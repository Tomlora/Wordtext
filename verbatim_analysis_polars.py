"""
Analyse de verbatims clients — Version haute volumétrie (20M+ conversations).

Optimisé avec :
    - Polars pour le traitement tabulaire (10-100x plus rapide que pandas)
    - Traitement par chunks pour limiter la RAM
    - Comptages pré-agrégés (pas de pivot géant)
    - LDA sur échantillon représentatif
    - Sentiment par lexique vectorisé

Prérequis :
    pip install polars numpy scikit-learn matplotlib seaborn

Entrée attendue :
    DataFrame Polars (ou Pandas, converti automatiquement) avec :
        - une colonne date
        - une colonne conversation_id
        - une colonne verbatims : liste de strings OU string repr de liste
          Ex : '["je veux un conseiller", "parler de ma facture", "oui"]'

Usage :
    analyzer = VerbatimAnalyzer(df, date_col='date', verbatim_col='verbatims')
    report = analyzer.run_full_analysis()
"""

import re
import ast
import json
import warnings
from collections import Counter
from datetime import timedelta
from typing import Optional

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

NEGATIVE_LEXICON: set[str] = {
    # Frustration / colère
    "scandaleux", "honteux", "inadmissible", "inacceptable", "honte",
    "nul", "nulle", "catastrophe", "catastrophique", "horrible",
    "lamentable", "déplorable", "minable", "pitoyable",
    # Problèmes techniques
    "panne", "bug", "bugge", "plante", "planté",
    "bloqué", "bloquée", "erreur", "impossible",
    "hs", "coupure", "coupé",
    # Insatisfaction
    "mécontent", "mécontente", "déçu", "déçue", "insatisfait",
    "insatisfaite", "plainte", "réclamation", "arnaque", "voleur",
    "voleurs", "menteur", "menteurs", "incompétent", "incompétente",
    "incompétents",
    # Attente / délais
    "attente", "attend", "attends", "relance", "relancé",
    # Résiliation / départ
    "résilier", "résiliation", "résilié", "partir", "quitter",
    "concurrence", "concurrent",
    # Facturation
    "surfacturation", "remboursement", "rembourser",
}

# Bigrams négatifs (détectés dans le texte nettoyé)
NEGATIVE_BIGRAMS: set[str] = {
    "marche pas", "fonctionne pas", "personne répond", "trop cher",
    "hors service", "pure simple", "porter plainte", "trois semaines",
}

STOPWORDS_FR: set[str] = {
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "votre", "leur", "leurs",
    "et", "ou", "mais", "donc", "car", "ni", "que", "qui", "dont",
    "où", "en", "dans", "sur", "sous", "avec", "sans", "pour", "par",
    "ne", "pas", "plus", "très", "bien", "aussi", "comme", "tout",
    "être", "avoir", "faire", "est", "sont", "suis", "ai", "a",
    "oui", "non", "merci", "bonjour", "bonsoir", "svp", "ok",
    "alors", "ça", "cela", "là", "ici", "voilà", "si", "se",
    "me", "te", "lui", "y", "à", "même", "quand", "peu", "peut",
    "encore", "après", "avant", "déjà", "tous", "toute", "toutes",
    "été", "fait", "dit", "mis", "pris", "rien", "quelque",
}


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES (vectorisées via Polars expressions)
# ══════════════════════════════════════════════════════════════════════

def _clean_expr(col: str = "verbatim") -> pl.Expr:
    """Expression Polars pour nettoyer un texte."""
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"http\S+", "")
        .str.replace_all(r"[^a-zàâäéèêëïîôùûüÿçœæ\s'-]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _neg_score_expr(col: str = "clean") -> pl.Expr:
    """
    Expression Polars pour scorer la négativité.
    Compte les mots du lexique négatif / nombre total de mots.
    """
    # On construit un pattern regex avec tous les mots négatifs
    # Pour les unigrams : word boundary match
    unigram_pattern = r"\b(" + "|".join(re.escape(w) for w in NEGATIVE_LEXICON) + r")\b"
    bigram_pattern = r"(" + "|".join(re.escape(b) for b in NEGATIVE_BIGRAMS) + r")"

    return (
        (
            pl.col(col).str.count_matches(unigram_pattern)
            + pl.col(col).str.count_matches(bigram_pattern)
        )
        .cast(pl.Float64)
        / pl.col(col).str.count_matches(r"\S+").cast(pl.Float64).clip(lower_bound=1)
    ).clip(upper_bound=1.0)


def _parse_verbatim_list(raw: str) -> list[str]:
    """Parse une string représentant une liste en vraie liste Python."""
    if not isinstance(raw, str):
        return [str(raw)] if raw is not None else []
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
        # Fallback : JSON
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, json.JSONDecodeError):
            pass
        # Fallback : split manuel
        inner = raw[1:-1]
        items = [item.strip().strip("'\"") for item in re.split(r"""['"],\s*['"]""", inner)]
        return [i for i in items if i]
    return [raw]


def _tokenize_batch(texts: list[str]) -> list[list[str]]:
    """Tokenise un batch de textes (pour la détection de termes)."""
    results = []
    for text in texts:
        words = text.split()
        results.append([w for w in words if w not in STOPWORDS_FR and len(w) > 2])
    return results


# ══════════════════════════════════════════════════════════════════════
# ANALYSEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class VerbatimAnalyzer:
    """
    Analyseur de verbatims clients — optimisé pour 20M+ conversations.

    Params:
        df                  : DataFrame source (Polars ou Pandas)
        date_col            : nom de la colonne date
        verbatim_col        : nom de la colonne contenant les listes de verbatims
        conversation_id_col : nom de la colonne id de conversation
        z_threshold         : seuil de z-score pour détecter un pic (défaut 2.0)
        growth_window       : fenêtre glissante en jours pour la croissance (défaut 7)
        growth_factor       : multiplicateur pour considérer un terme en croissance
        top_n_terms         : nombre de termes max dans les résultats
        lda_sample_size     : taille de l'échantillon pour la LDA (défaut 200_000)
        chunk_size          : taille des chunks pour le traitement par lots
    """

    def __init__(
        self,
        df,
        date_col: str = "date",
        verbatim_col: str = "verbatims",
        conversation_id_col: str = "conversation_id",
        z_threshold: float = 2.0,
        growth_window: int = 7,
        growth_factor: float = 3.0,
        top_n_terms: int = 20,
        lda_sample_size: int = 200_000,
        chunk_size: int = 500_000,
    ):
        self.date_col = date_col
        self.verbatim_col = verbatim_col
        self.conversation_id_col = conversation_id_col
        self.z_threshold = z_threshold
        self.growth_window = growth_window
        self.growth_factor = growth_factor
        self.top_n_terms = top_n_terms
        self.lda_sample_size = lda_sample_size
        self.chunk_size = chunk_size

        self.daily: Optional[pl.DataFrame] = None
        self.spike_details: list[dict] = []
        self.topics: list[dict] = []
        self.trending_terms = pl.DataFrame()
        self.novel_terms = pl.DataFrame()

        # Préparation
        self.df = self._prepare(df)

    # ── Préparation ──────────────────────────────────────────────────

    def _prepare(self, df) -> pl.DataFrame:
        """
        Prépare les données au niveau CONVERSATION.
        Concatène les verbatims de chaque liste en un seul texte.
        """
        print("⏳ Préparation des données...")

        # Conversion Pandas → Polars si nécessaire
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)

        # Parse des listes string → vraie liste, puis concaténation
        needs_parsing = df[self.verbatim_col].dtype == pl.Utf8
        if needs_parsing:
            print("   Parsing des listes string...")
            parsed = [
                _parse_verbatim_list(v)
                for v in df[self.verbatim_col].to_list()
            ]
            df = df.with_columns(
                pl.Series(name="verbatim_list", values=parsed)
            )
        else:
            # Déjà des listes Polars
            df = df.with_columns(
                pl.col(self.verbatim_col).alias("verbatim_list")
            )

        # Concaténation des messages en un seul texte par conversation
        df = df.with_columns(
            pl.col("verbatim_list")
            .list.join(". ")
            .alias("verbatim"),
            pl.col("verbatim_list")
            .list.len()
            .alias("n_messages"),
        )

        # Nettoyage vectorisé
        df = df.with_columns(_clean_expr("verbatim").alias("clean"))

        # Filtrage des conversations vides
        df = df.filter(pl.col("clean").str.len_chars() > 2)

        # Cast date
        if df[self.date_col].dtype != pl.Date:
            df = df.with_columns(pl.col(self.date_col).cast(pl.Date))

        n = df.height
        print(f"✅ {n:,} conversations prêtes")
        return df

    # ── 1. Sentiment & Détection de pics ─────────────────────────────

    def compute_sentiment(self) -> pl.DataFrame:
        """Score de négativité vectorisé via regex Polars."""
        print("⏳ Calcul du sentiment (lexique vectorisé)...")

        self.df = self.df.with_columns(
            _neg_score_expr("clean").alias("neg_score")
        ).with_columns(
            (pl.col("neg_score") > 0.15).alias("is_negative")
        )

        n_neg = self.df.filter(pl.col("is_negative")).height
        print(f"   {n_neg:,} conversations négatives "
              f"({n_neg / self.df.height * 100:.1f}%)")
        return self.df

    def detect_negative_spikes(self) -> pl.DataFrame:
        """
        Agrège par jour, détecte les pics via z-score glissant.
        Unité = CONVERSATION.
        """
        if "neg_score" not in self.df.columns:
            self.compute_sentiment()

        print("⏳ Détection des pics de négativité...")

        daily = (
            self.df
            .group_by(self.date_col)
            .agg(
                pl.count().alias("total_conversations"),
                pl.col("is_negative").sum().alias("negative_count"),
                pl.col("neg_score").mean().alias("mean_neg_score"),
            )
            .sort(self.date_col)
        )

        daily = daily.with_columns(
            (pl.col("negative_count") / pl.col("total_conversations"))
            .alias("negative_ratio")
        )

        # Z-score glissant sur 30 jours
        daily = daily.with_columns(
            pl.col("negative_ratio")
            .rolling_mean(window_size=30, min_periods=7)
            .alias("rolling_mean"),
            pl.col("negative_ratio")
            .rolling_std(window_size=30, min_periods=7)
            .alias("rolling_std"),
        ).with_columns(
            (
                (pl.col("negative_ratio") - pl.col("rolling_mean"))
                / pl.col("rolling_std").clip(lower_bound=1e-9)
            ).alias("z_score")
        ).with_columns(
            (pl.col("z_score") > self.z_threshold).alias("is_spike")
        )

        # Détails des pics : top termes négatifs + exemples
        spike_dates = (
            daily.filter(pl.col("is_spike"))
            .get_column(self.date_col)
            .to_list()
        )

        spike_details = []
        for d in spike_dates:
            day_neg = self.df.filter(
                (pl.col(self.date_col) == d) & pl.col("is_negative")
            )
            # Tokenise seulement les négatifs du jour (petit volume)
            all_tokens = []
            for text in day_neg.get_column("clean").to_list():
                words = text.split()
                all_tokens.extend(
                    w for w in words if w not in STOPWORDS_FR and len(w) > 2
                )
            top_terms = Counter(all_tokens).most_common(10)
            samples = day_neg.get_column("verbatim").head(5).to_list()
            spike_details.append({
                "date": d,
                "top_negative_terms": top_terms,
                "sample_verbatims": samples,
            })

        self.daily = daily
        self.spike_details = spike_details
        print(f"✅ {len(spike_dates)} pic(s) de négativité détecté(s)")
        return daily

    # ── 2. Sujets émergents (LDA sur échantillon) ───────────────────

    def detect_emerging_topics(
        self, n_topics: int = 10, recent_days: int = 14
    ) -> dict:
        """
        LDA sur un échantillon pour identifier les sujets émergents.
        Compare la distribution récente vs historique.
        """
        print(f"⏳ Détection des sujets émergents "
              f"(LDA sur échantillon de {self.lda_sample_size:,})...")

        max_date = self.df.get_column(self.date_col).max()
        cutoff = max_date - timedelta(days=recent_days)

        # Échantillonnage stratifié : garder la proportion recent/history
        recent = self.df.filter(pl.col(self.date_col) >= cutoff)
        history = self.df.filter(pl.col(self.date_col) < cutoff)

        # Proportions dans l'échantillon
        recent_ratio = recent.height / self.df.height
        n_recent = max(int(self.lda_sample_size * recent_ratio), 10_000)
        n_history = self.lda_sample_size - n_recent

        recent_sample = recent.sample(n=min(n_recent, recent.height), seed=42)
        history_sample = history.sample(n=min(n_history, history.height), seed=42)
        sample = pl.concat([history_sample, recent_sample])

        # Marqueur recent/history
        is_recent = [False] * history_sample.height + [True] * recent_sample.height

        print(f"   Échantillon : {history_sample.height:,} historique + "
              f"{recent_sample.height:,} récent")

        # Vectorisation
        texts = sample.get_column("clean").to_list()
        vectorizer = CountVectorizer(
            max_features=3000,
            min_df=10,
            max_df=0.8,
            stop_words=list(STOPWORDS_FR),
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=15,
            n_jobs=-1,  # parallélisation
        )
        doc_topics = lda.fit_transform(X)

        # Comparaison récent vs historique
        is_recent_arr = np.array(is_recent)
        recent_dist = doc_topics[is_recent_arr].mean(axis=0)
        history_dist = doc_topics[~is_recent_arr].mean(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            emergence_ratio = np.where(
                history_dist > 0.001,
                recent_dist / history_dist,
                recent_dist * 100,
            )

        topics = []
        for idx in range(n_topics):
            top_word_indices = lda.components_[idx].argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topics.append({
                "topic_id": idx,
                "top_words": top_words,
                "recent_weight": round(float(recent_dist[idx]), 4),
                "history_weight": round(float(history_dist[idx]), 4),
                "emergence_ratio": round(float(emergence_ratio[idx]), 2),
            })

        topics = sorted(topics, key=lambda x: x["emergence_ratio"], reverse=True)
        self.topics = topics
        emerging = [t for t in topics if t["emergence_ratio"] > 1.5]
        print(f"✅ {len(emerging)} sujet(s) émergent(s) détecté(s) (ratio > 1.5x)")
        return {"all_topics": topics, "emerging": emerging}

    # ── 3. Termes en forte croissance ────────────────────────────────

    def detect_trending_terms(self) -> pl.DataFrame:
        """
        Détecte les termes en forte croissance.
        Utilise des comptages pré-agrégés par jour (pas de pivot géant).
        """
        print("⏳ Détection des termes en forte croissance...")
        print("   Comptage des termes par jour (par chunks)...")

        # Comptage terme × jour via chunks pour limiter la RAM
        term_day_counts: Counter = Counter()
        n = self.df.height

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            chunk_texts = self.df.slice(start, end - start).get_column("clean").to_list()
            chunk_dates = self.df.slice(start, end - start).get_column(self.date_col).to_list()

            for text, d in zip(chunk_texts, chunk_dates):
                words = text.split()
                seen = set()  # compter un terme max 1 fois par conversation
                for w in words:
                    if w not in STOPWORDS_FR and len(w) > 2 and w not in seen:
                        term_day_counts[(str(d), w)] += 1
                        seen.add(w)

            if (start // self.chunk_size) % 10 == 0 and start > 0:
                print(f"   ... {start:,}/{n:,} conversations traitées")

        print(f"   {len(term_day_counts):,} paires (terme, jour) comptées")

        # Reconstruction en DataFrame Polars
        records = [
            {"date": k[0], "term": k[1], "count": v}
            for k, v in term_day_counts.items()
        ]
        term_df = pl.DataFrame(records).with_columns(
            pl.col("date").str.to_date()
        )

        # Agrégation : moyenne par terme sur les fenêtres
        max_date = term_df.get_column("date").max()
        recent_start = max_date - timedelta(days=self.growth_window)
        baseline_end = recent_start
        baseline_start = baseline_end - timedelta(days=30)

        # Moyenne récente par terme
        recent_avg = (
            term_df
            .filter(pl.col("date") >= recent_start)
            .group_by("term")
            .agg(
                (pl.col("count").sum() / self.growth_window)
                .alias("current_avg_daily")
            )
        )

        # Moyenne baseline par terme (30 jours avant la fenêtre récente)
        baseline_avg = (
            term_df
            .filter(
                (pl.col("date") >= baseline_start)
                & (pl.col("date") < baseline_end)
            )
            .group_by("term")
            .agg(
                (pl.col("count").sum() / 30.0).alias("baseline_avg_daily")
            )
        )

        # Jointure et calcul du ratio
        trending = (
            recent_avg
            .join(baseline_avg, on="term", how="left")
            .with_columns(
                pl.col("baseline_avg_daily").fill_null(0.0)
            )
            .with_columns(
                pl.when(pl.col("baseline_avg_daily") > 0.5)
                .then(pl.col("current_avg_daily") / pl.col("baseline_avg_daily"))
                .otherwise(pl.col("current_avg_daily") * 10)
                .alias("growth_ratio")
            )
            .filter(
                (pl.col("growth_ratio") >= self.growth_factor)
                & (pl.col("current_avg_daily") >= 2)
            )
            .sort("growth_ratio", descending=True)
            .head(self.top_n_terms)
            .with_columns(
                pl.col("current_avg_daily").round(2),
                pl.col("baseline_avg_daily").round(2),
                pl.col("growth_ratio").round(2),
            )
        )

        self.trending_terms = trending
        self.term_daily_df = term_df  # pour usage ultérieur si besoin
        print(f"✅ {trending.height} terme(s) en forte croissance "
              f"(x{self.growth_factor}+)")
        return trending

    # ── 4. Termes inhabituels (nouveaux) ─────────────────────────────

    def detect_novel_terms(
        self, recent_days: int = 7, min_count: int = 20
    ) -> pl.DataFrame:
        """
        Détecte les termes quasi absents historiquement mais soudainement fréquents.
        Seuil min_count relevé à 20 vu le volume.
        """
        print("⏳ Détection des termes inhabituels...")

        max_date = self.df.get_column(self.date_col).max()
        cutoff = max_date - timedelta(days=recent_days)
        min_date = self.df.get_column(self.date_col).min()
        total_hist_days = (cutoff - min_date).days or 1

        # Comptage sur recent et history par chunks
        recent_counts: Counter = Counter()
        history_counts: Counter = Counter()

        n = self.df.height
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            chunk = self.df.slice(start, end - start)
            chunk_texts = chunk.get_column("clean").to_list()
            chunk_dates = chunk.get_column(self.date_col).to_list()

            for text, d in zip(chunk_texts, chunk_dates):
                words = set(text.split())  # unique par conversation
                tokens = {w for w in words if w not in STOPWORDS_FR and len(w) > 2}
                if d >= cutoff:
                    recent_counts.update(tokens)
                else:
                    history_counts.update(tokens)

        # Identification des termes nouveaux
        novel = []
        for term, count in recent_counts.items():
            if count < min_count:
                continue
            hist = history_counts.get(term, 0)
            hist_daily = hist / total_hist_days
            recent_daily = count / recent_days

            if hist_daily < 1.0:  # quasi absent historiquement
                novel.append({
                    "term": term,
                    "recent_count": count,
                    "recent_daily_avg": round(recent_daily, 2),
                    "history_total": hist,
                    "history_daily_avg": round(hist_daily, 2),
                    "novelty": "nouveau" if hist == 0 else "quasi_nouveau",
                })

        novel_df = (
            pl.DataFrame(novel)
            .sort("recent_count", descending=True)
            .head(self.top_n_terms)
        ) if novel else pl.DataFrame()

        self.novel_terms = novel_df
        print(f"✅ {novel_df.height} terme(s) inhabituels détecté(s)")
        return novel_df

    # ── 5. Visualisations ────────────────────────────────────────────

    def plot_dashboard(self, save_path: Optional[str] = None):
        """Dashboard récapitulatif."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 14), constrained_layout=True)
        fig.suptitle(
            "📊 Dashboard Analyse Verbatims Clients (20M+)",
            fontsize=16, fontweight="bold",
        )

        # ── Graphe 1 : Volume + ratio négatif + pics ──
        if self.daily is not None:
            ax1 = axes[0]
            ax1.set_title("Volume de conversations et pics de négativité")
            daily_pd = self.daily.with_columns(
                pl.col("is_spike").fill_null(False)
            ).to_pandas()
            ax1.bar(
                daily_pd[self.date_col], daily_pd["total_conversations"],
                color="#c8d6e5", alpha=0.7, label="Total conversations",
            )
            ax1b = ax1.twinx()
            ax1b.plot(
                daily_pd[self.date_col], daily_pd["negative_ratio"],
                color="#e74c3c", linewidth=1.5, label="Ratio négatif",
            )
            spikes = daily_pd[daily_pd["is_spike"]]
            ax1b.scatter(
                spikes[self.date_col], spikes["negative_ratio"],
                color="red", s=80, zorder=5, marker="^", label="Pic détecté",
            )
            ax1.set_ylabel("Volume")
            ax1b.set_ylabel("Ratio négatif", color="#e74c3c")
            ax1.legend(loc="upper left")
            ax1b.legend(loc="upper right")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

        # ── Graphe 2 : Termes en croissance ──
        ax2 = axes[1]
        if self.trending_terms.height > 0:
            data = self.trending_terms.head(15).to_pandas()
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(data)))[::-1]
            ax2.barh(data["term"], data["growth_ratio"], color=colors)
            ax2.set_xlabel("Ratio de croissance")
            ax2.set_title("Termes en forte croissance (top 15)")
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, "Pas de données", ha="center", va="center")
            ax2.set_title("Termes en forte croissance")

        # ── Graphe 3 : Termes inhabituels ──
        ax3 = axes[2]
        if self.novel_terms.height > 0:
            data = self.novel_terms.head(15).to_pandas()
            colors = [
                "#e67e22" if n == "nouveau" else "#f39c12"
                for n in data["novelty"]
            ]
            ax3.barh(data["term"], data["recent_count"], color=colors)
            ax3.set_xlabel("Occurrences récentes")
            ax3.set_title("Termes inhabituels / nouveaux (top 15)")
            ax3.invert_yaxis()
        else:
            ax3.text(0.5, 0.5, "Pas de données", ha="center", va="center")
            ax3.set_title("Termes inhabituels / nouveaux")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📁 Dashboard sauvegardé : {save_path}")

        plt.close(fig)
        return fig

    # ── 6. Rapport complet ───────────────────────────────────────────

    def run_full_analysis(self, save_dashboard: Optional[str] = None) -> dict:
        """Lance l'analyse complète."""
        print("=" * 60)
        print("🔍 ANALYSE COMPLÈTE DES VERBATIMS CLIENTS")
        print("=" * 60)

        min_date = self.df.get_column(self.date_col).min()
        max_date = self.df.get_column(self.date_col).max()
        print(f"📅 Période : {min_date} → {max_date}")
        print(f"📝 Nombre de conversations : {self.df.height:,}")
        print()

        # 1. Sentiment + pics
        self.detect_negative_spikes()
        print()

        # 2. Sujets émergents (sur échantillon)
        topics_result = self.detect_emerging_topics()
        print()

        # 3. Termes en croissance
        trending = self.detect_trending_terms()
        print()

        # 4. Termes inhabituels
        novel = self.detect_novel_terms()
        print()

        # 5. Dashboard
        fig = self.plot_dashboard(save_path=save_dashboard)

        # ── Rapport console ──
        print("=" * 60)
        print("📋 RÉSUMÉ")
        print("=" * 60)

        if self.spike_details:
            print("\n🔴 PICS DE NÉGATIVITÉ :")
            for sp in self.spike_details[:5]:
                terms_str = ", ".join(
                    f"{t[0]} ({t[1]})" for t in sp["top_negative_terms"][:5]
                )
                print(f"  📅 {sp['date']} — Termes : {terms_str}")
                for v in sp["sample_verbatims"][:2]:
                    print(f"     💬 \"{str(v)[:100]}\"")

        if topics_result["emerging"]:
            print("\n🟠 SUJETS ÉMERGENTS :")
            for t in topics_result["emerging"][:5]:
                words = ", ".join(t["top_words"][:5])
                print(
                    f"  Topic #{t['topic_id']} (x{t['emergence_ratio']}) : "
                    f"{words}"
                )

        if trending.height > 0:
            print("\n🟡 TERMES EN FORTE CROISSANCE :")
            for row in trending.head(10).iter_rows(named=True):
                print(
                    f"  📈 {row['term']} : x{row['growth_ratio']} "
                    f"({row['baseline_avg_daily']}/j → "
                    f"{row['current_avg_daily']}/j)"
                )

        if novel.height > 0:
            print("\n🆕 TERMES INHABITUELS / NOUVEAUX :")
            for row in novel.head(10).iter_rows(named=True):
                tag = "🆕" if row["novelty"] == "nouveau" else "⚠️"
                print(
                    f"  {tag} {row['term']} : {row['recent_count']} "
                    f"occurrences ({row['novelty']})"
                )

        print("\n" + "=" * 60)

        return {
            "daily_stats": self.daily,
            "spike_details": self.spike_details,
            "topics": topics_result,
            "trending_terms": trending,
            "novel_terms": novel,
            "figure": fig,
        }


# ══════════════════════════════════════════════════════════════════════
# DÉMO
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    np.random.seed(42)
    from datetime import date as pydate
    dates = pl.date_range(
        pydate(2024, 6, 1), pydate(2025, 2, 25), eager=True
    ).to_list()

    normal_conversations = [
        ["Bonjour je souhaite modifier mon offre", "oui celle en cours", "d'accord merci"],
        ["Quand sera traitée ma demande", "celle du mois dernier", "ok je patiente"],
        ["Je veux des informations sur mon contrat", "oui c'est bien ça", "merci"],
        ["Pouvez-vous me rappeler", "demain matin si possible", "parfait"],
        ["J'aimerais changer mon mot de passe", "oui je confirme", "c'est bon merci"],
        ["Comment accéder à mon espace client", "je ne trouve pas le lien"],
        ["Je voudrais un conseiller", "pour parler de ma facture", "oui"],
    ]
    negative_conversations = [
        ["C'est scandaleux personne ne me répond", "ça fait des jours que j'attends",
         "je vais porter plainte"],
        ["Ça fait trois semaines que j'attends", "toujours rien", "c'est inadmissible"],
        ["Service incompétent je vais résilier", "j'en ai assez",
         "passez-moi un responsable"],
        ["Panne depuis ce matin rien ne fonctionne", "toujours pas rétabli",
         "c'est la troisième fois ce mois"],
        ["Bug sur l'application impossible de se connecter", "j'ai tout essayé",
         "ça ne marche toujours pas"],
        ["Je suis très déçu de votre service", "la qualité a vraiment baissé"],
        ["Arnaque pure et simple remboursez-moi", "je n'ai jamais demandé ça"],
        ["Votre service est lamentable", "je vais aller voir la concurrence"],
        ["Je vais aller chez le concurrent", "vous êtes trop cher",
         "j'ai reçu une meilleure offre"],
        ["Surfacturation sur ma dernière facture", "je veux un remboursement immédiat"],
    ]
    emerging_conversations = [
        ["Problème avec la nouvelle mise à jour fibre", "rien ne marche depuis",
         "quand est-ce que ça sera réparé"],
        ["Depuis la migration fibre plus rien ne marche", "internet coupé",
         "c'est inadmissible"],
        ["Migration fibre catastrophique", "débit ridicule", "je veux résilier"],
    ]

    # Génération des conversations
    conv_id = 0
    all_dates, all_ids, all_verbatims = [], [], []

    for d in dates:
        n = np.random.randint(50, 200)
        n_normal = int(n * 0.75)
        n_negative = n - n_normal

        for _ in range(n_normal):
            conv = normal_conversations[np.random.randint(len(normal_conversations))]
            all_dates.append(d)
            all_ids.append(f"conv_{conv_id:06d}")
            # Stocker en string repr de liste (comme les vraies données)
            all_verbatims.append(str(conv))
            conv_id += 1

        for _ in range(n_negative):
            conv = negative_conversations[np.random.randint(len(negative_conversations))]
            all_dates.append(d)
            all_ids.append(f"conv_{conv_id:06d}")
            all_verbatims.append(str(conv))
            conv_id += 1

        if d == pydate(2025, 1, 15):
            for _ in range(100):
                conv = negative_conversations[np.random.randint(len(negative_conversations))]
                all_dates.append(d)
                all_ids.append(f"conv_{conv_id:06d}")
                all_verbatims.append(str(conv))
                conv_id += 1

        if d >= pydate(2025, 2, 12):
            for _ in range(30):
                conv = emerging_conversations[np.random.randint(len(emerging_conversations))]
                all_dates.append(d)
                all_ids.append(f"conv_{conv_id:06d}")
                all_verbatims.append(str(conv))
                conv_id += 1

    df_demo = pl.DataFrame({
        "date": all_dates,
        "conversation_id": all_ids,
        "verbatims": all_verbatims,
    })

    print(f"📊 Dataset de démo : {df_demo.height:,} conversations\n")

    start_time = time.time()

    analyzer = VerbatimAnalyzer(
        df_demo,
        date_col="date",
        verbatim_col="verbatims",
        conversation_id_col="conversation_id",
        z_threshold=2.0,
        growth_window=7,
        growth_factor=3.0,
        lda_sample_size=50_000,  # réduit pour la démo
    )
    report = analyzer.run_full_analysis(save_dashboard="dashboard_verbatims.png")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Temps total : {elapsed:.1f}s")

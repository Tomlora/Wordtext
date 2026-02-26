"""
Analyse de verbatims clients : détection de pics, sujets émergents et termes inhabituels.

Prérequis :
    pip install pandas numpy scikit-learn transformers torch matplotlib seaborn

Entrée attendue :
    DataFrame avec au minimum :
        - une colonne date (ex: 'date')
        - une colonne verbatims sous forme de liste (ex: 'verbatims')
          Exemple : ['Je souhaiterai avoir un conseiller', 'évoquer mon offre', 'oui']

Usage :
    analyzer = VerbatimAnalyzer(df, date_col='date', verbatim_col='verbatims')
    report = analyzer.run_full_analysis()
"""

import re
import ast
import warnings
from collections import Counter
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

# Mots négatifs courants dans un contexte service client FR
NEGATIVE_LEXICON = {
    # Frustration / colère
    "scandaleux", "honteux", "inadmissible", "inacceptable", "honte",
    "nul", "nulle", "catastrophe", "catastrophique", "horrible",
    "lamentable", "déplorable", "minable", "pitoyable",
    # Problèmes techniques
    "panne", "bug", "bugge", "plante", "planté", "marche pas",
    "fonctionne pas", "bloqué", "bloquée", "erreur", "impossible",
    "hs", "hors service", "coupure", "coupé",
    # Insatisfaction
    "mécontent", "mécontente", "déçu", "déçue", "insatisfait",
    "insatisfaite", "plainte", "réclamation", "arnaque", "voleur",
    "voleurs", "menteur", "menteurs", "incompétent", "incompétente",
    "incompétents",
    # Attente / délais
    "attente", "attend", "attends", "heures", "jours", "semaines",
    "relance", "relancé", "rappelle", "personne ne répond",
    # Résiliation / départ
    "résilier", "résiliation", "résilié", "partir", "quitter",
    "concurrence", "concurrent", "changer", "fermer",
    # Facturation
    "facture", "surfacturation", "prélèvement", "remboursement",
    "rembourser", "trop cher", "augmentation", "prix",
}

# Stopwords FR minimaux pour le nettoyage
STOPWORDS_FR = {
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


# ──────────────────────────────────────────────────────────────────────
# NETTOYAGE
# ──────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Nettoyage basique d'un verbatim."""
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zàâäéèêëïîôùûüÿçœæ\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenise et retire les stopwords."""
    words = clean_text(text).split()
    return [w for w in words if w not in STOPWORDS_FR and len(w) > 2]


# ──────────────────────────────────────────────────────────────────────
# ANALYSE DE SENTIMENT (lexique)
# ──────────────────────────────────────────────────────────────────────

def sentiment_score_lexicon(text: str) -> float:
    """
    Score de négativité basé sur le lexique.
    Retourne un ratio entre 0 (neutre/positif) et 1 (très négatif).
    """
    words = clean_text(text).split()
    if not words:
        return 0.0
    neg_count = sum(1 for w in words if w in NEGATIVE_LEXICON)
    # On cherche aussi les bigrams courants
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    neg_count += sum(1 for b in bigrams if b in NEGATIVE_LEXICON)
    return min(neg_count / max(len(words), 1), 1.0)


def sentiment_score_transformer(texts: list[str], batch_size: int = 64) -> list[float]:
    """
    Sentiment via un modèle CamemBERT fine-tuné (optionnel).
    Retourne une liste de scores négatifs entre 0 et 1.
    Fallback sur le lexique si le modèle n'est pas disponible.
    """
    try:
        from transformers import pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="cmarkea/distilcamembert-base-sentiment",
            truncation=True,
            max_length=512,
        )
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = classifier(batch)
            for r in results:
                # Le modèle retourne 1-5 étoiles ; on convertit en score négatif
                label = int(r["label"].replace(" star", "").replace(" stars", ""))
                scores.append(1.0 - (label - 1) / 4.0)
        return scores
    except Exception:
        return [sentiment_score_lexicon(t) for t in texts]


# ──────────────────────────────────────────────────────────────────────
# ANALYSEUR PRINCIPAL
# ──────────────────────────────────────────────────────────────────────

class VerbatimAnalyzer:
    """
    Analyseur de verbatims clients.

    Params:
        df              : DataFrame source
        date_col        : nom de la colonne date
        verbatim_col    : nom de la colonne contenant les listes de verbatims
        use_transformer : True pour utiliser CamemBERT (plus lent, plus précis)
        z_threshold     : seuil de z-score pour détecter un pic (défaut 2.0)
        growth_window   : fenêtre glissante en jours pour détecter la croissance (défaut 7)
        growth_factor   : facteur multiplicatif pour considérer un terme en forte croissance
        top_n_terms     : nombre de termes à afficher dans les résultats
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        verbatim_col: str = "verbatims",
        conversation_id_col: str = "conversation_id",
        use_transformer: bool = False,
        z_threshold: float = 2.0,
        growth_window: int = 7,
        growth_factor: float = 3.0,
        top_n_terms: int = 20,
    ):
        self.date_col = date_col
        self.verbatim_col = verbatim_col
        self.conversation_id_col = conversation_id_col
        self.use_transformer = use_transformer
        self.z_threshold = z_threshold
        self.growth_window = growth_window
        self.growth_factor = growth_factor
        self.top_n_terms = top_n_terms

        # Préparation des données
        self.df = self._prepare(df.copy())
        self.daily: Optional[pd.DataFrame] = None
        self.term_daily: Optional[pd.DataFrame] = None

    # ── Préparation ──────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données au niveau CONVERSATION (1 ligne = 1 échange client).
        Concatène les verbatims de la liste en un seul texte par conversation.
        """
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # Gestion du cas où la colonne est une string repr de liste
        sample = df[self.verbatim_col].dropna().iloc[0]
        if isinstance(sample, str):
            df[self.verbatim_col] = df[self.verbatim_col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        # Concaténation : on joint les éléments de la liste en un seul texte
        # Chaque ligne = 1 conversation client complète
        df["verbatim_list"] = df[self.verbatim_col]
        df["n_messages"] = df[self.verbatim_col].apply(
            lambda x: len(x) if isinstance(x, list) else 1
        )
        df["verbatim"] = df[self.verbatim_col].apply(
            lambda x: ". ".join(str(v).strip() for v in x if str(v).strip())
            if isinstance(x, list)
            else str(x)
        )
        df = df[df["verbatim"].str.len() > 2]

        # Nettoyage et tokenisation sur le texte concaténé
        df["clean"] = df["verbatim"].apply(clean_text)
        df["tokens"] = df["clean"].apply(tokenize)

        return df.reset_index(drop=True)

    # ── 1. Sentiment & Détection de pics ─────────────────────────────

    def compute_sentiment(self) -> pd.DataFrame:
        """Calcule le score de négativité pour chaque verbatim."""
        print("⏳ Calcul du sentiment...")
        if self.use_transformer:
            self.df["neg_score"] = sentiment_score_transformer(
                self.df["clean"].tolist()
            )
        else:
            self.df["neg_score"] = self.df["clean"].apply(sentiment_score_lexicon)

        self.df["is_negative"] = self.df["neg_score"] > 0.15
        return self.df

    def detect_negative_spikes(self) -> pd.DataFrame:
        """
        Agrège par jour et détecte les pics de négativité via z-score.
        Unité d'analyse = la CONVERSATION (pas la phrase individuelle).

        Retourne un DataFrame journalier avec :
            - total_conversations, negative_count, negative_ratio
            - z_score, is_spike
        """
        if "neg_score" not in self.df.columns:
            self.compute_sentiment()

        daily = (
            self.df.groupby(self.df[self.date_col].dt.date)
            .agg(
                total_conversations=("verbatim", "count"),
                negative_count=("is_negative", "sum"),
                mean_neg_score=("neg_score", "mean"),
            )
            .reset_index()
        )
        daily.columns = [
            "date", "total_conversations", "negative_count", "mean_neg_score"
        ]
        daily["date"] = pd.to_datetime(daily["date"])
        daily["negative_ratio"] = (
            daily["negative_count"] / daily["total_conversations"]
        )

        # Z-score sur le ratio de négatifs
        mu = daily["negative_ratio"].rolling(30, min_periods=7).mean()
        sigma = daily["negative_ratio"].rolling(30, min_periods=7).std()
        daily["z_score"] = (daily["negative_ratio"] - mu) / sigma.replace(0, np.nan)
        daily["is_spike"] = daily["z_score"] > self.z_threshold

        # Pour chaque pic, récupérer les termes négatifs dominants
        spike_dates = daily.loc[daily["is_spike"], "date"].tolist()
        spike_details = []
        for d in spike_dates:
            day_neg = self.df[
                (self.df[self.date_col].dt.date == d.date()) & (self.df["is_negative"])
            ]
            all_tokens = [t for tokens in day_neg["tokens"] for t in tokens]
            top_terms = Counter(all_tokens).most_common(10)
            spike_details.append({
                "date": d,
                "top_negative_terms": top_terms,
                "sample_verbatims": day_neg["verbatim"].head(5).tolist(),
            })

        self.daily = daily
        self.spike_details = spike_details
        print(f"✅ {len(spike_dates)} pic(s) de négativité détecté(s)")
        return daily

    # ── 2. Sujets émergents (Topic Modeling) ─────────────────────────

    def detect_emerging_topics(
        self, n_topics: int = 10, recent_days: int = 14
    ) -> dict:
        """
        LDA pour identifier des clusters de sujets.
        Compare la distribution récente vs historique pour repérer les émergents.
        """
        print("⏳ Détection des sujets émergents...")

        cutoff = self.df[self.date_col].max() - timedelta(days=recent_days)
        mask_recent = self.df[self.date_col] >= cutoff
        mask_history = self.df[self.date_col] < cutoff

        # Vectorisation
        vectorizer = CountVectorizer(
            max_features=2000,
            min_df=5,
            max_df=0.8,
            stop_words=list(STOPWORDS_FR),
            ngram_range=(1, 2),
        )
        all_texts = self.df["clean"].tolist()
        X = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()

        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=20
        )
        doc_topics = lda.fit_transform(X)

        # Distribution des topics récents vs historiques
        recent_dist = doc_topics[mask_recent.values].mean(axis=0)
        history_dist = doc_topics[mask_history.values].mean(axis=0)

        # Ratio d'émergence
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
                "recent_weight": round(recent_dist[idx], 4),
                "history_weight": round(history_dist[idx], 4),
                "emergence_ratio": round(emergence_ratio[idx], 2),
            })

        # Tri par ratio d'émergence décroissant
        topics = sorted(topics, key=lambda x: x["emergence_ratio"], reverse=True)

        self.topics = topics
        emerging = [t for t in topics if t["emergence_ratio"] > 1.5]
        print(f"✅ {len(emerging)} sujet(s) émergent(s) détecté(s) (ratio > 1.5x)")
        return {"all_topics": topics, "emerging": emerging}

    # ── 3. Termes en forte croissance ────────────────────────────────

    def detect_trending_terms(self) -> pd.DataFrame:
        """
        Identifie les termes dont la fréquence a explosé récemment
        par rapport à leur moyenne historique.
        """
        print("⏳ Détection des termes en forte croissance...")

        # Comptage journalier par terme
        records = []
        for _, row in self.df.iterrows():
            d = row[self.date_col].date()
            for token in row["tokens"]:
                records.append({"date": d, "term": token})

        term_df = pd.DataFrame(records)
        if term_df.empty:
            print("⚠️ Pas assez de données pour détecter les tendances")
            return pd.DataFrame()

        term_daily = (
            term_df.groupby(["date", "term"]).size().reset_index(name="count")
        )
        term_daily["date"] = pd.to_datetime(term_daily["date"])

        # Pivot : une colonne par terme
        pivot = term_daily.pivot_table(
            index="date", columns="term", values="count", fill_value=0
        )

        # Moyenne mobile historique vs fenêtre récente
        window = self.growth_window
        rolling_mean = pivot.rolling(window=30, min_periods=7).mean()
        recent_mean = pivot.rolling(window=window, min_periods=3).mean()

        # Dernière ligne = situation actuelle
        last_date = pivot.index.max()
        current = recent_mean.loc[last_date]
        baseline = rolling_mean.loc[last_date - timedelta(days=window)]

        # Calcul du ratio de croissance
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = np.where(
                baseline > 0.5, current / baseline, current * 10
            )

        trending = pd.DataFrame({
            "term": pivot.columns,
            "current_avg_daily": current.values.round(2),
            "baseline_avg_daily": baseline.values.round(2),
            "growth_ratio": growth.round(2),
        })

        # Filtrage : termes significatifs en forte croissance
        trending = trending[
            (trending["growth_ratio"] >= self.growth_factor)
            & (trending["current_avg_daily"] >= 2)
        ].sort_values("growth_ratio", ascending=False)

        self.trending_terms = trending.head(self.top_n_terms)
        self.term_daily_pivot = pivot

        print(
            f"✅ {len(self.trending_terms)} terme(s) en forte croissance "
            f"(x{self.growth_factor}+)"
        )
        return self.trending_terms

    # ── 4. Termes inhabituels (nouveaux termes) ──────────────────────

    def detect_novel_terms(self, recent_days: int = 7, min_count: int = 5) -> pd.DataFrame:
        """
        Détecte les termes qui n'existaient pas (ou très peu) avant
        et qui apparaissent soudainement.
        """
        print("⏳ Détection des termes inhabituels (nouveaux)...")

        cutoff = self.df[self.date_col].max() - timedelta(days=recent_days)

        recent_tokens = [
            t
            for tokens in self.df.loc[
                self.df[self.date_col] >= cutoff, "tokens"
            ]
            for t in tokens
        ]
        history_tokens = [
            t
            for tokens in self.df.loc[
                self.df[self.date_col] < cutoff, "tokens"
            ]
            for t in tokens
        ]

        recent_counts = Counter(recent_tokens)
        history_counts = Counter(history_tokens)

        novel = []
        for term, count in recent_counts.items():
            if count < min_count:
                continue
            hist = history_counts.get(term, 0)
            # Normalisation par le nombre de jours
            total_days = (cutoff - self.df[self.date_col].min()).days or 1
            hist_daily = hist / total_days
            recent_daily = count / recent_days

            if hist_daily < 0.5:  # Terme quasi absent historiquement
                novel.append({
                    "term": term,
                    "recent_count": count,
                    "recent_daily_avg": round(recent_daily, 2),
                    "history_total": hist,
                    "history_daily_avg": round(hist_daily, 2),
                    "novelty": "nouveau" if hist == 0 else "quasi_nouveau",
                })

        novel_df = pd.DataFrame(novel).sort_values(
            "recent_count", ascending=False
        )
        self.novel_terms = novel_df.head(self.top_n_terms)
        print(f"✅ {len(self.novel_terms)} terme(s) inhabituels détecté(s)")
        return self.novel_terms

    # ── 5. Visualisations ────────────────────────────────────────────

    def plot_dashboard(self, save_path: Optional[str] = None):
        """Génère un dashboard récapitulatif."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 14), constrained_layout=True)
        fig.suptitle(
            "📊 Dashboard Analyse Verbatims Clients", fontsize=16, fontweight="bold"
        )

        # ── Graphe 1 : Volume + ratio négatif + pics ──
        if self.daily is not None:
            ax1 = axes[0]
            ax1.set_title("Volume de verbatims et pics de négativité")
            ax1.bar(
                self.daily["date"],
                self.daily["total_conversations"],
                color="#c8d6e5",
                alpha=0.7,
                label="Total conversations",
            )
            ax1b = ax1.twinx()
            ax1b.plot(
                self.daily["date"],
                self.daily["negative_ratio"],
                color="#e74c3c",
                linewidth=1.5,
                label="Ratio négatif",
            )
            # Marquer les pics
            spikes = self.daily[self.daily["is_spike"]]
            ax1b.scatter(
                spikes["date"],
                spikes["negative_ratio"],
                color="red",
                s=80,
                zorder=5,
                marker="^",
                label="Pic détecté",
            )
            ax1.set_ylabel("Volume")
            ax1b.set_ylabel("Ratio négatif", color="#e74c3c")
            ax1.legend(loc="upper left")
            ax1b.legend(loc="upper right")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

        # ── Graphe 2 : Top termes en croissance ──
        ax2 = axes[1]
        if hasattr(self, "trending_terms") and not self.trending_terms.empty:
            data = self.trending_terms.head(15)
            colors = plt.cm.Reds(
                np.linspace(0.4, 0.9, len(data))
            )[::-1]
            ax2.barh(data["term"], data["growth_ratio"], color=colors)
            ax2.set_xlabel("Ratio de croissance")
            ax2.set_title("Termes en forte croissance (top 15)")
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, "Pas de données", ha="center", va="center")
            ax2.set_title("Termes en forte croissance")

        # ── Graphe 3 : Termes inhabituels ──
        ax3 = axes[2]
        if hasattr(self, "novel_terms") and not self.novel_terms.empty:
            data = self.novel_terms.head(15)
            colors = ["#e67e22" if n == "nouveau" else "#f39c12"
                      for n in data["novelty"]]
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
        """
        Lance l'analyse complète et retourne un dictionnaire récapitulatif.

        Params:
            save_dashboard : chemin pour sauvegarder le dashboard (ex: 'dashboard.png')
        """
        print("=" * 60)
        print("🔍 ANALYSE COMPLÈTE DES VERBATIMS CLIENTS")
        print("=" * 60)
        print(f"📅 Période : {self.df[self.date_col].min().date()} → "
              f"{self.df[self.date_col].max().date()}")
        print(f"📝 Nombre de conversations : {len(self.df):,}")
        print()

        # 1. Sentiment + pics
        self.detect_negative_spikes()
        print()

        # 2. Sujets émergents
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
                    [f"{t[0]} ({t[1]})" for t in sp["top_negative_terms"][:5]]
                )
                print(f"  📅 {sp['date'].strftime('%d/%m/%Y')} — Termes : {terms_str}")
                for v in sp["sample_verbatims"][:2]:
                    print(f"     💬 \"{v[:100]}\"")

        if topics_result["emerging"]:
            print("\n🟠 SUJETS ÉMERGENTS :")
            for t in topics_result["emerging"][:5]:
                words = ", ".join(t["top_words"][:5])
                print(
                    f"  Topic #{t['topic_id']} (x{t['emergence_ratio']}) : {words}"
                )

        if not trending.empty:
            print("\n🟡 TERMES EN FORTE CROISSANCE :")
            for _, row in trending.head(10).iterrows():
                print(
                    f"  📈 {row['term']} : x{row['growth_ratio']} "
                    f"({row['baseline_avg_daily']}/j → {row['current_avg_daily']}/j)"
                )

        if not novel.empty:
            print("\n🆕 TERMES INHABITUELS / NOUVEAUX :")
            for _, row in novel.head(10).iterrows():
                tag = "🆕" if row["novelty"] == "nouveau" else "⚠️"
                print(
                    f"  {tag} {row['term']} : {row['recent_count']} occurrences "
                    f"({row['novelty']})"
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


# ──────────────────────────────────────────────────────────────────────
# EXEMPLE D'UTILISATION
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Génération de données de démo
    np.random.seed(42)
    dates = pd.date_range("2024-06-01", "2025-02-25", freq="D")

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
        ["Service incompétent je vais résilier", "j'en ai assez", "passez-moi un responsable"],
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
    # Conversations injectées pour tester la détection d'émergence
    emerging_conversations = [
        ["Problème avec la nouvelle mise à jour fibre", "rien ne marche depuis",
         "quand est-ce que ça sera réparé"],
        ["Depuis la migration fibre plus rien ne marche", "internet coupé",
         "c'est inadmissible"],
        ["Migration fibre catastrophique", "débit ridicule", "je veux résilier"],
    ]

    conv_id = 0
    rows = []
    for d in dates:
        n = np.random.randint(50, 200)
        n_normal = int(n * 0.75)
        n_negative = n - n_normal

        for _ in range(n_normal):
            conv = normal_conversations[np.random.randint(len(normal_conversations))]
            rows.append({
                "date": d,
                "conversation_id": f"conv_{conv_id:06d}",
                "verbatims": conv,
            })
            conv_id += 1

        for _ in range(n_negative):
            conv = negative_conversations[np.random.randint(len(negative_conversations))]
            rows.append({
                "date": d,
                "conversation_id": f"conv_{conv_id:06d}",
                "verbatims": conv,
            })
            conv_id += 1

        # Simuler un pic le 15 janvier 2025
        if d.date() == pd.Timestamp("2025-01-15").date():
            for _ in range(100):
                conv = negative_conversations[
                    np.random.randint(len(negative_conversations))
                ]
                rows.append({
                    "date": d,
                    "conversation_id": f"conv_{conv_id:06d}",
                    "verbatims": conv,
                })
                conv_id += 1

        # Simuler des termes émergents les 2 dernières semaines
        if d >= pd.Timestamp("2025-02-12"):
            for _ in range(30):
                conv = emerging_conversations[
                    np.random.randint(len(emerging_conversations))
                ]
                rows.append({
                    "date": d,
                    "conversation_id": f"conv_{conv_id:06d}",
                    "verbatims": conv,
                })
                conv_id += 1

    df_demo = pd.DataFrame(rows)

    # Lancement de l'analyse
    analyzer = VerbatimAnalyzer(
        df_demo,
        date_col="date",
        verbatim_col="verbatims",
        conversation_id_col="conversation_id",
        use_transformer=False,  # True pour CamemBERT (nécessite GPU)
        z_threshold=2.0,
        growth_window=7,
        growth_factor=3.0,
    )
    report = analyzer.run_full_analysis(save_dashboard="dashboard_verbatims.png")

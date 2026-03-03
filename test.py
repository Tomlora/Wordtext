    text = re.sub(r"(?<![a-zA-ZÀ-ÿ])'|'(?![a-zA-ZÀ-ÿ])", '', text)
    text = re.sub(r',\s*', ' ', text).strip()
    return text

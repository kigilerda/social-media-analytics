from pathlib import Path
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer


def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def load_wordlist(path):
    
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def build_stopwords():
    ensure_nltk()
    sw = set(nltk.corpus.stopwords.words("russian"))

    # extra_stop = load_wordlist(extra_stop_path)
    # keep_words = load_wordlist(keep_words_path)
    # sw.update(extra_stop)
    # sw.difference_update(keep_words)
    return sw


def duplicate_characters(word):
    # Убираем 'аааа', 'оооо', 'ахахах' и т.п.
    return re.search(r"(.)\1{2,}", word) is not None


def make_preprocessor():
    
    stopwords = build_stopwords()
    morph = MorphAnalyzer()

    def preprocess(text):
        if pd.isna(text):
            return ""

        s = str(text).lower()

        # ссылки/упоминания/почты
        s = re.sub(r"\[.*?\]\(https?:\/\/[^\s)]+\)", " ", s)   # markdown ссылки
        s = re.sub(r"http\S+|www\S+|https\S+", " ", s)         # обычные ссылки
        s = re.sub(r"tg://user\?id=\d+", " ", s)               # tg user id
        s = re.sub(r"@\w+", " ", s)                            # юзернеймы
        s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", s)      # почты

        # оставляем только буквы
        s = re.sub(r"[^a-zа-яё]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

        if not s:
            return ""

        tokens = word_tokenize(s, language="russian")

        # фильтры
        tokens = [t for t in tokens if len(t) > 1]
        tokens = [t for t in tokens if t not in stopwords]
        tokens = [t for t in tokens if not duplicate_characters(t)]

        # лемматизация
        lemmas = [morph.parse(t)[0].normal_form for t in tokens]
        return " ".join(lemmas)

    return preprocess

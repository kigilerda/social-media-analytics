# scripts/preprocess.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.text_cleaning import make_preprocessor


P = 0.90

RAW_PATH = Path("data/raw/data_base.csv")
SAMPLE_PATH = Path("data/sample/sample.csv")
OUT_DIR = Path("data/processed")


def main():
    # выбираем входной файл
    if RAW_PATH.exists():
        in_path = RAW_PATH
    elif SAMPLE_PATH.exists():
        in_path = SAMPLE_PATH
    else:
        raise FileNotFoundError("Нет ничего")

    print("[preprocess] input:", in_path)

    df = pd.read_csv(in_path)
    print("[preprocess] raw shape:", df.shape)

    # типы
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    for c in ["views", "reactions", "comments"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # фильтры качества
    before = len(df)
    df = df.dropna(subset=["date"])
    df = df[df["views"].fillna(0) > 0]
    df = df[df["reactions"].fillna(0) >= 0]
    df = df[df["comments"].fillna(0) >= 0]
    after = len(df)
    print(f"[preprocess] filters: {before} -> {after} (removed {before-after})")

    # таргет
    df["engagement"] = (df["reactions"].fillna(0) + df["comments"].fillna(0)) / df["views"]
    thr = df["engagement"].quantile(P)
    df["target"] = (df["engagement"] >= thr).astype(int)
    print(f"[preprocess] P={P} thr={thr:.6f}")
    print("[preprocess] target counts:\n", df["target"].value_counts())

    # фичи
    df["weekday"] = df["date"].dt.dayofweek
    df["hour"] = df["date"].dt.hour
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # фичи
    preprocess_fn = make_preprocessor()
    df["processed_text"] = df["text"].fillna("").apply(preprocess_fn)
    df["text_len"] = df["processed_text"].str.len()
    df["words_cnt"] = df["processed_text"].str.split().map(len)

    # split train/val/test
    train, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["target"])
    train, val = train_test_split(train, test_size=0.15, random_state=42, stratify=train["target"])
    print(f"[preprocess] split: train={len(train)} val={len(val)} test={len(test)}")

    # сохраняем
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "data_processed.csv", index=False)
    train.to_csv(OUT_DIR / "data_train.csv", index=False)
    val.to_csv(OUT_DIR / "data_val.csv", index=False)
    test.to_csv(OUT_DIR / "data_test.csv", index=False)

    print("[preprocess] saved to:", OUT_DIR)


if __name__ == "__main__":
    main()

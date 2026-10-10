"""Data and features shared by the benchmarks.

The cleaning and the TF-IDF settings are the ones in src/preprocess.py (a test checks that the
two produce the same matrix), so the CPU baselines see exactly the features the CUDA scripts see.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "fin_data_1.csv"
RESULTS_DIR = ROOT / "results"
SEED = 42            # the CUDA scripts that hold out data use random_state=42
TEST_SIZE = 0.2      # ... and test_size=0.2, stratified


def load_table(path: str | Path = DATA_PATH, deduplicate: bool = False) -> pd.DataFrame:
    """Read the CSV the way src/preprocess.py does; optionally remove repeated sentences.

    With `deduplicate=True` every sentence that appears with more than one label is dropped
    entirely (its true label is unknowable) and the other repeated sentences are kept once.
    """
    df = pd.read_csv(path)
    df.columns = ["Sentence", "Sentiment"]
    df = df.dropna(subset=["Sentence", "Sentiment"])
    df = df[df["Sentence"].apply(lambda x: isinstance(x, str))]
    if deduplicate:
        n_labels = df.groupby("Sentence")["Sentiment"].transform("nunique")
        df = df[n_labels == 1].drop_duplicates("Sentence")
    return df.reset_index(drop=True)


def data_quality(df: pd.DataFrame) -> dict:
    """Counts that bound what any model can score on this table."""
    labels_per_sentence = df.groupby("Sentence")["Sentiment"].nunique()
    return {
        "rows": int(len(df)),
        "unique_sentences": int(df["Sentence"].nunique()),
        "rows_in_repeated_sentences": int(df.duplicated("Sentence", keep=False).sum()),
        "sentences_with_conflicting_labels": int((labels_per_sentence > 1).sum()),
        "class_counts": {k: int(v) for k, v in df["Sentiment"].value_counts().sort_index().items()},
    }


def tfidf_features(df: pd.DataFrame) -> tuple[sparse.csr_matrix, "pd.Series", list[str]]:
    """TF-IDF (2,000 features, English stop words) of the cleaned sentences, plus integer labels."""
    cleaned = df["Sentence"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x.lower()))
    X = TfidfVectorizer(max_features=2000, stop_words="english").fit_transform(cleaned)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Sentiment"])
    return X.tocsr(), y, list(encoder.classes_)


def holdout_split(X, y):
    """The split used by the CUDA scripts that hold out data."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

"""Fixtures: a small synthetic sentiment table, so the tests need neither a GPU nor the real data."""
import numpy as np
import pandas as pd
import pytest

WORDS = {
    "positive": ["profit", "growth", "increase", "record", "strong", "gain"],
    "negative": ["loss", "decline", "fall", "weak", "layoffs", "drop"],
    "neutral": ["company", "board", "meeting", "located", "announced", "office"],
}


@pytest.fixture
def csv_path(tmp_path):
    """90 sentences, 30 per class, written to <tmp>/data/fin_data_1.csv with the real file's two columns."""
    rng = np.random.default_rng(0)
    rows = []
    for label, vocabulary in WORDS.items():
        for i in range(30):
            words = rng.choice(vocabulary, size=4).tolist() + [f"item{i}"]
            rows.append((f"The {' '.join(words)} was reported, 2024.", label))
    path = tmp_path / "data" / "fin_data_1.csv"
    path.parent.mkdir()
    pd.DataFrame(rows, columns=["Sentence", "Sentiment"]).to_csv(path, index=False)
    return path

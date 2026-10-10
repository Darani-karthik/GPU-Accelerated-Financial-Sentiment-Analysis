"""The dataset file and the figures the README quotes about it."""
import pandas as pd

from benchmarks.common import DATA_PATH, data_quality, load_table


def test_dataset_matches_the_readme():
    quality = data_quality(load_table(DATA_PATH))
    assert quality["rows"] == 5842
    assert quality["class_counts"] == {"negative": 860, "neutral": 3130, "positive": 1852}
    assert quality["unique_sentences"] == 5322
    assert quality["sentences_with_conflicting_labels"] == 514


def test_deduplicated_table_has_one_label_per_sentence():
    table = load_table(DATA_PATH, deduplicate=True)
    assert table["Sentence"].is_unique
    assert len(table) == 4808
    assert data_quality(table)["sentences_with_conflicting_labels"] == 0


def test_deduplication_drops_conflicting_sentences_entirely_and_keeps_agreeing_ones_once(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame(
        {"Sentence": ["a", "a", "b", "b", "c"], "Sentiment": ["positive", "negative", "neutral", "neutral", "positive"]}
    ).to_csv(path, index=False)
    table = load_table(path, deduplicate=True)
    assert sorted(table["Sentence"]) == ["b", "c"]

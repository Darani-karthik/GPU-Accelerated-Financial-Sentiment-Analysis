"""src/preprocess.py and the benchmarks must produce identical features."""
import numpy as np
import scipy.sparse

from benchmarks.common import load_table, tfidf_features
from src.preprocess import preprocess_and_save


def test_preprocess_writes_features_and_labels(csv_path, monkeypatch):
    monkeypatch.chdir(csv_path.parent.parent)
    preprocess_and_save("data/fin_data_1.csv")

    X = scipy.sparse.load_npz("data/processed/preprocessed_features.npz")
    y = np.load("data/processed/preprocessed_labels.npy")
    assert X.shape[0] == y.shape[0] == 90
    assert X.shape[1] <= 2000
    assert set(y) == {0, 1, 2}                              # negative, neutral, positive (alphabetical)
    assert np.bincount(y).tolist() == [30, 30, 30]


def test_benchmark_features_equal_the_cuda_scripts_features(csv_path, monkeypatch):
    monkeypatch.chdir(csv_path.parent.parent)
    preprocess_and_save("data/fin_data_1.csv")
    X_script = scipy.sparse.load_npz("data/processed/preprocessed_features.npz")
    y_script = np.load("data/processed/preprocessed_labels.npy")

    X, y, classes = tfidf_features(load_table(csv_path))

    assert classes == ["negative", "neutral", "positive"]
    assert (y == y_script).all()
    assert abs(X - X_script).max() < 1e-12


def test_preprocess_reports_a_missing_file_without_crashing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    preprocess_and_save("data/absent.csv")
    assert "was not found" in capsys.readouterr().out

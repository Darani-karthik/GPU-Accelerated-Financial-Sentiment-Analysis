"""The CPU baselines and the CUDA-script runner (everything that does not need a GPU)."""
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

from benchmarks import cpu_baselines, run_scripts
from benchmarks.common import ROOT, holdout_split, load_table, tfidf_features


# ── Output parsing ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "output, expected",
    [
        ("Block runtime: 1.2 seconds\n\nDeeper ANN Final Accuracy: 71.23%\n", 71.23),
        ("Runtime: 3.200s | Accuracy: 70.00%\n", 70.0),
        ("Train accuracy: 91.00%\nTest accuracy: 68.50%\n", 68.5),
        ("  Epoch 800/800 - Training Acc: 99.00%\n\nSVM (One-vs-Rest) Final Accuracy: 64.10%\n", 64.1),
        ("  - Accuracy: 55.55%\n  - Total Runtime: 3.0s\n", 55.55),
        ("Epoch 1/10 - Training Accuracy: 33%\n", None),
        ("no accuracy was printed\n", None),
    ],
)
def test_parse_accuracy(output, expected):
    assert run_scripts.parse_accuracy(output) == expected


# ── The script table ─────────────────────────────────────────────────────────
def test_every_cuda_script_is_in_the_runner_table():
    on_disk = {p.relative_to(ROOT).as_posix() for folder in ("tfidf", "svd") for p in (ROOT / "src" / folder).glob("*.py")}
    assert {s.path for s in run_scripts.SCRIPTS} == on_disk


def test_script_table_is_consistent():
    keys = [s.key for s in run_scripts.SCRIPTS]
    assert len(keys) == len(set(keys))
    assert all((ROOT / s.path).is_file() for s in run_scripts.SCRIPTS)
    assert {s.features for s in run_scripts.SCRIPTS} == {"tfidf", "svd"}
    assert {s.basis for s in run_scripts.SCRIPTS} == {run_scripts.ALL_ROWS, run_scripts.HELD_OUT}
    assert (ROOT / run_scripts.PREPROCESS).is_file()


def test_held_out_label_matches_scripts_that_split_the_data():
    """A script is 'held-out' exactly when it calls train_test_split."""
    for script in run_scripts.SCRIPTS:
        splits = "train_test_split(" in Path(ROOT / script.path).read_text(encoding="utf-8")
        assert splits == (script.basis == run_scripts.HELD_OUT), script.path


# ── Summaries ────────────────────────────────────────────────────────────────
def test_summarise_uses_the_median_and_keeps_the_range():
    runs = [{"status": "ok", "seconds": s, "accuracy": a} for s, a in [(3.0, 60.0), (1.0, 62.0), (2.0, 61.0)]]
    summary = run_scripts.summarise(runs)
    assert summary["seconds_median"] == 2.0 and (summary["seconds_min"], summary["seconds_max"]) == (1.0, 3.0)
    assert summary["accuracy"] == 61.0 and (summary["accuracy_min"], summary["accuracy_max"]) == (60.0, 62.0)


def test_summarise_reports_a_failed_run():
    summary = run_scripts.summarise([{"status": "ok", "seconds": 1.0, "accuracy": 50.0},
                                     {"status": "failed", "seconds": 0.5, "accuracy": None, "error": "boom"}])
    assert summary["status"] == "failed" and summary["error"] == "boom" and summary["accuracy"] is None


def test_markdown_table_marks_failures_and_missing_accuracy():
    report = {
        "environment": {"gpu": "Test GPU", "gpu_memory_gb": 8.0, "cupy": "13", "python": "3.12", "cuda_runtime": 12090},
        "repeats": 1,
        "scripts": {
            "svd/svm": {"status": "ok", "seconds_median": 12.34, "seconds_min": 12.34, "seconds_max": 12.34,
                        "accuracy": 64.5, "accuracy_min": 64.5, "accuracy_max": 64.5, "runs": 1},
            "tfidf/ann": {"status": "timeout", "seconds": 1800.0, "accuracy": None},
        },
    }
    text = run_scripts.to_markdown(report)
    assert "| `svd/svm.py` | Linear SVM | 64.50 | all rows (training data) | 12.3 | ok |" in text
    assert "| `tfidf/ann.py` | ANN (128-64) | n/a | all rows (training data) | n/a | timeout |" in text


# ── CPU baselines ────────────────────────────────────────────────────────────
SOFTMAX = {"Softmax": (lambda: LogisticRegression(max_iter=500), False)}


def test_holdout_split_is_stratified_80_20(csv_path):
    X, y, _ = tfidf_features(load_table(csv_path))
    X_train, X_test, y_train, y_test = holdout_split(X, y)
    assert (len(y_train), len(y_test)) == (72, 18)
    assert sorted(set(y_test)) == [0, 1, 2]


def test_cpu_baselines_score_a_separable_table(csv_path):
    setup = cpu_baselines.run_setup(csv_path, deduplicate=False, models=SOFTMAX)
    assert setup["n_train"] == 72 and setup["n_test"] == 18
    assert setup["models"]["Always the majority class"]["accuracy"] == pytest.approx(1 / 3, abs=0.05)
    assert setup["models"]["Softmax"]["accuracy"] > 0.9              # the synthetic classes use disjoint words
    assert setup["data_quality"]["rows"] == 90


def test_cpu_baselines_markdown(csv_path):
    setups = {key: cpu_baselines.run_setup(csv_path, key == "deduplicated", SOFTMAX) for key in ("as_is", "deduplicated")}
    text = cpu_baselines.to_markdown({"setups": setups})
    assert "## Table as it is" in text and "## Repeated sentences removed" in text and "| Softmax |" in text

"""CPU reference models on the CUDA scripts' TF-IDF features.

    python -m benchmarks.cpu_baselines

Standard scikit-learn models with default settings (no tuning) on the same features and the same
stratified 80/20 split (random_state=42) as the CUDA scripts that hold out data. They show what
accuracy this data supports, so the accuracy printed by a CUDA script can be put in context.
Two versions of the table are scored: as it is, and with repeated sentences removed
(see data_quality in the output).

Writes results/cpu_baselines.json and results/cpu_baselines.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from .common import DATA_PATH, RESULTS_DIR, SEED, data_quality, holdout_split, load_table, tfidf_features

#: name -> (factory, needs a dense matrix)
MODELS = {
    "Softmax regression": (lambda: LogisticRegression(max_iter=1000), False),
    "Linear SVM": (lambda: LinearSVC(), False),
    "Ridge classifier": (lambda: RidgeClassifier(), False),
    "Random forest (100 trees)": (lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), False),
    "Gradient boosting (histogram)": (lambda: HistGradientBoostingClassifier(max_iter=100, random_state=SEED), True),
    "MLP (128-64)": (lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=SEED), False),
}


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def run_setup(path: str | Path, deduplicate: bool, models: dict | None = None) -> dict:
    """Fit every model on the training part and score it on the held-out 20 %."""
    df = load_table(path, deduplicate=deduplicate)
    X, y, classes = tfidf_features(df)
    X_train, X_test, y_train, y_test = holdout_split(X, y)
    rows = {}
    majority = np.full_like(y_test, np.bincount(y_train).argmax())
    rows["Always the majority class"] = {**score(y_test, majority), "fit_seconds": 0.0}
    for name, (factory, dense) in (models or MODELS).items():
        train, test = (X_train.toarray(), X_test.toarray()) if dense else (X_train, X_test)
        model = factory()
        start = time.perf_counter()
        model.fit(train, y_train)
        elapsed = time.perf_counter() - start
        rows[name] = {**score(y_test, model.predict(test)), "fit_seconds": round(elapsed, 2)}
    return {"data_quality": data_quality(df), "classes": classes, "n_train": int(len(y_train)), "n_test": int(len(y_test)), "models": rows}


def to_markdown(results: dict) -> str:
    titles = {"as_is": "Table as it is", "deduplicated": "Repeated sentences removed"}
    out = ["# CPU baselines", "",
           "Held-out 20 % (stratified, random_state=42) of the TF-IDF features used by the CUDA scripts. "
           "Default scikit-learn settings, no tuning. Fit times are single-run wall-clock seconds on this machine.", ""]
    for key, setup in results["setups"].items():
        q = setup["data_quality"]
        out += [f"## {titles[key]}", "",
                f"{q['rows']:,} rows, {q['unique_sentences']:,} distinct sentences, {setup['n_train']:,} train / {setup['n_test']:,} test. "
                f"Sentences that appear with more than one label: {q['sentences_with_conflicting_labels']}.", "",
                "| Model | Accuracy (%) | Balanced accuracy (%) | Macro-F1 | Fit time (s) |", "|---|---|---|---|---|"]
        for name, r in setup["models"].items():
            out.append(f"| {name} | {r['accuracy'] * 100:.1f} | {r['balanced_accuracy'] * 100:.1f} | {r['macro_f1']:.3f} | {r['fit_seconds']:.2f} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    results = {"seed": SEED, "setups": {key: run_setup(args.data, dedupe) for key, dedupe in (("as_is", False), ("deduplicated", True))}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cpu_baselines.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = to_markdown(results)
    (args.out / "cpu_baselines.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

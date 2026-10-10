"""Run the CUDA scripts one by one and record how long each takes and what accuracy it prints.

    python -m benchmarks.run_scripts                 # every script, 1 run each
    python -m benchmarks.run_scripts --repeats 3     # median of 3 runs
    python -m benchmarks.run_scripts --only svd/svm tfidf/ann

Needs an NVIDIA GPU with CuPy. The scripts are run unchanged, from the repository root, exactly as the
README describes; this file only starts them and reads their output. Writes results/cuda_runs.json
and results/cuda_runs.md.

Timing is the wall-clock time of the whole `python script.py` process: it includes importing CuPy,
compiling the kernels at run time and copying data to the GPU, not just the kernel launches.
Accuracy is whatever the script prints, and the `basis` column says what it was measured on: some
scripts score the same rows they trained on, others a held-out 20 % (see README, "Notes and limitations").
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .common import RESULTS_DIR, ROOT

ALL_ROWS = "all rows (training data)"
HELD_OUT = "held-out 20 %"


@dataclass(frozen=True)
class Script:
    path: str            # relative to the repository root
    model: str
    features: str        # "tfidf" or "svd"
    basis: str           # what the printed accuracy is measured on

    @property
    def key(self) -> str:
        return self.path.removeprefix("src/").removesuffix(".py")


SCRIPTS = [
    Script("src/tfidf/ann.py", "ANN (128-64)", "tfidf", ALL_ROWS),
    Script("src/tfidf/softmax_regression.py", "Softmax regression", "tfidf", ALL_ROWS),
    Script("src/tfidf/softmax_regression_tiled.py", "Softmax regression, tiled", "tfidf", ALL_ROWS),
    Script("src/tfidf/svm.py", "Linear SVM", "tfidf", ALL_ROWS),
    Script("src/tfidf/ridge_classifier.py", "Ridge classifier", "tfidf", HELD_OUT),
    Script("src/tfidf/random_forest.py", "Random forest (stumps)", "tfidf", HELD_OUT),
    Script("src/tfidf/gradient_boosting.py", "Gradient boosting (stumps)", "tfidf", HELD_OUT),
    Script("src/svd/ann.py", "ANN (128-64)", "svd", ALL_ROWS),
    Script("src/svd/softmax_regression.py", "Softmax regression", "svd", ALL_ROWS),
    Script("src/svd/svm.py", "Linear SVM", "svd", ALL_ROWS),
    Script("src/svd/ridge_classifier.py", "Ridge classifier", "svd", HELD_OUT),
    Script("src/svd/random_forest.py", "Random forest (stumps)", "svd", HELD_OUT),
    Script("src/svd/gradient_boosting.py", "Gradient boosting (stumps)", "svd", HELD_OUT),
]

PREPROCESS = "src/preprocess.py"
_ACCURACY = re.compile(r"accuracy\D*?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


def parse_accuracy(output: str) -> float | None:
    """The last accuracy (in %) a script printed, ignoring per-epoch and training-accuracy lines."""
    found = None
    for line in output.splitlines():
        lowered = line.lower()
        if "epoch" in lowered or "train" in lowered:
            continue
        match = _ACCURACY.search(line)
        if match:
            found = float(match.group(1))
    return found


def run_one(path: str, timeout: float) -> dict:
    """Run one script as a subprocess from the repository root."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    start = time.perf_counter()
    try:
        done = subprocess.run([sys.executable, path], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": round(time.perf_counter() - start, 2), "accuracy": None}
    seconds = round(time.perf_counter() - start, 2)
    if done.returncode != 0:
        return {"status": "failed", "seconds": seconds, "accuracy": None, "error": (done.stderr or done.stdout).strip()[-600:]}
    return {"status": "ok", "seconds": seconds, "accuracy": parse_accuracy(done.stdout)}


def environment() -> dict:
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "gpu": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
        "gpu_memory_gb": round(props["totalGlobalMem"] / 1024 ** 3, 1),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cupy": cp.__version__,
        "python": sys.version.split()[0],
    }


def summarise(runs: list[dict]) -> dict:
    ok = [r for r in runs if r["status"] == "ok"]
    if len(ok) != len(runs):
        failed = next(r for r in runs if r["status"] != "ok")
        return {k: failed[k] for k in ("status", "seconds", "error") if k in failed} | {"accuracy": None}
    seconds = [r["seconds"] for r in ok]
    accuracies = [r["accuracy"] for r in ok if r["accuracy"] is not None]
    return {
        "status": "ok",
        "seconds_median": round(statistics.median(seconds), 2),
        "seconds_min": min(seconds),
        "seconds_max": max(seconds),
        "accuracy": round(statistics.median(accuracies), 2) if accuracies else None,
        "accuracy_min": min(accuracies) if accuracies else None,
        "accuracy_max": max(accuracies) if accuracies else None,
        "runs": len(ok),
    }


def to_markdown(report: dict) -> str:
    env = report["environment"]
    out = ["# CUDA script runs", "",
           f"{env['gpu']} ({env['gpu_memory_gb']} GB), CuPy {env['cupy']}, Python {env['python']}. "
           f"Each script run unchanged with `python <script>`, {report['repeats']} run(s) each; times are the median wall-clock seconds "
           "of the whole process (CuPy import, run-time kernel compilation and data transfer included).", ""]
    for features, title in (("tfidf", "TF-IDF features (`src/tfidf/`)"), ("svd", "SVD features (`src/svd/`)")):
        out += [f"## {title}", "", "| Script | Model | Accuracy printed (%) | Measured on | Time (s) | Status |", "|---|---|---|---|---|---|"]
        for s in SCRIPTS:
            if s.features != features:
                continue
            r = report["scripts"].get(s.key)
            if r is None:
                continue
            if r["status"] != "ok":
                out.append(f"| `{s.path.removeprefix('src/')}` | {s.model} | n/a | {s.basis} | n/a | {r['status']} |")
                continue
            spread = "" if r["runs"] == 1 or r["accuracy_min"] == r["accuracy_max"] else f" ({r['accuracy_min']:.1f} to {r['accuracy_max']:.1f})"
            accuracy = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.2f}{spread}"
            out.append(f"| `{s.path.removeprefix('src/')}` | {s.model} | {accuracy} | {s.basis} | {r['seconds_median']:.1f} | ok |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=1, help="runs per script (default 1)")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds before a script is stopped (default 1800)")
    parser.add_argument("--only", nargs="*", help="script keys such as svd/svm or tfidf/ann (default: all)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    chosen = [s for s in SCRIPTS if not args.only or s.key in args.only]
    unknown = set(args.only or []) - {s.key for s in SCRIPTS}
    if unknown:
        parser.error(f"unknown script(s): {sorted(unknown)}; choose from {[s.key for s in SCRIPTS]}")

    report = {"environment": environment(), "repeats": args.repeats, "scripts": {}}
    prep = run_one(PREPROCESS, args.timeout)
    if prep["status"] != "ok":
        raise SystemExit(f"{PREPROCESS} failed: {prep.get('error', prep['status'])}")
    for script in chosen:
        runs = [run_one(script.path, args.timeout) for _ in range(args.repeats)]
        report["scripts"][script.key] = summarise(runs)
        print(f"{script.key:32s} {report['scripts'][script.key]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cuda_runs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    (args.out / "cuda_runs.md").write_text(markdown + "\n", encoding="utf-8")
    print("\n" + markdown)


if __name__ == "__main__":
    main()

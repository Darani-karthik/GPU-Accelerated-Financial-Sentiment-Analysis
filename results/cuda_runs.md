# CUDA script runs

NVIDIA GeForce RTX 4070 Laptop GPU (8.0 GB), CuPy 14.2.0, Python 3.10.11. Each script run unchanged with `python <script>`, 3 run(s) each; times are the median wall-clock seconds of the whole process (CuPy import, run-time kernel compilation and data transfer included).

## TF-IDF features (`src/tfidf/`)

| Script | Model | Accuracy printed (%) | Measured on | Time (s) | Status |
|---|---|---|---|---|---|
| `tfidf/ann.py` | ANN (128-64) | 73.79 (73.3 to 74.2) | all rows (training data) | 13.0 | ok |
| `tfidf/softmax_regression.py` | Softmax regression | 68.45 | all rows (training data) | 1.4 | ok |
| `tfidf/softmax_regression_tiled.py` | Softmax regression, tiled | 68.45 | all rows (training data) | 1.5 | ok |
| `tfidf/svm.py` | Linear SVM | 53.58 | all rows (training data) | 7.3 | ok |
| `tfidf/ridge_classifier.py` | Ridge classifier | 67.32 | held-out 20 % | 1.8 | ok |
| `tfidf/random_forest.py` | Random forest (stumps) | 53.55 | held-out 20 % | 2.3 | ok |
| `tfidf/gradient_boosting.py` | Gradient boosting (stumps) | 56.89 (55.9 to 56.9) | held-out 20 % | 220.0 | ok |

## SVD features (`src/svd/`)

| Script | Model | Accuracy printed (%) | Measured on | Time (s) | Status |
|---|---|---|---|---|---|
| `svd/ann.py` | ANN (128-64) | 67.72 (66.9 to 67.8) | all rows (training data) | 3.8 | ok |
| `svd/softmax_regression.py` | Softmax regression | 66.86 | all rows (training data) | 6.0 | ok |
| `svd/svm.py` | Linear SVM | 62.86 | all rows (training data) | 3.3 | ok |
| `svd/ridge_classifier.py` | Ridge classifier | 67.49 | held-out 20 % | 3.1 | ok |
| `svd/random_forest.py` | Random forest (stumps) | 53.55 | held-out 20 % | 3.2 | ok |
| `svd/gradient_boosting.py` | Gradient boosting (stumps) | 53.55 | held-out 20 % | 8.0 | ok |


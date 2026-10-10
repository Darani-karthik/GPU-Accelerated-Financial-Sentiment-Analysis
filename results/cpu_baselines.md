# CPU baselines

Held-out 20 % (stratified, random_state=42) of the TF-IDF features used by the CUDA scripts. Default scikit-learn settings, no tuning. Fit times are single-run wall-clock seconds on this machine.

## Table as it is

5,842 rows, 5,322 distinct sentences, 4,673 train / 1,169 test. Sentences that appear with more than one label: 514.

| Model | Accuracy (%) | Balanced accuracy (%) | Macro-F1 | Fit time (s) |
|---|---|---|---|---|
| Always the majority class | 53.5 | 33.3 | 0.233 | 0.00 |
| Softmax regression | 69.5 | 55.4 | 0.562 | 0.06 |
| Linear SVM | 65.4 | 54.6 | 0.549 | 0.02 |
| Ridge classifier | 67.1 | 54.4 | 0.545 | 0.01 |
| Random forest (100 trees) | 63.3 | 51.3 | 0.518 | 0.52 |
| Gradient boosting (histogram) | 62.6 | 50.7 | 0.512 | 19.68 |
| MLP (128-64) | 60.6 | 51.3 | 0.514 | 23.37 |

## Repeated sentences removed

4,808 rows, 4,808 distinct sentences, 3,846 train / 962 test. Sentences that appear with more than one label: 0.

| Model | Accuracy (%) | Balanced accuracy (%) | Macro-F1 | Fit time (s) |
|---|---|---|---|---|
| Always the majority class | 54.3 | 33.3 | 0.234 | 0.00 |
| Softmax regression | 78.2 | 63.6 | 0.673 | 0.04 |
| Linear SVM | 77.0 | 67.2 | 0.692 | 0.01 |
| Ridge classifier | 78.2 | 66.2 | 0.693 | 0.01 |
| Random forest (100 trees) | 78.5 | 65.8 | 0.682 | 0.19 |
| Gradient boosting (histogram) | 72.2 | 59.4 | 0.618 | 17.96 |
| MLP (128-64) | 73.5 | 65.7 | 0.665 | 14.73 |


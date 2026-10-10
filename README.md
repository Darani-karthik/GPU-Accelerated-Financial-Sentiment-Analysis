# GPU-Accelerated Financial Sentiment Analysis

[![tests](https://github.com/Darani-karthik/GPU-Accelerated-Financial-Sentiment-Analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/Darani-karthik/GPU-Accelerated-Financial-Sentiment-Analysis/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Machine-learning and deep-learning models for financial sentiment classification (positive / neutral / negative), where the compute-heavy step of each model is written as a **custom CUDA C++ kernel** and driven from Python through [CuPy](https://cupy.dev/).

The goal is to look inside the models rather than call library routines: for each algorithm, find the operation that dominates the runtime (a sparse gradient update, a split search, a matrix multiply) and parallelise it by hand on the GPU.

## Contents

- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Approach](#approach)
- [Classical models](#classical-models)
- [Transformer experiments](#transformer-experiments)
- [Results](#results)
- [Getting started](#getting-started)
- [Notes and limitations](#notes-and-limitations)
- [Future work](#future-work)
- [Tech stack](#tech-stack)
- [Contributors](#contributors)
- [License](#license)

## Dataset

[`data/fin_data_1.csv`](data/fin_data_1.csv) holds English-language sentences from financial news, each labelled positive, neutral or negative. The project was originally described as using the Financial PhraseBank dataset, but this file has 5,842 rows where Financial PhraseBank has 4,846, and its origin and licence are not recorded here, so treat it as a financial-news sentiment dataset of unrecorded source.

| | |
|---|---|
| Rows | 5,842 |
| Columns | `Sentence`, `Sentiment` |
| neutral | 3,130 |
| positive | 1,852 |
| negative | 860 |
| Distinct sentences | 5,322 |
| Sentences that appear with more than one label | 514 |

The classes are imbalanced (neutral is about 54% of the data). The Transformer notebooks counter this with random oversampling of the minority classes. The repeated sentences with conflicting labels put a ceiling on the accuracy any model can reach; see [Results](#results).

## Repository structure

```
GPU-Accelerated-Financial-Sentiment-Analysis/
├── .github/workflows/tests.yml      # CI: the CPU-only tests on Python 3.10 and 3.12
├── benchmarks/                      # runs the CUDA scripts and the CPU reference models
│   ├── run_scripts.py
│   └── cpu_baselines.py
├── data/
│   └── fin_data_1.csv               # raw dataset
│                                    # (data/processed/ is generated, git-ignored)
├── results/                         # committed output of benchmarks/ (JSON and Markdown)
├── src/
│   ├── preprocess.py                # CSV -> TF-IDF features + labels
│   ├── tfidf/                       # models trained on sparse TF-IDF features
│   │   ├── ann.py
│   │   ├── ridge_classifier.py
│   │   ├── softmax_regression.py
│   │   ├── softmax_regression_tiled.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   └── gradient_boosting.py
│   └── svd/                         # the same families on dense SVD embeddings
│       ├── ann.py
│       ├── ridge_classifier.py
│       ├── softmax_regression.py
│       ├── svm.py
│       ├── random_forest.py
│       └── gradient_boosting.py
├── notebooks/                       # Transformer experiments, phases 1-9
├── tests/                           # CPU-only tests (no GPU needed)
├── LICENSE
├── pytest.ini
├── requirements.txt
└── README.md
```

## Approach

The project follows a **hybrid CUDA model**:

- **CuPy** handles the standard GPU work: array management, dense and sparse matrix products, reductions.
- **Custom CUDA C++ kernels** (compiled at runtime with `cupy.RawKernel`) handle the part that is specific to each algorithm. Patterns used across the code base:
  - **CSR sparse traversal**: forward products and gradient updates read the TF-IDF matrix directly in CSR form.
  - **`atomicAdd` scatter updates**: many threads accumulate into the same weight.
  - **`atomicMin` reductions**: thousands of threads compete to find the best tree split.
  - **Shared-memory tiling**: used in the tiled softmax kernels and in the Transformer matrix multiply.
  - **Conjugate Gradient**: the ridge classifier solves `(XᵀX + αI)w = Xᵀy` iteratively, with custom mat-vec kernels.

Two feature pipelines are compared:

1. **TF-IDF** (2,000 features, sparse), produced by `src/preprocess.py`.
2. **SVD embeddings** (dense; 256, 300 or 512 components depending on the script): TF-IDF reduced with truncated SVD.

## Classical models

Each model has a script for both feature pipelines.

| Model | What the custom kernel accelerates | `src/tfidf/` | `src/svd/` |
|---|---|---|---|
| **ANN** (hidden layers 128 → 64, ReLU) | First-layer weight update. Sparse version: CSR traversal with `atomicAdd`. Dense version: 2-D grid over the weight matrix. | `ann.py` | `ann.py` |
| **Softmax regression** | Cross-entropy gradient and weight update, one block per sample. Tiled variants use shared memory (sample-tiled and feature-tiled). | `softmax_regression.py`, `softmax_regression_tiled.py` | `softmax_regression.py` |
| **Linear SVM** (one-vs-rest, hinge loss) | Feature-parallel sub-gradient update. | `svm.py` | `svm.py` |
| **Ridge classifier** | Sparse (`csr_mv_vector`, `csr_mvt_improved`) or dense (`dense_mv`, `dense_mtv`) mat-vec products inside a Conjugate Gradient solver. | `ridge_classifier.py` | `ridge_classifier.py` |
| **Random forest** (10 stump trees) | Parallel best-split search over feature/threshold candidates (Gini impurity), plus a prediction kernel. | `random_forest.py` | `random_forest.py` |
| **Gradient boosting** (one-vs-rest regression stumps) | Best-split search minimising MSE, reduced with `atomicMin`. | `gradient_boosting.py` | `gradient_boosting.py` |

## Transformer experiments

[`notebooks/`](notebooks/) contains the second phase: small Transformer-style classifiers (`d_model` of 32 or 64, Q/K/V projections and a feed-forward layer) whose matrix multiplications, ReLU and row-softmax run on custom CUDA kernels. The notebooks compare different input representations and different matrix-multiply back-ends. Notebook 09 trains a plain MLP instead.

| Notebook | Focus |
|---|---|
| `01_ohe_transformer` | One-hot-encoded input with random oversampling; baseline Transformer. |
| `02_svd_embeddings_generation` | Builds word embeddings on the GPU: co-occurrence matrix → PPMI → SVD. |
| `03_svd_transformer` | Transformer on the SVD(PPMI) embeddings (co-occurrence and PPMI kernels). |
| `04_nmf_transformer` | Transformer on NMF features computed from GPU TF-IDF. |
| `05_matrix_tiling` | Shared-memory tiled matrix multiplication (`matmul_tiled`). |
| `05b_matrix_tiling_sample` | Variant of the tiled-matmul demo. |
| `06_tfidf_matrix_tiling` | TF-IDF + NMF features with the tiled matrix multiply. |
| `07_sentence_bert_transformer` | Pretrained Sentence-BERT embeddings (`all-MiniLM-L6-v2`) as input. |
| `08_mma_vs_tiling_transformer` | Compares FP16 Tensor-Core matmul (cuBLAS "MMA") against the custom tiled kernel inside the Transformer. |
| `09_mma_vs_tiling_mlp_training` | The same MMA vs. tiled comparison for MLP training on Sentence-BERT embeddings. |

## Results

The numbers below are produced by [`benchmarks/`](benchmarks/) and committed in [`results/`](results/). The CUDA runs were measured on an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB) with CuPy 14.2.0 and Python 3.10.11. The Transformer notebooks are not benchmarked.

### CUDA scripts

Every script was run unchanged, three times; the time is the median wall-clock seconds of the whole process, including the CuPy import, run-time kernel compilation and data transfer. *Measured on* says what the printed accuracy covers, which differs between scripts. Where the three runs disagreed, the range is in brackets. Full table: [results/cuda_runs.md](results/cuda_runs.md).

| Script | Model | Accuracy printed (%) | Measured on | Time (s) |
|---|---|---|---|---|
| `tfidf/ann.py` | ANN (128-64) | 73.79 (73.3 to 74.2) | all rows (training data) | 13.0 |
| `tfidf/softmax_regression.py` | Softmax regression | 68.45 | all rows (training data) | 1.4 |
| `tfidf/softmax_regression_tiled.py` | Softmax regression, tiled | 68.45 | all rows (training data) | 1.5 |
| `tfidf/svm.py` | Linear SVM | 53.58 | all rows (training data) | 7.3 |
| `tfidf/ridge_classifier.py` | Ridge classifier | 67.32 | held-out 20 % | 1.8 |
| `tfidf/random_forest.py` | Random forest (stumps) | 53.55 | held-out 20 % | 2.3 |
| `tfidf/gradient_boosting.py` | Gradient boosting (stumps) | 56.89 (55.9 to 56.9) | held-out 20 % | 220.0 |
| `svd/ann.py` | ANN (128-64) | 67.72 (66.9 to 67.8) | all rows (training data) | 3.8 |
| `svd/softmax_regression.py` | Softmax regression | 66.86 | all rows (training data) | 6.0 |
| `svd/svm.py` | Linear SVM | 62.86 | all rows (training data) | 3.3 |
| `svd/ridge_classifier.py` | Ridge classifier | 67.49 | held-out 20 % | 3.1 |
| `svd/random_forest.py` | Random forest (stumps) | 53.55 | held-out 20 % | 3.2 |
| `svd/gradient_boosting.py` | Gradient boosting (stumps) | 53.55 | held-out 20 % | 8.0 |

### CPU reference models

Standard scikit-learn models with default settings and no tuning, on the same TF-IDF features and the same stratified 80/20 split (`random_state=42`) as the CUDA scripts that hold out data. The fit time is a single-run wall-clock figure for the fit only. Full table, including a second version with repeated sentences removed: [results/cpu_baselines.md](results/cpu_baselines.md).

| Model | Accuracy (%) | Balanced accuracy (%) | Macro-F1 | Fit time (s) |
|---|---|---|---|---|
| Always the majority class | 53.5 | 33.3 | 0.233 | 0.00 |
| Softmax regression | 69.5 | 55.4 | 0.562 | 0.06 |
| Linear SVM | 65.4 | 54.6 | 0.549 | 0.02 |
| Ridge classifier | 67.1 | 54.4 | 0.545 | 0.01 |
| Random forest (100 trees) | 63.3 | 51.3 | 0.518 | 0.52 |
| Gradient boosting (histogram) | 62.6 | 50.7 | 0.512 | 19.68 |
| MLP (128-64) | 60.6 | 51.3 | 0.514 | 23.37 |

### What the numbers say

- **The CUDA ridge classifier matches its CPU counterpart** (67.3% against 67.1% on the same held-out rows), so that kernel is doing the same job.
- **The stump forests are no better than a constant guess.** The CUDA random forest and the SVD gradient boosting score 53.55%, the same as always predicting neutral (53.5%), because depth-one trees cannot do much here. The CPU random forest, with full trees, reaches 63.3%.
- **Accuracies marked "all rows" are training accuracy**, so they are not comparable with the held-out ones.
- **The data limits the accuracy.** With the 514 conflicting repeated sentences removed (4,808 rows left), the CPU models reach 72 to 79% instead of 61 to 70%, while the majority-class baseline stays near 54%.
- **There is no speed-up claim.** The CUDA times cover the whole process and the CPU times cover the fit only, so they cannot be compared. With 5,842 sentences the GPU has little to gain; the scripts demonstrate the kernels.

## Getting started

### Requirements

- An NVIDIA GPU with a CUDA toolkit installed (every model script and most notebooks run on the GPU).
- Python 3

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `cupy-cuda12x`; swap it for the wheel that matches your CUDA version (for example `cupy-cuda11x`).

### Running the classical models

All scripts expect to be run **from the repository root**, because they use paths relative to it (`data/...`).

```bash
# 1. TF-IDF features -> data/processed/preprocessed_features.npz and preprocessed_labels.npy
python src/preprocess.py

# 2. Any model on TF-IDF features
python src/tfidf/ann.py
python src/tfidf/svm.py

# 3. SVD-embedding models
python src/svd/ann.py
```

`src/preprocess.py` is a prerequisite for everything in `src/tfidf/` and for `src/svd/random_forest.py` and `src/svd/softmax_regression.py`. The other four `src/svd/` scripts read the CSV directly and build their own TF-IDF + SVD features.

### Reproducing the results and running the tests

```bash
python -m benchmarks.cpu_baselines               # CPU reference models (no GPU needed)
python -m benchmarks.run_scripts --repeats 3     # every CUDA script, median of 3 runs (needs the GPU)

pip install pytest
python -m pytest                                 # CPU-only tests
```

The tests cover the preprocessing, the benchmark code (output parsing, summaries, CPU baselines) and the dataset figures quoted in this README. They do not run the CUDA kernels; GitHub Actions runs them on Python 3.10 and 3.12.

### Running the notebooks

Start Jupyter from inside `notebooks/` so that the relative path `../data/fin_data_1.csv` resolves:

```bash
cd notebooks
jupyter notebook
```

On Google Colab, upload `fin_data_1.csv` and change the `read_csv` path in the notebook you want to run. Notebooks 07-09 download the Sentence-BERT model on first run.

## Notes and limitations

- **Accuracy numbers are not directly comparable across scripts.** Some scripts report accuracy on the full dataset they trained on (for example `tfidf/ann.py`, `tfidf/svm.py`, the softmax scripts), others on a held-out 20% split (random forest, gradient boosting, ridge). The notebooks print a mix of training and validation accuracy. Treat each script's output as a smoke test of that implementation, not as a leaderboard.
- **No speed-up is claimed.** The timings in [Results](#results) are whole-process wall-clock times on one laptop GPU, not kernel timings, and they are not set against CPU times for the same work.
- **The CUDA scripts need a GPU.** `cupy` and an NVIDIA device are required to run them. Only `src/preprocess.py`, the benchmarks' CPU baselines and the tests run on a CPU-only machine.
- **The dataset contains repeated sentences with conflicting labels** (514 of them), which caps the accuracy of every model.
- The Sentence-BERT and tensor-core notebooks need a recent NVIDIA GPU for FP16 Tensor-Core throughput.

## Future work

- Optimise the kernels further for scalability, especially the Transformer's attention.
- Try other embeddings (GloVe, FastText) and long-document architectures such as Longformer or BigBird.
- Run every CUDA script on the same held-out split, so that all of them can be compared with the CPU baselines directly.
- Replace the depth-one trees in the forest and boosting kernels with deeper ones.
- Explore deploying the final model on a live financial-news feed.

## Tech stack

- **Language:** Python, CUDA C++
- **GPU:** CuPy (`RawKernel`), cuBLAS Tensor Cores
- **Data and ML utilities:** NumPy, SciPy, pandas, scikit-learn
- **Deep learning / embeddings:** PyTorch, Sentence-Transformers (Sentence-BERT)

## Contributors

A team project, built by:

- Darani Karthik ([@Darani-karthik](https://github.com/Darani-karthik))
- Mahizhan S ([@Mahizhan-S](https://github.com/Mahizhan-S))
- Aakash Raj ([@Aakash-R1](https://github.com/Aakash-R1))
- Pavani Akshaya ([@Pavaniakshaya](https://github.com/Pavaniakshaya))

## License

The code is released under the [MIT License](LICENSE). The dataset in `data/` is not covered by it and keeps whatever terms it was published under.

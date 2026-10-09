# HPC_Project: GPU-Accelerated Financial Sentiment Analysis

Machine-learning and deep-learning models for financial sentiment classification (positive / neutral / negative), where the compute-heavy step of each model is written as a **custom CUDA C++ kernel** and driven from Python through [CuPy](https://cupy.dev/).

The goal is to look inside the models rather than call library routines: for each algorithm, find the operation that dominates the runtime (a sparse gradient update, a split search, a matrix multiply) and parallelise it by hand on the GPU.

## Contents

- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Approach](#approach)
- [Classical models](#classical-models)
- [Transformer experiments](#transformer-experiments)
- [Getting started](#getting-started)
- [Notes and limitations](#notes-and-limitations)
- [Future work](#future-work)
- [Tech stack](#tech-stack)

## Dataset

[`data/fin_data_1.csv`](data/fin_data_1.csv) holds English-language sentences from financial news, each labelled positive, neutral or negative. The project was originally described as using the Financial PhraseBank dataset.

| | |
|---|---|
| Rows | 5,842 |
| Columns | `Sentence`, `Sentiment` |
| neutral | 3,130 |
| positive | 1,852 |
| negative | 860 |

The classes are imbalanced (neutral is about 54% of the data). The Transformer notebooks counter this with random oversampling of the minority classes.

## Repository structure

```
HPC_Project/
├── data/
│   └── fin_data_1.csv               # raw dataset
│                                    # (data/processed/ is generated, git-ignored)
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

### Running the notebooks

Start Jupyter from inside `notebooks/` so that the relative path `../data/fin_data_1.csv` resolves:

```bash
cd notebooks
jupyter notebook
```

On Google Colab, upload `fin_data_1.csv` and change the `read_csv` path in the notebook you want to run. Notebooks 07-09 download the Sentence-BERT model on first run.

## Notes and limitations

- **Accuracy numbers are not directly comparable across scripts.** Some scripts report accuracy on the full dataset they trained on (for example `tfidf/ann.py`, `tfidf/svm.py`, the softmax scripts), others on a held-out 20% split (random forest, gradient boosting, ridge). The notebooks print a mix of training and validation accuracy. Treat each script's output as a smoke test of that implementation, not as a leaderboard.
- **No benchmark of the speed-up is included.** The scripts demonstrate the kernels; they do not time them against CPU baselines.
- **The CUDA scripts need a GPU.** They have been checked for syntax, but `cupy` and an NVIDIA device are required to execute them. Only `src/preprocess.py` runs on a CPU-only machine.
- The Sentence-BERT and tensor-core notebooks need a recent NVIDIA GPU for FP16 Tensor-Core throughput.

## Future work

- Optimise the kernels further for scalability, especially the Transformer's attention.
- Try other embeddings (GloVe, FastText) and long-document architectures such as Longformer or BigBird.
- Add CPU baselines and a common train/test protocol so models and kernels can be compared fairly.
- Explore deploying the final model on a live financial-news feed.

## Tech stack

- **Language:** Python, CUDA C++
- **GPU:** CuPy (`RawKernel`), cuBLAS Tensor Cores
- **Data and ML utilities:** NumPy, SciPy, pandas, scikit-learn
- **Deep learning / embeddings:** PyTorch, Sentence-Transformers (Sentence-BERT)

# Meridian Diagnostics ML

**Machine Learning Assignment 3**
Perceptron, Logistic Regression, Support Vector Machines and Clustering From Scratch

| Field | Value |
|---|---|
| Student | Abdul Azeem |
| Roll Number | 24i2013 |
| Section | BAI5A |
| Repository | [AbdulAzeemHashmi/meridian-diagnostics-ml](https://github.com/AbdulAzeemHashmi/meridian-diagnostics-ml) |

---

## Assignment Overview

Meridian Diagnostics is a pathology lab that screens breast mass samples as benign or malignant and reads handwritten digits from sample ID forms. This assignment builds standard linear classifiers and clustering methods entirely from scratch using only NumPy, pandas, and Matplotlib. No scikit-learn, scipy, or any ready-made classifier is used anywhere.

**Total marks:** 125

**Parts covered:**

| Part | Topic | Marks |
|---|---|---|
| Setup | Stratified split, standardisation, EDA, PCA by SVD, 8 asserts | 5 |
| A | Perceptron, Novikoff bound, averaged and kernel perceptron, multiclass | 20 |
| B | Stable loss, gradient check, four optimisers, L1/L2, cost decisions, softmax | 25 |
| C | Pegasos, SMO with KKT and duality gap, kernels, grid search, multiclass SVM | 25 |
| D | K-means, GMM with EM, hierarchical, DBSCAN, spectral, metrics | 20 |
| E | Comparison tables and clinical recommendation | 10 |
| F | Interactive playground with dragging and live stats | 10 |

---

## Roll Number Parameters

Roll: **2013** (digits d1=2, d2=0, d3=1, d4=3)

| Parameter | Formula | Value |
|---|---|---|
| seed | 2013 | 2013 |
| eta | 0.005 x (1 + d4) | 0.02 |
| lam | 0.002 x max(1, d1+d2+d3+d4) | 0.012 |
| C | 2 ^ ((d3+d4) mod 5 - 2) | 4.0 |
| gamma | 0.02 x (1 + (roll mod 7)) | 0.10 |
| r | 2 + (roll mod 4) | 3 |
| rho | 0.02 x max(1, roll mod 6) | 0.02 |

---

## Datasets

### D1: Breast Cancer Wisconsin Diagnostic

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Direct file:** `https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data`

**Local file:** `data.csv` (not committed to repository, download separately)

| Property | Value |
|---|---|
| Total samples | 569 |
| Features | 30 numeric features (columns 3 to 32 of wdbc.data) |
| Classes | Malignant (M) 212 samples, Benign (B) 357 samples |
| Label encoding perceptron/SVM | M=+1, B=1 |
| Label encoding logistic regression | M=1, B=0 |

**Split (stratified):**

| Split | Samples | B | M |
|---|---|---|---|
| Train (60%) | 342 | 214 | 128 |
| Validation (20%) | 114 | 72 | 42 |
| Test (20%) | 113 | 71 | 42 |

Test holdout uses `np.random.default_rng(2013 + 10)`. Train/validation split uses `np.random.default_rng(2013 + 11)`.

### D2: Optical Recognition of Handwritten Digits

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)

**Direct files:**
- `https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra`
- `https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes`

**Local files:** `optdigits.tra` and `optdigits.tes` (not committed, download separately)

| Property | Value |
|---|---|
| Total combined samples | 5620 |
| Features | 64 pixel values (8x8 image, values 0 to 16) |
| Classes | 10 digit labels (0 to 9) |
| Subset used | digits1500: 150 per class, sampled with rng(2013+6) |
| Split for tasks | 70% train (1050), 30% test (450) |

---

## File and Folder Structure

```
meridian-diagnostics-ml/
|
|-- 24i-2013_AbdulAzeem_BAI-5A.ipynb    # Main executed notebook (all parts)
|-- mlcode.py                            # Shared ML library imported by notebook and playground
|-- playground.py                        # Interactive visual playground (Part F)
|-- generate_screenshots.py             # Headless script to produce screenshots folder
|
|-- screenshots/                         # Required Part F screenshots (7 PNG files)
|   |-- perceptron.png                   # Perceptron decision boundary demo
|   |-- logistic.png                     # Logistic regression boundary demo
|   |-- svm.png                          # SVM with margin lines and support vectors
|   |-- kmeans.png                       # K-means centroids and cluster colours
|   |-- exp1.png                         # Experiment 1: outlier point vs all models
|   |-- exp2.png                         # Experiment 2: XOR pattern, linear failure
|   |-- exp3.png                         # Experiment 3: horizontal stripes vs K-means
|
|-- data.csv                             # D1 dataset (GITIGNORED, download separately)
|-- optdigits.tra                        # D2 training file (GITIGNORED, download separately)
|-- optdigits.tes                        # D2 test file (GITIGNORED, download separately)
|-- .gitignore                           # Git ignore rules
|-- README.md                            # This file
```

---

## File and Folder Details

### `24i-2013_AbdulAzeem_BAI-5A.ipynb`

The main Jupyter notebook. It is fully self-contained and must run top-to-bottom without errors. It imports from `mlcode.py` for all algorithm implementations. Structure:

| Section | Content |
|---|---|
| Cell 1 | Roll number parameters printed and asserted |
| Setup | Data loading, stratified split, standardisation, EDA, PCA, 8 asserts |
| Part A | Perceptron, Novikoff bound, averaged perceptron, kernel perceptron, multiclass |
| Part B | Stable sigmoid, gradient check, four optimisers, L1/L2 paths, cost threshold, softmax |
| Part C | Pegasos, SMO, KKT/duality assertions, kernels, grid search, multiclass SVM |
| Part D | K-means, GMM-EM, hierarchical clustering, DBSCAN, spectral, all metrics |
| Part E | Comparison tables, clinical recommendation for Meridian |
| Part F | Playground experiment answers with screenshots |
| Report | 6-section markdown report as final notebook section |

### `mlcode.py`

Pure NumPy/Matplotlib implementation of all required functions. No scikit-learn or scipy. Every function has a docstring. Groups of functions:

**Part A functions:**
`perceptron_train`, `perceptron_predict`, `novikoff_bound`, `averaged_perceptron`, `kernel_perceptron`, `multiclass_perceptron`, `accuracy`, `confusion_matrix`, `precision_recall_f1`, `macro_f1`

**Part B functions:**
`sigmoid`, `softplus`, `lr_objective`, `lr_gradient`, `lr_hessian`, `gradient_check`, `gd`, `sgd`, `momentum_gd`, `newton`, `soft_threshold`, `ista`, `kfold_stratified`, `cross_val_lambda`, `roc_curve`, `auc_ranksum`, `auc_trapezoid`, `expected_cost`, `softmax`, `softmax_regression`

**Part C functions:**
`hinge_objective`, `pegasos`, `svm_dual_objective`, `smo_train`, `kkt_violations`, `duality_gap`, `linear_kernel`, `poly_kernel`, `rbf_kernel`, `grid_search_cv`, `one_vs_rest`, `one_vs_one`

**Part D functions:**
`kmeans_pp_init`, `kmeans`, `logsumexp`, `gmm_em`, `bic`, `agglomerative`, `cut_tree`, `draw_dendrogram`, `dbscan`, `k_distance`, `spectral_clustering`, `ari`, `nmi`, `silhouette`

### `playground.py`

Interactive point-and-click ML visualiser. Uses only NumPy and Matplotlib (event system). Imports from `mlcode.py`. Must be run from a terminal as `python playground.py` (not inside Jupyter with inline plots).

**Controls:**

| Key / Action | Effect |
|---|---|
| Left click | Add blue point (class +1) |
| Right click | Add red point (class 1) |
| p | Switch to Perceptron mode |
| l | Switch to Logistic Regression mode |
| s | Switch to SVM mode |
| k | Switch to K-means mode |
| Space | Step the active model once |
| 2 to 6 | Set K for K-means |
| r | Reset model weights |
| u | Remove last point |
| c | Clear all points |
| Click and drag | Move an existing point; SVM and LR auto-update on release |

**Live title stats:**
- Perceptron: epoch number and mistakes in last epoch
- Logistic regression: iteration number and current loss
- SVM: number of support vectors and margin width
- K-means: K, iteration number, and inertia

### `generate_screenshots.py`

Headless (Agg backend) script that programmatically creates all 7 required screenshots. Run once to populate the `screenshots/` folder. This file is listed in `.gitignore` and is a utility helper only.

### `screenshots/`

Seven PNG files required by the assignment submission. All are committed to the repository.

| File | Content |
|---|---|
| `perceptron.png` | Perceptron boundary on a linearly separable 2-D set |
| `logistic.png` | Logistic regression boundary after gradient descent |
| `svm.png` | SVM with solid decision boundary, dashed margin lines, circled support vectors |
| `kmeans.png` | K-means clusters coloured by assignment with centroid X markers |
| `exp1.png` | One outlier point placed inside the wrong cluster, all three model responses |
| `exp2.png` | XOR four-group pattern showing linear separator oscillation and failure |
| `exp3.png` | Three horizontal stripes showing K-means inability to recover stripe structure |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AbdulAzeemHashmi/meridian-diagnostics-ml.git
cd meridian-diagnostics-ml
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib jupyter nbclient nbformat
```

Or with a requirements file if provided:

```bash
pip install -r requirements.txt
```

### 3. Download datasets (not in repository)

**D1 Breast Cancer Wisconsin:**

```bash
curl -L -o data.csv https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
```

The downloaded `wdbc.data` has no header. The notebook renames and loads it correctly as `data.csv`. Alternatively the included `data.csv` already has column names added.

**D2 Optical Digits:**

```bash
curl -L -o optdigits.tra https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra
curl -L -o optdigits.tes https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes
```

---

## Running the Notebook

### Open interactively in Jupyter

```bash
jupyter notebook 24i-2013_AbdulAzeem_BAI-5A.ipynb
```

Or with JupyterLab:

```bash
jupyter lab 24i-2013_AbdulAzeem_BAI-5A.ipynb
```

### Run the entire notebook non-interactively (top to bottom)

```bash
jupyter nbconvert --to notebook --execute 24i-2013_AbdulAzeem_BAI-5A.ipynb --output 24i-2013_AbdulAzeem_BAI-5A_executed.ipynb
```

### Export notebook to PDF

```bash
jupyter nbconvert --to pdf 24i-2013_AbdulAzeem_BAI-5A.ipynb
```

### Export notebook to HTML

```bash
jupyter nbconvert --to html 24i-2013_AbdulAzeem_BAI-5A.ipynb
```

---

## Running the Playground

The playground requires a live display. Run it from a terminal (not from Jupyter inline):

```bash
python playground.py
```

A matplotlib window opens at 8x8 inches fixed to the range [5, 5] x [5, 5]. Use the mouse and keyboard controls listed above.

---

## Generating Screenshots Without a Display

```bash
python generate_screenshots.py
```

This uses the Agg (non-interactive) backend and writes all 7 PNG files directly to `screenshots/`.

---

## Debugging and Development

### Run a quick sanity check on mlcode.py

```bash
python -c "import mlcode as ml, numpy as np; print('mlcode import ok'); rng = np.random.default_rng(42); X = rng.standard_normal((60, 4)); y = np.where(X[:, 0] > 0, 1, -1); w, b, h, ep = ml.perceptron_train(X, y); print('perceptron ok, epochs:', ep)"
```

### Test gradient check (should print error below 1e6)

```bash
python -c "import mlcode as ml, numpy as np; rng = np.random.default_rng(42); X = rng.standard_normal((40, 5)); y = rng.integers(0, 2, 40).astype(float); err = ml.gradient_check(np.zeros(5), 0.0, X, y, lam=0.01); print('Gradient relative error:', err)"
```

### Test sigmoid stability at extreme values

```bash
python -c "import mlcode as ml, numpy as np; z = np.array([-1000.0, 0.0, 1000.0]); print('sigmoid:', ml.sigmoid(z)); print('softplus:', ml.softplus(z))"
```

### Test SMO on a small dataset

```bash
python -c "import mlcode as ml, numpy as np; rng = np.random.default_rng(42); X = rng.standard_normal((30, 2)); y = np.where(X[:, 0] > 0, 1, -1); alpha, b, dh, _ = ml.smo_train(X, y, C=1.0, kernel_func=ml.linear_kernel, max_passes=5); print('SMO ok, SVs:', int(np.sum(alpha > 1e-4)))"
```

### Test K-means inertia non-increase assertion

```bash
python -c "import mlcode as ml, numpy as np; rng = np.random.default_rng(2013); X = rng.standard_normal((100, 2)); cents, lbls, inrt, hist = ml.kmeans(X, K=4, n_init=3, rng=rng); diffs = np.diff(hist); assert all(d <= 1e-8 for d in diffs), 'Inertia went up'; print('Inertia assertion passed. Final inertia:', inrt)"
```

### Test ARI equals 1 for permuted labels

```bash
python -c "import mlcode as ml, numpy as np; rng = np.random.default_rng(42); lbls = rng.integers(0, 3, 100); perm = np.where(lbls==0, 2, np.where(lbls==2, 0, lbls)); score = ml.ari(lbls, perm); print('ARI for permuted labels:', score); assert abs(score - 1.0) < 1e-6"
```

### Run notebook cells one by one interactively

```bash
jupyter console 24i-2013_AbdulAzeem_BAI-5A.ipynb
```

### Profile a slow cell or function

```bash
python -c "import cProfile, mlcode as ml, numpy as np; rng = np.random.default_rng(2013); X = rng.standard_normal((200, 10)); y = np.where(X[:, 0] > 0, 1, -1); cProfile.run('ml.smo_train(X, y, C=1.0, kernel_func=ml.linear_kernel, max_passes=3)')"
```

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: mlcode` | Notebook and mlcode.py must be in the same folder | Run notebook from the project root directory |
| `FileNotFoundError: data.csv` | Dataset not downloaded | Follow dataset download steps above |
| `playground.py` shows blank window | Running inside Jupyter with inline backend | Run `python playground.py` from a terminal, not from a notebook cell |
| Matplotlib cache warning | No write access to `.matplotlib` folder | Set env variable: `set MPLCONFIGDIR=%TEMP%` on Windows |
| SMO very slow on large datasets | Quadratic kernel matrix memory | Reduce `max_iter` and `max_passes` during development; full values in final run |
| Notebook does not run top to bottom | Cell output from previous partial run | Kernel: Restart and Run All |

---

## Library Restrictions (Assignment Rules)

**Allowed:**
- `numpy` (including `numpy.linalg`) for all model computation
- `pandas` for loading and displaying data
- `matplotlib` and `seaborn` for plots
- Python standard library: `math`, `time`, `collections`, `itertools`

**Not allowed:**
- `scikit-learn`
- `scipy`
- `statsmodels`
- Any ready-made classifier, optimiser, QP solver, clustering function, kernel, metric, cross-validation, or PCA function

PCA must use `numpy.linalg.svd`. Eigen problems must use `numpy.linalg.eigh`. Distance and kernel matrices must be vectorised with NumPy.

---

## Repository Layout After Full Setup

```
meridian-diagnostics-ml/
|-- 24i-2013_AbdulAzeem_BAI-5A.ipynb
|-- mlcode.py
|-- playground.py
|-- screenshots/
|   |-- perceptron.png
|   |-- logistic.png
|   |-- svm.png
|   |-- kmeans.png
|   |-- exp1.png
|   |-- exp2.png
|   |-- exp3.png
|-- .gitignore
|-- README.md
```

Datasets (`data.csv`, `optdigits.tra`, `optdigits.tes`) and Python cache (`__pycache__/`, `.ipynb_checkpoints/`) are excluded from the repository via `.gitignore`.

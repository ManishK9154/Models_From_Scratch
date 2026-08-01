# ML From Scratch — Regression

Implementations of core ML algorithms built from scratch using only NumPy/Pandas — no scikit-learn in the core logic. Each algorithm is validated against scikit-learn's implementation on real datasets to confirm correctness.

The goal isn't to reinvent scikit-learn — it's to genuinely understand what's happening under the hood (gradient descent, loss functions, scaling, evaluation metrics) before relying on library implementations.

## What's implemented

**`Regression.py`**
- `SimpleRegression` — Linear regression via batch gradient descent, with optional L1/L2 regularization and configurable convergence threshold
- `LogisticRegression` — Binary classification via gradient descent on binary cross-entropy loss, with sigmoid activation, L1/L2 regularization, and built-in evaluation (`model_accuracy` — a from-scratch confusion matrix implementation: accuracy, precision, recall, F1)
- `MinMaxScaler`, `StandardScaler`, `RobustScaler` — feature scalers following scikit-learn's `fit`/`transform` convention (fit on train, transform on test — no data leakage)
- `TrainTestSplit` — random train/test splitting using position-safe boolean masking (avoids label vs. positional-index bugs that come with `.drop()` on arbitrary indices)

## Results

### Linear Regression — California Housing dataset

| | MSE (test) |
|---|---|
| From-scratch (gradient descent) | 0.578 |
| scikit-learn `LinearRegression` | 0.539 |

The from-scratch model lands within ~7% of scikit-learn's closed-form (exact) solution — the gap is expected, since scikit-learn solves the least-squares problem analytically while this implementation approximates it iteratively via gradient descent with an early-stopping convergence threshold.

### Logistic Regression — Telco Customer Churn dataset

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| From-scratch (gradient descent) | 0.802 | 0.623 | 0.664 | 0.643 |
| scikit-learn `LogisticRegression` | 0.814 | — | — | — |

Class distribution in training data: ~74% no-churn, ~26% churn. The model beats the majority-class baseline (73.6% accuracy) by ~6.6 points, confirming it's genuinely learning the minority class rather than exploiting class imbalance.

## Notable bugs found and fixed during development

Documenting these because finding and fixing them was most of the actual learning:

- **Train/test scaling mismatch**: predicting on unscaled features using weights trained on scaled features caused wildly incorrect predictions (linear regression) and numerical overflow in `exp()` leading to a degenerate all-one-class model (logistic regression). Root cause in both cases: forgetting to call `.transform()` on test data before prediction.
- **Positional vs. label-based indexing**: an early `TrainTestSplit` implementation used `.drop()` with positional indices, which silently breaks when a DataFrame's index isn't a clean `0..n-1` range. Fixed using boolean masking with `.iloc`, which is unambiguous regardless of index state.
- **Exploding gradients from unscaled features**: training logistic regression on raw (unscaled) features caused loss to diverge to `inf`/`nan` within a few iterations, even at a reduced learning rate — resolved by properly scaling inputs before training, not just tuning the learning rate.

## Project structure

```
Models_From_Scratch/
├── models/
│   └── Regression.py     # Linear & Logistic Regression Core from-scratch implementations
├── requirements.txt
├── data/
│   └── Telco_Customer_Churn.csv
├── notebooks/
│   ├── linear_regression.ipynb    # California Housing — SimpleRegression
│   └── logistic_regression.ipynb  # Telco Churn — LogisticRegression
```

## Setup

```bash
pip install -r requirements.txt
```

Notebooks import from `Regression.py` at the repo root via:
```python
import sys
sys.path.append('..')
from Regression import LogisticRegression, SimpleRegression, StandardScaler, TrainTestSplit
```
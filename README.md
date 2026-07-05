# 🎯 PIT-CP

**pitcp** is a Python package for conformal prediction using **probability integral transform (PIT) pivotal scores**. Given any black-box nonconformity score, it fits a conditional density estimator on the score distribution and maps raw scores to PIT values, yielding valid marginal coverage at any user-specified level.

---

## ✨ Features

- **PIT Conformal Prediction**: Maps base nonconformity scores through a learned conditional CDF, producing asymptotically exact conditional coverage.
- **Model-agnostic**: Works with any callable nonconformity score `s(x, y)`, including distance, residual, or likelihood-based scores.
- **Flexible Density Estimation**: Supports normalizing flows and mixture density networks from the [zuko](https://github.com/probabilists/zuko) library.
- **Marginal Coverage Guarantee**: Provably valid conformal coverage at any target level via finite-sample calibration.
- **scikit-learn**: Native `BaseEstimator` integration with a familiar `fit` / `conformalize` / `predict` API.

---

## 🚀 Installation

```bash
pip install pitcp
```

## 🔧 Usage

### Example

```python
import torch
import zuko
from pitcp import PITCP


def std(x):
    return torch.where((x > -0.9) & (x < 0.9), torch.cos(torch.pi * x / 2), 1.0)


def gen_data(n):
    x = torch.rand(n, 1) * 2 - 1
    return x, torch.randn(n, 1) * std(x)


torch.manual_seed(42)

(X_train, y_train), (X_cal, y_cal), (X_test, y_test) = [
    gen_data(5000) for _ in range(3)
]


# Define a nonconformity score
def score(x, y):
    return y.abs()


# Build a normalizing flow density estimator
model = zuko.flows.NSF(features=1, context=1, bins=4, hidden_features=(32, 32, 32))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Compute nonconformity scores
s_train = score(X_train, y_train)
s_cal = score(X_cal, y_cal)
s_test = score(X_test, y_test)

# Fit and conformalize
pitcp = PITCP(model, optimizer, n_epochs=10, batch_size=128)
pitcp.fit(X_train, s_train)
pitcp.conformalize(X_cal, s_cal)

# Predict conformal regions (max score thresholds) at multiple quantiles
limits = pitcp.predict(X_test, quantile=[0.7, 0.8, 0.9])

# Predict conformal coverage
covered = pitcp.predict_coverage(X_test, s_test, quantile=[0.7, 0.8, 0.9])
print(f"Empirical coverages: {covered.mean(axis=0)}")
```

---

## 📖 Learn More

For tutorials, API reference, visit the official site:  
👉 [pitcp's documentation](https://felixlaplante0.github.io/pitcp)

---

## 📊 Reproducing Results

Clone the repository, create and activate a virtual environment, and install the local package in editable mode from the repository root:

```bash
python -m pip install -e .
```

The editable installation makes `pitcp` importable from any working directory and immediately exposes local source changes. The exact package versions used for the paper are recorded in `scripts/requirements.txt`. They can be installed before the editable package when exact environment reproduction is required:

```bash
python -m pip install -r scripts/requirements.txt
python -m pip install -e .
```

### Preparing the Real Data

The repository contains the raw SARCOS (`data/sarcos_inv.mat`) and Naval Propulsion Plants (`data/naval.txt`) datasets. To regenerate the training, validation-test, and prediction CSV files, install the TabPFN client, replace `PUT_YOUR_API_TOKEN_HERE` in `data/predict.py` with a valid access token, and run both dataset modes:

```bash
python -m pip install tabpfn-client
python data/predict.py --sarcos
python data/predict.py --naval
```

These commands write `{dataset}-train.csv`, `{dataset}-valtest.csv`, and `{dataset}-pred.csv` to `data/`. The generated CSV files are already included, so this step can be skipped unless the predictions must be regenerated.

### Running the Experiments

Run the experiment scripts from the repository root:

```bash
python scripts/convergence-plots.py
python scripts/synthetic-plots.py
python scripts/real-data-diagnostics.py --sarcos
python scripts/real-data-diagnostics.py --naval
```

The scripts resolve data and output paths from their file locations, so they do not depend on the current working directory. Figures and diagnostic tables are written to `figures/`.

### Script Descriptions

| Script | Description |
| :--- | :--- |
| `convergence-plots.py` | Evaluates the convergence of the PIT-CP procedure across different training sample sizes using various density estimators (SOSPF, GMM). |
| `real-data-diagnostics.py` | Benchmarks PIT-CP against other conformal prediction methods (SCP, CQR, HPD, CONTRA) on real-world datasets, calculating coverage gaps and prediction interval volumes. |
| `synthetic-plots.py` | Compares the conformal regions and conditional coverage of PIT-CP, CQR, and SCP on synthetic heteroscedastic data. |

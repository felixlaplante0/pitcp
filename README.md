# 🎯 PIT-CP

[![codecov](https://codecov.io/gh/felixlaplante0/pitcp/graph/badge.svg)](https://codecov.io/gh/felixlaplante0/pitcp)
[![documentation](https://readthedocs.org/projects/pitcp/badge/?version=latest)](https://pitcp.readthedocs.io/en/latest/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pitcp-app.streamlit.app/)

**pitcp** is a Python package for conformal prediction using **probability integral transform (PIT) pivotal scores**. Given any black-box nonconformity score, it fits a conditional density estimator on the score distribution and maps raw scores to PIT values, yielding valid marginal coverage at any user-specified level.

Our contribution is `PITCP`. The package also reimplements the state-of-the-art `SCP`, `CQR`, `HPD`, and `CONTRA` baselines behind a consistent scikit-learn API.

---

## ✨ Features

- **PIT conformal prediction**: `PITCP` maps base nonconformity scores through a learned conditional CDF.
- **Split conformal prediction**: `SCP` calibrates arbitrary scalar nonconformity scores without a learned correction.
- **Conformalized quantile regression**: `CQR` accepts multiple outputs and provides a scikit-learn gradient-boosting implementation of state-of-the-art conformalized quantile regression.
- **Highest-density regions**: `HPD` calibrates conditional highest-predictive-density sets.
- **Latent-space regions**: `CONTRA` maps targets through a conditional normalizing flow and calibrates a Euclidean norm-based score in latent space.
- **Model-agnostic**: Works with any callable nonconformity score `s(x, y)`, including distance, residual, or likelihood-based scores.
- **Flexible Density Estimation**: Supports normalizing flows and mixture density networks from the [zuko](https://github.com/probabilists/zuko) library.
- **Marginal Coverage Guarantee**: Provably valid conformal coverage at any target level via finite-sample calibration.
- **scikit-learn integration**: Native `BaseEstimator` integration with familiar `fit`, `conformalize`, `predict`, and `contains` methods.

---

## 🚀 Installation

```bash
python -m pip install pitcp
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
limits = pitcp.predict(X_test, confidence_level=[0.7, 0.8, 0.9])

# Predict conformal coverage
covered = pitcp.contains(X_test, s_test, confidence_level=[0.7, 0.8, 0.9])
print(f"Empirical coverages: {covered.mean(axis=0)}")
```

---

## 📖 Learn More

For tutorials and the API reference, visit the
[documentation](https://pitcp.readthedocs.io/en/latest/). Try the methods in the
[interactive playground](https://pitcp-app.streamlit.app/).

The method is described in [*A Post-Processing Conformal Prediction Approach
for Conditional Coverage via Pivotal Scores*](https://doi.org/10.48550/arXiv.2605.25852).

---

## 📊 Reproducing Results

Clone the repository, create and activate a virtual environment, then install the exact package versions used for the paper:

```bash
python -m pip install -r scripts/requirements.txt
```

Keep the repository folder layout unchanged: do not move, rename, or flatten its folders, because the experiment scripts resolve paths relative to their file locations.

This installs the frozen `pitcp` release from PyPI together with the
experimental dependencies recorded in `scripts/requirements.txt`.
Do not subsequently install the repository in editable mode when reproducing
the paper, because that would replace the frozen PyPI release with the local
source checkout.

For local development rather than exact paper reproduction, install the current checkout in editable mode from the repository root:

```bash
python -m pip install -e .
```

The editable installation makes local source changes immediately available without reinstalling the package.

### Preparing the Real Data

The repository contains the raw SARCOS (`data/sarcos_inv.mat`) and Naval Propulsion Plants (`data/naval.txt`) datasets. To regenerate the training, validation-test, and prediction CSV files, install the TabPFN client, set the `TABPFN_ACCESS_TOKEN` environment variable to a valid access token, and run both dataset modes:

```bash
python -m pip install tabpfn-client
```

On PowerShell:

```powershell
$env:TABPFN_ACCESS_TOKEN="your-token"
python data/predict.py --sarcos
python data/predict.py --naval
```

On Linux or macOS:

```bash
export TABPFN_ACCESS_TOKEN="your-token"
python data/predict.py --sarcos
python data/predict.py --naval
```

The token remains in the current terminal session and is not written to the
repository.

These commands write `{dataset}-train.csv`, `{dataset}-valtest.csv`, and `{dataset}-pred.csv` to `data/`. The generated CSV files are already included, so this step can be skipped unless the predictions must be regenerated.

The paper experiments use Python 3.13 and the dependencies in
`scripts/requirements.txt`. Verify downloaded and generated data against
`data/SHA256SUMS` before running the experiments. The committed prediction
files are the canonical reproduction artifacts because results returned by the
remote TabPFN service may change independently of this repository.

On systems with `sha256sum`, verify the artifacts with:

```bash
cd data
sha256sum --check SHA256SUMS
```

### Dataset Attribution and Licensing

The SARCOS inverse-dynamics dataset was provided by Sethu Vijayakumar and
is distributed through the [Gaussian Processes for Machine Learning dataset
repository](https://gaussianprocess.org/gpml/data/). Users should cite
Vijayakumar and Schaal (2000), *LWPR: An O(n) Algorithm for Incremental Real
Time Learning in High Dimensional Space*. No explicit redistribution license
has been identified for SARCOS. Its inclusion in this repository should not be
interpreted as granting reuse or redistribution rights.

The Naval Propulsion Plants dataset is licensed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
Users should cite Coraddu et al. (2014), *Condition Based Maintenance of Naval
Propulsion Plants*, [UCI Machine Learning Repository](https://doi.org/10.24432/C5K31K).

### Running the Experiments

Run the experiment scripts from the repository root:

```bash
python scripts/convergence-plots.py
python scripts/synthetic-plots.py
python scripts/real-data-diagnostics.py --sarcos
python scripts/real-data-diagnostics.py --naval
```

Alternatively, run all four experiments sequentially on Windows, Linux, or macOS:

```bash
python run-all.py
```

The runner uses the active Python interpreter and stops if an experiment fails. It does not run `data/predict.py`; prepare or verify the real-data files separately as described above.

These commands write figures and diagnostic tables to `figures/`. The paper settings are defined as constants near the top of each script.

### Script Descriptions

| Script | Description |
| :--- | :--- |
| `convergence-plots.py` | Evaluates the convergence of the PIT-CP procedure across different training sample sizes using various density estimators (SOSPF, GMM). |
| `real-data-diagnostics.py` | Benchmarks `PITCP` against other conformal prediction methods (`SCP`, `CQR`, `HPD`, `CONTRA`) on real-world datasets, calculating coverage gaps and prediction interval volumes. |
| `synthetic-plots.py` | Compares the conformal regions and conditional coverage of `PITCP`, `CQR`, and `SCP` on synthetic heteroscedastic data. |

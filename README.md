# 🎯 PIT-CP

**pitcp** is a Python package for conformal prediction using **probability integral transform (PIT) pivotal scores**. Given any black-box nonconformity score, it fits a conditional density estimator on the score distribution and maps raw scores to PIT values, yielding valid marginal coverage at any user-specified level.

---

## ✨ Features

- **PIT Conformal Prediction**: Maps base nonconformity scores through a learned conditional CDF, producing asymptotically exact conditional coverage.
- **Model-agnostic**: Works with any callable nonconformity score `s(X, y)`, including distance-, residual-, or likelihood-based scores.
- **Flexible Density Estimation**: Supports normalizing flows and mixture density networks from the [zuko](https://github.com/probabilists/zuko) library.
- **Marginal Coverage Guarantee**: Provably valid conformal coverage at any target level via finite-sample calibration.
- **scikit-learn**: Native `BaseEstimator` integration with a familiar `fit` / `predict` API.

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
    gen_data(n) for n in (5000, 1000, 5000)
]


# Define a nonconformity score
def score(x, y):
    return y.abs()


# Build a normalizing flow density estimator
model = zuko.flows.NSF(features=1, context=1, bins=4, hidden_features=(32, 32, 32))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Fit and conformalize
pitcp = PITCP(score, model, optimizer, n_epochs=100, batch_size=64)
pitcp.fit(X_train, y_train)
pitcp.conformalize(X_cal, y_cal)

# Predict coverage at 90%
covered = pitcp.predict(X_test, y_test, quantile=0.9)
print(f"Empirical coverage: {covered.float().mean():.3f}")
```

---

## 📖 Learn More

For tutorials, API reference, visit the official site:  
👉 [pitcp's documentation](https://felixlaplante0.github.io/pitcp)

import numpy as np
import torch
import zuko
from pitcp import PITCP


def base_score(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (y.squeeze(-1) - x.squeeze(-1)).abs()


torch.manual_seed(0)
np.random.seed(0)

n_train = 1000
n_cal = 500
n_test = 100

X = np.random.randn(n_train + n_cal + n_test, 1).astype("float32")
y = 2.0 * X + 0.3 * np.random.randn(*X.shape).astype("float32")

X_train, y_train = X[:n_train], y[:n_train]
X_cal, y_cal = X[n_train : n_train + n_cal], y[n_train : n_train + n_cal]
X_test, y_test = X[-n_test:], y[-n_test:]

estimator = zuko.flows.NSF(
    features=1,
    context=1,
    transforms=2,
    hidden_features=(16, 16),
)

optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)

model = PITCP(
    base_score=base_score,
    estimator=estimator,
    optimizer=optimizer,
    n_epochs=50,
    batch_size=32,
    verbose=True,
)

model.fit(X_train, y_train)
model.conformalize(X_cal, y_cal)

result = model.predict(X_test, y_test, quantile=0.9)

print("quantile:", result.quantile)
print("covered:", result.is_covered.detach().cpu().numpy())
print("empirical coverage:", result.is_covered.float().mean().item())

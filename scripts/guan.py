import matplotlib.pyplot as plt
import numpy as np
import torch
import zuko
from pitcp import PITCP
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import TensorDataset

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


class CQR:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.model_lo = HistGradientBoostingRegressor(
            max_iter=500, loss="quantile", quantile=alpha / 2, max_depth=1
        )
        self.model_hi = HistGradientBoostingRegressor(
            max_iter=500, loss="quantile", quantile=1 - alpha / 2, max_depth=1
        )
        self.q = 0.0

    def fit(self, X, y):
        self.model_lo.fit(X, y)
        self.model_hi.fit(X, y)
        return self

    def conformalize(self, X, y):
        q_lo = self.model_lo.predict(X)
        q_hi = self.model_hi.predict(X)
        scores = np.maximum(q_lo - y, y - q_hi)
        n = len(scores)
        self.q = np.quantile(
            scores,
            np.ceil((n + 1) * (1 - self.alpha)) / n,
            method="higher",
        )

        return self

    def predict(self, X):
        q_lo = self.model_lo.predict(X) - self.q
        q_hi = self.model_hi.predict(X) + self.q
        return np.minimum(q_lo, q_hi), np.maximum(q_lo, q_hi)


def std(x):
    return (1 - 2 * x**2).abs() + 0.1


def gen_data(n):
    x = torch.rand(n, 1) * 2 - 1
    return x, torch.randn(n, 1) * std(x)


torch.manual_seed(42)

(X_train, y_train), (X_cal, y_cal), (X_test, y_test) = [
    gen_data(n) for n in (5000, 1000, 5000)
]


def run(score, q):
    model = zuko.flows.SOSPF(features=1, context=1, hidden_features=(32, 32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pitcp = PITCP(score, model, optimizer, n_epochs=200, batch_size=512)
    pitcp.fit(TensorDataset(X_train, y_train))
    pitcp.conformalize(TensorDataset(X_cal, y_cal))

    cqr = CQR(alpha=1 - q).fit(X_train.numpy(), y_train.flatten().numpy())
    cqr.conformalize(X_cal.numpy(), y_cal.flatten().numpy())

    t = torch.quantile(score(X_cal, y_cal).flatten(), q).item()

    xb, yb = X_test.flatten(), y_test.flatten()

    xv = torch.linspace(-1, 1, 500)
    yv = torch.linspace(-5, 5, 1000)
    xg, yg = torch.meshgrid(xv, yv, indexing="ij")

    Xg, Yg = xg.unsqueeze(-1), yg.unsqueeze(-1)

    Zb = score(Xg, Yg).squeeze(-1).le(t)
    Zp = pitcp.predict(TensorDataset(Xg, Yg), quantile=q).squeeze(-1).bool()

    _, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].scatter(xb, yb, c="#7f8c8d", s=3, alpha=0.5)

    s = std(xv)

    configs = [
        (None, "CQR", "#e74c3c", "#c0392b", "dotted"),
        (Zb, "Base", "#3498db", "#2980b9", "dashed"),
        (Zp, "PIT", "#2ecc71", "#27ae60", "solid"),
    ]

    for Z, name, fill, dot, linestyle in configs:
        if Z is None:
            ymin, ymax = cqr.predict(xv.numpy().reshape(-1, 1))
        else:
            ymin = torch.where(Z, yv, torch.inf).min(1).values
            ymax = torch.where(Z, yv, -torch.inf).max(1).values

        ax[0].fill_between(
            xv,
            ymin,
            ymax,
            color=fill,
            alpha=0.3,
            label=f"{name}",
        )
        ax[0].plot(xv, ymin, c=dot, linewidth=2, linestyle=linestyle)
        ax[0].plot(xv, ymax, c=dot, linewidth=2, linestyle=linestyle)

        coverage = norm.cdf((ymax / s).numpy()) - norm.cdf((ymin / s).numpy())
        marginal_coverage = np.nanmean(coverage)
        l1 = torch.nanmean(torch.abs(torch.tensor(coverage) - marginal_coverage)).item()

        ax[1].plot(
            xv,
            coverage,
            linewidth=2,
            c=dot,
            label=f"{name} conditional",
            linestyle=linestyle,
        )
        ax[1].fill_between(
            xv,
            coverage,
            marginal_coverage,
            color=fill,
            alpha=0.3,
            label=f"{name} MAE: {l1:.3f}",
        )

    ax[0].set(
        title="Conformal region", xlabel="X", ylabel="Y", xlim=(-1, 1), ylim=(-3.5, 3.5)
    )
    ax[0].legend(loc="lower center", ncol=3)
    ax[1].axhline(q, c="black", lw=2, ls="dashed")
    ax[1].set(
        title="Coverage", xlabel="X", ylabel="Coverage", xlim=(-1, 1), ylim=(0, 1.05)
    )
    ax[1].legend(loc="lower center", ncol=3)

    plt.tight_layout()
    plt.savefig(f"../figures/quantile-{q}.pdf")
    plt.show()


def score_abs(x, y):
    return y.abs()


def score_hpd(x, y):
    v = std(x) ** 2
    return 0.5 * (torch.log(2 * torch.pi * v) + y**2 / v)


def score_y(x, y):
    return y


for score, q in [
    (score_abs, 0.7),
    (score_hpd, 0.8),
    (score_y, 0.9),
]:
    run(score, q)

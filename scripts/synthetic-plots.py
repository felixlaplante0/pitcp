import matplotlib.pyplot as plt
import numpy as np
import torch
import zuko
from pitcp import PITCP
from scipy.stats import norm
from utils.cqr import CQR
from utils.scp import SCP

# Set plot parameters
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


# Data generation helpers
def std(x):
    return np.abs(1 - 2 * x**2) + 0.1


def gen_data(n):
    x = np.random.rand(n, 1) * 2 - 1
    return x, np.random.randn(n, 1) * std(x)


# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

(X_train, y_train), (X_cal, y_cal), (X_test, y_test) = [
    gen_data(n) for n in (5000, 1000, 5000)
]


def run(score_fn, inv_score_fn, q):
    # CQR
    cqr = CQR(alpha=1 - q).fit(X_train, y_train.flatten())
    cqr.conformalize(X_cal, y_cal.flatten())

    # SCP
    scores_cal = score_fn(X_cal, y_cal)
    scp = SCP(alpha=1 - q).conformalize(X_cal, scores_cal)

    # PIT-CP
    model = zuko.flows.SOSPF(features=1, context=1, hidden_features=(32, 32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    pit = PITCP(model, opt, n_epochs=200, batch_size=512)
    pit.fit(X_train, score_fn(X_train, y_train))
    pit.conformalize(X_cal, scores_cal)

    xv = np.linspace(-1, 1, 500)

    # Plot results
    _, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].scatter(X_test, y_test, c="#7f8c8d", s=3, alpha=0.5)

    # Plot intervals and coverage
    for name, fill, dot, ls in [
        ("CQR", "#e74c3c", "#c0392b", "dotted"),
        ("SCP", "#3498db", "#2980b9", "dashed"),
        ("PIT", "#2ecc71", "#27ae60", "solid"),
    ]:
        if name == "CQR":
            y_min, y_max = cqr.predict(xv[:, None])
        elif name == "SCP":
            y_min, y_max = inv_score_fn(xv, scp.predict(xv))
        else:
            lim = pit.predict(xv[:, None], quantile=q).flatten()
            y_min, y_max = inv_score_fn(xv, lim)

        ax[0].fill_between(xv, y_min, y_max, color=fill, alpha=0.3, label=name)
        ax[0].plot(xv, y_min, c=dot, lw=2, ls=ls)
        ax[0].plot(xv, y_max, c=dot, lw=2, ls=ls)

        cov = norm.cdf(y_max / std(xv)) - norm.cdf(y_min / std(xv))
        ax[1].plot(xv, cov, lw=2, c=dot, ls=ls, label=f"{name} cond.")
        ax[1].fill_between(
            xv,
            cov,
            cov.mean(),
            color=fill,
            alpha=0.3,
            label=f"MAE: {np.abs(cov - cov.mean()).mean():.3f}",
        )

    ax[0].set(
        title="Conformal region", xlabel="X", ylabel="Y", xlim=(-1, 1), ylim=(-3.5, 3.5)
    )
    ax[1].set(
        title="Coverage", xlabel="X", ylabel="Coverage", xlim=(-1, 1), ylim=(0, 1.05)
    )
    ax[1].axhline(q, c="k", lw=2, ls="--")
    for a in ax:
        a.legend(loc="lower center", ncol=3)

    # Save figure
    plt.tight_layout()
    plt.savefig(f"../figures/synthetic-quantile-{q}.pdf")
    plt.show()


# Define scoring functions
def score_abs(x, y):
    return np.abs(y)


def inv_score_abs(x, s):
    return -s, s


def score_hpd(x, y):
    v = std(x) ** 2
    l = np.log(v)
    return 0.5 * (np.log(2 * np.pi) + l + y**2 / v)


def inv_score_hpd(x, s):
    v = std(x) ** 2
    l = np.log(v)
    y = np.sqrt(np.maximum((2 * s - np.log(2 * np.pi) - l) * v, 0))
    return -y, y


def score_y(x, y):
    return y


def inv_score_y(x, s):
    return np.full_like(s, -10), s


# Execute runs
for score_fn, inv_score_fn, q in [
    (score_abs, inv_score_abs, 0.7),
    (score_hpd, inv_score_hpd, 0.8),
    (score_y, inv_score_y, 0.9),
]:
    run(score_fn, inv_score_fn, q)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import zuko
from matplotlib.lines import Line2D
from pitcp import PITCP
from scipy.stats import norm

from utils._cqr import CQR
from utils._data import (
    gen_data,
    inv_score_abs,
    inv_score_hpd,
    inv_score_y,
    oracle_score_abs,
    oracle_score_hpd,
    oracle_score_y,
    score_abs,
    score_hpd,
    score_y,
    std,
)
from utils._scp import SCP

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


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SIZES = (5000, 1000, 5000)
QUANTILES = (0.7, 0.8, 0.9)
LEVELS = (0.6, 0.7, 0.8, 0.9)
Y_LIM = (-3.5, 3.5)
TITLES = ("Symmetric residual score", "Density-level score", "One-sided residual score")
METHOD_STYLES = (
    ("CQR", "#e74c3c", "#c0392b", "dotted"),
    ("SCP", "#3498db", "#2980b9", "dashed"),
    ("PIT", "#2ecc71", "#27ae60", "solid"),
)


def run(
    score_fn,
    inv_score_fn,
    oracle_score_fn,
    title,
    q,
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_test,
    y_test,
):
    # CQR
    cqr_gamma = 0 if score_fn is score_y else 1 / 2
    cqr = CQR(alpha=1 - q, gamma=cqr_gamma).fit(X_train[:, None], y_train)
    cqr.conformalize(X_cal[:, None], y_cal)

    # SCP
    scores_cal = score_fn(X_cal, y_cal)
    scp = SCP(alpha=1 - q).conformalize(X_cal[:, None], scores_cal)

    # PIT-CP
    model = zuko.flows.SOSPF(features=1, context=1, hidden_features=(32, 32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    pit = PITCP(model, opt, n_epochs=200, batch_size=512)
    pit.fit(X_train[:, None], score_fn(X_train, y_train))
    pit.conformalize(X_cal[:, None], scores_cal)

    xv = np.linspace(-1, 1, 500)

    # Plot results
    fig, ax = plt.subplots(3, 1, figsize=(8, 8.6))
    fig.suptitle(title)
    ax[0].scatter(X_test, y_test, c="#7f8c8d", s=3, alpha=0.5)

    # Plot intervals and coverage
    for name, fill, dot, ls in METHOD_STYLES:
        if name == "CQR":
            y_min, y_max = cqr.predict(xv[:, None])
        elif name == "SCP":
            y_min, y_max = inv_score_fn(xv, scp.predict(xv))
        else:
            lim = pit.predict(xv[:, None], quantile=q)
            y_min, y_max = inv_score_fn(xv, lim)

        y_min_plot = np.clip(y_min, *Y_LIM)
        y_max_plot = np.clip(y_max, *Y_LIM)
        ax[0].fill_between(
            xv, y_min_plot, y_max_plot, color=fill, alpha=0.3, label=name
        )
        ax[0].plot(xv, y_min_plot, c=dot, lw=2, ls=ls)
        ax[0].plot(xv, y_max_plot, c=dot, lw=2, ls=ls)

        coverage = norm.cdf(y_max / std(xv)) - norm.cdf(y_min / std(xv))
        mae = np.abs(coverage - coverage.mean()).mean()
        ax[1].plot(xv, coverage, lw=2, c=dot, ls=ls, label=f"{name} cond.")
        ax[1].fill_between(
            xv,
            coverage,
            coverage.mean(),
            color=fill,
            alpha=0.3,
            label=f"MAE: {mae:.3f}",
        )

    scores_test = score_fn(X_test, y_test)
    ax[2].scatter(X_test, scores_test, c="#7f8c8d", marker="+", s=12, alpha=0.5)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(LEVELS)))
    score_values = [scores_test]
    for level, color in zip(LEVELS, colors, strict=True):
        estimated = np.asarray(pit.predict(xv[:, None], quantile=level))
        oracle = oracle_score_fn(xv, level)
        score_values.extend((estimated, oracle))
        ax[2].plot(xv, estimated, c=color, lw=2, label=f"{level:.1f}")
        ax[2].plot(xv, oracle, c="k", lw=2, ls=":")

    ax[0].set(xlabel="X", ylabel="Y", xlim=(-1, 1), ylim=Y_LIM)
    ax[1].set(
        xlabel="X",
        ylabel="Coverage",
        xlim=(-1, 1),
        ylim=(0, 1.05),
    )
    score_values = np.concatenate([np.ravel(values) for values in score_values])
    score_lo, score_hi = np.nanquantile(score_values, [0.005, 0.995])
    score_pad = 0.05 * (score_hi - score_lo)
    ax[2].set(
        xlabel="X",
        ylabel="Score",
        xlim=(-1, 1),
        ylim=(score_lo - score_pad, score_hi + score_pad),
    )
    ax[1].axhline(q, c="k", lw=2, ls="--")
    ax[0].legend(loc="lower center", ncol=3)
    ax[1].legend(loc="lower center", ncol=3)
    level_legend = ax[2].legend(loc="upper center", ncol=len(LEVELS))
    ax[2].add_artist(level_legend)
    ax[2].legend(
        handles=[
            Line2D([], [], c="k", lw=2, label="PIT"),
            Line2D([], [], c="k", lw=2, ls=":", label="Oracle"),
        ],
        loc="lower center",
        ncol=2,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / f"synthetic-quantile-{q}.pdf")
    plt.show()


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = [
        gen_data(n) for n in SAMPLE_SIZES
    ]

    score_fns = (score_abs, score_hpd, score_y)
    inv_score_fns = (inv_score_abs, inv_score_hpd, inv_score_y)
    oracle_score_fns = (oracle_score_abs, oracle_score_hpd, oracle_score_y)

    for score_fn, inv_score_fn, oracle_score_fn, title, quantile in zip(
        score_fns, inv_score_fns, oracle_score_fns, TITLES, QUANTILES, strict=True
    ):
        run(
            score_fn,
            inv_score_fn,
            oracle_score_fn,
            title,
            quantile,
            X_train,
            y_train,
            X_cal,
            y_cal,
            X_test,
            y_test,
        )


if __name__ == "__main__":
    main()

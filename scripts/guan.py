import matplotlib.pyplot as plt
import torch
import zuko
from scipy.stats import norm

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


def run(score, name, q):
    model = zuko.flows.NSF(
        features=1,
        context=1,
        bins=4,
        hidden_features=(32, 32, 32),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    pitcp = PITCP(
        score,
        model,
        optimizer,
        n_epochs=100,
        batch_size=64,
    )

    pitcp.fit(X_train, y_train)
    pitcp.conformalize(X_cal, y_cal)

    t = torch.quantile(
        score(X_cal, y_cal).flatten(),
        q,
    ).item()

    xb, yb = X_test.flatten(), y_test.flatten()

    xv = torch.linspace(-1, 1, 500)
    yv = torch.linspace(yb.min(), yb.max(), 1000)
    xg, yg = torch.meshgrid(
        xv,
        yv,
        indexing="ij",
    )

    Xg, Yg = xg.unsqueeze(-1), yg.unsqueeze(-1)
    Zb = score(Xg, Yg).squeeze(-1).le(t).float()
    Zp = pitcp.predict(Xg, Yg, quantile=q).squeeze(-1).float()

    Cb = score(X_test, y_test).flatten().le(t)
    Cp = pitcp.predict(
        X_test,
        y_test,
        quantile=q,
    ).flatten()

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(name)

    for i, (Z, C, T) in enumerate(
        [
            (Zb, Cb, "Base"),
            (Zp, Cp, "PIT-corrected"),
        ]
    ):
        ax[0, i].contourf(
            xg, yg, Z, levels=[0, 0.5, 1], colors=["white", "#aae6be"], alpha=0.5
        )
        ax[0, i].contour(xg, yg, Z, levels=[0.5], colors="#5fb482")
        ax[0, i].scatter(xb[C], yb[C], c="#5fb482", s=3, label="Covered")
        ax[0, i].scatter(xb[~C], yb[~C], c="#dc6969", s=3, label="Uncovered")
        ax[0, i].set_title(f"{T} conformal region")
        ax[0, i].set_xlabel("X")
        ax[0, i].set_ylabel("Y")
        ax[0, i].legend(loc="upper center", markerscale=4)

        M = Z.bool()

        ymin = torch.where(
            M.any(1),
            torch.where(M, yv, torch.inf).min(1).values,
            torch.nan,
        )
        ymax = torch.where(
            M.any(1),
            torch.where(M, yv, -torch.inf).max(1).values,
            torch.nan,
        )

        s = std(xv)

        marginal_coverage = torch.mean(C.float()).item()
        coverage = norm.cdf((ymax / s).numpy()) - norm.cdf((ymin / s).numpy())
        l1 = torch.nanmean(torch.abs(torch.tensor(coverage) - q)).item()

        ax[1, i].plot(xv, coverage, linewidth=2, c="#7db9f5")
        ax[1, i].axhline(q, c="red", linestyle="dashed", label=f"Target coverage: {q}")
        ax[1, i].axhline(
            marginal_coverage,
            c="blue",
            linestyle="dashed",
            label=f"Marginal coverage: {marginal_coverage:.3f}",
        )
        ax[1, i].fill_between(
            xv,
            coverage,
            q,
            color="#7db9f5",
            alpha=0.3,
            label=f"Mean absolute deviation: {l1:.3f}",
        )

        ax[1, i].set_ylim(0, 1.05)
        ax[1, i].set_title(f"{T} conditional coverage")
        ax[1, i].set_xlabel("X")
        ax[1, i].set_ylabel("Coverage")
        ax[1, i].legend(loc="lower center")

    plt.tight_layout()
    plt.savefig(f"../figures/quantile-{q}.pdf")
    plt.show()


def score_abs(x, y):
    return y.abs()


def score_oracle(x, y):
    v = std(x) ** 2
    return 0.5 * (torch.log(2 * torch.pi * v) + y**2 / v)


def score_y(x, y):
    return y


for score, n, q in [
    (score_abs, r"Score $s(x, y) = \vert y \vert$", 0.5),
    (score_oracle, r"Score $s(x, y) = -\log p(y \mid x)$", 0.75),
    (score_y, r"Score $s(x, y) = y$", 0.9),
]:
    with plt.rc_context(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    ):
        run(score, n, q)
import matplotlib.pyplot as plt
import torch
import zuko
from scipy.stats import norm
from pitcp import PITCP


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


def std(x):
    return torch.where((x > -0.9) & (x < 0.9), torch.cos(torch.pi * x / 2), 1.0)


def gen_data(n):
    x = torch.rand(n, 1) * 2 - 1
    return x, torch.randn(n, 1) * std(x)


torch.manual_seed(42)

(X_train, y_train), (X_cal, y_cal), (X_test, y_test) = [
    gen_data(n) for n in (5000, 1000, 5000)
]


def run(score, q):
    model = zuko.flows.SOSPF(features=1, context=1, hidden_features=(32, 32, 32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pitcp = PITCP(model, optimizer, n_epochs=100)
    pitcp.fit(X_train, score(X_train, y_train))
    pitcp.conformalize(X_cal, score(X_cal, y_cal))

    t = torch.quantile(score(X_cal, y_cal).flatten(), q).item()

    xb, yb = X_test.flatten(), y_test.flatten()

    xv = torch.linspace(-1, 1, 500)
    yv = torch.linspace(yb.min(), yb.max(), 1000)
    xg, yg = torch.meshgrid(xv, yv, indexing="ij")

    Xg, Yg = xg.unsqueeze(-1), yg.unsqueeze(-1)

    Zb = score(Xg, Yg).squeeze(-1).le(t)
    Zp = pitcp.predict(Xg, score(Xg, Yg), quantile=q).squeeze(-1).bool()

    Cb = score(X_test, y_test).flatten().le(t)
    Cp = pitcp.predict(X_test, score(X_test, y_test), quantile=q).flatten()

    _, ax = plt.subplots(2, 1, figsize=(7, 8))
    ax[0].scatter(xb, yb, c="#7f8c8d", s=3, alpha=0.5)

    s = std(xv)

    configs = [
        (Zb, Cb, "Base", "#3498db", "#2980b9"),
        (Zp, Cp, "PIT", "#2ecc71", "#27ae60"),
    ]

    for Z, C, name, fill, dot in configs:
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
        ax[0].plot(xv, ymin, c=dot, linewidth=2)
        ax[0].plot(xv, ymax, c=dot, linewidth=2)

        marginal_coverage = C.float().mean().item()

        coverage = norm.cdf((ymax / s).numpy()) - norm.cdf((ymin / s).numpy())
        l1 = torch.nanmean(torch.abs(torch.tensor(coverage) - q)).item()

        ax[1].plot(xv, coverage, linewidth=2, c=dot, label=f"{name} conditional")
        ax[1].axhline(
            marginal_coverage,
            linestyle="dashed",
            linewidth=2,
            c=dot,
            label=f"{name} marginal: {marginal_coverage:.2f}",
        )
        ax[1].fill_between(
            xv,
            coverage,
            q,
            color=fill,
            alpha=0.3,
            label=f"{name} MAE: {l1:.3f}",
        )

    ax[0].set(title="Conformal region", xlabel="X", ylabel="Y", xlim=(-1, 1))
    ax[0].legend(loc="lower center", ncol=2)
    ax[1].set(
        title="Coverage", xlabel="X", ylabel="Coverage", xlim=(-1, 1), ylim=(0, 1.05)
    )
    ax[1].legend(loc="lower center", ncol=2)

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


for score, q in [
    (score_abs, 0.7),
    (score_oracle, 0.8),
    (score_y, 0.9),
]:
    run(score, q)

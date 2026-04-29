import matplotlib.pyplot as plt
import torch
import zuko
from scipy.stats import norm

from pitcp import PITCP


def gen_data(n):
    x = torch.rand(n, 1) * 2 - 1
    mask = (x > -0.95) & (x < 0.95)
    s = torch.where(mask, torch.cos(torch.pi * x / 2), 1.0)
    y = torch.randn(n, 1) * s
    return x, y


torch.manual_seed(42)
X_train, y_train = gen_data(10000)
X_cal, y_cal = gen_data(1000)
X_test, y_test = gen_data(5000)


def run(base_score, name, q=0.75):
    model = zuko.flows.RealNVP(features=1, context=1, hidden_features=(128, 128))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pitcp = PITCP(base_score, model, optimizer, n_epochs=100)

    pitcp.fit(X_train, y_train)
    pitcp.conformalize(X_cal, y_cal)

    score_cal = base_score(X_cal, y_cal).flatten()
    threshold = torch.quantile(score_cal, q).item()

    score_test = base_score(X_test, y_test).flatten()
    covered_base = score_test <= threshold
    covered_pitcp = pitcp.predict(X_test, y_test, quantile=q).flatten()

    x, y = X_test.flatten(), y_test.flatten()
    xv = torch.linspace(-1, 1, 400)
    yv = torch.linspace(y.min(), y.max(), 400)
    xg, yg = torch.meshgrid(xv, yv, indexing="ij")

    X_grid, Y_grid = xg.reshape(-1, 1), yg.reshape(-1, 1)
    Z_base = (
        base_score(X_grid, Y_grid).flatten().le(threshold).reshape(xg.shape).float()
    )
    Z_pitcp = pitcp.predict(X_grid, Y_grid, quantile=q).reshape(xg.shape).float()

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(name, fontsize=16)

    for i, (Z, covered, title) in enumerate(
        [(Z_base, covered_base, "Base"), (Z_pitcp, covered_pitcp, "PIT-corrected")]
    ):
        ax[0, i].contourf(
            xg, yg, Z, levels=[0, 0.5, 1], colors=["white", "#aae6be"], alpha=0.5
        )
        ax[0, i].contour(xg, yg, Z, levels=[0.5], colors="#5fb482", linewidths=1.5)

        ax[0, i].scatter(x[covered], y[covered], c="#5fb482", s=3, alpha=0.4)
        ax[0, i].scatter(x[~covered], y[~covered], c="#ac2222", s=3, alpha=0.4)

        ax[0, i].set_title(f"{title} Conformal region")
        ax[0, i].set_xlabel("X")
        ax[0, i].set_ylabel("Y")

        Zc = Z.numpy()
        ymin = torch.full((len(xv),), float("nan"))
        ymax = torch.full((len(xv),), float("nan"))

        for k in range(len(xv)):
            idx = Zc[k] == 1
            if idx.any():
                ymin[k] = yv[idx].min()
                ymax[k] = yv[idx].max()

        mask = (xv > -0.95) & (xv < 0.95)
        sigma = torch.where(mask, torch.cos(torch.pi * xv / 2), 1.0)

        cov = norm.cdf((ymax / sigma).numpy()) - norm.cdf((ymin / sigma).numpy())

        ax[1, i].plot(xv, cov, linewidth=2, c="#7db9f5")
        ax[1, i].axhline(
            q, c="#ac2222", lw=1.2, linestyle="dashed", label=f"Target coverage {q}"
        )

        mean_l1 = torch.nanmean(torch.abs(torch.tensor(cov) - q)).item()
        ax[1, i].fill_between(
            xv, cov, q, color="#7db9f5", alpha=0.3, label=f"Mean L1: {mean_l1:.4f}"
        )

        ax[1, i].set_ylim(0, 1)
        ax[1, i].set_title(f"{title} Conditional Coverage")
        ax[1, i].set_xlabel("X")
        ax[1, i].set_ylabel("Coverage")
        ax[1, i].legend()

    plt.tight_layout()
    plt.show()


def score_abs(x, y):
    return y.abs()


def score_oracle(x, y):
    mask = (x > -0.95) & (x < 0.95)
    v = torch.where(mask, torch.cos(torch.pi * x / 2) ** 2, 1.0)
    return 0.5 * (torch.log(2 * torch.pi * v) + y**2 / v)


def score_y(x, y):
    return y


run(score_abs, r"Score $s(x, y) = |y|$")
run(score_oracle, r"Score $s(x, y) = -\log p(y \mid x)$")
run(score_y, r"Score $s(x, y) = y$")

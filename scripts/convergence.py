import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import zuko
from pitcp import PITCP
from scipy.stats import norm

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
    return (1 - 2 * x**2).abs() + 0.1


def gen_data(n):
    x = torch.rand(n, 1) * 2 - 1
    return x, torch.randn(n, 1) * std(x)


def score_abs(x, y):
    return y.abs()


torch.manual_seed(42)

X_cal, y_cal = gen_data(1000)

Ns = torch.linspace(0, 5000, 6, dtype=int)
qs = np.linspace(0.0, 1.0, 100).tolist()[1:-1]
n_runs = 10

xv = torch.linspace(-1, 1, 500)
yv = torch.linspace(-5, 5, 1000)
xg, yg = torch.meshgrid(xv, yv, indexing="ij")

Xg, Yg = xg.unsqueeze(-1), yg.unsqueeze(-1)
s = std(xv)

data = []
for _ in range(n_runs):
    for name in ["SOSPF", "GMM"]:
        for n in Ns:
            X_train, y_train = gen_data(n.item())

            if name == "SOSPF":
                model = zuko.flows.SOSPF(
                    features=1, context=1, hidden_features=(32, 32)
                )
            else:
                model = zuko.mixtures.GMM(
                    features=1,
                    context=1,
                    components=4,
                    hidden_features=(32, 32),
                    covariance_type="diagonal",
                )

            if n.item() == 0:
                for param in model.parameters():
                    param.data.zero_()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            pitcp = PITCP(score_abs, model, optimizer, n_epochs=200, batch_size=256)
            if n.item() > 0:
                pitcp.fit(X_train, y_train)
            pitcp.conformalize(X_cal, y_cal)

            Zp = pitcp.predict(Xg, Yg, quantile=qs).bool()

            ymin = torch.where(Zp, yv.view(1, -1, 1), torch.inf).min(1).values
            ymax = torch.where(Zp, yv.view(1, -1, 1), -torch.inf).max(1).values
            ymax = torch.maximum(ymin, ymax)

            coverage = norm.cdf((ymax / s.view(-1, 1)).numpy()) - norm.cdf(
                (ymin / s.view(-1, 1)).numpy()
            )
            marginal_coverage = np.nanmean(coverage, axis=0)
            l1 = np.nanmean(np.abs(coverage - marginal_coverage).max(axis=1))
            data.append({"Model": name, "N": n.item(), "MAE": l1})

df = pd.DataFrame(data)

_, ax = plt.subplots(figsize=(8, 5))

sns.lineplot(
    data=df,
    x="N",
    y="MAE",
    hue="Model",
    ax=ax,
    marker="o",
    palette={"SOSPF": "#2980B9", "GMM": "#C0392B"},
    linewidth=2,
    err_style="bars",
)

ax.set(
    title="Convergence of the PIT-CP procedure",
    xlabel="N (training samples)",
    ylabel=r"$\mathbb{E}[\widehat{\Delta}(X)]$",
)
ax.legend()

plt.tight_layout()
plt.savefig("../figures/convergence.pdf")
plt.show()

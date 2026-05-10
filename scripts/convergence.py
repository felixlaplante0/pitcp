import matplotlib.pyplot as plt
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
    return (0.9 * x**2 + 0.1).sqrt()


def gen_data(n):
    x = torch.rand(n, 1) * 2 - 1
    return x, torch.randn(n, 1) * std(x)


def score_abs(x, y):
    return y.abs()


torch.manual_seed(42)

(X_cal, y_cal), (X_test, y_test) = [gen_data(n) for n in (1000, 5000)]

Ns = torch.linspace(0, 5000, 5, dtype=int)
q = 0.9
n_runs = 5

xv = torch.linspace(-1, 1, 500)
yv = torch.linspace(y_test.min(), y_test.max(), 1000)
xg, yg = torch.meshgrid(xv, yv, indexing="ij")

Xg, Yg = xg.unsqueeze(-1), yg.unsqueeze(-1)
s = std(xv)

data = []
for _ in range(n_runs):
    for name in ["SOSPF", "GMM"]:
        for n in Ns:
            X_train, y_train = gen_data(n.item())

            if name == "SOSPF" or n.item() == 0:
                model = zuko.flows.SOSPF(
                    features=1, context=1, hidden_features=(16, 16)
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            else:
                model = zuko.mixtures.GMM(
                    features=1,
                    context=1,
                    components=10,
                    hidden_features=(16, 16),
                    covariance_type="diagonal",
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

            batch_size = max(1, n.item() // 10)
            pitcp = PITCP(
                score_abs, model, optimizer, n_epochs=100, batch_size=batch_size
            )
            if n.item() > 0:
                pitcp.fit(X_train, y_train)
            pitcp.conformalize(X_cal, y_cal)

            Zp = pitcp.predict(Xg, Yg, quantile=q).squeeze(-1).bool()
            Cp = pitcp.predict(X_test, y_test, quantile=q).flatten()

            ymin = torch.where(Zp, yv, torch.inf).min(1).values
            ymax = torch.where(Zp, yv, -torch.inf).max(1).values

            marginal_coverage = Cp.float().mean().item()
            coverage = norm.cdf((ymax / s).numpy()) - norm.cdf((ymin / s).numpy())
            l1 = torch.nanmean(
                torch.abs(torch.tensor(coverage) - marginal_coverage)
            ).item()
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
    ylabel="MAE",
)
ax.legend()

plt.tight_layout()
plt.savefig("../figures/convergence.pdf")
plt.show()

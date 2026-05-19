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
    return np.abs(1 - 2 * x**2) + 0.1


def gen_data(n):
    x = np.random.rand(n) * 2 - 1
    return x, np.random.randn(n) * std(x)


# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate data
X_cal, y_cal = gen_data(1000)
Ns, qs = np.linspace(0, 5000, 6, dtype=int), np.linspace(0.01, 0.99, 98).tolist()
n_runs = 10

xv = np.linspace(-1, 1, 500)[:, None]

data = []
for _ in range(n_runs):
    for name in ["SOSPF", "GMM"]:
        for n in Ns:
            X_train, y_train = gen_data(n)
            if name == "SOSPF":
                model = zuko.flows.SOSPF(
                    features=1, context=1, hidden_features=(32, 32)
                )
            else:
                model = zuko.mixtures.GMM(
                    features=1, context=1, components=5, hidden_features=(32, 32)
                )

            if n == 0:
                for p in model.parameters():
                    p.data.zero_()

            # PIT-CP
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            pit = PITCP(model, optimizer, n_epochs=200, batch_size=512)
            if n > 0:
                pit.fit(X_train[:, None], np.abs(y_train))
            # Conformalize
            pit.conformalize(X_cal[:, None], np.abs(y_cal))

            lims = pit.predict(xv, quantile=qs)
            y_min, y_max = -lims, lims

            # Calculate coverage and L1 error
            cov = norm.cdf(y_max / std(xv)) - norm.cdf(y_min / std(xv))
            data.append(
                {"Model": name, "N": n, "MAE": np.abs(cov - cov.mean(0)).max(1).mean()}
            )

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

# Save figure
plt.tight_layout()
plt.savefig("../figures/convergence.pdf")
plt.show()

plt.show()

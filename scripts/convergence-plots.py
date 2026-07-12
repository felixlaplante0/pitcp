from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import zuko
from _utils import gen_data, score_abs, std
from pitcp import PITCP
from scipy.stats import norm

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
N_RUNS = 10
TRAINING_SIZES = np.linspace(0, 5000, 6, dtype=int)
CALIBRATION_SIZE = 1000
QUANTILES = np.linspace(0.01, 0.99, 98).tolist()


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    X_cal, y_cal = gen_data(CALIBRATION_SIZE)
    xv = np.linspace(-1, 1, 500)[:, None]

    data = []
    for _ in range(N_RUNS):
        for name in ["SOSPF", "GMM"]:
            for n in TRAINING_SIZES:
                X_train, y_train = gen_data(n)
                if name == "SOSPF":
                    model = zuko.flows.SOSPF(
                        features=1, context=1, hidden_features=(32, 32)
                    )
                else:
                    model = zuko.mixtures.GMM(
                        features=1,
                        context=1,
                        components=5,
                        hidden_features=(32, 32),
                    )

                if n == 0:
                    for parameter in model.parameters():
                        parameter.data.zero_()

                # Train PIT-CP model
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                pit = PITCP(model, optimizer, n_epochs=200, batch_size=512)
                if n > 0:
                    pit.fit(X_train[:, None], score_abs(X_train, y_train))
                pit.conformalize(X_cal[:, None], score_abs(X_cal, y_cal))

                # Compute coverage
                limits = pit.predict(xv, confidence_level=QUANTILES)
                y_min, y_max = -limits, limits
                coverage = norm.cdf(y_max / std(xv)) - norm.cdf(y_min / std(xv))
                data.append(
                    {
                        "Model": name,
                        "N": n,
                        "MAE": np.abs(coverage - coverage.mean(0)).max(1).mean(),
                    }
                )

    frame = pd.DataFrame(data)
    _, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=frame,
        x="N",
        y="MAE",
        hue="Model",
        style="Model",
        ax=ax,
        markers={"SOSPF": "o", "GMM": "s"},
        dashes=False,
        palette={"SOSPF": "#2980B9", "GMM": "#C0392B"},
        linewidth=2,
        err_style="bars",
    )
    ax.set(
        title="Convergence of the PIT-CP procedure",
        xlabel="N (training samples)",
        ylabel=r"$\mathbb{E}\left[ \widehat{\Delta}(X) \right] \downarrow$",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "convergence.pdf")
    plt.show()


if __name__ == "__main__":
    main()

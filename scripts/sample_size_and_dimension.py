import time

import lingam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

from otlingam import ExhaustiveLiNGAM, GreedyLiNGAM, ICALiNGAM, disorder
from utils import gen_laplace

MODELS = {
    "ExhaustiveLiNGAM": ExhaustiveLiNGAM,
    "GreedyLiNGAM": GreedyLiNGAM,
    "OT ICA-LiNGAM": ICALiNGAM,
    "ICA-LiNGAM": lingam.ICALiNGAM,
    "DirectLiNGAM": lingam.DirectLiNGAM,
}


def fit(data, weights, seed):
    for name, factory in MODELS.items():
        model = factory(random_state=seed) if "ICA" in name or "Direct" in name else factory()
        start = time.perf_counter()
        model.fit(data)
        estimated = np.abs(model.adjacency_matrix_) > 0.1
        truth = weights != 0
        yield {
            "Method": name,
            "Disorder": disorder(model.causal_order_, weights),
            "SHD": np.count_nonzero(truth != estimated),
            "F1": f1_score(truth.ravel(), estimated.ravel()),
            "Time": time.perf_counter() - start,
        }


results = []
for sweep, grid, fixed_n, fixed_d in (
    ("n", (250, 500, 1000, 2000, 4000), None, 7),
    ("d", (4, 6, 8, 10), 2000, None),
):
    for value in grid:
        for repetition in range(10):
            rng = np.random.default_rng(1000 + 100 * value + repetition)
            data, weights = gen_laplace(fixed_n or value, fixed_d or value, 0.4, rng)
            results.extend(
                {"Sweep": sweep, "Value": value, **result}
                for result in fit(data, weights, repetition)
            )

results = pd.DataFrame(results)
figure, axes = plt.subplots(2, 4, figsize=(12, 5.4), layout="constrained")
for row, (sweep, xlabel) in enumerate(
    (("n", "Sample size n (d = 7)"), ("d", "Dimension d (n = 2000)"))
):
    for axis, metric in zip(axes[row], ("Disorder", "SHD", "F1", "Time")):
        sns.lineplot(
            data=results[results["Sweep"] == sweep],
            x="Value",
            y=metric,
            hue="Method",
            marker="o",
            errorbar="sd",
            ax=axis,
            legend=metric == "Disorder",
        )
        axis.set(xlabel=xlabel, title=metric)
        axis.grid(alpha=0.3)
        if metric == "Time":
            axis.set_yscale("log")

figure.savefig("../figures/sample_size_and_dimension.pdf")
plt.close(figure)

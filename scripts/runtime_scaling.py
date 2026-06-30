import time

import lingam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from otlingam import ExhaustiveLiNGAM, GreedyLiNGAM, ICALiNGAM
from utils import gen_laplace

MODELS = {
    "ExhaustiveLiNGAM": ExhaustiveLiNGAM,
    "GreedyLiNGAM": GreedyLiNGAM,
    "OT ICA-LiNGAM": ICALiNGAM,
    "ICA-LiNGAM": lingam.ICALiNGAM,
    "DirectLiNGAM": lingam.DirectLiNGAM,
}

results = []
for d in (6, 8, 10, 12, 16, 20):
    for repetition in range(3):
        data, _ = gen_laplace(2000, d, 0.4, np.random.default_rng(9000 + repetition))
        for name, factory in MODELS.items():
            if d > 12 and name == "ExhaustiveLiNGAM":
                continue
            model = factory(random_state=repetition) if "ICA" in name or "Direct" in name else factory()
            start = time.perf_counter()
            model.fit(data)
            results.append(
                {"Dimension": d, "Method": name, "Time": time.perf_counter() - start}
            )

figure, axis = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
sns.lineplot(
    data=pd.DataFrame(results),
    x="Dimension",
    y="Time",
    hue="Method",
    marker="o",
    errorbar="sd",
    ax=axis,
)
axis.set_yscale("log")
axis.grid(alpha=0.3)
figure.savefig("../figures/runtime_scaling.pdf")
plt.close(figure)

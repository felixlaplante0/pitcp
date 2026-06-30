import lingam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from otlingam import ExhaustiveLiNGAM, GreedyLiNGAM, ICALiNGAM, disorder
from utils import gen_t

MODELS = {
    "ExhaustiveLiNGAM": ExhaustiveLiNGAM,
    "GreedyLiNGAM": GreedyLiNGAM,
    "OT ICA-LiNGAM": ICALiNGAM,
    "ICA-LiNGAM": lingam.ICALiNGAM,
    "DirectLiNGAM": lingam.DirectLiNGAM,
}

results = []
for maximum_df in (3, 5, 10, 20, 40):
    for repetition in range(12):
        rng = np.random.default_rng(7000 + repetition)
        data, weights = gen_t(3000, 8, 0.4, np.linspace(3, maximum_df, 8), rng)
        for name, factory in MODELS.items():
            model = factory(random_state=repetition) if "ICA" in name or "Direct" in name else factory()
            model.fit(data)
            results.append(
                {
                    "Maximum df": maximum_df,
                    "Method": name,
                    "Disorder": disorder(model.causal_order_, weights),
                }
            )

figure, axis = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
sns.lineplot(
    data=pd.DataFrame(results),
    x="Maximum df",
    y="Disorder",
    hue="Method",
    marker="o",
    errorbar="sd",
    ax=axis,
)
axis.grid(alpha=0.3)
figure.savefig("../figures/noise_heterogeneity.pdf")
plt.close(figure)

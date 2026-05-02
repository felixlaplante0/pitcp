import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update(
    {"axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12}
)

t = np.linspace(1e-4, 6, 700)


def cond_cdf(t, x):
    return 1 - np.exp(-(x + 1) * t)


def marg_cdf(t):
    return 1 - (np.exp(-t) - np.exp(-2 * t)) / t


marg = marg_cdf(t)

fig, ax = plt.subplots(figsize=(8, 4.2))

x_vals = np.linspace(0, 1, 30)
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#27ae60", "#c0392b"])

for x_val in x_vals:
    ax.plot(t, cond_cdf(t, x_val), color=cmap(x_val), lw=1, alpha=0.5)

ax.plot(t, marg, color="black", lw=2, zorder=4, label=r"$F_S$ (Marginal)")

for x_val, color, label, ks_text, offset in [
    (0, "#27ae60", r"$F_{S \mid X=0}$", r"$d_{KS}(F_{S \mid X=0}, F_S)$", -0.16),
    (1, "#c0392b", r"$F_{S \mid X=1}$", r"$d_{KS}(F_{S \mid X=1}, F_S)$", 0.13),
]:
    cond = cond_cdf(t, x_val)
    ax.plot(t, cond, color=color, lw=2, zorder=5, label=label)

    diff = np.abs(cond - marg)
    idx = np.argmax(diff)

    ax.plot(
        [t[idx], t[idx]],
        [marg[idx], cond[idx]],
        color=color,
        lw=2,
        linestyle="dotted",
        zorder=6,
    )
    ax.scatter([t[idx], t[idx]], [marg[idx], cond[idx]], color=color, s=40, zorder=7)

    ax.text(
        t[idx] - 0.5 * offset,
        (marg[idx] + cond[idx]) / 2 + offset,
        ks_text,
        color=color,
        fontsize=16,
        va="center",
        ha="center",
        rotation=45,
        zorder=8,
    )

ax.set(xlabel=r"$t$", ylabel=r"$F(t)$", xlim=(-0.05, 4), ylim=(0, 1.05))
ax.legend(loc="lower right", frameon=False, fontsize=16)

plt.tight_layout()
plt.savefig("../figures/ks.pdf")
plt.show()

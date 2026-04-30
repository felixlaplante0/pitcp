import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

rng = np.random.default_rng(42)


def under_curve(dist, n, lo, hi):
    x = np.array([])
    while x.size < n:
        z = dist.rvs(4 * n, random_state=rng)
        x = np.r_[x, z[(lo <= z) & (z <= hi)]][:n]
    return x, rng.random(n) * dist.pdf(x)


x = np.linspace(0.03, 3.2, 700)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))

curves = [
    (
        lognorm(0.38, scale=np.exp(-0.35)),
        "#aae6be",
        "#5fb482",
        r"$P_{S\mid X=a}$",
        (0.85, 1.35),
    ),
    (
        lognorm(0.52, scale=np.exp(0.28)),
        "#ffbebe",
        "#dc6969",
        r"$P_{S\mid X=b}$",
        (2.0, 0.35),
    ),
]

for d, fill, dot, label, pos in curves:
    xs, ys = under_curve(d, 35, 0.03, 3.2)
    ax[0].fill_between(x, d.pdf(x), color=fill, alpha=0.5)
    ax[0].plot(x, d.pdf(x), color=dot, lw=1)
    ax[0].scatter(xs, ys, s=20, color=dot)
    ax[0].text(*pos, label, color=dot, fontsize=18)

xu, yu = rng.random(70), 0.5 * rng.random(70)
ax[1].fill([0, 1, 1, 0], [0, 0, 0.5, 0.5], color="#b4dcff", alpha=0.5)
ax[1].plot([0, 1, 1, 0, 0], [0, 0, 0.5, 0.5, 0], color="#7db9f5", lw=1)
ax[1].scatter(xu, yu, s=20, color="#7db9f5")
ax[1].text(
    0.5, 0.54, r"$\mathrm{Uniform}(0,1)$", ha="center", color="#7db9f5", fontsize=18
)

ax[0].set(xlim=(0, 3.2), ylim=(0, 1.8))
ax[1].set(xlim=(0, 1), ylim=(0, 0.62))

for a in ax:
    a.set_xticks([])
    a.set_yticks([])

plt.tight_layout()
plt.savefig("../figures/ot.pdf")
plt.show()

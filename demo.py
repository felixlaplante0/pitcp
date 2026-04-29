import matplotlib.pyplot as plt
import torch
import zuko
from pitcp import PITCP


def gen(n):
    x = torch.rand(n, 1) * 2 - 1
    m = (x > -0.95) & (x < 0.95)
    s = torch.where(m, torch.cos(torch.pi * x / 2), 1.0)
    y = torch.randn(n, 1) * s
    return x, y


torch.manual_seed(0)
Xt, yt = gen(4000)
Xc, yc = gen(1000)
Xs, ys = gen(50000)


def smooth(x, y, xg, h=0.15, batch_size=5000):
    num = torch.zeros_like(xg)
    den = torch.zeros_like(xg)
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]
        w = torch.exp(-0.5 * ((xb - xg) / h) ** 2)
        num += (w * yb).sum(0)
        den += w.sum(0)
    return num / den


def run(score, name):
    est = zuko.flows.RealNVP(
        features=1,
        context=1,
        hidden_features=(64, 64),
    )

    opt = torch.optim.Adam(est.parameters(), lr=1e-3)

    m = PITCP(score, est, opt, n_epochs=50, batch_size=64)

    m.fit(Xt, yt)
    m.conformalize(Xc, yc)

    q = 0.9

    qb = torch.quantile(score(Xc, yc).flatten(), q).item()

    sb = score(Xs, ys).flatten()
    cb = sb <= qb

    cp = m.predict(Xs, ys, quantile=q).flatten()

    x = Xs.flatten()
    y = ys.flatten()

    xv = torch.linspace(-1, 1, 200)
    yv = torch.linspace(y.min(), y.max(), 200)

    xg, yg = torch.meshgrid(xv, yv, indexing="ij")

    Xgf = xg.reshape(-1, 1)
    Ygf = yg.reshape(-1, 1)

    Zb = score(Xgf, Ygf).flatten().le(qb).reshape(xg.shape).float()

    Zp = m.predict(Xgf, Ygf, quantile=q).reshape(xg.shape).float()

    xp = torch.linspace(-1, 1, 200)

    _, ax = plt.subplots(2, 2, figsize=(12, 9))

    for i, (Z, c, t) in enumerate([(Zb, cb, "Base"), (Zp, cp, "PITCP")]):
        ax[0, i].contourf(
            xg,
            yg,
            Z,
            levels=[0.5, 1.5],
            alpha=0.25,
        )

        ax[0, i].scatter(
            x[::10],
            y[::10],
            c=c[::10],
            cmap="RdYlGn",
            s=6,
            alpha=0.3,
        )

        s = smooth(
            x.unsqueeze(1),
            c.float().unsqueeze(1),
            xp.unsqueeze(0),
        )

        ax[1, i].plot(xp, s.flatten())

        ax[1, i].axhline(q, linestyle="dashed", c="red")
        ax[1, i].set_ylim(0, 1)

        ax[0, i].set_title(name + " " + t)
        ax[1, i].set_title(name + " " + t)

    plt.tight_layout()
    plt.show()


def score_abs(x, y):
    return y.abs()


def score_oracle(x, y):
    m = (x > -0.95) & (x < 0.95)
    v = torch.where(
        m,
        torch.cos(torch.pi * x / 2) ** 2,
        torch.ones_like(x),
    )
    return 0.5 * (torch.log(2 * torch.pi * v) + y**2 / v)


def score_y(x, y):
    return y


run(score_abs, "abs(y)")
run(score_oracle, "HPD")
run(score_y, "y")

"""Data, model, metric, and plotting helpers for the playground."""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import torch
import zuko
from sklearn.ensemble import HistGradientBoostingRegressor

from pitcp import CONTRA, CQR, HPD, PITCP, SCP

SEED = 42
SAMPLE_SIZES = (5000, 1000, 5000)
METHODS = ("PITCP", "SCP", "CQR", "HPD", "CONTRA")
COLORS = {
    "PITCP": "#6750a4",
    "SCP": "#2673b8",
    "CQR": "#d1495b",
    "HPD": "#23856d",
    "CONTRA": "#d17b0f",
}


def _scale(x: np.ndarray) -> np.ndarray:
    return np.abs(1 - 2 * x**2) + 0.1


def _data() -> dict[str, np.ndarray]:
    rng = np.random.RandomState(SEED)

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.rand(n) * 2 - 1
        return x[:, None], rng.randn(n) * _scale(x)

    (x_train, y_train), (x_cal, y_cal), (x_test, y_test) = [
        sample(n) for n in SAMPLE_SIZES
    ]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_test": x_test,
        "y_test": y_test,
    }


def _pitcp(path: Path) -> PITCP:
    torch.manual_seed(SEED)
    density = zuko.flows.SOSPF(features=1, context=1, hidden_features=(32, 32))
    model = PITCP(
        density,
        torch.optim.Adam(density.parameters(), lr=1e-3),
        n_epochs=200,
        batch_size=512,
        verbose=False,
        random_state=SEED,
    )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.scores_ = saved["scores"].numpy()
    model.n_features_in_ = 1
    model.eval()
    return model


def _density_models(path: Path) -> tuple[HPD, CONTRA]:
    torch.manual_seed(SEED)
    density = zuko.flows.MAF(features=1, context=1, hidden_features=(32, 32))
    optimizer = torch.optim.Adam(density.parameters(), lr=1e-3)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    density.load_state_dict(saved["state_dict"])

    hpd = HPD(
        density,
        optimizer,
        n_epochs=500,
        n_samples=512,
        batch_size=512,
        verbose=False,
        random_state=SEED,
    )
    hpd.scores_ = saved["hpd_scores"].numpy()
    hpd.is_fitted_ = True
    hpd.n_features_in_ = 1

    contra = CONTRA(density, optimizer, batch_size=512, verbose=False)
    contra.scores_ = saved["contra_scores"].numpy()
    contra.n_features_in_ = 1
    hpd.eval()
    contra.eval()
    return hpd, contra


def _models(root: Path) -> dict[str, object]:
    data = _data()
    hpd, contra = _density_models(root / "models" / "density_example.pt")
    return {
        "PITCP": _pitcp(root / "models" / "pitcp_example.pt"),
        "SCP": SCP().conformalize(np.abs(data["y_cal"])),
        "HPD": hpd,
        "CONTRA": contra,
    }


def _fit_cqr(confidence: float, data: dict[str, np.ndarray]) -> CQR:
    model = CQR(
        HistGradientBoostingRegressor(max_iter=100, random_state=SEED),
        confidence_level=confidence,
    )
    return model.fit(data["x_train"], data["y_train"]).conformalize(
        data["x_cal"], data["y_cal"]
    )


@torch.no_grad()
def _hpd_bounds(
    model: HPD,
    x: np.ndarray,
    confidence: float,
    *,
    n_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = next(model.estimator.parameters()).dtype
    context = torch.as_tensor(x, dtype=dtype)
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        dist = model.estimator(context)
        sample_nll = -dist.log_prob(dist.sample((n_samples,)))
    cutoff = torch.quantile(sample_nll, float(model.thresholds(confidence)[0]), dim=0)
    y_grid = torch.linspace(-5, 5, 1001, dtype=dtype)[:, None, None].expand(
        -1, len(x), -1
    )
    inside = -dist.log_prob(y_grid) <= cutoff
    values = y_grid[:, 0, 0]
    lower = torch.where(inside, values[:, None], torch.inf).min(dim=0).values
    upper = torch.where(inside, values[:, None], -torch.inf).max(dim=0).values
    return lower.numpy(), upper.numpy()


@torch.no_grad()
def _contra_bounds(
    model: CONTRA, x: np.ndarray, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    dtype = next(model.estimator.parameters()).dtype
    context = torch.as_tensor(x, dtype=dtype)
    radius = float(model.thresholds(confidence)[0])
    latent = torch.tensor([-radius, radius], dtype=dtype)[:, None, None]
    values = model.estimator(context).transform.inv(latent).squeeze(-1)
    return values.min(dim=0).values.numpy(), values.max(dim=0).values.numpy()


def _prediction(
    name: str,
    model: object,
    x_grid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name in {"PITCP", "SCP"}:
        limits = model.predict(  # type: ignore[attr-defined]
            x_grid, confidence_level=confidence
        )
        lower, upper = -limits, limits
        covered = (
            model.contains(  # type: ignore[attr-defined]
                x_test, np.abs(y_test), confidence_level=confidence
            )
            if name == "PITCP"
            else model.contains(  # type: ignore[attr-defined]
                np.abs(y_test), confidence_level=confidence
            )
        )
    elif name == "CQR":
        bounds = model.predict(x_grid)  # type: ignore[attr-defined]
        lower, upper = bounds.T
        covered = model.contains(x_test, y_test)  # type: ignore[attr-defined]
    elif name == "HPD":
        lower, upper = _hpd_bounds(model, x_grid, confidence)  # type: ignore[arg-type]
        covered = model.contains(  # type: ignore[attr-defined]
            x_test, y_test, confidence_level=confidence
        )
    else:
        lower, upper = _contra_bounds(  # type: ignore[arg-type]
            model, x_grid, confidence
        )
        covered = model.contains(  # type: ignore[attr-defined]
            x_test, y_test, confidence_level=confidence
        )
    return np.asarray(lower), np.asarray(upper), np.asarray(covered)


def _theoretical_coverage(
    x: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    sigma = _scale(x)
    root_two = np.sqrt(2)
    hi = torch.erf(torch.as_tensor(upper / (sigma * root_two)))
    lo = torch.erf(torch.as_tensor(lower / (sigma * root_two)))
    return ((hi - lo) / 2).numpy()


def _summary(
    name: str,
    lower: np.ndarray,
    upper: np.ndarray,
    theoretical: np.ndarray,
) -> dict[str, str]:
    return {
        "Method": name,
        "Marginal coverage": f"{theoretical.mean():.1%}",
        "Conditional MAD": f"{np.abs(theoretical - theoretical.mean()).mean():.3f}",
        "Mean width": f"{np.mean(upper - lower):.3f}",
    }


def _method_scale(methods: tuple[str, ...]) -> alt.Scale:
    return alt.Scale(domain=methods, range=[COLORS[name] for name in methods])


def _region_chart(
    results: dict[str, tuple[np.ndarray, ...]],
    methods: tuple[str, ...],
    x_grid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> alt.LayerChart:
    region = pd.concat(
        [
            pd.DataFrame(
                {
                    "x": x_grid[:, 0],
                    "lower": results[name][0],
                    "upper": results[name][1],
                    "method": name,
                }
            )
            for name in methods
        ],
        ignore_index=True,
    )
    boundary = pd.concat(
        [
            pd.DataFrame(
                {
                    "x": x_grid[:, 0],
                    "value": values,
                    "bound": bound,
                    "method": name,
                }
            )
            for name in methods
            for bound, values in (
                ("lower", results[name][0]),
                ("upper", results[name][1]),
            )
        ],
        ignore_index=True,
    )
    points = pd.DataFrame(
        {
            "x": x_test[:, 0],
            "y": y_test,
        }
    )
    band = (
        alt.Chart(region)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("x:Q", title="X"),
            y="lower:Q",
            y2="upper:Q",
            color=alt.Color("method:N", scale=_method_scale(methods), legend=None),
            tooltip=["method:N", "x:Q", "lower:Q", "upper:Q"],
        )
    )
    bounds = (
        alt.Chart(boundary)
        .mark_line(strokeWidth=2)
        .encode(
            x="x:Q",
            y=alt.Y("value:Q", title="Y"),
            color=alt.Color("method:N", scale=_method_scale(methods), legend=None),
            detail="bound:N",
        )
    )
    observations = (
        alt.Chart(points)
        .mark_circle(size=24, opacity=0.6)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.value("#87909f"),
            tooltip=["x:Q", "y:Q"],
        )
    )
    return (band + bounds + observations).properties(height=360)


def _coverage_chart(
    results: dict[str, tuple[np.ndarray, ...]],
    methods: tuple[str, ...],
    x_grid: np.ndarray,
    confidence: float,
) -> alt.LayerChart:
    theory = pd.concat(
        [
            pd.DataFrame(
                {
                    "x": x_grid[:, 0],
                    "coverage": results[name][3],
                    "method": name,
                }
            )
            for name in methods
        ],
        ignore_index=True,
    )
    curve = (
        alt.Chart(theory)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("x:Q", title="X"),
            y=alt.Y(
                "coverage:Q",
                title="Coverage",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("method:N", scale=_method_scale(methods), legend=None),
        )
    )
    target = (
        alt.Chart(pd.DataFrame({"coverage": [confidence]}))
        .mark_rule(color="#d1495b", strokeDash=[4, 4], strokeWidth=2)
        .encode(y="coverage:Q")
    )
    return (curve + target).properties(height=250)

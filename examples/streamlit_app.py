"""Interactive comparison of PITCP and split conformal prediction."""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import zuko

from pitcp import PITCP, SCP

st.set_page_config(page_title="PIT-CP playground", page_icon="🎯", layout="wide")


def _scale(x):
    return np.abs(1 - 2 * x**2) + 0.1


@st.cache_resource(show_spinner="Loading the pretrained example model…")
def example():
    """Load the pretrained model and recreate its deterministic example data."""
    rng = np.random.RandomState(42)

    def sample(n):
        x = rng.rand(n) * 2 - 1
        y = rng.randn(n) * _scale(x)
        return x[:, None], y

    sample(5000)
    _, y_cal = sample(1000)
    x_test, y_test = sample(5000)

    torch.manual_seed(42)
    density = zuko.flows.SOSPF(features=1, context=1, hidden_features=(32, 32))
    optimizer = torch.optim.Adam(density.parameters(), lr=1e-3)
    pitcp = PITCP(
        density,
        optimizer,
        n_epochs=200,
        batch_size=512,
        verbose=False,
        random_state=42,
    )
    saved = torch.load(
        Path(__file__).with_name("pitcp_example.pt"),
        map_location="cpu",
        weights_only=True,
    )
    pitcp.load_state_dict(saved["state_dict"])
    pitcp.scores_ = saved["scores"].numpy()
    pitcp.n_features_in_ = 1
    pitcp.eval()
    scp = SCP().conformalize(np.abs(y_cal))
    return pitcp, scp, x_test, y_test


st.title("PIT-CP playground")
st.write(
    "PITCP learns how score distributions change with the input. Compare its "
    "adaptive region with the constant-width region from split conformal prediction."
)

method = st.selectbox("Method", ["PITCP", "SCP"])
confidence = st.slider("Confidence level", 0.50, 0.99, 0.90, 0.01)

pitcp, scp, x_test, y_test = example()
model = pitcp if method == "PITCP" else scp
x_grid = np.linspace(-1, 1, 250).reshape(-1, 1)

limits = model.predict(x_grid, confidence_level=confidence)
covered = (
    model.contains(x_test, np.abs(y_test), confidence_level=confidence)
    if method == "PITCP"
    else model.contains(np.abs(y_test), confidence_level=confidence)
)

region = pd.DataFrame({"x": x_grid[:, 0], "lower": -limits, "upper": limits})
points = pd.DataFrame(
    {
        "x": x_test[::5, 0],
        "y": y_test[::5],
        "covered": np.where(covered[::5], "Covered", "Missed"),
    }
)

band = (
    alt.Chart(region)
    .mark_area(color="#6750a4", opacity=0.25)
    .encode(x=alt.X("x:Q", title="x"), y="lower:Q", y2="upper:Q")
)
boundary = (
    alt.Chart(region)
    .transform_fold(["lower", "upper"])
    .mark_line(color="#6750a4")
    .encode(x="x:Q", y=alt.Y("value:Q", title="y"), detail="key:N")
)
observations = (
    alt.Chart(points)
    .mark_circle(size=25, opacity=0.65)
    .encode(
        x="x:Q",
        y="y:Q",
        color=alt.Color(
            "covered:N",
            scale=alt.Scale(domain=["Covered", "Missed"], range=["#327a4c", "#c43c35"]),
        ),
        tooltip=["x:Q", "y:Q", "covered:N"],
    )
)

left, right = st.columns([2, 1])
with left:
    st.altair_chart((band + boundary + observations).interactive(), width="stretch")

edges = np.linspace(-1, 1, 9)
bin_index = np.digitize(x_test[:, 0], edges[1:-1])
coverage_by_bin = pd.DataFrame(
    {
        "x": (edges[:-1] + edges[1:]) / 2,
        "coverage": [covered[bin_index == i].mean() for i in range(8)],
    }
)
target = pd.DataFrame({"coverage": [confidence]})
theoretical = pd.DataFrame(
    {
        "x": x_grid[:, 0],
        "coverage": torch.erf(
            torch.as_tensor(limits / (_scale(x_grid[:, 0]) * np.sqrt(2)))
        ).numpy(),
    }
)

with right:
    st.metric(
        "Empirical coverage",
        f"{covered.mean():.1%}",
        f"{covered.mean() - confidence:+.1%}",
    )
    bars = (
        alt.Chart(coverage_by_bin)
        .mark_bar(color="#6750a4", opacity=0.35, size=25)
        .encode(
            x=alt.X("x:Q", title="Input bin"),
            y=alt.Y("coverage:Q", title="Coverage", scale=alt.Scale(domain=[0, 1])),
            tooltip=["x:Q", alt.Tooltip("coverage:Q", format=".1%")],
        )
    )
    theoretical_line = (
        alt.Chart(theoretical)
        .mark_line(color="#6750a4", strokeDash=[3, 3], strokeWidth=3)
        .encode(x="x:Q", y="coverage:Q")
    )
    target_line = alt.Chart(target).mark_rule(color="#c43c35").encode(y="coverage:Q")
    st.altair_chart(bars + theoretical_line + target_line, width="stretch")
    st.caption("Bars: empirical · Dotted: theoretical · Red: target")

with st.expander("Show the core API calls"):
    st.code(
        """density = zuko.flows.SOSPF(features=1, context=1, hidden_features=(32, 32))
optimizer = torch.optim.Adam(density.parameters(), lr=1e-3)
model = PITCP(density, optimizer, n_epochs=200, batch_size=512)
model.fit(X_train, abs(y_train)).conformalize(X_cal, abs(y_cal))

limits = model.predict(X_test, confidence_level=0.9)
covered = model.contains(X_test, abs(y_test), confidence_level=0.9)""",
        language="python",
    )

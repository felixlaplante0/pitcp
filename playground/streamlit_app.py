"""Streamlit application comparing every conformal model in pitcp."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v2 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground.utils import (
    COLORS,
    METHODS,
    _coverage_chart,
    _data,
    _fit_cqr,
    _models,
    _prediction,
    _region_chart,
    _summary,
    _theoretical_coverage,
)

ROOT = Path(__file__).resolve().parent
N_TEST = 2500

_METHOD_SELECTOR = components.component(
    "pitcp_method_selector",
    html='<div class="label">Methods</div><div class="methods" role="group"></div>',
    css="""
        .label { margin-bottom: .45rem; font-size: .875rem; font-weight: 600; }
        .methods {
            display: grid;
            grid-template-columns: repeat(5, minmax(5rem, 1fr));
            gap: .4rem;
        }
        button {
            min-height: 2.4rem;
            border: 1px solid var(--method-color);
            border-radius: .5rem;
            background: transparent;
            color: var(--method-color);
            font: inherit;
            font-weight: 650;
            cursor: pointer;
            opacity: .45;
            transition: opacity .12s ease, background .12s ease,
                transform .12s ease;
        }
        button:hover { opacity: .8; transform: translateY(-1px); }
        button.active { background: var(--method-color); color: white; opacity: 1; }
        button:focus-visible {
            outline: 2px solid var(--method-color);
            outline-offset: 2px;
        }
        @media (max-width: 700px) {
            .methods { grid-template-columns: repeat(2, 1fr); }
        }
    """,
    js="""
        export default function(component) {
            const { data, parentElement, setStateValue } = component;
            const container = parentElement.querySelector('.methods');
            const selected = new Set(data.selected);
            container.replaceChildren();

            for (const method of data.methods) {
                const button = document.createElement('button');
                const active = selected.has(method.name);
                button.type = 'button';
                button.textContent = method.name;
                button.className = active ? 'active' : '';
                button.setAttribute('aria-pressed', active);
                button.style.setProperty('--method-color', method.color);
                button.onclick = () => {
                    if (selected.has(method.name)) {
                        selected.delete(method.name);
                    } else {
                        selected.add(method.name);
                    }
                    setStateValue(
                        'selected',
                        data.methods
                            .map(item => item.name)
                            .filter(name => selected.has(name)),
                    );
                };
                container.appendChild(button);
            }
        }
    """,
)


def _method_selector() -> tuple[str, ...]:
    state = st.session_state.get("method-selector", {})
    selected = state.get("selected", ["PITCP"])
    result = _METHOD_SELECTOR(
        key="method-selector",
        data={
            "methods": [{"name": name, "color": COLORS[name]} for name in METHODS],
            "selected": selected,
        },
        default={"selected": selected},
        on_selected_change=lambda: None,
    )
    selected = result.selected or []
    return tuple(name for name in METHODS if name in selected)


@st.cache_data
def _cached_data() -> dict[str, np.ndarray]:
    return _data()


@st.cache_resource(show_spinner="Loading pretrained density models…")
def _cached_models() -> dict[str, object]:
    return _models(ROOT)


@st.cache_resource(show_spinner=False)
def _cached_cqr(confidence: float) -> object:
    return _fit_cqr(confidence, _cached_data())


@st.cache_data(show_spinner=False)
def _cached_results(
    confidence: float,
) -> tuple[dict[str, tuple[np.ndarray, ...]], list[dict[str, str]]]:
    data = _cached_data()
    models = dict(_cached_models())
    models["CQR"] = _cached_cqr(confidence)
    x_grid = np.linspace(-1, 1, 250).reshape(-1, 1)
    results = {}
    summaries = []
    for name in METHODS:
        x_test = data["x_test"]
        y_test = data["y_test"]
        lower, upper, covered = _prediction(
            name,
            models[name],
            x_grid,
            x_test,
            y_test,
            confidence,
        )
        theoretical = _theoretical_coverage(x_grid[:, 0], lower, upper)
        results[name] = lower, upper, covered, theoretical
        summaries.append(
            _summary(
                name,
                lower,
                upper,
                theoretical,
            )
        )
    return results, summaries


def main():  # noqa: D103
    st.set_page_config(page_title="PIT-CP playground", page_icon="🎯", layout="wide")
    st.title("PIT-CP playground")
    st.write(
        "Compare all five estimators on the heteroscedastic candy example used in "
        "the experiments. Density models are pretrained; CQR is fitted for the "
        "confidence level you request."
    )

    st.session_state.setdefault("applied_confidence", 0.9)
    model_column, level_column = st.columns([2, 3])
    with model_column:
        selected_methods = _method_selector()
    with level_column, st.form("level"):
        slider_column, apply_column = st.columns([4, 1], vertical_alignment="bottom")
        requested_confidence = slider_column.slider(
            "Confidence level", 0.50, 0.99, 0.90, 0.01
        )
        submitted = apply_column.form_submit_button("Apply", type="primary")
    if submitted:
        st.session_state["applied_confidence"] = requested_confidence
    confidence = st.session_state["applied_confidence"]
    st.caption(f"Applied confidence level: {confidence:.0%}")
    if not selected_methods:
        st.info("Select at least one method to show its regions.")
        return

    data = _cached_data()
    x_grid = np.linspace(-1, 1, 250).reshape(-1, 1)
    with st.spinner("Preparing the five-method comparison…"):
        results, summaries = _cached_results(confidence)

    region_chart = _region_chart(
        results,
        selected_methods,
        x_grid,
        data["x_test"],
        data["y_test"],
    )
    coverage_chart = _coverage_chart(
        results,
        selected_methods,
        x_grid,
        confidence,
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Prediction regions")
        st.altair_chart(region_chart.interactive(), width="stretch")
    with right:
        st.subheader("Conditional coverage")
        st.altair_chart(coverage_chart, width="stretch")
        st.caption("Solid: theoretical · Dotted red: target")

    summary_by_method = {summary["Method"]: summary for summary in summaries}
    st.subheader("Selected methods at this confidence level")
    st.dataframe(
        pd.DataFrame([summary_by_method[name] for name in selected_methods]).set_index(
            "Method"
        ),
        width="stretch",
    )


if __name__ == "__main__":
    main()

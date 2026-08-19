"""Smoke test for the interactive documentation example."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from pitcp import PITCP


def test_streamlit_playground(monkeypatch):
    """The saved model loads and reruns without fitting or raising exceptions."""
    confidence = 0.8
    app = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
    monkeypatch.setattr(PITCP, "fit", None)
    page = AppTest.from_file(app, default_timeout=60).run()

    assert not page.exception
    assert page.selectbox[0].value == "PITCP"

    page.slider[0].set_value(confidence).run()

    assert not page.exception
    assert page.slider[0].value == confidence

    page.selectbox[0].select("SCP").run()

    assert not page.exception
    assert page.selectbox[0].value == "SCP"

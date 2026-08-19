# Documentation pattern

Use the same small structure for sibling Python projects:

1. Write the project-specific explanation, theory, and quick start in
   `docs/source/index.rst`.
2. Generate the API reference from the package docstrings with Sphinx autodoc and
   autosummary. Keep the documentation in the source code authoritative.
3. Put an optional interactive example in `examples/streamlit_app.py` and embed its
   public Streamlit URL in `docs/source/playground.rst`.
4. Build the documentation on Read the Docs with `.readthedocs.yaml`; deploy the
   example separately on Streamlit Community Cloud.

Local checks:

```bash
python -m pip install -e ".[test]" -r docs/requirements.txt
sphinx-build -W -b html docs/source docs/_build/html
python -m pytest tests/test_streamlit_app.py
```

For another project, copy this structure and change only the introduction, API
autosummary entries, package name, and Streamlit example.

"""Configures the Sphinx documentation builder."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pitcp"
release = ""
version = ""
copyright = "2026, Félix Laplante"
author = "Félix Laplante"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_nb",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = True
autosummary_generate = True
add_module_names = False
napoleon_use_ivar = True
napoleon_attr_annotations = True
suppress_warnings = ["docutils"]
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "PIT-CP"
html_logo = "_static/pitcp-ribbon.svg"
html_favicon = "_static/pitcp-ribbon.svg"
html_theme_options = {
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/felixlaplante0/pitcp",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Playground",
            "url": "https://pitcp-app.streamlit.app/",
            "icon": "fa-solid fa-chart-line",
            "type": "fontawesome",
        },
    ],
}
html_sidebars = {"index": []}

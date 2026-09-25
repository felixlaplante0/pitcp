"""Configures the Sphinx documentation builder."""

project = "pitcp"
copyright = "2026, Félix Laplante"
author = "Félix Laplante"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_nb",
    "sphinx_design",
]

templates_path = ["_templates"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
add_module_names = False
napoleon_use_ivar = True
suppress_warnings = ["docutils"]
nb_execution_mode = "off"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "PIT-CP"
html_logo = "_static/pitcp-logo.svg"
html_favicon = "_static/pitcp-logo.svg"
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
html_sidebars = {"**": []}

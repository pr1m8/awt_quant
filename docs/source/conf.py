# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'awt_quant'
copyright = '2025, William R. Astley, Pedro Gaissler, Nico Jacobs'
author = 'William R. Astley, Pedro Gaissler, Nico Jacobs'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('../../awt_quant'))
# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',        # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',       # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.autosummary',    # Create summary tables for modules
    'sphinx_rtd_theme',          # ReadTheDocs theme
]

# Enable auto-generating summaries
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

# -- Autodoc settings --------------------------------------------------------
autodoc_member_order = 'bysource'  # Show members in the order they appear
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Napoleon settings (for docstrings) --------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
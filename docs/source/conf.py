import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'AWT-Quant'
copyright = '2025, William R. Astley, Pedro Gaissler, Nico Jacobs'
author = 'William R. Astley, Pedro Gaissler, Nico Jacobs'
release = '0.1.0'
import os
os.environ["NIXTLA_API_KEY"] = "dummy_key"

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'autoapi.extension',  # Enables automatic API documentation
    'sphinx.ext.autosummary',
]

# AutoAPI settings (auto-generates docs for all modules)
autoapi_dirs = ['../../awt_quant']
autoapi_type = 'python'
autoapi_generate_api_docs = True
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'special-members',
]

# Napoleon settings (for better docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- HTML Output Settings ---------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# Configuration file for the Sphinx documentation builder
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# Add the project root to sys.path for proper module discovery
sys.path.insert(0, os.path.abspath('../../'))

# Set environment variables required for certain modules
os.environ["NIXTLA_API_KEY"] = "dummy_key"

# -- Project information -----------------------------------------------------
project = 'AWT-Quant'
copyright = f'2023-{datetime.now().year}, William R. Astley, Pedro Gaissler, Nico Jacobs'
author = 'William R. Astley, Pedro Gaissler, Nico Jacobs'
release = '0.1.4'
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings
extensions = [
    # Core Sphinx extensions
    'sphinx.ext.autodoc',         # Auto-generate API documentation
    'sphinx.ext.napoleon',        # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',        # Add links to view source code
    'sphinx.ext.githubpages',     # GitHub Pages support
    'sphinx.ext.mathjax',         # Render mathematical equations
    'sphinx.ext.intersphinx',     # Link to other projects' documentation
    'sphinx.ext.autosummary',     # Generate summary tables for API docs
    'sphinx.ext.todo',            # Support for TODO items
    
    # Third-party extensions
    'autoapi.extension',          # Automatic API documentation generation
    'sphinx_copybutton',          # Add copy buttons to code blocks
    'myst_parser',                # Markdown support
]

# Configure MathJax for financial equations
mathjax3_config = {
    'tex': {
        'macros': {
            'E': '{\\mathbb{E}}',
            'Var': '{\\mathrm{Var}}',
            'Cov': '{\\mathrm{Cov}}',
        }
    }
}

# AutoAPI settings for automatic API doc generation
autoapi_type = 'python'
autoapi_dirs = ['../../awt_quant']
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'special-members',
]
autoapi_python_class_content = 'both'
autoapi_template_dir = '_templates/autoapi' if os.path.exists('_templates/autoapi') else None
autoapi_keep_files = True

# Napoleon settings for parsing docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
}

# Enable todo items
todo_include_todos = True

# Add any paths that contain templates
templates_path = ['_templates']

# The suffix(es) of source filenames
source_suffix = ['.rst', '.md']

# The master toctree document
master_doc = 'index'

# List of patterns to exclude from source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'furo'

# Create required directories if they don't exist
os.makedirs(os.path.join(os.path.dirname(__file__), '_static'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '_static/css'), exist_ok=True)

# Theme-specific options
html_theme_options = {
    # Sidebar options
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    
    # Announcement at the top of the page
    "announcement": "AWT Quant v0.1.4 - Advanced Quantitative Forecasting",
    
    # Light mode variables
    "light_css_variables": {
        # Primary accent color for links, buttons
        "color-brand-primary": "#1a73e8",
        "color-brand-content": "#1a73e8",
        
        # Colors for specific admonitions and notices
        "color-admonition-background": "#e8f0fe",
        
        # Background colors
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f8f9fa",
        
        # Code block colors
        "color-code-background": "#f5f7f9",
        "color-code-foreground": "#24292e",
    },
    
    # Dark mode variables
    "dark_css_variables": {
        "color-brand-primary": "#8ab4f8",
        "color-brand-content": "#8ab4f8",
        "color-admonition-background": "#303134",
        "color-background-primary": "#202124",
        "color-background-secondary": "#303134",
        "color-code-background": "#1a1a1a",
        "color-code-foreground": "#e6e6e6",
    },
    
    # Footer icons
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/pr1m8/awt_quant",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Add any paths that contain custom static files
html_static_path = ['_static']
html_css_files = ['css/custom.css']

# Logo and favicon - these will be used if available
html_logo = '_static/images/logo.png' if os.path.exists(os.path.join(os.path.dirname(__file__), '_static/images/logo.png')) else None
html_favicon = '_static/images/favicon.ico' if os.path.exists(os.path.join(os.path.dirname(__file__), '_static/images/favicon.ico')) else None

# -- Options for HTMLHelp output ---------------------------------------------
htmlhelp_basename = 'AWTQuantdoc'

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{mathtools}
        % Additional commands for financial mathematics
        \newcommand{\E}{\mathbb{E}}
        \newcommand{\Var}{\mathrm{Var}}
        \newcommand{\Cov}{\mathrm{Cov}}
    ''',
    'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'AWT-Quant.tex', 'AWT Quant Documentation',
     author, 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'awt-quant', 'AWT Quant Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'AWT-Quant', 'AWT Quant Documentation',
     author, 'AWT-Quant', 'Advanced Quantitative Forecasting with SPDE, GARCH, and LLMs',
     'Miscellaneous'),
]

# -- Options for epub output -------------------------------------------------
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']

# -- Setup for autodoc ---------------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

def setup(app):
    """Set up custom processing for the documentation."""
    # Add custom stylesheets or JavaScript
    app.add_css_file('css/custom.css')
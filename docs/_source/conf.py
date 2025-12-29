"""
Configuration file for the Sphinx documentation builder.

This file contains the configuration for building comprehensive API documentation
for the AI Hydra agent using the Read the Docs theme.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Debug information for ReadTheDocs
print(f"Python path: {sys.path}")
print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")

# Try to import the project to verify it's available
try:
    import ai_hydra
    print(f"Successfully imported ai_hydra from {ai_hydra.__file__}")
except ImportError as e:
    print(f"Warning: Could not import ai_hydra: {e}")
    # This is expected during initial setup

# -- Project information -----------------------------------------------------

project = 'AI Hydra'
copyright = '2024, AI Hydra Team'
author = 'AI Hydra Team'

# The full version, including alpha/beta/rc tags
release = '0.4.5'
version = '0.4.5'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'myst_parser',
]

# Try to import and use Read the Docs theme
try:
    import sphinx_rtd_theme
    extensions.append('sphinx_rtd_theme')
except ImportError:
    pass

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension ------------------------------------------

# This value selects what content will be inserted into the main body of an autoclass directive
autoclass_content = 'both'

# This value selects how the signature will be displayed for the class defined by autoclass directive
autodoc_class_signature = 'mixed'

# This value controls the docstrings inheritance
autodoc_inherit_docstrings = True

# This value controls the behavior of sphinx.ext.autodoc-skip-member event
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for autosummary extension --------------------------------------

# Boolean indicating whether to scan all found documents for autosummary directives
autosummary_generate = True

# -- Options for napoleon extension -----------------------------------------

# Enable parsing of Google style docstrings
napoleon_google_docstring = True

# Enable parsing of NumPy style docstrings
napoleon_numpy_docstring = True

# Include init docstrings in class docstring
napoleon_include_init_with_doc = False

# Include private members (like _membername) with docstrings in the documentation
napoleon_include_private_with_doc = False

# Include special members (like __membername__) with docstrings in the documentation
napoleon_include_special_with_doc = True

# Use the :param: role for each function parameter
napoleon_use_param = True

# Use the :type: role for each function parameter
napoleon_use_rtype = True

# Use the :keyword: role for each function keyword argument
napoleon_use_keyword = True

# -- Options for intersphinx extension --------------------------------------

# Mapping to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- Options for todo extension ---------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for coverage extension -----------------------------------------

# Set to True to enable coverage checking
coverage_show_missing_items = True
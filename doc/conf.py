#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" NNGT documentation build configuration file. """

import sys
import os, errno
import shlex
import re
import fnmatch
import importlib.util as imputil

import sphinx_bootstrap_theme


# Paths
current_directory = os.path.abspath('.')
sys.path.append(current_directory)
sys.path.append(current_directory + "/extensions")  # custom extensions folder

# Simlink to geometry.examples
src = os.path.abspath('../nngt/geometry/examples')
tgt = os.path.abspath('modules/examples/')

try:
    os.remove(tgt)               # remove existing simlink
except OSError as e:
    if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
        raise                    # raise if a different error occurred

os.symlink(src, tgt)


'''
If on rtd, the graph libraries are not available so they need to be mocked
'''

try:
    if imputil.find_spec("nest") is None:
        # NEST is not available
        raise ImportError
except:
    import mock
    mock_object = mock.Mock(__name__ = "Mock", __bases__ = (object,))

    class Mock(object):

        def __init__(self, *args, **kwargs):
            super(Mock, self).__init__()

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            return self

        def __getitem__(self, name):
            return self

        def __setitem__(self, name, value):
            pass

        @property
        def __version__(self):
            return ""

        def __iter__(self):
            return self

        def next(self):
            raise StopIteration

        def __next__(self):
            self.next()

        @property
        def __name__(self):
            return "Mock"

        @property
        def __bases__(self):
            return (object,)

    sys.modules["nest"] = Mock()


# -- Setup all autosum then start --------------------------------------------

# import nngt
import nngt

# import simulation & geospatial explicitely to avoid import conflict with
# lazy load
try:
    import nngt.geospatial
except:
    pass

import nngt.simulation

from autosum import gen_autosum

# find all *.in files

inputs = []
skip   = ('main-functions.rst.in', 'side-classes.rst.in', 'geometry.rst.in',
          'graph-classes.rst.in')

for root, dirnames, filenames in os.walk('.'):
    for filename in fnmatch.filter(filenames, '*.in'):
        if filename not in skip:
            inputs.append(os.path.join(root, filename))

# list of classes to ignore for each module
ignore = {
    'nngt.core': ("Graph", "Network", "SpatialGraph", "SpatialNetwork",
                  "Group", "MetaGroup", "MetaNeuralGroup", "NeuralPop",
                  "NeuralGroup"),
    'nngt.lib': ("custom", "decorate", "deprecated", "graph_tool_check",
                 "mpi_barrier", "mpi_checker", "mpi_random", "not_implemented",
                 "num_mpi_processes", "on_master_process", "seed"),
    'nngt.generation': ('connect_nodes', 'connect_groups', 'connect_types',
                        'connect_neural_types', 'random_rewire',
                        'lattice_rewire', 'connect_neural_groups')
}

for f in inputs:
    target = f[:-3]  # remove '.in'
    # find the module (what will come after nngt, it is the name of the file)
    last_dot = target.rfind('.')
    last_slash = target.rfind('/')
    module = target[last_slash + 1:last_dot]
    if module != 'nngt':
        module = 'nngt.' + module
    gen_autosum(f, target, module, 'full', ignore=ignore.get(module, None))

# Add nngt (functions)
source = current_directory + "/modules/nngt/main-functions.rst.in"
target = current_directory + "/modules/nngt/main-functions.rst"
gen_autosum(source, target, 'nngt', 'summary', dtype="func")
gen_autosum(target, target, 'nngt', 'autofunction', dtype="func")

# nngt (main classes)
source = current_directory + "/modules/nngt/graph-classes.rst.in"
target = current_directory + "/modules/nngt/graph-classes.rst"
gen_autosum(source, target, 'nngt.Graph', 'summary', dtype="classmembers")
gen_autosum(target, target, 'nngt.SpatialGraph', 'summary',
            dtype="classmembers")
gen_autosum(target, target, 'nngt.Network', 'summary', dtype="classmembers")
gen_autosum(target, target, 'nngt.SpatialNetwork', 'summary', dtype="classmembers")

# nngt (side classes)
source = current_directory + "/modules/nngt/side-classes.rst.in"
target = current_directory + "/modules/nngt/side-classes.rst"

gen_autosum(source, target, 'nngt', 'summary', dtype="class",
            ignore=("Graph", "Network", "SpatialGraph", "SpatialNetwork",
                    "GroupProperty"))

gen_autosum(target, target, 'nngt.Group', 'summary', dtype="classmembers")
gen_autosum(target, target, 'nngt.NeuralGroup', 'summary', 
            dtype="classmembers")
gen_autosum(target, target, 'nngt.Structure', 'summary', dtype="classmembers")
gen_autosum(target, target, 'nngt.NeuralPop', 'summary', dtype="classmembers")

gen_autosum(target, target, 'nngt', 'autoclass', dtype="class",
            ignore=("Graph", "Network", "SpatialGraph", "SpatialNetwork",
                    "GroupProperty"))

# geometry
source = current_directory + "/modules/geometry.rst.in"
target = current_directory + "/modules/geometry.rst"
gen_autosum(source, target, 'nngt.geometry', 'summary', dtype="all")

# generation
source = current_directory + "/modules/generation.rst.in"
target = current_directory + "/modules/generation.rst"
gen_autosum(source, target, 'nngt.generation', 'summary', dtype="func",
            ignore=ignore['nngt.generation'])
gen_autosum(target, target, 'nngt.generation', 'autofunction', dtype="func")




# -- NNGT setup -----------------------------------------------------------

from nngt import __version__ as nngt_version

# set database
try:
    import peewee
    nngt.set_config("use_database", True)
except ImportError:
    pass


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'linksourcecode',
    'extlinks_fancy',
    'sphinx_gallery.gen_gallery',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'NNGT'
copyright = u'2015, Tanguy Fardet'
author = u'Tanguy Fardet'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = nngt_version
while version[-1].isalpha():
    version = version[:-1]
while version.count('.') > 1:
    idx_last_dot = version.rfind('.')
    version = version[:idx_last_dot]
# The full version, including alpha/beta/rc tags.
release = nngt_version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['.build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#~ html_theme = 'sphinx_rtd_theme'

html_theme = 'nngt_theme'
html_theme_path = ["."] + sphinx_bootstrap_theme.get_html_theme_path()
html_use_smartypants = False

html_theme_options = {
    # A list of tuples containing pages or urls to link to.
    # Valid tuples should be in the following forms:
    #    (name, page)                 # a link to a page
    #    (name, "/aa/bb", 1)          # a link to an arbitrary relative url
    #    (name, "http://example.com", True) # arbitrary absolute url
    # Note the "1" or "True" value above as the third argument to indicate
    # an arbitrary url.
    'navbar_links': [
        ("Modules", "py-modindex"),
        ("Index", "genindex"),
        ("SourceHut", "https://git.sr.ht/~tfardet/NNGT", True),
        ("GitHub", "https://github.com/tfardet/NNGT", True),
    ],

    # Render the next and previous page links in navbar. (Default: true)
    'navbar_sidebarrel': False,

    # Render the current pages TOC in the navbar. (Default: true)
    'navbar_pagenav': True,

    # Tab name for the current pages TOC. (Default: "Page")
    'navbar_pagenav_name': "Current",

    # Global TOC depth for "site" navbar tab. (Default: 1)
    # Switching to -1 shows all levels.
    'globaltoc_depth': 2,

    # Include hidden TOCs in Site navbar?
    #
    # Note: If this is "false", you cannot have mixed ``:hidden:`` and
    # non-hidden ``toctree`` directives in the same page, or else the build
    # will break.
    #
    # Values: "true" (default) or "false"
    'globaltoc_includehidden': "true",

    # Fix navigation bar to top of page?
    # Values: "true" (default) or "false"
    'navbar_fixed_top': "false",

    # Location of link to source.
    # Options are "nav" (default), "footer" or anything else to exclude.
    'source_link_position': "",

    # Bootswatch (http://bootswatch.com/) theme.
    'bootswatch_theme': "yeti"
}

html_sidebars = {'**': ['customtoc.html', 'searchbox.html']}


# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#~ html_logo = 'images/nngt_logo.png'
#~ html_logo = 'images/nngt_ico.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'images/nngt_ico.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# Set differently depending on the location?

html_static_path = ['_static']

# Add permalinks to headers
html_permalinks_icon = "#"

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#~ html_sidebars = {
   #~ '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
#~ }

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'NNGTdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',

# Latex figure (float) alignment
#'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, 'NNGT.tex', u'NNGT Documentation',
   u'Tanguy Fardet', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = 'images/nngt_logo.pdf'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'nngt', u'NNGT Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'NNGT', u'NNGT Documentation',
   author, 'NNGT', 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
texinfo_show_urls = 'inline'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False

autodoc_default_flags = ['members', 'undoc-members']
autodoc_docstring_signature = True
autosummary_generate = True
autodoc_order = 'bysource'
autoclass_content = 'both'
napoleon_include_special_with_doc = False
napoleon_use_param = False
napoleon_use_rtype = False
imported_members = True

intersphinx_mapping = {
    'cartopy': ('https://scitools.org.uk/cartopy/docs/latest/', None),
    'geopandas': ('https://geopandas.org/', None),
    'gt': ('https://graph-tool.skewed.de/static/doc/', None),
    'ipython': ('https://ipython.org/ipython-doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'shapely': ('https://shapely.readthedocs.io/en/latest/', None),
}

extlinks_fancy = {
    'doi': (['https://dx.doi.org/{0}'], ['DOI: {0}']),
    'arxiv': (['https://arxiv.org/abs/{0}'], ['arXiv: {0}']),
    'gtdoc': (['https://graph-tool.skewed.de/static/doc/{0}.html#graph_tool.{1}'], ['graph-tool - {0}']),
    'igdoc': (['https://igraph.org/python/api/latest/igraph._igraph.GraphBase.html#{0}'], ['igraph - {0}']),
    'nxdoc': (['https://networkx.org/documentation/stable/reference/{0}generated/networkx.{1}.html'], ['networkx - {0}'])
}

# sphinx gallery parameters

sphinx_gallery_conf = {
     'examples_dirs': ['examples/graph_structure', 'examples/graph_properties'],  # path to your example scripts
     'gallery_dirs': ['gallery/graph_structure', 'gallery/graph_properties'],  # path to where to save gallery generated output
     'thumbnail_size': (400, 400),
     'capture_repr': (),
}

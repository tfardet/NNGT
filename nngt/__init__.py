#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
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

"""
====
NNGT
====

Neural Networks Growth and Topology analyzing tool.

Provides algorithms for
	1. growing networks
	2. analyzing their activity
	3. studying the graph theoretical properties of those networks


Available subpackages
=====================

analysis
  Tools to study graph topology and neuronal activity.
core
  Contains the main network classes.
  classes and methods for users are loaded at the main module (`nngt`) level
  when the library is imported, so :class:`nngt.core` should not be used.
generation
  Functions to generate specific networks.
geometry
  Tools to work on metric graphs (see
  `PyNCulture <https://github.com/SENeC-Initiative/PyNCulture>`_).
io
  Tools for input/output operations.
lib
  Basic functions used by several sub-packages.
simulation
  Tools to provide complex network generation with NEST and help analyze the
  influence of the network structure on neuronal activity.
plot
  plot data or graphs using matplotlib and graph_tool.


Utilities
---------
get_config
	Show library _configuration
set_config
	Set library _configuration (graph library, multithreading...)
version
	NNGT version string


Units
=====

Functions related to spatial embedding of networks are using micrometers
(um) as default unit; other units from the metric system can also be
provided:

- `mm` for milimeters
- `cm` centimeters
- `dm` for decimeters
- `m` for meters


Main classes and functions
==========================
"""

import os as _os
import shutil as _shutil
import sys as _sys
import logging


version = '0.6.a'
''' :obj:`str`, current NNGT version '''


# ----------------------- #
# Requirements and config #
# ----------------------- #

# Python > 2.6
if _sys.hexversion < 0x02070000:
    _logger.critical('NNGT requires Python 2.7 or higher.')
    raise ImportError('NNGT requires Python 2.7 or higher.')

# configuration
_config = {
    'color_lib': 'matplotlib',
    'db_folder': "~/.nngt/database",
    'db_to_file': False,
    'db_url': "mysql:///nngt_db",
    'graph': object,
    'graph_library': "",
    'library': None,
    'load_nest': False,
    'log_folder': "~/.nngt/log",
    'log_level': 10,
    'log_to_file': False,
    'mpl_backend': None,
    'multithreading': False,
    'omp': 1,
    'palette': 'Set1',
    'seed': None,
    'use_database': False,
    'use_tex': False,
    'set_omp_graph_tool': False,
    'with_nest': False,
    'with_plot': False,
}

_lib_folder = _os.path.expanduser('~') + '/.nngt'
_new_config = _os.path.expanduser('~') + '/.nngt/nngt.conf'
_default_config = _os.path.dirname(_os.path.realpath(__file__)) + \
                  '/nngt.conf.default'

# check that library config folder exists
if not _os.path.isdir(_lib_folder):
    _os.mkdir(_lib_folder)

# IMPORTANT: first create logger
from .lib.logger import _configure_logger

_logger = logging.getLogger(__name__)
_configure_logger(_logger)

# IMPORTANT: afterwards, import config
from .lib.nngt_config import get_config, set_config, _load_config, _convert

# check that config file exists
if not _os.path.isfile(_new_config):  # if it does not, create it
    _shutil.copy(_default_config, _new_config)
else:                                 # if it does check it is up-to-date
    with open(_new_config, 'r') as fconfig:
        options = [l.strip() for l in fconfig if l.strip() and l[0] != "#"]
        config_version = ""
        for opt in options:
            sep = opt.find("=")
            opt_name = opt[:sep].strip()
            opt_val = _convert(opt[sep+1:].strip())
            if opt_name == "version":
                config_version = opt_val
        if config_version != version:
            _shutil.copy(_default_config, _new_config)
            _logger.warning("Updating the configuration file, your previous "
                            "settings have be overwritten.")


_load_config(_new_config)

# multithreading
_config["omp"] = int(_os.environ.get("OMP", 1))
if _config["omp"] > 1:
    _config["multithreading"] = True


# --------------------- #
# Loading graph library #
#---------------------- #

from .lib.graph_backends import use_library, analyze_graph

_libs = [ 'graph-tool', 'igraph', 'networkx' ]

try:
    use_library(_config['graph_library'], False)
except ImportError:
    idx = _libs.index(_config['graph_library'])
    del _libs[idx]
    keep_trying = True
    while _libs and keep_trying:
        try:
            use_library(_libs[-1], False)
            keep_trying = False
        except ImportError:
            _libs.pop()

if not _libs:
    raise ImportError("This module needs one of the following graph libraries "
                      "to work:  `graph_tool`, `igraph`, or `networkx`.")


# ------- #
# Modules #
# ------- #

# import some tools into main namespace

from .core.graph_datastruct import NeuralPop, NeuralGroup, GroupProperty
from .core.graph_classes import Graph, SpatialGraph, Network, SpatialNetwork
from .generation.graph_connectivity import generate
from .lib.rng_tools import seed


# import modules

from . import analysis
from . import core
from . import generation
from . import geometry
from . import lib


__all__ = [
    "analysis",
    "analyze_graph",
    "Connections",
    "core",
    "generate",
    "generation",
    "geometry",
    "get_config",
    "Graph",
    "GroupProperty",
    "lib",
    "Network",
    "NeuralGroup",
    "NeuralPop",
    "seed",
    "set_config",
    "SpatialGraph",
    "SpatialNetwork",
    "use_library",
    "version"
]


# test if plot module is supported

from . import plot
try:
    from . import plot
    _config['with_plot'] = True
    __all__.append('plot')
except ImportError as e:
    _logger.debug("Error, plot module will not be loaded: " + str(e))
    _config['with_plot'] = False


# look for nest

if _config['load_nest']:
    try:
        _sys.argv.append('--quiet')
        import nest
        from . import simulation
        _config['with_nest'] = nest.version()
        __all__.append("simulation")
    except ImportError as e:
        _logger.debug("NEST not found; nngt.simulation not loaded: " + str(e))
        _config["with_nest"] = False


# load database module if required

if _config["use_database"]:
    if _config["db_to_file"]:
        if not _os.path.isdir(_config["db_folder"]):
            _os.mkdir(_config["db_folder"])
    try:
        from .database import db
        __all__.append('db')
    except ImportError as e:
        _logger.debug("Could not load database module: " + str(e))


# ------------------------ #
# Print config information #
# ------------------------ #

_log_info = '''
    -----------
    NNGT loaded
    -----------
Graph library:  {gl}
Multithreading: {thread} ({omp} thread{s})
Plotting:       {plot}
NEST support:   {nest}
Database:       {db}
    -----------
'''.format(
    gl=_config["graph_library"],
    thread=_config["multithreading"],
    plot=_config["with_plot"],
    nest=_config["with_nest"],
    db=_config["use_database"],
    omp=_config["omp"],
    s="s" if _config["omp"] > 1 else ""
)

_logger.info(_log_info)

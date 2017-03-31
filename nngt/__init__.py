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

core
	Contains the main network classes.
	These are loaded in nngt at import so specifying :class:`nngt.core` is not necessary
generation
	Functions to generate specific networks
lib
	Basic functions used by several sub-packages.
io
	@todo: Tools for input/output operations
nest
	NEST integration tools
growth
	@todo: Growing networks tools
plot
	plot data or graphs (@todo) using matplotlib and graph_tool


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

Functions related to spatial embedding of networks are using milimeters
(mm) as default unit; other units from the metric system can also be
provided:

- `um` for micrometers
- `cm` centimeters
- `dm` for decimeters
- `m` for meters


Main classes and functions
==========================
"""

import os as _os
import shutil as _shutil
import sys as _sys


# ----------------------- #
# Requirements and config #
# ----------------------- #

# Python > 2.6
assert _sys.hexversion > 0x02060000, 'NNGT requires Python > 2.6'

# _configuration
_lib_folder = _os.path.expanduser('~') + '/.nngt'
_path_config = _os.path.expanduser('~') + '/.nngt/nngt.conf'
_nngt_root = _os.path.dirname(_os.path.realpath(__file__))
if not _os.path.isdir(_lib_folder):
    _os.mkdir(_lib_folder)
if not _os.path.isfile(_path_config):
    _shutil.copy(_nngt_root + '/nngt.conf.default', _path_config)

from .globals import (analyze_graph, _config, use_library, version, set_config,
                      get_config, seed)

# multithreading
_config["omp"] = int(_os.environ.get("OMP", 1))
if _config["omp"] > 1:
    _config["multithreading"] = True


# ------- #
# Modules #
# ------- #

# importing core directly
from .core import *
from .core.graph_datastruct import Shape, NeuralPop, GroupProperty, Connections
from .core.graph_classes import Graph, SpatialGraph, Network, SpatialNetwork
from .generation.graph_connectivity import generate

# import modules
from . import analysis
from . import core
from . import generation
from . import lib


__all__ = [
    "analysis",
    "analyze_graph",
    "Connections",
    "core",
    "generate",
    "generation",
    "get_config",
    "Graph",
    "GroupProperty",
    "lib",
    "Network",
    "NeuralPop",
    "seed",
    "set_config",
    "Shape",
    "SpatialGraph",
    "SpatialNetwork",
    "use_library",
    "version"
]

# test if plot module is supported
try:
    from . import plot
    _config['with_plot'] = True
    __all__.append('plot')
except ImportError as e:
    ImportWarning("Error, plot module will not be loaded...", e)
    _config['with_plot'] = False

# look for nest
if _config['load_nest']:
    try:
        _sys.argv.append('--quiet')
        import nest
        from . import simulation
        _config['with_nest'] = True
        __all__.append("simulation")
    except ImportError as e:
        ImportWarning("NEST not found; nngt.simulation not loaded...", e)
        _config["with_nest"] = False
    
# load database module if required
if _config["set_logging"]:
    if _config["to_file"]:
        if not _os.path.isdir(_config["log_folder"]):
            _os.mkdir(_config["log_folder"])
    try:
        from .database import db
        __all__.append('db')
    except ImportError as e:
        ImportWarning("Could not load database module", e)

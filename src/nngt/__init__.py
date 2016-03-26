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
show_config
	@todo: Show build configuration
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

from __future__ import absolute_import
import os
import shutil
import sys



#-----------------------------------------------------------------------------#
# Requirements and config
#------------------------
#

# Python > 2.6
assert(sys.hexversion > 0x02060000)

# configuration
lib_folder = os.path.expanduser('~') + '/.nngt'
path_config = os.path.expanduser('~') + '/.nngt/nngt.conf'
nngt_root = os.path.dirname(os.path.realpath(__file__))
if not os.path.isdir(lib_folder):
    os.mkdir(lib_folder)
if not os.path.isfile(path_config):
    shutil.copy(nngt_root + '/nngt.conf.default', path_config)

from .globals import analyze_graph, config, use_library, version


#-----------------------------------------------------------------------------#
# Modules
#------------------------
#

from .core import *
from .core.graph_classes import Graph, SpatialGraph, Network, SpatialNetwork
from .core.graph_datastruct import Shape, NeuralPop, GroupProperty, Connections
from .generation.graph_connectivity import generate

from . import core
from . import generation
from . import analysis
from . import lib


__all__ = [
    "analysis",
    "analyze_graph",
    "config",
    "Connections",
    "core",
    "generate",
    "generation",
    "Graph",
    "GroupProperty",
    "lib",
    "Network",
    "NeuralPop",
    "Shape",
    "SpatialGraph",
    "SpatialNetwork",
    "use_library",
    "version"
]

# test if plot module is supported
try:
    from . import plot
    config['with_plot'] = True
    __all__.append('plot')
except ImportError as e:
    print("Error, plot module will not be loaded...", e)

# look for nest
try:
    sys.argv.append('--quiet')
    import nest
    from . import simulation
    config['with_nest'] = True
    __all__.append("simulation")
except ImportError as e:
    print("NEST not found; nest module will not be loaded...", e)
    
# load database module if required
if config["set_logging"]:
    if config["to_file"]:
        if not os.path.isdir(config["log_folder"]):
            os.mkdir(config["log_folder"])
    try:
        from .database import db
        __all__.append('db')
    except ImportError as e:
        print("Could not load database module", e)

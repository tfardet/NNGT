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
import sys
import socket


#-----------------------------------------------------------------------------#
# Requirements
#------------------------
#

# Python > 2.6
assert(sys.hexversion > 0x02060000)

# version and graph library test
from .globals import version, use_library


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
_with_plot = False
try:
    from . import plot
    _with_plot = True
    __all__.append('plot')
except:
    print("Uncompatibility, plot module will not be loaded...")

# look for nest
_with_nest = False
try:
    sys.argv.append('--quiet')
    import nest
    from . import simulation
    _with_nest = True
    __all__.append("simulation")
except:
    print("NEST not found; nest module will not be loaded...")

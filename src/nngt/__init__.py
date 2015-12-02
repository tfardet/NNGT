"""
NNGT
=====

Neural Networks Growth and Topology analyzing tool.

Provides algorithms for
	1. growing networks
	2. analyzing their activity
	3. studying the graph theoretical properties of those networks

How to use the documentation
----------------------------
Documentation is not yet really available. I will try to implement more
extensive docstrings within the code.
I recommend exploring the docstrings using
`IPython <http://ipython.scipy.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.
The docstring examples assume that `numpy` has been imported as `np`::

	>>> import numpy as np

Code snippets are indicated by three greater-than signs::

	>>> x = 42
	>>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

	>>> help(nggt.GraphClass)

Available subpackages
---------------------
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
random
	@todo? Random numbers generation tools
growth
	@todo: Growing networks tools

Utilities
---------
plot
	plot data or graphs (@todo) using matplotlib and graph_tool
show_config
	@todo: Show build configuration
version
	NNGT version string

Units
-----
Functions related to spatial embedding of networks are using milimeters
(mm) as default unit; other units from the metric system can also be
provided:

	- `um` for micrometers
	- `cm` centimeters
	- `dm` for decimeters
	- `m` for meters

"""

from __future__ import absolute_import
import sys


#-----------------------------------------------------------------------------#
# Requirements
#------------------------
#

# Python > 2.6
assert sys.hexversion > 0x02060000

# graph library
try:
    import graph_tool
except:
    try:
        import snap
    except:
        raise ImportError(
            "This module needs one of the following graph libraries to work: \
`graph_tool`, `apgl`, `igraph` or `SNAP`.")


#-----------------------------------------------------------------------------#
# Modules
#------------------------
#

from .core import *
from .core.graph_classes import (Graph, SpatialGraph, Network,
                                 SpatialNetwork)
from .core.graph_datastruct import Shape, NeuralPop, GroupProperty, Connections

from . import core
from . import generation
from . import analysis
from . import plot
from . import lib

try:
    import nest
    from . import nest
except:
    print("NEST not found; nest module will not be loaded")

#~ from . import io
#~ from . import random


#-----------------------------------------------------------------------------#
# Dict
#------------------------
#

# @todo: ensure classes cannot be instantiated from both nngt and nngt.core

__all__ = [
    "Graph",
    "SpatialGraph",
    "Network",
    "SpatialNetwork",
    "Shape",
    "NeuralPop",
    "GroupProperty",
    "Connections",
    "core",
    "generation",
    "analysis",
    "plot",
    "lib"
]

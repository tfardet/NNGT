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
	Contains the main network classes
	These are loaded in nngt at import so specifying :class:`nngt.core` is not necessary
generation
	Functions to generate specific networks
lib
	Basic functions used by several sub-packages.
io
	Tools for input/output operations
nest
	NEST integration tools
random
	Random numbers generation tools
growth
	Growing networks tools

Utilities
---------
plot
	@todo: plots graphs or data using matplotlib and graph_tool
show_config
	@todo: Show build configuration
__version__
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

from .constants import *
from .core import *
from . import core
from . import generation
from . import lib
#~ from . import io
#~ from . import random

__all__ = core.__all__ + cst__all__

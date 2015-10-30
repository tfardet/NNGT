"""
======
AGNet
======

Provides tools for
  1. growing networks
  2. analyzing their activity
  3. studying the graph theoretical properties of those networks
  
How to use the documentation
----------------------------
Documentation is not yet really available. I will try to implement more
extensive docstrings within the code.
I recommend exploring the docstrings using
`IPython <http://ipython.scipy.org>', an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.
The docstring examples assume that `numpy' has been imported as `np'::
  >>> import numpy as np
Code snippets are indicated by three greater-than signs::
  >>> x = 42
  >>> x = x + 1
Use the built-in ``help'' function to view a function's docstring::
  >>> help(agnet.Graph)

Available subpackages
---------------------
core
    Contains the main network classes
generation
	Functions to generate specific networks
lib
    Basic functions used by several sub-packages.
io
	Tools for input/output operations
nest
	NEST integration tools
random
    Core Random Tools
growth
    Growing networks tools
    
Utilities
---------
plot
    @todo: plots graphs or data using matplotlib and graph_tool
show_config
    @todo: Show build configuration
__version__
    AGNet version string

Units
------
Functions related to spatial embedding of networks are using milimeters
(mm) as default unit; other units from the metric system can also be
provided:
	- ``um'' for micrometers
	- ``cm'' centimeters
	- ``dm'' for decimeters
	- ``m'' for meters

"""

from __future__ import absolute_import

from graph_tool import Graph

from .core import *
#~ from . import lib
#~ from . import io
#~ from . import random

__version__ = '0.1'

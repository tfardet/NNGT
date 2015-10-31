"""
========================
NEST INTEGRATION MODULE
========================
==================== =========================================================
Functions
==============================================================================
make_nest_network    Create a network in NEST from a Graph object
get_nest_network     Create a Graph object from a NEST network
==================== =========================================================

"""

from __future__ import absolute_import
from sys import modules
from .nest_graph import *



#
#---
# Dependencies
#---------------------

depends = ['nest', 'graph_tool', 'AGNet.core']

try:
	'nest' in modules.keys()
except:
	import nest


#
#---
# Declare functions
#---------------------

__all__ = [
	'make_nest_network',
	'get_nest_network'
]

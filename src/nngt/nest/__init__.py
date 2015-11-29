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

from sys import modules



#
#---
# Dependencies
#---------------------

depends = ['nest', 'graph_tool', 'nngt.core']

from .nest_graph import *


#
#---
# Declare functions
#---------------------

__all__ = [
	'make_nest_network',
	'get_nest_network'
]

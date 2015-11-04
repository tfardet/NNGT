"""
Core module
===========

================== =====================================================
Classes
========================================================================
GraphClass			Main object
SpatialGraph        Spatially-embedded graph
Network		        More detailed network that inherits from GraphClass
SpatialNetwork      Spatially-embedded network
InputConnect		Connectivity to input analogic signals on a graph
================== =====================================================


Contents
--------

"""

from .graph_measures import *


__all__ = [ 
	"degree_list",
	"betweenness_list",
	"assortativity",
	"reciprocity",
	"clustering",
	"num_iedges",
	"num_scc",
	"num_wcc",
	"diameter",
	"spectral_radius"
]

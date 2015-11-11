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

from .graph_objects import GraphObject
from .graph_measures import *
#~ from .InputConnect import InputConnect # there is some problem in this file


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

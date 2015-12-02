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


__all__ = [ 
	"GraphObject"
]

"""
Core module
===========

================== =====================================================
Classes
========================================================================
GraphClass			Main object
InputConnect		Connectivity to input analogic signals on a graph
NeuralNetwork		More detailed network that inherits from GraphClass
================== =====================================================


Contents
--------

"""

from .GraphClass import GraphClass
from .SpatialGraph import SpatialGraph
from .NeuralNetwork import NeuralNetwork
from .SpatialNetwork import SpatialNetwork
from .InputConnect import InputConnect
from ..lib import errors

depends = ['graph_tool']

__all__ = [
	'GraphClass',
    'SpatialGraph',
	'NeuralNetwork',
    'SpatialNetwork',
	'InputConnect'
]

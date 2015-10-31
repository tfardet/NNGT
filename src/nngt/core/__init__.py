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

================== =====================================================
Functions
========================================================================
make_nest_network	Create a network in NEST from a Graph object
get_nest_network	Create a Graph object from a NEST network
================== =====================================================

================== =====================================================
Errors
========================================================================
InvalidArgument		Argument passed to the function are not valid
================== =====================================================

Summary
+++++++

.. autosummary::
	:nosignatures:

	nngt.core.GraphClass
	nngt.core.InputConnect

Contents
++++++++

"""

from .GraphClass import GraphClass
from NeuralNetwork import NeuralNetwork
from .InputConnect import InputConnect
from .errors import InvalidArgument

depends = ['graph_tool']

__all__ = [
	'GraphClass',
	'InputConnect',
	'NeuralNetwork',
	'InvalidArgument'
]

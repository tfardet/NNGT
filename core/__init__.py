"""
========================
CORE MODULE
========================

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
make_nest_network    Create a network in NEST from a Graph object
get_nest_network     Create a Graph object from a NEST network
================== =====================================================

================== =====================================================
Errors
========================================================================
InvalidArgument		Argument passed to the function are not valid
================== =====================================================

"""

depends = ['graph_tool']

__all__ = [
	'GraphClass',
	'InputConnect',
	'NeuralNetwork',
	'InvalidArgument'
]

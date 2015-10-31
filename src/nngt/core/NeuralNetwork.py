#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" NeuralNetwork: detailed biological network """

from GraphClass import GraphClass


#
#---
# NeuralNetwork
#------------------------

class NeuralNetwork(GraphClass):
	
	"""
	.. py:currentmodule:: core
	
	The detailed class that inherits from :class:`GraphClass` and implements
	additional properties to describe various biological functions
	and interact with the NESTsimulator.
	
	Parameters
	----------
	di_prop : dict, optional
		A dictionary containing the desired properties of the graph that
		will be generated. By default an empty graph is generated.
	graph : :class:`graph_tool.Graph`, optional
		An optional :class:`graph_tool.Graph` to serve as base.
	
	Returns
	-------
	self : :class:`~core.GraphClass`

	Attributes
	----------
	di_properties : dict
		Dictionary containing the properties of the :class:`graph_tool.Graph` object
	"""

	def __init__ (self, dicProp={"Name": "Graph", "Type": "None", "Weighted": False}, graph=None):
		''' init from properties '''
		None

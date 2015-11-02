#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" GraphClass: graph generation and management """

import warnings
from copy import deepcopy
from numpy import multiply

from graph_tool import Graph
from graph_tool.spectral import adjacency

from graph_measure import *



#
#---
# GraphClass
#------------------------

class GraphClass(object):
	
	"""
	.. py:currentmodule:: nggt.core
	
	The basic class that contains a :class:`graph_tool.Graph` and some
	of is properties or methods to easily access them.

	:ivar id: :class:`int`
		unique id that identifies the instance.
	:ivar graph: :class:`graph_tool.Graph`
		main attribute of the class instance.
	"""

	#------------------#
	# Class attributes #
	#------------------#

	__num_graphs = 0
	__max_id = 0
	__di_property_func = {
			"reciprocity": reciprocity, "clustering": clustering,
			"assortativity": assortativity, "diameter": diameter,
			"scc": num_scc, "wcc": num_wcc, "radius": spectral_radius, 
			"edges_inhib_frac": edge_inhib_frac }
	__properties = __di_property_func.keys()
	
	@classmethod
	def num_graphs(cls):
		''' Returns the number of alive instances. '''
		return cls.__num_graphs

	#----------------------------#
	# Instance-related functions #
	#----------------------------#

	def __init__ (self, nodes=0, name="Graph",
				  weighted=True, directed=True, graph=None):
		'''
		Initialize GraphClass instance

		Parameters
		----------
		nodes : int, optional (default: 0)
			Number of nodes in the graph.
		name : string, optional (default: "Graph")
			The name of this :class:`GraphClass` instance.
		weighted : bool, optional (default: True)
			Whether the graph edges have weight properties.
		directed : bool, optional (default: True)
			Whether the graph is directed or undirected.
		graph : :class:`graph_tool.Graph`, optional
			An optional :class:`graph_tool.Graph` to serve as base.
		
		Returns
		-------
		self : :class:`~nggt.core.GraphClass`
		'''
		self.__id = self.__class__.max_id
		self.__di_prop = {
			"id": self.__id,
			"name": name,
			"weighted": weighted,
			"directed": directed,
		}
		if graph != None:
			self._graph = graph
		else:
			self._graph = Graph(directed=directed)
		self.__class__.num_graphs += 1
		self.__class__.max_id += 1

	@property
	def id(self):
		''' unique :class:`int` identifying the instance '''
		return self.__id
	
	@property
	def graph(self):
		''' :class:`graph_tool.Graph` attribute of the instance '''
		return self._graph

	@graph.setter
	def graph(self, new_graph):
		if new_graph.__class__ == Graph:
			self._graph = new_graph
		else:
			raise TypeError("The object passed is not a\
			< class 'graph_tool.Graph' > but a {}".format(new_graph.__class__))

	def copy(self):
		'''
		Returns a deepcopy of the current :class:`~nngt.core.GraphClass`
		instance
		'''
		gc_instance = GraphClass(
							name=self.__di_prop["name"]+'_copy',
							weighted=self.__di_prop["weighted"],
							graph=self._graph.copy())
		return gc_instance

	#---------------------------#
	# Manipulating the gt graph #
	#---------------------------#

	def inhibitory_subgraph(self):
		''' Create a :class:`~nngt.core.GraphClass` instance which graph
		contains only the inhibitory connections of the current instance's
		:class:`graph_tool.Graph` '''
		eprop_b_type = self._graph.new_edge_property(
					   "bool",-self._graph.edge_properties["type"].a+1)
		self._graph.set_edge_filter(eprop_b_type)
		inhib_graph = GraphClass(
							name=self.__di_prop["name"]+'_inhib',
							weighted=self.__di_prop["weighted"],
							graph=Graph(self._graph,prune=True))
		self._graph.clear_filters()
		return inhib_graph

	def excitatory_subgraph(self):
		''' create a :class:`~nngt.core.GraphClass` instance which graph
		contains only the excitatory connections of the current instance's
		:class:`graph_tool.Graph` '''
		eprop_b_type = self._graph.new_edge_property(
					   "bool",self._graph.edge_properties["type"].a+1)
		self._graph.set_edge_filter(eprop_b_type)
		exc_graph = GraphClass(
							name=self.__di_prop["name"]+'_exc',
							weighted=self.__di_prop["weighted"],
							graph=Graph(self._graph,prune=True))
		self._graph.clear_filters()
		return exc_graph

	#-------------------------#
	# Set or update functions #
	#-------------------------#
		
	def set_name(self,name=""):
		''' set graph name '''
		if name != "":
			self.__di_prop["name"] = name
		else:
			strName = self.__di_prop["type"]
			tplIgnore = ("type", "name", "weighted")
			for key,value in self.__di_prop.items():
				if key not in tplIgnore and (value.__class__ != dict):
					strName += '_' + key[0] + str(value)
			self.__di_prop["name"] = strName

	#---------------#
	# Get functions #
	#---------------#

	## basic properties

	def get_name(self):
		return self.__di_prop["name"]
	
	def num_vertices(self):
		return self._graph.num_vertices()

	def num_edges(self):
		return self._graph.num_edges()

	def get_density(self):
		return self._graph.num_edges()/float(self._graph.num_vertices()**2)

	def is_weighted(self):
		return self.__di_prop["weighted"]

	def is_directed(self):
		return self.__di_prop["directed"]

	## adjacency matrix

	def get_mat_adjacency(self):
		return adjacency(self._graph, self.get_weights())

	## complex properties
	
	def get_property(self, s_property):
		''' Return the desired property or None for an incorrect one. '''
		if s_property in GraphClass.__properties:
			return GraphClass.__di_property_func[s_property](self._graph)
		else:
			warnings.warn("Ignoring request for unknown property \
						  '{}'".format(s_property))
			return None

	def get_properties(self, a_properties):
		'''
		Return a dictionary containing the desired properties

		Parameters
		----------
		a_properties : sequence
			List or tuple of strings of the property names.

		Returns
		-------
		di_result : dict
			A dictionary of values with the property names as keys.
		'''
		di_result = { prop: self.get_property(prop) for prop in a_properties }
		return di_result

	def get_degrees(self, strType="total", bWeights=True):
		lstValidTypes = ["in", "out", "total"]
		if strType in lstValidTypes:
			return degree_list(self._graph, strType, bWeights)
		else:
			warnings.warn("Ignoring invalid degree type '{}'".format(strType))
			return None

	def get_betweenness(self, bWeights=True):
		if bWeights:
			if not self.bWBetwToDate:
				self.wBetweeness = betweenness_list(self._graph, bWeights)
				self.wBetweeness = True
			return self.wBetweeness
		if not self.bBetwToDate and not bWeights:
			self.betweenness = betweenness_list(self._graph, bWeights)
			self.bBetwToDate = True
			return self.betweenness

	def get_edge_types(self):
		if "type" in self._graph.edge_properties.keys():
			return self._graph.edge_properties["type"].a
		else:
			return repeat(1, self._graph.num_edges())
	
	def get_weights(self):
		if self.is_weighted():
			epropW = self._graph.edge_properties["weight"].copy()
			epropW.a = multiply(epropW.a,
								self._graph.edge_properties["type"].a)
			return epropW
		else:
			return self._graph.edge_properties["type"].copy()

	#--------#
	# Delete #
	#--------#

	def __del__(self):
		self.__class__.__num_graphs -= 1

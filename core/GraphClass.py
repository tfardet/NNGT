#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" GraphClass: graph generation and management """

from __future__ import absolute_import

from copy import deepcopy
from numpy import multiply

from graph_tool.spectral import adjacency

#~ from .graph_generation import gen_er, gen_fs, gen_edr
#~ from .graph_measure import *



#
#---
# GraphClass
#------------------------

class GraphClass:
	
	"""
	.. py:currentmodule:: core
	
	The basic class that contains a graph_tool graph and its properties,
	with the possibility of generating it out of various topologies.
	
	Parameters
	----------
	di_prop: dict (optional)
		A dictionary containing the desired properties of the graph that
		will be generated. By default an empty graph is generated.
	graph: graph_tool.Graph (optional)
		An optional graph_tool.Graph to serve as base.
	
	Returns
    -------
    GraphClass object.
	"""

	#------------#
	# Initialize #
	#------------#

	def __init__ (self, dicProp={"Name": "Graph", "Type": "None", "Weighted": False}, graph=None):
		''' init from properties '''
		#~ self.dicProperties = deepcopy(dicProp)
		#~ self.dicGetProp = { "Reciprocity": get_reciprocity, "Clustering": get_clustering, "Assortativity": get_assortativity,
							#~ "Diameter": get_diameter, "SCC": get_num_scc, #"Spectral radius": get_spectral_radius, 
							#~ "WCC": get_num_wcc, "InhibFrac": get_inhib_frac }
		#~ self.dicGenGraph = { "Erdos-Renyi": gen_er, "Free-scale": gen_fs, "EDR": gen_edr }
		#~ # create a graph
		#~ if graph != None:
			#~ # use the one furnished
			#~ self.__graph = graph
			#~ self.update_prop()
			#~ self.bPropToDate = True
		#~ elif dicProp["Type"] == "None":
			#~ # create an empty graph
			#~ self.__graph = Graph()
			#~ self.bPropToDate = False
		#~ else:
			#~ # generate a graph of the requested type
			#~ self.__graph = self.dicGenGraph[dicProp["Type"]](self.dicProperties)
			#~ self.update_prop()
			#~ self.set_name()
			#~ self.bPropToDate = True

	@classmethod
	def from_graph_class(cls, graphToCopy):
		''' create new GraphClass instance as a deepcopy of another '''
		dicProperties = deepcopy(graphToCopy.get_dict_properties())
		gtGraph = graphToCopy.get_graph().copy()
		# create
		graphClass = cls(dicProperties, gtGraph)
		# set state of properties
		bPropToDate = deepcopy(graphToCopy.bPropToDate)
		bBetwToDate = deepcopy(graphToCopy.bBetwToDate)
		graphClass.bPropToDate = bPropToDate
		graphClass.bBetwToDate = bBetwToDate
		return graphClass

	def copy(self):
		''' returns a deepcopy of the graphClass instance '''
		graphCopy = GraphClass()
		graphCopy.set_graph(self.__graph.copy())
		graphCopy.update_prop()
		graphCopy.set_name(self.dicProperties["Name"]+'_copy')
		return graphCopy

	#---------------------------#
	# Manipulating the gt graph #
	#---------------------------#

	def set_graph(self, gtGraph):
		''' acquire a graph_tool graph as its own '''
		if gtGraph.__class__ == Graph:
			self.__graph = gtGraph
		else:
			raise TypeError("The object passed to 'copy_gt_graph' is not a < class 'graph_tool.Graph' > but a {}".format(gtGraph.__class__))

	def inhibitory_subgraph(self):
		''' create a GraphClass instance which graph contains only
		the inhibitory connections of the current instance's graph '''
		graph = self.graph.copy()
		epropType = graph.new_edge_property("bool",-graph.edge_properties["type"].a+1)
		graph.set_edge_filter(epropType)
		inhibGraph = GraphClass()
		inhibGraph.set_graph(Graph(graph,prune=True))
		inhibGraph.set_prop("Weighted", True)
		return inhibGraph

	def excitatory_subgraph(self):
		''' create a GraphClass instance which graph contains only
		the excitatory connections of the current instance's graph '''
		graph = self.graph.copy()
		epropType = graph.new_edge_property("bool",graph.edge_properties["type"].a+1)
		graph.set_edge_filter(epropType)
		excGraph = GraphClass()
		excGraph.set_graph(Graph(graph,prune=True))
		excGraph.set_prop("Weighted", True)
		return excGraph

	#-------------------------#
	# Set or update functions #
	#-------------------------#
		
	def set_name(self,name=""):
		''' set graph name '''
		if name != "":
			self.dicProperties["Name"] = name
		else:
			strName = self.dicProperties["Type"]
			tplIgnore = ("Type", "Name", "Weighted")
			for key,value in self.dicProperties.items():
				if key not in tplIgnore and (value.__class__ != dict):
					strName += '_' + key[0] + str(value)
			self.dicProperties["Name"] = strName

	def update_prop(self, lstProp=[]):
		''' update part or all of the graph properties '''
		if lstProp:
			for strPropName in lstProp:
				if strPropName in self.dicGetProp.keys():
					self.dicProperties[strPropName] = self.dicGetProp[strPropName](self.__graph)
				else:
					print("Ignoring unknown property '{}'".format(strPropName))
		else:
			self.dicProperties.update({ strPropName: self.dicGetProp[strPropName](self.__graph) for strPropName in self.dicGetProp.keys() })
			self.bPropToDate = True

	#---------------#
	# Get functions #
	#---------------#

	## basic properties

	def get_name(self):
		return self.dicProperties["Name"]
	
	def num_vertices(self):
		return self.__graph.num_vertices()

	def num_edges(self):
		return self.__graph.num_edges()

	def get_density(self):
		return self.__graph.num_edges()/float(self.__graph.num_vertices()**2)

	def is_weighted(self):
		return self.dicProperties["Weighted"]

	## graph and adjacency matrix
	
	def get_graph(self):
		self.bPropToDate = False
		self.bBetwToDate = False
		self.wBetweeness = False
		return self.__graph

	def get_mat_adjacency(self):
		return adjacency(self.__graph, self.get_weights())

	## complex properties
	
	def get_prop(self, strPropName):
		if strPropName in self.dicProperties.keys():
			if not self.bPropToDate:
				self.dicProperties[strPropName] = self.dicGetProp[strPropName](self.__graph)
			return self.dicProperties[strPropName]
		else:
			print("Ignoring request for unknown property '{}'".format(strPropName))

	def get_dict_properties(self):
		return self.dicProperties

	def get_degrees(self, strType="total", bWeights=True):
		lstValidTypes = ["in", "out", "total"]
		if strType in lstValidTypes:
			return degree_list(self.__graph, strType, bWeights)
		else:
			print("Ignoring invalid degree type '{}'".format(strType))
			return None

	def get_betweenness(self, bWeights=True):
		if bWeights:
			if not self.bWBetwToDate:
				self.wBetweeness = betweenness_list(self.__graph, bWeights)
				self.wBetweeness = True
			return self.wBetweeness
		if not self.bBetwToDate and not bWeights:
			self.betweenness = betweenness_list(self.__graph, bWeights)
			self.bBetwToDate = True
			return self.betweenness

	def get_types(self):
		if "type" in self.graph.edge_properties.keys():
			return self.__graph.edge_properties["type"].a
		else:
			return repeat(1, self.__graph.num_edges())
	
	def get_weights(self):
		if self.dicProperties["Weighted"]:
			epropW = self.__graph.edge_properties["weight"].copy()
			epropW.a = multiply(epropW.a, self.__graph.edge_properties["type"].a)
			return epropW
		else:
			return self.__graph.edge_properties["type"].copy()

	#--------#
	# Delete #
	#--------#

	#~ def __del__(self):
		#~ print("graph died")

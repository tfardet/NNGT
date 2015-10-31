#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Graph measurements on graph_tool.Graph objects """


import numpy as np
import scipy as sp
import scipy.sparse.linalg as spl

from graph_tool.centrality import betweenness
from graph_tool.correlations import assortativity
from graph_tool.stats import *
from graph_tool.topology import edge_reciprocity, label_components, pseudo_diameter
from graph_tool.clustering import global_clustering
from graph_tool.util import *
from graph_tool.spectral import *



#
#---
# Distributions
#------------------------

def degree_list(gtGraph, strType, bWeights=True):
	degPropMap = gtGraph.degree_property_map(strType)
	if "weight" in gtGraph.edge_properties.keys() and bWeights:
		degPropMap = gtGraph.degree_property_map(strType, gtGraph.edge_properties["weight"])
	return degPropMap.a

def betweenness_list(gtGraph, bWeights=True):
	if "weight" in gtGraph.edge_properties.keys() and bWeights:
		weightPropMap = gtGraph.copy_property(gtGraph.edge_properties["weight"])
		#~ weightPropMap.a = np.divide(np.repeat(1,len(weightPropMap.a)),weightPropMap.a) # this drastically changes the distribution
		weightPropMap.a = weightPropMap.a.max() - weightPropMap.a
		return betweenness(gtGraph, weight=weightPropMap)
	else:
		return betweenness(gtGraph)


#
#---
# Scalar pproperties
#------------------------

def get_assortativity(gtGraph):
	return assortativity(gtGraph,"total")[0]

def get_reciprocity(gtGraph):
	return edge_reciprocity(gtGraph)

def get_clustering(gtGraph):
	return global_clustering(gtGraph)[0]

def get_inhib_frac(gtGraph):
	numInhib = len(gtGraph.edge_properties["type"].a < 0)
	return float(numInhib)/gtGraph.num_edges()

def get_num_scc(gtGraph):
	vpropComp,lstHisto = label_components(gtGraph,directed=True)
	return len(lstHisto)

def get_num_wcc(gtGraph):
	vpropComp,lstHisto = label_components(gtGraph,directed=False)
	return len(lstHisto)

def get_diameter(gtGraph):
	return pseudo_diameter(gtGraph)[0]

def get_spectral_radius(gtGraph):
	weights = gtGraph.edge_properties["type"].copy()
	if "weight" in gtGraph.edge_properties.keys():
		weights.a = np.multiply(weights.a,gtGraph.edge_properties["weight"].a)
	matAdj = adjacency(gtGraph,weights)
	eigVal = [0]
	try:
		eigVal = spl.eigs(matAdj,return_eigenvectors=False)
	except spl.eigen.arpack.ArpackNoConvergence,err:
		eigVal = err.eigenvalues
	return np.max(np.absolute(eigVal))

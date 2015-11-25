#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Graph measurements on graph_tool.Graph objects """


import scipy as sp
import scipy.sparse.linalg as spl

from graph_tool.spectral import adjacency
from graph_tool.centrality import betweenness
from graph_tool.correlations import assortativity
from graph_tool.stats import *
from graph_tool.topology import edge_reciprocity, label_components, pseudo_diameter
from graph_tool.clustering import global_clustering
from graph_tool.util import *
from graph_tool.spectral import *



#
#---
# Scalar properties
#------------------------

def assortativity(lib_graph):
	return assortativity(lib_graph,"total")[0]

def reciprocity(lib_graph):
	return edge_reciprocity(lib_graph)

def clustering(lib_graph):
	return global_clustering(lib_graph)[0]

def num_iedges(lib_graph):
	numInhib = len(lib_graph.edge_properties["type"].a < 0)
	return float(numInhib)/lib_graph.num_edges()

def num_scc(lib_graph):
	vpropComp,lstHisto = label_components(lib_graph,directed=True)
	return len(lstHisto)

def num_wcc(lib_graph):
	vpropComp,lstHisto = label_components(lib_graph,directed=False)
	return len(lstHisto)

def diameter(lib_graph):
	return pseudo_diameter(lib_graph)[0]


#
#---
# Spectral properties
#------------------------

def spectral_radius(lib_graph):
	weights = lib_graph.edge_properties["type"].copy()
	if "weight" in lib_graph.edge_properties.keys():
		weights.a = sp.multiply(weights.a,
                                lib_graph.edge_properties["weight"].a)
	matAdj = adjacency(lib_graph,weights)
	eigVal = [0]
	try:
		eigVal = spl.eigs(matAdj,return_eigenvectors=False)
	except spl.eigen.arpack.ArpackNoConvergence,err:
		eigVal = err.eigenvalues
	return sp.max(sp.absolute(eigVal))

def adjacency_matrix(lib_graph):
    return adjacency(lib_graph)

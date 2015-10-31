#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Utilities from GraphClass generation """


#
#---
# Wrappers
#------------------------

def _check_connect_arguments(f):
	'''
	Wrapper to check the validity the arguments for a connectivity
	generating function from AGNet.generation.
	'''
	def wrapper(*args, **kw):
		''' @todo: check types
		check that density is in [0,1] also for given edges
		check that fractions are also in [0,1] '''
		return f(*args, **kw)
	return wrapper


#
#---
# Parameter calculations
#------------------------

def _compute_inhib_exc(nodes, edges, inodes_frac, iedges_frac):
	"""
	Compute the number of inhibitory and excitatory neurons in the
	network, as well as the number of inhibitory and excitatory
	connections.
	
	Parameters
    ----------
	nodes: int
		number of nodes in the graph.
	edges: int
		Number of connections in the graph.
	inodes_frac:
		Fraction of inhibitory neurons.
	iedges_frac: double
		Fraction of inhibitory connections.
	
	Returns
    -------
	enodes: int
		Number of excitatory nodes.
    inodes: int
		Number of inhibitory nodes in the network.
	eedges: int
		Number of excitatry connections.
	iedges: int
		Number of inhibitory connections.
	"""
	
	inodes = int(nodes * inodes_frac)
	enodes = nodes - inodes
	iedges = int(edges * iedges_frac)
	eedges = edges - iedges
	
	return enodes, inodes, eedges, iedges
	

def _compute_edges(nodes, density, edges, avg_deg):
	"""
	Compute the number of edges in the network.
	
	Parameters
    ----------
	nodes: int
		number of nodes in the graph.
	density: double
		Defined as edges / (nodes*nodes).
	edges: int
		Number of connections in the graph.
	avg_deg:
		Average degree of the nodes, defined as edges/nodes.
	
	Returns
    -------
    edges: int
		Number of edges in the graph.
	"""
	
	if edges == -1:
		if avg_deg != -1:
			edges = int(nodes*avg_deg)
		else:
			edges = int(np.square(nodes)*density)
	
	return edges
	

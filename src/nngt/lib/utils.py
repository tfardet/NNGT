#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

from ..core.graph_classes import SpatialGraph, Network, SpatialNetwork
from ..core.graph_datastruct import Shape


__all__ = [
    "make_spatial",
    "make_network",
    "delete_self_loops",
    "delete_parallel_edges",
    "adjacency_matrix"
]


#
#---
# Conversion utils
#------------------------

def make_spatial(graph, shape=Shape(), positions=None):
    if isinstance(graph, Network):
        graph.__class__ = SpatialNetwork
    else:
        graph.__class__ = SpatialGraph
    graph._init_spatial_properties(shape, positions)

def make_network(graph, neural_pop):
    if isinstance(graph, SpatialGraph):
        graph.__class__ = SpatialNetwork
    else:
        graph.__class__ = Network
    graph.population = neural_pop



#
#---
# Adjacency utils
#------------------------

def __del_parallel_edges(graph):
    pass

def __adjacency_matrix(graph):
    pass

try:
    from graph_tool.stats import remove_self_loops as delete_self_loops
    from graph_tool.stats import remove_parallel_edges as delete_parallel_edges
    from graph_tool.spectral import adjacency as adjacency_matrix
except:
    from snap import DelSelfEdges as delete_self_loops
    delete_parallell_edges = __del_parallel_edges
    adjacency_matrix = __adjacency_matrix

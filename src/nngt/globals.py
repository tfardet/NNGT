#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Constant values for NNGT """

import sys
from types import ModuleType

import scipy as sp
import scipy.sparse as ssp



#-----------------------------------------------------------------------------#
# All and version
#------------------------
#

__all__ = [
    "version",
    "glib_data",
    "glib_func",
    "default_neuron",
    "default_synapse",
    "POS",
    "DIST"
    "WEIGHT",
    "DELAY",
    "TYPE",
    "use_library"
]

version = '0.4a'
''' :class:`string`, the current version '''


#-----------------------------------------------------------------------------#
# Graph libraries
#------------------------
#

# name, lib and Graph object
glib_data = {
    "name": "",
    "library": None,
    "graph": object
}

# store the main functions
glib_func = {
    "adjacency": None,
    "assortativity": None,
    "betweenness": None,
    "clustering": None,
    "components": None,
    "diameter": None,
    "reciprocity": None
}

# switch libraries
def use_library(library, reloading=True):
    '''
    Allows the user to switch to a specific graph library.
    
    .. warning:
        If :class:`~nngt.Graph` objects have already been created, they will no
        longer be compatible with NNGT methods.

    Parameters
    ----------
    library : string
        Name of a graph library among 'graph_tool', 'igraph', 'networkx'.
    reloading : bool, optional (default: True)
        Whether the graph objects should be reloaded (this should always be set
        to True except when NNGT is first initiated!)
    '''
        
    if library == "graph_tool":
        # library and graph object
        import graph_tool as glib
        from graph_tool import Graph as GraphLib
        glib_data["name"] = "graph_tool"
        glib_data["library"] = glib
        glib_data["graph"] = GraphLib
        # analysis functions
        from graph_tool.spectral import adjacency as _adj
        from graph_tool.centrality import betweenness
        from graph_tool.correlations import assortativity as assort
        from graph_tool.topology import (edge_reciprocity,
                                        label_components, pseudo_diameter)
        from graph_tool.clustering import global_clustering
        glib_func["assortativity"] = assort
        glib_func["betweenness"] = betweenness
        glib_func["clustering"] = global_clustering
        glib_func["components"] = label_components
        glib_func["diameter"] = pseudo_diameter
        glib_func["reciprocity"] = edge_reciprocity
        # defining the adjacency function
        def adj_mat(graph, weights=None):
            if weights is not None:
                weights = graph.edge_properties[weights]
            return _adj(graph, weights).T
        glib_func["adjacency"] = adj_mat
    elif library == "igraph":
        # library and graph object
        import igraph as glib
        from igraph import Graph as GraphLib
        glib_data["name"] = "igraph"
        glib_data["library"] = glib
        glib_data["graph"] = GraphLib
        # defining the adjacency function
        def adj_mat(graph, weights=None):
            n = graph.node_nb()
            if graph.edge_nb():
                xs, ys = map(sp.array, zip(*graph.get_edgelist()))
                if not graph.is_directed():
                    xs, ys = sp.hstack((xs, ys)).T, sp.hstack((ys, xs)).T
                else:
                    xs, ys = xs.T, ys.T
                data = sp.ones(xs.shape)
                if weights is not None:
                    data = weights
                    if not graph.is_directed():
                        data.extend(data)
                coo_adj = ssp.coo_matrix((data, (xs, ys)))
                return coo_adj.tocsr()
            else:
                return ssp.csr_matrix((n,n))
        glib_func["adjacency"] = adj_mat
    elif library == "networkx":
        # library and graph object
        import networkx as glib
        from networkx import DiGraph as GraphLib
        glib_data["name"] = "networkx"
        glib_data["library"] = glib
        glib_data["graph"] = GraphLib
        # defining the adjacency function
        from networkx import to_scipy_sparse_matrix as adj_mat
        glib_func["adjacency"] = adj_mat
    if reloading:
        reload(sys.modules["nngt"].core.graph_objects)
        reload(sys.modules["nngt"].core)
        reload(sys.modules["nngt"].analysis)
        reload(sys.modules["nngt"].analysis.gt_analysis)
        reload(sys.modules["nngt"].generation)
        reload(sys.modules["nngt"].generation.graph_connectivity)
        reload(sys.modules["nngt"].plot)
        reload(sys.modules["nngt"].lib) #@todo: make price algo and remove this
        reload(sys.modules["nngt"].core.graph_classes)
        from nngt.core.graph_classes import (Graph, SpatialGraph, Network,
                                             SpatialNetwork)
        sys.modules["nngt"].Graph = Graph
        sys.modules["nngt"].SpatialGraph = SpatialGraph
        sys.modules["nngt"].Network = Network
        sys.modules["nngt"].SpatialNetwork = SpatialNetwork

# import the graph libraries the first time
try:
    use_library("graph_tool", False)
except:
    try:
        use_library("igraph", False)
    except:
        try:
            use_library("networkx", False)
        except:
            raise ImportError( "This module needs one of the following graph \
libraries to work:  `graph_tool`, `igraph`, or `networkx`.")


#-----------------------------------------------------------------------------#
# Names
#------------------------
#

POS = "position"
DIST = "distance"
WEIGHT = "weight"
DELAY = "delay"
TYPE = "type"


#-----------------------------------------------------------------------------#
# Basic values
#------------------------
#

#~ default_neuron = "iaf_neuron"
default_neuron = "aeif_cond_alpha"
''' :class:`string`, the default NEST neuron model '''
default_synapse = "static_synapse"
''' :class:`string`, the default NEST synaptic model '''
default_delay = 1.
''' :class:`double`, the default synaptic delay in NEST '''

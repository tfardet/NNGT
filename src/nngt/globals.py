#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Constant values for NNGT """

import sys

import scipy as sp
import scipy.sparse as ssp

import nngt



#-----------------------------------------------------------------------------#
# All and version
#------------------------
#

__all__ = [
    "version",
    "default_neuron",
    "default_synapse",
    "POS",
    "DIST"
    "WEIGHT",
    "DELAY",
    "TYPE",
    "use_library"
]

version = '0.5'
''' :class:`string`, the current version '''


#-----------------------------------------------------------------------------#
# Graph libraries and python 2/3 compatibility
#------------------------
#

# compatible reload

reload_module = None
if sys.hexversion >= 0x03000000 and sys.hexversion < 0x03040000:
    import imp
    reload_module = imp.reload
elif sys.hexversion >= 0x03040000:
    import importlib
    reload_module = importlib.reload
else:
    reload_module = reload

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
    "reciprocity": None,
    "get_edges": None
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
    reload_moduleing : bool, optional (default: True)
        Whether the graph objects should be reload_moduleed (this should always be set
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
        glib_func["scc"] = label_components
        glib_func["wcc"] = label_components
        glib_func["diameter"] = pseudo_diameter
        glib_func["reciprocity"] = edge_reciprocity
        # defining the adjacency function
        def adj_mat(graph, weight=None):
            if weight is not None:
                weight = graph.edge_properties[weight]
            return _adj(graph, weight).T
        def get_edges(graph):
            return graph.edges()
    elif library == "igraph":
        # library and graph object
        import igraph as glib
        from igraph import Graph as GraphLib
        glib_data["name"] = "igraph"
        glib_data["library"] = glib
        glib_data["graph"] = GraphLib
        # functions
        glib_func["assortativity"] = None
        glib_func["nbetweenness"] = None
        glib_func["ebetweenness"] = None
        glib_func["clustering"] = None
        glib_func["scc"] = None
        glib_func["wcc"] = None
        glib_func["diameter"] = None
        glib_func["reciprocity"] = None
        # defining the adjacency function
        def adj_mat(graph, weight=None):
            n = graph.node_nb()
            if graph.edge_nb():
                xs, ys = map(sp.array, zip(*graph.get_edgelist()))
                if not graph.is_directed():
                    xs, ys = sp.hstack((xs, ys)).T, sp.hstack((ys, xs)).T
                else:
                    xs, ys = xs.T, ys.T
                data = sp.ones(xs.shape)
                if issubclass(weight.__class__, str):
                    data *= sp.array(graph.es[weight])
                    if not graph.is_directed():
                        data.extend(data)
                else:
                    data *= sp.array(weight)
                coo_adj = ssp.coo_matrix((data, (xs, ys)), shape=(n,n))
                return coo_adj.tocsr()
            else:
                return ssp.csr_matrix((n,n))
        def get_edges(graph):
            return graph.get_edgelist()
    elif library == "networkx":
        # library and graph object
        import networkx as glib
        from networkx import DiGraph as GraphLib
        glib_data["name"] = "networkx"
        glib_data["library"] = glib
        glib_data["graph"] = GraphLib
        # functions
        from networkx.algorithms import ( diameter, 
            strongly_connected_components, weakly_connected_components,
            degree_assortativity_coefficient )
        def overall_reciprocity(g):
            num_edges = g.number_of_edges()
            num_recip = (num_edges - g.to_undirected().number_of_edges()) * 2
            if n_all_edge == 0:
                raise ArgumentError("Not defined for empty graphs")
            else:
                return num_recip/float(num_edges)
        nx_version = glib.__version__
        try:
            from networkx.algorithms import overall_reciprocity
            glib_func["reciprocity"] = overall_reciprocity
        except:
            def mock_recip(*args, **kwargs):
                return NotImplementedError("Not implemented for networkx {}; \
try to install latest version.".format(nx_version))
            glib_func["reciprocity"] = mock_recip
        glib_func["assortativity"] = degree_assortativity_coefficient
        glib_func["diameter"] = diameter
        glib_func["scc"] = strongly_connected_components
        glib_func["wcc"] = diameter
        # defining the adjacency function
        from networkx import to_scipy_sparse_matrix
        def adj_mat(graph, weight=None):
            return to_scipy_sparse_matrix(graph, weight=weight)
        def get_edges(graph):
            return graph.edges_iter(data=False)
    glib_func["adjacency"] = adj_mat
    glib_func["get_edges"] = get_edges
    if reloading:
        reload_module(sys.modules["nngt"].core.graph_objects)
        reload_module(sys.modules["nngt"].core)
        reload_module(sys.modules["nngt"].analysis)
        reload_module(sys.modules["nngt"].analysis.gt_analysis)
        reload_module(sys.modules["nngt"].generation)
        reload_module(sys.modules["nngt"].generation.graph_connectivity)
        if nngt._with_plot:
            reload_module(sys.modules["nngt"].plot)
        if nngt._with_nest:
            reload_module(sys.modules["nngt"].simulation)
        reload_module(sys.modules["nngt"].lib) #@todo: make price algo and remove this
        reload_module(sys.modules["nngt"].core.graph_classes)
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
BWEIGHT = "bweight"
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

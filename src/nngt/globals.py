#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Constant values for NNGT """

cst__all__ = ["version",
                "default_neuron",
                "default_synapse"]

version = '0.3'
''' :class:`string`, the current version '''


#-----------------------------------------------------------------------------#
# Graph libraries
#------------------------
#

s_glib, glib, GraphLib = "", None, object
adjacency = None
try:
    import graph_tool as glib
    from graph_tool import Graph as GraphLib
    from graph_tool.spectral import adjacency as adjacency
    from graph_tool.centrality import betweenness
    from graph_tool.correlations import assortativity as assort
    from graph_tool.topology import (edge_reciprocity, label_components,
                                     pseudo_diameter)
    from graph_tool.clustering import global_clustering
    s_glib = "graph_tool"
except:
    try:
        import igraph as glib
        from igraph import Graph as GraphLib
        s_glib = "igraph"
        def adj(graph, weights=None):
            xs, ys = map(array, zip(*graph.get_edgelist()))
            if not graph.is_directed():
                xs, ys = hstack((xs, ys)).T, hstack((ys, xs)).T
            else:
                xs, ys = xs.T, ys.T
            data = ones(xs.shape)
            if weights is not None:
                data = weights
                if not graph.is_directed():
                    data.extend(data)
            coo_adj = ssp.coo_matrix((data, (xs, ys)))
            return coo_adj.tocsr()
        adjacency = adj
    except:
        try:
            import networkx as glib
            from networkx import DiGraph as GraphLib
            from networkx import to_scipy_sparse_matrix as adjacency
            s_glib = "networkx"
        except:
            try:
                import snap as glib
                from snap import TNEANet as GraphLib
                s_glib = "snap"
            except:
                pass

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

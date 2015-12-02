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

#------------#
# graph_tool #
#------------#
try:
    # library and graph object
    s_glib = "graph_tool"
    import graph_tool as glib
    from graph_tool import Graph as GraphLib
    # analysis functions
    from graph_tool.spectral import adjacency as _adj
    from graph_tool.centrality import betweenness
    from graph_tool.correlations import assortativity as assort
    from graph_tool.topology import (edge_reciprocity, label_components,
                                     pseudo_diameter)
    from graph_tool.clustering import global_clustering
    # defining the adjacency function
    def adj_mat(graph, weights=None):
        if weights is not None:
            weights = graph.new_edge_property("double", weights)
        return _adj(graph, weights)
    adjacency = adj_mat
except:
    #--------#
    # igraph #
    #--------#
    try:
        # library and graph object
        s_glib = "igraph"
        import igraph as glib
        from igraph import Graph as GraphLib
        # defining the adjacency function
        def adj_mat(graph, weights=None):
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
        adjacency = adj_mat
    except:
        #----------#
        # networkx #
        #----------#
        try:
            # library and graph object
            s_glib = "networkx"
            import networkx as glib
            from networkx import DiGraph as GraphLib
            # defining the adjacency function
            from networkx import to_scipy_sparse_matrix as _adj
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

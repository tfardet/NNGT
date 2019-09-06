#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Tools to interact with the graph libraries backends """

import logging
import sys

import numpy as np
import scipy.sparse as ssp

import nngt
from .errors import not_implemented
from .logger import _log_message
from .reloading import reload_module
from .test_functions import nonstring_container, mpi_barrier


logger = logging.getLogger(__name__)


# ------------------- #
# Graph library usage #
# ------------------- #

analyze_graph = {
    'adjacency': not_implemented,
    'assortativity': not_implemented,
    'betweenness': not_implemented,
    'clustering': not_implemented,
    'diameter': not_implemented,
    'ebetweenness': not_implemented,
    'get_edges': not_implemented,
    'nbetweenness': not_implemented,
    'reciprocity': not_implemented,
    'scc': not_implemented,
    'wcc': not_implemented,
}


# use library function

@mpi_barrier
def use_backend(backend, reloading=True, silent=False):
    '''
    Allows the user to switch to a specific graph library as backend.

    .. warning ::
        If :class:`~nngt.Graph` objects have already been created, they will no
        longer be compatible with NNGT methods.

    Parameters
    ----------
    backend : string
        Name of a graph library among 'graph_tool', 'igraph', 'networkx', or
        'nngt'.
    reloading : bool, optional (default: True)
        Whether the graph objects should be reloaded through `reload_module`
        (this should always be set to True except when NNGT is first initiated!)
    silent : bool, optional (default: False)
        Whether the changes made to the configuration should be logged at the
        DEBUG (True) or INFO (False) level.
    '''
    # save old config except for graph-library data
    old_config = nngt.get_config(detailed=True)
    for k in ("graph", "backend", "library"):
        del old_config[k]
    # try to switch graph library
    success = False
    error = None
    if backend == "graph-tool":
        try:
            success = _set_graph_tool()
        except Exception as e:
            error = e
    elif backend == "igraph":
        try:
            success = _set_igraph()
        except Exception as e:
            error = e
    elif backend == "networkx":
        try:
            success = _set_networkx()
        except Exception as e:
            error = e
    elif backend == "nngt":
        try:
            success = _set_nngt()
        except Exception as e:
            error = e
    else:
        raise ValueError("Invalid graph library requested.")
    if reloading:
        reload_module(sys.modules["nngt"].core.base_graph)
        reload_module(sys.modules["nngt"].core.gt_graph)
        reload_module(sys.modules["nngt"].core.ig_graph)
        reload_module(sys.modules["nngt"].core.nx_graph)
        reload_module(sys.modules["nngt"].core)
        reload_module(sys.modules["nngt"].generation)
        reload_module(sys.modules["nngt"].generation.graph_connectivity)
        if nngt._config['with_plot']:
            reload_module(sys.modules["nngt"].plot)
        if nngt._config['with_nest']:
            reload_module(sys.modules["nngt"].simulation)
        reload_module(sys.modules["nngt"].lib)
        reload_module(sys.modules["nngt"].core.graph_classes)
        reload_module(sys.modules["nngt"].analysis)
        from nngt.core.graph_classes import (Graph, SpatialGraph, Network,
                                             SpatialNetwork)
        sys.modules["nngt"].Graph = Graph
        sys.modules["nngt"].SpatialGraph = SpatialGraph
        sys.modules["nngt"].Network = Network
        sys.modules["nngt"].SpatialNetwork = SpatialNetwork
    # restore old config
    nngt.set_config(old_config, silent=True)
    # log
    if success:
        if silent:
            _log_message(logger, "DEBUG",
                         "Successfuly switched to " + backend + ".")
        else:
            _log_message(logger, "INFO",
                         "Successfuly switched to " + backend + ".")
    else:
        if silent:
            _log_message(logger, "DEBUG",
                         "Error, could not switch to " + backend + ": "
                         "{}.".format(error))
        else:
            _log_message(logger, "WARNING",
                         "Error, could not switch to " + backend + ": "
                         "{}.".format(error))
        if error is not None:
            raise error


# ----------------- #
# Loading functions #
# ----------------- #

def _set_graph_tool():
    '''
    Set graph-tool as graph library, store relevant items in config and
    analyze graph dictionaries.
    '''
    import graph_tool as glib
    from graph_tool import Graph as GraphLib
    nngt._config["backend"] = "graph-tool"
    nngt._config["library"] = glib
    nngt._config["graph"]   = GraphLib
    # analysis functions
    from graph_tool.spectral import adjacency as _adj
    from graph_tool.centrality import betweenness, closeness
    from graph_tool.correlations import scalar_assortativity as assort
    from graph_tool.topology import (edge_reciprocity,
                                    label_components, pseudo_diameter)
    from graph_tool.clustering import global_clustering, local_clustering
    def global_clustering_coeff(g):
        return global_clustering(g)[0]
    def local_clustering_coeff(g, nodes=None):
        lc = local_clustering(g).a
        if nodes is None:
            return lc
        return lc[nodes]
    def _closeness(graph, nodes, weights):
        if weights is True and graph.is_weighted():
            weights = graph.edge_properties[weight]
        else:
            weights=None
        c = closeness(graph, weight=weights)
        if nodes is None:
            return c.a
        return c.a[nodes]
    # defining the adjacency function
    def adj_mat(graph, weight=None):
        if weight not in (None, False):
            weight = graph.edge_properties[weight]
            return _adj(graph, weight).T
        return _adj(graph).T
    def get_edges(graph):
        return graph.edges()
    # store the functions
    nngt.analyze_graph["assortativity"] = assort
    nngt.analyze_graph["betweenness"] = betweenness
    nngt.analyze_graph["closeness"] = _closeness
    nngt.analyze_graph["clustering"] = global_clustering_coeff
    nngt.analyze_graph["local_clustering"] = local_clustering_coeff
    nngt.analyze_graph["scc"] = label_components
    nngt.analyze_graph["wcc"] = label_components
    nngt.analyze_graph["diameter"] = pseudo_diameter
    nngt.analyze_graph["reciprocity"] = edge_reciprocity
    nngt.analyze_graph["adjacency"] = adj_mat
    nngt.analyze_graph["get_edges"] = get_edges
    return True


def _set_igraph():
    '''
    Set igraph as graph library, store relevant items in config and
    analyze graph dictionaries.
    '''
    import igraph as glib
    from igraph import Graph as GraphLib
    nngt._config["backend"] = "igraph"
    nngt._config["library"] = glib
    nngt._config["graph"]   = GraphLib
    # define
    def _closeness(graph, nodes, weights):
        if weights is True and graph.is_weighted():
            weights = weight
        else:
            weights=None
        return graph.closeness(nodes, mode="out", weights=weights)
    # defining the adjacency function
    def adj_mat(graph, weight=None):
        n = graph.node_nb()
        if graph.edge_nb():
            xs, ys = map(np.array, zip(*graph.get_edgelist()))
            xs, ys = xs.T, ys.T
            data = np.ones(xs.shape)
            if weight in graph.edges_attributes:
                data *= np.array(graph.es[weight])
            elif nonstring_container(weight):
                data *= np.array(weight)
            elif weight:
                data *= np.array(graph.es["weight"])
            coo_adj = ssp.coo_matrix((data, (xs, ys)), shape=(n,n))
            return coo_adj.tocsr()
        else:
            return ssp.csr_matrix((n,n))
    def get_edges(graph):
        return graph.get_edgelist()
    # store functions
    nngt.analyze_graph["assortativity"] = not_implemented
    nngt.analyze_graph["nbetweenness"] = not_implemented
    nngt.analyze_graph["ebetweenness"] = not_implemented
    nngt.analyze_graph["clustering"] = not_implemented
    nngt.analyze_graph["closeness"] = _closeness
    nngt.analyze_graph["local_clustering"] = not_implemented
    nngt.analyze_graph["scc"] = not_implemented
    nngt.analyze_graph["wcc"] = not_implemented
    nngt.analyze_graph["diameter"] = not_implemented
    nngt.analyze_graph["reciprocity"] = not_implemented
    nngt.analyze_graph["adjacency"] = adj_mat
    nngt.analyze_graph["get_edges"] = get_edges
    return True


def _set_networkx():
    import networkx as glib
    if glib.__version__[0] < '2':
        raise ImportError("`networkx {} is ".format(glib.__version__) +\
                          "installed while version 2+ is required.")
    from networkx import DiGraph as GraphLib
    nngt._config["backend"] = "networkx"
    nngt._config["library"] = glib
    nngt._config["graph"]   = GraphLib
    # analysis functions
    from networkx.algorithms import ( diameter,
        strongly_connected_components, weakly_connected_components,
        degree_assortativity_coefficient )
    def _closeness(graph, nodes, weights):
        if weights is True and graph.is_weighted():
            weights = graph.edge_properties[weight]
        else:
            weights=None
        if nodes is None:
            return glib.closeness_centrality(graph, distance=weights)
        else:
            c = [glib.closeness_centrality(graph, u=n, distance=weights)
                 for n in nodes]
            return c
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
    except ImportError:
        def overall_reciprocity(*args, **kwargs):
            return NotImplementedError("Not implemented for networkx " +
                                       str(nx_version) + "; try installing "
                                       "the latest version.")
    def local_clustering(g, nodes=None):
        return np.array(glib.clustering(g, nodes).values())
    # defining the adjacency function
    from networkx import to_scipy_sparse_matrix
    def adj_mat(graph, weight=None):
        return to_scipy_sparse_matrix(graph, weight=weight)
    def get_edges(graph):
        return graph.edges(data=False)
    # store functions
    nngt.analyze_graph["assortativity"] = degree_assortativity_coefficient
    nngt.analyze_graph["diameter"] = diameter
    nngt.analyze_graph["closeness"] = _closeness
    nngt.analyze_graph["clustering"] = glib.transitivity
    nngt.analyze_graph["local_clustering"] = local_clustering
    nngt.analyze_graph["reciprocity"] = overall_reciprocity
    nngt.analyze_graph["scc"] = strongly_connected_components
    nngt.analyze_graph["wcc"] = diameter
    nngt.analyze_graph["adjacency"] = adj_mat
    nngt.analyze_graph["get_edges"] = get_edges
    return True


def _set_nngt():
    nngt._config["backend"] = "nngt"
    nngt._config["library"] = nngt
    nngt._config["graph"]   = object
    # analysis functions
    def _notimplemented(*args, **kwargs):
        raise NotImplementedError("Install a graph library to use.")
    def adj_mat(graph, weight=None):
        if weight in graph.edges_attributes and weight != "weight":
            edges     = graph.edges_array
            prop      = graph.get_edge_attributes(name=weight)
            num_nodes = graph.node_nb()
            mat       = ssp.coo_matrix((prop, (edges[:, 0], edges[:, 1])),
                                       shape=(num_nodes, num_nodes))
            return mat.tocsr()
        elif weight is False:
            mat = graph._adj_mat.tocsr()
            mat.data = np.ones(len(mat.data))
            return mat
        else:
            return graph._adj_mat.tocsr()
    def get_edges(graph):
        return graph.edges_array()
    # store functions
    nngt.analyze_graph["assortativity"] = _notimplemented
    nngt.analyze_graph["diameter"] = _notimplemented
    nngt.analyze_graph["closeness"] = _notimplemented
    nngt.analyze_graph["clustering"] = _notimplemented
    nngt.analyze_graph["local_clustering"] = _notimplemented
    nngt.analyze_graph["reciprocity"] = _notimplemented
    nngt.analyze_graph["scc"] = _notimplemented
    nngt.analyze_graph["wcc"] = _notimplemented
    nngt.analyze_graph["adjacency"] = adj_mat
    nngt.analyze_graph["get_edges"] = get_edges
    return True

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

from importlib import reload
import logging
import sys

import numpy as np
import scipy.sparse as ssp

import nngt
from .errors import not_implemented
from .logger import _log_message
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
        Whether the graph objects should be reloaded through `reload`
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
        reload(sys.modules["nngt"].core.graph_interface)
        reload(sys.modules["nngt"].core.nngt_graph)
        reload(sys.modules["nngt"].core.gt_graph)
        reload(sys.modules["nngt"].core.ig_graph)
        reload(sys.modules["nngt"].core.nx_graph)
        reload(sys.modules["nngt"].core)
        reload(sys.modules["nngt"].generation)
        reload(sys.modules["nngt"].generation.graph_connectivity)
        if nngt._config['with_plot']:
            reload(sys.modules["nngt"].plot)
        if nngt._config['with_nest']:
            reload(sys.modules["nngt"].simulation)
        reload(sys.modules["nngt"].lib)
        reload(sys.modules["nngt"].core.graph)
        reload(sys.modules["nngt"].core.spatial_graph)
        reload(sys.modules["nngt"].core.networks)
        reload(sys.modules["nngt"].analysis)

        from nngt.core.graph import Graph
        from nngt.core.spatial_graph import SpatialGraph
        from nngt.core.networks import Network, SpatialNetwork

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

    # store the functions
    from ..analysis import gt_functions

    _store_functions(nngt.analyze_graph, gt_functions)

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

    # store the functions
    from ..analysis import gt_functions

    _store_functions(nngt.analyze_graph, gt_functions)

    return True


def _set_networkx():
    import networkx as glib
    if glib.__version__ < '2.4':
        raise ImportError("`networkx {} is ".format(glib.__version__) +\
                          "installed while version >= 2.4 is required.")
    from networkx import DiGraph as GraphLib
    nngt._config["backend"] = "networkx"
    nngt._config["library"] = glib
    nngt._config["graph"]   = GraphLib
    # analysis functions
    from networkx.algorithms import ( diameter,
        strongly_connected_components, weakly_connected_components,
        degree_assortativity_coefficient )

    def diam(g):
        return diameter(g.graph)

    def scc(g):
        return strongly_connected_components(g.graph)

    def wcc(g):
        return weakly_connected_components(g.graph)

    def deg_assort_coef(g):
        return degree_assortativity_coefficient(g.graph)

    def clustering(g):
        return glib.transitivity(g.graph)

    def _closeness(g, nodes, weights):
        if weights is True and g.is_weighted():
            weights = g.edge_properties[weight]
        else:
            weights=None
        if nodes is None:
            return glib.closeness_centrality(g.graph, distance=weights)
        else:
            c = [glib.closeness_centrality(g.graph, u=n, distance=weights)
                 for n in nodes]
            return c

    def overall_reciprocity(g):
        num_edges = g.graph.number_of_edges()
        num_recip = (num_edges - g.graph.to_undirected().number_of_edges()) * 2
        if n_all_edge == 0:
            raise ArgumentError("Not defined for empty graphs")
        else:
            return num_recip/float(num_edges)

    nx_version = glib.__version__

    from networkx.algorithms import overall_reciprocity

    def overall_recip(g):
        return overall_reciprocity(g.graph)

    def local_clustering(g, nodes=None):
        return np.array(glib.clustering(g.graph, nodes).values())

    # defining the adjacency function
    from networkx import to_scipy_sparse_matrix
    def adj_mat(g, weight=None):
        return to_scipy_sparse_matrix(g.graph, weight=weight)
    def get_edges(g):
        return g.graph.edges(data=False)
    # store functions
    nngt.analyze_graph["assortativity"] = deg_assort_coef
    nngt.analyze_graph["diameter"] = diam
    nngt.analyze_graph["closeness"] = _closeness
    nngt.analyze_graph["clustering"] = clustering
    nngt.analyze_graph["local_clustering"] = local_clustering
    nngt.analyze_graph["reciprocity"] = overall_recip
    nngt.analyze_graph["scc"] = scc
    nngt.analyze_graph["wcc"] = wcc
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
        data = None

        if weight in graph.edges_attributes:
            data = graph.get_edge_attributes(name=weight)
        else:
            data = np.ones(graph.edge_nb())
            
        edges     = graph.edges_array
        num_nodes = graph.node_nb()
        mat       = ssp.coo_matrix((data, (edges[:, 0], edges[:, 1])),
                                   shape=(num_nodes, num_nodes))

        return mat.tocsr()

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


def _store_functions(analysis_dict, module):
    ''' Store functions from module '''
    analysis_dict["assortativity"] = module.assortativity
    analysis_dict["betweenness"] = module.betwenness
    analysis_dict["closeness"] = module.closeness
    analysis_dict["clustering"] = module.global_clustering
    analysis_dict["local_clustering"] = module.local_clustering
    analysis_dict["scc"] = module.connected_components
    analysis_dict["wcc"] = module.connected_components
    analysis_dict["diameter"] = module.diameter
    analysis_dict["reciprocity"] = module.reciprocity
    analysis_dict["adjacency"] = module.adj_mat
    analysis_dict["get_edges"] = module.get_edges

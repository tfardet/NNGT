#-*- coding:utf-8 -*-
#
# nx_functions.py
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

""" Tools to analyze graphs with the networkx backend """

import numpy as np

from ..lib.test_functions import nonstring_container

import networkx as nx
from networkx.algorithms import diameter as nx_diam
from networkx.algorithms import (strongly_connected_components,
                                 weakly_connected_components,
                                 degree_assortativity_coefficient)


def global_clustering(g, weights=None):
    '''
    Returns the global clustering coefficient.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.

    Note
    ----
    If `weights` is None, returns the transitivity (number of existing
    triangles over total number of possible triangles); otherwise returns
    the average clustering.

    References
    ----------
    .. [nx-global-clustering] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.transitivity.html
    .. [nx-average-clustering] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html
    '''
    w = _get_weights(g, weights)

    if w is None:
        return nx.transitivity(g.graph)

    return np.average(nx.clustering(g.graph, weight=weights))


def local_clustering(g, weights=None, nodes=None):
    '''
    Returns the local clustering coefficient of some `nodes`.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    nodes : list, optional (default: all nodes)
        The list of nodes for which the clutering will be returned

    Returns
    -------
    lc : :class:`numpy.ndarray`
        The list of clustering coefficients, on per node.

    References
    ----------
    .. [nx-local-clustering] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html
    '''
    ww = _get_weights(g, weights)

    lc = nx.clustering(nx.to_undirected(g.graph), nodes=nodes, weight=weights)
    lc = np.array([lc[i] for i in range(g.node_nb())], dtype=float)

    return lc


def assortativity(g, degree, weights=None):
    '''
    Returns the assortativity of the graph.
    This tells whether nodes are preferentially connected together depending
    on their degree.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    degree : str
        The type of degree that should be considered.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.

    References
    ----------
    .. [nx-assortativity] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.degree_assortativity_coefficient.html
    '''
    w = _get_weights(g, weights)

    return degree_assortativity_coefficient(g.graph, x=degree, y=degree,
                                            weight=w)


def reciprocity(g):
    '''
    Calculate the edge reciprocity of the graph.

    @todo: check whether we can get this for single nodes for all libraries.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.

    References
    ----------
    .. [nx-reciprocity] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.reciprocity.overall_reciprocity.html
    '''
    return nx.overall_reciprocity(g.graph)


def closeness(g, weighted=False, nodes=None):
    '''
    Returns the closeness centrality of some `nodes`.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    nodes : list, optional (default: all nodes)
        The list of nodes for which the clutering will be returned

    Returns
    -------
    c : :class:`numpy.ndarray`
        The list of closeness centralities, on per node.

    Note
    ----
    When requesting a subset of nodes, check whether it is faster to use
    `nodes`, or to compute all closeness centralities, then take the subset.

    References
    ----------
    .. [nx-closeness] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
    '''
    w = _get_weights(g, weights)

    if nodes is None:
        return nx.closeness_centrality(g.graph, distance=w)

    c = [nx.closeness_centrality(g.graph, u=n, distance=w)
         for n in nodes]

    return c


def connected_components(g, ctype=None):
    '''
    Returns the connected component to which each node belongs.

    @todo: check if the components are labeled from 0.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    ctype : str, optional (default 'scc')
        Type of component that will be searched: either strongly connected
        ('scc', by default) or weakly connected ('wcc').

    Returns
    -------
    cc, hist : :class:`numpy.ndarray`
        The component associated to each node (`cc`) and the number of nodes in
        each of the component (`hist`).

    References
    ----------
    .. [nx-scc] https://graph-tool.skewed.de/static/doc/topology.html#graph_tool.topology.label_components
    .. [nx-wcc]
    '''
    res = None

    if ctype == "scc":
        res = strongly_connected_components(g.graph)
    elif ctype == "wcc":
        res = weakly_connected_components(g.graph)
    else:
        raise ValueError("Invalid `ctype`, only 'scc' and 'wcc' are allowed.")

    cc   = np.zeros(g.node_nb(), dtype=int)
    hist = []

    for i, nodes in enumerate(res):
        cc[nodes] = i

        hist.append(len(nodes))

    return cc, np.array(hist, dtype=int)


def diameter(g, weights=False):
    '''
    Returns the pseudo-diameter of the graph.

    @todo: check what happens if some nodes are disconnected (with and
    without weights)

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.

    References
    ----------
    .. [nx-diameter] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.distance_measures.diameter.html
    '''
    w = _get_weights(g, weights)

    if w is not None:
        raise NotImplementedError("Weighted diameter is not available for "
                                  "networkx backend.")

    return nx.diameter(g.graph)[0]


def adj_mat(g, weights=None):
    r'''
    Returns the adjacency matrix :math:`A` of the graph.
    With edge :math:`i \leftarrow j` corresponding to entry :math:`A_{ij}`.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then returns the binary adjacency matrix; if ``True``, returns the
        weighted matrix, otherwise fills the matrix with any valid edge
        attribute values.

    Returns
    -------
    The adjacency matrix as a :class:`scipy.sparse.csr_matrix`.

    References
    ----------
    .. [gt-adjacency] https://graph-tool.skewed.de/static/doc/spectral.html#graph_tool.spectral.adjacency
    '''
    w = _get_weights(g, weights)

    return nx.to_scipy_sparse_matrix(g.graph, weight=w)


def get_edges(g):
    '''
    Returns the edges in the graph by order of creation.
    '''
    return g.edges_array


def _get_weights(g, weights):
    if weights in g.edges_attributes:
        # existing edge attribute
        return weights
    elif nonstring_container(weights):
        # user-provided array
        return ValueError("networkx backend does not support custom arrays "
                          "as `weights`.")
    elif weights is True:
        # "normal" weights
        return 'weight'
    elif not weights:
        # unweighted
        return None

    raise ValueError("Unknown attribute '{}' for `weights`.".format(weights))
    

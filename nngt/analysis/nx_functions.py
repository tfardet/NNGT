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


def global_clustering(g, weights=None):
    '''
    Returns the undirected global clustering coefficient.
    This corresponds to the ratio of undirected triangles to the number of
    undirected triads.

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
    '''
    w = _get_weights(g, weights)

    if w is None:
        return nx.transitivity(g.graph.to_undirected(as_view=True))

    raise NotImplementedError("Weighted global clustering is not implemented "
                              "for networkx backend.")


def undirected_local_clustering(g, weights=None, nodes=None,
                                combine_weights="sum"):
    '''
    Returns the undirected local clustering coefficient of some `nodes`.

    If `g` is directed, then it is converted to a simple undirected graph
    (no parallel edges).

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
    combine_weights : str, optional (default: "sum")
        How the weights of directed edges between two nodes should be combined,
        among:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "product": the product of the edge attribute values will be used for
          the new edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "median": the median of the edge attribute values will be used for
          the new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge. 

    Returns
    -------
    lc : :class:`numpy.ndarray`
        The list of clustering coefficients, on per node.

    References
    ----------
    .. [nx-local-clustering] :nxdoc:`algorithms.cluster.clustering`
    '''
    ww = _get_weights(g, weights)

    if g.is_directed() and ww is not None:
        raise NotImplementedError("networkx backend currently does not "
                                  "provide weighted clustering for directed "
                                  "graphs.")

    lc = nx.clustering(g.graph.to_undirected(as_view=True), nodes=nodes,
                       weight=weights)
       
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
    .. [nx-assortativity]
       :nxdoc:`algorithms.assortativity.degree_assortativity_coefficient`
    '''
    if weights is not None:
        raise NotImplementedError("Weighted assortatibity is not yet "
                                  "implemented for networkx backend.")

    w = _get_weights(g, weights)

    return nx.degree_assortativity_coefficient(g.graph, x=degree, y=degree,
                                               weight=w)


def reciprocity(g):
    '''
    Calculate the edge reciprocity of the graph.

    The reciprocity is defined as the number of edges that have a reciprocal
    edge (an edge between the same nodes but in the opposite direction)
    divided by the total number of edges.
    This is also the probability for any given edge, that its reciprocal edge
    exists.
    By definition, the reciprocity of undirected graphs is 1.

    @todo: check whether we can get this for single nodes for all libraries.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.

    References
    ----------
    .. [nx-reciprocity] :nxdoc:`algorithms.reciprocity.overall_reciprocity`
    '''
    if not g.is_directed():
        return 1.

    return nx.overall_reciprocity(g.graph)


def closeness(g, weights=None, nodes=None, mode="out", harmonic=False,
              default=np.NaN):
    '''
    Returns the closeness centrality of some `nodes`.

    Closeness centrality of a node `u` is defined, for the harmonic version,
    as the sum of the reciprocal of the shortest path distance :math:`d_{uv}`
    from `u` to the N - 1 other nodes in the graph (if `mode` is "out",
    reciprocally :math:`d_{vu}`, the distance to `u` from another node v,
    if `mode` is "in"):

    .. math::

        C(u) = \frac{1}{N - 1} \sum_{v \neq u} \frac{1}{d_{uv}},

    or, using the arithmetic definition, as the reciprocal of the
    average shortest path distance to/from `u` over to all other nodes:

    .. math::

        C(u) = \frac{n - 1}{\sum_{v \neq u} d_{uv}},

    where `d_{uv}` is the shortest-path distance from `u` to `v`,
    and `n` is the number of nodes in the component.

    By definition, the distance is infinite when nodes are not connected by
    a path in the harmonic case (such that :math:`\frac{1}{d(v, u)} = 0`),
    while the distance itself is taken as zero for unconnected nodes in the
    first equation.

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
    mode : str, optional (default: "out")
        For directed graphs, whether the distances are computed from ("out") or
        to ("in") each of the nodes.
    harmonic : bool, optional (default: False)
        Whether the arithmetic (default) or the harmonic (recommended) version
        of the closeness should be used.

    Returns
    -------
    c : :class:`numpy.ndarray`
        The list of closeness centralities, on per node.

    .. warning ::
        For compatibility reasons (harmonic closeness is not implemented for
        igraph), the arithmetic version is used by default; however, it is
        recommended to use the harmonic version instead whenever possible.

    References
    ----------
    .. [nx-harmonic] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.harmonic_centrality.html
    .. [nx-closeness] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
    '''
    w = _get_weights(g, weights)

    graph = g.graph

    if graph.is_directed() and mode == "out":
        graph = g.graph.reverse(copy=False)

    c = None

    if harmonic:
        c = nx.harmonic_centrality(graph, distance=w)
    else:
        c = nx.closeness_centrality(graph, distance=w, wf_improved=False)

    c = np.array([v for _, v in c.items()])

    # normalize
    if harmonic:
        c *= 1 / (len(graph) - 1)
    elif default != 0:
        c[c == 0.] = default

    if nodes is None:
        return c

    return c[nodes]


def betweenness(g, btype="both", weights=None):
    '''
    Returns the normalized betweenness centrality of the nodes and edges.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    btype : str, optional (default 'both')
        The centrality that should be returned (either 'node', 'edge', or
        'both'). By default, both betweenness centralities are computed.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.

    Returns
    -------
    nb : :class:`numpy.ndarray`
        The nodes' betweenness if `btype` is 'node' or 'both'
    eb : :class:`numpy.ndarray`
        The edges' betweenness if `btype` is 'edge' or 'both'

    References
    ----------
    .. [nx-ebetw] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html#networkx.algorithms.centrality.edge_betweenness_centrality
    .. [nx-nbetw] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html#networkx.algorithms.centrality.betweenness_centrality
    '''
    w = _get_weights(g, weights)

    nb, eb = None, None

    if btype in ("both", "node"):
        di_nb = nx.betweenness_centrality(g.graph, weight=w)
        nb    = np.array([di_nb[i] for i in g.get_nodes()])

    if btype in ("both", "edge"):
        di_eb = nx.edge_betweenness_centrality(g.graph, weight=w)
        eb    = np.array([di_eb[tuple(e)] for e in g.edges_array])

    if btype == "node":
        return nb
    elif btype == "edge":
        return eb

    return nb, eb


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
    .. [nx-scc] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.strongly_connected_components.html#networkx.algorithms.components.strongly_connected_components
    .. [nx-wcc] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.weakly_connected_components.html#networkx.algorithms.components.weakly_connected_components
    '''
    ctype = "scc" if ctype is None else ctype
    res   = None

    if not g.is_directed():
        res = nx.connected_components(g.graph)
    elif ctype == "scc":
        res = nx.strongly_connected_components(g.graph)
    elif ctype == "wcc":
        res = nx.weakly_connected_components(g.graph)
    else:
        raise ValueError("Invalid `ctype`, only 'scc' and 'wcc' are allowed.")

    cc   = np.zeros(g.node_nb(), dtype=int)
    hist = []

    for i, nodes in enumerate(res):
        cc[list(nodes)] = i

        hist.append(len(nodes))

    return cc, np.array(hist, dtype=int)


def diameter(g, weights=False):
    '''
    Returns the diameter of the graph.

    It returns infinity if the graph is not connected (strongly connected for
    directed graphs).

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

    num_nodes = g.node_nb()

    if w is not None:
        res = []

        for _, (d, _) in nx.all_pairs_dijkstra(g.graph, weight=w):
            if len(d) < num_nodes:
                return np.inf

            res.extend(d.values())

        return np.max(res)

    try:
        return nx.diameter(g.graph)
    except nx.exception.NetworkXError:
        return np.inf


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
    

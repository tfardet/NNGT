#-*- coding:utf-8 -*-
#
# analysis/nx_functions.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Tools to analyze graphs with the networkx backend """

import numpy as np
import scipy.sparse as ssp

from ..lib.test_functions import nonstring_container, is_integer
from ..lib.graph_helpers import _get_nx_weights, _get_nx_graph

import networkx as nx


def global_clustering_binary_undirected(g):
    '''
    Returns the undirected global clustering coefficient.

    This corresponds to the ratio of undirected triangles to the number of
    undirected triads.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.

    References
    ----------
    .. [nx-global-clustering] :nxdoc:`algorithms.cluster.transitivity`
    '''
    return nx.transitivity(g.graph.to_undirected(as_view=True))


def local_clustering_binary_undirected(g, nodes=None):
    '''
    Returns the undirected local clustering coefficient of some `nodes`.

    If `g` is directed, then it is converted to a simple undirected graph
    (no parallel edges).

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    nodes : list, optional (default: all nodes)
        The list of nodes for which the clustering will be returned

    Returns
    -------
    lc : :class:`numpy.ndarray`
        The list of clustering coefficients, on per node.

    References
    ----------
    .. [nx-local-clustering] :nxdoc:`algorithms.cluster.clustering`
    '''
    num_nodes = g.node_nb()

    if nonstring_container(nodes):
        num_nodes = len(nodes)
    elif nodes is not None:
        num_nodes = 1

    lc = nx.clustering(g.graph.to_undirected(as_view=True), nodes=nodes,
                       weight=None)

    if num_nodes == 1:
        return lc

    if nodes is None:
        nodes = list(range(num_nodes))

    return np.array([lc[n] for n in nodes], dtype=float)


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
    w = _get_nx_weights(g, weights)

    return nx.degree_pearson_correlation_coefficient(
        g.graph, x=degree, y=degree, weight=w)


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


def closeness(g, weights=None, nodes=None, mode="out", harmonic=True,
              default=np.NaN):
    r'''
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
    harmonic : bool, optional (default: True)
        Whether the arithmetic or the harmonic (recommended) version
        of the closeness should be used.

    Returns
    -------
    c : :class:`numpy.ndarray`
        The list of closeness centralities, on per node.

    References
    ----------
    .. [nx-harmonic] :nxdoc:`algorithms.centrality.harmonic_centrality`
    .. [nx-closeness] :nxdoc:`algorithms.centrality.closeness_centrality`
    '''
    w = _get_nx_weights(g, weights)

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
    .. [nx-ebetw] :nxdoc:`algorithms.centrality.edge_betweenness_centrality`
    .. [nx-nbetw] :nxdoc:`networkx.algorithms.centrality.betweenness_centrality`
    '''
    w = _get_nx_weights(g, weights)

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
    .. [nx-ucc] :nxdoc:`algorithms.components.connected_components`
    .. [nx-scc] :nxdoc:`algorithms.components.strongly_connected_components`
    .. [nx-wcc] :nxdoc:`algorithms.components.weakly_connected_components`
    '''
    if ctype is None:
        ctype = "scc" if g.is_directed() else "wcc"

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


def shortest_path(g, source, target, directed=None, weights=None,
                  combine_weights="mean"):
    '''
    Returns a shortest path between `source`and `target`.
    The algorithms returns an empty list if there is no path between the nodes.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    source : int
        Node from which the path starts.
    target : int
        Node where the path ends.
    directed : bool, optional (default: ``g.is_directed()``)
        Whether the edges should be considered as directed or not
        (automatically set to False if `g` is undirected).
    weights : str or array, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.

    Returns
    -------
    path : list of ints
        Order of the nodes making up the path from `source` to `target`.

    References
    ----------
    .. [nx-sp] :nxdoc:`algorithms.shortest_paths.generic.shortest_path`
    '''
    g, graph, w = _get_nx_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_nx_weights(g, w)

    try:
        return nx.shortest_path(graph, source, target, weight=w)
    except nx.NetworkXNoPath:
        return []


def all_shortest_paths(g, source, target, directed=None, weights=None,
                       combine_weights="mean"):
    '''
    Yields all shortest paths from `source` to `target`.
    The algorithms returns an empty generator if there is no path between the
    nodes.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    source : int
        Node from which the paths starts.
    target : int, optional (default: all nodes)
        Node where the paths ends.
    directed : bool, optional (default: ``g.is_directed()``)
        Whether the edges should be considered as directed or not
        (automatically set to False if `g` is undirected).
    weights : str or array, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.

    Returns
    -------
    all_paths : generator
        Generator yielding paths as lists of ints.

    References
    ----------
    .. [nx-sp] :nxdoc:`algorithms.shortest_paths.generic.all_shortest_paths`
    '''
    g, graph, w = _get_nx_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_nx_weights(g, w)

    try:
        return nx.all_shortest_paths(graph, source, target, weight=w)
    except nx.NetworkXNoPath:
        return (_ for _ in [])


def shortest_distance(g, sources=None, targets=None, directed=None,
                      weights=None, combine_weights="mean"):
    '''
    Returns the length of the shortest paths between `sources`and `targets`.
    The algorithms return infinity if there are no paths between nodes.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    sources : list of nodes, optional (default: all)
        Nodes from which the paths must be computed.
    targets : list of nodes, optional (default: all)
        Nodes to which the paths must be computed.
    directed : bool, optional (default: ``g.is_directed()``)
        Whether the edges should be considered as directed or not
        (automatically set to False if `g` is undirected).
    weights : str or array, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.

    Returns
    -------
    distance : float, or 1d/2d numpy array of floats
        Distance (if single source and single target) or distance array.
        For multiple sources and targets, the shape of the matrix is (S, T),
        with S the number of sources and T the number of targets; for a single
        source or target, return a 1d-array of length T or S.

    References
    ----------
    .. [nx-sp] :nxdoc:`algorithms.shortest_paths.weighted.multi_source_dijkstra`
    '''
    num_nodes = g.node_nb()

    # check consistency for weights and directed
    g, graph, w = _get_nx_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_nx_weights(g, w)

    # check for single source/target case and convert sources and targets
    if is_integer(sources):
        if is_integer(targets):
            try:
                return nx.shortest_path_length(graph, sources, targets,
                                               weight=w)
            except Exception as e:
                return np.inf

        sources = [sources]
    elif sources is None:
        sources = range(num_nodes)

    if is_integer(targets):
        targets = [targets]

    # compute distances
    data, ii, jj = [], [], []

    def _nx_sp(nx_graph, s, weight):
        if weight is None:
            return nx.single_source_shortest_path_length(nx_graph, s)

        dist, _ = nx.multi_source_dijkstra(graph, [s], weight=weight)

        return dist

    for s in sources:
        dist = _nx_sp(graph, s, w)

        if targets is None:
            data.extend(dist.values())
            ii.extend((s for _ in range(len(dist))))
            jj.extend(dist.keys())
        else:
            for t in targets:
                if t in dist:
                    data.append(dist[t])
                    ii.append(s)
                    jj.append(t)

    num_sources = num_nodes if sources is None else len(sources)
    num_targets = num_nodes if targets is None else len(targets)

    mat_dist = np.full((num_sources, num_targets), np.inf)
    mat_dist[ii, jj] = data

    if num_sources == 1:
        return mat_dist[0]

    if num_targets == 1:
        return mat_dist.T[0]

    return mat_dist


def average_path_length(g, sources=None, targets=None, directed=None,
                        weights=None, combine_weights="mean",
                        unconnected=False):
    r'''
    Returns the average shortest path length between `sources` and `targets`.
    The algorithms raises an error if all nodes are not connected unless
    `unconnected` is set to True.

    The average path length is defined as

    .. math::

       L = \frac{1}{N_p} \sum_{u,v} d(u, v),

    where :math:`N_p` is the number of paths between `sources` and `targets`,
    and :math:`d(u, v)` is the shortest path distance from u to v.

    If `sources` and `targets` are both None, then the total number of paths is
    :math:`N_p = N(N - 1)`, with :math:`N` the number of nodes in the graph.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    sources : list of nodes, optional (default: all)
        Nodes from which the paths must be computed.
    targets : list of nodes, optional (default: all)
        Nodes to which the paths must be computed.
    directed : bool, optional (default: ``g.is_directed()``)
        Whether the edges should be considered as directed or not
        (automatically set to False if `g` is undirected).
    weights : str or array, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.
    unconnected : bool, optional (default: False)
        If set to true, ignores unconnected nodes and returns the average path
        length of the existing paths.

    References
    ----------
    .. [nx-sp] :nxdoc:`algorithms.shortest_paths.generic.average_shortest_path_length`
    '''
    directed = g.is_directed() if directed is None else directed

    if sources is None and targets is None and not unconnected:
        g, graph, w = _get_nx_graph(g, directed, weights, combine_weights,
                                 return_all=True)

        w = _get_nx_weights(g, w)

        return nx.average_shortest_path_length(graph, weight=w)

    mat_dist = shortest_distance(g, sources=sources, targets=targets,
                                 directed=directed, weights=weights)

    if not unconnected and np.any(np.isinf(mat_dist)):
        raise nx.NetworkXNoPath("`sources` and `target` do not belong to the "
                                "same connected component.")

    # compute the number of path
    num_paths = np.sum(mat_dist != 0)

    # compute average path length
    if unconnected:
        num_paths -= np.sum(np.isinf(mat_dist))

        return np.nansum(mat_dist) / num_paths

    return np.sum(mat_dist) / num_paths


def diameter(g, directed=None, weights=None, combine_weights="mean",
             is_connected=False):
    '''
    Returns the diameter of the graph.

    .. versionchanged:: 2.3
        Added `combine_weights` argument.

    .. versionchanged:: 2.0
        Added `directed` and `is_connected` arguments.

    It returns infinity if the graph is not connected (strongly connected for
    directed graphs) unless `is_connected` is True, in which case it returns
    the longest existing shortest distance.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    directed : bool, optional (default: ``g.is_directed()``)
        Whether to compute the directed diameter if the graph is directed.
        If False, then the graph is treated as undirected. The option switches
        to False automatically if `g` is undirected.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.
    is_connected : bool, optional (default: False)
        If False, check whether the graph is connected or not and return
        infinite diameter if graph is unconnected. If True, the graph is
        assumed to be connected.

    See also
    --------
    :func:`nngt.analysis.shortest_distance`

    References
    ----------
    .. [nx-diameter] :nxdoc:`algorithms.distance_measures.diameter`
    .. [nx-dijkstra] :nxdoc:`algorithms.shortest_paths.weighted.all_pairs_dijkstra`
    '''
    w = _get_nx_weights(g, weights)

    # weighted or "connected" cases
    if w is not None or is_connected:
        dist = shortest_distance(g, directed=directed, weights=weights,
                                 combine_weights=combine_weights)

        if is_connected:
            return np.max(dist[~np.isinf(dist)])

        return np.max(dist)

    # unweighted case
    graph = _get_nx_graph(g, directed, w, combine_weights)

    try:
        return nx.diameter(graph)
    except nx.exception.NetworkXError:
        return np.inf


def adj_mat(g, weights=None, mformat="csr"):
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
    mformat : str, optional (default: "csr")
        Type of :mod:`scipy.sparse` matrix that will be returned, by
        default :class:`scipy.sparse.csr_matrix`.

    Returns
    -------
    The adjacency matrix as a :class:`scipy.sparse.csr_matrix`.

    References
    ----------
    .. [nx-adjacency] :nxdoc:`.convert_matrix.to_scipy_sparse_matrix`
    '''
    w = _get_nx_weights(g, weights)

    return nx.to_scipy_sparse_matrix(g.graph, nodelist=range(g.node_nb()),
                                     weight=w, format=mformat)


def get_edges(g):
    '''
    Returns the edges in the graph by order of creation.
    '''
    return g.edges_array

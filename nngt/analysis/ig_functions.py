#-*- coding:utf-8 -*-
#
# analysis/ig_functions.py
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

""" Tools to analyze graphs with the igraph backend """

import numpy as np
import scipy.sparse as ssp

from ..lib.test_functions import nonstring_container, is_integer
from ..lib.graph_helpers import _get_ig_weights, _get_ig_graph


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
    .. [ig-global-clustering] :igdoc:`transitivity_undirected`
    '''
    graph = g.graph

    if graph.is_loop():
        graph = graph.copy()
        graph.simplify(multiple=False, loops=True)

    return np.array(graph.as_undirected().transitivity_undirected())


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
    .. [ig-local-clustering] :igdoc:`transitivity_local_undirected`
    '''
    graph = g.graph

    if graph.is_loop():
        graph = graph.copy()
        graph.simplify(multiple=False, loops=True)

    u = graph.as_undirected()

    return np.array(
        u.transitivity_local_undirected(nodes, mode="zero", weights=None))


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
    .. [ig-assortativity] :igdoc:`assortativity`
    '''
    ww = _get_ig_weights(g, weights)

    node_attr = g.get_degrees(degree, weights=ww)

    return g.graph.assortativity(node_attr, directed=g.is_directed())


def reciprocity(g):
    '''
    Calculate the edge reciprocity of the graph.

    The reciprocity is defined as the number of edges that have a reciprocal
    edge (an edge between the same nodes but in the opposite direction)
    divided by the total number of edges.
    This is also the probability for any given edge, that its reciprocal edge
    exists.
    By definition, the reciprocity of undirected graphs is 1.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.

    References
    ----------
    .. [ig-reciprocity] :igdoc:`reciprocity`
    '''
    return g.graph.reciprocity(ignore_loops=True, mode="default")


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
    and `n` is the number of nodes in the strongly-connected component (SCC).

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
        Whether the arithmetic or the harmonic (recommended) version of the
        closeness centrality should be used.

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
    .. [ig-closeness] :igdoc:`closeness`
    '''
    if harmonic:
        ww = _get_ig_weights(g, weights)

        try:
            return np.asarray(g.graph.harmonic_centrality(nodes, mode=mode,
                                                          weights=ww))
        except:
            raise RuntimeError("This function requires igraph >= 0.9.0.")

    ww = _get_ig_weights(g, weights)

    return np.asarray(g.graph.closeness(nodes, mode=mode, weights=ww))


def betweenness(g, btype="both", weights=None):
    '''
    Returns the normalized betweenness centrality of the nodes and edges.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    ctype : str, optional (default 'both')
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
    .. [ig-ebetw] :igdoc:`edge_betweenness`
    .. [ig-nbetw] :igdoc:`betweenness`
    '''
    w = _get_ig_weights(g, weights)

    n  = g.node_nb()

    nb, eb = None, None

    if btype in ("both", "node"):
        di_nb = np.array(g.graph.betweenness(weights=w))
        nb    = np.array([di_nb[i] for i in g.get_nodes()])
        norm  = 1.

        if n > 2:
            norm = (1 if g.is_directed() else 2) / ((n - 1) * (n - 2))

        nb *= norm

    if btype in ("both", "edge"):
        eb = np.array(g.graph.edge_betweenness(weights=w))

        norm = (1 if g.is_directed() else 2) / (n * (n - 1))
        eb *= norm

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
        The components are numbered from 0.

    References
    ----------
    .. [ig-connected-components] :igdoc:`clusters`
    '''
    if ctype is None:
        ctype = "scc" if g.is_directed() else "wcc"

    ig_type = "strong"

    if ctype == "wcc":
        ig_type = "weak"
    elif ctype != "scc":
        raise ValueError("`ctype` must be either 'scc' or 'wcc'.")

    clusters = g.graph.clusters(ig_type)

    cc = np.zeros(g.node_nb(), dtype=int)

    for i, nodes in enumerate(clusters):
        cc[nodes] = i

    hist = np.array([len(c) for c in clusters], dtype=int)

    return cc, hist


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
    path : array of ints
        Order of the nodes making up the path from `source` to `target`.

    References
    ----------
    .. [ig-sp] :igdoc:`get_shortest_paths`
    '''
    g, graph, w = _get_ig_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_ig_weights(g, w)

    path = graph.get_shortest_paths(source, target, mode="out", weights=w)[0]

    if source != target and len(path) == 1:
        # weird igraph issue
        return []

    return path


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
    .. [ig-sp] :igdoc:`get_all_shortest_paths`
    '''
    g, graph, w = _get_ig_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_ig_weights(g, w)

    return (p for p in graph.get_all_shortest_paths(source, target, weights=w))


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
    .. [ig-sp] :igdoc:`shortest_paths`
    '''
    # weighted or selective algorithm
    g, graph, w = _get_ig_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_ig_weights(g, w)

    # special case for one source/one target
    if is_integer(sources) and is_integer(targets):
        return graph.shortest_paths(source=sources, target=targets,
                                    weights=w)[0][0]

    # multiple sources/targets
    mat_dist = graph.shortest_paths(source=sources, target=targets,
                                    weights=w)

    mat_dist = np.array(mat_dist, dtype=float)

    if mat_dist.shape[0] == 1:
        return mat_dist[0]

    if mat_dist.shape[1] == 1:
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
    .. [ig-sp] :igdoc:`shortest_paths`
    '''
    directed = g.is_directed() if directed is None else directed

    mat_dist = shortest_distance(g, sources=sources, targets=targets,
                                 directed=directed, weights=weights)

    if not unconnected and np.any(np.isinf(mat_dist)):
        raise RuntimeError("`sources` and `target` do not belong to the "
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
    .. [ig-diameter] :igdoc:`diameter`
    '''
    g, graph, w = _get_ig_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_ig_weights(g, w)

    if not is_connected:
        mode = "strong" if g.is_directed() else "weak"

        if not graph.is_connected(mode):
            return np.inf

    return graph.diameter(weights=w, unconn=True)


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

    Note
    ----
    This function does not use a builtin igraph method.

    Returns
    -------
    The adjacency matrix as a :class:`scipy.sparse.csr_matrix`.
    '''
    n = g.node_nb()

    w = _get_ig_weights(g, weights)

    if g.edge_nb():
        xs, ys = map(np.array, zip(*g.graph.get_edgelist()))
        xs, ys = xs.T, ys.T

        data = np.ones(xs.shape)

        if w is not None:
            data *= w

        coo_adj = ssp.coo_matrix((data, (xs, ys)), shape=(n, n))

        if not g.is_directed():
            coo_adj += coo_adj.T

        return coo_adj.asformat(mformat)

    m = ssp.coo_matrix((n, n))

    return m.asformat(mformat)


def get_edges(g):
    '''
    Returns the edges in the graph by order of creation.
    '''
    return g.graph.get_edgelist()

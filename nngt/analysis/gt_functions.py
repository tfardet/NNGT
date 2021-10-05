#-*- coding:utf-8 -*-
#
# analysis/gt_functions.py
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

""" Tools to analyze graphs with the graph-tool backend """

import numpy as np
import scipy.sparse as ssp

from graph_tool import GraphView, _prop
from graph_tool.centrality import closeness as gt_closeness
from graph_tool.centrality import betweenness as gt_betweenness
from graph_tool.correlations import scalar_assortativity
from graph_tool.spectral import libgraph_tool_spectral
from graph_tool.stats import label_parallel_edges

import graph_tool.topology as gtt
import graph_tool.clustering as gtc

from ..lib.test_functions import nonstring_container
from ..lib.graph_helpers import _get_gt_weights, _get_gt_graph


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
    .. [gt-global-clustering] :gtdoc:`clustering.global_clustering`
    '''
    # use undirected graph view, filter parallel edges
    u = GraphView(g.graph, directed=False)
    u = GraphView(u, efilt=label_parallel_edges(u).fa == 0)

    return gtc.global_clustering(u, weight=None)[0]


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
    .. [gt-local-clustering] :gtdoc:`clustering.locall_clustering`
    '''
    # use undirected graph view, filter parallel edges
    u = GraphView(g.graph, directed=False)
    u = GraphView(u, efilt=label_parallel_edges(u).fa == 0)

    # compute clustering
    lc = gtc.local_clustering(u, weight=None, undirected=None).a

    if nodes is None:
        return lc

    return lc[nodes]


def assortativity(g, degree, weights=None):
    '''
    Returns the assortativity of the graph.

    This tells whether nodes are preferentially connected together depending
    on their degree.
    For directed graphs, assortativity is performed with the same degree (e.g.
    in/in, out/out, total/total), for undirected graph, they are all the same
    and equivalent to "total".

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    degree : str
        The type of degree that should be considered ("in", "out", or "total").
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.

    References
    ----------
    .. [gt-assortativity] :gtdoc:`correlations.scalar_assortativity`
    '''
    # graph-tool expects "total" for undirected graphs
    if not g.is_directed():
        degree = "total"

    if not nonstring_container(weights) and weights in {None, False}:
        return scalar_assortativity(g.graph, degree)[0]

    # for weighted assortativity, use node strength
    strength = g.get_degrees(degree, weights=weights)
    ep = g.graph.new_vertex_property("double", vals=strength)

    return scalar_assortativity(g.graph, ep)[0]


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
    .. [gt-reciprocity] :gtdoc:`topology.edge_reciprocity`
    '''
    return gtt.edge_reciprocity(g.graph)


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
    c : :class:`np.ndarray`
        The list of closeness centralities, on per node.

    References
    ----------
    .. [gt-closeness] :gtdoc:`centrality.closeness`
    '''
    ww = _get_gt_weights(g, weights)

    if mode == "in":
        g.graph.set_reversed(True)

    c = gt_closeness(g.graph, weight=ww, harmonic=harmonic).a

    g.graph.set_reversed(False)

    if not np.isnan(default):
        c[np.isnan(c)] = default

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
    ctype : str, optional (default 'both')
        The centrality that should be returned (either 'node', 'edge', or
        'both'). By default, both betweenness centralities are computed.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.

    Returns
    -------
    nb : :class:`np.ndarray`
        The nodes' betweenness if `btype` is 'node' or 'both'
    eb : :class:`np.ndarray`
        The edges' betweenness if `btype` is 'edge' or 'both'

    References
    ----------
    .. [gt-betw] :gtdoc:`centrality.betweenness`
    '''
    w = _get_gt_weights(g, weights)

    n  = g.node_nb()

    nb, eb = gt_betweenness(g.graph, weight=w)

    if btype == "node":
        return nb.a
    elif btype == "edge":
        return eb.a

    return nb.a, eb.a


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
    cc, hist : :class:`np.ndarray`
        The component associated to each node (`cc`) and the number of nodes in
        each of the component (`hist`).

    References
    ----------
    .. [gt-connected-components] :gtdoc:`topology.label_components`
    '''
    if ctype is None:
        ctype = "scc" if g.is_directed() else "wcc"

    if ctype not in ("scc", "wcc"):
        raise ValueError("`ctype` must be either 'scc' or 'wcc'.")

    directed  = True if ctype == "scc" else False
    directed *= g.is_directed()

    cc, hist = gtt.label_components(g.graph, directed=directed)

    return cc.a, hist


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
    .. [gt-sp] :gtdoc:`topology.shortest_path`
    '''
    # source == target case
    if source == target:
        return [source]

    # non-trivial cases
    g, graph, w = _get_gt_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_gt_weights(g, w)

    path, _ = gtt.shortest_path(graph, source, target, weights=w)

    return [int(v) for v in path]


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
    .. [gt-sd] :gtdoc:`topology.all_shortest_paths`
    '''
    # source == target case
    if source == target:
        return ([source] for _ in range(1))

    # not trivial cases
    g, graph, w = _get_gt_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_gt_weights(g, w)

    all_paths = gtt.all_shortest_paths(graph, source, target, weights=w)

    return ([int(v) for v in path] for path in all_paths)


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
    .. [gt-sd] :gtdoc:`topology.shortest_distance`
    '''
    num_nodes = g.node_nb()

    g, graph, w = _get_gt_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_gt_weights(g, w)

    dist_emap = None
    tgt_vtx   = None

    maxint = np.iinfo(np.int32).max
    maxflt = np.finfo(np.float64).max

    # convert sources and targets
    if sources is not None:
        if nonstring_container(sources):
            sources = [graph.vertex(s) for s in sources]
        else:
            sources = [graph.vertex(sources)]

    if nonstring_container(targets):
        tgt_vtx = [graph.vertex(t) for t in targets]
    elif targets is not None:
        tgt_vtx = [graph.vertex(targets)]
        targets = [targets]

    # compute only specific paths
    if sources is not None:
        # single source/target case
        if targets is not None and len(sources) == 1 and len(targets) == 1:
            distance = gtt.shortest_distance(
                graph, source=sources[0], target=tgt_vtx[0], weights=w)

            if w is None:
                if distance == maxint:
                    return np.inf
            elif distance == maxflt:
                return np.inf

            return float(distance)

        # multiple sources
        num_sources = len(sources)
        num_targets = num_nodes if targets is None else len(targets)

        mat_dist = np.full((num_sources, num_targets), np.NaN)

        for s in sources:
            s_int = int(s)
            dist = gtt.shortest_distance(graph, source=s, target=tgt_vtx,
                                         weights=w)

            mat_dist[s_int] = dist.a

        # convert max int and float to inf
        if w is None:
            mat_dist[mat_dist == maxint] = np.inf
        else:
            mat_dist[mat_dist == maxflt] = np.inf

        if num_sources == 1:
            return mat_dist[0]

        if num_targets == 1:
            return mat_dist[0]

        return mat_dist

    # if source is None, then we compute all paths
    dist = gtt.shortest_distance(graph, weights=w)

    # transpose (graph-tool uses columns as sources)
    mat_dist = dist.get_2d_array([i for i in range(num_nodes)]).astype(float).T

    if w is None:
        # check unconnected with int32
        mat_dist[mat_dist == maxint] = np.inf
    else:
        # check float max
        mat_dist[mat_dist == maxflt] = np.inf

    if targets is not None:
        if len(targets) == 1:
            return mat_dist.T[0]

        return mat_dist[:, targets]

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
    unconnected : bool, optional (default: False)
        If set to true, ignores unconnected nodes and returns the average path
        length of the existing paths.
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

    References
    ----------
    .. [gt-sd] :gtdoc:`topology.shortest_distance`
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
    .. [gt-diameter] :gtdoc:`topology.pseudo_diameter`
    '''
    g, graph, w = _get_gt_graph(g, directed, weights, combine_weights,
                             return_all=True)

    w = _get_gt_weights(g, w)

    # first check whether the graph is fully connected
    ctype = "scc" if directed else "wcc"

    if not is_connected:
        cc, hist = connected_components(g, ctype)

        if len(hist) > 1:
            return np.inf

    return gtt.pseudo_diameter(graph, weights=w)[0]


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
    .. [gt-adjacency] :gtdoc:`spectral.adjacency`
    '''
    ww = _get_gt_weights(g, weights)

    graph = g.graph
    index = None

    if graph.get_vertex_filter()[0] is not None:
        index = graph.new_vertex_property("int64_t")
        index.fa = np.arange(graph.num_vertices())
    else:
        index = graph.vertex_index

    E = graph.num_edges() if graph.is_directed() else 2 * graph.num_edges()

    data = np.zeros(E, dtype="double")
    i = np.zeros(E, dtype="int32")
    j = np.zeros(E, dtype="int32")

    libgraph_tool_spectral.adjacency(
        graph._Graph__graph, _prop("v", graph, index),
        _prop("e", graph, ww), data, i, j)

    if E > 0:
        V = max(graph.num_vertices(), max(i.max() + 1, j.max() + 1))
    else:
        V = graph.num_vertices()

    # we take the convention using rows for outgoing connections
    m = ssp.coo_matrix((data, (j, i)), shape=(V, V))

    return m.asformat(mformat)


def get_edges(g):
    '''
    Returns the edges in the graph by order of creation.
    '''
    return g.graph.edges()

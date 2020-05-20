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
import scipy.sparse as ssp

from ..lib.test_functions import nonstring_container, is_integer
from ..lib.graph_helpers import _get_nx_weights

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
    lc = nx.clustering(g.graph.to_undirected(as_view=True), nodes=nodes,
                       weight=None)
       
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

    w = _get_nx_weights(g, weights)

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


def shortest_distance(g, sources=None, targets=None, directed=True,
                      weights=None, mformat='dense'):
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
    directed : bool, optional (default: True)
        Whether the edges should be considered as directed or not
        (automatically set to False if `g` is undirected).
    weights : str, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.
    mformat : str, optional (default: 'dense')
        Format of the distance matrix returned: either a dense numpy matrix
        or one of the :mod:`scipy.sparse` matrices ('bsr', 'coo', 'csr', 'csc',
        'lil').

    Returns
    -------
    distance : float or matrix
        Distance (if single source and single target) or distance matrix.
        If `mformat` is 'dense', then the shape of the matrix is (S, T),
        with S the number of sources and T the number of targets;
        otherwise (for sparse matrices) only the relevant entries are present
        but the matrix size is (N, N) with N the number of nodes in the graph.

    References
    ----------
    .. [nx-sp] :nxdoc:`algorithms.shortest_paths.generic.shortest_path_length`
    '''
    # networkx raises NetworkXNoPath if nodes are not connected so no
    # additional check is necessary
    graph = g.graph if directed else g.graph.to_undirected(as_view=True)

    num_nodes = g.node_nb()

    # check consistency for weights and directed
    w = _get_nx_weights(g, weights)

    if g.graph.is_directed() and not directed and w is not None:
        raise ValueError("Cannot make graph undirected if `weights` are used.")

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

    if w is None:
        for s in sources:
            dist = nx.single_source_shortest_path_length(graph, s)

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
    else:
        dist, _ = multi_source_dijkstra(graph, sources, weight=w)

        for s, d in dist.items():
            if targets is None:
                data.extend(d.values())
                ii.extend((s for _ in range(len(d))))
                jj.extend(d.keys())
            else:
                for t in targets:
                    if t in d:
                        data.append(d[t])
                        ii.append(s)
                        jj.append(t)

    num_sources = num_nodes if sources is None else len(sources)
    num_targets = num_nodes if targets is None else len(targets)

    if mformat == 'dense':
        mat_dist = np.full((num_sources, num_targets), np.inf)
        mat_dist[ii, jj] = data

        if num_sources == 1:
            return mat_dist[0]

        if num_targets == 1:
            return mat_dist.T[0]

        return mat_dist

    coo = ssp.coo_matrix((data, (ii, jj)), shape=(num_nodes, num_nodes))

    return coo.asformat(mformat)


def average_path_length(g, sources=None, targets=None, directed=True,
                        weights=None):
    r'''
    Returns the average shortest path length between `sources` and `targets`.
    The algorithms raises an error if all nodes are not connected.

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
    directed : bool, optional (default: True)
        Whether the edges should be considered as directed or not
        (automatically set to False if `g` is undirected).
    weights : str, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.

    References
    ----------
    .. [ig-sp] :igdoc:`shortest_paths`
    '''
    mat_dist = shortest_paths_length(g, sources=sources, targets=targets,
                                     directed=directed, weights=weights,
                                     mformat='coo')

    return mat_dist.mean()


def average_path_length(g, weights=None):
    r'''
    Returns the average shortest path length.

    The average shortest path length is

    .. math::

       L = \sum_{u,v} \frac{d(u, v)}{N(N-1)}

    where :math:`N` is the number of nodes in `g` and :math:`d(u, v)` is the
    shortest path distance from u to v.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    weights : str, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.

    References
    ----------
    .. [nx-apl] :nxdoc:`algorithms.shortest_paths.generic.average_shortest_path_length`
    '''
    return nx.average_shortest_path_length(g.graph, weight=weights)


def diameter(g, weights=False):
    '''
    Returns the diameter of the graph.

    It returns infinity if the graph is not connected (strongly connected for
    directed graphs).

    For weighted graphs, uses the Dijkstra algorithm to find all shortests
    paths and returns the longest.

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
    .. [nx-diameter] :nxdoc:`algorithms.distance_measures.diameter`
    .. [nx-dijkstra] :nxdoc:`algorithms.shortest_paths.weighted.all_pairs_dijkstra`
    '''
    w = _get_nx_weights(g, weights)

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

    return nx.to_scipy_sparse_matrix(g.graph, weight=w, format=mformat)


def get_edges(g):
    '''
    Returns the edges in the graph by order of creation.
    '''
    return g.edges_array

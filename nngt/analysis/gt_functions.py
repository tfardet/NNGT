#-*- coding:utf-8 -*-
#
# gt_functions.py
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
from ..lib.graph_helpers import _get_gt_weights


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
    lc = gtc.local_clustering(u, weight=None, undirected=True).a

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
    c : :class:`np.ndarray`
        The list of closeness centralities, on per node.

    .. warning ::
        For compatibility reasons (harmonic closeness is not implemented for
        igraph), the arithmetic version is used by default; however, it is
        recommended to use the harmonic version instead whenever possible.

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
    ctype = "scc" if ctype is None else ctype

    if ctype not in ("scc", "wcc"):
        raise ValueError("`ctype` must be either 'scc' or 'wcc'.")

    directed  = True if ctype == "scc" else False
    directed *= g.is_directed()

    cc, hist = gtt.label_components(g.graph, directed=directed)

    return cc.a, hist


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
    .. [gt-sd] :gtdoc:`topology.shortest_distance`
    '''
    # graph-tool returns infinity for unconnected nodes so there is no need
    # to check connectedness
    graph = g.graph

    num_nodes = g.node_nb()

    w = _get_gt_weights(g, weights)

    if not directed and graph.is_directed():
        if w is not None:
            raise ValueError(
                "Cannot make graph undirected if `weights` are used.")

        graph = GraphView(g.graph, directed=False)
        graph = GraphView(graph, efilt=label_parallel_edges(u).fa == 0)

    dist_emap = None
    tgt_vtx   = None

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
                graph, source=sources[0], target=tgt_vtx[0], weights=weights)

            if weights is None:
                if distance == 2147483647:
                    return np.inf

            return float(distance)

        # multiple sources
        data, ii, jj = [], [], []

        for s in sources:
            s_int = int(s)
            dist = gtt.shortest_distance(graph, source=s, target=tgt_vtx,
                                         weights=weights)

            if targets is None:
                # dist is a 1d property map
                data.extend(dist.a)
                ii.extend((s_int for _ in range(num_nodes)))
                jj.extend((t for t in range(num_nodes)))
            else:
                # dist is an array
                data.extend(dist)
                ii.extend((s_int for _ in range(num_nodes)))
                jj.extend((t for t in targets))

        # convert 2147483647 to inf for binary graphs
        if weights is None:
            data = np.array(data, dtype=float)

            data[data == 2147483647] = np.inf

        num_sources = len(sources)
        num_targets = num_nodes if targets is None else len(targets)

        if mformat == 'dense':
            mat_dist = np.full((num_sources, num_targets), np.NaN)
            mat_dist[ii, jj] = data

            if num_sources == 1:
                return mat_dist[0]

            if num_targets == 1:
                return mat_dist[0]

            return mat_dist

        coo = ssp.coo_matrix((data, (ii, jj)), shape=(num_nodes, num_nodes))

        return coo.asformat(mformat)

    # if source is None, then we compute all paths
    dist = gtt.shortest_distance(graph, weights=weights)

    # transpose (graph-tool uses columns as sources)
    mat_dist = dist.get_2d_array([i for i in range(num_nodes)]).astype(float).T

    if weights is None:
        # check unconnected with int32
        mat_dist[mat_dist == 2147483647] = np.inf

    if mformat == 'dense':
        if targets is not None:
            if len(targets) == 1:
                return mat_dist.T[0]

            return mat_dist[:, targets]

        return mat_dist

    sp_mat = None

    if targets is None:
        sp_mat = ssp.coo_matrix(mat_dist)
    else:
        sp_mat = ssp.lil_matrix((num_nodes, num_nodes))
        sp_mat[:, targets] = mat_dist[:, targets]

    return sp_mat.asformat(mformat)


def diameter(g, weights=False):
    '''
    Returns the pseudo-diameter of the graph.

    This function returns an approximmation of the graph diameter.
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
    .. [gt-diameter] :gtdoc:`topology.pseudo_diameter`
    '''
    ww = _get_gt_weights(g, weights)

    # first check whether the graph is fully connected
    cc, hist = connected_components(g)

    if len(hist) > 1:
        return np.inf

    return gtt.pseudo_diameter(g.graph, weights=ww)[0]


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

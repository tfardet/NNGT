#-*- coding:utf-8 -*-
#
# ig_functions.py
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

""" Tools to analyze graphs with the igraph backend """

import numpy as np
import scipy.sparse as ssp

from ..lib.test_functions import nonstring_container
from ..lib.graph_helpers import _get_ig_weights


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

    References
    ----------
    .. [ig-global-clustering] :igdoc:`transitivity_undirected`
    '''
    ww = _get_ig_weights(g, weights)

    if ww is not None:
        # raise warning
        return np.average(
            g.graph.as_undirected().transitivity_local_undirected(weights=ww))

    return np.array(g.graph.as_undirected().transitivity_undirected())


def undirected_local_clustering(g, weights=None, nodes=None,
                                combine_weights="sum"):
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
    .. [ig-local-clustering] :igdoc:`transitivity_local_undirected`
    '''
    if weights is not None and not isinstance(weights, str):
        raise ValueError("Only existing attributes can be used as weights.")

    u = g.graph.as_undirected(combine_edges=combine_weights)
    u.simplify()

    return np.array(u.transitivity_local_undirected(nodes, weights=weights))


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

    node_attr = g.get_degrees(degree, use_weights=ww)

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

    Note
    ----
    When requesting a subset of nodes, check whether it is faster to use
    `nodes`, or to compute all closeness centralities, then take the subset.

    References
    ----------
    .. [ig-closeness] :igdoc:`closeness`
    '''
    if harmonic:
        raise NotImplementedError("`harmonic` closeness is not available with "
                                  "igraph backend.")

    if not np.all(g.get_degrees("in")) or not np.all(g.get_degrees("out")):
        raise RuntimeError("igraph backend does not support closeness for "
                           "graphs containing nodes with zero in/out-degree.")

    ww = _get_ig_weights(g, weights)

    return g.graph.closeness(nodes, mode=mode, weights=ww)


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
    ctype   = "scc" if ctype is None else ctype
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


def diameter(g, weights=None):
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
    .. [ig-diameter] :igdoc:`diameter`
    '''
    ww = _get_ig_weights(g, weights)

    mode = "strong" if g.is_directed() else "weak"

    if not g.graph.is_connected(mode):
        return np.inf

    return g.graph.diameter(weights=ww)


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

        return coo_adj.tocsr()

    return ssp.csr_matrix((n, n))


def get_edges(g):
    '''
    Returns the edges in the graph by order of creation.
    '''
    return g.graph.get_edgelist()

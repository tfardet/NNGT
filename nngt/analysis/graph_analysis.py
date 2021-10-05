#-*- coding:utf-8 -*-
#
# analysis/graph_analysis.py
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

""" Tools for graph analysis using the graph libraries """

import logging

import numpy as np
import scipy.sparse.linalg as spl

import nngt
import nngt.generation as ng

from nngt.lib import InvalidArgument, nonstring_container, is_integer
from nngt.lib.logger import _log_message
from . import clustering
from .activity_analysis import get_b2, get_firing_rate
from .bayesian_blocks import bayesian_blocks
from .clustering import *


logger = logging.getLogger(__name__)


# implemented function; import from proper backend is done at the bottom

__all__ = [
    "adjacency_matrix",
    "all_shortest_paths",
    "assortativity",
    "average_path_length",
    "betweenness",
    "betweenness_distrib",
    "binning",
	"closeness",
	"connected_components",
    "degree_distrib",
	"diameter",
    "node_attributes",
	"num_iedges",
	"reciprocity",
    "shortest_distance",
    "shortest_path",
    "small_world_propensity",
	"spectral_radius",
    "subgraph_centrality",
    "transitivity",
]


__all__.extend(clustering.__all__)


_backend_required = "Please install either networkx, igraph, or graph-tool " \
                    "to use this function."


# ---------------- #
# Graph properties #
# ---------------- #

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
    .. [newman-mixing-2003] M. E. J. Newman, "Mixing patterns in networks",
        Phys. Rev. E 67, 026126 (2003), see graph-tool below for links.
    .. [gt-assortativity] :gtdoc:`correlations.scalar_assortativity`
    .. [ig-assortativity] :igdoc:`assortativity`
    .. [nx-assortativity]
       :nxdoc:`algorithms.assortativity.degree_assortativity_coefficient`
    '''
    raise NotImplementedError(_backend_required)


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
    .. [wasserman-1994] S. Wasserman and K. Faust, "Social Network Analysis".
       (Cambridge University Press, Cambridge, 1994)
    .. [lopez-2007] Gorka Zamora-López, Vinko Zlatić, Changsong
       Zhou, Hrvoje Štefančić, and Jürgen Kurths "Reciprocity of networks with
       degree correlations and arbitrary degree sequences", Phys. Rev. E 77,
       016106 (2008) :doi:`10.1103/PhysRevE.77.016106`, :arxiv:`0706.3372`
    .. [gt-reciprocity] :gtdoc:`topology.edge_reciprocity`
    .. [ig-reciprocity] :igdoc:`reciprocity`
    .. [nx-reciprocity] :nxdoc:`algorithms.reciprocity.overall_reciprocity`
    '''
    raise NotImplementedError(_backend_required)


def transitivity(g, directed=True, weights=None):
    '''
    Same as :func:`~nngt.analysis.global_clustering`.
    '''
    return global_clustering(g, directed=directed, weights=weights)


def num_iedges(graph):
    '''
    Returns the number of inhibitory connections.

    For :class:`~nngt.Network` objects, this corresponds to the number of edges
    stemming from inhibitory nodes (given by
    :meth:`nngt.NeuralPop.inhibitory`).
    Otherwise, counts the edges where the type attribute is -1.
    '''
    if graph.is_network():
        inhib_nodes = graph.population.inhibitory

        return np.sum(graph.get_degrees("out", node_list=inhib_nodes))

    if "type" in graph.edge_attributes:
        return np.sum(graph.get_edge_attributes(name="type") < 0)

    return 0.


def connected_components(g, ctype=None):
    '''
    Returns the connected component to which each node belongs.

    .. versionadded:: 2.0

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
    .. [gt-cc] :gtdoc:`topology.label_components`
    .. [ig-cc] :igdoc:`clusters`
    .. [nx-ucc] :nxdoc:`algorithms.components.connected_components`
    .. [nx-scc] :nxdoc:`algorithms.components.strongly_connected_components`
    .. [nx-wcc] :nxdoc:`algorithms.components.weakly_connected_components`
    '''
    raise NotImplementedError(_backend_required)


def diameter(g, directed=True, weights=False, is_connected=False):
    '''
    Returns the diameter of the graph.

    .. versionchanged:: 2.0
        Added `directed` and `is_connected` arguments.

    It returns infinity if the graph is not connected (strongly connected for
    directed graphs) unless `is_connected` is True, in which case it returns
    the longest existing shortest distance.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    directed : bool, optional (default: True)
        Whether to compute the directed diameter if the graph is directed.
        If False, then the graph is treated as undirected. The option switches
        to False automatically if `g` is undirected.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    is_connected : bool, optional (default: False)
        If False, check whether the graph is connected or not and return
        infinite diameter if graph is unconnected. If True, the graph is
        assumed to be connected.

    Warning
    -------
    For graph-tool, the [pseudo-diameter]_ is returned, which may sometime
    lead to inexact results.

    See also
    --------
    :func:`nngt.analysis.shortest_distance`

    References
    ----------
    .. [pseudo-diameter] http://en.wikipedia.org/wiki/Distance_%28graph_theory%29
    .. [dijkstra] E. Dijkstra, "A note on two problems in connexion with
       graphs." Numerische Mathematik, 1:269-271, 1959.
    .. [gt-diameter] :gtdoc:`topology.pseudo_diameter`
    .. [ig-diameter] :igdoc:`diameter`
    .. [nx-diameter] :nxdoc:`algorithms.distance_measures.diameter`
    .. [nx-dijkstra] :nxdoc:`algorithms.shortest_paths.weighted.all_pairs_dijkstra`
    '''
    raise NotImplementedError(_backend_required)


def small_world_propensity(g, directed=None, use_global_clustering=False,
                           use_diameter=False, weights=None,
                           combine_weights="mean", clustering="continuous",
                           lattice=None, random=None, return_deviations=False):
    r'''
    Returns the small-world propensity of the graph as first defined in
    [Muldoon2016]_.

    .. versionadded: 2.0

    .. math::

        \phi = 1 - \sqrt{\frac{\Pi_{[0, 1]}(\Delta_C^2) + \Pi_{[0, 1]}(\Delta_L^2)}{2}}

    with :math:`\Delta_C` the clustering deviation, i.e. the relative global or
    average clustering of `g` compared to two reference graphs

    .. math::

        \Delta_C = \frac{C_{latt} - C_g}{C_{latt} - C_{rand}}

    and :math:`Delta_L` the deviation of the average path length or diameter,
    i.e. the relative average path length of `g` compared to that of the
    reference graphs

    .. math::

        \Delta_L = \frac{L_g - L_{rand}}{L_{latt} - L_{rand}}.

    In both cases, *latt* and *rand* refer to the equivalent lattice and
    Erdos-Renyi (ER) graphs obtained by rewiring `g` to obtain respectively the
    highest and lowest combination of clustering and average path length.

    Both deviations are clipped to the [0, 1] range in case some graphs have a
    higher clustering than the lattice or a lower average path length than the
    ER graph.

    Parameters
    ----------
    g : :class:`~nngt.Graph` object
        Graph to analyze.
    directed : bool, optional (default: True)
        Whether to compute the directed clustering if the graph is directed.
        If False, then the graph is treated as undirected. The option switches
        to False automatically if `g` is undirected.
    use_global_clustering : bool, optional (default: True)
        If False, then the average local clustering is used instead of the
        global clustering.
    use_diameter : bool, optional (default: False)
        Use the diameter instead of the average path length to have more global
        information. Ccan also be much faster in some cases, especially using
        graph-tool as the backend.
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
    clustering : str, optional (default: 'continuous')
        Method used to compute the weighted clustering coefficients, either
        'barrat' [Barrat2004]_, 'continuous' (recommended), or 'onnela'
        [Onnela2005]_.
    lattice : :class:`nngt.Graph`, optional (default: generated from `g`)
        Lattice to use as reference (since its generation is deterministic,
        enables to avoid multiple generations when running the algorithm
        several times with the same graph)
    random : :class:`nngt.Graph`, optional (default: generated from `g`)
        Random graph to use as reference. Can be useful for reproducibility or
        for very sparse graphs where ER algorithm would statistically lead to
        a disconnected graph.
    return_deviations : bool, optional (default: False)
        If True, the deviations are also returned, in addition to the
        small-world propensity.

    Note
    ----
    If `weights` are provided, the distance calculation uses the inverse of
    the weights.
    This implementation differs slightly from the `original implementation
    <https://github.com/KordingLab/nctpy>`_ as it can also use the global
    instead of the average clustering coefficient, the diameter instead of
    the avreage path length, and it is generalized to directed networks.

    References
    ----------
    .. [Muldoon2016] Muldoon, Bridgeford, Bassett. Small-World Propensity and
        Weighted Brain Networks. Sci Rep 2016, 6 (1), 22057.
        :doi:`10.1038/srep22057`, :arxiv:`1505.02194`.
    .. [Barrat2004] Barrat, Barthelemy, Pastor-Satorras, Vespignani. The
        Architecture of Complex Weighted Networks. PNAS 2004, 101 (11).
        :doi:`10.1073/pnas.0400087101`.
    .. [Onnela2005] Onnela, Saramäki, Kertész, Kaski. Intensity and Coherence
        of Motifs in Weighted Complex Networks. Phys. Rev. E 2005, 71 (6),
        065103. :doi:`10.1103/physreve.71.065103`, arxiv:`cond-mat/0408629`.

    Returns
    -------
    phi : float in [0, 1]
        The small-world propensity.
    delta_l : float
        The average path-length deviation (if `return_deviations` is True).
    delta_c : float
        The clustering deviation (if `return_deviations` is True).

    See also
    --------
    :func:`nngt.analysis.average_path_length`
    :func:`nngt.analysis.diameter`
    :func:`nngt.analysis.global_clustering`
    :func:`nngt.analysis.local_clustering`
    :func:`nngt.generation.lattice_rewire`
    :func:`nngt.generation.random_rewire`
    '''
    # special case for too sparse (unconnected) graphs
    if g.edge_nb() < g.node_nb():
        if return_deviations:
            return np.NaN, np.NaN, np.NaN

        return np.NaN

    # check graph directedness
    directed = g.is_directed() if directed is None else directed

    if g.is_directed() and not directed:
        g = g.to_undirected(combine_weights)

    # rewired graph
    latt = ng.lattice_rewire(g, weight=weights) if lattice is None else lattice
    rand = ng.random_rewire(g) if random is None else random

    # compute average path-length using the inverse of the weights
    inv_w, inv_wl, inv_wr = None, None, None

    if weights not in (None, False):
        inv_w = 1 / g.edge_attributes[weights]
        inv_wl = 1 / latt.edge_attributes[weights]
        inv_wr = 1 / rand.edge_attributes[weights]

    l_latt, l_rand, l_g = None, None, None

    if use_diameter:
        l_latt = diameter(latt, directed=directed, weights=inv_wl)
        l_rand = diameter(rand, directed=directed, weights=inv_wr)
        l_g    = diameter(g, directed=directed, weights=inv_w)
    else:
        l_latt = average_path_length(latt, directed=directed, weights=inv_wl)
        l_rand = average_path_length(rand, directed=directed, weights=inv_wr)
        l_g    = average_path_length(g, directed=directed, weights=inv_w)

    # compute clustering
    c_latt, c_rand, c_g = None, None, None

    if use_global_clustering:
        c_latt = global_clustering(
            latt, directed=directed, weights=weights, method=clustering)

        c_rand = global_clustering(
            rand, directed=directed, weights=weights, method=clustering)

        c_g = global_clustering(
            g, directed=directed, weights=weights, method=clustering)
    else:
        c_latt = np.average(local_clustering(
            latt, directed=directed, weights=weights, method=clustering))

        c_rand = np.average(local_clustering(
            rand, directed=directed, weights=weights, method=clustering))

        c_g = np.average(local_clustering(
            g, directed=directed, weights=weights, method=clustering))

    # compute deltas
    delta_l = (l_g - l_rand) / (l_latt - l_rand) if l_latt != l_rand \
              else float(l_g > l_rand)
    delta_c = (c_latt - c_g) / (c_latt - c_rand)

    if np.isinf(l_rand):
        _log_message(logger, "WARNING", 'Randomized graph was unconnected.')

    if return_deviations:
        return 1 - np.sqrt(
            0.5*(np.clip(delta_l, 0, 1)**2 + np.clip(delta_c, 0, 1)**2)), \
            delta_l, delta_c
    else:
        return 1 - np.sqrt(
            0.5*(np.clip(delta_l, 0, 1)**2 + np.clip(delta_c, 0, 1)**2))


def shortest_path(g, source, target, directed=True, weights=None):
    '''
    Returns a shortest path between `source`and `target`.
    The algorithms returns an empty list if there is no path between the nodes.

    .. versionadded:: 2.0

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    source : int
        Node from which the path starts.
    target : int
        Node where the path ends.
    directed : bool, optional (default: True)
        Whether the edges should be considered as directed or not
        (automatically set to False if `g` is undirected).
    weights : str or array, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.

    Returns
    -------
    path : array of ints
        Order of the nodes making up the path from `source` to `target`.

    References
    ----------
    .. [gt-sd] :gtdoc:`topology.shortest_distance`
    .. [ig-sp] :igdoc:`shortest_paths`
    .. [nx-sp] :nxdoc:`algorithms.shortest_paths.generic.shortest_path`
    '''
    raise NotImplementedError(_backend_required)


def all_shortest_paths(g, source, target, directed=True, weights=None):
    '''
    Yields all shortest paths from `source` to `target`.
    The algorithms returns an empty generator if there is no path between the
    nodes.

    .. versionadded:: 2.0

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    source : int
        Node from which the paths starts.
    target : int, optional (default: all nodes)
        Node where the paths ends.
    directed : bool, optional (default: True)
        Whether the edges should be considered as directed or not
        (automatically set to False if `g` is undirected).
    weights : str or array, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.

    Returns
    -------
    all_paths : generator
        Generator yielding paths as lists of ints.

    References
    ----------
    .. [gt-sd] :gtdoc:`topology.all_shortest_paths`
    .. [ig-sp] :igdoc:`get_all_shortest_paths`
    .. [nx-sp] :nxdoc:`algorithms.shortest_paths.generic.all_shortest_paths`
    '''
    raise NotImplementedError(_backend_required)


def shortest_distance(g, sources=None, targets=None, directed=True,
                      weights=None):
    '''
    Returns the length of the shortest paths between `sources`and `targets`.
    The algorithms return infinity if there are no paths between nodes.

    .. versionadded:: 2.0

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
    weights : str or array, optional (default: binary)
        Whether to use weighted edges to compute the distances. By default,
        all edges are considered to have distance 1.

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
    .. [ig-sp] :igdoc:`shortest_paths`
    .. [nx-sp] :nxdoc:`algorithms.shortest_paths.weighted.multi_source_dijkstra`
    '''
    raise NotImplementedError(_backend_required)


def average_path_length(g, sources=None, targets=None, directed=None,
                        weights=None, unconnected=False):
    r'''
    Returns the average shortest path length between `sources` and `targets`.
    The algorithms raises an error if all nodes are not connected unless
    `unconnected` is set to True.

    .. versionadded:: 2.0

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
    unconnected : bool, optional (default: False)
        If set to true, ignores unconnected nodes and returns the average path
        length of the existing paths.

    References
    ----------
    .. [gt-sd] :gtdoc:`topology.shortest_distance`
    .. [ig-sp] :igdoc:`shortest_paths`
    .. [nx-sp] :nxdoc:`algorithms.shortest_paths.generic.average_shortest_path_length`

    See also
    --------
    :func:`nngt.analysis.shortest_distance`
    '''
    raise NotImplementedError(_backend_required)


# ------------ #
# Centralities #
# ------------ #

def closeness(g, weights=None, nodes=None, mode="out", harmonic=True,
              default=np.NaN):
    r'''
    Returns the closeness centrality of some `nodes`.

    .. versionadded:: 2.0

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
        Whether the arithmetic or the harmonic (recommended) version of the
        closeness should be used.

    Returns
    -------
    c : :class:`numpy.ndarray`
        The list of closeness centralities, on per node.

    References
    ----------
    .. [gt-closeness] :gtdoc:`centrality.closeness`
    .. [ig-closeness] :igdoc:`closeness`
    .. [nx-harmonic] :nxdoc:`algorithms.centrality.harmonic_centrality`
    .. [nx-closeness] :nxdoc:`algorithms.centrality.closeness_centrality`
    '''
    raise NotImplementedError(_backend_required)


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
    .. [wiki-betw] http://en.wikipedia.org/wiki/Centrality#Betweenness_centrality
    .. [gt-betw] :gtdoc:`centrality.betweenness`
    .. [ig-ebetw] :igdoc:`edge_betweenness`
    .. [ig-nbetw] :igdoc:`betweenness`
    .. [nx-ebetw] :nxdoc:`algorithms.centrality.edge_betweenness_centrality`
    .. [nx-nbetw] :nxdoc:`networkx.algorithms.centrality.betweenness_centrality`
    '''
    raise NotImplementedError(_backend_required)


def subgraph_centrality(graph, weights=True, nodes=None,
                        normalize="max_centrality"):
    '''
    Returns the subgraph centrality for each node in the graph.

    Defined according to [Estrada2005]_ as:

    .. math::

        sc(i) = e^{W}_{ii}

    where :math:`W` is the (potentially weighted and normalized) adjacency
    matrix.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Network to analyze.
    weights : bool or string, optional (default: True)
        Whether weights should be taken into account; if True, then connections
        are weighed by their synaptic strength, if False, then a binary matrix
        is returned, if `weights` is a string, then the ponderation is the
        correponding value of the edge attribute (e.g. "distance" will return
        an adjacency matrix where each connection is multiplied by its length).
    nodes : array-like container with node ids, optional (default = all nodes)
        Nodes for which the subgraph centrality should be returned (all
        centralities are computed anyway in the algorithm).
    normalize : str or False, optional (default: "max_centrality")
        Whether the centrality should be normalized. Accepted normalizations
        are "max_eigenvalue" (the matrix is divided by its largest eigenvalue),
        "max_centrality" (the largest centrality is one), and ``False`` to get
        the non-normalized centralities.

    Returns
    -------
    centralities : :class:`numpy.ndarray`
        The subgraph centrality of each node.

    References
    ----------
    .. [Estrada2005] Ernesto Estrada and Juan A. Rodríguez-Velázquez,
       Subgraph centrality in complex networks, PHYSICAL REVIEW E 71, 056103
       (2005), :doi:`10.1103/PhysRevE.71.056103`, :arxiv:`cond-mat/0504730`.
    '''
    adj_mat = graph.adjacency_matrix(types=False, weights=weights).tocsc()

    centralities = None

    if normalize == "max_centrality":
        centralities = spl.expm(adj_mat / adj_mat.max()).diagonal()
        centralities /= centralities.max()
    elif normalize == "max_eigenvalue":
        norm, _ = spl.eigs(adj_mat, k=1)
        centralities = spl.expm(adj_mat / norm).diagonal()
    elif normalize is False:
        centralities = spl.expm(adj_mat).diagonal()
    else:
        raise InvalidArgument('`normalize` should be either False, "eigenmax",'
                              ' or "centralmax".')

    if nodes is None:
        return centralities

    return centralities[nodes]


# ------------------- #
# Spectral properties #
# ------------------- #

def spectral_radius(graph, typed=True, weights=True):
    '''
    Spectral radius of the graph, defined as the eigenvalue of greatest module.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Network to analyze.
    typed : bool, optional (default: True)
        Whether the excitatory/inhibitory type of the connnections should be
        considered.
    weights : bool, optional (default: True)
        Whether weights should be taken into account, defaults to the "weight"
        edge attribute if present.

    Returns
    -------
    the spectral radius as a float.
    '''
    mat_adj  = graph.adjacency_matrix(types=typed,
                                      weights=weights).astype(float)
    eigenval = []

    try:
        eigenval = spl.eigs(mat_adj, return_eigenvectors=False)
    except spl.eigen.arpack.ArpackNoConvergence as err:
        eigenval = err.eigenvalues

    if len(eigenval):
        return np.amax(np.absolute(eigenval))

    raise spl.eigen.arpack.ArpackNoConvergence()


def adjacency_matrix(graph, types=False, weights=False):
    '''
    Adjacency matrix of the graph.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Network to analyze.
    types : bool, optional (default: False)
        Whether the excitatory/inhibitory type of the connnections should be
        considered (only if the weighing factor is the synaptic strength).
    weights : bool or string, optional (default: False)
        Whether weights should be taken into account; if True, then connections
        are weighed by their synaptic strength, if False, then a binary matrix
        is returned, if `weights` is a string, then the ponderation is the
        correponding value of the edge attribute (e.g. "distance" will return
        an adjacency matrix where each connection is multiplied by its length).

    Returns
    -------
    a :class:`~scipy.sparse.csr_matrix`.

    References
    ----------
    .. [gt-adjacency] :gtdoc:`spectral.adjacency`
    .. [nx-adjacency] :nxdoc:`.convert_matrix.to_scipy_sparse_matrix`
    '''
    return graph.adjacency_matrix(types=types, weights=weights)


# --------------- #
# Node properties #
# --------------- #

def node_attributes(network, attributes, nodes=None, data=None):
    '''
    Return node `attributes` for a set of `nodes`.

    Parameters
    ----------
    network : :class:`~nngt.Graph`
        The graph where the `nodes` belong.
    attributes : str or list
        Attributes which should be returned, among:
        * "betweenness"
        * "clustering"
        * "closeness"
        * "in-degree", "out-degree", "total-degree"
        * "subgraph_centrality"
    nodes : list, optional (default: all nodes)
        Nodes for which the attributes should be returned.
    data : :class:`numpy.array` of shape (N, 2), optional (default: None)
        Potential data on the spike events; if not None, it must contain the
        sender ids on the first column and the spike times on the second.

    Returns
    -------
    values : array-like or dict
        Returns the attributes, either as an array if only one attribute is
        required (`attributes` is a :obj:`str`) or as a :obj:`dict` of arrays.
    '''
    if nonstring_container(attributes):
        values = {}

        for attr in attributes:
            values[attr] = _get_attribute(network, attr, nodes, data)

        return values

    return _get_attribute(network, attributes, nodes, data)


def find_nodes(network, attributes, equal=None, upper_bound=None,
               lower_bound=None, upper_fraction=None, lower_fraction=None,
               data=None):
    '''
    Return the nodes in the graph which fulfill the given conditions.

    Parameters
    ----------
    network : :class:`~nngt.Graph`
        The graph where the `nodes` belong.
    attributes : str or list
        Properties on which the conditions apply, among:
        * "B2" (requires NEST or `data` entry)
        * "betweenness"
        * "clustering"
        * "firing_rate" (requires NEST or `data` entry)
        * "in-degree", "out-degree", "total-degree"
        * "subgraph_centrality"
        * any custom property formerly set by the user
    equal : optional (default: None)
        Value to which `attributes` should be equal. For a given
        property, this entry is cannot be used together with any of the
        others.
    upper_bound : optional (default: None)
        Value which should strictly major `attributes` in the desired
        nodes. Can be combined with all other entries, except `equal`.
    lower_bound : optional (default: None)
        Value which should minor or be equal to the value of `attributes`
        in the desired nodes. Can be combined with all other entries,
        except `equal`.
    upper_fraction : optional (default: None)
        Only the nodes that belong to the `upper_fraction` with the highest
        values for `attributes` are kept.
    lower_fraction : optional (default: None)
        Only the nodes that belong to the `lower_fraction` with the lowest
        values for `attributes` are kept.
    data : :class:`numpy.array` of shape (N, 2), optional (default: None)
        Potential data on the spike events; if not None, it must contain the
        sender ids on the first column and the spike times on the second.

    Notes
    -----
    When combining both `*_fraction` and `*_bound` entries, their effects
    are cumulated, i.e. only the nodes belonging to the fraction AND
    displaying a value that is consistent with the boundary are kept.

    Examples
    --------

        nodes = g.find("in-degree", upper_bound=15, lower_bound=10)
        nodes2 = g.find(["total-degree", "clustering"], equal=[20, None],
            lower=[None, 0.1])
    '''
    if not nonstring_container(attributes):
        attributes = [attributes]
        equal = [equal]
        upper_bound = [upper_bound]
        lower_bound = [lower_bound]
        upper_fraction = [upper_fraction]
        lower_fraction = [lower_fraction]
        assert not np.any([
            len(attributes)-len(equal), len(upper_bound)-len(equal),
            len(lower_bound)-len(equal), len(upper_fraction)-len(equal),
            len(lower_fraction)-len(equal)])

    nodes = set(range(self.node_nb()))

    # find the nodes
    di_attr = node_attributes(self, attributes)
    keep = np.ones(self.node_nb(), dtype=bool)

    for i in range(len(attributes)):
        attr, eq = attributes[i], equal[i]
        ub, lb = upper_bound[i], lower_bound[i]
        uf, lf = upper_fraction[i], lower_fraction[i]
        # check that the combination is valid
        if eq is not None:
            assert (ub is None)*(lb is None)*(uf is None)*(lf is None), \
            "`equal` entry is incompatible with all other entries."
            keep *= (_get_attribute(self, attr) == eq)
        if ub is not None:
            keep *= (_get_attribute(self, attr) < ub)
        if lb is not None:
            keep *= (_get_attribute(self, attr) >= lb)
        values = None
        if uf is not None or lf is not None:
            values = _get_attribute(self, attr)
        if uf is not None:
            num_keep = int(self.node_nb()*uf)
            sort = np.argsort(values)[:-num_keep]
            keep_tmp = np.ones(self.node_nb(), dtype=bool)
            keep_tmp[sort] = 0
            keep *= keep_tmp
        if lf is not None:
            num_keep = int(self.node_nb()*lf)
            sort = np.argsort(values)[:num_keep]
            keep_tmp = np.zeros(self.node_nb(), dtype=bool)
            keep_tmp[sort] = 1
            keep *= keep_tmp

    nodes = nodes.intersection_update(np.array(nodes)[keep])

    return nodes


# ------------- #
# Distributions #
# ------------- #

def degree_distrib(graph, deg_type="total", nodes=None, weights=None,
                   log=False, num_bins='bayes'):
    '''
    Degree distribution of a graph.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        the graph to analyze.
    deg_type : string, optional (default: "total")
        type of degree to consider ("in", "out", or "total").
    nodes : list of ints, optional (default: None)
        Restrict the distribution to a set of nodes (default: all nodes).
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    log : bool, optional (default: False)
        use log-spaced bins.
    num_bins : int, list or str, optional (default: 'bayes')
        Any of the automatic methodes from :func:`numpy.histogram`, or 'bayes'
        will provide automatic bin optimization. Otherwise, an int for the
        number of bins can be provided, or the direct bins list.

    See also
    --------
    :func:`numpy.histogram`, :func:`~nngt.analysis.binning`

    Returns
    -------
    counts : :class:`numpy.array`
        number of nodes in each bin
    deg : :class:`numpy.array`
        bins
    '''
    degrees = graph.get_degrees(deg_type, nodes, weights)

    if num_bins == 'bayes' or is_integer(num_bins):
        num_bins = binning(degrees, bins=num_bins, log=log)
    elif log:
        deg = degrees[degrees > 0]
        counts, bins = np.histogram(np.log(deg), num_bins)

        return counts, np.exp(bins)

    return np.histogram(degrees, num_bins)


def betweenness_distrib(graph, weights=None, nodes=None, num_nbins='bayes',
                        num_ebins='bayes', log=False):
    '''
    Betweenness distribution of a graph.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        the graph to analyze.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    nodes : list or numpy.array of ints, optional (default: all nodes)
        Restrict the distribution to a set of nodes (only impacts the node
        attribute).
    log : bool, optional (default: False)
        use log-spaced bins.
    num_bins : int, list or str, optional (default: 'bayes')
        Any of the automatic methodes from :func:`numpy.histogram`, or 'bayes'
        will provide automatic bin optimization. Otherwise, an int for the
        number of bins can be provided, or the direct bins list.

    Returns
    -------
    ncounts : :class:`numpy.array`
        number of nodes in each bin
    nbetw : :class:`numpy.array`
        bins for node betweenness
    ecounts : :class:`numpy.array`
        number of edges in each bin
    ebetw : :class:`numpy.array`
        bins for edge betweenness
    '''
    ia_nbetw, ia_ebetw = betweenness(graph, btype="both", weights=weights)

    if nodes is not None:
        ia_nbetw = ia_nbetw[nodes]

    ra_nbins, ra_ebins = None, None

    if num_ebins == 'bayes' or log:
        ra_ebins = binning(ia_ebetw, bins=num_ebins, log=log)
    else:
        ra_ebins = num_ebins

    if num_nbins == 'bayes' or log:
        ra_nbins = binning(ia_nbetw, bins=num_nbins, log=log)
    else:
        ra_nbins = num_nbins

    ncounts, nbetw = np.histogram(ia_nbetw, ra_nbins)
    ecounts, ebetw = np.histogram(ia_ebetw, ra_ebins)

    return ncounts, nbetw, ecounts, ebetw


# ----- #
# Tools #
# ----- #

def binning(x, bins='bayes', log=False):
    """
    Binning function providing automatic binning using Bayesian blocks in
    addition to standard linear and logarithmic uniform bins.

    .. versionadded:: 0.7

    Parameters
    ----------
    x : array-like
        Array of data to be histogrammed
    bins : int, list or 'auto', optional (default: 'bayes')
        If `bins` is 'bayes', in use bayesian blocks for dynamic bin widths; if
        it is an int, the interval will be separated into
    log : bool, optional (default: False)
        Whether the bins should be evenly spaced on a logarithmic scale.
    """
    x = np.asarray(x)
    new_bins = None

    if bins == 'bayes':
        return bayesian_blocks(x)
    elif nonstring_container(bins) or bins == "auto":
        if log:
            ordered = np.sort(x)
            nonzero_min = ordered[ordered > 0][0]
            return np.logspace(np.log10(nonzero_min), np.log10(x.max()), 20)
        return bins
    elif is_integer(bins):
        if log:
            ordered = np.sort(x)
            nonzero_min = ordered[ordered > 0][0]
            return np.logspace(np.log10(nonzero_min), np.log10(x.max()), bins)
        else:
            return np.linspace(x.min(), x.max(), bins)

    raise ValueError("unrecognized bin code: '" + str(bins) + "'.")


def _get_attribute(network, attribute, nodes=None, data=None):
    '''
    If data is not None, must be an np.array of shape (N, 2).
    '''
    if attribute.lower() == "b2":
        return get_b2(network, nodes=nodes, data=data)
    elif attribute == "betweenness":
        betw = network.get_betweenness("node")
        if nodes is not None:
            return betw[nodes]
        return betw
    elif attribute == "closeness":
        return closeness(network, nodes=nodes)
    elif attribute == "clustering":
        return local_clustering(network, nodes=nodes)
    elif "degree" in attribute.lower():
        dtype = attribute[:attribute.index("-")]
        if dtype.startswith("w"):
            return network.get_degrees(
                dtype[1:], nodes=nodes, weights=True)
        else:
            return network.get_degrees(dtype, nodes=nodes)
    elif "strength" in attribute.lower():
        dtype = attribute[:attribute.index("-")]
        return network.get_degrees(dtype, nodes=nodes, weights=True)
    elif attribute == "firing_rate":
        return get_firing_rate(network, nodes=nodes, data=data)
    elif attribute == "subgraph_centrality":
        sc = subgraph_centrality(network)
        if nodes is not None:
            return sc[nodes]
        return sc
    elif attribute in network.node_attributes:
        return network.get_node_attributes(nodes=nodes, name=attribute)

    raise RuntimeError("Attribute '{}' is not available.".format(attribute))


# ------------------------------------ #
# Importing backend-specific functions #
# ------------------------------------ #

if nngt._config["backend"] == "networkx":
    from .nx_functions import *

if nngt._config["backend"] == "igraph":
    from .ig_functions import *

if nngt._config["backend"] == "graph-tool":
    from .gt_functions import *

if nngt._config["backend"] == "nngt":
    from .nngt_functions import *

# update analyze_graph dict
for func in __all__:
    nngt.analyze_graph[func] = locals()[func]

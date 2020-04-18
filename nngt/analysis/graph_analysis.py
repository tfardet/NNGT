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

""" Tools for graph analysis using the graph libraries """

import numpy as np
import scipy.sparse.linalg as spl

import nngt
from nngt.lib import InvalidArgument, nonstring_container, is_integer
from .activity_analysis import get_b2, get_firing_rate
from .bayesian_blocks import bayesian_blocks


__all__ = [
    "adjacency_matrix",
    "assortativity",
    "betweenness_distrib",
    "binning",
	"closeness",
	"global_clustering",
    "degree_distrib",
	"diameter",
    "local_clustering",
    "node_attributes",
	"num_iedges",
	"num_scc",
	"num_wcc",
	"reciprocity",
	"spectral_radius",
    "subgraph_centrality",
    "transitivity"
]


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
    .. [gt-global-clustering] :gtdoc:`clustering.global_clustering`
    .. [ig-global-clustering] :igdoc:`transitivity_undirected`
    .. [nx-global-clustering] :nxdoc:`algorithms.cluster.transitivity`
    '''
    raise NotImplementedError(_backend_required)


def transitivity(g, weights=None):
    '''
    Same as :func:`~nngt.analysis.global_clustering`.
    '''
    return global_clustering(g, weights=weights)


def num_iedges(graph):
    '''
    Returns the number of inhibitory connections.

    For :class:`~nngt.Network` objects, this corresponds to the number of edges
    stemming from inhibitory nodes (given by
    :meth:`~nngt.NeuralPop.inhibitory`).
    Otherwise, counts the edges where the type attribute is -1.
    '''
    if graph.is_network():
        inhib_nodes = graph.population.inhibitory

        return np.sum(graph.get_degrees("out", node_list=inhib_nodes))

    return np.sum(graph.get_edge_attributes(name="type") < 0)


def num_scc(graph, listing=False):
    '''
    Returns the number of strongly connected components (SCCs).
    SCC are ensembles where all contained nodes can reach any other node in
    the ensemble using the directed edges.

    See also
    --------
    num_wcc
    '''
    lst_histo = None

    if nngt._config["backend"] == "graph-tool":
        vprop_comp, lst_histo = nngt.analyze_graph["scc"](graph, directed=True)
    elif nngt._config["backend"] == "igraph":
        lst_histo = graph.graph.clusters()
        lst_histo = [cluster for cluster in lst_histo]
    else:
        lst_histo = [comp for comp in nngt.analyze_graph["scc"](graph)]

    if listing:
        return len(lst_histo), lst_histo

    return len(lst_histo)


def num_wcc(graph, listing=False):
    '''
    Connected components if the directivity of the edges is ignored.
    (i.e. all edges are considered bidirectional).

    See also
    --------
    num_scc
    '''
    lst_histo = None

    if nngt._config["backend"] == "graph-tool":
        _, lst_histo = nngt.analyze_graph["wcc"](graph, directed=False)
    elif nngt._config["backend"] == "igraph":
        lst_histo = graph.graph.clusters("WEAK")
        lst_histo = [cluster for cluster in lst_histo]
    else:
        if listing:
            raise RuntimeError("Not implemented for networkx.")
        return nngt.analyze_graph["wcc"](graph)

    if listing:
        return len(lst_histo), lst_histo

    return len(lst_histo)


def diameter(graph):
    '''
    Pseudo-diameter of the graph

    @todo: weighted diameter
    '''
    if nngt._config["backend"] == "igraph":
        return graph.graph.diameter()
    elif nngt._config["backend"] == "networkx":
        return nngt.analyze_graph["diameter"](graph)

    # graph-tool
    return nngt.analyze_graph["diameter"](graph)[0]


# ------------ #
# Centralities #
# ------------ #

def closeness(graph, nodes=None, use_weights=False):
    '''
    Return the closeness centrality for each node in `nodes`.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` object
        Graph to analyze.
    nodes : array-like container with node ids, optional (default = all nodes)
        Nodes for which the local clustering coefficient should be computed.
    use_weights : bool, optional (default: False)
        Whether weighted closeness should be used.
    '''
    raise NotImplementedError(_backend_required)


def betweenness(g, btype="both", weights=None, norm=True):
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
    .. [betweenness-wikipedia] http://en.wikipedia.org/wiki/Centrality#Betweenness_centrality
    .. [gt-betw] https://graph-tool.skewed.de/static/doc/centrality.html#graph_tool.centrality.betweenness
    .. [ig-betw] https://igraph.org/python/doc/igraph.GraphBase-class.html#betweenness
    .. [ig-edge-betw] https://igraph.org/python/doc/igraph.GraphBase-class.html#edge_betweenness
    .. [nx-ebetw] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html#networkx.algorithms.centrality.edge_betweenness_centrality
    .. [nx-nbetw] https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html#networkx.algorithms.centrality.betweenness_centrality
    '''
    raise NotImplementedError(_backend_required)


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
    .. [gt-local-clustering] :gtdoc:`clustering.locall_clustering`
    .. [ig-local-clustering] :igdoc:`transitivity_local_undirected`
    .. [nx-local-clustering] :nxdoc:`algorithms.cluster.clustering`
    '''
    raise NotImplementedError(_backend_required)


def local_clustering(g, weights=None, nodes=None):
    '''
    Local clustering coefficient of the nodes.

    Parameters
    ----------
    g : :class:`~nngt.Graph` object
        Graph to analyze.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    nodes : array-like container with node ids, optional (default = all nodes)
        Nodes for which the local clustering coefficient should be computed.
    '''
    if not graph.is_directed():
        return undirected_local_clustering(g, weights=weights, nodes=nodes)

    raise NotImplementedError("`local_clustering` is not yet implemented for "
                              "directed graphs.")


def subgraph_centrality(graph, weights=True, normalize="max_centrality"):
    '''
    Subgraph centrality, accordign to [Estrada2005], for each node in the
    graph.

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
    normalize : str, optional (default: "max_centrality")
        Whether the centrality should be normalized. Accepted normalizations
        are "max_eigenvalue" and "max_centrality"; the first rescales the
        adjacency matrix by the its largest eigenvalue before taking the
        exponential, the second sets the maximum centrality to one.

    Returns
    -------
    centralities : :class:`numpy.ndarray`
        The subgraph centrality of each node.

    References
    ----------
    .. [Estrada2005] Ernesto Estrada and Juan A. Rodríguez-Velázquez,
    Subgraph centrality in complex networks, PHYSICAL REVIEW E 71, 056103
    (2005), `available on ArXiv <http://www.arxiv.org/pdf/cond-mat/0504730>`_.
    '''
    adj_mat = graph.adjacency_matrix(types=False, weights=weights).tocsc()

    centralities = None

    if normalize == "max_centrality":
        centralities = spl.expm(adj_mat / adj_mat.max()).diagonal()
        centralities /= centralities.max()
    elif normalize == "max_eigenvalue":
        norm, _ = spl.eigs(adj_mat, k=1)
        centralities = spl.expm(adj_mat / norm).diagonal()
    else:
        raise InvalidArgument('`normalize` should be either False, "eigenmax",'
                              ' or "centralmax".')
    return centralities


# ------------------- #
# Spectral properties #
# ------------------- #

def spectral_radius(graph, typed=True, weighted=True):
    '''
    Spectral radius of the graph, defined as the eigenvalue of greatest module.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Network to analyze.
    typed : bool, optional (default: True)
        Whether the excitatory/inhibitory type of the connnections should be
        considered.
    weighted : bool, optional (default: True)
        Whether the weights should be taken into account.

    Returns
    -------
    the spectral radius as a float.
    '''
    mat_adj  = graph.adjacency_matrix(types=typed, weights=weighted)
    eigenval = []

    try:
        eigenval = spl.eigs(mat_adj, return_eigenvectors=False)
    except spl.eigen.arpack.ArpackNoConvergence as err:
        eigenval = err.eigenvalues

    if len(eigenval):
        return np.amax(np.absolute(eigenval))

    raise spl.eigen.arpack.ArpackNoConvergence()


def adjacency_matrix(graph, types=True, weights=True):
    '''
    Adjacency matrix of the graph.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Network to analyze.
    types : bool, optional (default: True)
        Whether the excitatory/inhibitory type of the connnections should be
        considered (only if the weighing factor is the synaptic strength).
    weights : bool or string, optional (default: True)
        Whether weights should be taken into account; if True, then connections
        are weighed by their synaptic strength, if False, then a binary matrix
        is returned, if `weights` is a string, then the ponderation is the
        correponding value of the edge attribute (e.g. "distance" will return
        an adjacency matrix where each connection is multiplied by its length).

    Returns
    -------
    a :class:`~scipy.sparse.csr_matrix`.
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

    .. versionchanged:: 0.7

    Inclusion of automatic binning.

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
    degrees = graph.get_degrees(deg_type, node_list, use_weights)

    if num_bins == 'bayes' or is_integer(num_bins):
        num_bins = binning(degrees, bins=num_bins, log=log)

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
    ia_nbetw, ia_ebetw = betweenness(graph, btype="both", weights=use_weights)

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

    ra_ebins = binning(ia_ebetw, bins=num_ebins, log=log)

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
        return bins
    elif is_integer(bins):
        if log:
            return np.logspace(np.log10(np.maximum(x.min(), 1e-10)),
                               np.log10(x.max()), bins)
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
                dtype[1:], node_list=nodes, use_weights=True, use_types=True)
        else:
            return network.get_degrees(dtype, node_list=nodes)
    elif attribute == "firing_rate":
        return get_firing_rate(network, nodes=nodes, data=data)
    elif attribute == "subgraph_centrality":
        sc = subgraph_centrality(network)
        if nodes is not None:
            return sc[nodes]
        return sc
    elif attribute in network.nodes_attributes:
        return network.get_node_attributes(nodes=nodes, name=attribute)

    raise RuntimeError("Attribute '{}' is not available.".format(attribute))

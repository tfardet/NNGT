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

import scipy.sparse as ssp


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

    References
    ----------
    .. [ig-global-clustering] https://igraph.org/python/doc/igraph.GraphBase-class.html#transitivity_undirected
    '''
    ww = _get_weights(g, weights)

    if ww is not None:
        # raise warning
        return np.average(g.graph.transitivity_local_undirected(weights=ww))

    return np.array(g.graph.transitivity_undirected())


def local_clustering(g, weights=False, nodes=None):
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
    .. [ig-local-clustering] https://igraph.org/python/doc/igraph.GraphBase-class.html#transitivity_local_undirected
    '''
    ww = _get_weights(g, weights)

    return np.array(g.graph.transitivity_local_undirected(nodes, weights=ww))


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
    .. [gt-assortativity] https://graph-tool.skewed.de/static/doc/correlations.html#graph_tool.correlations.scalar_assortativity
    '''
    use_weights = True if weights in (True, 'weight') else False

    if not use_weights:
        raise RuntimeError("igraph backend does not support specific attributes "
        "as weights yet, will come soon.")

    degrees = graph.get_degrees(deg_type=deg_type, use_weights=use_weights)

    return g.graph.assortativity(degrees, directed=graph.is_directed())


def reciprocity(g):
    '''
    Calculate the edge reciprocity of the graph.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.

    References
    ----------
    .. [gt-reciprocity] https://graph-tool.skewed.de/static/doc/topology.html#graph_tool.topology.edge_reciprocity
    '''
    return edge_reciprocity(g.graph)


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

    References
    ----------
    .. [ig-closeness] https://igraph.org/python/doc/igraph.GraphBase-class.html#closeness
    '''
    ww = _get_weights(g, weights)

    return g.graph.closeness(nodes, mode="out", weights=ww)


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
    .. [gt-connected-components] https://graph-tool.skewed.de/static/doc/topology.html#graph_tool.topology.label_components
    '''
    cc, hist = label_components(g.graph, *args, **kwargs)
    return cc.a, hist


def diameter(g, weighted=False):
    '''
    Returns the pseudo-diameter of the graph.

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
    .. [gt-diameter] https://graph-tool.skewed.de/static/doc/topology.html#graph_tool.topology.pseudo_diameter
    '''
    ww = _get_weights(g, weights)

    return pseudo_diameter(g.graph, weights=ww)


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

    if g.edge_nb():
        xs, ys = map(np.array, zip(*g.graph.get_edgelist()))
        xs, ys = xs.T, ys.T

        data = np.ones(xs.shape) * _get_weights(g, weights)

        coo_adj = ssp.coo_matrix((data, (xs, ys)), shape=(n,n))

        return coo_adj.tocsr()

    return ssp.csr_matrix((n,n))


def get_edges(g):
    '''
    Returns the edges in the graph by order of creation.
    '''
    return g.graph.get_edgelist()


def _get_weights(g, weights):
    if weights in g.edges_attributes:
        # existing edge attribute
        return np.array(g.graph.es[weights])
    elif nonstring_container(weights):
        # user-provided array
        return np.array(weights)
    elif weights is True:
        # "normal" weights
        return np.array(g.graph.es["weight"])
    elif not weights:
        # unweighted
        return None

    raise ValueError("Unknown edge attribute '" + str(weights) + "'.")
    

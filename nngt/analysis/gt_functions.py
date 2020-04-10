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

from graph_tool.spectral import adjacency
from graph_tool.centrality import closeness as gt_closeness
from graph_tool.correlations import scalar_assortativity
from graph_tool.topology import (edge_reciprocity,
                                 label_components, pseudo_diameter)
from graph_tool.clustering import global_clustering as gt_gc
from graph_tool.clustering import local_clustering as gt_lc


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
    .. [gt-global-clustering] https://graph-tool.skewed.de/static/doc/clustering.html#graph_tool.clustering.global_clustering
    '''
    ww = _get_weights(g, weights)

    return gt_gc(g.graph, weight=ww)[0]


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
    .. [gt-local-clustering] https://graph-tool.skewed.de/static/doc/clustering.html#graph_tool.clustering.local_clustering
    '''
    ww = _get_weights(g, weights)

    lc = gt_lc(g.graph, weight=ww).a

    if nodes is None:
        return lc

    return lc[nodes]


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
    ww = _get_weights(g, weights)

    return scalar_assortativity(g.graph, degree, eweight=ww)


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
    .. [gt-closeness] https://graph-tool.skewed.de/static/doc/centrality.html#graph_tool.centrality.closeness
    '''
    ww = _get_weights(g, weights)

    c = closeness(g.graph, weight=ww)

    if nodes is None:
        return c.a

    return c.a[nodes]


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

    Returns
    -------
    The adjacency matrix as a :class:`scipy.sparse.csr_matrix`.

    References
    ----------
    .. [gt-adjacency] https://graph-tool.skewed.de/static/doc/spectral.html#graph_tool.spectral.adjacency
    '''
    ww = _get_weights(g, weights)

    return _adj(g.graph, ww).T


def get_edges(g):
    '''
    Returns the edges in the graph by order of creation.
    '''
    return g.graph.edges()


def _get_weights(g, weights):
    if weights in g.edges_attributes:
        # existing edge attribute
        return g.graph.edge_properties[weights]
    elif nonstring_container(weights):
        # user-provided array
        return g.graph.new_edge_property("double", vals=weights)
    elif weights is True:
        # "normal" weights
        return g.graph.edge_properties['weight']
    elif not weights:
        # unweighted
        return None

    raise ValueError("Unknown edge attribute '" + str(weights) + "'.")
    

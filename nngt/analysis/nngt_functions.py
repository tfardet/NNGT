#-*- coding:utf-8 -*-
#
# nngt_functions.py
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

""" Tools to analyze graphs with the nngt backend """

import numpy as np
import scipy.sparse as ssp


def adj_mat(g, weight=None, mformat="csr"):
    data = None

    if weight in g.edge_attributes:
        data = g.get_edge_attributes(name=weight)
    else:
        data = np.ones(g.edge_nb())

    if not g.is_directed():
        data = np.repeat(data, 2)
        
    edges     = np.array(list(g._graph._edges), dtype=int)
    num_nodes = g.node_nb()
    mat       = ssp.coo_matrix((data, (edges[:, 0], edges[:, 1])),
                               shape=(num_nodes, num_nodes))

    return mat.asformat(mformat)


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
    if not g.is_directed():
        return 1.

    num_edges = g.edge_nb()

    g = g._graph

    num_recip = sum((1 if e[::-1] in g._edges else 0 for e in g._edges))

    return num_recip / num_edges

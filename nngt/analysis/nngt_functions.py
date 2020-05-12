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

#-*- coding:utf-8 -*-
#
# analysis/nngt_functions.py
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

""" Tools to analyze graphs with the nngt backend """

from collections import deque

import numpy as np
import scipy.sparse as ssp


def adj_mat(g, weight=None, mformat="csr"):
    data = None

    num_nodes = g.node_nb()
    num_edges = g.edge_nb()

    if weight in g.edge_attributes:
        data = g.get_edge_attributes(name=weight)
    else:
        data = np.ones(num_edges)

    if not g.is_directed():
        data = np.repeat(data, 2)
        
    edges = np.array(list(g._graph._edges), dtype=int)
    edges = (edges[:, 0], edges[:, 1]) if num_edges else [[], []]

    mat = ssp.coo_matrix((data, edges), shape=(num_nodes, num_nodes))

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
    ctype = "scc" if ctype is None else ctype

    num_nodes = g.node_nb()

    all_nodes = set(g.get_nodes())

    all_seen = set()

    components = []

    # get adjacency matrix
    A = g.adjacency_matrix()

    if ctype == "wcc" and g.is_directed():
        A = A + A.T

    start_nodes = []

    while len(all_seen) < num_nodes:
        start = next(iter(all_nodes.difference(all_seen)))

        start_nodes.append(start)

        visited = _dfs(A, start)

        all_seen.update(visited)

        components.append(visited)

    # do reverse search
    if ctype == "scc" and g.is_directed():
        A = A.T

        count = 0

        components2 = []

        all_seen = set()

        while len(all_seen) < num_nodes:
            start = start_nodes[count] if count < len(start_nodes) - 1 \
                    else next(iter(all_nodes.difference(all_seen)))

            visited = _dfs(A, start)

            all_seen.update(visited)

            components2.append(visited)

            count += 1

        # generate sccs
        components1 = components.copy()

        components = []

        # sort components by size
        order1 = np.argsort([len(s) for s in components1])
        order2 = np.argsort([len(s) for s in components2])

        for i in order1:
            s1 = components1[i]

            if len(s1) == 1:
                components.append(s1)
                for j in order1:
                    components1[j] = components1[j].difference(s1)

                for j in order2:
                    components2[j] = components2[j].difference(s1)
            else:
                for j in order2:
                    s2 = components2[j]

                    intsct = s1.intersection(s2)

                    if intsct:
                        components.append(intsct)

                        for k in order1:
                            components1[k] = components1[k].difference(intsct)

                        for k in order2:
                            components2[k] = components2[k].difference(intsct)

    # make labels and histogram
    labels = np.zeros(num_nodes, dtype=int)

    hist = np.array([len(s) for s in components], dtype=int)

    order = np.argsort(hist)[::-1]

    for i, s in zip(order, components):
        labels[list(s)] = i

    return labels, hist[order]


def _dfs(adjacency, start):
    '''
    Depth-first search returning all nodes that can be reached from a
    '''
    todo  = deque([start])

    visited = {start}

    while todo:
        n = todo.popleft()

        neighbours = np.where(adjacency.getrow(n).todense().A1)[0]
        todo.extend([v for v in neighbours if v not in visited])

        visited.update(neighbours)

    return visited

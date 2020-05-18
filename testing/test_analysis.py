#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Test the graph analysis functions """

import os

import numpy as np
import pytest

import nngt
import nngt.analysis as na
import nngt.generation as ng


methods = ('barrat', 'continuous', 'onella')


@pytest.mark.mpi_skip
def test_binary_undirected_clustering():
    '''
    Check that directed local clustering returns undirected value if graph is
    not directed.
    '''
    g = ng.erdos_renyi(avg_deg=10, nodes=100, directed=False)

    ccu = na.local_clustering_binary_undirected(g)
    cc  = na.local_clustering(g)

    assert np.all(np.isclose(cc, ccu))


@pytest.mark.mpi_skip
def test_weighted_undirected_clustering():
    '''
    Check relevant properties of weighted clustering:

    * give back the binary definition if all weights are one
    * corner cases for specific networks, see [Saramaki2007]
    * equivalence between no edge and zero-weight edge for 'continuous' method

    Note: onella and barrat are already check against networkx and igraph
    implementations in libarry_compatibility.py
    '''
    g = ng.erdos_renyi(avg_deg=10, nodes=100, directed=False)

    # recover binary
    ccb = na.local_clustering_binary_undirected(g)

    for method in methods:
        ccw = na.local_clustering(g, weights='weight', method=method)

        assert np.all(np.isclose(ccb, ccw))

    # corner cases
    eps = 1e-20

    # 3 nodes
    num_nodes = 3
    edge_list = [(0, 1), (1, 2), (2, 0)]

    # all epsilon
    weights = [eps, eps, eps]

    g = nngt.Graph(nodes=num_nodes, directed=False)
    g.new_edges(edge_list, attributes={"weight": weights})

    for method in methods:
        cc = na.local_clustering(g, weights='weight', method=method)
        assert np.array_equal(cc, [1, 1, 1])

    # one weight is one
    g.set_weights(np.array([eps, eps, 1]))

    for method in methods:
        cc = na.local_clustering(g, weights='weight', method=method)

        if method == "barrat":
            assert np.all(np.isclose(cc, 1))
        else:
            assert np.all(np.isclose(cc, 0))

    # two weights are one
    g.set_weights(np.array([eps, eps, 1]))

    for method in methods:
        cc = na.local_clustering(g, weights='weight', method=method)

        if method == "barrat":
            assert np.all(np.isclose(cc, 1))
        else:
            assert np.all(np.isclose(cc, 0))

    # 4 nodes
    num_nodes = 4
    edge_list = [(0, 1), (1, 2), (2, 0), (2, 3)]

    g = nngt.Graph(nodes=num_nodes, directed=False)
    g.new_edges(edge_list)

    # out of triangle edge is epsilon
    g.set_weights([1, 1, 1, eps])

    for method in methods:
        cc = na.local_clustering(g, weights='weight', method=method)

        if method == 'barrat':
            assert np.all(np.isclose(cc, [1, 1, 0.5, 0]))
        elif method == "continuous":
            assert np.all(np.isclose(cc, [1, 1, 1, 0]))
        else:
            assert np.all(np.isclose(cc, [1, 1, 1/3, 0]))

    # out of triangle edge is 1 others are epsilon
    g.set_weights([eps, eps, eps, 1])

    for method in methods:
        cc = na.local_clustering(g, weights='weight', method=method)

        if method == 'barrat':
            assert np.all(np.isclose(cc, [1, 1, 0, 0]))
        else:
            assert np.all(np.isclose(cc, 0))

    # opposite triangle edge is 1 others are epsilon
    g.set_weights([1, eps, eps, eps])

    for method in methods:
        cc = na.local_clustering(g, weights='weight', method=method)

        if method == 'barrat':
            assert np.all(np.isclose(cc, [1, 1, 1/3, 0]))
        else:
            assert np.all(np.isclose(cc, 0))

    # adjacent triangle edge is 1 others are epsilon
    g.set_weights([eps, 1, eps, eps])

    for method in methods:
        cc = na.local_clustering(g, weights='weight', method=method)

        if method == 'barrat':
            assert np.all(np.isclose(cc, [1, 1, 1/2, 0]))
        else:
            assert np.all(np.isclose(cc, 0))

    # check zero-weight edge/no edge equivalence for continuous method
    num_nodes = 6
    edge_list = [(0, 1), (1, 2), (2, 0), (2, 3), (4, 5)]

    g = nngt.Graph(nodes=num_nodes, directed=False)
    g.new_edges(edge_list)

    g.set_weights([1/4, 1/9, 1/4, 1/9, 1])

    expected = [1/36, 1/24, 1/64, 0, 0, 0]

    cc = na.local_clustering(g, weights='weight', method='continuous')

    assert np.all(np.isclose(cc, expected))

    # 0-weight case
    g.set_weights([1/4, 1/9, 1/4, 0, 1])

    cc0 = na.local_clustering(g, weights='weight', method='continuous')

    # no-edge case
    edge_list = [(0, 1), (1, 2), (2, 0), (4, 5)]

    g = nngt.Graph(nodes=num_nodes, directed=False)
    g.new_edges(edge_list)
    g.set_weights([1/4, 1/9, 1/4, 1])

    expected = [1/36, 1/24, 1/24, 0, 0, 0]

    ccn = na.local_clustering(g, weights='weight', method='continuous')

    assert np.all(np.isclose(cc0, ccn))
    assert np.all(np.isclose(cc0, expected))
    


if __name__ == "__main__":
    if not nngt.get_config("mpi"):
        test_binary_undirected_clustering()
        test_weighted_undirected_clustering()

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


nngt_backend = (nngt.get_config("backend") == "nngt")

methods = ('barrat', 'continuous', 'onnela', 'zhang')


@pytest.mark.mpi_skip
def test_binary_undirected_clustering():
    '''
    Check that directed local clustering returns undirected value if graph is
    not directed.
    '''
    # pre-defined graph
    num_nodes = 5
    edge_list = [
        (0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 2), (4, 0)
    ]

    # expected results
    loc_clst  = [2/3., 2/3., 1., 1., 0.5]
    glob_clst = 0.6428571428571429

    # create graph
    g = nngt.Graph(nodes=num_nodes)
    g.new_edges(edge_list)

    # check all 3 ways of computing the local clustering
    assert np.all(np.isclose(
        na.local_clustering_binary_undirected(g), loc_clst))

    assert np.all(np.isclose(
        na.local_clustering(g, directed=False), loc_clst))

    assert np.all(np.isclose(
        nngt.analyze_graph["local_clustering"](g, directed=False),
        loc_clst))

    # check all 4 ways of computing the global clustering
    assert np.isclose(
        na.global_clustering(g, directed=False), glob_clst)

    assert np.isclose(na.transitivity(g, directed=False), glob_clst)

    assert np.isclose(
        na.global_clustering_binary_undirected(g), glob_clst)

    assert np.isclose(
        nngt.analyze_graph["global_clustering"](g, directed=False),
        glob_clst)

    # check that self-loops are ignore
    g.new_edge(0, 0, self_loop=True)

    assert np.all(np.isclose(
        na.local_clustering_binary_undirected(g), loc_clst))

    assert np.isclose(
        na.global_clustering_binary_undirected(g), glob_clst)

    # sanity check for local clustering on undirected unweighted graph
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

    Note: onnela and barrat are already check against networkx and igraph
    implementations in libarry_compatibility.py
    '''
    g = ng.erdos_renyi(avg_deg=10, nodes=100, directed=False)

    # recover binary
    ccb = na.local_clustering_binary_undirected(g)

    for method in methods:
        ccw = na.local_clustering(g, weights='weight', method=method)

        assert np.all(np.isclose(ccb, ccw))

    # corner cases
    eps = 1e-15

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
        elif method == "zhang":
            assert np.all(np.isclose(cc, [0, 1, 0]))
        else:
            assert np.all(np.isclose(cc, 0, atol=1e-4))

    # two weights are one
    g.set_weights(np.array([eps, 1, 1]))

    for method in methods:
        cc = na.local_clustering(g, weights='weight', method=method)

        if method == "barrat":
            assert np.all(np.isclose(cc, 1))
        elif method == "zhang":
            # due to floating point rounding errors, we get 0.9 instead of 1
            assert np.all(np.isclose(cc, (0.9, 0.9, 0), atol=1e-3))
        else:
            assert np.all(np.isclose(cc, 0, atol=1e-2))

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
        elif method in ("continuous", "zhang"):
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
        elif method == "zhang":
            assert np.all(np.isclose(cc, [0, 0, 1/3, 0]))
        else:
            assert np.all(np.isclose(cc, 0, atol=1e-5))

    # adjacent triangle edge is 1 others are epsilon
    g.set_weights([eps, 1, eps, eps])

    for method in methods:
        cc = na.local_clustering(g, weights='weight', method=method)

        if method == 'barrat':
            assert np.all(np.isclose(cc, [1, 1, 1/2, 0]))
        elif method == "zhang":
            assert np.all(np.isclose(cc, [1, 0, 0, 0]))
        else:
            assert np.all(np.isclose(cc, 0, atol=1e-4))

    # check zero-weight edge/no edge equivalence for continuous method
    num_nodes = 6
    edge_list = [(0, 1), (1, 2), (2, 0), (2, 3), (4, 5)]

    g = nngt.Graph(nodes=num_nodes, directed=False)
    g.new_edges(edge_list)

    g.set_weights([1/64, 1/729, 1/64, 1/729, 1])

    # triangle is 1/20736
    # triplets are [1/64, 1/216, 62/5832, 0, 0, 0]
    expected = [1/324, 1/96, 9/1984, 0, 0, 0]

    cc = na.local_clustering(g, weights='weight', method='continuous')

    assert np.all(np.isclose(cc, expected))

    # 0-weight case
    g.set_weights([1/64, 1/729, 1/64, 0, 1])

    cc0 = na.local_clustering(g, weights='weight', method='continuous')

    # no-edge case
    edge_list = [(0, 1), (1, 2), (2, 0), (4, 5)]

    g = nngt.Graph(nodes=num_nodes, directed=False)
    g.new_edges(edge_list)
    g.set_weights([1/64, 1/729, 1/64, 1])

    expected = [1/324, 1/96, 1/96, 0, 0, 0]

    ccn = na.local_clustering(g, weights='weight', method='continuous')

    assert np.all(np.isclose(cc0, ccn))
    assert np.all(np.isclose(cc0, expected))


@pytest.mark.mpi_skip
def test_weighted_directed_clustering():
    num_nodes = 6
    edge_list = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 3), (4, 5)]

    g = nngt.Graph(nodes=num_nodes)
    g.new_edges(edge_list)

    # continuous
    g.set_weights([1/64, 1/729, 1/64, 1/729, 1/64, 1/729, 1])

    # expected triangles and triplets
    # s_sqrt_tot  = np.array([89/216, 31/108, 51/216, 1/27, 1, 1])
    # s_tot       = np.array([2251/46656, 761/23328, 307/15552, 1/729, 1, 1])
    # s_recip     = np.array([35/1728, 1/64, 1/216, 0, 0, 0])
    # triplets_c  = np.square(s_sqrt_tot) - s_tot - 2*s_recip
    triangles_c = np.array([97/839808, 97/839808, 97/839808, 0, 0, 0])
    triplets_c  = np.array([35/432, 1/54, 13/486, 0, 0, 0])

    assert np.all(np.isclose(
        triangles_c,
        na.triangle_count(g, weights='weight', method='continuous')))

    assert np.all(np.isclose(
        triplets_c,
        na.triplet_count(g, weights='weight', method='continuous')))

    triplets_c[-3:] = 1
    expected = triangles_c / triplets_c

    cc = na.local_clustering(g, weights='weight', method='continuous')

    assert np.all(np.isclose(cc, expected))

    # barrat (clemente version for reciprocal strength)
    g.set_weights([1/4, 1/9, 1/4, 1/9, 1/4, 1/9, 1])

    # d_tot       = np.array([4, 3, 4, 1, 1, 1])
    # s_recip     = np.array([31/72, 1/4, 0, 0, 0])
    # triplets_b  = s_tot*(d_tot - 1) - s_recip
    triangles_b = np.array([31/36, 13/18, 7/12, 0, 0, 0])
    triplets_b  = np.array([31/18, 13/18, 25/18, 0, 0, 0])

    assert np.all(np.isclose(
        triangles_b, na.triangle_count(g, weights='weight', method='barrat')))

    assert np.all(np.isclose(
        triplets_b, na.triplet_count(g, weights='weight', method='barrat')))

    triplets_b[-3:] = 1
    expected = triangles_b / triplets_b

    cc = na.local_clustering(g, weights='weight', method='barrat')

    assert np.all(np.isclose(cc, expected))

    # onnela
    triplets_o  = np.array([8, 4, 10, 0, 0, 0])
    triangles_o = np.array(
        [0.672764902429877, 0.672764902429877, 0.672764902429877, 0, 0, 0])

    assert np.array_equal(
        triplets_o, na.triplet_count(g, weights='weight', method='onnela'))

    assert np.all(np.isclose(
        triangles_o, na.triangle_count(g, weights='weight', method="onnela")))
    
    triplets_o[-3:] = 1
    expected = triangles_o / triplets_o

    cc = na.local_clustering(g, weights='weight', method='onnela')

    assert np.all(np.isclose(cc, expected))

    # zhang
    g.set_weights([1/2, 1/3, 1/2, 1/3, 1/2, 1/3, 1])

    triangles_z = np.array([5/18, 5/18, 5/18, 0, 0, 0])
    triplets_z  = np.array([5/3, 2/3, 4/3, 0, 0, 0])

    assert np.all(np.isclose(
        triplets_z, na.triplet_count(g, weights='weight', method='zhang')))

    assert np.all(np.isclose(
        triangles_z, na.triangle_count(g, weights='weight', method="zhang")))
    
    triplets_z[-3:] = 1
    expected = triangles_z / triplets_z

    cc = na.local_clustering(g, weights='weight', method='zhang')

    assert np.all(np.isclose(cc, expected))


@pytest.mark.mpi_skip
def test_clustering_parameters():
    W = np.array([
        [0, 1, 1, 1.],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ])

    g = nngt.Graph.from_matrix(W, directed=True)

    cc_bu = na.local_clustering_binary_undirected(g)

    for m in methods:
        # combine
        for combine in ("sum", "max", "min", "mean"):
            cc_wu = na.local_clustering(g, directed=False, weights=True,
                                        combine_weights=combine, method=m)

            assert np.all(np.isclose(cc_wu, cc_bu))

        # subset of nodes
        nodes = [1, [1, 2]]

        for n in nodes:
            cc = na.local_clustering_binary_undirected(g, nodes=n)

            assert np.all(np.isclose(cc, cc_bu[n]))

            cc = na.local_clustering(g, nodes=n, method=m)

            assert np.all(np.isclose(cc, cc_bu[n]))

    # check different but equivalent matrices
    for m in methods:
        W = np.array([
            [0, 1, 1, 0.6],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0.4, 0, 0, 0]
        ])

        g = nngt.Graph.from_matrix(W, directed=True)

        cc_sum = na.local_clustering(g, directed=False, weights="weight",
                                     combine_weights="mean", method=m)

        W = np.array([
            [0, 1, 1, 0.5],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0.5, 0, 0, 0]
        ])

        g = nngt.Graph.from_matrix(W, directed=False)

        cc_sym = na.local_clustering(g, weights="weight", method=m)

        assert np.all(np.isclose(cc_sum, cc_sym))

        # subset of nodes
        nodes = [1, [1, 2]]

        cc_dir = na.local_clustering(g, weights="weight", method=m)

        for n in nodes:
            cc = na.local_clustering(g, nodes=n, weights="weight", method=m)

            assert np.all(np.isclose(cc, cc_dir[n]))


@pytest.mark.mpi_skip
def test_partial_directed_clustering():
    # cycle
    num_nodes = 3
    edge_list = [
        (0, 1), (1, 2), (2, 0)
    ]

    g = nngt.Graph(num_nodes)
    g.new_edges(edge_list)

    for m in methods:
        assert np.array_equal(
            na.local_clustering(g, mode="cycle", method=m), [1, 1, 1])

        assert np.array_equal(
            na.local_clustering(g, mode="cycle", method=m, weights="weight"),
            [1, 1, 1])

    # middleman, fan-in, fan-out
    num_nodes = 3
    edge_list = [
        (0, 1), (0, 2), (1, 2)
    ]

    g = nngt.Graph(num_nodes)
    g.new_edges(edge_list)

    for m in methods:
        assert np.array_equal(
            na.local_clustering(g, mode="cycle", method=m), [0, 0, 0])

        assert np.array_equal(
            na.local_clustering(g, mode="cycle", method=m, weights="weight"),
            [0, 0, 0])

        assert np.array_equal(
            na.local_clustering(g, mode="middleman", method=m), [0, 1, 0])

        assert np.array_equal(
            na.local_clustering(g, mode="middleman", method=m,
                                weights="weight"),
            [0, 1, 0])

        assert np.array_equal(
            na.local_clustering(g, mode="fan-in", method=m), [0, 0, 0.5])

        assert np.array_equal(
            na.local_clustering(g, mode="fan-in", method=m, weights="weight"),
            [0, 0, 0.5])

        assert np.array_equal(
            na.local_clustering(g, mode="fan-out", method=m), [0.5, 0, 0])

        assert np.array_equal(
            na.local_clustering(g, mode="fan-out", method=m, weights="weight"),
            [0.5, 0, 0])

    # complex graph
    g = nngt.Graph(4)
    g.new_edges([(0, 3), (1, 0), (1, 2), (1, 3), (2, 1), (3, 2)])


    for m in methods:
        assert np.array_equal(
            na.local_clustering(g, mode="cycle", method=m), [0, 0.5, 1, 0.5])

        assert np.array_equal(
            na.local_clustering(g, mode="middleman", method=m), [1, 0, 0, 0.5])

        assert np.array_equal(
            na.local_clustering(g, mode="fan-in", method=m), [0, 0, 0.5, 0.5])

        assert np.array_equal(
            na.local_clustering(g, mode="fan-out", method=m), [0, 1/3, 0, 0])

    # weighted (barrat)
    g.set_weights([1, 0.5, 0.4, 1, 0.6, 1])

    assert np.all(np.isclose(
        na.local_clustering(g, mode="fan-out", weights="weight",
                            method="barrat"),
        [0, 1.45/3.8, 0, 0]))

    assert np.array_equal(
        na.local_clustering(g, mode="fan-in", weights="weight",
                            method="barrat"), [0, 0, 0.5, 0.5])

    assert np.all(np.isclose(
        na.local_clustering(g, mode="cycle", weights="weight", method="barrat"),
        [0, 0.8/1.35, 1, 0.5]))

    assert np.array_equal(
        na.local_clustering(g, mode="middleman", weights="weight",
                            method="barrat"), [1, 0, 0, 0.5])

    # weighted (onnela)
    g.set_weights([1, 1/8, 1, 1, 1, 1/8])

    assert np.array_equal(
        na.local_clustering(g, mode="middleman", weights="weight",
                            method="onnela"), [0.5, 0, 0, 0.25])

    assert np.array_equal(
        na.local_clustering(g, mode="cycle", weights="weight", method="onnela"),
        [0, 0.25, 0.5, 0.25])

    assert np.all(np.isclose(
        na.local_clustering(g, mode="fan-out", weights="weight",
                            method="onnela"), [0, 1/6, 0, 0]))

    assert np.array_equal(
        na.local_clustering(g, mode="fan-in", weights="weight",
                            method="onnela"), [0, 0, 0.25, 0.25])

    # weighted (continuous)
    g.set_weights([1, 1/64, 1/64, 1, 1, 1])

    assert np.all(np.isclose(
        na.local_clustering(g, mode="fan-out", weights="weight"),
        [0, 4/17, 0, 0]))

    assert np.all(np.isclose(
        na.local_clustering(g, mode="fan-in", weights="weight"),
        [0, 0, 0.25, 1/32]))

    assert np.all(np.isclose(
        na.local_clustering(g, mode="cycle", weights="weight"),
        [0, 8/9, 1, 0.5]))

    assert np.all(np.isclose(
        na.local_clustering(g, mode="middleman", weights="weight"),
        [0.5, 0, 0, 1/32]))

    # weighted (zhang)
    g.set_weights([1, 0.25, 0.5, 1, 1, 0.5])

    assert np.all(np.isclose(
        na.local_clustering(g, mode="fan-out", weights="weight", method="zhang"),
        [0, 2/7, 0, 0]))

    assert np.all(np.isclose(
        na.local_clustering(g, mode="fan-in", weights="weight", method="zhang"),
        [0, 0, 0.5, 0.125]))

    assert np.all(np.isclose(
        na.local_clustering(g, mode="cycle", weights="weight", method="zhang"),
        [0, 2/5, 1, 0.5]))

    assert np.all(np.isclose(
        na.local_clustering(g, mode="middleman", weights="weight", method="zhang"),
        [1, 0, 0, 0.25]))


@pytest.mark.mpi_skip
def test_global_clustering():
    num_nodes = 3

    g = nngt.Graph(num_nodes)
    g.new_edges([(0, 1), (0, 2), (2, 1)])

    with pytest.raises(AssertionError):
        na.global_clustering(g, method='plop')

    modes   = ('total', 'cycle', 'middleman', 'fan-in', 'fan-out')
    results = (0.5, 0, 1, 0.5, 0.5)

    for m in methods:
        for mode, res in zip(modes, results):
            gc = na.global_clustering(g, method=m, mode=mode)

            assert gc == res


@pytest.mark.mpi_skip
def test_reciprocity():
    ''' Check reciprocity result '''
    num_nodes  = 5
    edge_list1 = [(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 2)]
    edge_list2 = [(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3)]

    recip = 2/7

    # directed
    g = nngt.Graph(nodes=num_nodes, directed=True)
    g.new_edges(edge_list1)

    assert np.isclose(
        nngt.analyze_graph["reciprocity"](g), recip)

    # undirected
    g = nngt.Graph(nodes=num_nodes, directed=False)
    g.new_edges(edge_list2)

    assert nngt.analyze_graph["reciprocity"](g) == 1.


@pytest.mark.mpi_skip
def test_iedges():
    ''' Check the computation of the number of inhibitory edges '''
    num_nodes = 5
    edge_list = [(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3)]

    types = [1, -1, 1, 1, -1, 1]

    for directed in (True, False):
        g = nngt.Graph(nodes=num_nodes, directed=directed)
        g.new_edges(edge_list)

        # from list
        g.set_types(types)

        assert na.num_iedges(g) == 2

        # from node list
        nodes = [1, 2]
        num_inhib = 3

        g.set_types(-1, nodes=nodes)

        assert na.num_iedges(g) == 3

        # from edge fraction
        g.set_types(-1, fraction=0.5)

        assert na.num_iedges(g) == 3


@pytest.mark.mpi_skip
@pytest.mark.skipif(nngt_backend, reason="Not implemented")
def test_swp():
    ''' Check small-world propensity '''
    num_nodes = 500
    k_latt = 20

    # SWP for different extreme p values (0, 1)
    expected = 1 - 1/np.sqrt(2)

    weights  = {"distribution": "uniform", "lower": 0.5, "upper": 5}

    for directed in (True, False):
        for w in (None, weights):
            for p in (0, 1):
                use_weights = None if w is None else "weight"

                if use_weights and not p:
                    # for weighted SWP, it only works for the ER graph limit,
                    # not for the lattice limit since the weights are uniform
                    continue

                g = ng.watts_strogatz(k_latt, p, nodes=num_nodes,
                                      directed=directed, weights=w)

                swp = na.small_world_propensity(
                    g, use_diameter=True, weights=use_weights)

                if w is None:
                    assert np.isclose(swp, expected, atol=0.01)
                else:
                    assert np.isclose(swp, expected, atol=0.02)

    # check options for binary only
    g = ng.watts_strogatz(k_latt, 0, nodes=num_nodes, directed=True)

    assert np.isclose(
        na.small_world_propensity(g, use_global_clustering=False),
        expected, atol=0.01)

    assert np.isclose(
        na.small_world_propensity(g, use_diameter=False), expected, atol=0.01)

    # run a random version just to check with remaining edges in the
    # algorithm
    for directed in (False, True):
        g = ng.newman_watts(10, 0.03, nodes=num_nodes, directed=directed)
        na.small_world_propensity(g, use_diameter=True)

    # check that undirected option works
    g = ng.newman_watts(10, 0.03, nodes=num_nodes, weighted=False)
    u = g.to_undirected()

    swp_g = na.small_world_propensity(g, use_diameter=True, directed=False)
    swp_u = na.small_world_propensity(u, use_diameter=True)

    if not (np.isnan(swp_g) or np.isnan(swp_u)):
        # check is necessary as the rewired graph might be unconnected
        assert np.isclose(swp_g, swp_u, 0.02)


@pytest.mark.mpi_skip
def test_local_closure():
    # undirected
    g = nngt.Graph(4, directed=False)

    g.new_edges([(0, 1), (0, 3), (1, 2), (1, 3)])

    assert np.all(np.isclose(na.local_closure(g), [2/3, 1, 0, 2/3]))

    # undirected weighted (normal/zhang)
    g.set_weights([1, 0.5, 0.25, 1])

    assert np.all(np.isclose(
        na.local_closure(g, weights="weight"),
        [4/7, 1, 0, 4/7]))

    # check weight normalization
    g.set_weights(2*np.array([1, 0.5, 0.25, 1]))

    assert np.all(np.isclose(
        na.local_closure(g, weights="weight"),
        [4/7, 1, 0, 4/7]))

    # undirected weighted (continuous)
    g.set_weights([1, 1/64, 1/4, 1])

    assert np.all(np.isclose(
        na.local_closure(g, weights="weight", method="continuous"),
        [1/13, 0.5, 0, 1/13]))

    # directed (circle)
    g = nngt.Graph(3)

    g.new_edges([(0, 1), (1, 2), (2, 0)])

    assert np.array_equal(na.local_closure(g), [1, 1, 1])
    assert np.array_equal(na.local_closure(g, mode="cycle-out"), [1, 1, 1])
    assert np.array_equal(na.local_closure(g, mode="cycle-in"), [1, 1, 1])

    for mode in ("fan-in", "fan-out"):
        assert np.array_equal(na.local_closure(g, mode=mode), [0, 0, 0])
    
    with pytest.raises(ValueError):
        na.local_closure(g, method="plop")
    
    with pytest.raises(ValueError):
        na.local_closure(g, mode="circle")

    # directed (fan-in/out)
    g = nngt.Graph(3)

    g.new_edges([(0, 1), (0, 2), (1, 2)])

    assert np.array_equal(na.local_closure(g, mode="fan-out"), [1, 0, 0])
    assert np.array_equal(na.local_closure(g, mode="fan-in"), [0, 0, 1])

    for mode in ("cycle-in", "cycle-out"):
        assert np.array_equal(na.local_closure(g, mode=mode), [0, 0, 0])

    # directed weighted
    g = nngt.Graph(4)

    g.new_edges([(0, 1), (1, 2), (1, 3), (2, 1), (3, 0), (3, 1)])

    # weights for normal/zhang
    g.set_weights([1, 0.25, 0.5, 0.25, 0.5, 1])

    modes = ("cycle-out", "cycle-in", "fan-in", "fan-out")
    clsrs = [[1/3, 1, 0, 1/3], [1, 0.5, 0, 2/5], [0, 1, 0, 0], [0, 0, 0, 2/3]]

    for mode, res in zip(modes, clsrs):
        assert np.all(np.isclose(
            na.local_closure(g, weights="weight", mode=mode), res))

    # weights for continuous
    g.set_weights([1, 0.25, 1, 0.25, 1/64, 1])
    
    clsrs = [
        [1/24, 0.5, 0, 0.1],
        [0.5, 0.5, 0, 1/24],
        [0, 0.5, 0, 0],
        [0, 0, 0, 0.1]
    ]
    
    for mode, res in zip(modes, clsrs):
        assert np.all(np.isclose(
            na.local_closure(g, weights="weight", method="continuous",
                             mode=mode),
            res))


if __name__ == "__main__":
    if not nngt.get_config("mpi"):
        test_binary_undirected_clustering()
        test_weighted_undirected_clustering()
        test_weighted_directed_clustering()
        test_reciprocity()
        test_iedges()
        test_swp()
        test_partial_directed_clustering()
        test_clustering_parameters()
        test_global_clustering()
        test_local_closure()

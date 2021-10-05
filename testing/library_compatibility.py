#!/usr/bin/env python
#-*- coding:utf-8 -*-

# library_compatibility.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test that use some existing graph libraries for graph analysis, or that compare
the results with an existing implementation.
"""

import numpy as np

import nngt
import nngt.analysis as na


backends = ["networkx", "igraph", "graph-tool"]

all_backends = ["networkx", "igraph", "graph-tool", "nngt"]


def test_weighted_undirected_clustering():
    '''
    Compare the onnela implementation with networkx and Barrat with igraph.
    '''
    # import networkx and igraph
    import networkx as nx
    import igraph as ig

    # create a pre-defined graph
    num_nodes = 5
    edge_list = [
        (0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 0)
    ]
    weights = [0.53, 0.45, 0.8, 0.125, 0.66, 0.31, 0.78]

    # create nx graph and compute reference onnela clustering
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))

    arr_edges = [e + (w,) for e, w in zip(edge_list, weights)]
    nx_graph.add_weighted_edges_from(arr_edges, weight="weight")

    onnela = list(nx.clustering(nx_graph, weight="weight").values())

    triplets = [3, 3, 1, 1, 6]
    gc_onnela = np.sum(np.multiply(onnela, triplets)) / np.sum(triplets)

    # create ig graph and compute reference Barrat clustering
    ig_graph = ig.Graph(num_nodes, directed=False)
    ig_graph.add_edges(edge_list)
    ig_graph.es["weight"] = weights

    barrat = ig_graph.transitivity_local_undirected(mode="zero",
                                                    weights="weight")

    strength = np.array(ig_graph.strength(weights='weight'))
    degree   = np.array(ig_graph.degree())
    triplets = strength*(degree - 1)

    gc_barrat = np.sum(np.multiply(barrat, triplets)) / np.sum(triplets)

    # check for all backends
    for bckd in all_backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=False)
        g.new_edges(edge_list, attributes={"weight": weights})

        # onnela
        assert np.all(np.isclose(
            na.local_clustering(g, weights="weight", method="onnela"),
            onnela))

        assert np.isclose(
            na.global_clustering(g, weights="weight", method="onnela"),
            gc_onnela)

        # barrat
        assert np.all(np.isclose(
            na.local_clustering(g, weights="weight", method="barrat"),
            barrat))

        assert np.isclose(
            na.global_clustering(g, weights="weight", method="barrat"),
            gc_barrat)

        # fully reciprocal directed version
        g = nngt.Graph(nodes=num_nodes, directed=True)

        g.new_edges(edge_list, attributes={"weight": weights})

        g.new_edges(np.array(edge_list, dtype=int)[:, ::-1],
                    attributes={"weight": weights})

        assert np.all(np.isclose(
            na.local_clustering(g, weights="weight", method="onnela"),
            onnela))

        assert np.isclose(
            na.global_clustering(g, weights="weight", method="onnela"),
            gc_onnela)

        assert np.all(np.isclose(
            na.local_clustering(g, weights="weight", method="barrat"),
            barrat))

        assert np.isclose(
            na.global_clustering(g, weights="weight", method="barrat"),
            gc_barrat)


def test_assortativity():
    ''' Check assortativity result for all backends '''
    # DIRECTED
    num_nodes = 5
    edge_list = [(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 2)]
    weights   = [0.53, 0.45, 0.8, 0.125, 0.66, 0.31, 0.78]

    # expected results
    assort_unweighted = -0.47140452079103046
    assort_weighted   = -0.5457956719785911

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes)
        g.new_edges(edge_list, attributes={"weight": weights})

        assert np.isclose(
            nngt.analyze_graph["assortativity"](g, "in"), assort_unweighted)

        # not check weighted version for networkx for now
        assert np.isclose(
            nngt.analyze_graph["assortativity"](g, "in", weights=True),
            assort_weighted)

    # UNDIRECTED
    edge_list = [(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3)]
    weights = weights[:len(edge_list)]

    # expected results
    assort_unweighted = -0.33333333333333215
    assort_weighted   = -0.27351320394915296

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=False)
        g.new_edges(edge_list, attributes={"weight": weights})

        assert np.isclose(
            nngt.analyze_graph["assortativity"](g, "in"), assort_unweighted)

        # not check weighted version for networkx for now
        if bckd != "networkx":
            assert np.isclose(
                nngt.analyze_graph["assortativity"](g, "in", weights=True),
                assort_weighted)


def test_closeness():
    ''' Check closeness results for all backends '''
    num_nodes = 5
    edge_list = [(0, 1), (0, 3), (1, 3), (2, 0), (3, 2), (3, 4), (4, 2)]

    weights = [0.54881, 0.71518, 0.60276, 0.54488, 0.42365, 0.64589, 0.43758]

    expected = [2/3, 0.5, 0.5, 0.5714285714285714, 0.4444444444444444]
    weighted = [1.06273031, 0.89905622, 0.83253895, 1.12504606, 0.86040934]

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=True)
        g.new_edges(edge_list, attributes={"weight": weights})

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, harmonic=False), expected))

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, weights=True, harmonic=False),
            weighted))

    # with zero degrees and harmonic implementation
    # closeness does not work for igraph if some nodes have zero in/out-degrees
    edge_list = [(0, 1), (0, 2), (0, 3), (1, 3), (3, 2), (3, 4), (4, 2)]

    # DIRECTED
    harmonic   = [7/8, 0.5, 0, 0.5, 1/4]
    arithmetic = [0.8, 0.6, np.NaN, 1., 1.]

    harmonic_in   = [0, 1/4, 7/8, 0.5, 0.5]
    arithmetic_in = [np.NaN, 1, 0.8, 1., 0.6]

    harmonic_wght   = [1.42006842, 0.92688794, 0., 0.97717257, 0.5713241 ]
    arithmetic_wght = [1.28394428, 1.10939361, np.NaN, 1.86996279, 2.2852964]

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=True)
        g.new_edges(edge_list, attributes={"weight": weights})

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, harmonic=True), harmonic))

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, harmonic=False), arithmetic,
            equal_nan=True))

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, mode="in", harmonic=True),
            harmonic_in))

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, mode="in", harmonic=False),
            arithmetic_in, equal_nan=True))

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, weights=True, harmonic=True),
            harmonic_wght))

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, weights=True, harmonic=False),
            arithmetic_wght, equal_nan=True))

    # UNDIRECTED
    harmonic   = [7/8, 3/4, 7/8, 1., 3/4]
    arithmetic = [0.8, 2/3, 0.8, 1., 2/3]

    harmonic_wght   = [1.436723, 1.382419, 1.76911934, 1.85074797, 1.38520591]
    arithmetic_wght = [1.3247182, 1.2296379, 1.5717462, 1.8040934, 1.16720163]

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=False)
        g.new_edges(edge_list, attributes={"weight": weights})

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, harmonic=True), harmonic))

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, weights=True, harmonic=True),
            harmonic_wght))

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, harmonic=False), arithmetic,
            equal_nan=True))

        assert np.all(np.isclose(
            nngt.analyze_graph["closeness"](g, weights="weight",
                                            harmonic=False),
            arithmetic_wght, equal_nan=True))


def test_betweenness():
    ''' Check betweenness results for all backends '''
    num_nodes  = 5

    # UNDIRECTED
    edge_list  = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (3, 4)]

    weights = [0.1, 0.3, 1.5, 0.8, 5.6, 4., 2.3]

    nb_expect = [0.083333, 0.0, 0.083333, 0.3333333, 0.0]
    eb_expect = [0.15, 0.2, 0.15, 0.25, 0.15, 0.15, 0.25]

    nb_exp_wght = [0.5, 2/3, 0, 0.5, 0]
    eb_exp_wght = [0.6, 0.4, 0, 0.6, 0, 0, 0.4]

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=False)
        g.new_edges(edge_list, attributes={"weight": weights})

        nb, eb = nngt.analyze_graph["betweenness"](g)

        assert np.all(np.isclose(nb, nb_expect))
        assert np.all(np.isclose(eb, eb_expect))

        # weighted
        nb, eb = nngt.analyze_graph["betweenness"](g, weights=True)

        assert np.all(np.isclose(nb, nb_exp_wght))
        assert np.all(np.isclose(eb, eb_exp_wght))

    # DIRECTED
    edge_list  = [
        (0, 1), (0, 2), (0, 3), (1, 3), (3, 2), (3, 4), (4, 2), (4, 0)
    ]

    weights = [0.1, 0.3, 1.5, 0.8, 5.6, 4., 2.3, 0.9]

    nb_expect = [0.25, 0, 0, 1/3, 0.25]
    eb_expect = [0.15, 0.05, 0.15, 0.2, 0.1, 0.3, 0.05, 0.3]

    nb_exp_wght = [0.5, 0.25, 0, 1/3, 5/12]
    eb_exp_wght = [0.3, 0.2, 0, 0.35, 0, 0.4, 0, 0.45]

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=True)
        g.new_edges(edge_list, attributes={"weight": weights})

        nb, eb = nngt.analyze_graph["betweenness"](g)

        assert np.all(np.isclose(nb, nb_expect))
        assert np.all(np.isclose(eb, eb_expect))

        # weighted
        nb, eb = nngt.analyze_graph["betweenness"](g, weights=True)

        assert np.all(np.isclose(nb, nb_exp_wght))
        assert np.all(np.isclose(eb, eb_exp_wght))


def test_components():
    ''' Check connected components for all backends '''
    num_nodes = 8
    edge_list = [(0, 1), (0, 2), (1, 2)]

    for i in range(3, num_nodes - 1):
        edge_list.append((i, i+1))

    edge_list.append((7, 3))

    for bckd in all_backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=True)
        g.new_edges(edge_list)

        # scc
        cc, hist = \
            nngt.analyze_graph["connected_components"](g, ctype="scc")

        idx = np.array([i for i in range(num_nodes)], dtype=int)

        # nodes are assigned to the correct components
        assert np.all(cc[0] not in cc[1:])        # 3 first are isolated
        assert np.all(cc[1] not in cc[idx != 1])  # 3 first are isolated
        assert np.all(cc[2] not in cc[idx != 2])  # 3 first are isolated
        assert np.all(cc[3:] == cc[3])            # 5 last together
        # hence counts should be 1, 1, 1, and 5
        assert hist[cc[0]] == 1
        assert hist[cc[1]] == 1
        assert hist[cc[2]] == 1
        assert hist[cc[5]] == 5

        # wcc
        cc, hist = \
            nngt.analyze_graph["connected_components"](g, ctype="wcc")

        # nodes are assigned to the correct components
        assert np.all(cc[:3] == cc[0])  # 3 first together
        assert np.all(cc[3:] == cc[3])  # 5 last together
        # hence counts should be 3 and 5
        assert hist[cc[0]] == 3
        assert hist[cc[5]] == 5

        # undirected
        g = nngt.Graph(nodes=num_nodes, directed=False)
        g.new_edges(edge_list)

        cc, hist = \
            nngt.analyze_graph["connected_components"](g)

        # nodes are assigned to the correct components
        assert np.all(cc[:3] == cc[0])  # 3 first together
        assert np.all(cc[3:] == cc[3])  # 5 last together
        # hence counts should be 3 and 5
        assert hist[cc[0]] == 3
        assert hist[cc[5]] == 5


def test_diameter():
    ''' Check connected components for all backends '''
    # unconnected
    num_nodes = 8
    edge_list = [(0, 1), (0, 2), (1, 2)]

    for i in range(3, num_nodes - 1):
        edge_list.append((i, i+1))

    edge_list.append((7, 3))

    weights = [0.58, 0.59, 0.88, 0.8, 0.61, 0.66, 0.62, 0.28]

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=True)
        g.new_edges(edge_list, attributes={"weight": weights})

        assert np.isinf(nngt.analyze_graph["diameter"](g, weights=None))

        assert np.isinf(nngt.analyze_graph["diameter"](g, weights=True))

    # connected
    num_nodes = 5
    edge_list = [(0, 1), (0, 3), (1, 3), (2, 0), (3, 2), (3, 4), (4, 2)]

    weights = [0.58, 0.59, 0.88, 0.8, 0.61, 0.66, 0.28]

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=True)
        g.new_edges(edge_list, attributes={"weight": weights})

        d = nngt.analyze_graph["diameter"](g)

        assert nngt.analyze_graph["diameter"](g, weights=None) == 3

        assert np.isclose(
            nngt.analyze_graph["diameter"](g, weights="weight"), 2.29)


def test_binary_shortest_distance():
    ''' Check shortest distance '''
    num_nodes = 5

    for bckd in backends:
        nngt.use_backend(bckd)

        # UNDIRECTED
        edge_list = [
            (0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 0)
        ]

        g = nngt.Graph(num_nodes, directed=False)
        g.new_edges(edge_list)

        mat_dist = na.shortest_distance(g)

        undirected_dist = np.array([
            [0, 1, 2, 1, 1],
            [1, 0, 1, 2, 1],
            [2, 1, 0, 2, 1],
            [1, 2, 2, 0, 1],
            [1, 1, 1, 1, 0],
        ])

        assert np.array_equal(mat_dist, undirected_dist)

        # undirected, sources
        mat_dist = na.shortest_distance(g, sources=[0, 1, 2])

        assert np.array_equal(mat_dist, undirected_dist[:3])

        # undirected targets
        mat_dist = na.shortest_distance(g, targets=[0, 1, 2])

        assert np.array_equal(mat_dist[:, :3], undirected_dist[:, :3])

        # single source/target
        assert na.shortest_distance(g, sources=0, targets=2) == 2

        # DIRECTED
        g = nngt.Graph(num_nodes, directed=True)
        edge_list = [
            (0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 2), (4, 0)
        ]
        g.new_edges(edge_list)

        directed_dist = np.array([
            [0.,     np.inf, np.inf, 1.,  np.inf],
            [1.,     0.,     1.,     2.,  2.    ],
            [2.,     2.,     0.,     2.,  1.    ],
            [np.inf, np.inf, np.inf, 0.,  np.inf],
            [ 1.,    1.,     1.,     1.,  0.    ]
        ])

        mat_dist = na.shortest_distance(g)

        assert np.array_equal(mat_dist, directed_dist)

        # check undirected from directed
        mat_dist = na.shortest_distance(g, directed=False)

        assert np.array_equal(mat_dist, undirected_dist)

        # single source
        mat_dist = na.shortest_distance(g, sources=[0])

        assert np.array_equal(mat_dist, directed_dist[:1].ravel())

        # single target
        mat_dist = na.shortest_distance(g, targets=0)

        assert np.array_equal(mat_dist, directed_dist[:, 0].ravel())

        # single source/target directed
        assert np.isinf(na.shortest_distance(g, sources=0, targets=2))


def test_weighted_shortest_distance():
    ''' Check shortest distance '''
    for bckd in backends:
        nngt.use_backend(bckd)

        num_nodes = 5

        edge_list = [
            (0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 0)
        ]
        weights = [2., 1., 1., 3., 2., 3., 2.]

        # UNDIRECTED
        g = nngt.Graph(num_nodes, directed=False)
        g.new_edges(edge_list)

        mat_dist = na.shortest_distance(g, weights=weights)

        undirected_dist = np.array([
            [0, 1, 2, 2, 2],
            [1, 0, 1, 3, 2],
            [2, 1, 0, 4, 3],
            [2, 3, 4, 0, 3],
            [2, 2, 3, 3, 0],
        ])

        assert np.array_equal(mat_dist, undirected_dist)

        # undirected, sources
        g.set_weights(weights)
        mat_dist = na.shortest_distance(g, sources=[0, 1, 2], weights='weight')

        assert np.array_equal(mat_dist, undirected_dist[:3])

        # undirected targets
        mat_dist = na.shortest_distance(g, targets=[0, 1, 2], weights='weight')

        assert np.array_equal(mat_dist[:, :3], undirected_dist[:, :3])

        # single source/target
        dist = na.shortest_distance(g, sources=3, targets=2, weights='weight')

        assert dist == 4

        # DIRECTED
        g = nngt.Graph(num_nodes, directed=True)
        g.new_edges(edge_list)

        directed_dist = np.array([
            [0.,     np.inf, np.inf, 2.,  np.inf],
            [1.,     0.,     1.,     3.,  4.    ],
            [5.,     5.,     0.,     6.,  3.    ],
            [np.inf, np.inf, np.inf, 0.,  np.inf],
            [ 2.,    2.,     3.,     3.,  0.    ]
        ])

        mat_dist = na.shortest_distance(g, weights=weights)

        assert np.array_equal(mat_dist, directed_dist)

        # check undirected from directed gives back undirected results
        mat_dist = na.shortest_distance(g, directed=False, weights=weights)

        assert np.array_equal(mat_dist, undirected_dist)

        # single source
        g.set_weights(weights)
        mat_dist = na.shortest_distance(g, sources=[0], weights='weight')

        assert np.array_equal(mat_dist, directed_dist[:1].ravel())

        # single target
        mat_dist = na.shortest_distance(g, targets=0, weights='weight')

        assert np.array_equal(mat_dist, directed_dist[:, 0].ravel())

        # single source/target directed
        assert np.isinf(
            na.shortest_distance(g, sources=0, targets=2, weights='weight'))


def test_binary_shortest_paths():
    ''' Test shortests paths with BFS algorithm '''
    for bckd in backends:
        nngt.use_backend(bckd)

        num_nodes = 5

        # UNDIRECTED
        edge_list = [
            (0, 1), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)
        ]

        g = nngt.Graph(num_nodes, directed=False)
        g.new_edges(edge_list)

        assert na.shortest_path(g, 0, 0) == [0]
        assert na.shortest_path(g, 0, 1) == [0, 1]
        assert na.shortest_path(g, 0, 2) in ([0, 1, 2], [0, 3, 2])

        count = 0

        for p in na.all_shortest_paths(g, 4, 2):
            assert p in ([4, 1, 2], [4, 3, 2])
            count += 1

        assert count == 2

        count = 0

        for p in na.all_shortest_paths(g, 1, 1):
            assert p == [1]
            count += 1

        assert count == 1

        # DIRECTED
        edge_list = [
            (0, 1), (0, 3), (1, 2), (1, 4), (3, 2), (4, 3)
        ]

        g = nngt.Graph(num_nodes, directed=True)
        g.new_edges(edge_list)

        assert na.shortest_path(g, 0, 0) == [0]
        assert na.shortest_path(g, 2, 4) == []
        assert na.shortest_path(g, 0, 2) in ([0, 1, 2], [0, 3, 2])

        count = 0

        for p in na.all_shortest_paths(g, 4, 2):
            assert p == [4, 3, 2]
            count += 1

        assert count == 1

        count = 0

        for p in na.all_shortest_paths(g, 1, 1):
            assert p == [1]
            count += 1

        assert count == 1


def test_weighted_shortest_paths():
    ''' Test shortest paths with Dijsktra algorithm '''
    for bckd in backends:
        nngt.use_backend(bckd)

        num_nodes = 5

        # UNDIRECTED
        edge_list = [
            (0, 1), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)
        ]
        weights = [5, 0.1, 3., 0.5, 1, 3.5]

        g = nngt.Graph(num_nodes, directed=False)
        g.new_edges(edge_list)
        g.set_weights(weights)

        assert na.shortest_path(g, 0, 0, weights='weight') == [0]
        assert na.shortest_path(g, 0, 1, weights='weight') == [0, 3, 2, 1]

        sp = na.shortest_path(g, 1, 3, weights=weights)
        assert sp in ([1, 4, 3], [1, 2, 3])

        count = 0

        for p in na.all_shortest_paths(g, 1, 3, weights='weight'):
            assert p in ([1, 4, 3], [1, 2, 3])
            count += 1

        assert count == 2

        count = 0

        for p in na.all_shortest_paths(g, 1, 1, weights='weight'):
            assert p == [1]
            count += 1

        assert count == 1

        # DIRECTED
        edge_list = [
            (0, 1), (0, 3), (1, 2), (1, 4), (3, 2), (4, 3)
        ]
        weights = [1., 0.1, 0.1, 0.5, 1, 3.5]

        g = nngt.Graph(num_nodes, directed=True)
        g.new_edges(edge_list, attributes={"weight": weights})

        assert na.shortest_path(g, 0, 0, weights='weight') == [0]
        assert na.shortest_path(g, 2, 4, weights='weight') == []
        assert na.shortest_path(g, 1, 3, weights=weights) == [1, 4, 3]

        count = 0

        for p in na.all_shortest_paths(g, 0, 2, weights='weight'):
            assert p in ([0, 1, 2], [0, 3, 2])
            count += 1

        assert count == 2

        count = 0

        for p in na.all_shortest_paths(g, 1, 1, weights='weight'):
            assert p == [1]
            count += 1

        assert count == 1

        # UNDIRECTED FROM DIRECTED
        weights = [5, 0.1, 3., 0.5, 1, 3.5]
        # reset weights
        g.set_weights(weights)

        assert na.shortest_path(g, 0, 0, False, weights='weight') == [0]
        assert na.shortest_path(
            g, 0, 1, False, weights='weight') == [0, 3, 2, 1]

        sp = na.shortest_path(g, 1, 3, False, weights=weights)
        assert sp in ([1, 4, 3], [1, 2, 3])

        count = 0

        for p in na.all_shortest_paths(g, 1, 3, False, weights='weight'):
            assert p in ([1, 4, 3], [1, 2, 3])
            count += 1

        assert count == 2

        count = 0

        for p in na.all_shortest_paths(g, 1, 1, False, weights='weight'):
            assert p == [1]
            count += 1

        assert count == 1



def test_subgraph_centrality():
    ''' Check subgraph centrality with networkx implementation '''
    num_nodes  = 5
    edge_list  = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (3, 4)]

    # this test requires networkx as backend
    nngt.use_backend("networkx")

    g = nngt.Graph(num_nodes, directed=False)
    g.new_edges(edge_list)

    # test full centralities
    import networkx as nx

    sc_nx = nx.subgraph_centrality(g.graph)
    sc_nx = np.array([sc_nx[i] for i in range(num_nodes)])

    sc_nngt = nngt.analysis.subgraph_centrality(g, weights=False,
                                                normalize=False)

    assert np.all(np.isclose(sc_nx, sc_nngt))

    # test max_centrality
    sc_nngt = nngt.analysis.subgraph_centrality(g, weights=False,
                                                normalize="max_centrality")

    assert np.all(np.isclose(sc_nx / sc_nx.max(), sc_nngt))

    # test subpart
    sc_nngt = nngt.analysis.subgraph_centrality(g, weights=False,
                                                normalize=False, nodes=[0, 1])

    assert np.all(np.isclose(sc_nx[:2], sc_nngt))


if __name__ == "__main__":
    test_weighted_undirected_clustering()
    test_assortativity()
    test_closeness()
    test_betweenness()
    test_components()
    test_diameter()
    test_binary_shortest_distance()
    test_weighted_shortest_distance()
    test_binary_shortest_paths()
    test_weighted_shortest_paths()
    test_subgraph_centrality()

#!/usr/bin/env python
#-*- coding:utf-8 -*-

# library_compatibility.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Test the graph analysis function for all backends """

import numpy as np

import nngt
import nngt.analysis as na


backends_clustering = ["networkx", "igraph", "graph-tool", "nngt"]

backends = ["networkx", "igraph", "graph-tool"]


def test_binary_undirected_clustering():
    ''' Check the clustering coefficient results for all backends '''
    # create a pre-defined graph
    num_nodes = 5
    edge_list = [
        (0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 2), (4, 0)
    ]

    # expected results
    loc_clst  = [2/3., 2/3., 1., 1., 0.5]
    glob_clst = 0.6428571428571429

    for bckd in backends_clustering:
        nngt.set_config("backend", bckd)

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

        # check all 3 ways of computing the global clustering
        assert np.isclose(
            na.global_clustering(g, directed=False), glob_clst)

        assert np.isclose(
            na.global_clustering_binary_undirected(g), glob_clst)

        assert np.isclose(
            nngt.analyze_graph["global_clustering"](g, directed=False),
            glob_clst)


def test_weighted_undirected_clustering():
    '''
    Compare the Onella implementation with networkx and Barrat with igraph.
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

    # create nx graph and compute reference Onella clustering
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))

    arr_edges = [e + (w,) for e, w in zip(edge_list, weights)]
    nx_graph.add_weighted_edges_from(arr_edges, weight="weight")

    onella = list(nx.clustering(nx_graph, weight="weight").values())

    triplets = [3, 3, 1, 1, 6]
    gc_onella = np.sum(np.multiply(onella, triplets)) / np.sum(triplets)

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
    for bckd in backends_clustering:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes, directed=False)
        g.new_edges(edge_list, attributes={"weight": weights})

        # onella
        assert np.all(np.isclose(
            na.local_clustering(g, weights="weight", method="onella"),
            onella))

        assert np.isclose(
            na.global_clustering(g, weights="weight", method="onella"),
            gc_onella)

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
            na.local_clustering(g, weights="weight", method="onella"),
            onella))

        assert np.isclose(
            na.global_clustering(g, weights="weight", method="onella"),
            gc_onella)

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
        if bckd != "networkx":
            assert np.isclose(
                nngt.analyze_graph["assortativity"](g, "in", weights=True),
                assort_weighted)

    # UNDIRECTED
    edge_list = [(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3)]

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


def test_reciprocity():
    ''' Check reciprocity result for all backends '''
    num_nodes  = 5
    edge_list1 = [(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 2)]
    edge_list2 = [(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3)]

    recip = 2/7

    for bckd in backends:
        nngt.set_config("backend", bckd)

        # directed
        g = nngt.Graph(nodes=num_nodes, directed=True)
        g.new_edges(edge_list1)

        assert np.isclose(
            nngt.analyze_graph["reciprocity"](g), recip)

        # undirected
        g = nngt.Graph(nodes=num_nodes, directed=False)
        g.new_edges(edge_list2)

        assert nngt.analyze_graph["reciprocity"](g) == 1.


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

        if bckd == "igraph":
            # assert igraph fails as expected
            try:
                print(nngt.analyze_graph["closeness"](g, harmonic=True))
                errored = False
            except NotImplementedError:
                errored = True

            assert errored

            try:
                nngt.analyze_graph["closeness"](g, harmonic=False)
                runtime_error = False
            except RuntimeError:
                runtime_error = True

            assert runtime_error
        else:
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
                nngt.analyze_graph["closeness"](g, weights=True,
                                                harmonic=True),
                harmonic_wght))

            assert np.all(np.isclose(
                nngt.analyze_graph["closeness"](g, weights=True,
                                                harmonic=False),
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

        if bckd != "igraph":
            assert np.all(np.isclose(
                nngt.analyze_graph["closeness"](g, harmonic=True), harmonic))

            assert np.all(np.isclose(
                nngt.analyze_graph["closeness"](g, weights=True,
                                                harmonic=True),
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

    for bckd in backends:
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


def test_subgraph_centrality():
    ''' Check subgraph centrality with networkx implementation '''
    num_nodes  = 5
    edge_list  = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (3, 4)]

    nngt.set_config("backend", "networkx")

    g = nngt.Graph(num_nodes, directed=False)
    g.new_edges(edge_list)

    import networkx as nx

    sc_nx = nx.subgraph_centrality(g.graph)
    sc_nx = [sc_nx[i] for i in range(num_nodes)]

    sc_nngt = nngt.analysis.subgraph_centrality(g, weights=False,
                                                normalize=False)

    assert np.all(np.isclose(sc_nx, sc_nngt))


if __name__ == "__main__":
    test_binary_undirected_clustering()
    test_weighted_undirected_clustering()
    test_assortativity()
    test_reciprocity()
    test_closeness()
    test_betweenness()
    test_components()
    test_diameter()
    test_subgraph_centrality()

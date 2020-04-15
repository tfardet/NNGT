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


backends = ["networkx", "igraph", "graph-tool"]


def test_clustering():
    ''' Check the clustering coefficient results for all backends '''
    # create a pre-defined graph
    num_nodes = 5
    edge_list = [
        (0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3), (4, 2), (4, 0)
    ]

    # expected results
    loc_clst  = [2/3., 2/3., 1., 1., 0.5]
    glob_clst = 0.6428571428571429

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes)
        g.new_edges(edge_list)

        assert np.all(np.isclose(
            nngt.analyze_graph["local_clustering"](g), loc_clst))

        assert np.isclose(
            nngt.analyze_graph["global_clustering"](g), glob_clst)


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


if __name__ == "__main__":
    test_clustering()
    test_assortativity()
    test_reciprocity()

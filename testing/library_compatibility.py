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


def test_clustering():
    ''' Check the clustering coefficient results '''
    backends = ["networkx", "igraph", "graph-tool"]
    results  = {}

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


if __name__ == "__main__":
    test_clustering()

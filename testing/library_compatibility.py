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
    # ~ edge_list = [(0, 3), (1, 0), (1, 2), (2, 4), (4, 1), (4, 3)]
    edge_list = [(0, 3), (3, 0), (1, 0), (0, 1), (1, 2), (2, 1), (2, 4), (4, 2), (4, 1), (1, 4), (4, 3), (3, 4)]

    for bckd in backends:
        nngt.set_config("backend", bckd)

        g = nngt.Graph(nodes=num_nodes)
        g.new_edges(edge_list)

        results[bckd] = nngt.analyze_graph["local_clustering"](g)

    print(results)

    nngt.plot.draw_network(g, show=True)

    assert np.all(np.isclose(results[backends[0]], results[backends[1]]) &
                  np.isclose(results[backends[0]], results[backends[2]])), \
        "Differing clustering coefficients: {}".format(results)


if __name__ == "__main__":
    test_clustering()

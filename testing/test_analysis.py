#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Test the graph analysis functions """

import numpy as np

import nngt


def test_directed_clustering():
    '''
    Check directed clustering.
    (should return undirected value if graph is not directed)
    '''
    g = nngt.generation.erdos_renyi(avg_deg=10, nodes=100, directed=False)

    ccu = nngt.analysis.undirected_local_clustering(g)
    cc  = nngt.analysis.local_clustering(g)

    assert np.all(np.isclose(cc, ccu))


if __name__ == "__main__":
    test_directed_clustering()

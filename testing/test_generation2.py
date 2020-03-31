#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_generation2.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the new methods of the :mod:`~nngt.generation` module.
"""

import numpy as np

import nngt
import nngt.generation as ng


def test_from_degree_list():
    '''
    Check that the degrees generated using `from_degree_list` indeed
    correspond to the provided list
    '''
    num_nodes = 1000
    deg_list  = np.random.randint(0, 100, size=num_nodes)

    # test for in
    g = ng.from_degree_list(deg_list, degree_type="in", nodes=num_nodes)

    assert np.all(g.get_degrees("in") == deg_list)

    assert g.edge_nb() == np.sum(deg_list)

    # test for out
    g = ng.from_degree_list(deg_list, degree_type="out", nodes=num_nodes)

    assert np.all(g.get_degrees("out") == deg_list)

    assert g.edge_nb() == np.sum(deg_list)


if __name__ == "__main__":
    test_from_degree_list()

#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_weights.py

"""
Test weights setting.
"""

import numpy as np
import pytest

import nngt
import nngt.generation as ng


@pytest.mark.mpi_skip
def test_set_weights():
    w = 10.
    g = ng.erdos_renyi(nodes=100, density=0.1, weights=w)

    assert set(g.get_weights()) == {w}

    w2 = 5.
    g.set_weights(w2)
    
    assert set(g.get_weights()) == {w2}

    elist = g.get_edges()[:10]  # keep 10 first edges

    w3 = 2.
    g.set_weights(w3, elist=elist)

    assert set(g.get_weights()) == {w2, w3}
    assert set(g.get_weights(edges=elist)) == {w3}


def test_weighted_degrees():
    g = nngt.Graph()
    g.new_node(10)

    ww = [5.]*4
    g.new_edges([(0, 4), (8, 2), (7, 0), (1, 3)], attributes={"weight": ww})

    # check degrees
    in_deg  = [1., 0., 1., 1., 1., 0., 0., 0., 0., 0.]
    out_deg = [1., 1., 0., 0., 0., 0., 0., 1., 1., 0.]
    tot_deg = [2., 1., 1., 1., 1., 0., 0., 1., 1., 0.]

    assert np.all(g.get_degrees("in") == in_deg)
    assert np.all(
        g.get_degrees("in", weights=True) == np.multiply(ww[0], in_deg))
    assert np.all(g.get_degrees("out") == out_deg)
    assert np.all(g.get_degrees("total") == tot_deg)

    # set all weights to zero
    g.set_weights(0.)

    assert set(g.get_weights()) == {0.}
    assert set(g.get_edge_attributes(name="weight")) == {0.}

    assert not np.any(g.get_degrees("in", weights=True))

    # set two edges to 2.
    elist = [(0, 4), (8, 2)]
    g.set_weights(2., elist=elist)

    assert set(g.get_weights()) == {0., 2.}
    assert set(g.get_edge_attributes(name="weight")) == {0., 2.}

    # test new weighted degree
    w_indeg  = [0., 0., 2., 0., 2., 0., 0., 0., 0., 0.]
    w_outdeg = [2., 0., 0., 0., 0., 0., 0., 0., 2., 0.]

    assert np.all(g.get_degrees("in", weights=True) == w_indeg)

    assert np.all(g.get_degrees("out", weights=True) == w_outdeg)

    # check the node and edges
    assert g.node_nb() == 10
    assert g.edge_nb() == 4


if __name__ == "__main__":
    test_set_weights()
    test_weighted_degrees()

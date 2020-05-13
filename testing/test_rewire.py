#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_rewire.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the rewire methods of the :mod:`~nngt.generation` module.
"""

import os

import numpy as np
import pytest

import nngt
import nngt.analysis as na
import nngt.generation as ng


if os.environ.get("MPI"):
    nngt.set_config("mpi", True)


@pytest.mark.mpi_skip
def test_random_rewire():
    ''' Check random rewire '''
    num_nodes = 10
    coord_nb  = 2
    recip     = 0.7
    shortcut  = 0.2

    g = ng.newman_watts(coord_nb, shortcut, reciprocity_circular=recip,
                        nodes=num_nodes)

    # get graph properties
    final_recip = na.reciprocity(g)
    num_edges   = g.edge_nb()

    # make some node and edge attributes
    g.new_node_attribute("random_int", "int",
                         values=[2, 5, 33, 6, 4, 1, 98, 45, 30, 10])
    g.new_node_attribute("attr2", "float",
                         values=nngt._rng.uniform(size=num_nodes))

    ww = np.arange(1, num_edges + 1, dtype=float)
    g.set_weights(ww)

    g.new_edge_attribute("my-edge-attr", "int", values=-ww[::-1].astype(int))

    # completely random rewiring
    r1 = ng.random_rewire(g)

    assert r1.node_nb() == num_nodes
    assert r1.edge_nb() == num_edges
    assert g.node_attributes == r1.node_attributes
    assert g.edge_attributes == r1.edge_attributes
    assert not np.array_equal(g.node_attributes["random_int"],
                              r1.node_attributes["random_int"])
    assert not np.array_equal(g.node_attributes["attr2"],
                              r1.node_attributes["attr2"])
    assert not np.array_equal(g.edge_attributes["weights"],
                              r1.edge_attributes["weights"])
    assert not np.array_equal(g.edge_attributes["my-edge-attr"],
                              r1.edge_attributes["my-edge-attr"])

    # keep degrees
    for deg_type in ["in-degree", "out-degree", "total-degree"]:
        degrees  = g.get_degrees(deg_type)

        edge_constraint = "together"

        if deg_type == "in-degree":
            edge_constraint = "preserve_in"
        elif deg_type == "out-degree":

        r2 = ng.random_rewire(g, constraints=deg_type,
                              node_attr_constraints="preserve",
                              )

        assert r2.node_nb() == num_nodes
        assert r2.edge_nb() == num_edges
        
        assert g.node_attributes == r2.node_attributes
        assert np.array_equal(g.node_attributes["random_int"],
                              r2.node_attributes["random_int"])
        assert np.array_equal(g.node_attributes["attr2"],
                              r2.node_attributes["attr2"])

        assert np.array_equal(r2.get_degrees(deg_type), degrees)

    # 


if __name__ == "__main__":
    nngt.set_config("multithreading", False)
    if not nngt.get_config("mpi"):
        test_random_rewire()

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
    assert not np.array_equal(g.edge_attributes["weight"],
                              r1.edge_attributes["weight"])
    assert not np.array_equal(g.edge_attributes["my-edge-attr"],
                              r1.edge_attributes["my-edge-attr"])

    # keep degrees
    for deg_type in ["in-degree", "out-degree", "total-degree"]:
        degrees  = g.get_degrees(deg_type)

        edge_constraint = "together"

        if deg_type == "in-degree":
            edge_constraint = "preserve_in"
        elif deg_type == "out-degree":
            edge_constraint = "preserve_out"

        r2 = ng.random_rewire(g, constraints=deg_type,
                              node_attr_constraints="preserve",
                              edge_attr_constraints=edge_constraint)

        # check basics
        assert r2.node_nb() == num_nodes
        assert r2.edge_nb() == num_edges

        assert np.array_equal(r2.get_degrees(deg_type), degrees)

        # check node attributes
        assert g.node_attributes == r2.node_attributes
        assert np.array_equal(g.node_attributes["random_int"],
                              r2.node_attributes["random_int"])
        assert np.array_equal(g.node_attributes["attr2"],
                              r2.node_attributes["attr2"])

        # check edge attributes
        if deg_type == "total-degree":
            # check that attributes were moved together
            weights  = r2.get_weights()
            my_eattr = r2.edge_attributes["my-edge-attr"]

            assert np.array_equal(my_eattr,
                                  (weights - (num_edges + 1)).astype(int))
        else:
            node_type = ("source_node" if edge_constraint == "preserve_out"
                         else "target_node")

            for i in range(num_nodes):
                kwarg = {node_type: i}

                old_edges = g.get_edges(**kwarg)
                new_edges = r2.get_edges(**kwarg)

                for name in g.edge_attributes:
                    old_attr = g.get_edge_attributes(old_edges, name)
                    new_attr = r2.get_edge_attributes(new_edges, name)

                    assert set(old_attr) == set(new_attr)


@pytest.mark.mpi_skip
def test_clst_rewire():
    ''' Test rewire preserving clustering '''
    num_nodes = 200
    coord_nb  = 8
    recip     = 0.7
    shortcut  = 0.1

    g = ng.newman_watts(coord_nb, shortcut, reciprocity_circular=recip,
                        nodes=num_nodes)

    num_edges = g.edge_nb()

    # make some node and edge attributes
    g.new_node_attribute("random_int", "int",
                         values=nngt._rng.integers(1, 2000, num_nodes))
    g.new_node_attribute("attr2", "float",
                         values=nngt._rng.uniform(size=num_nodes))

    ww = np.arange(1, num_edges + 1, dtype=float)
    g.set_weights(ww)

    g.new_edge_attribute("my-edge-attr", "int", values=-ww[::-1].astype(int))

    # rewire
    rtol = 0.1

    rc = ng.random_rewire(g, constraints="clustering",
                          node_attr_constraints="preserve",
                          edge_attr_constraints="together", rtol=rtol)

    r1 = ng.random_rewire(g)

    c0 = nngt.analysis.local_clustering(g).mean()
    c1 = nngt.analysis.local_clustering(r1).mean()
    cc = nngt.analysis.local_clustering(rc).mean()

    assert c0 - c1 > c0 - cc
    assert np.abs(c0 - cc) / c0 < rtol

    # check node attributes
    assert g.node_attributes == rc.node_attributes
    assert np.array_equal(g.node_attributes["random_int"],
                          rc.node_attributes["random_int"])
    assert np.all(np.isclose(g.node_attributes["attr2"],
                             rc.node_attributes["attr2"]))
    # check that attributes were moved together
    weights  = rc.get_weights()
    my_eattr = rc.edge_attributes["my-edge-attr"]

    assert np.array_equal(my_eattr, (weights - (num_edges + 1)).astype(int))


@pytest.mark.mpi_skip
def test_complete_lattice_rewire():
    ''' Check lattice rewiring method. '''
    num_nodes = 10
    degree = 4

    g = ng.fixed_degree(degree, "total", nodes=num_nodes)

    # node attributes
    g.new_node_attribute("random_int", "int",
                         values=[2, 5, 33, 6, 4, 1, 98, 45, 30, 10])
    g.new_node_attribute("attr2", "float",
                         values=nngt._rng.uniform(size=num_nodes))

    # edge attributes
    ww = nngt._rng.uniform(1, 5, size=g.edge_nb())
    g.set_weights(ww)
    g.new_edge_attribute("my-edge-attr", "int", values=-ww.astype(int))

    # rewire
    l1 = ng.lattice_rewire(g, weight="weight",
                           node_attr_constraints="preserve")

    assert g.node_nb() == l1.node_nb()
    assert g.edge_nb() == l1.edge_nb()

    # check node attributes
    assert g.node_attributes == l1.node_attributes
    assert np.array_equal(g.node_attributes["random_int"],
                          l1.node_attributes["random_int"])
    assert np.array_equal(g.node_attributes["attr2"],
                          l1.node_attributes["attr2"])

    # check the order of the first weights
    srt = np.sort(l1.get_weights())[::-1]
    oth = -srt.astype(int)

    all_other = True

    for i in range(num_nodes - 1):
        e = (i,  i + 1)
        assert l1.get_edge_attributes(e, name='weight') == srt[2*i]
        all_other *= l1.get_edge_attributes(e, name='my-edge-attr') == oth[2*i]
        
        e = (i + 1, i)
        assert l1.get_edge_attributes(e, name='weight') == srt[2*i + 1]
        all_other *= (l1.get_edge_attributes(e, name='my-edge-attr')
                      == oth[2*i + 1])

    assert not all_other


@pytest.mark.mpi_skip
def test_incomplete_lattice_rewire():
    ''' Lattice rewire when node degrees are odd. '''
    num_nodes = 10
    degree = 7

    g = ng.fixed_degree(degree, "total", nodes=num_nodes)

    # node attributes
    g.new_node_attribute("random_int", "int",
                         values=[2, 5, 33, 6, 4, 1, 98, 45, 30, 10])
    g.new_node_attribute("attr2", "float",
                         values=nngt._rng.uniform(size=num_nodes))

    # edge attributes
    ww = nngt._rng.uniform(1, 5, size=g.edge_nb())
    g.set_weights(ww)
    g.new_edge_attribute("my-edge-attr", "int", values=-ww.astype(int))

    # rewire
    l2 = ng.lattice_rewire(g, weight="weight", distance_sort="linear",
                           edge_attr_constraints="together")

    assert g.node_nb() == l2.node_nb()
    assert g.edge_nb() == l2.edge_nb()

    # check node attributes (may fail every million trials or so)
    assert g.node_attributes == l2.node_attributes
    assert not np.array_equal(g.node_attributes["random_int"],
                              l2.node_attributes["random_int"])
    assert not np.array_equal(g.node_attributes["attr2"],
                              l2.node_attributes["attr2"])

    # check edge attributes
    mea = l2.edge_attributes["my-edge-attr"]
    assert np.array_equal(mea, -l2.get_weights().astype(int))

    # check the order of the first weights
    srt = np.sort(ww)
    oth = -srt.astype(int)

    for i in range(num_nodes - 1):
        e = (i,  i + 1)
        assert l2.get_edge_attributes(e, name='weight') == srt[2*i]
        assert l2.get_edge_attributes(e, name='my-edge-attr') == oth[2*i]
        
        e = (i + 1, i)
        assert l2.get_edge_attributes(e, name='weight') == srt[2*i + 1]
        assert l2.get_edge_attributes(e, name='my-edge-attr') == oth[2*i + 1]


@pytest.mark.mpi_skip
def test_reciprocity_lattice_rewire():
    ''' Lattice rewire with variable reciprocity. '''
    num_nodes = 1000
    degree = 7

    g = ng.fixed_degree(degree, "total", nodes=num_nodes)

    # edge attributes
    ww = nngt._rng.uniform(1, 5, size=g.edge_nb())
    g.set_weights(ww)
    g.new_edge_attribute("my-edge-attr", "int", values=-ww.astype(int))

    # rewire
    reciprocity = 0.7

    l = ng.lattice_rewire(g, weight="weight", target_reciprocity=reciprocity,
                          edge_attr_constraints="together")

    assert np.isclose(na.reciprocity(l), reciprocity)

    srt = np.sort(ww)[::-1]
    oth = -srt.astype(int)

    for i in range(num_nodes - 1):
        e = (i,  i + 1)
        assert l.get_edge_attributes(e, name='weight') == srt[2*i]
        assert l.get_edge_attributes(e, name='my-edge-attr') == oth[2*i]
        
        e = (i + 1, i)
        assert l.get_edge_attributes(e, name='weight') == srt[2*i + 1]
        assert l.get_edge_attributes(e, name='my-edge-attr') == oth[2*i + 1]


if __name__ == "__main__":
    if not nngt.get_config("mpi"):
        test_random_rewire()
        test_clst_rewire()
        test_complete_lattice_rewire()
        test_incomplete_lattice_rewire()
        test_reciprocity_lattice_rewire()

#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_generation2.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the new methods of the :mod:`~nngt.generation` module.
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

    # test for total-degree
    deg_list = 2*np.random.randint(0, 50, size=num_nodes)

    try:
        g = ng.from_degree_list(deg_list, degree_type="total", nodes=num_nodes,
                                directed=True)

        assert np.all(g.get_degrees("total") == deg_list)

        assert g.edge_nb() == int(0.5*np.sum(deg_list))
    except ValueError:
        # non-graphical sequence was provided
        print("Skipping non graphical sequence for 'total-degree'.")

    # test for undirected
    deg_list = 2*np.random.randint(0, 50, size=num_nodes)

    try:
        g = ng.from_degree_list(deg_list, nodes=num_nodes, directed=False)

        assert np.all(g.get_degrees("total") == deg_list)

        assert g.edge_nb() == int(0.5*np.sum(deg_list))
    except ValueError:
        # non-graphical sequence was provided
        print("Skipping non graphical sequence for undirected graph.")


def test_newman_watts():
    '''
    Check the newman_watts generation method.
    '''
    nngt.use_backend("networkx")
    num_nodes  = 5
    k_lattice  = 2
    p_shortcut = 0.2

    ## USING EDGES

    # undirected
    g = ng.newman_watts(k_lattice, edges=6, nodes=num_nodes, directed=False)

    subset   = {(0, 1), (1, 2), (2, 3), (3, 4)}

    # connection between 0 and 4 depends on the library
    try:
        g.edge_id((0, 4))
        subset.add((0, 4))
    except:
        subset.add((4, 0))

    edge_set = {tuple(e) for e in g.edges_array}

    assert edge_set.issuperset(subset)
    assert g.edge_nb() == 6  # min_edges + one shortcut

    # directed
    reciprocity = 0.

    g = ng.newman_watts(k_lattice, reciprocity_circular=reciprocity, edges=6,
                        nodes=num_nodes, directed=True)

    assert g.edge_nb() == 6  # 5 lattice edge + one shortcut
    assert 0. <= na.reciprocity(g) <= 1/3.

    reciprocity = 1.

    g = ng.newman_watts(k_lattice, reciprocity_circular=reciprocity, edges=12,
                         nodes=num_nodes, directed=True)

    assert g.edge_nb() == 12  # 10 lattice edges + 2 shortcuts
    assert 5/6. <= na.reciprocity(g) <= 1

    reciprocity = 0.5
    g = ng.newman_watts(k_lattice, reciprocity_circular=reciprocity, edges=8,
                        nodes=num_nodes, directed=True)

    assert g.edge_nb() == 8  # 7 lattice edges + 1 shortcuts
    assert 0.5 <= na.reciprocity(g) <= 0.75

    ## USING PROBABILITY

    # undirected
    g = ng.newman_watts(k_lattice, p_shortcut, nodes=num_nodes, directed=False)

    assert 0.5*k_lattice*num_nodes <= g.edge_nb() <= k_lattice*num_nodes

    # directed
    reciprocity = 0.

    g = ng.newman_watts(k_lattice, p_shortcut, reciprocity, nodes=num_nodes,
                        directed=True)

    assert 0.5*k_lattice*num_nodes <= g.edge_nb() <= k_lattice*num_nodes

    reciprocity = 1.

    g = ng.newman_watts(k_lattice, p_shortcut, reciprocity, nodes=num_nodes,
                        directed=True)

    assert k_lattice*num_nodes <= g.edge_nb() <= 2*k_lattice*num_nodes

    reciprocity = 0.5
    g = ng.newman_watts(k_lattice, p_shortcut, reciprocity, nodes=num_nodes,
                        directed=True)

    recip_fact = 0.5*(1 + reciprocity)
    min_edges  = int(recip_fact*k_lattice*num_nodes)
    assert min_edges <= g.edge_nb() <= 2*recip_fact*k_lattice*num_nodes


@pytest.mark.mpi
def test_mpi_from_degree_list():
    '''
    Check that the degrees generated using `from_degree_list` indeed
    correspond to the provided list
    '''
    num_nodes = 1000

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    deg_list = np.random.randint(0, 100, size=num_nodes)

    # test for in
    g = ng.from_degree_list(deg_list, degree_type="in", nodes=num_nodes)

    deg = g.get_degrees("in")
    deg = comm.gather(deg, root=0)

    if nngt.on_master_process():
        deg = np.sum(deg, axis=0)
        assert np.all(deg == deg_list)

    num_edges = g.edge_nb()
    num_edges = comm.gather(num_edges, root=0)

    if nngt.on_master_process():
        num_edges = np.sum(num_edges)
        assert num_edges == np.sum(deg_list)

    # test for out
    g = ng.from_degree_list(deg_list, degree_type="out", nodes=num_nodes)

    deg = g.get_degrees("out")
    deg = comm.gather(deg, root=0)

    if nngt.on_master_process():
        deg = np.sum(deg, axis=0)
        assert np.all(deg == deg_list)

    num_edges = g.edge_nb()
    num_edges = comm.gather(num_edges, root=0)

    if nngt.on_master_process():
        num_edges = np.sum(num_edges)
        assert num_edges == np.sum(deg_list)


def test_total_undirected_connectivities():
    ''' Test total-degree connectivities '''
    num_nodes = 1000

    # erdos-renyi
    density = 0.1
    g = ng.erdos_renyi(density=density, nodes=num_nodes, directed=False)

    assert g.edge_nb() / (num_nodes*num_nodes) == density

    for directed in (True, False):
        # fixed-degree
        deg = 50
        g = ng.fixed_degree(deg, "total", nodes=num_nodes, directed=directed)

        assert {deg} == set(g.get_degrees())

        # gaussian degree
        avg = 50.
        std = 5.

        g = ng.gaussian_degree(avg, std, degree_type="total", nodes=num_nodes,
                               directed=directed)

        deviation = 20. / np.sqrt(num_nodes)
        average   = np.average(g.get_degrees())

        assert avg - deviation <= average <= avg + deviation


if __name__ == "__main__":
    test_newman_watts()

    if not nngt.get_config("mpi"):
        test_from_degree_list()
        test_total_undirected_connectivities()

    if nngt.get_config("mpi"):
        test_mpi_from_degree_list()

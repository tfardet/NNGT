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

    assert np.array_equal(g.get_degrees("in"), deg_list)

    assert g.edge_nb() == np.sum(deg_list)

    # test for out
    g = ng.from_degree_list(deg_list, degree_type="out", nodes=num_nodes)

    assert np.array_equal(g.get_degrees("out"), deg_list)

    assert g.edge_nb() == np.sum(deg_list)

    # test for total-degree
    deg_list = 2*np.random.randint(0, 50, size=num_nodes)

    try:
        g = ng.from_degree_list(deg_list, degree_type="total", nodes=num_nodes,
                                directed=True)

        assert np.array_equal(g.get_degrees("total"), deg_list)

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


@pytest.mark.mpi_skip
def test_newman_watts():
    '''
    Check the newman_watts generation method.
    '''
    num_nodes  = 5
    k_lattice  = 2
    p_shortcut = 0.2

    ## USING EDGES

    # undirected
    g = ng.newman_watts(k_lattice, edges=6, nodes=num_nodes, directed=False)

    lattice_edges = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]

    for e in lattice_edges:
        assert g.has_edge(e)
    
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

    recip_fact = 0.5*(1 + reciprocity / (2 - reciprocity))
    min_edges  = int(np.round(recip_fact*k_lattice*num_nodes))

    assert min_edges <= g.edge_nb() <= 2*recip_fact*k_lattice*num_nodes


@pytest.mark.mpi_skip
def test_watts_strogatz():
    '''
    Check the watts_strogatz generation method.
    '''
    num_nodes  = 5
    k_lattice  = 2
    p_shortcut = 0.2

    # undirected
    g = ng.watts_strogatz(k_lattice, p_shortcut, nodes=num_nodes,
                          directed=False)

    assert g.edge_nb() == int(0.5*k_lattice*num_nodes)

    # directed
    reciprocity = 0.

    g = ng.watts_strogatz(k_lattice, p_shortcut, reciprocity, nodes=num_nodes,
                          directed=True)

    assert g.edge_nb() == 0.5*k_lattice*num_nodes

    reciprocity = 1.

    g = ng.watts_strogatz(k_lattice, p_shortcut, reciprocity, nodes=num_nodes,
                          shuffle="sources", directed=True)

    assert g.edge_nb() == k_lattice*num_nodes
    assert np.all(g.get_degrees("in") == k_lattice)

    g = ng.watts_strogatz(k_lattice, p_shortcut, reciprocity, nodes=num_nodes,
                          shuffle="targets", directed=True)

    assert np.all(g.get_degrees("out") == k_lattice)

    reciprocity = 0.5

    g = ng.watts_strogatz(k_lattice, p_shortcut, reciprocity, nodes=num_nodes,
                          directed=True)

    recip_fact = 0.5*(1 + reciprocity / (2 - reciprocity))

    assert g.edge_nb() == int(np.round(recip_fact*k_lattice*num_nodes))

    # limit cases
    for p_shortcut in (0, 1):
        g = ng.watts_strogatz(k_lattice, p_shortcut, nodes=num_nodes,
                              directed=False)

        assert g.edge_nb() == 0.5*k_lattice*num_nodes

        g = ng.watts_strogatz(k_lattice, p_shortcut, nodes=num_nodes,
                              directed=True)

        assert g.edge_nb() == k_lattice*num_nodes


@pytest.mark.mpi
def test_mpi_from_degree_list():
    '''
    Check that the degrees generated using `from_degree_list` indeed
    correspond to the provided list
    '''
    num_nodes = 1000

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    deg_list = np.random.randint(0, int(0.1*num_nodes), size=num_nodes)

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


@pytest.mark.mpi_skip
def test_total_undirected_connectivities():
    ''' Test total-degree connectivities '''
    num_nodes = 1000

    # erdos-renyi
    density = 0.1

    lower, upper = 0.3, 5.4

    weights = {"distribution": "uniform", "lower": lower, "upper": upper}

    g = ng.erdos_renyi(density=density, nodes=num_nodes, directed=False,
                       weights=weights)

    assert g.edge_nb() / (num_nodes*num_nodes) == density

    # check weights

    ww = g.get_weights()

    assert np.all((lower <= ww) * (ww <= upper))

    # check other graph types
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


@pytest.mark.mpi_skip
def test_all_to_all():
    ''' Test all-to-all connection scheme '''
    num_nodes = 4

    # via direct generation call
    g = ng.all_to_all(nodes=num_nodes, directed=False)

    assert np.array_equal(
        g.edges_array, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    g = ng.all_to_all(nodes=num_nodes, directed=True)

    assert np.array_equal(
        g.edges_array, [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3),
                        (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)])

    # via connector call
    g = nngt.Graph(num_nodes)

    ng.connect_nodes(g, [0, 1], [2, 3], "all_to_all")

    assert np.array_equal(g.edges_array, [(0, 2), (0, 3), (1, 2), (1, 3)])

    g = nngt.Graph(num_nodes)

    ng.connect_nodes(g, [0, 1], [1, 2, 3], "all_to_all")

    assert np.array_equal(g.edges_array,
                          [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])


@pytest.mark.mpi_skip
def test_distances():
    ''' Check that distances are properly generated for SpatialGraphs '''
    # simple graph
    # ~ num_nodes = 4

    # ~ pos = [(0, 0), (1, 0), (2, 0), (3, 0)]
    
    # ~ g = nngt.SpatialGraph(num_nodes, positions=pos)

    # ~ edges = [(0, 1), (0, 3), (1, 2), (2, 3)]

    # ~ g.new_edges(edges)

    # ~ dist = g.edge_attributes["distance"]

    # ~ expected = np.abs(np.diff(g.edges_array, axis=1)).ravel()

    # ~ assert np.array_equal(dist, expected)

    # ~ g.new_node(positions=[(4, 0)])
    # ~ g.new_edge(1, 4)

    # ~ assert g.get_edge_attributes((1, 4), "distance") == 3

    # ~ # distance rule
    # ~ g = ng.distance_rule(2.5, rule="lin", nodes=num_nodes, avg_deg=2,
                         # ~ positions=pos)

    # ~ dist = g.edge_attributes["distance"]

    # ~ expected = np.abs(np.diff(g.edges_array, axis=1)).ravel()

    # ~ assert np.array_equal(dist, expected)
    # ~ assert np.all(dist < 3)

    # using the connector functions
    num_nodes = 20

    pop = nngt.NeuralPop.exc_and_inhib(num_nodes)
    pos = np.array([(i, 0) for i in range(num_nodes)])

    net = nngt.SpatialNetwork(pop, positions=pos)

    inh = pop["inhibitory"]
    exc = pop["excitatory"]

    ng.connect_groups(net, exc, pop, "erdos_renyi", avg_deg=5)
    ng.connect_groups(net, inh, pop, "random_scale_free", in_exp=2.1,
                      out_exp=2.1, avg_deg=5)

    dist = net.edge_attributes["distance"]

    expected = np.abs(np.diff(net.edges_array, axis=1)).ravel()

    assert np.array_equal(dist, expected)


@pytest.mark.mpi_skip
def test_price():
    ''' Test Price network '''
    # directed
    m = 5
    g = ng.price_scale_free(m, nodes=100)

    in_degrees  = g.get_degrees("in")
    out_degrees = g.get_degrees("out")

    assert set(out_degrees) == {i for i in range(m + 1)}
    assert in_degrees.min() == 0

    # undirected
    g = ng.price_scale_free(m, nodes=100, directed=False)

    degrees = g.get_degrees()

    assert np.all(degrees >= m)

    # reciprocity
    g = ng.price_scale_free(m, nodes=100, reciprocity=1)

    assert np.all(degrees >= m)
    assert na.reciprocity(g) == 1

    r = 0.3

    g = ng.price_scale_free(m, nodes=100, reciprocity=r)

    E  = g.edge_nb()
    Er = 2 * r * E  / (1 + r)

    rmin = (Er - 4*np.sqrt(Er)) / E
    rmax = (Er + 4*np.sqrt(Er)) / E

    assert rmin < na.reciprocity(g) < rmax


@pytest.mark.mpi_skip
def test_connect_switch_distance_rule_max_proba():
    num_omp = nngt.get_config("omp")
    mthread = nngt.get_config("multithreading")

    # switch multithreading to False
    nngt.set_config("multithreading", False)

    pop = nngt.NeuralPop.exc_and_inhib(1000)

    radius = 100.

    shape = nngt.geometry.Shape.disk(radius)

    net = nngt.SpatialNetwork(population=pop, shape=shape)

    max_proba = 0.1

    avg, std = 10., 1.5

    weights = {"distribution": "gaussian", "avg": avg, "std": std}

    ng.connect_nodes(net, pop.inhibitory, pop.excitatory, "distance_rule",
                     scale=5*radius, max_proba=max_proba, weights=weights)

    assert net.edge_nb() <= len(pop.inhibitory)*len(pop.excitatory)*max_proba

    # check weights
    ww = net.get_weights()

    assert avg - 0.5*std < ww.mean() < avg + 0.5*std
    assert 0.75*std < ww.std() < 1.25*std

    # restore mt parameters
    nngt.set_config("mpi", False)
    nngt.set_config("omp", num_omp)
    nngt.set_config("multithreading", mthread)


@pytest.mark.mpi_skip
def test_circular():
    ''' Test the circular graph generation methods. '''
    num_nodes = 1000
    coord_nb  = 4

    # undirected
    gc = ng.circular(coord_nb, nodes=num_nodes, directed=False)

    assert gc.node_nb() == num_nodes
    assert gc.edge_nb() == int(0.5*num_nodes*coord_nb)
    assert np.array_equal(gc.get_degrees(), np.full(num_nodes, coord_nb))

    # directed (reciprocity one)
    gc = ng.circular(coord_nb, nodes=num_nodes)

    assert gc.node_nb() == num_nodes
    assert gc.edge_nb() == int(num_nodes*coord_nb)
    assert np.array_equal(gc.get_degrees(), np.full(num_nodes, 2*coord_nb))
    assert np.array_equal(gc.get_degrees("in"), np.full(num_nodes, coord_nb))
    assert np.array_equal(gc.get_degrees("out"), np.full(num_nodes, coord_nb))

    # directed reciprocity = 0.5
    recip = 0.5
    gc = ng.circular(coord_nb, nodes=num_nodes, reciprocity=recip)

    num_edges = int(np.round(
        0.5 * (1 + recip / (2 - recip)) * num_nodes * coord_nb))

    num_recip = 2 * int(np.round(
        0.5 * recip / (2 - recip) * num_nodes * coord_nb))

    assert gc.node_nb() == num_nodes
    assert gc.edge_nb() == num_edges
    assert np.isclose(na.reciprocity(gc), num_recip / num_edges)
    assert np.isclose(na.reciprocity(gc), recip, 1e-3)

    # directed reciprocity = 0.5
    recip = 0.3
    gc = ng.circular(coord_nb, nodes=num_nodes, reciprocity=recip)

    num_edges = int(np.round(
        0.5 * (1 + recip / (2 - recip)) * num_nodes * coord_nb))

    num_recip = 2 * int(np.round(
        0.5 * recip / (2 - recip) * num_nodes * coord_nb))

    assert gc.node_nb() == num_nodes
    assert gc.edge_nb() == num_edges
    assert np.isclose(na.reciprocity(gc), num_recip / num_edges)
    assert np.isclose(na.reciprocity(gc), recip, 1e-3)


@pytest.mark.mpi_skip
def test_sparse_clustered():
    ccs = np.linspace(0.1, 0.9, 4)

    num_nodes = 500

    degrees = [10, 40]

    methods = ["star-component", "sequential", "random", "central-node"]

    for directed in (True, False):
        # check errors
        with pytest.raises(ValueError):
            g = ng.sparse_clustered(0, nodes=num_nodes, avg_deg=10,
                                    directed=directed)

        with pytest.raises(RuntimeError):
            g = ng.sparse_clustered(1, nodes=num_nodes, avg_deg=10,
                                    directed=directed, rtol=1e-10)

        # check graphs
        for i, c in enumerate(ccs):
            for deg in degrees:
                if c*num_nodes > deg:
                    g = ng.sparse_clustered(
                        c, nodes=num_nodes, avg_deg=deg, connected=False,
                        directed=directed, rtol=0.09)

                    g = ng.sparse_clustered(
                        c, nodes=num_nodes, avg_deg=deg, directed=directed,
                        exact_edge_nb=True)

                    assert g.edge_nb() == deg*num_nodes

                    g = ng.sparse_clustered(
                        c, nodes=num_nodes, avg_deg=deg, directed=directed,
                        connected=True, method=methods[i])

                    assert g.is_connected()


if __name__ == "__main__":
    if not nngt.get_config("mpi"):
        test_circular()
        test_newman_watts()
        test_from_degree_list()
        test_total_undirected_connectivities()
        test_watts_strogatz()
        test_all_to_all()
        test_distances()
        test_price()
        test_connect_switch_distance_rule_max_proba()
        test_sparse_clustered()

    if nngt.get_config("mpi"):
        test_mpi_from_degree_list()

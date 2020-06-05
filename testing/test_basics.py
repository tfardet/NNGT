#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_basics.py


"""
Test the validity of the most basic graph functions.
"""

import pytest

import numpy as np

import nngt
from nngt.lib import InvalidArgument


tolerance = 1e-6


def test_config():
    '''
    Check get/set_config functions.
    '''
    old_cfg = nngt.get_config(detailed=True)

    # get config
    cfg = nngt.get_config()

    for k, v in cfg.items():
        assert v == nngt.get_config(k)

    cfg_detailed = nngt.get_config(detailed=True)

    for k, v in cfg_detailed.items():
        assert v == nngt.get_config(k)

    # set config (omp)
    num_omp = 2
    nngt.set_config("omp", num_omp)

    assert nngt.get_config("multithreading")
    assert nngt.get_config("omp") == num_omp

    # set config (mpi)
    nngt.set_config("mpi", True)

    assert nngt.get_config("mpi")
    assert not nngt.get_config("multithreading")
    assert nngt.get_config("omp") == 1

    # key error
    key_error = False

    try:
        nngt.set_config("random_entry", "plop")
    except KeyError:
        key_error = True

    assert key_error

    # except for "palette"
    nngt.set_config("palette", "viridis")

    # restore old config
    nngt.set_config(old_cfg)


@pytest.mark.mpi_skip
def test_node_creation():
    '''
    When making graphs, test node creation function.
    '''
    g = nngt.Graph(100, name="new_node_test")

    assert g.node_nb() == 100, \
        "Error on '{}': invalid initial nodes ({} vs {} expected).".format(
            g.name, g.node_nb(), 100)

    n = g.new_node()
    assert g.node_nb() == 101 and n == 100, \
        "Error on '{}': ({}, {}) vs (101, 100) expected.".format(
            g.name, g.node_nb(), n)

    nn = g.new_node(2)

    assert g.node_nb() == 103 and tuple(nn) == (101, 102), \
        "Error on '{}': ({}, {}, {}) vs (103, 101, 102) expected.".format(
            g.name, g.node_nb(), nn[0], nn[1])


@pytest.mark.mpi_skip
def test_edge_creation():
    ''' Check edge checks '''
    num_nodes = 10

    # DIRECTED
    edges = [(0, 1), (2, 4)]

    g = nngt.Graph(num_nodes)
    g.new_edges(edges)

    error_raised = False

    # all following should trigger an error
    for e in [(11, 10), (4, 4), (0, 1)]:
        error_raised = False

        try:
            g.new_edge(*e)
        except InvalidArgument:
            error_raised = True

        assert error_raised

    # all following should also trigger an error
    lst_edges = [
        [(1, 1), (0, 2), (3, 4)],   # self-loop
        [(9, 4), (5, 6), (9, 4)],   # duplicate
        [(2, 4), (3, 4), (7, 2)],   # existing
        [(20, 1), (4, 8), (3, 2)],  # out-of-range
    ]

    for elist in lst_edges:
        error_raised = False

        try:
            g.new_edges(elist)
        except InvalidArgument:
            error_raised = True

        assert error_raised

    # check specific filters
    # self-loop
    error_raised = False

    try:
        g.new_edges(lst_edges[0], check_duplicates=False, check_existing=False)
    except InvalidArgument:
        error_raised = True

    assert error_raised

    # duplicate
    error_raised = False

    try:
        g.new_edges(lst_edges[1], check_duplicates=True,
                    check_self_loops=False, check_existing=False)
    except InvalidArgument:
        error_raised = True

    assert error_raised

    # existing
    error_raised = False

    try:
        g.new_edges(lst_edges[2], check_self_loops=False)
    except InvalidArgument:
        error_raised = True

    assert error_raised

    # out-of-range
    error_raised = False

    try:
        g.new_edges(lst_edges[3], check_self_loops=False, check_existing=False)
    except InvalidArgument:
        error_raised = True

    assert error_raised

    # working calls
    g.new_edge(0, 1, ignore=True)
    assert g.edge_id((0, 1)) == 0

    g.set_weights(5.)
    g.new_edge(0, 1, attributes={"weight": 3.}, ignore=True)
    assert g.get_weights(edges=(0, 1)) == 5

    g.new_edges(lst_edges[0], check_self_loops=False)
    assert g.edge_nb() == 5

    for elist in lst_edges[:-1]:
        g.new_edges(elist, ignore_invalid=True)

    assert g.edge_nb() == 8

    # UNDIRECTED
    g = nngt.Graph(num_nodes, directed=False)
    g.new_edges(edges)

    # all following should trigger an error
    for e in [(1, 0), (0, 1)]:
        error_raised = False

        try:
            g.new_edge(*e)
        except InvalidArgument:
            error_raised = True

        assert error_raised

    # all following should also trigger an error
    lst_edges = [
        [(1, 1), (0, 2), (3, 4)],  # self-loop
        [(9, 4), (5, 6), (9, 4)],  # duplicate
        [(2, 4), (3, 4), (7, 2)],  # existing
        [(4, 2), (4, 8), (3, 2)],  # existing 2
    ]

    for elist in lst_edges:
        error_raised = False

        try:
            g.new_edges(elist)
        except InvalidArgument:
            error_raised = True

        assert error_raised

    # working calls
    g.new_edge(0, 1, ignore=True)
    assert g.edge_id((0, 1)) == 0

    g.set_weights(5.)
    g.new_edge(1, 0, attributes={"weight": 3.}, ignore=True)
    assert g.get_weights(edges=(0, 1)) == 5

    g.new_edges(lst_edges[0], check_self_loops=False)
    assert g.edge_nb() == 5


    for elist in lst_edges:
        g.new_edges(elist, ignore_invalid=True)

    assert g.edge_nb() == 10


@pytest.mark.mpi_skip
def test_has_edges_edge_id():
    ''' Test the ``has_edge`` and ``edge_id`` methods '''
    num_nodes = 10

    # DIRECTED
    edges = [(0, 1), (2, 4)]

    g = nngt.Graph(num_nodes)
    g.new_edges(edges)

    for i, e in enumerate(edges):
        assert g.has_edge(e)
        assert g.edge_id(e) == i

    # UNDIRECTED
    g = nngt.Graph(num_nodes, directed=False)
    g.new_edges(edges)

    for i, e in enumerate(edges):
        assert g.has_edge(e)
        assert g.edge_id(e) == i
        assert g.has_edge(e[::-1])
        assert g.edge_id(e[::-1]) == i


def test_new_node_attr():
    '''
    Test node creation with attributes.
    '''
    shape = nngt.geometry.Shape.rectangle(1000., 1000.)
    g = nngt.SpatialGraph(100, shape=shape, name="new_node_spatial")

    assert g.node_nb() == 100, \
        "Error on '{}': invalid initial nodes ({} vs {} expected).".format(
            g.name, g.node_nb(), 100)

    n = g.new_node(positions=[(0, 0)])

    assert np.all(np.isclose(g.get_positions(n), (0, 0), tolerance)), \
        "Error on '{}': last position is ({}, {}) vs (0, 0) expected.".format(
            g.name, *g.get_positions(n))


def test_graph_copy():
    '''
    Test partial and full graph copy.
    '''
    # partial copy
    # non-spatial graph
    avg = 20
    std = 4

    g = nngt.generation.gaussian_degree(avg, std, nodes=100)

    h = nngt.Graph(copy_graph=g)

    assert g.node_nb() == h.node_nb()
    assert g.edge_nb() == h.edge_nb()

    assert np.array_equal(g.edges_array, h.edges_array)

    # spatial network
    pop   = nngt.NeuralPop.exc_and_inhib(100)
    shape = nngt.geometry.Shape.rectangle(1000., 1000.)

    g = nngt.generation.gaussian_degree(avg, std, population=pop, shape=shape,
                                        name="new_node_spatial")

    h = nngt.Graph(copy_graph=g)

    assert g.node_nb() == h.node_nb()
    assert g.edge_nb() == h.edge_nb()

    assert np.array_equal(g.edges_array, h.edges_array)

    assert not h.is_network()
    assert not h.is_spatial()
    
    # full copy
    copy = g.copy()

    assert g.node_nb() == h.node_nb()
    assert g.edge_nb() == h.edge_nb()

    assert np.array_equal(g.edges_array, h.edges_array)

    assert g.population == copy.population
    assert g.population is not copy.population

    assert g.shape == copy.shape
    assert g.shape is not copy.shape


def test_degrees_neighbors():
    '''
    Check ``Graph.get_degrees`` method.
    '''
    edge_list = [(0, 1), (0, 2), (0, 3), (1, 3), (3, 2), (3, 4), (4, 2)]
    weights   = [0.54881, 0.71518, 0.60276, 0.54488, 0.42365, 0.64589, 0.43758]

    out_degrees = np.array([3, 1, 0, 2, 1])
    in_degrees  = np.array([0, 1, 3, 2, 1])
    tot_degrees = in_degrees + out_degrees

    out_strengths = np.array([1.86675, 0.54488, 0, 1.06954, 0.43758])
    in_strengths  = np.array([0, 0.54881, 1.57641, 1.14764, 0.64589])
    tot_strengths = in_strengths + out_strengths

    # DIRECTED
    g = nngt.Graph(5, directed=True)
    g.new_edges(edge_list, attributes={"weight": weights})

    assert np.all(g.get_degrees(mode="in") == in_degrees)
    assert np.all(g.get_degrees(mode="out") == out_degrees)
    assert np.all(g.get_degrees() == tot_degrees)

    assert np.all(
        np.isclose(g.get_degrees(mode="in", weights=True), in_strengths))
    assert np.all(
        np.isclose(g.get_degrees(mode="out", weights=True), out_strengths))
    assert np.all(np.isclose(g.get_degrees(weights="weight"), tot_strengths))

    assert g.neighbours(3, "in")  == {0, 1}
    assert g.neighbours(3, "out") == {2, 4}
    assert g.neighbours(3, "all") == {0, 1, 2, 4}

    # UNDIRECTED
    g = nngt.Graph(5, directed=False)
    g.new_edges(edge_list, attributes={"weight": weights})

    assert np.all(g.get_degrees(mode="in") == tot_degrees)
    assert np.all(g.get_degrees(mode="out") == tot_degrees)
    assert np.all(g.get_degrees() == tot_degrees)

    assert np.all(
        np.isclose(g.get_degrees(mode="in", weights=True), tot_strengths))
    assert np.all(
        np.isclose(g.get_degrees(mode="out", weights=True), tot_strengths))
    assert np.all(np.isclose(g.get_degrees(weights="weight"), tot_strengths))

    assert g.neighbours(3, "in")  == {0, 1, 2, 4}
    assert g.neighbours(3, "out") == {0, 1, 2, 4}
    assert g.neighbours(3, "all") == {0, 1, 2, 4}


def test_directed_adjacency():
    ''' Check directed adjacency matrix '''
    num_nodes = 5
    edge_list = [(0, 1), (0, 3), (1, 3), (2, 0), (3, 2), (3, 4), (4, 2)]
    weights   = [0.54881, 0.71518, 0.60276, 0.54488, 0.42365, 0.64589, 0.43758]
    etypes    = [-1, 1, 1, -1, -1, 1, 1]

    g = nngt.Graph(num_nodes)
    g.new_edges(edge_list, attributes={"weight": weights})
    g.new_edge_attribute("type", "int", values=etypes)

    adj_mat = np.array([
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0]
    ])

    assert np.all(np.isclose(g.adjacency_matrix(weights=False).todense(),
                             adj_mat))

    w_mat = np.array([
        [0,       0.54881, 0,       0.71518, 0      ],
        [0,       0,       0,       0.60276, 0      ],
        [0.54488, 0,       0,       0,       0      ],
        [0,       0,       0.42365, 0,       0.64589],
        [0,       0,       0.43758, 0,       0]
    ])

    assert np.all(np.isclose(g.adjacency_matrix(weights=True).todense(),
                             w_mat))

    # for typed edges
    tpd_mat = np.array([
        [ 0, -1,  0, 1, 0],
        [ 0,  0,  0, 1, 0],
        [-1,  0,  0, 0, 0],
        [ 0,  0, -1, 0, 1],
        [ 0,  0,  1, 0, 0]
    ])

    assert np.all(np.isclose(g.adjacency_matrix(types=True).todense(),
                             tpd_mat))

    wt_mat = np.array([
        [ 0,       -0.54881,  0,       0.71518, 0      ],
        [ 0,        0,        0,       0.60276, 0      ],
        [-0.54488,  0,        0,       0,       0      ],
        [ 0,        0,       -0.42365, 0,       0.64589],
        [ 0,        0,        0.43758, 0,       0]
    ])

    assert np.all(np.isclose(
        g.adjacency_matrix(types=True, weights=True).todense(), wt_mat))

    # for Network and node attribute type
    num_nodes = 5
    edge_list = [(0, 1), (0, 3), (1, 3), (2, 0), (3, 2), (3, 4), (4, 2)]
    weights   = [0.54881, 0.71518, 0.60276, 0.54488, 0.42365, 0.64589, 0.43758]

    inh = nngt.NeuralGroup([0, 2, 4], neuron_type=-1, name="inh")
    exc = nngt.NeuralGroup([1, 3], neuron_type=1, name="exc")
    pop = nngt.NeuralPop.from_groups((inh, exc), with_models=False)

    net = nngt.Network(population=pop)
    net.new_edges(edge_list, attributes={"weight": weights})

    g = nngt.Graph(num_nodes)
    g.new_node_attribute("type", "int", values=[-1, 1, -1, 1, -1])
    g.new_edges(edge_list, attributes={"weight": weights})

    tpd_mat = np.array([
        [ 0, -1,  0, -1, 0],
        [ 0,  0,  0,  1, 0],
        [-1,  0,  0,  0, 0],
        [ 0,  0,  1,  0, 1],
        [ 0,  0, -1,  0, 0]
    ])

    assert np.all(np.isclose(net.adjacency_matrix(types=True).todense(),
                             tpd_mat))

    assert np.all(np.isclose(g.adjacency_matrix(types=True).todense(),
                             tpd_mat))

    wt_mat = np.array([
        [ 0,       -0.54881,  0,       -0.71518, 0      ],
        [ 0,        0,        0,        0.60276, 0      ],
        [-0.54488,  0,        0,        0,       0      ],
        [ 0,        0,        0.42365,  0,       0.64589],
        [ 0,        0,       -0.43758,  0,       0]
    ])

    assert np.all(np.isclose(
        net.adjacency_matrix(types=True, weights=True).todense(), wt_mat))

    assert np.all(np.isclose(
        g.adjacency_matrix(types=True, weights=True).todense(), wt_mat))


def test_undirected_adjacency():
    ''' Check undirected adjacency matrix '''
    num_nodes = 5
    edge_list = [(0, 1), (0, 3), (1, 3), (2, 0), (3, 2), (3, 4), (4, 2)]
    weights   = [0.54881, 0.71518, 0.60276, 0.54488, 0.42365, 0.64589, 0.43758]
    etypes    = [-1, 1, 1, -1, -1, 1, 1]

    g = nngt.Graph(num_nodes, directed=False)
    g.new_edges(edge_list, attributes={"weight": weights})
    g.new_edge_attribute("type", "int", values=etypes)

    adj_mat = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])

    assert np.all(np.isclose(g.adjacency_matrix(weights=False).todense(),
                             adj_mat))

    w_mat = np.array([
        [0,       0.54881, 0.54488, 0.71518, 0      ],
        [0.54881, 0,       0,       0.60276, 0      ],
        [0.54488, 0,       0,       0.42365, 0.43758],
        [0.71518, 0.60276, 0.42365, 0,       0.64589],
        [0,       0,       0.43758, 0.64589, 0      ]
    ])

    assert np.all(np.isclose(g.adjacency_matrix(weights=True).todense(),
                             w_mat))

    # for typed edges
    tpd_mat = np.array([
        [ 0, -1, -1,  1, 0],
        [-1,  0,  0,  1, 0],
        [-1,  0,  0, -1, 1],
        [ 1,  1, -1,  0, 1],
        [ 0,  0,  1,  1, 0]
    ])

    assert np.all(np.isclose(g.adjacency_matrix(types=True).todense(),
                             tpd_mat))

    wt_mat = np.array([
        [ 0,       -0.54881, -0.54488,  0.71518, 0      ],
        [-0.54881,  0,        0,        0.60276, 0      ],
        [-0.54488,  0,        0,       -0.42365, 0.43758],
        [ 0.71518,  0.60276, -0.42365,  0,       0.64589],
        [ 0,        0,        0.43758,  0.64589, 0      ]
    ])

    assert np.all(np.isclose(
        g.adjacency_matrix(types=True, weights=True).todense(), wt_mat))


# ---------- #
# Test suite #
# ---------- #

if __name__ == "__main__":
    test_directed_adjacency()
    test_undirected_adjacency()
    test_config()
    test_new_node_attr()
    test_graph_copy()
    test_degrees_neighbors()

    if not nngt.get_config('mpi'):
        test_node_creation()
        test_edge_creation()
        test_has_edges_edge_id()

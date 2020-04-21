#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_basics.py


"""
Test the validity of the most basic graph functions.
"""

import pytest

import numpy as np

import nngt


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
    g = nngt.generation.erdos_renyi(density=0.1, nodes=100)

    h = nngt.Graph(copy_graph=g)

    assert g.node_nb() == h.node_nb()
    assert g.edge_nb() == h.edge_nb()

    assert np.array_equal(g.edges_array, h.edges_array)

    # spatial network
    pop   = nngt.NeuralPop.exc_and_inhib(100)
    shape = nngt.geometry.Shape.rectangle(1000., 1000.)

    g = nngt.generation.erdos_renyi(density=0.1, population=pop, shape=shape,
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


def test_adjacency():
    ''' Check adjacency matrix '''
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


# ---------- #
# Test suite #
# ---------- #

if __name__ == "__main__":
    test_adjacency()
    test_config()
    test_new_node_attr()
    test_graph_copy()

    if not nngt.get_config('mpi'):
        test_node_creation()

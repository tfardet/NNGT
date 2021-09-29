#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_basics.py


"""
Test the validity of the most basic graph functions.
"""

import pytest

import numpy as np
import numpy.testing as npt

import nngt
import nngt.generation as ng
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
    has_mpi = False

    try:
        import mpi4py
        has_mpi = True
    except:
        pass

    if has_mpi:
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

    # except for palettes
    nngt.set_config("palette_continuous", "viridis")
    nngt.set_config("palette_discrete", "Set2")

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

    # empty edge list
    assert not g.new_edges([])

    # all following should trigger an error
    for e in [(11, 10), (4, 4), (0, 1)]:
        with pytest.raises(InvalidArgument):
            g.new_edge(*e)

    # all following should also trigger an error
    lst_edges = [
        [(1, 1), (0, 2), (3, 4)],   # self-loop
        [(9, 4), (5, 6), (9, 4)],   # duplicate
        [(2, 4), (3, 4), (7, 2)],   # existing
        [(20, 1), (4, 8), (3, 2)],  # out-of-range
    ]

    for elist in lst_edges:
        with pytest.raises(InvalidArgument):
            g.new_edges(elist)

    # check specific filters
    # self-loop
    with pytest.raises(InvalidArgument):
        g.new_edges(lst_edges[0], check_duplicates=False, check_existing=False)

    # duplicate
    with pytest.raises(InvalidArgument):
        g.new_edges(lst_edges[1], check_duplicates=True,
                    check_self_loops=False, check_existing=False)

    # existing
    with pytest.raises(InvalidArgument):
        g.new_edges(lst_edges[2], check_self_loops=False)

    # out-of-range
    with pytest.raises(InvalidArgument):
        g.new_edges(lst_edges[3], check_self_loops=False, check_existing=False)

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
        with pytest.raises(InvalidArgument):
            g.new_edge(*e)

    # all following should also trigger an error
    lst_edges = [
        [(1, 1), (0, 2), (3, 4)],  # self-loop
        [(9, 4), (5, 6), (9, 4)],  # duplicate
        [(2, 4), (3, 4), (7, 2)],  # existing
        [(4, 2), (4, 8), (3, 2)],  # existing 2
    ]

    for elist in lst_edges:
        with pytest.raises(InvalidArgument):
            g.new_edges(elist)

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


@pytest.mark.mpi_skip
def test_graph_copy():
    '''
    Test partial and full graph copy.
    '''
    # partial copy
    # non-spatial graph
    avg = 20
    std = 4

    g = ng.gaussian_degree(avg, std, nodes=100)

    h = nngt.Graph(copy_graph=g)

    assert g.node_nb() == h.node_nb()
    assert g.edge_nb() == h.edge_nb()

    assert np.array_equal(g.edges_array, h.edges_array)

    # spatial network
    pop   = nngt.NeuralPop.exc_and_inhib(100)
    shape = nngt.geometry.Shape.rectangle(1000., 1000.)

    g = ng.gaussian_degree(avg, std, population=pop, shape=shape,
                           name="new_node_spatial")

    h = nngt.Graph(copy_graph=g)

    assert g.node_nb() == h.node_nb()
    assert g.edge_nb() == h.edge_nb()

    assert np.array_equal(g.edges_array, h.edges_array)

    assert not h.is_network()
    assert not h.is_spatial()
    
    # full copy
    rng = np.random.default_rng()

    g.set_weights(rng.uniform(0, 10, g.edge_nb()))

    g.new_node_attribute("plop", "int", rng.integers(1, 50, g.node_nb()))
    g.new_node_attribute("bip", "double", rng.uniform(0, 1, g.node_nb()))
    g.new_edge_attribute("test", "int", rng.integers(1, 200, g.edge_nb()))

    copy = g.copy()

    assert g.node_nb() == copy.node_nb()
    assert g.edge_nb() == copy.edge_nb()

    assert np.array_equal(g.edges_array, copy.edges_array)

    for k, v in g.edge_attributes.items():
        npt.assert_array_equal(v, copy.edge_attributes[k])

    for k, v in g.node_attributes.items():
        npt.assert_array_equal(v, copy.node_attributes[k])

    assert g.population == copy.population
    assert g.population is not copy.population

    assert g.shape == copy.shape
    assert g.shape is not copy.shape

    # check that undirected graph stays undirected
    g = ng.erdos_renyi(nodes=100, avg_deg=10, directed=False)

    h = g.copy()

    assert g.is_directed() == h.is_directed() == False

    # eid is protected and should not be copied to a visible edge attribute
    assert "eid" not in h.edge_attributes


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


@pytest.mark.mpi_skip
def test_get_edges():
    ''' Check that correct edges are returned '''
    # directed
    g = nngt.Graph(4, directed=True)

    edges = [(0, 1), (1, 0), (1, 2), (2, 3)]

    g.new_edges(edges)

    def to_set(ee):
        return {tuple(e) for e in ee}

    assert g.get_edges(source_node=0) == [(0, 1)]
    assert g.get_edges(target_node=2) == [(1, 2)]
    assert to_set(g.get_edges(source_node=[0, 1])) == to_set(edges[:3])
    assert to_set(g.get_edges(target_node=[0, 1])) == to_set(edges[:2])
    assert to_set(g.get_edges(source_node=[0, 2],
                              target_node=[0, 1])) == {(0, 1)}

    # undirected
    g = nngt.Graph(4, directed=False)

    edges = [(0, 1), (1, 2), (2, 3)]

    g.new_edges(edges)
    
    res = {(0, 1), (1, 2)}

    assert g.get_edges(source_node=0) == [(0, 1)]
    assert to_set(g.get_edges(target_node=2)) == {(1, 2), (2, 3)}
    assert to_set(g.get_edges(source_node=[0, 1])) == res
    assert to_set(g.get_edges(target_node=[0, 1])) == res
    assert to_set(g.get_edges(source_node=[0, 2],
                              target_node=[0, 1])) == res

    assert to_set(g.get_edges(source_node=0, target_node=1)) == {(0, 1)}
    assert to_set(g.get_edges(source_node=0, target_node=[0, 1])) == {(0, 1)}


@pytest.mark.mpi_skip
def test_delete():
    ''' Test node and edge deletion '''
    mat = np.array([
        [0.,  0.5, 0.,  0.2, 0.,  1. ],
        [0.,  0.,  0.5, 0.,  0.3, 0. ],
        [0.1, 0.,  0.,  0.,  1.,  0. ],
        [0.,  0.2, 1.,  0.,  0.,  0. ],
        [0.5, 0.,  0.,  0.,  0.,  0.5],
        [0.,  0.1, 0.,  0.,  0.,  0. ],
    ])

    num_nodes = 6
    num_edges = 12

    in_deg = [2, 3, 2, 1, 2, 2]
    out_deg = [3, 2, 2, 2, 2, 1]

    # make positions and structure
    rng = np.random.default_rng()

    positions = rng.uniform(-5, 5, size=(num_nodes, 2))

    g1 = nngt.Group([1, 2, 3])
    g2 = nngt.Group([0, 4, 5])
    struct = nngt.Structure.from_groups([g1, g2], ["g1", "g2"])

    g = nngt.Graph.from_matrix(mat, positions=positions, structure=struct)

    g.new_node_attribute("test", "int", values=list(range(num_nodes)))

    assert num_edges == g.edge_nb()

    # delete one edge (eid = 2)
    edge = (0, 5)

    in_deg[5] -= 1
    out_deg[0] -= 1

    g.delete_edges(edge)

    num_edges -= 1

    assert g.edge_nb() == num_edges

    assert len(g.get_weights()) == num_edges

    assert np.array_equal(in_deg, g.get_degrees("in"))
    assert np.array_equal(out_deg, g.get_degrees("out"))

    mat[edge] = 0

    adj = g.adjacency_matrix(weights=True, mformat="dense")

    assert np.all(np.isclose(mat, adj))

    # delete several edges (eids = (3, 6))
    edges = [(1, 4), (3, 1)]

    g.delete_edges(edges)

    in_deg[4] -= 1
    out_deg[1] -= 1
    in_deg[1] -= 1
    out_deg[3] -= 1

    num_edges -= len(edges)

    assert g.edge_nb() == num_edges

    assert len(g.get_weights()) == num_edges

    assert np.array_equal(in_deg, g.get_degrees("in"))
    assert np.array_equal(out_deg, g.get_degrees("out"))

    for e in edges:
        mat[e] = 0

    adj = g.adjacency_matrix(weights=True, mformat="dense")

    assert np.all(np.isclose(mat, adj))

    # deleting one node
    g.delete_nodes([0])

    adj = g.adjacency_matrix(weights=True, mformat="dense")

    assert g.node_nb() == 5

    assert np.array_equal(adj, mat[1:, 1:])

    assert np.array_equal(g.node_attributes["test"], [1, 2, 3, 4, 5])

    assert set(g.structure["g2"].ids) == {3, 4}

    assert np.array_equal(g.get_positions(), positions[1:])

    # delete two nodes
    g.delete_nodes([1, 2])

    assert g.node_nb() == 3

    adj = g.adjacency_matrix(weights=True, mformat="dense")

    assert np.array_equal(adj, mat[[1, 4, 5]][:, [1, 4, 5]])

    assert np.array_equal(g.node_attributes["test"], [1, 4, 5])

    assert set(g.structure["g1"].ids) == {0}
    assert set(g.structure["g2"].ids) == {1, 2}

    assert np.array_equal(g.get_positions(), positions[[1, 4, 5]])

    # readd nodes
    g.new_node(2, attributes={"test": [-1, 2]}, positions=[(-2, 1), (0.5, 3)])

    assert g.node_nb() == 5

    assert np.array_equal(g.node_attributes["test"], [1, 4, 5, -1, 2])

    # readd edges
    g.new_edges([(1, 4), (4, 3)])

    assert g.edge_nb() == 4

    assert np.array_equal(g.edges_array, [(1, 2), (2, 0), (1, 4), (4, 3)])

    assert np.all(np.isclose(
        g.get_weights(), [0.5, 0.1, 1., 1.], equal_nan=True))

    # test delete from get_edges (issue #136)
    edges = g.get_edges()

    g.delete_edges(edges[:2])

    # test copy after edge deletion
    h = g.copy()

    assert np.all(np.isclose(h.get_weights(), g.get_weights()))
    assert np.all(np.isclose(h.edge_attributes["distance"],
                             g.edge_attributes["distance"]))


def test_to_undirected():
    mat = np.array([
        [0,   2., 0.5, 0],
        [0,   0,   1., 0],
        [1.5, 0,    0, 1],
        [0,   1,  0.5, 0]
    ])

    g = nngt.Graph.from_matrix(mat)

    g.new_node_attribute("test", "int", [10, 20, 30, 40])
    g.new_node_attribute("alph", "string", ["d", "c", "b", "a"])

    g.new_edge_attribute("rnd", "int", [2, 6, 8, 4, 5, 3, 9])
    g.new_edge_attribute("alph", "string",
                         ["a", "e", "i", "o", "u", "y", "aa"])

    # undirected sum
    u = g.to_undirected()

    assert np.array_equal(u.node_attributes["test"], g.node_attributes["test"])
    assert list(u.node_attributes["alph"]) == list(g.node_attributes["alph"])

    assert set(u.edge_attributes) == {"weight", "rnd"}

    assert np.all(np.isclose(
        mat + mat.T, u.adjacency_matrix(weights="weight").todense()
    ))

    assert np.array_equal(u.edge_attributes["rnd"], [2, 10, 8, 3, 14])

    # make spatial
    pos = nngt._rng.uniform(size=(g.node_nb(), 2))
    g.make_spatial(g, positions=pos)

    # undirected max
    u = g.to_undirected("max")

    m = np.maximum(mat, mat.T)

    assert np.all(np.isclose(
        m, u.adjacency_matrix(weights="weight").todense()
    ))

    assert np.array_equal(u.edge_attributes["rnd"], [2, 6, 8, 3, 9])

    # make network
    pop = nngt.NeuralPop.uniform(g.node_nb())
    g.make_network(g, pop)

    # undirected min
    u = g.to_undirected("min")

    nnz = np.where(np.multiply(mat, mat.T))
    m   = mat + mat.T
    m[nnz] = np.minimum(mat[nnz], mat.T[nnz])

    assert np.all(np.isclose(
        m, u.adjacency_matrix(weights="weight").todense()
    ))

    assert np.array_equal(u.edge_attributes["rnd"], [2, 4, 8, 3, 5])

    # undirected mean
    u = g.to_undirected("mean")

    nnz = np.where(np.multiply(mat, mat.T))
    m   = mat + mat.T
    m[nnz] = 0.5*(mat[nnz] + mat.T[nnz])

    assert np.all(np.isclose(
        m, u.adjacency_matrix(weights="weight").todense()
    ))

    assert np.array_equal(u.edge_attributes["rnd"], [2, 5, 8, 3, 7])

    # undirected mean/max
    u = g.to_undirected({"weight": "mean", "rnd": "max"})

    assert np.all(np.isclose(
        m, u.adjacency_matrix(weights="weight").todense()
    ))

    assert np.array_equal(u.edge_attributes["rnd"], [2, 6, 8, 3, 9])

    # for an unweighted graph
    g = nngt.Graph.from_matrix(mat, weighted=False)
    u = g.to_undirected()

    assert not u.edge_attributes

    m = mat + mat.T
    nnz = np.where(m)
    m[nnz] = 1

    assert np.array_equal(u.adjacency_matrix().todense(), m)



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
    test_get_edges()
    test_to_undirected()

    if not nngt.get_config('mpi'):
        test_node_creation()
        test_edge_creation()
        test_has_edges_edge_id()
        test_delete()

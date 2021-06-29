#-*- coding:utf-8 -*-

# test_graphclasses.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the main methods of the :class:`~nngt.Graph` class and its subclasses.
"""

import unittest
import pytest

import numpy as np
import scipy.sparse as ssp

import nngt
import nngt.generation as ng

from base_test import TestBasis, XmlHandler, network_dir
from tools_testing import foreach_graph


# ---------- #
# Test class #
# ---------- #

class TestGraphClasses(TestBasis):

    '''
    Class testing the main methods of :class:`~nngt.Graph` and its subclasses.
    '''

    matrices = {}
    mat_gen = {
        "from_scipy_sparse_rand": ssp.rand,
        "from_numpy_randint": np.random.randint
    }

    @property
    def test_name(self):
        return "test_graphclasses"

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    def gen_graph(self, graph_name):
        di_instructions = self.parser.get_graph_options(graph_name)
        mat = self.mat_gen[graph_name](**di_instructions)
        self.matrices[graph_name] = mat
        graph = nngt.Graph.from_matrix(mat)
        graph.set_name(graph_name)
        return graph, di_instructions

    @foreach_graph
    def test_adj_mat(self, graph, **kwargs):
        '''
        When generating graphs from :class:`numpy.ndarray`s or
        :mod:`scipy.sparse` matrices, check that the result of
        graph.adjacency_matrix() is the same as the initial matrix.
        '''
        ref_result = ssp.csr_matrix(self.matrices[graph.name])
        computed_result = graph.adjacency_matrix(weights=True)
        self.assertTrue(
            (ref_result != computed_result).nnz == 0,
            "AdjMat test failed for graph {}:\nref = {} vs exp {}\
            ".format(graph.name, ref_result, computed_result))

    @foreach_graph
    def test_copy_clear(self, graph, **kwargs):
        '''
        Test that the copied graph is indeed the same as the original, but that
        all its properties are deep copies.
        Then check that clear_edges() removes all edges and no nodes.
        '''
        ref_result = (graph.node_nb(), graph.edge_nb(), graph.node_nb(), 0)
        copied = graph.copy()
        self.assertIsNot(copied, graph)
        computed_result = [copied.node_nb(), copied.edge_nb()]
        copied.clear_all_edges()
        computed_result.extend((copied.node_nb(), copied.edge_nb()))
        self.assertEqual(
            ref_result, tuple(computed_result),
            "Copy test failed for graph {}:\nref = {} vs exp {}\
            ".format(graph.name, ref_result, computed_result))


@pytest.mark.mpi_skip
def test_structure_graph():
    gclass = (nngt.Group, nngt.NeuralGroup)
    sclass = (nngt.Structure, nngt.NeuralPop)
    nclass = (nngt.Graph, nngt.Network)

    for gc, sc, nc in zip(gclass, sclass, nclass):
        room1 = gc(25, neuron_type=1)
        room2 = gc(50, neuron_type=1)
        room3 = gc(40, neuron_type=1)
        room4 = gc(35, neuron_type=1)

        names = ["R1", "R2", "R3", "R4"]

        kwargs = {"with_models": False} if sc == nngt.NeuralPop else {}
        struct = sc.from_groups((room1, room2, room3, room4), names, **kwargs)

        kwargs = ({"population": struct} if nc == nngt.Network
                  else {"structure": struct})

        g = nc(**kwargs)

        # connect groups
        for room in struct:
            ng.connect_groups(g, room, room, "all_to_all")

        d1 = 5
        ng.connect_groups(g, room1, room2, "erdos_renyi", avg_deg=d1)
        ng.connect_groups(g, room1, room3, "erdos_renyi", avg_deg=d1)
        ng.connect_groups(g, room1, room4, "erdos_renyi", avg_deg=d1)

        d2 = 5
        ng.connect_groups(g, room2, room3, "erdos_renyi", avg_deg=d2)
        ng.connect_groups(g, room2, room4, "erdos_renyi", avg_deg=d2,
                          weights=2)

        d3 = 20
        ng.connect_groups(g, room3, room1, "erdos_renyi", avg_deg=d3)

        d4 = 10
        ng.connect_groups(g, room4, room3, "erdos_renyi", avg_deg=d4)

        # get structure graph
        sg = g.get_structure_graph()

        assert sg.node_nb() == len(struct)

        eset = set([tuple(e) for e in sg.edges_array])
        expected = [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 2),
            (3, 2), (3, 3)
        ]

        assert eset == set(expected)

        # check weights
        w1 = room1.size * d1
        w2 = room2.size * d2
        w3 = room3.size * d3
        w4 = room4.size * d4

        expected_weights = [
            room1.size*(room1.size - 1), w1, w1, w1,
            room2.size*(room2.size - 1), w2, w2*2,
            w3, room3.size*(room3.size - 1),
            w4, room4.size*(room4.size - 1)
        ]

        assert np.array_equal(sg.get_weights(edges=expected), expected_weights)


@pytest.mark.mpi_skip
def test_autoclass():
    '''
    Check that Graph is automatically converted to Network or SpatialGraph
    if the relevant arguments are provided.
    '''
    pop = nngt.NeuralPop.exc_and_inhib(100)

    g = nngt.Graph(population=pop)

    assert isinstance(g, nngt.Network)

    shape = nngt.geometry.Shape.disk(50.)

    g = nngt.Graph(shape=shape)

    assert isinstance(g, nngt.SpatialGraph)

    g = nngt.Graph(population=pop, shape=shape)

    assert isinstance(g, nngt.SpatialNetwork)


# ---------- #
# Test suite #
# ---------- #

if not nngt.get_config('mpi'):
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphClasses)

    if __name__ == "__main__":
        unittest.main()
        test_structure_graph()
        test_autoclass()

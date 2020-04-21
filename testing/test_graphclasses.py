#-*- coding:utf-8 -*-

# test_graphclasses.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the main methods of the :class:`~nngt.Graph` class and its subclasses.
"""

import unittest

import numpy as np
import scipy.sparse as ssp

import nngt
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


# ---------- #
# Test suite #
# ---------- #

if not nngt.get_config('mpi'):
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphClasses)

    if __name__ == "__main__":
        unittest.main()

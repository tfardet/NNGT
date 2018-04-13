#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the IO functions.
"""

import os
import unittest

import numpy as np

import nngt
from nngt.lib.test_functions import _old_graph_tool
from base_test import TestBasis, XmlHandler, network_dir
from tools_testing import foreach_graph


current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
error = 'Wrong {{val}} for {graph}.'

nngt.set_config("multithreading", False)


# ---------- #
# Test class #
# ---------- #

class TestIO(TestBasis):
    
    '''
    Class testing saving and loading functions.
    '''
    
    @classmethod
    def tearDownClass(cls):
        for graphname in cls.graphs:
            try:
                os.remove(current_dir + graphname + '.el')
            except:
                pass
        try:
            os.remove(current_dir + 'test.el')
        except:
            pass
    
    @property
    def test_name(self):
        return "test_io"

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    @unittest.skipIf(_old_graph_tool('2.22'), 'Skip for graph-tool < 2.22.')
    def gen_graph(self, graph_name):
        # check whether we are loading from file
        if "." in graph_name:
            abspath = network_dir + graph_name
            di_instructions = self.parser.get_graph_options(graph_name)
            graph = nngt.Graph.from_file(abspath, **di_instructions)
            graph.set_name(graph_name)
            return graph, None
        else:
            di_instructions = self.parser.get_graph_options(graph_name)
            graph = nngt.generate(di_instructions)
            graph.set_name(graph_name)
            graph.to_file(current_dir + graph_name + '.el')
            return graph, di_instructions

    @foreach_graph
    def test_identical(self, graph, instructions, **kwargs):
        '''
        Test that the generated graph and the one loaded from the saved file
        are indeed identical.
        '''
        err = error.format(graph=graph.get_name())
        if instructions is not None:  # working with generated graph
            # load graph
            h = nngt.Graph.from_file(current_dir + graph.get_name() + '.el')
            attributes = h.edges_attributes
            # test properties
            self.assertTrue(h.node_nb() == graph.node_nb(),
                            err.format(val='node number'))
            self.assertTrue(h.edge_nb() == graph.edge_nb(),
                            err.format(val='edge number'))
            if graph.is_spatial():
                self.assertTrue(np.allclose(h.get_positions(),
                                            graph.get_positions()),
                                err.format(val='positions'))
            for attr, values in graph.edges_attributes.items():
                # different results probably because of rounding problems
                allclose = np.allclose(h.edges_attributes[attr], values, 1e-4)
                if not allclose:
                    print("Error: expected")
                    print(h.edges_attributes[attr])
                    print("but got")
                    print(values)
                    print("max error is: {}".format(
                        np.max(np.abs(np.subtract(
                            h.edges_attributes[attr], values)))))
                self.assertTrue(allclose, err.format(val=attr))
        else:  # working with loaded graph
            nodes = self.get_expected_result(graph, "nodes")
            edges = self.get_expected_result(graph, "edges")
            # check
            self.assertEqual(
                nodes, graph.node_nb(), err.format(val='node number'))
            self.assertEqual(
                edges, graph.edge_nb(), err.format(val='edge number'))

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    @unittest.skipIf(_old_graph_tool('2.22'), 'Skip for graph-tool < 2.22.')
    def test_custom_attributes(self):
        '''
        Test that custom attributes are saved and loaded correctly
        '''
        num_nodes = 1000
        avg_deg = 100

        g = nngt.Graph(nodes=num_nodes)
        g.new_edge_attribute("test_attr", "int")

        for i in range(num_nodes):
            targets = np.unique(np.random.randint(0, num_nodes, avg_deg))
            elist = np.zeros((len(targets), 2), dtype=int)
            elist[:, 0] = i
            elist[:, 1] = targets
            ids  = np.random.randint(0, avg_deg*num_nodes, len(targets))
            ids *= 2*np.random.randint(0, 2, len(targets)) - 1
            g.new_edges(elist, attributes={"test_attr": ids},
                        check_edges=False)

        g.to_file('test.el')
        h = nngt.Graph.from_file('test.el')

        allclose = np.allclose(g.get_edge_attributes(name="test_attr"),
                               h.get_edge_attributes(name="test_attr"))
        if not allclose:
            print("Results differed for '{}'.".format(g.name))
            print(g.get_edge_attributes(name="test_attr"))
            print(h.get_edge_attributes(name="test_attr"))
            with open('test.el', 'r') as f:
                for line in f.readlines():
                    print(line.strip())

        self.assertTrue(allclose)


# ---------- #
# Test suite #
# ---------- #

suite = unittest.TestLoader().loadTestsFromTestCase(TestIO)

if __name__ == "__main__":
    unittest.main()

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
            for fmt in ("nn", "el", "gml"):
                os.remove(current_dir + 'test.' + fmt)
        except:
            pass
    
    @property
    def test_name(self):
        return "test_io"

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    def gen_graph(self, graph_name):
        # check whether we are loading from file
        if "." in graph_name:
            with_nngt = nngt.get_config("backend") == "nngt"

            if "graphml" in graph_name and with_nngt:
                return None

            abspath = network_dir + graph_name
            di_instructions = self.parser.get_graph_options(graph_name)
            graph = nngt.Graph.from_file(abspath, **di_instructions,
                                         cleanup=True)
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
        err = error.format(graph=graph.name)
        if instructions is not None:  # working with generated graph
            # load graph
            h = nngt.Graph.from_file(current_dir + graph.name + '.el')
            attributes = h.edge_attributes

            # test properties
            self.assertTrue(h.node_nb() == graph.node_nb(),
                            err.format(val='node number'))
            self.assertTrue(h.edge_nb() == graph.edge_nb(),
                            err.format(val='edge number'))

            if graph.is_spatial():
                self.assertTrue(np.allclose(h.get_positions(),
                                            graph.get_positions()),
                                err.format(val='positions'))

            for attr, values in graph.edge_attributes.items():
                # different results probably because of rounding problems
                # note, here we are using the edge list format so edge order
                # is the same in both graphs
                new_val = h.get_edge_attributes(name=attr)
                allclose = np.allclose(new_val, values, 1e-4)
                if not allclose:
                    print("Error: expected")
                    print(values)
                    print("but got")
                    print(new_val)
                    print("max error is: {}".format(
                        np.max(np.abs(np.subtract(
                            new_val, values)))))
                self.assertTrue(allclose, err.format(val=attr))
        else:  # working with loaded graph
            nodes = self.get_expected_result(graph, "nodes")
            edges = self.get_expected_result(graph, "edges")
            directed = self.get_expected_result(graph, "directed")
            # check
            self.assertEqual(
                nodes, graph.node_nb(), err.format(val='node number'))
            self.assertEqual(
                edges, graph.edge_nb(), err.format(val='edge number'))
            self.assertTrue(directed == graph.is_directed(),
                            err.format(val='directedness'))

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    def test_custom_attributes(self):
        '''
        Test that custom attributes are saved and loaded correctly
        '''
        num_nodes = 100
        avg_deg = 10

        g = nngt.Graph(nodes=num_nodes)
        g.new_edge_attribute("test_attr", "int")

        for i in range(num_nodes):
            targets = np.random.choice(num_nodes, size=avg_deg, replace=False)
            elist = np.zeros((len(targets), 2), dtype=int)
            elist[:, 0] = i
            elist[:, 1] = targets
            ids  = np.random.randint(0, avg_deg*num_nodes, len(targets))
            ids *= 2*np.random.randint(0, 2, len(targets)) - 1
            g.new_edges(elist, attributes={"test_attr": ids},
                        check_duplicates=False, check_self_loops=False,
                        check_existing=False)

        old_edges = g.edges_array

        for fmt in ("nn", "el", "gml"):
            g.to_file(current_dir + 'test.' + fmt)
            h = nngt.Graph.from_file(current_dir + 'test.' + fmt)

            # for neighbour list, we need to give the edge list to have
            # the edge attributes in the same order as the original graph
            allclose = np.allclose(g.get_edge_attributes(name="test_attr"),
                                   h.get_edge_attributes(edges=old_edges,
                                                         name="test_attr"))
            if not allclose:
                print("Results differed for '{}'.".format(g.name))
                print("using file 'test.{}'.".format(fmt))
                print(g.get_edge_attributes(name="test_attr"))
                print(h.get_edge_attributes(edges=old_edges, name="test_attr"))
                with open(current_dir + 'test.' + fmt, 'r') as f:
                    for line in f.readlines():
                        print(line.strip())

            self.assertTrue(allclose)


# ---------- #
# Test suite #
# ---------- #

suite = unittest.TestLoader().loadTestsFromTestCase(TestIO)

if __name__ == "__main__":
    unittest.main()

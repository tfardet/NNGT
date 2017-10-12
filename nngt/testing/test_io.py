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
    
    @property
    def test_name(self):
        return "test_io"

    def gen_graph(self, graph_name):
        di_instructions = self.parser.get_graph_options(graph_name)
        graph = nngt.generate(di_instructions)
        graph.set_name(graph_name)
        graph.to_file(current_dir + graph_name + '.el')
        return graph, di_instructions

    @foreach_graph
    def test_identical(self, graph, **kwargs):
        err = error.format(graph=graph.get_name())
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
            self.assertTrue(np.allclose(h.edges_attributes[attr], values),
                            err.format(val=attr))


#-----------------------------------------------------------------------------#
# Test suite
#------------------------
#

suite = unittest.TestLoader().loadTestsFromTestCase(TestIO)

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the IO functions as well as the :mod:`~nngt.analysis` module on
reference graphs.
"""

import unittest

import nngt
from base_test import TestBasis, XmlHandler, network_dir
from tools_testing import foreach_graph



#-----------------------------------------------------------------------------#
# Test class
#------------------------
#

class Test_Analysis(TestBasis):
    
    '''
    Class testing the functions of the :mod:`~nngt.analysis` and
    :mod:`~nngt.io` modules using a set of known graphs.
    '''
    
    @property
    def test_name(self):
        return "test_analysis"

    def gen_graph(self, graph_name):
        abspath = network_dir + graph_name
        di_instructions = self.parser.get_graph_options(graph_name)
        graph = nngt.Graph.from_file(abspath, **di_instructions)
        graph.set_name(graph_name)
        return graph, None

    @foreach_graph
    def test_node_nb(self, graph, **kwargs):
        ref_result = self.get_expected_result(graph, "nodes")
        computed_result = graph.node_nb()
        self.assertEqual(
            ref_result, computed_result,
            "Test failed for graph {}:\nref = {} vs exp {}\
            ".format(graph.name, ref_result, computed_result))

    @foreach_graph
    def test_edge_nb(self, graph, **kwargs):
        ref_result = self.get_expected_result(graph, "edges")
        computed_result = graph.edge_nb()
        self.assertEqual(
            ref_result, computed_result,
            "Test failed for graph {}:\nref = {} vs exp {}\
            ".format(graph.name, ref_result, computed_result))


#-----------------------------------------------------------------------------#
# Test suite
#------------------------
#

suite = unittest.TestLoader().loadTestsFromTestCase(Test_Analysis)

if __name__ == "__main__":
    unittest.main()

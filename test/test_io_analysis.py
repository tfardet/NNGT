#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

import unittest

import numpy as np

import nngt
from ..base_test import TestBasis, directory
from test_tools import foreach_graph, set_path



#-----------------------------------------------------------------------------#
# Test class
#------------------------
#

class TestIO_Analysis(TestBasis):
    
    '''
    Class testing the functions of the :mod:`~nngt.analysis` module using a set
    of known graphs.
    '''

    def __init__(self):
        super(TestBasis, self).init()
        self._name = "test_analysis"
    
    @property
    def test_name(self):
        return self._name

    @set_path(directory)
    def gen_graph(self, graph_name):
        di_instructions = self.parser.get_graph_options(graph_name)
        graph = nngt.Graph.from_file(graph_name, **di_instructions)
        graph.set_name(graph_name)
        return graph

    @foreach_graph(TestIO_Analysis.graphs)
    def test_node_nb(self, graph, **kwargs):
        assert( self.get_expected_result(graph, "nodes") == graph.node_nb() )

    @foreach_graph(TestIO_Analysis.graphs)
    def test_edge_nb(self, graph, **kwargs):
        assert( self.get_expected_result(graph, "edges") == graph.edge_nb() )


#-----------------------------------------------------------------------------#
# Test suite
#------------------------
#

def suite():
    suite = unittest.makeSuite(TestAnalysis, 'test_analysis')
    return suite

def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

if __name__ == "__main__":
    run()

#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_basics.py


"""
Test the validity of the most basic graph functions.
"""

import unittest

import numpy as np

import nngt


# ---------- #
# Test class #
# ---------- #

class TestBasics(unittest.TestCase):
    
    '''
    Class testing the basic methods of the Graph object.
    '''
    
    tolerance = 1e-6
    
    @property
    def test_name(self):
        return "test_basics"

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    def test_node_creation(self):
        '''
        When making graphs, test node creation function.
        '''
        g = nngt.Graph(100, name="new_node_test")
        self.assertTrue(g.node_nb() == 100,
            '''Error on graph {}: invalid initial nodes ({} vs {} expected).
            '''.format(g.name, g.node_nb(), 100))
        n = g.new_node()
        self.assertTrue(g.node_nb() == 101 and n == 100,
            '''Error on graph {}: ({}, {}) vs (101, 100) expected.
            '''.format(g.name, g.node_nb(), n))
        nn = g.new_node(2)
        self.assertTrue(g.node_nb() == 103 and nn[0] == 101 and nn[1] == 102,
            '''Error on graph {}: ({}, {}, {}) vs (103, 101, 102) expected.
            '''.format(g.name, g.node_nb(), nn[0], nn[1]))

    def test_new_node_attr(self):
        '''
        Test node creation with attributes.
        '''
        shape = nngt.geometry.Shape.rectangle(1000., 1000.)
        g = nngt.SpatialGraph(100, shape=shape, name="new_node_spatial")
        self.assertTrue(g.node_nb() == 100,
            '''Error on graph {}: invalid initial nodes ({} vs {} expected).
            '''.format(g.name, g.node_nb(), 100))
        n = g.new_node(positions=[(0, 0)])
        self.assertTrue(
            np.all(np.isclose(g.get_positions(n), (0, 0), self.tolerance)),
            '''Error on graph {}: last position is ({}, {}) vs (0, 0) expected.
            '''.format(g.name, *g.get_positions(n)))


# ---------- #
# Test suite #
# ---------- #

if not nngt.get_config('mpi'):
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBasics)

    if __name__ == "__main__":
        unittest.main()

#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

import unittest

import numpy as np

import nngt

from ..base_test import TestBasis, foreach_graph



#-----------------------------------------------------------------------------#
# Test class
#------------------------
#

class TestAnalysis(TestBasis):
    
    '''
    Class testing the functions of the :mod:`~nngt.analysis` module using a set
    of known graphs.
    '''

    def __init__(self):
        super(TestBasis, self).init()
    
    def get_expected_result(self, graph_instruction):
        pass
    
    def make_graph(self, graph_instruction):
        pass


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

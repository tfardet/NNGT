#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Check that the examples work.
"""

import os
from os import environ
from os.path import dirname, abspath, isfile, join
import unittest

import pytest
from scipy.special import lambertw

import nngt

# set example dir
current_dir = dirname(abspath(__file__))
idx_testing = current_dir.find('testing')
example_dir = current_dir[:idx_testing] + 'doc/examples/'

# remove plotting and NEST
nngt.set_config("with_plot", False)
nngt.set_config("with_nest", False)

# set globals
glob = {"lambertw": lambertw}


# ---------- #
# Test class #
# ---------- #

@pytest.mark.mpi_skip
class TestExamples(unittest.TestCase):
    
    '''
    Class testing saving and loading functions.
    '''
    
    example_files = [
        example_dir + f for f in os.listdir(example_dir)
        if isfile(join(example_dir, f))
    ]

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove("sp_graph.el")
        except:
            pass
    
    @property
    def test_name(self):
        return "test_examples"

    @unittest.skipIf(int(environ.get("OMP", 1)) == 1, 'Check only with OMP')
    def test_examples(self):
        '''
        Test that the example files execute correctly.
        '''
        for example in self.example_files:
            if example.endswith('.py'):
                with open(example) as f:
                    code = compile(f.read(), example, 'exec')
                    exec(code, glob)


# ---------- #
# Test suite #
# ---------- #

suite = unittest.TestLoader().loadTestsFromTestCase(TestExamples)

if __name__ == "__main__":
    unittest.main()

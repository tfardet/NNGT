#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test that the examples work.
"""

import os
from os.path import dirname, abspath, isfile, join
import unittest

import nngt


# set example dir

current_dir = dirname(abspath(__file__))
idx_nngt    = current_dir.find('nngt/testing')
example_dir = current_dir[:idx_nngt] + 'doc/examples/'


# define global variables
import importlib

global_vars = {
    "importlib": importlib,
    # ~ "nngt": nngt,
}


# ---------- #
# Test class #
# ---------- #

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
        for f in cls.example_files:
            if f.endswith(".el"):
                try:
                    os.remove(f)
                except:
                    pass
                local_f = f.rfind("/") + 1
                try:
                    os.remove(f[local_f:])
                except:
                    pass
    
    @property
    def test_name(self):
        return "test_examples"

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    def test_examples(self):
        '''
        Test that the example files execute correctly.
        '''

        for example in example_files:
            print(example)
            if example.endswith('.py'):
                try:
                    execfile(example)
                except NameError:
                    with open(example) as f:
                        code = compile(f.read(), example, 'exec')
                        global_vars["__file__"] = example
                        exec(code, global_vars)


#-----------------------------------------------------------------------------#
# Test suite
#------------------------
#

suite = unittest.TestLoader().loadTestsFromTestCase(TestExamples)

if __name__ == "__main__":
    unittest.main()

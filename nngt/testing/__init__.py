#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# test_analysis.py
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
==============
Testing module
==============

This module tests the various functionalities of NNGT to make sure that all
implementations remain compatible with the graph libraries and versions 2.7 and
3.x of python.

note ::
    When adding new tests, filename should be of the form `test_xxx.py` and the
    code should contain:
    * a ``TestXXX`` class,
    * a ``suite = unittest.TestLoader().loadTestsFromTestCase(TestXXX)``
      declaration.
"""

# std imports
import sys
import importlib
from os import listdir, environ
from os.path import abspath, dirname, isfile
import unittest

# personal library
import nngt



#-----------------------------------------------------------------------------#
# Get the tests
#------------------------
#

# get the arguments
graph_library = environ.get("GL", None)
if graph_library == "gt":
    nngt.use_library("graph-tool")
    assert nngt.get_config('graph_library') == "graph-tool", \
           "Loading graph-tool failed..."
elif graph_library == "ig":
    nngt.use_library("igraph")
    assert nngt.get_config('graph_library') == "igraph", \
           "Loading igraph failed..."
elif graph_library == "nx":
    nngt.use_library("networkx")
    assert nngt.get_config('graph_library') == "networkx", \
           "Loading networkx failed..."

omp = int(environ.get("OMP", 1))
nngt.set_config({"omp": omp})

# get the tests
current_dir = dirname(abspath(__file__))
dir_files = listdir(current_dir)
sys.path.insert(0, current_dir)
testfiles = [fname[:-3] for fname in dir_files if (fname.startswith("test_") 
             and fname.endswith(".py"))]
tests = [importlib.import_module(name) for name in testfiles]


#-----------------------------------------------------------------------------#
# Run if main
#------------------------
#

if __name__ == "__main__":
    for test in tests:
        unittest.TextTestRunner(verbosity=2).run(test.suite)

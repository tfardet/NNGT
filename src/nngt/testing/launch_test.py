#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

import sys
from os import listdir
from os.path import isfile

import nngt

dir_tests = [ d for d in listdir(".") if not isfile(d) ]
for dir_test in dir_tests:
    sys.path.insert(0, dir_test)



#-----------------------------------------------------------------------------#
# Get graph library
#------------------------
#


args = sys.argv
num_args = len(args)
graph_library = args[0]

if graph_library == "GT":
    nngt.use_library("graph_tool")
    print("Using graph_tool")
elif graph_library == "IG":
    nngt.use_library("graph_tool")
    print("Using igraph")
else:
    nngt.use_library("networkx")
    print("Using networkx")


#-----------------------------------------------------------------------------#
# Launch tests
#------------------------
#


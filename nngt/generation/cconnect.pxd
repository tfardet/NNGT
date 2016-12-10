# distutils: language = c++
# distutils: sources = connect.cpp
#!/usr/bin/env cython
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

cimport numpy as np

from libcpp.vector cimport vector
from libcpp cimport bool



__all__ = [
    "_distance_rule",
    "_erdos_renyi",
    "_filter",
    "_fixed_degree",
    "_gaussian_degree",
    "_newman_watts",
    "_no_self_loops",
    "_price_scale_free",
    "_random_scale_free",
    "_unique_rows",
    "price_network",
]


#-----------------------------------------------------------------------------#
# Load the c++ functions
#------------------------
#

cdef extern from "func_connect.h" namespace "generation":
    cdef vector[size_t] _gen_edge_complement(
      long seed, vector[size_t] nodes, size_t other_end, size_t degree,
      const vector[ vector[size_t] ]* existing_edges, bool multigraph) nogil

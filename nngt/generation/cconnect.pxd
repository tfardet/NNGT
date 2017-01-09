#!/usr/bin/env cython
#-*- coding:utf-8 -*-

from libcpp.vector cimport vector
from libcpp cimport bool

""" Generation tools for NNGT """


#-----------------------------------------------------------------------------#
# Load the c++ functions
#------------------------
#

cdef extern from "func_connect.h" namespace "generation":
    cdef void _gen_edges(
      size_t* ia_edges, const vector [size_t]& first_nodes,
      const vector[size_t]& degrees, const vector[size_t]& second_nodes,
      const vector[ vector[size_t] ]& existing_edges, unsigned int idx,
      bool multigraph, bool directed, long msd, unsigned int omp) except +

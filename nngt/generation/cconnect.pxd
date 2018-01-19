#!/usr/bin/env cython
#-*- coding:utf-8 -*-

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

""" Generation tools for NNGT """


#-----------------------------------------------------------------------------#
# Load the c++ functions
#------------------------
#

cdef extern from "func_connect.h" namespace "generation":
    cdef void _gen_edges(
      size_t* ia_edges, const vector[size_t]& first_nodes,
      const vector[unsigned int]& degrees, const vector[size_t]& second_nodes,
      const vector[ vector[size_t] ]& existing_edges, unsigned int idx,
      bool multigraph, bool directed, long msd, unsigned int omp) except +

    cdef void _cdistance_rule(
      size_t* ia_edges, const vector[size_t]& source_nodes,
      const vector[ vector[size_t] ]& target_nodes, const string& rule,
      float scale, float norm, const vector[float]& x, const vector[float]& y,
      size_t num_neurons, size_t num_edges,
      const vector[ vector[size_t] ]& existing_edges, vector[float]& dist,
      bool multigraph, long msd, unsigned int omp) except +

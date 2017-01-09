#!/usr/bin/env cython
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """


#-----------------------------------------------------------------------------#
# Load the c++ functions
#------------------------
#

cdef extern from "func_connect.h" namespace "generation":
#~     cdef vector[size_t] _gen_edge_complement(
#~       long seed, vector[size_t]& nodes, size_t other_end, size_t degree,
#~       const vector[ vector[size_t] ]* existing_edges, bool multigraph) nogil
      
#~     cdef vector[ vector[size_t] ] _gen_edges(
#~       const vector [size_t]& first_nodes, const vector[size_t]& degrees,
#~       const vector[size_t]& second_nodes,
#~       const vector[ vector[size_t] ]& existing_edges,
#~       bool multigraph, bool directed, long msd, size_t omp) except +
    cdef void _gen_edges(
      size_t* ia_edges, const vector [size_t]& first_nodes,
      const vector[size_t]& degrees, const vector[size_t]& second_nodes,
      const vector[ vector[size_t] ]& existing_edges, unsigned int idx,
      bool multigraph, bool directed, long msd, unsigned int omp) except +

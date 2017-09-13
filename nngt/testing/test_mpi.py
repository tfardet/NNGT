#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_generation.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the main methods of the :mod:`~nngt.generation` module.
"""

import unittest

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    pass

import numpy as np

import nngt
from nngt.analysis import *
from nngt.lib.connect_tools import _compute_connections

from base_test import TestBasis, XmlHandler, network_dir
from test_generation import _distance_rule_theo, _distance_rule_exp
from tools_testing import foreach_graph


# -------- #
# Test MPI #
# -------- #

def _get_connections(instruct):
    nodes = instruct["nodes"]
    density = instruct.get("density", -1)
    edges = instruct.get("edges", -1)
    average_degree = instruct.get("avg_deg", -1)
    reciprocity = instruct.get("reciprocity", -1)
    directed = instruct.get("directed", True)
    #~ weighted = instruct.get("weighted", True))
    return nodes, density, edges, average_degree, directed, reciprocity



# ---------- #
# Test class #
# ---------- #

class TestMPI(TestBasis):
    
    '''
    Class testing the main methods of the :mod:`~nngt.generation` module.
    '''
    
    theo_prop = {
        "distance_rule": _distance_rule_theo,
    }
    
    exp_prop = {
        "distance_rule": _distance_rule_exp,
    }

    tolerance = 0.01
    
    @property
    def test_name(self):
        return "test_mpi"

    def gen_graph(self, graph_name):
        if rank == 0:
            di_instructions = self.parser.get_graph_options(graph_name)
            graph = nngt.generate(di_instructions)
            graph.set_name(graph_name)
            return graph, di_instructions

    @foreach_graph
    @unittest.skipIf(not nngt.get_config('mpi'), "Not using MPI.")
    def test_model_properties(self, graph, instructions, **kwargs):
        '''
        When generating graphs from on of the preconfigured models, check that
        the expected properties are indeed obtained.
        '''
        if rank == 0:
            graph_type = instructions["graph_type"]
            ref_result = self.theo_prop[graph_type](instructions)
            computed_result = self.exp_prop[graph_type](graph, instructions)
            self.assertTrue(np.allclose(
                ref_result, computed_result, self.tolerance),
                "Test for graph {} failed:\nref = {} vs exp {}\
                ".format(graph.name, ref_result, computed_result))


# ---------- #
# Test suite #
# ---------- #

suite = unittest.TestLoader().loadTestsFromTestCase(TestMPI)

if __name__ == "__main__":
    unittest.main()

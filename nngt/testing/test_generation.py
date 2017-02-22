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
import numpy as np

import nngt
from nngt.analysis import *
from nngt.generation import _compute_connections

from base_test import TestBasis, XmlHandler, network_dir
from tools_testing import foreach_graph



#-----------------------------------------------------------------------------#
# Test tools
#------------------------
#

def _get_connections(instruct):
    nodes = instruct["nodes"]
    density = instruct.get("density", -1)
    edges = instruct.get("edges", -1)
    average_degree = instruct.get("avg_deg", -1)
    reciprocity = instruct.get("weighted", -1)
    directed = instruct.get("directed", True)
    #~ weighted = instruct.get("weighted", True))
    return nodes, density, edges, average_degree, directed, reciprocity


def _fixed_deg_theo(instruct):
    fixed_degree = instruct["degree"]
    avg_free_degree = fixed_degree
    deg_type = instruct["degree_type"]
    std_free_degree = np.sqrt(avg_free_degree) if deg_type != "total" else 0
    return fixed_degree, avg_free_degree, std_free_degree


def _fixed_deg_exp(graph, instruct):
    n, d, e, a_deg, directed, weighted = _get_connections(instruct)
    deg_type = instruct["degree_type"]
    deg_types = [ 'in', 'out' ]
    counts, bins = degree_distrib(graph, deg_type)
    max_count = counts.max()
    fixed_degree = bins[np.where(counts-max_count == 0)][0]
    # check that they are indeed all in the fixed_degree
    assert(n == max_count)
    # compute free_degree properties
    avg_free_degree, std_free_degree = fixed_degree, 0
    if deg_type != 'total':
        del deg_types[deg_types.index(deg_type)]
        degrees = graph.get_degrees(deg_types[0], use_weights=False)
        avg_free_degree = np.average(degrees)
        std_free_degree = np.std(degrees)
    return fixed_degree, avg_free_degree, std_free_degree


def _gaussian_deg_theo(instruct):
    avg_degree = instruct["avg"]
    std_degree = instruct["std"]
    avg_free_degree = avg_degree
    deg_type = instruct["degree_type"]
    std_free_degree = np.sqrt(avg_free_degree)
    return avg_degree, std_degree, avg_free_degree, std_free_degree


def _gaussian_deg_exp(graph, instruct):
    n, d, e, a_deg, directed, weighted = _get_connections(instruct)
    deg_type = instruct["degree_type"]
    deg_types = [ 'in', 'out' ]
    degrees = graph.get_degrees(deg_type, use_weights=False)
    avg_degree = np.average(degrees)
    std_degree = np.std(degrees)
    # compute free_degree properties
    avg_free_degree, std_free_degree = avg_degree, std_degree
    if deg_type != 'total':
        del deg_types[deg_types.index(deg_type)]
        degrees = graph.get_degrees(deg_types[0], use_weights=False)
        avg_free_degree = np.average(degrees)
        std_free_degree = np.std(degrees)
    return avg_degree, std_degree, avg_free_degree, std_free_degree


def _erdos_renyi_theo(instruct):
    pass


def _erdos_renyi_exp(graph, instruct):
    pass


def _random_scale_free_theo(instruct):
    pass


def _random_scale_free_exp(graph, instruct):
    pass


def _newman_watts_theo(instruct):
    pass


def _newman_watts_exp(graph, instruct):
    pass


def _distance_rule_theo(instruct):
    pass


def _distance_rule_exp(graph, instruct):
    pass


#-----------------------------------------------------------------------------#
# Test class
#------------------------
#

class TestGeneration(TestBasis):
    
    '''
    Class testing the main methods of the :mod:`~nngt.generation` module.
    '''
    
    theo_prop = {
        "fixed_degree": _fixed_deg_theo,
        "gaussian_degree": _gaussian_deg_theo,
        "erdos_renyi": _erdos_renyi_theo,
        "random_scale_free": _random_scale_free_theo,
        "newman_watts": _newman_watts_theo,
        "distance_rule": _distance_rule_theo,
    }
    
    exp_prop = {
        "fixed_degree": _fixed_deg_exp,
        "gaussian_degree": _gaussian_deg_exp,
        "erdos_renyi": _erdos_renyi_exp,
        "random_scale_free": _random_scale_free_exp,
        "newman_watts": _newman_watts_exp,
        "distance_rule": _distance_rule_exp,
    }

    tolerance = 0.05
    
    @property
    def test_name(self):
        return "test_generation"

    def gen_graph(self, graph_name):
        di_instructions = self.parser.get_graph_options(graph_name)
        graph = nngt.generate(di_instructions)
        graph.set_name(graph_name)
        return graph, di_instructions

    @foreach_graph
    def test_model_properties(self, graph, instructions, **kwargs):
        '''
        When generating graphs from on of the preconfigured models, check that
        the expected properties are indeed obtained.
        '''
        graph_type = instructions["graph_type"]
        print(graph_type)
        ref_result = self.theo_prop[graph_type](instructions)
        computed_result = self.exp_prop[graph_type](graph, instructions)
        self.assertTrue(np.allclose(
            ref_result, computed_result, self.tolerance),
            "Test for graph {} failed:\nref = {} vs exp {}\
            ".format(graph.name, ref_result, computed_result))


#-----------------------------------------------------------------------------#
# Test suite
#------------------------
#

suite = unittest.TestLoader().loadTestsFromTestCase(TestGeneration)

if __name__ == "__main__":
    unittest.main()

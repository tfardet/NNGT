#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_attributes.py


"""
Test the validity of graph, node, and edge attributes as well as the
distribution generators.
"""

import unittest
import numpy as np

import nngt

from base_test import TestBasis, XmlHandler, network_dir
from tools_testing import foreach_graph


# ---------- #
# Test tools #
# ---------- #

def _results_theo(instruct):
    di_param = instruct["weights"]
    if (di_param["distribution"] == "uniform"
        or "corr" in di_param["distribution"]):
        return di_param["lower"], di_param["upper"]
    elif di_param["distribution"] == "gaussian":
        return di_param["avg"], di_param["std"]
    elif di_param["distribution"] == "lognormal":
        return di_param["position"], di_param["scale"]
    else:
        raise NotImplementedError("This distribution is not supported yet.")


def _results_exp(attrib, instruct):
    di_param = instruct["weights"]
    if (di_param["distribution"] == "uniform"
        or "corr" in di_param["distribution"]):
        return attrib.min(), attrib.max()
    elif di_param["distribution"] == "gaussian":
        return np.average(attrib), np.std(attrib)
    elif di_param["distribution"] == "lognormal":
        m = np.average(attrib)
        v = np.var(attrib)
        return np.log(m/np.sqrt(1+v/m**2)), np.sqrt(np.log(1+v/m**2))
    else:
        raise NotImplementedError("This distribution is not supported yet.")


# ---------- #
# Test class #
# ---------- #

class TestAttributes(TestBasis):
    
    '''
    Class testing the main methods of the :mod:`~nngt.generation` module.
    '''
    
    tolerance = 0.02
    
    @property
    def test_name(self):
        return "test_attributes"

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    def gen_graph(self, graph_name):
        di_instructions = self.parser.get_graph_options(graph_name)
        graph = nngt.generate(di_instructions)
        graph.set_name(graph_name)
        return graph, di_instructions

    def test_node_attr(self):
        '''
        When generating graphs with weights, check that the expected properties
        are indeed obtained.
        '''
        g = nngt.Graph(100)
        ref_result = np.random.uniform(-1, 4, g.node_nb())
        g.set_node_attribute("nud", values=ref_result, value_type="double")
        computed_result = g.get_node_attributes(name="nud")
        self.assertTrue(np.allclose(ref_result, computed_result),
            '''Error on graph {}: unequal 'nud' attribute for tolerance {}.
            '''.format(g.name, self.tolerance))

    def test_user_defined(self):
        '''
        When generating graphs with weights, check that the expected properties
        are indeed obtained.
        '''
        g = nngt.generation.erdos_renyi(avg_deg=50, nodes=200)
        ref_result = np.random.uniform(0, 5, g.edge_nb())
        g.set_edge_attribute("ud", values=ref_result, value_type="double")
        computed_result = g.get_edge_attributes(name="ud")
        self.assertTrue(np.allclose(ref_result, computed_result),
            '''Error on graph {}: unequal 'ud' attribute for tolerance {}.
            '''.format(g.name, self.tolerance))

    def test_user_defined2(self):
        '''
        When generating graphs with weights, check that the expected properties
        are indeed obtained.
        '''
        g = nngt.generation.erdos_renyi(avg_deg=50, nodes=200)
        ref_result = np.full(g.edge_nb(), 4.)
        g.set_edge_attribute("ud2", val=4., value_type="double")
        computed_result = g.get_edge_attributes(name="ud2")
        self.assertTrue(np.allclose(ref_result, computed_result),
            '''Error on graph {}: unequal 'ud2' attribute for tolerance {}.
            '''.format(g.name, self.tolerance))

    @foreach_graph
    def test_weights(self, graph, instructions, **kwargs):
        '''
        When generating graphs with weights, check that the expected properties
        are indeed obtained.
        '''
        ref_result = _results_theo(instructions)
        weights = graph.get_weights()
        computed_result = _results_exp(weights, instructions)
        self.assertTrue(np.allclose(
            ref_result, computed_result, self.tolerance),
            '''Error on graph {}: unequal weights for tolerance {}.
            '''.format(graph.name, self.tolerance))

    @foreach_graph
    def test_delays(self, graph, instructions, **kwargs):
        '''
        Test entirely run only if NEST is present on the computer.
        Check that delay distribution generated in NNGT, then in NEST, is
        conform to what was instructed.
        '''
        # get the informations from the weights
        di_distrib = instructions["weights"]
        distrib = di_distrib["distribution"]
        delays = graph.set_delays(distribution=distrib, parameters=di_distrib)
        ref_result = _results_theo(instructions)
        computed_result = _results_exp(delays, instructions)
        self.assertTrue(np.allclose(
            ref_result, computed_result, self.tolerance),
            '''Error on graph {}: unequal delays for tolerance {}.
            '''.format(graph.name, self.tolerance))
        # @todo
        #~ if nngt._config['with_nest']:
            #~ from nngt.simulation import make_nest_network
            #~ gids = make_nest_network(graph)


# ---------- #
# Test suite #
# ---------- #

if not nngt.get_config('mpi'):
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAttributes)

    if __name__ == "__main__":
        unittest.main()

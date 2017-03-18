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



#-----------------------------------------------------------------------------#
# Test tools
#------------------------
#

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


#-----------------------------------------------------------------------------#
# Test class
#------------------------
#

class TestAttributes(TestBasis):
    
    '''
    Class testing the main methods of the :mod:`~nngt.generation` module.
    '''
    
    tolerance = 0.02
    
    @property
    def test_name(self):
        return "test_attributes"

    def gen_graph(self, graph_name):
        print(graph_name)
        di_instructions = self.parser.get_graph_options(graph_name)
        graph = nngt.generate(di_instructions)
        graph.set_name(graph_name)
        return graph, di_instructions

    @foreach_graph
    def test_weights(self, graph, instructions, **kwargs):
        '''
        When generating graphs with weights, check that the expected properties
        are indeed obtained.
        '''
        if graph.name == "erdos_uniform_weights":
            print(instructions, graph.get_weights()[0], len(graph.get_weights()), graph.edge_nb())
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
        di_distrib = instructions["weights"]
        distrib = di_distrib["distribution"]
        delays = graph.set_delays(distribution=distrib, parameters=di_distrib)
        ref_result = _results_theo(instructions)
        computed_result = _results_exp(delays, instructions)
        self.assertTrue(np.allclose(ref_result,computed_result,self.tolerance))
        # @todo
        #~ if nngt.config['with_nest']:
            #~ from nngt.simulation import make_nest_network
            #~ gids = make_nest_network(graph)


#-----------------------------------------------------------------------------#
# Test suite
#------------------------
#

suite = unittest.TestLoader().loadTestsFromTestCase(TestAttributes)

if __name__ == "__main__":
    unittest.main()

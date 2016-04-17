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
from nngt.generation import _compute_connections

from base_test import TestBasis, XmlHandler, network_dir
from tools_testing import foreach_graph



#-----------------------------------------------------------------------------#
# Test tools
#------------------------
#

def _weights_theo(instruct):
    wprop = instruct["weights"]
    if wprop["distribution"] == "uniform" or "corr" in wprop["distribution"]:
        return wprop["lower"], wprop["upper"]
    elif wprop["distribution"] == "gaussian":
        return wprop["avg"], wprop["std"]
    elif wprop["distribution"] == "lognormal":
        return wprop["position"], wprop["scale"]
    else:
        raise NotImplementedError("This distribution is not supported yet.")

def _weights_exp(weights, instruct):
    wprop = instruct["weights"]
    if wprop["distribution"] == "uniform" or "corr" in wprop["distribution"]:
        return weights.min(), weights.max()
    elif wprop["distribution"] == "gaussian":
        return np.average(weights), np.std(weights)
    elif wprop["distribution"] == "lognormal":
        m = np.average(weights)
        v = np.var(weights)
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
    
    theo_prop = {
        "weights": _weights_theo,
    }
    
    exp_prop = {
        "weights": _weights_exp,
    }

    tolerance = 0.02
    
    @property
    def test_name(self):
        return "test_attributes"

    def gen_graph(self, graph_name):
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
        graph_type = instructions["graph_type"]
        ref_result = _weights_theo(instructions)
        weights = graph.get_weights()
        computed_result = _weights_exp(weights, instructions)
        self.assertTrue(np.allclose(ref_result,computed_result,self.tolerance))


#-----------------------------------------------------------------------------#
# Test suite
#------------------------
#

suite = unittest.TestLoader().loadTestsFromTestCase(TestAttributes)

if __name__ == "__main__":
    unittest.main()

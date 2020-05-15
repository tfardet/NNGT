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
    
    def test_nattr_default_values(self):
        g2 = nngt.Graph()

        # add a new node with attributes
        attributes = {
            'size': 2.,
            'color': 'blue',
            'a': 5,
            'blob': []
        }

        attribute_types = {
            'size': 'double',
            'color': 'string',
            'a': 'int',
            'blob': 'object'
        }

        g2.new_node(attributes=attributes, value_types=attribute_types)
        g2.new_node(2)
        g2.new_node(3, attributes={'size': [4., 5., 1.],
                    'color': ['r', 'g', 'b']},
                    value_types={'size': 'double', 'color': 'string'})

        # check all values
        # for the doubles:
        # NaN == NaN is false, so we need to check separately equality between
        # non-NaN entries and position of NaN entries
        double_res = np.array([2., np.NaN, np.NaN, 4., 5., 1.])
        isnan1     = np.isnan(g2.node_attributes['size'])
        isnan2     = np.isnan(double_res)
        self.assertTrue(np.all(isnan1 == isnan2))
        self.assertTrue(
            np.all(np.isclose(
                g2.node_attributes['size'][~isnan1], double_res[~isnan2]))
        )
        # for the others, just compare the lists
        self.assertEqual(
            g2.node_attributes['color'].tolist(),
            ['blue', '', '', 'r', 'g', 'b'])
        self.assertEqual(
            g2.node_attributes['a'].tolist(), [5, 0, 0, 0, 0, 0])
        self.assertEqual(
            g2.node_attributes['blob'].tolist(),
            [[], None, None, None, None, None])

    def test_user_defined(self):
        '''
        When generating graphs with weights, check that the expected properties
        are indeed obtained.
        '''
        avg = 50
        std = 6
        g = nngt.generation.gaussian_degree(avg, std, nodes=200)

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
        avg = 50
        std = 6
        g = nngt.generation.gaussian_degree(avg, std, nodes=200)

        ref_result = np.full(g.edge_nb(), 4.)
        g.set_edge_attribute("ud2", val=4., value_type="double")

        computed_result = g.get_edge_attributes(name="ud2")
        self.assertTrue(np.allclose(ref_result, computed_result),
            '''Error on graph {}: unequal 'ud2' attribute for tolerance {}.
            '''.format(g.name, self.tolerance))

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    def test_list_attributes(self):
        '''
        For list attributes, test that they are preserved as lists, and that
        some nodes or edges do not own references to the same list.
        '''
        avg = 25
        std = 3

        graph = nngt.generation.gaussian_degree(avg, std, nodes=1000)

        # --------------- #
        # node attributes #
        # --------------- #

        graph.new_node_attribute("nlist", value_type="object", val=[])

        nodes = [i for i in range(8, 49)]
        graph.set_node_attribute("nlist", val=[1], nodes=nodes)

        # update a fraction of the previously updated nodes
        nodes = [i for i in range(0, 41)]

        # to update the values, we need to get them to update the lists
        nlists = graph.get_node_attributes(name="nlist", nodes=nodes)

        for l in nlists:
            l.append(2)

        graph.set_node_attribute("nlist", values=nlists, nodes=nodes)

        # check that all lists are present
        nlists = graph.get_node_attributes(name="nlist")

        self.assertTrue(
            np.all(np.unique(nlists) == np.unique([[], [1], [2], [1, 2]])))

        # check that all nodes from 0 to 48 were updated
        self.assertTrue([] not in nlists[:49].tolist())

        # --------------- #
        # edge attributes #
        # --------------- #

        graph.new_edge_attribute("elist", value_type="object", val=[])

        nodes = [i for i in range(8, 49)]
        edges = graph.get_edges(source_node=nodes, target_node=nodes)
        graph.set_edge_attribute("elist", val=[1], edges=edges)

        # update a fraction of the previously updated nodes
        nodes  = [i for i in range(0, 41)]
        edges2 = graph.get_edges(source_node=nodes, target_node=nodes)

        # to update the values, we need to get them to update the lists
        elists = graph.get_edge_attributes(name="elist", edges=edges2)

        for l in elists:
            l.append(2)

        graph.set_edge_attribute("elist", values=elists, edges=edges2)

        # check that all lists are present
        elists = graph.get_edge_attributes(name="elist")

        self.assertTrue(
            np.all(np.unique(elists) == np.unique([[], [1], [2], [1, 2]])))

        # check that all edges where updated
        eattr1 = graph.get_edge_attributes(name="elist", edges=edges).tolist()
        eattr2 = graph.get_edge_attributes(name="elist", edges=edges2).tolist()

        self.assertTrue([] not in eattr1 and [] not in eattr2)

    @foreach_graph
    def test_weights(self, graph, instructions, **kwargs):
        '''
        When generating graphs with weights, check that the expected properties
        are indeed obtained.
        '''
        ref_result = _results_theo(instructions)

        weights    = graph.get_weights()
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

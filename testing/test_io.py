#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the IO functions.
"""

import os
import unittest

import numpy as np
import pytest

import nngt
from base_test import TestBasis, XmlHandler, network_dir
from tools_testing import foreach_graph


# --------------------- #
# File creation/removal #
# --------------------- #

current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
error = 'Wrong {{val}} for {graph}.'

formats = ("neighbour", "edge_list", "gml", "graphml")

filetypes = ("nn", "el", "gml", "graphml")

gfilename = current_dir + 'g.graph'


def teardown_function(function):
    ''' Cleanup the file '''
    if nngt.get_config("mpi"):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD.Clone()

        comm.Barrier()

    try:
        os.remove(gfilename)
    except:
        pass


# ---------- #
# Test class #
# ---------- #

class TestIO(TestBasis):

    '''
    Class testing saving and loading functions.
    '''

    @classmethod
    def tearDownClass(cls):
        for graphname in cls.graphs:
            try:
                os.remove(current_dir + graphname + '.el')
            except:
                pass
        try:
            for ft in filetypes:
                os.remove(current_dir + 'test.' + ft)
        except:
            pass
    
    @property
    def test_name(self):
        return "test_io"

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    def gen_graph(self, graph_name):
        # check whether we are loading from file
        if "." in graph_name:
            abspath = network_dir + graph_name
            di_instructions = self.parser.get_graph_options(graph_name)
            graph = nngt.Graph.from_file(abspath, **di_instructions,
                                         cleanup=True)
            graph.set_name(graph_name)
            return graph, None
        else:
            di_instructions = self.parser.get_graph_options(graph_name)
            graph = nngt.generate(di_instructions)
            graph.set_name(graph_name)
            graph.to_file(current_dir + graph_name + '.el')

            return graph, di_instructions

    @foreach_graph
    def test_identical(self, graph, instructions, **kwargs):
        '''
        Test that the generated graph and the one loaded from the saved file
        are indeed identical.
        '''
        err = error.format(graph=graph.name)
        if instructions is not None:  # working with generated graph
            # load graph
            h = nngt.Graph.from_file(current_dir + graph.name + '.el')
            attributes = h.edge_attributes

            # test properties
            self.assertTrue(h.node_nb() == graph.node_nb(),
                            err.format(val='node number'))
            self.assertTrue(h.edge_nb() == graph.edge_nb(),
                            err.format(val='edge number'))

            if graph.is_spatial():
                self.assertTrue(np.allclose(h.get_positions(),
                                            graph.get_positions()),
                                err.format(val='positions'))

            for attr, values in graph.edge_attributes.items():
                # different results probably because of rounding problems
                # note, here we are using the edge list format so edge order
                # is the same in both graphs
                new_val = h.get_edge_attributes(name=attr)
                allclose = np.allclose(new_val, values, 1e-4)
                if not allclose:
                    print("Error: expected")
                    print(values)
                    print("but got")
                    print(new_val)
                    print("max error is: {}".format(
                        np.max(np.abs(np.subtract(
                            new_val, values)))))
                self.assertTrue(allclose, err.format(val=attr))
        else:  # working with loaded graph
            nodes = self.get_expected_result(graph, "nodes")
            edges = self.get_expected_result(graph, "edges")
            directed = self.get_expected_result(graph, "directed")
            # check
            self.assertEqual(
                nodes, graph.node_nb(), err.format(val='node number'))
            self.assertEqual(
                edges, graph.edge_nb(), err.format(val='edge number'))
            self.assertTrue(directed == graph.is_directed(),
                            err.format(val='directedness'))

    @unittest.skipIf(nngt.get_config('mpi'), 'Not checking for MPI')
    def test_custom_attributes(self):
        '''
        Test that custom attributes are saved and loaded correctly
        '''
        num_nodes = 100
        avg_deg = 10

        g = nngt.Graph(nodes=num_nodes)
        g.new_edge_attribute("test_attr", "int")

        for i in range(num_nodes):
            targets = np.random.choice(num_nodes, size=avg_deg, replace=False)
            elist = np.zeros((len(targets), 2), dtype=int)
            elist[:, 0] = i
            elist[:, 1] = targets
            ids  = np.random.randint(0, avg_deg*num_nodes, len(targets))
            ids *= 2*np.random.randint(0, 2, len(targets)) - 1
            g.new_edges(elist, attributes={"test_attr": ids},
                        check_duplicates=False, check_self_loops=False,
                        check_existing=False)

        old_edges = g.edges_array

        for ft in filetypes:
            g.to_file(current_dir + 'test.' + ft)
            h = nngt.Graph.from_file(current_dir + 'test.' + ft)

            # for neighbour list, we need to give the edge list to have
            # the edge attributes in the same order as the original graph
            allclose = np.allclose(g.get_edge_attributes(name="test_attr"),
                                   h.get_edge_attributes(edges=old_edges,
                                                         name="test_attr"))
            if not allclose:
                print("Results differed for '{}'.".format(g.name))
                print("using file 'test.{}'.".format(ft))
                print(g.get_edge_attributes(name="test_attr"))
                print(h.get_edge_attributes(edges=old_edges, name="test_attr"))
                with open(current_dir + 'test.' + ft, 'r') as f:
                    for line in f.readlines():
                        print(line.strip())

            self.assertTrue(allclose)


@pytest.mark.mpi_skip
def test_empty_out_degree():
    g = nngt.Graph(2)

    g.new_edge(0, 1)

    for fmt in formats:
        nngt.save_to_file(g, gfilename, fmt=fmt)

        h = nngt.load_from_file(gfilename, fmt=fmt)

        assert np.array_equal(g.edges_array, h.edges_array)


@pytest.mark.mpi_skip
def test_str_attributes():
    g = nngt.Graph(2)

    g.new_edge(0, 1)

    g.new_edge_attribute("type", "string")
    g.set_edge_attribute("type", val='odd')

    g.new_node_attribute("rnd", "string")
    g.set_node_attribute("rnd", values=["s'adf", 'sd fr"'])

    for fmt in formats:
        nngt.save_to_file(g, gfilename, fmt=fmt)

        h = nngt.load_from_file(gfilename, fmt=fmt)

        assert np.array_equal(g.edges_array, h.edges_array)

        assert np.array_equal(g.edge_attributes["type"],
                              h.edge_attributes["type"])

        assert np.array_equal(g.node_attributes["rnd"],
                              h.node_attributes["rnd"])


@pytest.mark.mpi_skip
def test_structure():
    # with a structure
    room1 = nngt.Group(25)
    room2 = nngt.Group(50)
    room3 = nngt.Group(40)
    room4 = nngt.Group(35)

    names = ["R1", "R2", "R3", "R4"]

    struct = nngt.Structure.from_groups((room1, room2, room3, room4), names)

    g = nngt.Graph(structure=struct)

    for fmt in formats:
        g.to_file(gfilename, fmt=fmt)

        h = nngt.load_from_file(gfilename, fmt=fmt)

        assert g.structure == h.structure

    # with a neuronal population
    g = nngt.Network.exc_and_inhib(100)

    for fmt in formats:
        g.to_file(gfilename, fmt=fmt)

        h = nngt.load_from_file(gfilename, fmt=fmt)

        assert g.population == h.population


@pytest.mark.mpi_skip
def test_spatial():
    from nngt.geometry import Shape

    shape = Shape.disk(100, default_properties={"plop": 0.2, "height": 1.})
    area = Shape.rectangle(10, 10, default_properties={"height": 10.})
    shape.add_area(area, name="center")

    g = nngt.SpatialGraph(20, shape=shape)

    for fmt in formats:
        g.to_file(gfilename, fmt=fmt)

        h = nngt.load_from_file(gfilename, fmt=fmt)

        assert np.all(np.isclose(g.get_positions(), h.get_positions()))
        assert g.shape.almost_equals(h.shape)

        for name, area in g.shape.areas.items():
            assert area.almost_equals(h.shape.areas[name])

            assert area.properties == h.shape.areas[name].properties


@pytest.mark.mpi_skip
def test_node_attributes():
    num_nodes = 10
    g = nngt.generation.erdos_renyi(nodes=num_nodes, avg_deg=3, directed=False)

    g.new_node_attribute("size", "int", [2*(i+1) for i in range(num_nodes)])

    for fmt in formats:
        g.to_file(gfilename, fmt=fmt)

        h = nngt.load_from_file(gfilename, fmt=fmt)

        assert np.array_equal(g.node_attributes["size"],
                              h.node_attributes["size"])


# ---------- #
# Test suite #
# ---------- #

suite = unittest.TestLoader().loadTestsFromTestCase(TestIO)

if __name__ == "__main__":
    if not nngt.get_config("mpi"):
        test_empty_out_degree()
        test_str_attributes()
        test_structure()
        test_node_attributes()
        test_spatial()
        unittest.main()

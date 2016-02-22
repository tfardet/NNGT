#!/usr/bin/env python
#-*- coding:utf-8 -*-

# base_test.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

import sys
from os import listdir
from os.path import isfile, join
import unittest
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import xml.etree.ElementTree as xmlet

import nngt
from test_tools import ( _bool_from_string, _make_graph_list, _xml_to_dict,
                         with_metaclass )



#-----------------------------------------------------------------------------#
# Path to input files
#------------------------
#

# folder containing all networks
directory = "Networks/"

# file containing networks' properties and instructions
xml_file = "graph_tests.xml"

if xml_file not in listdir("."):
    xml_file = "test/graph_tests.xml"
    directory = "test/Networks/"
    if xml_file not in listdir("."):
        raise RuntimeError("File `graph_tests.xml` not found! Tests must be \
run from the same folder as `setup.py` or from the `test` folder.")


#-----------------------------------------------------------------------------#
# XmlResultParser
#------------------------
#

class XmlHandler:

    ''' Class parsing the XML reference files. '''
    
    di_type = {
        "float": float,
        "double": float,
        "string": str,
        "str": str,
        "int": int,
        "bool": _bool_from_string
    }
    
    def __init__(self):
        tree = xmlet.parse(xml_file)
        self.root = tree.getroot()
        self.graphs = self.root.find("graphs")
    
    def result(self, elt):
        return di_type[elt.tag](elt.text)
    
    def get_graph_list(self, test):
        elt_test = self.root.find('./test[@name="{}"]'.format(test))
        return _make_graph_list(elt_test.attrib["graph_list"])

    def get_graph_options(self, graph):
        graph_elt = self.graphs.find('./graph[@name="{}"]'.format(graph))
        elt_options = None
        for child in graph_elt:
            if child.tag in ("load_options", "generate_options"):
                elt_options = child
        return _xml_to_dict(elt_options, di_type)
    
    def get_reference_result(self, graph, result_name):
        elt_test = self.root.find('./graph[@name="{}"]'.format(graph))
        elt_result = elt_test.find('./[@name="{}"]'.format(result_name))
        return self.result(elt_result)


#-----------------------------------------------------------------------------#
# TestBasis class
#------------------------
#

@with_metaclass(ABCMeta)
class TestBasis(unittest.TestCase):

    '''
    Class defining the graphs and the conditions in which they will be tested.
    
    warning ::
        All test methods should be of the form:
        ``def test_method(self, graph, **kwargs)``
    '''
    
    #-------------------------------------------------------------------------#
    # Class properties
    
    tolerance = 1e-5
    parser = XmlHandler()
    graphs = []
    
    #-------------------------------------------------------------------------#
    # Instance
    
    def __init__(self):
        super(unittest.TestCase, self).__init__()
        self.make_graphs()

    @abstractproperty
    def test_name(self):
        pass
    
    def get_expected_result(self, graph, res_name):
        return self.parser.get_result(graph.get_name(), res_name)

    def make_graphs(self):
        for graph_name in parser.get_graph_list(self.test_name):
            self.__class__.graphs.append(self.gen_graph(graph_name))

    @abstractmethod
    def gen_graph(self, graph_name):
        pass
    

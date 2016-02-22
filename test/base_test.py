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
from abc import ABCMeta, abstractmethod

import numpy as np
import xml.etree.ElementTree as xmlet

import nngt



# xml file containing the graphs, their properties, and the instructions
xml_file = "graph_tests.xml"
if xml_file not in listdir("."):
    xml_file = "test/graph_tests.xml"
    if xml_file not in listdir("."):
        raise RuntimeError("File `graph_tests.xml` not found! Tests must be \
run from the same folder as `setup.py` or from the `test` folder.")


#-----------------------------------------------------------------------------#
# Tools
#------------------------
#

def _bool_from_string(string):
	return True if (string.lower() == "true") else False

def _make_instructions(string):
    pass

#-----------------------------------------------------------------------------#
# Decorators
#------------------------
#

def with_metaclass(mcls):
    ''' Python 2/3 compatible metaclass declaration. '''
    def decorator(cls):
        body = vars(cls).copy()
        # clean out class body
        body.pop('__dict__', None)
        body.pop('__weakref__', None)
        return mcls(cls.__name__, cls.__bases__, body)
    return decorator

def foreach_graph(graphs):
    '''
    Decorator that automatically does the test for all graph instructions
    provided in the argument.
    '''
    def decorator(func):          
        def wrapper(*args, **kwargs):
            self = args[0]
            for graph_instruction in graphs:
                graph = self.make_graph(graph_instruction)
                reference = self.get_expected_result(graph_instruction)
                computed = func(self, graph, **kwargs)
                assert( (reference-computed)/reference < self.tolerance )
        return wrapper
    return decorator
        

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
    
    def get_instructions(self, test):
        elt_test = self.root.find('./test[@name="{}"]'.format(test))
        return _make_instructions(elt_test.attrib["instructions"])
    
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
        ``def method(self, graph, **kwargs)``
    '''
    
    #-------------------------------------------------------------------------#
    # Class properties
    
    tolerance = 1e-5
    dir_graph = "test/Networks"
    lst_gfiles = [f for f in listdir(dir_graph) if isfile(join(dir_graph, f))]
    
    load_options = {
        "p2p-Gnutella04.txt": { format:"edge_list", delimiter:"\t" }
    }
    
    @staticmethod
    def 
    
    #-------------------------------------------------------------------------#
    # Instance
    
    def __init__(self):
        super(unittest.TestCase, self).__init__()
    
    @abstractmethod
    def get_expected_result(self, graph_instruction):
        pass
    
    @abstractmethod
    def make_graph(self, graph_instruction):
        pass
    

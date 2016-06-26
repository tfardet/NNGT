#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" GraphObject for subclassing the libraries graphs """

from collections import OrderedDict
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from six import add_metaclass
import nngt.globals
from nngt.globals import BWEIGHT
from nngt.lib import InvalidArgument



adjacency = nngt.globals.analyze_graph["adjacency"]

#-----------------------------------------------------------------------------#
# Library-dependent graph properties
#------------------------
#

class MetaAbcProp(ABCMeta, type):
    pass

@add_metaclass(MetaAbcProp)
class BaseProperty(dict):
    
    def __init__(self, parent):
        self.parent = parent
        self.stored = {}
    
    @abstractmethod
    def __getitem__(self, name):
        pass
    
    @abstractmethod
    def __setitem__(self, name, value):
        pass


#-----------------------------------------------------------------------------#
# BaseGraph
#------------------------
#

class MetaAbcGraph(ABCMeta, type):
    pass

@add_metaclass(ABCMeta)
class BaseGraph(nngt.globals.config["graph"]):
    
    #-------------------------------------------------------------------------#
    # Class methods and attributes

    nattr_class = None
    eattr_class = None
    
    @classmethod
    def to_graph_object(cls, obj, weighted=True, directed=True):
        obj.__class__ = cls
        edges = nngt.globals.analyze_graph["get_edges"](obj)
        obj._nattr = cls.nattr_class(obj)
        obj._eattr = cls.eattr_class(obj)
        obj._edges = OrderedDict()
        obj._directed = directed
        obj._weighted = weighted
        for i, edge in enumerate(edges):
            obj._edges[tuple(edge)] = i
        return obj
    
    #-------------------------------------------------------------------------#
    # Shared properties methods
    
    @property
    def edge_index(self):
        ''' :class:`OrderedDict` containing the edges as keys (2-tuple) and
        their index at the time of their creation as value '''
        return self._edges
        
    @property
    def edges_array(self):
        ''' Edges of the graph, sorted by order of creation, as an array of
        2-tuple. '''
        return np.array(tuple(self._edges.keys()))

    @property
    def nproperties(self):
        return self._nattr

    @property
    def eproperties(self):
        return self._eattr
    
    def new_node_attribute(self, name, value_type, values=None, val=None):
         self._nattr.new_na(name, value_type, values, val)

    def new_edge_attribute(self, name, value_type, values=None, val=None):
         self._eattr.new_ea(name, value_type, values, val)

    def remove_edge(self, edge):
        raise NotImplementedError("This function has been removed because it \
            makes using edge properties too complicated")

    def remove_vertex(self, node, fast=False):
        raise NotImplementedError("This function has been removed because it \
            makes using node properties too complicated")
        
    def adjacency_matrix(self, types=True, weights=True):
        weights = "weight" if weights is True else weights
        types = "type" if types is True else False
        mat = adjacency(self, weights)
        if types in self.attributes() and weights in (True, "weight"):
            mtype = adjacency(self, weight="type")
            return mat*mtype
        else:
            return mat
    
    #-------------------------------------------------------------------------#
    # Properties and methods to implement
    
    #~ @abstractmethod
    #~ def inhibitory_subgraph(self):
        #~ pass
    #~ 
    #~ @abstractmethod
    #~ def excitatory_subgraph(self, n=1, ntype=1):
        #~ pass
    
    @abstractmethod
    def new_node(self, n=1, ntype=1):
        pass
    
    @abstractmethod
    def new_edge(self, source, target, weight=1.):
        pass
        
    @abstractmethod
    def new_edges(self, edge_list, eprops=None):
        pass
        
    @abstractmethod
    def node_nb(self):
        pass

    @abstractmethod
    def edge_nb(self):
        pass
    
    @abstractmethod
    def degree_list(self, node_list=None, deg_type="total", use_weights=True):
        pass

    @abstractmethod
    def betweenness_list(self, use_weights=True, as_prop=False, norm=True):
        pass

    @abstractmethod
    def clear_all_edges(self):
        pass

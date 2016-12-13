#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" GraphObject for subclassing the libraries graphs """

from collections import OrderedDict
from abc import ABCMeta, abstractmethod, abstractproperty
from six import add_metaclass
from weakref import ref

import numpy as np

import nngt.globals
from nngt.globals import BWEIGHT
from nngt.lib import InvalidArgument



adjacency = nngt.globals.analyze_graph["adjacency"]

#-----------------------------------------------------------------------------#
# Library-dependent graph properties
#------------------------
#

class BaseProperty(dict):
    
    def __init__(self, parent):
        self.parent = ref(parent)
        
    def value_type(self, key=None):
        if key is not None:
            return super(BaseProperty, self).__getitem__(key)
        else:
            return { k:super(BaseProperty, self).__getitem__(k) for k in self }
    
    def values(self):
        return [ self[k] for k in self ]
    
    def itervalues(self):
        return ( self[k] for k in self )
    
    def items(self):
        return [ (k, self[k]) for k in self ]
    
    def iteritems(self):
        return ( (k, self[k]) for k in self )


#-----------------------------------------------------------------------------#
# BaseGraph
#------------------------
#

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
    @abstractmethod
    def edges_array(self):
        pass

    @property
    def nproperties(self):
        return self._nattr

    @property
    def eproperties(self):
        return self._eattr
    
    def new_node_attribute(self, name, value_type, values=None, val=None):
         self._nattr.new_na(name, value_type, values, val)
         
    def remove_edge(self, edge):
        raise NotImplementedError("This function has been removed because it \
            makes using edge properties too complicated")

    def remove_vertex(self, node, fast=False):
        raise NotImplementedError("This function has been removed because it \
            makes using node properties too complicated")
        
    def adjacency_matrix(self, types=True, weights=True):
        '''
        Return the graph adjacency matrix.
        NB : source nodes are represented by the rows, targets by the
        corresponding columns.

        Parameters
        ----------
        types : bool, optional (default: True)
            Wether the edge types should be taken into account (negative values
            for inhibitory connections).
        weights : bool or string, optional (default: True)
            Whether the adjacecy matrix should be weighted. If True, all
            connections are multiply bythe associated synaptic strength; if
            weight is a string, the connections are scaled bythe corresponding
            edge attribute.

        Returns
        -------
        mat : :class:`scipy.sparse.csr` matrix
            The adjacency matrix of the graph.
        '''
        weights = "weight" if weights is True else weights
        types = "type" if types is True else False
        mat = adjacency(self, weights)
        if types in self.attributes() and weights in (True, "weight"):
            mtype = adjacency(self, weight="type")
            return mat * mtype
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
    def edge_id(self, edge):
        pass
    
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
    def neighbours(self, node, mode="all"):
        pass

    @abstractmethod
    def clear_all_edges(self):
        pass

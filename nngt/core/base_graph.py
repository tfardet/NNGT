#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" GraphObject for subclassing the libraries graphs """

from collections import OrderedDict
from abc import ABCMeta, abstractmethod, abstractproperty
from six import add_metaclass
from weakref import ref

import numpy as np

import nngt
from nngt.lib import InvalidArgument, BWEIGHT, nonstring_container


# ---------------------------------- #
# Library-dependent graph properties #
# ---------------------------------- #

class BaseProperty(dict):
    
    def __init__(self, parent):
        self.parent = ref(parent)
        self._num_values_set = {}
        
    def value_type(self, key=None):
        if key is not None:
            return super(BaseProperty, self).__getitem__(key)
        else:
            return {k:super(BaseProperty, self).__getitem__(k) for k in self}

    # redefine dict values/items to use the __getitem__ that will be
    # overwritten by the child classes

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
class BaseGraph(nngt._config["graph"]):
    
    #-------------------------------------------------------------------------#
    # Class methods and attributes

    nattr_class = None
    eattr_class = None
    
    @classmethod
    def to_graph_object(cls, obj, weighted=True, directed=True):
        obj.__class__ = cls
        edges = nngt.analyze_graph["get_edges"](obj)
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
         
    def remove_edge(self, edge):
        raise NotImplementedError(
            "This function has been removed because it makes using edge "
            "properties too complicated.")

    def remove_vertex(self, node, fast=False):
        raise NotImplementedError(
            "This function has been removed because it makes using node"
            "properties too complicated.")
        
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
        mat = nngt.analyze_graph["adjacency"](self, weights)
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
    def new_node(self, n=1, ntype=1, attributes=None):
        pass

    @abstractmethod
    def new_edge(self, source, target, weight=1.):
        pass

    @abstractmethod
    def new_edges(self, edge_list, attributes=None):
        pass
        
    def attr_new_edges(self, edge_list, attributes=None):
        if attributes:
            for k in attributes.keys():
                if k not in self.attributes() and k in ("weight", "delay"):
                    self._eattr.new_attribute(name=k, value_type="double")
            # take care of classic attributes
            if "weight" in attributes:
                self._eattr.set_attribute(
                    "weight", attributes["weight"], edges=edge_list)
            if "delay" in attributes:
                self._eattr.set_attribute(
                    "delay", attributes["delay"], edges=edge_list)
            if "distance" in attributes:
                raise NotImplementedError("distance not implemented yet")
                #~ self.set_distances(elist=edge_list,
                                   #~ dlist=attributes["distance"])
            # take care of potential additional attributes
            if "names" in attributes:
                num_attr = len(attributes["names"])
                for i in range(num_attr):
                    v = attributes["values"]
                    if not nonstring_container(v):
                        v = np.repeat(v, self.edge_nb())
                    self._eattr.new_attribute(attributes["names"][i],
                                              attributes["types"][i], values=v)
        
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

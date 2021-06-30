#-*- coding:utf-8 -*-
#
# core/graph_interface.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Base classes to interface the various libraries' graphs """

from abc import abstractmethod, abstractproperty
from collections import OrderedDict
from copy import deepcopy
from weakref import ref

import logging

import numpy as np
from scipy.sparse import csr_matrix

import nngt
from nngt.lib import InvalidArgument, BWEIGHT, nonstring_container, is_integer
from nngt.lib.graph_helpers import _get_edge_attr, _get_syn_param
from nngt.lib.converters import _np_dtype, _to_np_array
from nngt.lib.logger import _log_message


logger = logging.getLogger(__name__)


# ---------------------------------- #
# Library-dependent graph properties #
# ---------------------------------- #

class BaseProperty(dict):

    def __init__(self, parent):
        self.parent = ref(parent)
        self._num_values_set = {}

    def value_type(self, key=None):
        if key is not None:
            return super(type(self), self).__getitem__(key)
        else:
            return {k: super(type(self), self).__getitem__(k) for k in self}

    # redefine dict values/items to use the __getitem__ that will be
    # overwritten by the child classes

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]


# -------------- #
# GraphInterface #
# -------------- #

class GraphInterface:

    #------------------------------------------------------------------#
    # Class attributes

    _nattr_class = None
    _eattr_class = None

    #------------------------------------------------------------------#
    # Shared properties methods

    @property
    @abstractmethod
    def edges_array(self):
        pass

    #------------------------------------------------------------------#
    # Properties and methods to implement

    #~ @abstractmethod
    #~ def inhibitory_subgraph(self):
        #~ pass
    #~
    #~ @abstractmethod
    #~ def excitatory_subgraph(self, n=1, neuron_type=1):
        #~ pass

    @abstractmethod
    def edge_id(self, edge):
        pass

    @abstractmethod
    def has_edge(self, edge):
        pass

    @abstractmethod
    def new_node(self, n=1, neuron_type=1, attributes=None):
        pass

    @abstractmethod
    def delete_nodes(self, nodes):
        pass

    @abstractmethod
    def new_edge(self, source, target, attributes=None, ignore=False):
        pass

    @abstractmethod
    def new_edges(self, edge_list, attributes=None, check_duplicates=False,
                  check_self_loops=True, check_existing=True,
                  ignore_invalid=False):
        pass

    @abstractmethod
    def delete_edges(self, edges):
        pass

    @abstractmethod
    def node_nb(self):
        pass

    @abstractmethod
    def edge_nb(self):
        pass

    @abstractmethod
    def get_degrees(self, mode="total", nodes=None, weights=False):
        pass

    @abstractmethod
    def neighbours(self, node, mode="all"):
        pass

    @abstractmethod
    def clear_all_edges(self):
        pass

    @abstractmethod
    def _from_library_graph(self, graph, copy=True):
        pass

    def _attr_new_edges(self, edge_list, attributes=None):
        ''' Generate attributes for newly created edges. '''
        num_edges = len(edge_list)

        if num_edges:
            attributes = {} if attributes is None else attributes
            specials = ("weight", "delay", 'distance')

            for k in attributes.keys():
                if k not in self.edge_attributes and k in specials:
                    self._eattr.new_attribute(name=k, value_type="double")

            # take care of classic attributes
            bio_weights = False
            bio_delays = False

            # distance must come first
            if self.is_spatial() or "distance" in attributes:
                prop = attributes.get("distance", None)
                values = _get_edge_attr(
                    self, edge_list, 'distance', prop, last_edges=True)
                self._eattr.set_attribute(
                    "distance", values, edges=edge_list, last_edges=True)

            # first check for potential syn_spec if Network
            if self.is_network():
                for syn_param in self.population.syn_spec.values():
                    bio_weights += ("weight" in syn_param)
                    bio_delays  += ("delay" in syn_param)

            # then weights
            if bio_weights:
                syn_spec = self.population.syn_spec

                mat = csr_matrix(
                    (np.repeat(1., num_edges),
                    (edge_list[:, 0], edge_list[:, 1])),
                    (self.population.size, self.population.size))

                for name1, g1 in self.population.items():
                    for name2, g2 in self.population.items():
                        src_slice = slice(g1.ids[0], g1.ids[-1]+1)
                        tgt_slice = slice(g2.ids[0], g2.ids[-1]+1)

                        e12 = mat[src_slice, tgt_slice].nonzero()
    
                        syn_prop = _get_syn_param(
                            name1, g1, name2, g2, syn_spec, "weight")
                        syn_prop = 1. if syn_prop is None else syn_prop

                        if isinstance(syn_prop, dict):
                            # through set_weights for dict
                            distrib = syn_prop["distribution"]
                            del syn_prop["distribution"]
                            self.set_weights(elist=e12, distribution=distrib,
                                             parameters=syn_prop)
                        elif nonstring_container(syn_prop):
                            # otherwise direct attribute set
                            self.set_edge_attribute(
                                "weight", values=syn_prop, value_type="double",
                                edges=edge_list)
                        else:
                            self.set_edge_attribute(
                                "weight", val=syn_prop, value_type="double",
                                edges=edge_list)
            elif self.is_weighted() or "weight" in attributes:
                values = _get_edge_attr(
                    self, edge_list, 'weight', attributes.get("weight", None),
                    last_edges=True)

                self._eattr.set_attribute(
                    "weight", values, edges=edge_list, last_edges=True)

            # then delay
            if self.is_network() or "delay" in self.edge_attributes:
                prop = attributes.get("delay", None)

                values = _get_edge_attr(
                    self, edge_list, 'delay', prop, last_edges=True)

                self._eattr.set_attribute(
                    "delay", values, edges=edge_list, last_edges=True)

            for k in attributes.keys():
                if k not in specials:
                    if k in self.edge_attributes:
                        values = _get_edge_attr(
                            self, edge_list, k, attributes[k], last_edges=True)

                        self._eattr.set_attribute(k, values, edges=edge_list,
                                                  last_edges=True)
                    else:
                        raise RuntimeError("Unknown attribute: '" + k + "'.")

            # take care of potential new attributes
            if "names" in attributes:
                num_attr = len(attributes["names"])

                for i in range(num_attr):
                    v = attributes["values"][i]

                    if not nonstring_container(v):
                        v = np.repeat(v, self.edge_nb())

                    self._eattr.new_attribute(attributes["names"][i],
                                              attributes["types"][i], values=v)



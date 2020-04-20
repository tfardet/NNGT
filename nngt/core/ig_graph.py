#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
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

""" iGraph subclassing """

from collections import OrderedDict
from copy import deepcopy
import logging

import numpy as np
import scipy.sparse as ssp

import nngt
from nngt.lib import InvalidArgument, nonstring_container, BWEIGHT, is_integer
from nngt.lib.graph_helpers import _to_np_array, _get_dtype
from nngt.lib.io_tools import _np_dtype
from nngt.lib.logger import _log_message
from .graph_interface import GraphInterface, BaseProperty


logger = logging.getLogger(__name__)


# ---------- #
# Properties #
# ---------- #

class _IgNProperty(BaseProperty):

    '''
    Class for generic interactions with nodes properties (igraph)
    '''

    def __getitem__(self, name):
        g = self.parent()._graph
        dtype = _np_dtype(super(_IgNProperty, self).__getitem__(name))
        return _to_np_array(g.vs[name], dtype=dtype)

    def __setitem__(self, name, value):
        g    = self.parent()._graph
        size = g.vcount()

        if name in self:
            if len(value) == size:
                g.vs[name] = value
            else:
                raise ValueError("A list or a np.array with one entry per "
                                 "node in the graph is required")
        else:
            raise InvalidArgument("Attribute does not exist yet, use "
                                  "set_attribute to create it.")

    def new_attribute(self, name, value_type, values=None, val=None):
        g = self.parent()._graph

        if val is None:
            if value_type == "int":
                val = int(0)
            elif value_type == "double":
                val = np.NaN
            elif value_type == "string":
                val = ""
            else:
                val = None
                value_type = "object"

        if values is None:
            values = [deepcopy(val) for _ in range(g.vcount())]

        # store name and value type in the dict
        super(_IgNProperty,self).__setitem__(name, value_type)
        # store the real values in the attribute
        self[name] = values
        self._num_values_set[name] = len(values)

    def set_attribute(self, name, values, nodes=None):
        '''
        Set the node attribute.

        Parameters
        ----------
        name : str
            Name of the node attribute.
        values : array, size N
            Values that should be set.
        nodes : array-like, optional (default: all nodes)
            Nodes for which the value of the property should be set. If `nodes`
            is not None, it must be an array of size N.
        '''
        g = self.parent()._graph

        num_nodes = g.vcount()
        num_n = len(nodes) if nodes is not None else num_nodes

        if num_n == num_nodes:
            self[name] = values
        elif num_n:
            if num_n != len(values):
                raise ValueError("`nodes` and `values` must have the same "
                                 "size; got respectively " + str(num_n) + \
                                 " and " + str(len(values)) + " entries.")
            if self._num_values_set[name] == num_nodes - num_n:
                g.vs[-num_n:][name] = values
            else:
                for n, val in zip(nodes, values):
                    g.vs[n][name] = val
        if num_n:
            self._num_values_set[name] = num_nodes


class _IgEProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (igraph) '''

    def __getitem__(self, name):
        g = self.parent()._graph

        if isinstance(name, slice):
            eprop = {}
            for k in self.keys():
                dtype = _np_dtype(super(_IgEProperty, self).__getitem__(k))
                eprop[k] = _to_np_array(g.es[k], dtype=dtype)[name]
            return eprop
        elif nonstring_container(name):
            eprop = {}
            if nonstring_container(name[0]):
                eids = [g.get_eid(*e) for e in name]
                for k in self.keys():
                    dtype = _np_dtype(super(_IgEProperty, self).__getitem__(k))
                    eprop[k] = _to_np_array(g.es[k], dtype=dtype)[eids]
            else:
                eid = g.get_eid(*name)
                for k in self.keys():
                    eprop[k] = g.es[k][eid]
            return eprop

        dtype = _np_dtype(super(_IgEProperty, self).__getitem__(name))

        return _to_np_array(g.es[name], dtype=dtype)

    def __setitem__(self, name, value):
        g = self.parent()._graph

        if name in self:
            size = g.ecount()
            if len(value) == size:
                g.es[name] = value
            else:
                raise ValueError("A list or a np.array with one entry per "
                                 "edge in the graph is required")
        else:
            raise InvalidArgument("Attribute does not exist yet, use "
                                  "set_attribute to create it.")

    def new_attribute(self, name, value_type, values=None, val=None):
        g = self.parent()._graph

        if val is None:
            if value_type == "int":
                val = int(0)
            elif value_type == "double":
                val = np.NaN
            elif value_type == "string":
                val = ""
            else:
                val = None
                value_type = 'object'

        if values is None:
            values = [deepcopy(val) for _ in range(g.ecount())]

        # store name and value type in the dict
        super(_IgEProperty, self).__setitem__(name, value_type)
        # store the real values in the attribute
        self[name] = values
        self._num_values_set[name] = len(values)

    def set_attribute(self, name, values, edges=None):
        '''
        Set the edge property.

        Parameters
        ----------
        name : str
            Name of the edge property.
        values : array
            Values that should be set.
        edges : array-like, optional (default: None)
            Edges for which the value of the property should be set. If `edges`
            is not None, it must be an array of shape `(len(values), 2)`.
        '''
        g = self.parent()._graph

        num_edges = g.ecount()
        num_e = len(edges) if edges is not None else num_edges

        if num_e == num_edges:
            self[name] = values
        elif num_e:
            if num_e != len(values):
                raise ValueError("`edges` and `values` must have the same "
                                 "size; got respectively " + str(num_e) + \
                                 " and " + str(len(values)) + " entries.")
            if self._num_values_set[name] == num_edges - num_e:
                g.es[-num_e:][name] = values
            else:
                for e, val in zip(edges, values):
                    eid = g.get_eid(*e)
                    g.es[eid][name] = val

        if num_e:
            self._num_values_set[name] = num_edges


# ----- #
# Graph #
# ----- #

class _IGraph(GraphInterface):

    '''
    Subclass of :class:`igraph.Graph`.
    '''

    _nattr_class = _IgNProperty
    _eattr_class = _IgEProperty    

    #-------------------------------------------------------------------------#
    # Constructor and instance properties

    def __init__(self, nodes=0, copy_graph=None, directed=True, weighted=False,
                 **kwargs):
        self._nattr = _IgNProperty(self)
        self._eattr = _IgEProperty(self)
        self._weighted = weighted
        self._directed = directed

        g = copy_graph.graph if copy_graph is not None else None

        if g is None:
            self._graph = nngt._config["graph"](n=nodes, directed=True)
        else:
            self._from_library_graph(g, copy=True)

    #-------------------------------------------------------------------------#
    # Graph manipulation

    def edge_id(self, edge):
        '''
        Return the ID a given edge or a list of edges in the graph.
        Raises an error if the edge is not in the graph or if one of the
        vertices in the edge is nonexistent.

        Parameters
        ----------
        edge : 2-tuple or array of edges
            Edge descriptor (source, target).

        Returns
        -------
        index : int or array of ints
            Index of the given `edge`.
        '''
        if is_integer(edge[0]):
            return self._graph.get_eid(*edge)
        elif nonstring_container(edge[0]):
            return self._graph.get_eids(edge)
        else:
            raise AttributeError("`edge` must be either a 2-tuple of ints or "
                                 "an array of 2-tuples of ints.")

    def has_edge(self, edge):
        '''
        Whether `edge` is present in the graph.

        .. versionadded:: 2.0
        '''
        try:
            self._graph.get_eid(*edge)
            return True
        except:
            return False

    @property
    def edges_array(self):
        '''
        Edges of the graph, sorted by order of creation, as an array of
        2-tuple.
        '''
        g = self._graph
        return np.array([(e.source, e.target) for e in g.es], dtype=int)

    def new_node(self, n=1, neuron_type=1, attributes=None, value_types=None,
                 positions=None, groups=None):
        '''
        Adding a node to the graph, with optional properties.

        Parameters
        ----------
        n : int, optional (default: 1)
            Number of nodes to add.
        neuron_type : int, optional (default: 1)
            Type of neuron (1 for excitatory, -1 for inhibitory)

        Returns
        -------
        The node or an iterator over the nodes created.
        '''
        g = self._graph

        first_node_idx = g.vcount()

        g.add_vertices(n)

        nodes = list(range(first_node_idx, first_node_idx + n))

        attributes = {} if attributes is None else deepcopy(attributes)

        if attributes:
            for k, v in attributes.items():
                if k not in self._nattr:
                    self._nattr.new_attribute(k, value_types[k], val=v)
                else:
                    v = v if nonstring_container(v) else [v]
                    self._nattr.set_attribute(k, v, nodes=nodes)

        # set default values for all attributes that were not set
        for k in self.nodes_attributes:
            if k not in attributes:
                dtype = self.get_attribute_type(k)
                if dtype == "string":
                    self._nattr.set_attribute(k, ["" for _ in nodes],
                                              nodes=nodes)
                elif dtype == "int":
                    self._nattr.set_attribute(k, [0 for _ in nodes],
                                              nodes=nodes)
                elif dtype == "double":
                    self._nattr.set_attribute(k, [np.NaN for _ in nodes],
                                              nodes=nodes)

        g.vs[nodes[0]:nodes[-1] + 1]['type'] = neuron_type

        if self.is_spatial():
            old_pos      = self._pos
            self._pos    = np.full((self.node_nb(), 2), np.NaN)
            num_existing = len(old_pos) if old_pos is not None else 0
            if num_existing != 0:
                self._pos[:num_existing, :] = old_pos
        if positions is not None:
            assert self.is_spatial(), \
                "`positions` argument requires a SpatialGraph/SpatialNetwork."
            self._pos[nodes] = positions

        if groups is not None:
            assert self.is_network(), \
                "`positions` argument requires a Network/SpatialNetwork."
            if nonstring_container(groups):
                assert len(groups) == n, "One group per neuron required."
                for g, node in zip(groups, nodes):
                    self.population.add_to_group(g, node)
            else:
                self.population.add_to_group(groups, nodes)

        if n == 1:
            return nodes[0]

        return nodes

    def new_edge(self, source, target, attributes=None, ignore=False):
        '''
        Adding a connection to the graph, with optional properties.

        Parameters
        ----------
        source : :class:`int/node`
            Source node.
        target : :class:`int/node`
            Target node.
        attributes : :class:`dict`, optional (default: ``{}``)
            Dictionary containing optional edge properties. If the graph is
            weighted, defaults to ``{"weight": 1.}``, the unit weight for the
            connection (synaptic strength in NEST).
        ignore : bool, optional (default: False)
            If set to True, ignore attempts to add an existing edge, otherwise
            raises an error.

        Returns
        -------
        The new connection.
        '''
        attributes = {} if attributes is None else deepcopy(attributes)

        # set default values for attributes that were not passed
        for k in self.edges_attributes:
            if k not in attributes:
                dtype = self.get_attribute_type(k)
                if dtype == "string":
                    attributes[k] = [""]
                elif dtype == "double" and k != "weight":
                    attributes[k] = [np.NaN]
                elif dtype == "int":
                    attributes[k] = [0]

        self.new_edges(((source, target),), attributes)

    def new_edges(self, edge_list, attributes=None, check_edges=True):
        '''
        Add a list of edges to the graph.

        .. warning ::
            This function currently does not check for duplicate edges between
            the existing edges and the added ones, but only inside `edge_list`!

        Parameters
        ----------
        edge_list : list of 2-tuples or np.array of shape (edge_nb, 2)
            List of the edges that should be added as tuples (source, target)
        attributes : :class:`dict`, optional (default: ``{}``)
            Dictionary containing optional edge properties. If the graph is
            weighted, defaults to ``{"weight": ones}``, where ``ones`` is an
            array the same length as the `edge_list` containing a unit weight
            for each connection (synaptic strength in NEST).
        check_edges : bool, optional (default: True)
            Check for duplicate edges and self-loops.

        @todo: add example

        Returns
        -------
        Returns new edges only.
        '''
        attributes = {} if attributes is None else deepcopy(attributes)
        num_edges  = len(edge_list)

        # set default values for attributes that were not passed
        for k in self.edges_attributes:
            if k not in attributes:
                dtype = self.get_attribute_type(k)
                if dtype == "string":
                    attributes[k] = ["" for _ in range(num_edges)]
                elif dtype == "double" and k != "weight":
                    attributes[k] = [np.NaN for _ in range(num_edges)]
                elif dtype == "int":
                    attributes[k] = [0 for _ in range(num_edges)]

        new_attr = None

        if check_edges:
            # @todo: speed this up
            new_attr = {key: [] for key in attributes}
            eweight_list = OrderedDict()
            for i, e in enumerate(edge_list):
                tpl_e = tuple(e)
                if tpl_e in eweight_list:
                    eweight_list[tpl_e] += 1
                elif e[0] == e[1]:
                    _log_message(logger, "WARNING",
                                 "Self-loop on {} ignored.".format(e[0]))
                else:
                    eweight_list[tpl_e] = 1
                    for k, vv in attributes.items():
                        new_attr[k].append(vv[i])
            edge_list = np.array(list(eweight_list.keys()))
        else:
            edge_list = np.array(edge_list)
            new_attr = attributes

        if not self._directed:
            recip_edges = edge_list[:,::-1]
            # slow but works
            unique = ~(recip_edges[..., np.newaxis]
                      == edge_list[..., np.newaxis].T).all(1).any(1)
            edge_list = np.concatenate((edge_list, recip_edges[unique]))

            for key, val in new_attr.items():
                new_attr[key] = np.concatenate((val, val[unique]))

        self._graph.add_edges(edge_list)

        # call parent function to set the attributes
        self._attr_new_edges(edge_list, attributes=new_attr)

        return edge_list

    def clear_all_edges(self):
        ''' Remove all edges from the graph '''
        self._graph.delete_edges(None)
        self._eattr.clear()

    #-------------------------------------------------------------------------#
    # Getters

    def node_nb(self):
        ''' Number of nodes in the graph '''
        return self._graph.vcount()

    def edge_nb(self):
        ''' Number of edges in the graph '''
        return self._graph.ecount()

    def get_degrees(self, mode="total", nodes=None, weights=None):
        g = self._graph
        w = _get_weights(self, weights)
    
        mode = 'all' if mode == 'total' else mode

        if weights not in {False, None}:
            return np.array(g.strength(nodes, mode=mode, weights=w))

        return np.array(g.degree(nodes, mode=mode), dtype=int)

    def betweenness_list(self, btype="both", use_weights=False, norm=True,
                         **kwargs):
        g = self._graph

        n = g.vcount()
        e = g.ecount()

        ncoeff_norm = (n-1)*(n-2)
        ecoeff_norm = (e-1)*(e-2)/2.

        w, nbetw, ebetw = None, None, None

        if use_weights:
            if "bweight" in g.es:
                w = g.es['bweight']
            else:
                w  = self.get_weights()
                w  = np.max(w) - w
                minw = np.min(w)
                w += 1e-5*minw if minw > 0 else 1e-5

        if btype in ("both", "node"):
            nbetw = np.array(g.betweenness(weights=w))

        if btype in ("both", "edge"):
            ebetw = np.array(g.edge_betweenness(weights=w))

        if btype == "node":
            return nbetw/ncoeff_norm if norm else nbetw
        elif btype == "edge":
            return ebetw/ecoeff_norm if norm else ebetw
        elif norm:
            return nbetw/ncoeff_norm, ebetw/ecoeff_norm

        return nbetw, ebetw

    def neighbours(self, node, mode="all"):
        '''
        Return the neighbours of `node`.

        Parameters
        ----------
        node : int
            Index of the node of interest.
        mode : string, optional (default: "all")
            Type of neighbours that will be returned: "all" returns all the
            neighbours regardless of directionality, "in" returns the
            in-neighbours (also called predecessors) and "out" retruns the
            out-neighbours (or successors).

        Returns
        -------
        neighbours : tuple
            The neighbours of `node`.
        '''
        g = self._graph

        if mode == "all":
            return (n for n in g.neighbors(node, mode=3))
        elif mode == "in":
            return (n for n in g.neighbors(node, mode=2))
        elif mode == "out":
            return (n for n in g.neighbors(node, mode=1))

        raise ArgumentError('Invalid `mode` argument {}; possible values are '
                            '"all", "out" or "in".'.format(mode))

    def _from_library_graph(self, graph, copy=True):
        ''' Initialize `self._graph` from existing library object. '''
        nodes = graph.vcount()
        edges = graph.ecount()

        self._graph = graph.copy() if copy else graph

        # get attributes names and "types" and initialize them
        if nodes:
            for key, val in graph.vs[0].attributes().items():
                super(type(self._nattr), self._nattr).__setitem__(
                    key, _get_dtype(val))

        if edges:
            for key, val in graph.es[0].attributes().items():
                super(type(self._eattr), self._eattr).__setitem__(
                    key, _get_dtype(val))


def _get_weights(g, weights):
    if weights in g.edges_attributes:
        # existing edge attribute
        return np.array(g._graph.es[weights])
    elif nonstring_container(weights):
        # user-provided array
        return np.array(weights)
    elif weights is True:
        # "normal" weights
        return np.array(g._graph.es["weight"])
    elif not weights:
        # unweighted
        return None

    raise ValueError("Unknown edge attribute '" + str(weights) + "'.")

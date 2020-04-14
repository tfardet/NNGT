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

""" Networkx subclassing """

from collections import OrderedDict, deque
from copy import deepcopy
import logging
import sys

import numpy as np
import scipy.sparse as ssp

import nngt
from nngt.lib import InvalidArgument, BWEIGHT, nonstring_container, is_integer
from nngt.lib.graph_helpers import _to_np_array, _get_dtype
from nngt.lib.io_tools import _np_dtype
from nngt.lib.logger import _log_message
from .graph_interface import GraphInterface, BaseProperty


logger = logging.getLogger(__name__)


# ---------- #
# Properties #
# ---------- #

class _NxNProperty(BaseProperty):

    '''
    Class for generic interactions with nodes properties (networkx)
    '''

    def __getitem__(self, name):
        g = self.parent()._graph

        lst = [g.nodes[i][name] for i in range(g.number_of_nodes())]

        dtype = _np_dtype(super(_NxNProperty, self).__getitem__(name))

        return _to_np_array(lst, dtype=dtype)

    def __setitem__(self, name, value):
        g    = self.parent()._graph
        size = g.number_of_nodes()

        if name in self:
            if len(value) == size:
                for i in range(size):
                    g.nodes[i][name] = value[i]
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
            values = [deepcopy(val)
                      for _ in range(g.number_of_nodes())]

        # store name and value type in the dict
        super(_NxNProperty, self).__setitem__(name, value_type)
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

        num_nodes = g.number_of_nodes()
        num_n = len(nodes) if nodes is not None else num_nodes

        if num_n == num_nodes:
            self[name] = values
        else:
            if num_n != len(values):
                raise ValueError("`nodes` and `values` must have the same "
                                 "size; got respectively " + str(num_n) + \
                                 " and " + str(len(values)) + " entries.")
            else:
                for n, val in zip(nodes, values):
                    g.nodes[n][name] = val
        self._num_values_set[name] = num_nodes


class _NxEProperty(BaseProperty):

    ''' Class for generic interactions with edge properties (networkx)  '''

    def __getitem__(self, name):
        g     = self.parent()._graph
        edges = None

        if isinstance(name, slice):
            edges = self.parent().edges_array[name]
        elif nonstring_container(name):
            if nonstring_container(name[0]):
                edges = name
            else:
                if len(name) != 2:
                    raise InvalidArgument(
                        "key for edge attribute must be one of the following: "
                        "slice, list of edges, edges or attribute name.")
                return g[name[0]][name[1]]

        if isinstance(name, str):
            dtype = _np_dtype(super(_NxEProperty, self).__getitem__(name))
            eprop = np.empty(g.number_of_edges(), dtype=dtype)

            for d, eid in zip(g.edges(data=name), g.edges(data="eid")):
                eprop[eid[2]] = d[2]

            return eprop

        eprop = {k: [] for k in self.keys()}

        for edge in edges:
            data = g.get_edge_data(edge[0], edge[1])
            if data is None:
                raise ValueError("Edge {} does not exist.".format(edge))
            for k, v in data.items():
                if k != "eid":
                    eprop[k].append(v)
        dtype = None

        for k, v in eprop.items():
            dtype    = _np_dtype(super(_NxEProperty, self).__getitem__(k))
            eprop[k] = _to_np_array(v, dtype)

        return eprop

    def __setitem__(self, name, value):
        g = self.parent()._graph

        if name in self:
            size = g.number_of_edges()
            if len(value) == size:
                for e in g.edges(data="eid"):
                    g.edges[e[0], e[1]][name] = value[e[2]]
            else:
                raise ValueError(
                    "A list or a np.array with one entry per edge in the "
                    "graph is required. For attribute "
                    "'{}', got {} entries vs {} edges.".format(
                        name, len(value), size))
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
            values = [deepcopy(val)
                      for _ in range(g.number_of_edges())]

        # store name and value type in the dict
        super(_NxEProperty, self).__setitem__(name, value_type)

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

        num_edges = g.number_of_edges()
        num_e = len(edges) if edges is not None else num_edges

        if num_e == num_edges:
            self[name] = values
        else:
            if num_e != len(values):
                raise ValueError("`edges` and `values` must have the same "
                                 "size; got respectively " + str(num_e) + \
                                 " and " + str(len(values)) + " entries.")
            for i, e in enumerate(edges):
                try:
                    edict = g[e[0]][e[1]]
                except:
                    edict = {}
                edict[name] = values[i]
                g.add_edge(e[0], e[1], **edict)
        self._num_values_set[name] = num_edges


# ----- #
# Graph #
# ----- #

class _NxGraph(GraphInterface):

    '''
    Subclass of networkx Graph
    '''

    _nattr_class = _NxNProperty
    _eattr_class = _NxEProperty

    #-------------------------------------------------------------------------#
    # Class properties

    di_value = { "string": "", "double": 0., "int": int(0) }

    #-------------------------------------------------------------------------#
    # Constructor and instance properties

    def __init__(self, nodes=0, copy_graph=None, directed=True, weighted=False,
                 **kwargs):
        self._directed = directed
        self._weighted = weighted
        self._nattr = _NxNProperty(self)
        self._eattr = _NxEProperty(self)

        g = copy_graph.graph if copy_graph is not None else None

        if g is not None:
            if not directed and g.is_directed():
                g = g.to_undirected()
            elif directed and not g.is_directed():
                g = g.to_directed()

            self._from_library_graph(g, copy=True)
        else:
            nx = nngt._config["library"]

            self._graph = nx.DiGraph() if directed else nx.Graph()

            if nodes:
                self._graph.add_nodes_from(range(nodes))

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
        g = self._graph

        if is_integer(edge[0]):
            return g[edge[0]][edge[1]]["eid"]
        elif nonstring_container(edge[0]):
            return [g[e[0]][e[1]]["eid"] for e in edge]

        raise AttributeError("`edge` must be either a 2-tuple of ints or "
                             "an array of 2-tuples of ints.")

    def has_edge(self, edge):
        '''
        Whether `edge` is present in the graph.

        .. versionadded:: 2.0
        '''
        return self._graph.has_edge(edge)

    @property
    def edges_array(self):
        '''
        Edges of the graph, sorted by order of creation, as an array of
        2-tuple.
        '''
        g     = self._graph
        edges = np.zeros((g.number_of_edges(), 2), dtype=int)

        # fast iteration using list comprehension
        # could also be done with deque and map (deque forces lazy map to run)
        # deque(map(lambda x: _gen_edges(edges, x), g.edges(data="eid")))

        [_gen_edges(edges, x) for x in g.edges(data="eid")]

        return edges

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
        The node or a list of the nodes created.
        '''
        g = self._graph
        num_nodes = g.number_of_nodes()

        new_nodes = list(range(num_nodes, num_nodes + n))

        for v in new_nodes:
            g.add_node(v)

        attributes = {} if attributes is None else deepcopy(attributes)

        if attributes:
            for k, v in attributes.items():
                if k not in self._nattr:
                    self._nattr.new_attribute(k, value_types[k], val=v)
                else:
                    v = v if nonstring_container(v) else [v]
                    self._nattr.set_attribute(k, v, nodes=new_nodes)

        # set default values for all attributes that were not set
        for k in self.nodes_attributes:
            if k not in attributes:
                dtype = self.get_attribute_type(k)
                filler = [None for _ in new_nodes]

                # change for strings, doubles and ints
                if dtype == "string":
                    filler = ["" for _ in new_nodes]
                elif dtype == "double":
                    filler = [np.NaN for _ in new_nodes]
                elif dtype == "int":
                    filler = [0 for _ in new_nodes]

                self._nattr.set_attribute(k, filler, nodes=new_nodes)

        if self.is_spatial():
            old_pos      = self._pos
            self._pos    = np.full((self.node_nb(), 2), np.NaN)
            num_existing = len(old_pos) if old_pos is not None else 0
            if num_existing != 0:
                self._pos[:num_existing, :] = old_pos

        if positions is not None and len(positions):
            assert self.is_spatial(), \
                "`positions` argument requires a SpatialGraph/SpatialNetwork."
            self._pos[new_nodes, :] = positions

        if groups is not None:
            assert self.is_network(), \
                "`positions` argument requires a Network/SpatialNetwork."
            if nonstring_container(groups):
                assert len(groups) == n, "One group per neuron required."
                for g, node in zip(groups, new_nodes):
                    self.population.add_to_group(g, node)
            else:
                self.population.add_to_group(groups, new_nodes)

        if len(new_nodes) == 1:
            return new_nodes[0]

        return new_nodes

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
        g = self._graph

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

        if g.has_edge(source, target):
            if not ignore:
                raise InvalidArgument("Trying to add existing edge.")
        else:
            for attr in attributes:
                if "_corr" in attr:
                    raise NotImplementedError("Correlated attributes are not "
                                              "available with networkx.")
            if self._weighted and "weight" not in attributes:
                attributes["weight"] = 1.

            g.add_edge(source, target)
            g[source][target]["eid"] = g.number_of_edges() - 1

            # call parent function to set the attributes
            self._attr_new_edges([(source, target)], attributes=attributes)

            if not self._directed:
                g.add_edge(target,source)
                g[source][target]["eid"] = g.number_of_edges() - 1
                for key, val in attributes.items():
                    g[target][source][key] = val
                self._attr_new_edges([(target, source)], attributes=attributes)

        return (source, target)

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
        g = self._graph

        attributes = {} if attributes is None else deepcopy(attributes)
        num_edges  = len(edge_list)

        for attr in attributes:
            if "_corr" in attr:
                raise NotImplementedError("Correlated attributes are not "
                                          "available with networkx.")

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

        initial_edges = g.number_of_edges()
        new_attr = None

        # @todo use map for speedup
        if check_edges:
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

        # create the edges
        num_added = len(edge_list)

        if num_added:
            arr_edges = np.zeros((num_added, 3), dtype=int)

            arr_edges[:, :2] = edge_list
            arr_edges[:, 2]  = np.arange(initial_edges,
                initial_edges + num_added)

            if not self._directed:
                recip_edges = edge_list[:, ::-1]
                # slow but works
                unique = ~(recip_edges[..., np.newaxis]
                          == edge_list[..., np.newaxis].T).all(1).any(1)
                edge_list = np.concatenate((edge_list, recip_edges[unique]))
                for key, val in new_attr.items():
                    new_attr[key] = np.concatenate((val, val[unique]))

            # create the edges with an eid attribute
            g.add_weighted_edges_from(arr_edges, weight="eid")

            # call parent function to set the attributes
            self._attr_new_edges(edge_list, attributes=new_attr)

        return edge_list

    def clear_all_edges(self):
        ''' Remove all edges from the graph '''
        g = self._graph
        g.remove_edges_from(tuple(g.edges()))
        self._eattr.clear()

    #-------------------------------------------------------------------------#
    # Getters

    def node_nb(self):
        ''' Number of nodes in the graph '''
        return self._graph.number_of_nodes()

    def edge_nb(self):
        ''' Number of edges in the graph '''
        return self._graph.number_of_edges()

    def degree_list(self, node_list=None, deg_type="total", use_weights=False):
        g = self._graph

        weight = 'weight' if use_weights else None
        di_deg = None

        if deg_type == 'total':
            di_deg = g.degree(node_list, weight=weight)
        elif deg_type == 'in':
            di_deg = g.in_degree(node_list, weight=weight)
        else:
            di_deg = g.out_degree(node_list, weight=weight)

        return np.array([d[1] for d in di_deg])

    def betweenness_list(self, btype="both", use_weights=False, **kwargs):
        g  = self._graph
        nx = nngt._config["library"]

        di_nbetw, di_ebetw = None, None

        wattr = None

        if use_weights:
            wattr = "bweight"

            if "bweight" not in self._eattr:
                w = self.get_weights()
                w = w.max() - w if use_weights else None
                self.new_edge_attribute("bweight", "double", values=w)

        if btype in ("both", "node"):
            di_nbetw = nx.betweenness_centrality(g, weight=wattr)
        if btype in ("both", "edge"):
            di_ebetw = nx.edge_betweenness_centrality(g, weight=wattr)

        if btype == "node":
            return np.array(tuple(di_nbetw.values()))
        elif btype == "edge":
            return np.array(tuple(di_ebetw.values()))

        return (np.array(tuple(di_nbetw.values())),
                np.array(tuple(di_ebetw.values())))

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
        neighbours : numpy array
            The neighbours of `node`.
        '''
        g = self._graph

        if mode == "all":
            return np.fromiter(
                chain(g.successors(node), g.predecessors(node)), dtype=int)
        elif mode == "in":
            return np.fromiter(self.predecessors(node), dtype=int)
        elif mode == "out":
            return np.fromiter(self.successors(node), dtype=int)

        raise ArgumentError('Invalid `mode` argument {}; possible values are '
                            '"all", "out" or "in".'.format(mode))

    def _from_library_graph(self, graph, copy=True):
        ''' Initialize `self._graph` from existing library object. '''
        # @todo check if nodes start from 0 and are continuous
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()

        self._graph = nngt._config["graph"](graph) if copy else graph

        # get attributes names and "types" and initialize them
        if nodes:
            for key, val in graph[0].items():
                super(type(self._nattr), self._nattr).__setitem__(
                    key, _get_dtype(val))

        if edges:
            e0 = next(iter(graph.edges))
            for key, val in graph.edges[e0].items():
                super(type(self._eattr), self._eattr).__setitem__(
                    key, _get_dtype(val))


# tool function to generate the edges_array

def _gen_edges(array, edata):
    source, target, eid = edata
    array[eid] = (source, target)

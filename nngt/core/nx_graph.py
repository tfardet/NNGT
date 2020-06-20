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
from itertools import chain
import logging
import sys

import numpy as np
import scipy.sparse as ssp

import nngt
from nngt.lib import InvalidArgument, BWEIGHT, nonstring_container, is_integer
from nngt.lib.connect_tools import (_cleanup_edges, _set_dist_new_edges,
                                    _set_default_edge_attributes)
from nngt.lib.graph_helpers import _get_dtype, _get_nx_weights
from nngt.lib.converters import _np_dtype, _to_np_array
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
            if len(name) == 0:
                return []

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
        return self._graph.has_edge(*edge)

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
        for k in self.node_attributes:
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
                self._pos[:num_existing] = old_pos

        if positions is not None and len(positions):
            assert self.is_spatial(), \
                "`positions` argument requires a SpatialGraph/SpatialNetwork."
            self._pos[new_nodes] = positions

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

    def new_edge(self, source, target, attributes=None, ignore=False,
                 self_loop=False):
        '''
        Adding a connection to the graph, with optional properties.

        .. versionchanged :: 2.0
            Added `self_loop` argument to enable adding self-loops.

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
            If set to True, ignore attempts to add an existing edge and accept
            self-loops; otherwise an error is raised.
        self_loop : bool, optional (default: False)
            Whether to allow self-loops or not.

        Returns
        -------
        The new connection or None if nothing was added.
        '''
        g = self._graph

        attributes = {} if attributes is None else deepcopy(attributes)

        # check that nodes exist
        num_nodes = g.number_of_nodes()

        if source >= num_nodes or target >= num_nodes:
            raise InvalidArgument("`source` or `target` does not exist.")

        # set default values for attributes that were not passed
        _set_default_edge_attributes(self, attributes, num_edges=1)

        if g.has_edge(source, target):
            if not ignore:
                raise InvalidArgument("Trying to add existing edge.")

            _log_message(logger, "WARNING",
                         "Existing edge {} ignored.".format((source, target)))
        else:
            if source == target:
                if not ignore and not self_loop:
                    raise InvalidArgument("Trying to add a self-loop.")
                elif ignore:
                    _log_message(logger, "WARNING",
                                 "Self-loop on {} ignored.".format(source))

                    return None

            for attr in attributes:
                if "_corr" in attr:
                    raise NotImplementedError("Correlated attributes are not "
                                              "available with networkx.")

            if self.is_weighted() and "weight" not in attributes:
                attributes["weight"] = 1.

            # check distance
            _set_dist_new_edges(attributes, self, [(source, target)])

            g.add_edge(source, target)

            g[source][target]["eid"] = g.number_of_edges() - 1

            # call parent function to set the attributes
            self._attr_new_edges([(source, target)], attributes=attributes)

        return (source, target)

    def new_edges(self, edge_list, attributes=None, check_duplicates=False,
                  check_self_loops=True, check_existing=True,
                  ignore_invalid=False):
        '''
        Add a list of edges to the graph.

        .. versionchanged:: 2.0
            Can perform all possible checks before adding new edges via the
            ``check_duplicates`` ``check_self_loops``, and ``check_existing``
            arguments.

        Parameters
        ----------
        edge_list : list of 2-tuples or np.array of shape (edge_nb, 2)
            List of the edges that should be added as tuples (source, target)
        attributes : :class:`dict`, optional (default: ``{}``)
            Dictionary containing optional edge properties. If the graph is
            weighted, defaults to ``{"weight": ones}``, where ``ones`` is an
            array the same length as the `edge_list` containing a unit weight
            for each connection (synaptic strength in NEST).
        check_duplicates : bool, optional (default: False)
            Check for duplicate edges within `edge_list`.
        check_self_loops : bool, optional (default: True)
            Check for self-loops.
        check_existing : bool, optional (default: True)
            Check whether some of the edges in `edge_list` already exist in the
            graph or exist multiple times in `edge_list` (also performs
            `check_duplicates`).
        ignore_invalid : bool, optional (default: False)
            Ignore invalid edges: they are not added to the graph and are
            silently dropped. Unless this is set to true, an error is raised
            whenever one of the three checks fails.

        .. warning::

            Setting `check_existing` to False will lead to undefined behavior
            if existing edges are provided! Only use it (for speedup) if you
            are sure that you are indeed only adding new edges.

        Returns
        -------
        Returns new edges only.
        '''
        g = self._graph

        attributes = {} if attributes is None else deepcopy(attributes)
        num_edges  = len(edge_list)

        # check that all nodes exist
        if np.max(edge_list) >= g.number_of_nodes():
            raise InvalidArgument("Some nodes do no exist.")

        for attr in attributes:
            if "_corr" in attr:
                raise NotImplementedError("Correlated attributes are not "
                                          "available with networkx.")

        # set default values for attributes that were not passed
        _set_default_edge_attributes(self, attributes, num_edges)

        # check edges
        new_attr = None

        if check_duplicates or check_self_loops or check_existing:
            edge_list, new_attr = _cleanup_edges(
                self, edge_list, attributes, check_duplicates,
                check_self_loops, check_existing, ignore_invalid)
        else:
            new_attr = attributes

        # create the edges
        initial_edges = g.number_of_edges()

        num_added = len(edge_list)

        if num_added:
            arr_edges = np.zeros((num_added, 3), dtype=int)

            arr_edges[:, :2] = edge_list
            arr_edges[:, 2]  = np.arange(initial_edges,
                initial_edges + num_added)

            # create the edges with an eid attribute
            g.add_weighted_edges_from(arr_edges, weight="eid")

            # check distance
            _set_dist_new_edges(new_attr, self, edge_list)

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

    def get_degrees(self, mode="total", nodes=None, weights=None):
        g = self._graph
        w = _get_nx_weights(self, weights)

        nodes  = range(g.number_of_nodes()) if nodes is None else nodes
        dtype  = int if weights in {False, None} else float
        di_deg = None

        if mode == 'total' or not self._graph.is_directed():
            di_deg = g.degree(nodes, weight=w)
        elif mode == 'in':
            di_deg = g.in_degree(nodes, weight=w)
        elif mode == 'out':
            di_deg = g.out_degree(nodes, weight=w)
        else:
            raise ValueError("Unknown `mode` '{}'".format(mode))

        return np.array([di_deg[i] for i in nodes], dtype=dtype)

    def is_connected(self, mode="strong"):
        '''
        Return whether the graph is connected.

        Parameters
        ----------
        mode : str, optional (default: "strong")
            Whether to test connectedness with directed ("strong") or
            undirected ("weak") connections.
        '''
        g = self._graph

        if g.is_directed() and mode == "weak":
            g = g.to_undirected(as_view=True)

        try:
            import networkx as nx
            nx.diameter(g)
            return True
        except nx.exception.NetworkXError:
            return False

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
        neighbours : set
            The neighbours of `node`.
        '''
        g = self._graph

        # special case for undirected
        if not g.is_directed():
            return set(g.neighbors(node))

        if mode == "all":
            # for directed graphs, neighbors ~ successors
            return set(g.successors(node)).union(g.predecessors(node))
        elif mode == "in":
            return set(g.predecessors(node))
        elif mode == "out":
            return set(g.successors(node))

        raise ArgumentError('Invalid `mode` argument {}; possible values are '
                            '"all", "out" or "in".'.format(mode))

    def _from_library_graph(self, graph, copy=True):
        ''' Initialize `self._graph` from existing library object. '''
        import networkx as nx

        nodes = {n: i for i, n in enumerate(graph)}

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        # check if nodes start from 0 and are continuous
        if set(nodes.keys()) != set(range(num_nodes)):
            # forced copy to restore nodes to [0, num_nodes[
            g = None

            if graph.is_directed():
                g = nx.DiGraph()
            else:
                g = nx.Graph()

            # add nodes
            for i, (n, attr) in enumerate(graph.nodes(data=True)):
                attr["id"] = n
                g.add_node(i, **attr)

            # add edges
            [g.add_edge(nodes[u], nodes[v], **attr)
             for u, v, attr in graph.edges(data=True)]

            # make edges ids
            def set_eid(e, eid):
                g.edges[e]["eid"] = eid

            [set_eid(e, i) for i, e in enumerate(g.edges)]

            graph = g

            self._graph = g
        else:
            # all good
            self._graph = graph.copy() if copy else graph

        # get attributes names and "types" and initialize them
        if num_nodes:
            for key, val in graph.nodes[0].items():
                super(type(self._nattr), self._nattr).__setitem__(
                    key, _get_dtype(val))

        if num_edges:
            e0 = next(iter(graph.edges))
            for key, val in graph.edges[e0].items():
                super(type(self._eattr), self._eattr).__setitem__(
                    key, _get_dtype(val))


# tool function to generate the edges_array

def _gen_edges(array, edata):
    source, target, eid = edata
    array[eid] = (source, target)

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
from nngt.lib import InvalidArgument, BWEIGHT, nonstring_container, is_integer
from nngt.lib.connect_tools import (_cleanup_edges, _set_dist_new_edges,
                                    _set_default_edge_attributes)
from nngt.lib.converters import _to_np_array
from nngt.lib.graph_helpers import _get_dtype, _get_gt_weights
from nngt.lib.logger import _log_message
from .graph_interface import GraphInterface, BaseProperty


logger = logging.getLogger(__name__)


# ---------- #
# Properties #
# ---------- #

class _GtNProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __getitem__(self, name):
        dtype = super(_GtNProperty, self).__getitem__(name)

        g = self.parent()._graph

        if dtype == "string":
            return g.vertex_properties[name].get_2d_array([0])[0]
        elif dtype == "object":
            vprop = g.vertex_properties[name]
            return _to_np_array(
                [vprop[i] for i in range(g.num_vertices())], dtype)

        return _to_np_array(g.vertex_properties[name].a, dtype)

    def __setitem__(self, name, value):
        g = self.parent()._graph

        if name in self:
            size = g.num_vertices()
            if len(value) == size:
                g.vertex_properties[name].a = np.array(value)
            else:
                raise ValueError("A list or a np.array with one entry per "
                                 "node in the graph is required")
        else:
            raise InvalidArgument("Attribute does not exist yet, use "
                                  "set_attribute to create it.")

    def new_attribute(self, name, value_type, values=None, val=None):
        dtype = object
        g     = self.parent()._graph

        if val is None:
            if value_type == "int":
                val = int(0)
                dtype = int
            elif value_type == "double":
                val = np.NaN
                dtype = float
            elif value_type == "string":
                val = ""
            else:
                val = None

        if values is None:
            values = _to_np_array([deepcopy(val)
                                   for _ in range(g.num_vertices())], dtype)

        if len(values) != g.num_vertices():
            raise ValueError("A list or a np.array with one entry per "
                             "node in the graph is required")

        # store name and value type in the dict
        super(_GtNProperty, self).__setitem__(name, value_type)

        # store the real values in the attribute
        nprop = g.new_vertex_property(value_type, vals=values)
        g.vertex_properties[name] = nprop
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

        num_nodes = g.num_vertices()
        num_n = len(nodes) if nodes is not None else num_nodes

        if num_n == num_nodes:
            self[name] = values
            self._num_values_set[name] = num_nodes
        elif num_n:
            if num_n != len(values):
                raise ValueError("`nodes` and `values` must have the same "
                                 "size; got respectively " + str(num_n) + \
                                 " and " + str(len(values)) + " entries.")

            non_obj = (super(_GtNProperty, self).__getitem__(name)
                       not in ('string', 'object'))

            if self._num_values_set[name] == num_nodes - num_n and non_obj:
                g.vertex_properties[name].a[-num_n:] = values
            else:
                for n, val in zip(nodes, values):
                    g.vertex_properties[name][n] = val
        if num_n:
            self._num_values_set[name] = num_nodes


class _GtEProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __getitem__(self, name):
        '''
        Return the attributes of an edge or a list of edges.
        '''
        g = self.parent()._graph

        Edge = g.edge

        if isinstance(name, slice):
            eprop = {}
            for k in self:
                eprop[k] = g.edge_properties[k].a[name]

            return eprop
        elif nonstring_container(name):
            # name is and edge or contains edges
            if len(name) == 0:
                return {k: [] for k in self}

            eprop = {}
            if nonstring_container(name[0]):
                # name is a list of edges
                eids = [g.edge_index[Edge(*e)] for e in name]
                for k in self.keys():
                    dtype = super(_GtEProperty, self).__getitem__(k)
                    if dtype == "string":
                        eprop[k] = \
                            g.edge_properties[k].get_2d_array([0])[0][eids]
                    elif dtype == "object":
                        tmp = g.edge_properties[k]
                        eprop[k] = _to_np_array(
                            [tmp[Edge(*e)] for e in name], dtype)
                    else:
                        eprop[k] = g.edge_properties[k].a[eids]
            else:
                # name is a single edge
                for k in self.keys():
                    eprop[k] = g.edge_properties[k][Edge(*name)]

            return eprop

        dtype = super(_GtEProperty, self).__getitem__(name)

        if dtype == "string":
            return g.edge_properties[name].get_2d_array([0])[0]
        elif dtype == "object":
            tmp   = g.edge_properties[name]
            edges = g.get_edges()
            return _to_np_array([tmp[Edge(*e)] for e in edges], dtype)

        return _to_np_array(g.edge_properties[name].a, dtype)

    def __setitem__(self, name, value):
        g = self.parent()._graph

        if name in self:
            size = g.num_edges()

            if size:
                dtype = super(_GtEProperty, self).__getitem__(name)

                # check for list value for one edge
                if dtype == "object" and size == 1 and isinstance(value, list):
                    value = [value]

                if len(value) == size:
                    g.edge_properties[name].a = _to_np_array(value, dtype)
                else:
                    raise ValueError("A list or a np.array with one entry per "
                                 "edge in the graph is required")
        else:
            raise InvalidArgument("Attribute does not exist yet, use "
                                  "set_attribute to create it.")

        self._num_values_set[name] = len(value)

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

        num_edges = g.num_edges()
        num_e     = len(edges) if edges is not None else num_edges

        if num_e == num_edges:
            self[name] = values
            self._num_values_set[name] = num_edges
        elif num_e:
            if num_e != len(values):
                raise ValueError("`edges` and `values` must have the same "
                                 "size; got respectively " + str(num_e) + \
                                 " and " + str(len(values)) + " entries.")
            if self._num_values_set[name] == num_edges - num_e:
                g.edge_properties[name].a[-num_e:] = values
                self._num_values_set[name] = num_edges
            else:
                for e, val in zip(edges, values):
                    gt_e = g.edge(*e)
                    g.edge_properties[name][gt_e] = val
                self._num_values_set[name] += num_e

    def new_attribute(self, name, value_type, values=None, val=None):
        g = self.parent()._graph

        num_edges = g.num_edges()

        if values is None and val is None:
            self._num_values_set[name] = num_edges

        if val is None:
            if value_type == "int":
                val = int(0)
            elif value_type == "double":
                val = np.NaN
            elif value_type == "string":
                val = ""
            else:
                val = None

        if values is None:
            values = _to_np_array(
                [deepcopy(val) for _ in range(num_edges)],
                value_type)

        if len(values) != num_edges:
            self._num_values_set[name] = 0
            raise ValueError("A list or a np.array with one entry per "
                             "edge in the graph is required")

        # store name and value type in the dict
        super(_GtEProperty, self).__setitem__(name, value_type)

        # store the real values in the attribute
        eprop = g.new_edge_property(value_type, vals=values)
        g.edge_properties[name] = eprop
        self._num_values_set[name] = len(values)


# ----- #
# Graph #
# ----- #

class _GtGraph(GraphInterface):

    '''
    Subclass of :class:`gt.Graph` that (with
    :class:`~nngt.core._SnapGraph`) unifies the methods to work with either
    `graph-tool` or `SNAP`.
    '''

    _nattr_class = _GtNProperty
    _eattr_class = _GtEProperty

    #-------------------------------------------------------------------------#
    # Constructor and instance properties

    def __init__(self, nodes=0, copy_graph=None, weighted=True, directed=True,
                 **kwargs):
        '''
        @todo: document that
        see :class:`gt.Graph`'s constructor '''
        self._nattr = _GtNProperty(self)
        self._eattr = _GtEProperty(self)

        g = copy_graph.graph if copy_graph is not None else None

        if g is not None:
            from graph_tool.stats import remove_parallel_edges
            if not directed and g.is_directed():
                g = g.copy()
                g.set_directed(False)
                remove_parallel_edges(g)
            elif directed and not g.is_directed():
                g.set_directed(True)

            self._from_library_graph(g, copy=True)
        else:
            self._graph = nngt._config["graph"](directed=directed)

            if nodes:
                self._graph.add_vertex(nodes)

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

        if nonstring_container(edge) and len(edge):
            if is_integer(edge[0]):
                return g.edge_index[g.edge(*edge)]
            elif nonstring_container(edge[0]):
                idx = [g.edge_index[g.edge(*e)] for e in edge]
                return idx

        raise AttributeError("`edge` must be either a 2-tuple of ints or "
                             "an array of 2-tuples of ints.")

    def has_edge(self, edge):
        '''
        Whether `edge` is present in the graph.

        .. versionadded:: 2.0
        '''
        try:
            e = self._graph.edge(*edge)

            if e is None:
                return False

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
        edges = g.get_edges([g.edge_index])
        order = np.argsort(edges[:, 2])

        return edges[order, :2]

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
        attributes : dict, optional (default: None)
            Dictionary containing the attributes of the nodes.
        value_types : dict, optional (default: None)
            Dict of the `attributes` types, necessary only if the `attributes`
            do not exist yet.

        Returns
        -------
        The node or a list of the nodes created.
        '''
        nodes = self._graph.add_vertex(n)
        nodes = [int(nodes)] if n == 1 else [int(node) for node in nodes]

        attributes = {} if attributes is None else deepcopy(attributes)

        if attributes:
            for k, v in attributes.items():
                v = deepcopy(v)
                if k not in self._nattr:
                    self._nattr.new_attribute(k, value_types[k], val=v)
                else:
                    v = v if nonstring_container(v) else [v]
                    self._nattr.set_attribute(k, v, nodes=nodes)

        # set default values for double attributes that were not set
        # (others are properly handled automatically)
        for k in self.node_attributes:
            if k not in attributes and self.get_attribute_type(k) == "double":
                self.set_node_attribute(k, nodes=nodes, val=np.NaN)

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

        _set_default_edge_attributes(self, attributes, num_edges=1)

        # check that the edge does not already exist and that nodes are valid
        try:
            edge = g.edge(source, target)
        except ValueError:
            raise InvalidArgument("`source` or `target` does not exist.")

        if edge is None:
            if source == target:
                if not ignore and not self_loop:
                    raise InvalidArgument("Trying to add a self-loop.")
                elif ignore:
                    _log_message(logger, "WARNING",
                                 "Self-loop on {} ignored.".format(source))

                    return None

            g.add_edge(source, target, add_missing=False)

            # check distance
            _set_dist_new_edges(attributes, self, [(source, target)])

            # set the attributes
            self._attr_new_edges([(source, target)], attributes=attributes)
        else:
            if not ignore:
                raise InvalidArgument("Trying to add existing edge.")

            _log_message(logger, "WARNING",
                         "Existing edge {} ignored.".format((source, target)))

            return None

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
        attributes = {} if attributes is None else deepcopy(attributes)
        num_edges  = len(edge_list)
        num_nodes  = self.node_nb()

        # check that all nodes exist
        if np.max(edge_list) >= num_nodes:
            raise InvalidArgument("Some nodes do no exist.")

        # set default values for attributes that were not passed
        _set_default_edge_attributes(self, attributes, num_edges)

        # check edges
        new_attr = None

        if check_duplicates or check_self_loops or check_existing:
            edge_list, new_attr = _cleanup_edges(
                self, edge_list, attributes, check_duplicates,
                check_self_loops, check_existing, ignore_invalid)
        else:
            edge_list = np.asarray(edge_list)
            new_attr  = attributes

        # check distance
        _set_dist_new_edges(new_attr, self, edge_list)

        # create the edges
        if len(edge_list):
            self._graph.add_edge_list(edge_list)

            # call parent function to set the attributes
            self._attr_new_edges(edge_list, attributes=new_attr)

        return edge_list

    def clear_all_edges(self):
        ''' Remove all edges from the graph '''
        self._graph.clear_edges()
        self._eattr.clear()

    #-------------------------------------------------------------------------#
    # Getters

    def node_nb(self):
        ''' Number of nodes in the graph '''
        return self._graph.num_vertices()

    def edge_nb(self):
        ''' Number of edges in the graph '''
        return self._graph.num_edges()

    def get_degrees(self, mode="total", nodes=None, weights=None):
        w = _get_gt_weights(self, weights)

        if not self._graph.is_directed():
            mode = "total"

        if nodes is not None:
            return self._graph.degree_property_map(
                mode, weight=w).a[nodes]

        return self._graph.degree_property_map(mode, weight=w).a.flatten()

    def is_connected(self, mode="strong"):
        '''
        Return whether the graph is connected.

        Parameters
        ----------
        mode : str, optional (default: "strong")
            Whether to test connectedness with directed ("strong") or
            undirected ("weak") connections.
        '''
        from graph_tool.topology import label_components

        directed  = True if mode == "strong" else False
        directed *= self._graph.is_directed()

        _, hist = label_components(self._graph, directed=directed)

        return len(hist) == 1

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
        v = g.vertex(node)

        if mode == "all" or not g.is_directed():
            return set(int(n) for n in v.all_neighbours())
        elif mode == "in":
            return set(int(n) for n in v.in_neighbours())
        elif mode == "out":
            return set(int(n) for n in v.out_neighbours())
        else:
            raise ArgumentError('''Invalid `mode` argument {}; possible values
                                are "all", "out" or "in".'''.format(mode))

    def _from_library_graph(self, graph, copy=True):
        ''' Initialize `self._graph` from existing library object. '''
        nodes = graph.num_vertices()
        edges = graph.num_edges()

        if copy:
            self._graph = nngt._config["graph"](g=graph, directed=True)
        else:
            self._graph = graph

        # get attributes names and "types" and initialize them
        if nodes:
            for key, val in graph.vertex_properties.items():
                try:
                    super(type(self._nattr), self._nattr).__setitem__(
                        key, _get_dtype(val.a[0]))
                except:
                    pass

        if edges:
            for key, val in graph.edge_properties.items():
                if val.value_type() == 'string':
                    super(type(self._eattr), self._eattr).__setitem__(
                        key, 'string')
                elif val.value_type().startswith('vector'):
                    super(type(self._eattr), self._eattr).__setitem__(
                        key, 'object')
                else:
                    super(type(self._eattr), self._eattr).__setitem__(
                        key, _get_dtype(val.a[0]))

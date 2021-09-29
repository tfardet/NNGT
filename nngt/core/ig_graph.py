#-*- coding:utf-8 -*-
#
# core/ig_graph.py
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

""" iGraph subclassing """

from collections import OrderedDict
from copy import deepcopy
import logging

import numpy as np
import scipy.sparse as ssp

import nngt
from nngt.lib import InvalidArgument, nonstring_container, BWEIGHT, is_integer
from nngt.lib.connect_tools import (_cleanup_edges, _set_dist_new_edges,
                                    _set_default_edge_attributes)
from nngt.lib.graph_helpers import (_get_dtype, _get_ig_weights,
                                    _post_del_update)
from nngt.lib.converters import _np_dtype, _to_np_array
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
            if value_type in ("int", "integer"):
                val = int(0)
            elif value_type in ("double", "float"):
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

    def get_eattr(self, edges, name=None):
        g = self.parent()._graph

        if nonstring_container(edges[0]):
            # many edges
            eids = [g.get_eid(*e) for e in edges]

            if name is None:
                eprop = {}

                if nonstring_container(name[0]):
                    for k in self.keys():
                        dtype = _np_dtype(super().__getitem__(k))
                        eprop[k] = _to_np_array(
                            [g.es[eid][k] for eid in eids], dtype=dtype)

            dtype = _np_dtype(super().__getitem__(name))
            return _to_np_array([g.es[eid][name] for eid in eids], dtype=dtype)
        elif not nonstring_container(edges):
            raise ValueError("Invalid `edges` entry: {}.".format(edges))

        # single edge
        eid = g.get_eid(*edges)

        if name is None:
            eprop = {}

            for k in self.keys():
                eprop[k] = g.es[eid][k]

            return eprop

        return g.es[eid][name]

    def new_attribute(self, name, value_type, values=None, val=None):
        g = self.parent()._graph

        if val is None:
            if value_type in ("int", "integer"):
                val = int(0)
            elif value_type in ("double", "float"):
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

    def set_attribute(self, name, values, edges=None, last_edges=False):
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

        if num_e != len(values):
            raise ValueError(
                "`edges` and `values` must have the same  size; got "
                "respectively " + str(num_e) + " and " + str(len(values))
                + " entries.")

        if edges is None:
            self[name] = values
        else:
            if last_edges:
                g.es[-num_e:][name] = values
            else:
                for e, val in zip(edges, values):
                    eid = g.get_eid(*e)
                    g.es[eid][name] = val

                self._num_values_set[name] += num_e

        if num_e:
            self._num_values_set[name] = num_edges


# ----- #
# Graph #
# ----- #

class _IGraph(GraphInterface):

    '''
    Container for :class:`igraph.Graph`.
    '''

    _nattr_class = _IgNProperty
    _eattr_class = _IgEProperty    

    #-------------------------------------------------------------------------#
    # Constructor and instance properties

    def __init__(self, nodes=0, copy_graph=None, directed=True, weighted=False,
                 **kwargs):
        self._nattr = _IgNProperty(self)
        self._eattr = _IgEProperty(self)

        g = copy_graph.graph if copy_graph is not None else None

        if g is None:
            self._graph = nngt._config["graph"](n=nodes, directed=directed)
        else:
            # convert graph if necessary
            if directed and not g.is_directed():
                g = g.copy()
                g.to_directed()
            elif not directed and g.is_directed():
                g = g.as_undirected(mode="collapse", combine_edges="sum")
                g.simplify(combine_edges="sum")

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
        return np.array([e.tuple for e in g.es], dtype=int)

    def _get_edges(self, source_node=None, target_node=None):
        '''
        Called by Graph.get_edges if source_node and target_node are not both
        integers.
        '''
        g = self._graph

        edges = None

        if source_node is None:
            if target_node is None:
                edges = g.es
            elif is_integer(target_node):
                edges = g.es.select(_target_eq=target_node)
            else:
                edges = g.es.select(_target_in=target_node)
        elif is_integer(source_node):
            if target_node is None:
                edges = g.es.select(_source_eq=source_node)
            else:
                edges = g.es.select(_source_eq=source_node,
                                    _target_in=target_node)
        else:
            if target_node is None:
                edges = g.es.select(_source_in=source_node)
            elif is_integer(target_node):
                edges = g.es.select(_source_in=source_node,
                                    _target_eq=target_node)
            else:
                edges = g.es.select(_source_in=source_node,
                                    _target_in=target_node)

        return [e.tuple for e in edges]

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
        positions : array of shape (n, 2), optional (default: None)
            Positions of the neurons. Valid only for
            :class:`~nngt.SpatialGraph` or :class:`~nngt.SpatialNetwork`.
        groups : str, int, or list, optional (default: None)
            :class:`~nngt.core.NeuralGroup` to which the neurons belong. Valid
            only for :class:`~nngt.Network` or :class:`~nngt.SpatialNetwork`.

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
        for k in self.node_attributes:
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

    def delete_nodes(self, nodes):
        '''
        Remove nodes (and associated edges) from the graph.
        '''
        if nodes is None:
            self._graph.delete_vertices()
        else:
            self._graph.delete_vertices(nodes)

        for key in self._nattr:
            self._nattr._num_values_set[key] = self.node_nb()

        for key in self._eattr:
            self._eattr._num_values_set[key] = self.edge_nb()

        # check spatial and structure properties
        _post_del_update(self, nodes)

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
        attributes = {} if attributes is None \
                     else {k: [v] for k, v in attributes.items()}

        if source == target:
            if not ignore and not self_loop:
                raise InvalidArgument("Trying to add a self-loop.")
            elif ignore:
                _log_message(logger, "INFO",
                             "Self-loop on {} ignored.".format(source))

                return None

        return self.new_edges(((source, target),), attributes,
                              check_self_loops=(not ignore and not self_loop),
                              ignore_invalid=ignore)

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

        # check that all nodes exist
        if num_edges:
            if np.max(edge_list) >= self.node_nb():
                raise InvalidArgument("Some nodes do no exist.")

            for k, v in attributes.items():
                assert nonstring_container(v) and len(v) == num_edges, \
                    "One attribute per edge is required."

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

            self._graph.add_edges(edge_list)

            # check distance
            _set_dist_new_edges(new_attr, self, edge_list)

            # call parent function to set the attributes
            self._attr_new_edges(edge_list, attributes=new_attr)

            if len(edge_list) == 0:
                return None
            elif len(edge_list) == 1:
                return edge_list[0]

        return edge_list

    def delete_edges(self, edges):
        ''' Remove a list of edges '''
        if len(edges):
            if nonstring_container(edges[0]):
                if isinstance(edges[0], tuple):
                    self._graph.delete_edges(edges)
                else:
                    self._graph.delete_edges([tuple(e) for e in edges])
            else:
                self._graph.delete_edges([edges])

            for key in self._eattr:
                self._eattr._num_values_set[key] = self.edge_nb()

    def clear_all_edges(self):
        ''' Remove all edges from the graph '''
        self._graph.delete_edges()
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
        w = _get_ig_weights(self, weights)
    
        mode = 'all' if mode == 'total' else mode

        if nonstring_container(weights) or weights not in {False, None}:
            return np.array(g.strength(nodes, mode=mode, weights=w))

        return np.array(g.degree(nodes, mode=mode), dtype=int)

    def is_connected(self, mode="strong"):
        '''
        Return whether the graph is connected.

        Parameters
        ----------
        mode : str, optional (default: "strong")
            Whether to test connectedness with directed ("strong") or
            undirected ("weak") connections.
        '''
        return self._graph.is_connected(mode)

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

        if mode == "all":
            return set(n for n in g.neighbors(node, mode=3))
        elif mode == "in":
            return set(n for n in g.neighbors(node, mode=2))
        elif mode == "out":
            return set(n for n in g.neighbors(node, mode=1))

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
                if key != 'eid':
                    super(type(self._eattr), self._eattr).__setitem__(
                        key, _get_dtype(val))


def _get_weights(g, weights):
    if weights in g.edge_attributes:
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

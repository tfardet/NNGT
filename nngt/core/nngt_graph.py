#-*- coding:utf-8 -*-
#
# core/nngt_graph.py
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

""" Default (limited) graph if none of the graph libraries are available """

from collections import OrderedDict
from copy import deepcopy
import logging

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix

import nngt
from nngt.analysis.nngt_functions import _dfs
from nngt.lib import InvalidArgument, nonstring_container, is_integer
from nngt.lib.connect_tools import (_cleanup_edges, _set_dist_new_edges,
                                    _set_default_edge_attributes)
from nngt.lib.graph_helpers import _get_edge_attr, _get_dtype, _post_del_update
from nngt.lib.converters import _np_dtype, _to_np_array
from nngt.lib.logger import _log_message
from .graph_interface import GraphInterface, BaseProperty


logger = logging.getLogger(__name__)


# ---------- #
# Properties #
# ---------- #

class _NProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prop = OrderedDict()

    def __getitem__(self, name):
        dtype = _np_dtype(super().__getitem__(name))
        return _to_np_array(self.prop[name], dtype=dtype)

    def __setitem__(self, name, value):
        if name in self:
            size = self.parent().node_nb()
            if len(value) == size:
                self.prop[name] = list(value)
            else:
                raise ValueError("A list or a np.array with one entry per "
                                 "node in the graph is required")
        else:
            raise InvalidArgument("Attribute does not exist yet, use "
                                  "set_attribute to create it.")

    def new_attribute(self, name, value_type, values=None, val=None):
        dtype = object

        if val is None:
            if value_type in ("int", "integer"):
                val = int(0)
                dtype = int
            elif value_type in ("double", "float"):
                val = np.NaN
                dtype = float
            elif value_type == "string":
                val = ""
            else:
                val = None
                value_type = "object"

        if values is None:
            values = _to_np_array(
                [deepcopy(val) for _ in range(self.parent().node_nb())],
                value_type)

        if len(values) != self.parent().node_nb():
            raise ValueError("A list or a np.array with one entry per "
                             "node in the graph is required")

        # store name and value type in the dict
        super().__setitem__(name, value_type)

        # store the real values in the attribute
        self.prop[name] = list(values)
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
        num_nodes = self.parent().node_nb()
        num_n = len(nodes) if nodes is not None else num_nodes
        if num_n == num_nodes:
            self[name] = list(values)
            self._num_values_set[name] = num_nodes
        else:
            if num_n != len(values):
                raise ValueError("`nodes` and `values` must have the same "
                                 "size; got respectively " + str(num_n) + \
                                 " and " + str(len(values)) + " entries.")

            if self._num_values_set[name] == num_nodes - num_n:
                self.prop[name].extend(values)
            else:
                for n, val in zip(nodes, values):
                    self.prop[name][n] = val

        self._num_values_set[name] = num_nodes

    def remove(self, nodes):
        ''' Remove entries for a set of nodes '''
        for key in self:
            for n in reversed(sorted(nodes)):
                self.prop[key].pop(n)

            self._num_values_set[key] -= len(nodes)


class _EProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prop = OrderedDict()

    def __getitem__(self, name):
        '''
        Return the attributes of an edge or a list of edges.
        '''
        eprop = {}
        graph = self.parent()

        if isinstance(name, slice):
            for k in self.keys():
                dtype = _np_dtype(super().__getitem__(k))
                eprop[k] = _to_np_array(self.prop[k], dtype)[name]

            return eprop
        elif nonstring_container(name):
            if nonstring_container(name[0]):
                eids = [graph.edge_id(e) for e in name]

                for k in self.keys():
                    dtype = _np_dtype(super().__getitem__(k))
                    eprop[k] = _to_np_array(self.prop[k], dtype=dtype)[eids]
            else:
                eid = graph.edge_id(name)

                for k in self.keys():
                    eprop[k] = self.prop[k][eid]

            return eprop
            
        dtype = _np_dtype(super().__getitem__(name))

        return _to_np_array(self.prop[name], dtype=dtype)

    def __setitem__(self, name, value):
        if name in self:
            size = self.parent().edge_nb()
            if len(value) == size:
                self.prop[name] = list(value)
            else:
                raise ValueError("A list or a np.array with one entry per "
                                 "edge in the graph is required")
        else:
            raise InvalidArgument("Attribute does not exist yet, use "
                                  "set_attribute to create it.")
        self._num_values_set[name] = len(value)

    def get_eattr(self, edges, name=None):
        g = self.parent()

        eid = g.edge_id

        if nonstring_container(edges[0]):
            # many edges
            if name is None:
                eprop = {}

                for k in self.keys():
                    prop = self.prop[k]

                    dtype = super().__getitem__(k)

                    eprop[k] = _to_np_array(
                        [prop[eid(tuple(e))] for e in edges], dtype)

                return eprop

            prop = self.prop[name]

            dtype = super().__getitem__(name)

            return _to_np_array([prop[eid(tuple(e))] for e in edges], dtype)

        # single edge
        if name is None:
            eprop = {}
            for k in self.keys():
                eprop[k] = self.prop[k][eid(tuple(edges))]

            return eprop

        return self.prop[name][eid(tuple(edges))]

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
        num_edges = self.parent().edge_nb()
        num_e     = len(edges) if edges is not None else num_edges

        if num_e != len(values):
            raise ValueError("`edges` and `values` must have the same "
                             "size; got respectively " + str(num_e) + \
                             " and " + str(len(values)) + " entries.")

        if edges is None:
            self[name] = list(values)
        else:
            if last_edges:
                self.prop[name].extend(values)
            else:
                eid  = self.parent().edge_id
                prop = self.prop[name]

                # using list comprehension for fast loop
                [_set_prop(prop, eid(e), val) for e, val in zip(edges, values)]

        if num_e:
            self._num_values_set[name] = num_edges

    def new_attribute(self, name, value_type, values=None, val=None):
        num_edges = self.parent().edge_nb()

        if values is None and val is None:
            self._num_values_set[name] = num_edges

        if val is None:
            if value_type in ("int", "integer"):
                val = int(0)
            elif value_type in ("double", "float"):
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
        super().__setitem__(name, value_type)

        # store the real values in the attribute
        self.prop[name] = list(values)
        self._num_values_set[name] = len(values)

    def edges_deleted(self, eids):
        ''' Remove the attributes of a set of edge ids '''
        for key, val in self.prop.items():
            self.prop[key] = [v for i, v in enumerate(val) if i not in eids]
            self._num_values_set[key] -= len(eids)


# ----------------- #
# NNGT backup graph #
# ----------------- #

class _NNGTGraphObject:
    '''
    Minimal implementation of the GraphObject, which does not rely on any
    graph-library.
    '''

    def __init__(self, nodes=0, weighted=True, directed=True):
        ''' Initialized independent graph '''
        self._nodes    = set(i for i in range(nodes))
        self._out_deg  = [0]*nodes
        self._in_deg   = [0]*nodes

        if directed:
            # for directed networks, edges and unique are the same
            self._edges = self._unique = OrderedDict()
            assert self._edges is self._unique
        else:
            # for undirected networks
            self._edges  = OrderedDict()
            self._unique = OrderedDict()

        self._directed = directed
        self._weighted = weighted

    def copy(self):
        ''' Returns a deep copy of the graph object '''
        copy = _NNGTGraphObject(len(self._nodes), weighted=self._weighted,
                                directed=self._directed)

        copy._nodes   = self._nodes.copy()

        if self._directed:
            copy._unique = copy._edges = self._edges.copy()
            assert copy._unique is copy._edges
        else:
            copy._edges  = self._edges.copy()
            copy._unique = self._unique.copy()

        copy._out_deg = self._out_deg.copy()
        copy._in_deg  = self._in_deg.copy()

        return copy

    def is_directed(self):
        return self._directed

    @property
    def nodes(self):
        return list(self._nodes)


class _NNGTGraph(GraphInterface):
    ''' NNGT wrapper class for _NNGTGraphObject '''

    #------------------------------------------------------------------#
    # Constructor and instance properties

    def __init__(self, nodes=0, weighted=True, directed=True,
                 copy_graph=None, **kwargs):
        ''' Initialized independent graph '''
        self._nattr = _NProperty(self)
        self._eattr = _EProperty(self)

        self._max_eid = 0

        # test if copying graph
        if copy_graph is not None:
            self._from_library_graph(copy_graph, copy=True)
            self._max_eid = copy_graph._max_eid
        else:
            self._graph = _NNGTGraphObject(
                nodes=nodes, weighted=weighted, directed=directed)

    #------------------------------------------------------------------#
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
            return g._edges[tuple(edge)]
        elif nonstring_container(edge[0]):
            idx = [g._edges[tuple(e)] for e in edge]
            return idx
        else:
            raise AttributeError("`edge` must be either a 2-tuple of ints or "
                                 "an array of 2-tuples of ints.")

    def has_edge(self, edge):
        '''
        Whether `edge` is present in the graph.

        .. versionadded:: 2.0
        '''
        e = tuple(edge)

        return e in self._graph._edges

    @property
    def edges_array(self):
        '''
        Edges of the graph, sorted by order of creation, as an array of
        2-tuple.
        '''
        return np.asarray(list(self._graph._unique), dtype=int)

    def _get_edges(self, source_node=None, target_node=None):
        '''
        Called by Graph.get_edges if source_node and target_node are not both
        integers.
        '''
        g = self._graph

        if source_node is not None:
            source_node = \
                {source_node} if is_integer(source_node) else set(source_node)

            # source only
            if target_node is None:
                if g.is_directed():
                    return [e for e in g._unique if e[0] in source_node]

                return [e for e in g._unique
                        if e[0] in source_node or e[1] in source_node]

            # source and target
            target_node = \
                {target_node} if is_integer(target_node) else set(target_node)

            if g.is_directed():
                return [e for e in g._unique
                        if e[0] in source_node and e[1] in target_node]

            return [e for e in g._unique
                    if e[0] in source_node and e[1] in target_node
                    or e[1] in source_node and e[0] in target_node]

        if target_node is None:
            # return all edges
            return list(g._unique)

        # target only
        target_node = \
            {target_node} if is_integer(target_node) else set(target_node)

        if g.is_directed():
            return [e for e in g._unique if e[1] in target_node]

        return [e for e in g._unique
                if e[0] in target_node or e[1] in target_node]

    def is_connected(self, mode="strong"):
        '''
        Return whether the graph is connected.

        Parameters
        ----------
        mode : str, optional (default: "strong")
            Whether to test connectedness with directed ("strong") or
            undirected ("weak") connections.
        '''
        num_nodes = self.node_nb()

        # get adjacency matrix
        A = self.adjacency_matrix()

        if mode == "weak" and self.is_directed():
            A = A + A.T

        visited = _dfs(A, 0)

        if len(visited) != num_nodes:
            return False
        elif mode == "strong" and self.is_directed():
            visited = _dfs(A.T, 0)

            if len(visited) != num_nodes:
                return False

        return True

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
        The node or a tuple of the nodes created.
        '''
        nodes = []

        g = self._graph

        if n == 1:
            nodes.append(len(g._nodes))
            g._in_deg.append(0)
            g._out_deg.append(0)
        else:
            num_nodes = len(g._nodes)
            nodes.extend(
                [i for i in range(num_nodes, num_nodes + n)])
            g._in_deg.extend([0 for _ in range(n)])
            g._out_deg.extend([0 for _ in range(n)])

        g._nodes.update(nodes)

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
                if dtype == "double":
                    values = [np.NaN for _ in nodes]
                    self._nattr.set_attribute(k, values, nodes=nodes)
                elif dtype == "int":
                    values = [0 for _ in nodes]
                    self._nattr.set_attribute(k, values, nodes=nodes)
                elif dtype == "string":
                    values = ["" for _ in nodes]
                    self._nattr.set_attribute(k, values, nodes=nodes)
                else:
                    values = [None for _ in nodes]
                    self._nattr.set_attribute(k, values, nodes=nodes)

        if self.is_spatial():
            old_pos      = self._pos
            self._pos    = np.full((self.node_nb(), 2), np.NaN, dtype=float)
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
        g = self._graph

        old_nodes = range(self.node_nb())

        # update node set and degrees
        if nonstring_container(nodes):
            nodes = set(nodes)

            g._nodes = g._nodes.difference(nodes)
        else:
            g._nodes.remove(nodes)
            g._out_deg.pop(nodes)
            g._in_deg.pop(nodes)

            nodes = {nodes}

        # remove node attributes
        self._nattr.remove(nodes)

        # map from old nodes to new node ids
        idx = 0

        remapping = {}

        for n in old_nodes:
            if n not in nodes:
                remapping[n] = idx
                idx += 1

        # remove edges and remap edges
        remove_eids = set()

        new_edges  = OrderedDict()

        new_eid = 0

        for e, eid in g._unique.items():
            if e[0] in nodes or e[1] in nodes:
                remove_eids.add(eid)
            else:
                new_edges[(remapping[e[0]], remapping[e[1]])] = new_eid
                new_eid += 1

        g._unique = new_edges

        if not g._directed:
            g._edges = new_edges.copy()
            g._edges.update({e[::-1]: i for e, i in new_edges.items()})
        else:
            g._edges = g._unique

        # tell edge attributes
        self._eattr.edges_deleted(remove_eids)

        g._nodes = set(range(len(g._nodes)))

        # check spatial and structure properties
        _post_del_update(self, nodes, remapping=remapping)

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

        # set default values for attributes that were not passed
        _set_default_edge_attributes(self, attributes, num_edges=1)

        # check that the edge does not already exist
        edge = (source, target)

        if source not in g._nodes:
            raise InvalidArgument("There is no node {}.".format(source))

        if target not in g._nodes:
            raise InvalidArgument("There is no node {}.".format(target))

        if source == target:
            if not ignore and not self_loop:
                raise InvalidArgument("Trying to add a self-loop.")
            elif ignore:
                _log_message(logger, "INFO",
                             "Self-loop on {} ignored.".format(source))

                return None

        if (g._directed and edge not in g._unique) or edge not in g._edges:
            edge_id             = self._max_eid
            # ~ edge_id             = len(g._unique)
            g._unique[edge]     = edge_id
            g._out_deg[source] += 1
            g._in_deg[target]  += 1

            self._max_eid += 1

            # check distance
            _set_dist_new_edges(attributes, self, [edge])

            # attributes
            self._attr_new_edges([(source, target)], attributes=attributes)

            if not g._directed:
                # edges and unique are different objects, so update _edges
                g._edges[edge] = edge_id
                # add reciprocal
                e_recip             = (target, source)
                g._edges[e_recip]   = edge_id
                g._out_deg[target] += 1
                g._in_deg[source]  += 1
        else:
            if not ignore:
                raise InvalidArgument("Trying to add existing edge.")

            _log_message(logger, "INFO",
                         "Existing edge {} ignored.".format((source, target)))

        return edge

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

        g = self._graph

        # set default values for attributes that were not passed
        _set_default_edge_attributes(self, attributes, num_edges)

        # check that all nodes exist
        if num_edges:
            if np.max(edge_list) >= self.node_nb():
                raise InvalidArgument("Some nodes do no exist.")

            for k, v in attributes.items():
                assert nonstring_container(v) and len(v) == num_edges, \
                    "One attribute per edge is required."

            # check edges
            new_attr = None

            if check_duplicates or check_self_loops or check_existing:
                edge_list, new_attr = _cleanup_edges(
                    self, edge_list, attributes, check_duplicates,
                    check_self_loops, check_existing, ignore_invalid)
            else:
                new_attr = attributes

            # create the edges
            initial_eid = self._max_eid

            ws        = None
            num_added = len(edge_list)

            if "weight" in new_attr:
                if nonstring_container(new_attr["weight"]):
                    ws = new_attr["weight"]
                else:
                    ws = (new_attr["weight"] for _ in range(num_added))
            else:
                ws = _get_edge_attr(self, edge_list, "weight", last_edges=True)

            for i, (e, w) in enumerate(zip(edge_list, ws)):
                eid = self._max_eid

                g._unique[tuple(e)] = eid

                self._max_eid += 1

                g._out_deg[e[0]] += 1
                g._in_deg[e[1]]  += 1

                if not g._directed:
                    # edges and unique are different objects, so update _edges
                    g._edges[tuple(e)] = eid
                    # reciprocal edge
                    g._edges[tuple(e[::-1])] = eid

                    g._out_deg[e[1]] += 1
                    g._in_deg[e[0]]  += 1

            # check distance
            _set_dist_new_edges(new_attr, self, edge_list)

            # call parent function to set the attributes
            self._attr_new_edges(edge_list, attributes=new_attr)

        return edge_list

    def delete_edges(self, edges):
        ''' Remove a list of edges '''
        if len(edges):
            g = self._graph

            old_enum = len(g._unique)

            if not nonstring_container(edges[0]):
                edges = [edges]

            if not isinstance(edges[0], tuple):
                edges = [tuple(e) for e in edges]

            # get edge ids
            e_to_eid = g._unique

            eids = {e_to_eid[e] for e in edges}

            # remove
            directed = g._directed

            for e in edges:
                if e in g._unique:
                    del g._unique[e]

                if not directed:
                    if e in g._edges:
                        del g._edges[e]
                        del g._edges[e[::-1]]

                # update in and out degrees
                s, t = e

                g._in_deg[t] -= 1
                g._out_deg[s] -= 1

            # reset eids
            for i, e in enumerate(g._unique):
                g._unique[e] = i

                if not directed:
                    e._edges[e] = i
                    e._edges[e[::-1]] = i

            self._eattr.edges_deleted(eids)

    def clear_all_edges(self):
        g = self._graph

        if g._directed:
            g._edges = g._unique = OrderedDict()
            assert g._edges is g._unique
        else:
            g._edges  = OrderedDict()
            g._unique = OrderedDict()

        g._out_deg = [0 for _ in range(self.node_nb())]
        g._out_deg = [0 for _ in range(self.node_nb())]

        self._eattr.clear()

    #------------------------------------------------------------------#
    # Getters

    def node_nb(self):
        '''
        Returns the number of nodes.

        .. warning:: When using MPI, returns only the local number of nodes.
        '''
        return len(self._graph._nodes)

    def edge_nb(self):
        '''
        Returns the number of edges.

        .. warning:: When using MPI, returns only the local number of edges.
        '''
        return len(self._graph._unique)

    def is_directed(self):
        return g._directed

    def get_degrees(self, mode="total", nodes=None, weights=None):
        '''
        Returns the degree of the nodes.

        .. warning ::
            When using MPI, returns only the degree related to local edges.
        '''
        g = self._graph

        num_nodes = None
        weights   = 'weight' if weights is True else weights

        if nodes is None:
            num_nodes = self.node_nb()
            nodes = slice(num_nodes)
        elif nonstring_container(nodes):
            nodes = list(nodes)
            num_nodes = len(nodes)
        else:
            nodes = [nodes]
            num_nodes = 1

        # weighted
        if nonstring_container(weights) or weights in self._eattr:
            degrees = np.zeros(num_nodes)
            adj_mat = self.adjacency_matrix(types=False, weights=weights)

            if mode in ("in", "total") or not self.is_directed():
                degrees += adj_mat.sum(axis=0).A1[nodes]
            if mode in ("out", "total") and self.is_directed():
                degrees += adj_mat.sum(axis=1).A1[nodes]

            return degrees
        elif weights not in {None, False}:
            raise ValueError("Invalid `weights` {}".format(weights))

        # unweighted
        degrees = np.zeros(num_nodes, dtype=int)

        if not g._directed or mode in ("in", "total"):
            if isinstance(nodes, slice):
                degrees += g._in_deg[nodes]
            else:
                degrees += [g._in_deg[i] for i in nodes]

        if g._directed and mode in ("out", "total"):
            if isinstance(nodes, slice):
                degrees += g._out_deg[nodes]
            else:
                degrees += [g._out_deg[i] for i in nodes]

        if num_nodes == 1:
            return degrees[0]

        return degrees

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
        edges = self.edges_array

        if mode == "all" or not self._graph._directed:
            neighbours = set(edges[edges[:, 1] == node, 0])
            return neighbours.union(edges[edges[:, 0] == node, 1])

        if mode == "in":
            return set(edges[edges[:, 1] == node, 0])

        if mode == "out":
            return set(edges[edges[:, 0] == node, 1])

        raise ValueError(('Invalid `mode` argument {}; possible values'
                          'are "all", "out" or "in".').format(mode))

    def _from_library_graph(self, graph, copy=True):
        ''' Initialize `self._graph` from existing library object. '''
        self._graph = graph._graph.copy() if copy else graph._graph

        for key, val in graph._nattr.items():
            dtype = graph._nattr.value_type(key)
            self._nattr.new_attribute(key, dtype, values=val)

        for key, val in graph._eattr.items():
            dtype = graph._eattr.value_type(key)
            self._eattr.new_attribute(key, dtype, values=val)


# tool function to set edge properties

def _set_prop(array, eid, val):
    array[eid] = val

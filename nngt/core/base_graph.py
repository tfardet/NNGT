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

""" GraphObject for subclassing the libraries graphs """

from collections import defaultdict, OrderedDict
from copy import deepcopy
from abc import ABCMeta, abstractmethod, abstractproperty
from six import add_metaclass
from weakref import ref
import logging

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

import nngt
from nngt.lib import InvalidArgument, BWEIGHT, nonstring_container, is_integer
from nngt.lib.graph_helpers import _get_edge_attr, _get_syn_param, _to_np_array
from nngt.lib.io_tools import _np_dtype
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
            return super(BaseProperty, self).__getitem__(key)
        else:
            return {k: super(BaseProperty, self).__getitem__(k) for k in self}

    # redefine dict values/items to use the __getitem__ that will be
    # overwritten by the child classes

    def values(self):
        return [self[k] for k in self]

    def itervalues(self):
        return (self[k] for k in self)

    def items(self):
        return [(k, self[k]) for k in self]

    def iteritems(self):
        return ((k, self[k]) for k in self)


# -------------- #
# GraphInterface #
# -------------- #

@add_metaclass(ABCMeta)
class GraphInterface(nngt._config["graph"]):

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
        mat : :class:`scipy.sparse.csr_matrix` matrix
            The adjacency matrix of the graph.
        '''
        weights = "weight" if weights is True else weights
        mat = nngt.analyze_graph["adjacency"](self, weights)
        if types and 'type' in self.nodes_attributes:
            tarray = np.where(self.nodes_attributes['type'] < 0)[0]
            if np.any(tarray):
                mat[tarray] *= -1.
        elif types and 'type' in self.edges_attributes:
            raise NotImplementedError()
        return mat

    #-------------------------------------------------------------------------#
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
    def new_node(self, n=1, neuron_type=1, attributes=None):
        pass

    @abstractmethod
    def new_edge(self, source, target, weight=1.):
        pass

    @abstractmethod
    def new_edges(self, edge_list, attributes=None):
        pass

    def attr_new_edges(self, edge_list, attributes=None):
        num_edges = len(edge_list)
        if num_edges:
            attributes = {} if attributes is None else attributes
            specials = ("weight", "delay", 'distance')
            for k in attributes.keys():
                if k not in self.edges_attributes and k in specials:
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
                    "distance", values, edges=edge_list)
            # first check for potential syn_spec if Network
            if self.is_network():
                for syn_param in self.population.syn_spec.values():
                    bio_weights += ("weight" in syn_param)
                    bio_delays += ("delay" in syn_param)
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
                    "weight", values, edges=edge_list)
            # then delay
            if self.is_network() or "delay" in self.edges_attributes:
                prop = attributes.get("delay", None)
                values = _get_edge_attr(
                    self, edge_list, 'delay', prop, last_edges=True)
                self._eattr.set_attribute(
                    "delay", values, edges=edge_list)
            for k in attributes.keys():
                if k not in specials:
                    if k in self.edges_attributes:
                        values = _get_edge_attr(
                            self, edge_list, k, attributes[k], last_edges=True)
                        self._eattr.set_attribute(k, values, edges=edge_list)
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


# --------- #
# BaseGraph #
# --------- #

class BaseGraph(GraphInterface):
    '''
    Minimal implementation of the GraphObject, which does not rely on any
    graph-library.
    '''

    #-------------------------------------------------------------------------#
    # Constructor and instance properties

    def __init__(self, nodes=0, weighted=True, directed=True,
                 g=None, **kwargs):
        ''' Initialized independent graph '''
        self._nodes    = set()
        self._out_deg  = []
        self._in_deg   = []
        self._edges    = OrderedDict()
        self._adj_mat  = lil_matrix((nodes, nodes))
        super(BaseGraph, self).__init__()
        # test if copying graph
        if g is not None:
            # create nodes and node attributes
            self._directed = g.is_directed()
            self._weighted = g.is_weighted()
            self._edges    = OrderedDict()
            self._nattr    = deepcopy(g._nattr)
            self._eattr    = _EProperty(self)

            self.new_node(g.node_nb())

            self._adj_mat  = lil_matrix((g.node_nb(), g.node_nb()))
            # create edges and edge attributes
            attributes = g.get_edge_attributes()
            self.new_edges(g.edges_array, attributes=attributes,
                           check_edges=False)
        else:
            self._nattr    = _NProperty(self)
            self._eattr    = _EProperty(self)
            self._directed = directed
            self._weighted = weighted
            self.new_node(nodes)

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
            return self._edges[tuple(edge)]
        elif nonstring_container(edge[0]):
            idx = [self._edges[tuple(e)] for e in edge]
            return idx
        else:
            raise AttributeError("`edge` must be either a 2-tuple of ints or "
                                 "an array of 2-tuples of ints.")

    @property
    def edges_array(self):
        '''
        Edges of the graph, sorted by order of creation, as an array of
        2-tuple.
        '''
        return np.array(list(self._edges.keys()), dtype=int)

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

        if n == 1:
            nodes.append(len(self._nodes))
            self._in_deg.append(0)
            self._out_deg.append(0)
        else:
            num_nodes = len(self._nodes)
            nodes.extend(
                [i for i in range(num_nodes, num_nodes + n)])
            self._in_deg.extend([0 for _ in range(n)])
            self._out_deg.extend([0 for _ in range(n)])
        self._nodes.update(nodes)

        old_mat = self._adj_mat.tocoo()
        if self.edge_nb():
            tmp = coo_matrix((old_mat.data, (old_mat.row, old_mat.col)),
                             shape=(len(self._nodes), len(self._nodes)))
            self._adj_mat = tmp.tolil()
        else:
            self._adj_mat = lil_matrix((len(self._nodes), len(self._nodes)))

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
                else:
                    attributes[k] = [None]

        # check that the edge does not already exist
        edge = (source, target)

        if source not in self._nodes:
            raise ValueError("There is no node {}.".format(source))
        if target not in self._nodes:
            raise ValueError("There is no node {}.".format(target))

        if edge not in self._edges:
            edge_id                = len(self._edges)
            self._edges[edge]      = edge_id
            self._out_deg[source] += 1
            self._in_deg[target]  += 1
            # attributes
            self.attr_new_edges([(source, target)], attributes=attributes)
            # update matrix
            w = _get_edge_attr(self, [edge], "weight", last_edges=True)
            self._adj_mat[source, target] = w
            if not self._directed:
                e_recip                = (target, source)
                self._edges[e_recip]   = edge_id + 1
                self._out_deg[target] += 1
                self._in_deg[source]  += 1
                for k, v in attributes.items():
                    self.set_edge_attribute(k, val=v, edges=[e_recip])
                self._adj_mat[source, target] = w
        else:
            if not ignore:
                raise InvalidArgument("Trying to add existing edge.")
        return edge

    def new_edges(self, edge_list, attributes=None, check_edges=True):
        '''
        Add a list of edges to the graph.

        .. versionchanged:: 1.0
            new_edges checks for duplicate edges and self-loops

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
                elif dtype == "double":
                    if k != "weight":
                        attributes[k] = [np.NaN for _ in range(num_edges)]
                elif dtype == "int":
                    attributes[k] = [0 for _ in range(num_edges)]
                else:
                    attributes[k] = [None for _ in range(num_edges)]

        assert self._nodes.issuperset(np.ravel(edge_list)), \
            "Some nodes in `edge_list` do not exist in the network."

        initial_edges = self.edge_nb()
        new_attr      = None

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
        if not self._directed:
            recip_edges = edge_list[:,::-1]
            # slow but works
            unique = ~(recip_edges[..., np.newaxis]
                       == edge_list[..., np.newaxis].T).all(1).any(1)
            edge_list = np.concatenate((edge_list, recip_edges[unique]))
            for key, val in new_attr.items():
                new_attr[key] = np.concatenate((val, val[unique]))
        # create the edges
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
            self._edges[tuple(e)]     = initial_edges + i
            self._out_deg[e[0]]  += 1
            self._in_deg[e[1]]   += 1
            self._adj_mat[e[0], e[1]] = w
        # call parent function to set the attributes
        self.attr_new_edges(edge_list, attributes=new_attr)
        return edge_list

    def clear_all_edges(self):
        self._edges   = OrderedDict()
        self._out_deg = [0 for _ in range(self.node_nb())]
        self._out_deg = [0 for _ in range(self.node_nb())]
        self._adj_mat = lil_matrix((self.node_nb(), self.node_nb()))
        self._eattr.clear()

    #-------------------------------------------------------------------------#
    # Getters

    def node_nb(self):
        '''
        Returns the number of nodes.

        .. warning:: When using MPI, returns only the local number of nodes.
        '''
        return len(self._nodes)

    def edge_nb(self):
        '''
        Returns the number of edges.

        .. warning:: When using MPI, returns only the local number of edges.
        '''
        return len(self._edges)

    def degree_list(self, node_list=None, deg_type="total", use_weights=False):
        '''
        Returns the degree of the nodes.

        .. warning::
        When using MPI, returns only the degree related to local edges.
        '''
        if node_list is None:
            node_list = slice(self.node_nb())

        degrees = np.zeros(self.node_nb())

        if "weight" in self._eattr and use_weights:
            if not self._directed:
                degrees += self._adj_mat.sum(axis=1).A1[node_list]
            else:
                if deg_type in ("in", "total"):
                    degrees += self._adj_mat.sum(axis=0).A1[node_list]
                if deg_type in ("out", "total"):
                    degrees += self._adj_mat.sum(axis=1).A1[node_list]
        else:
            if not self._directed:
                degrees += self._in_deg[node_list]
            else:
                if deg_type in ("in", "total"):
                    degrees += self._in_deg[node_list]
                if deg_type in ("out", "total"):
                    degrees += self._out_deg[node_list]
        return degrees

    def betweenness_list(self, btype="both", use_weights=False, as_prop=False,
                         norm=True):
        raise NotImplementedError("BaseGraph does not support betweenness, "
                                  "install a graph library to use it.")

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
        neighbours = []
        if mode in ("in", "all"):
            neighbours.extend(self._adj_mat[node])
        elif mode in ("out", "all"):
            neighbours.extend(self._adj_mat[:, node])
        else:
            raise ValueError('''Invalid `mode` argument {}; possible values
                                are "all", "out" or "in".'''.format(mode))
        return list(set(neighbours))


class _NProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __init__(self, *args, **kwargs):
        super(_NProperty, self).__init__(*args, **kwargs)
        self.prop = OrderedDict()

    def __getitem__(self, name):
        dtype = _np_dtype(super(_NProperty, self).__getitem__(name))
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
                value_type = "object"

        if values is None:
            values = _to_np_array(
                [deepcopy(val) for _ in range(self.parent().node_nb())],
                value_type)

        if len(values) != self.parent().node_nb():
            raise ValueError("A list or a np.array with one entry per "
                             "node in the graph is required")

        # store name and value type in the dict
        super(_NProperty, self).__setitem__(name, value_type)
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


class _EProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __init__(self, *args, **kwargs):
        super(_EProperty, self).__init__(*args, **kwargs)
        self.prop = OrderedDict()

    def __getitem__(self, name):
        '''
        Return the attributes of an edge or a list of edges.
        '''
        eprop = {}
        if isinstance(name, slice):
            for k in self.keys():
                dtype = _np_dtype(super(_EProperty, self).__getitem__(k))
                eprop[k] = _to_np_array(self.prop[k], dtype)[name]
            return eprop
        elif nonstring_container(name):
            if nonstring_container(name[0]):
                eids = [self.parent().edge_id(e) for e in name]
                for k in self.keys():
                    dtype = _np_dtype(super(_EProperty, self).__getitem__(k))
                    eprop[k] = _to_np_array(self.prop[k], dtype=dtype)[eids]
            else:
                eid = self.parent().get_eid(*name)
                for k in self.keys():
                    eprop[k] = self.prop[k][eid]
            return eprop

        dtype = _np_dtype(super(_EProperty, self).__getitem__(name))

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
        num_edges = self.parent().edge_nb()
        num_e     = len(edges) if edges is not None else num_edges

        if num_e == num_edges:
            self[name] = list(values)
            self._num_values_set[name] = num_edges
        else:
            if num_e != len(values):
                raise ValueError("`edges` and `values` must have the same "
                                 "size; got respectively " + str(num_e) + \
                                 " and " + str(len(values)) + " entries.")
            if self._num_values_set[name] == num_edges - num_e:
                self.prop[name].extend(values)
                self._num_values_set[name] = num_edges
            else:
                for e, val in zip(edges, values):
                    idx = self.parent().edge_id(e)
                    self.prop[name][idx] = val

    def new_attribute(self, name, value_type, values=None, val=None):
        if values is None and val is None:
            self._num_values_set[name] = self.parent().edge_nb()
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
                [deepcopy(val) for _ in range(self.parent().edge_nb())],
                value_type)

        if len(values) != self.parent().edge_nb():
            self._num_values_set[name] = 0
            raise ValueError("A list or a np.array with one entry per "
                             "edge in the graph is required")
        # store name and value type in the dict
        super(_EProperty, self).__setitem__(name, value_type)
        # store the real values in the attribute
        self.prop[name] = list(values)
        self._num_values_set[name] = len(values)

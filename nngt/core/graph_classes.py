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

""" Graph classes for graph generation and management """

from copy import deepcopy
import logging
import weakref

import numpy as np
import scipy.sparse as ssp

import nngt
from nngt import save_to_file
import nngt.analysis as na
from nngt.lib import (InvalidArgument, nonstring_container, default_neuron,
                      default_synapse, POS, WEIGHT, DELAY, DIST, TYPE)
from nngt.lib.graph_helpers import _edge_prop
from nngt.lib.io_tools import _as_string, _load_from_file
from nngt.lib.logger import _log_message
from nngt.lib.test_functions import graph_tool_check, deprecated, is_integer

if nngt._config['with_nest']:
    from nngt.simulation import make_nest_network


__all__ = ['Graph', 'SpatialGraph', 'Network', 'SpatialNetwork']

logger = logging.getLogger(__name__)


# ----- #
# Graph #
# ----- #

class Graph(nngt.core.GraphObject):

    """
    The basic graph class, which inherits from a library class such as
    :class:`graph_tool.Graph`, :class:`networkx.DiGraph`, or ``igraph.Graph``.

    The objects provides several functions to easily access some basic
    properties.
    """

    #-------------------------------------------------------------------------#
    # Class properties

    __num_graphs = 0
    __max_id = 0

    @classmethod
    def num_graphs(cls):
        ''' Returns the number of alive instances. '''
        return cls.__num_graphs

    @classmethod
    def from_library(cls, library_graph, weighted=True, directed=True,
                     **kwargs):
        library_graph = nngt.core.GraphObject.to_graph_object(library_graph)
        library_graph.__class__ = cls
        if weighted:
            library_graph._w = _edge_prop(kwargs.get("weights", 1.))
        library_graph._d = _edge_prop(kwargs.get("delays", 1.))
        library_graph.__id = cls.__max_id
        library_graph._name = "Graph" + str(cls.__num_graphs)
        cls.__max_id += 1
        cls.__num_graphs += 1
        return library_graph

    @classmethod
    def from_matrix(cls, matrix, weighted=True, directed=True):
        '''
        Creates a :class:`~nngt.Graph` from a :mod:`scipy.sparse` matrix or
        a dense matrix.

        Parameters
        ----------
        matrix : :mod:`scipy.sparse` matrix or :class:`numpy.ndarray`
            Adjacency matrix.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.

        Returns
        -------
        :class:`~nngt.Graph`
        '''
        shape = matrix.shape
        graph_name = "FromYMatrix_Z"
        nodes = max(shape[0], shape[1])
        if issubclass(matrix.__class__, ssp.spmatrix):
            graph_name = graph_name.replace('Y', 'Sparse')
            if not directed:
                if shape[0] != shape[1] or not (matrix.T != matrix).nnz == 0:
                    raise InvalidArgument('Incompatible `directed=False` '
                                          'option provided for non symmetric '
                                          'matrix.')
        else:
            graph_name = graph_name.replace('Y', 'Dense')
            if not directed:
                if shape[0] != shape[1] or not (matrix.T == matrix).all():
                    raise InvalidArgument('Incompatible `directed=False` '
                                          'option provided for non symmetric '
                                          'matrix.')
        edges = np.array(matrix.nonzero()).T
        graph = cls(nodes, name=graph_name.replace("Z", str(cls.__num_graphs)),
                    weighted=weighted, directed=directed)
        weights = None
        if weighted:
            if issubclass(matrix.__class__, ssp.spmatrix):
                weights = np.array(matrix[edges[:, 0], edges[:, 1]])[0]
            else:
                weights = matrix[edges[:, 0], edges[:, 1]]
        graph.new_edges(edges, {"weight": weights}, check_edges=False)
        return graph

    @staticmethod
    @graph_tool_check('2.22')
    def from_file(filename, fmt="auto", separator=" ", secondary=";",
                  attributes=None, notifier="@", ignore="#",
                  from_string=False):
        '''
        Import a saved graph from a file.
        @todo: implement gml, dot, xml, gt

        Parameters
        ----------
        filename: str
            The path to the file.
        fmt : str, optional (default: "neighbour")
            The format used to save the graph. Supported formats are:
            "neighbour" (neighbour list, default if format cannot be deduced
            automatically), "ssp" (scipy.sparse), "edge_list" (list of all the
            edges in the graph, one edge per line, represented by a ``source
            target``-pair), "gml" (gml format, default if `filename` ends with
            '.gml'), "graphml" (graphml format, default if `filename` ends
            with '.graphml' or '.xml'), "dot" (dot format, default if
            `filename` ends with '.dot'), "gt" (only when using
            `graph_tool`<http://graph-tool.skewed.de/>_ as library, detected
            if `filename` ends with '.gt').
        separator : str, optional (default " ")
            separator used to separate inputs in the case of custom formats
            (namely "neighbour" and "edge_list")
        secondary : str, optional (default: ";")
            Secondary separator used to separate attributes in the case of
            custom formats.
        attributes : list, optional (default: [])
            List of names for the attributes present in the file. If a
            `notifier` is present in the file, names will be deduced from it;
            otherwise the attributes will be numbered.
            This argument can also be used to load only a subset of the saved
            attributes.
        notifier : str, optional (default: "@")
            Symbol specifying the following as meaningfull information.
            Relevant information is formatted ``@info_name=info_value``, where
            ``info_name`` is in ("attributes", "directed", "name", "size") and
            associated ``info_value`` are of type (``list``, ``bool``,
            ``str``, ``int``).
            Additional notifiers are ``@type=SpatialGraph/Network/
            SpatialNetwork``, which must be followed by the relevant notifiers
            among ``@shape``, ``@population``, and ``@graph``.
        from_string : bool, optional (default: False)
            Load from a string instead of a file.

        Returns
        -------
        graph : :class:`~nngt.Graph` or subclass
            Loaded graph.
        '''
        info, edges, nattr, eattr, pop, shape, pos = _load_from_file(
            filename=filename, fmt=fmt, separator=separator,
            secondary=secondary, attributes=attributes, notifier=notifier)
        # create the graph
        graph = Graph(nodes=info["size"], name=info["name"],
                      directed=info["directed"])
        # make the nodes attributes
        lst_attr, dtpes, lst_values = [], [], []
        if info["node_attributes"]:  # edge attributes to add to the graph
            lst_attr   = info["node_attributes"]
            dtpes      = info["node_attr_types"]
            lst_values = [nattr[name] for name in info["node_attributes"]]
        for nattr, dtype, values in zip(lst_attr, dtpes, lst_values):
            graph.new_node_attribute(nattr, dtype, values=values)
        # make the edges and their attributes
        lst_attr, dtpes, lst_values = [], [], []
        if info["edge_attributes"]:  # edge attributes to add to the graph
            lst_attr   = info["edge_attributes"]
            dtpes      = info["edge_attr_types"]
            lst_values = [eattr[name] for name in info["edge_attributes"]]
        graph.new_edges(edges, check_edges=False)
        for eattr, dtype, values in zip(lst_attr, dtpes, lst_values):
            graph.new_edge_attribute(eattr, dtype, values=values)
        if pop is not None:
            Network.make_network(graph, pop)
            pop._parent = weakref.ref(graph)
            for g in pop.values():
                g._pop = weakref.ref(pop)
                g._net = weakref.ref(graph)
        if pos is not None or shape is not None:
            SpatialGraph.make_spatial(graph, shape=shape, positions=pos)
        return graph

    @staticmethod
    def make_spatial(graph, shape=None, positions=None, copy=False):
        '''
        Turn a :class:`~nngt.Graph` object into a :class:`~nngt.SpatialGraph`,
        or a :class:`~nngt.Network` into a :class:`~nngt.SpatialNetwork`.

        Parameters
        ----------
        graph : :class:`~nngt.Graph` or :class:`~nngt.SpatialGraph`
            Graph to convert.
        shape : :class:`~nngt.geometry.Shape`, optional (default: None)
            Shape to associate to the new :class:`~nngt.SpatialGraph`.
        positions : (N, 2) array
            Positions, in a 2D space, of the N neurons.
        copy : bool, optional (default: ``False``)
            Whether the operation should be made in-place on the object or if a
            new object should be returned.

        Notes
        -----
        In-place operation that directly converts the original graph if `copy`
        is ``False``, else returns the copied :class:`~nngt.Graph` turned into
        a :class:`~nngt.SpatialGraph`.
        The `shape` argument can be skipped if `positions` are given; in that
        case, the neurons will be embedded in a rectangle that contains them
        all.
        '''
        if copy:
            graph = graph.copy()
        if isinstance(graph, Network):
            graph.__class__ = SpatialNetwork
        else:
            graph.__class__ = SpatialGraph
        graph._init_spatial_properties(shape, positions)
        if copy:
            return graph

    @staticmethod
    def make_network(graph, neural_pop, copy=False, **kwargs):
        '''
        Turn a :class:`~nngt.Graph` object into a :class:`~nngt.Network`, or a
        :class:`~nngt.SpatialGraph` into a :class:`~nngt.SpatialNetwork`.

        Parameters
        ----------
        graph : :class:`~nngt.Graph` or :class:`~nngt.SpatialGraph`
            Graph to convert
        neural_pop : :class:`~nngt.NeuralPop`
            Population to associate to the new :class:`~nngt.Network`
        copy : bool, optional (default: ``False``)
            Whether the operation should be made in-place on the object or if a
            new object should be returned.

        Notes
        -----
        In-place operation that directly converts the original graph if `copy`
        is ``False``, else returns the copied :class:`~nngt.Graph` turned into
        a :class:`~nngt.Network`.
        '''
        if copy:
            graph = graph.copy()
        if isinstance(graph, SpatialGraph):
            graph.__class__ = SpatialNetwork
        else:
            graph.__class__ = Network
        # set delays to 1. or to provided value if they are not already set
        if "delays" not in kwargs and not hasattr(graph, '_d'):
            graph._d = {"distribution": "constant", "value": 1.}
        elif "delays" in kwargs and not hasattr(graph, '_d'):
            graph._d = kwargs["delays"]
        elif "delays" in kwargs:
            _log_message(logger, "WARNING",
                         'Graph already had delays set, ignoring new ones.')
        graph._init_bioproperties(neural_pop)
        if copy:
            return graph

    #-------------------------------------------------------------------------#
    # Constructor/destructor and properties

    def __init__(self, nodes=0, name="Graph", weighted=True, directed=True,
                 from_graph=None, **kwargs):
        '''
        Initialize Graph instance

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`Graph` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        from_graph : :class:`~nngt.core.GraphObject`, optional
            An optional :class:`~nngt.core.GraphObject` to serve as base.
        kwargs : optional keywords arguments
            Optional arguments that can be passed to the graph, e.g. a dict
            containing information on the synaptic weights
            (``weights={"distribution": "constant", "value": 2.3}`` which is
            equivalent to ``weights=2.3``), the synaptic `delays`, or a
            ``type`` information.

        Returns
        -------
        self : :class:`~nngt.Graph`
        '''
        self.__id = self.__class__.__max_id
        self._name = name
        self._graph_type = kwargs["type"] if "type" in kwargs else "custom"
        # Init the core.GraphObject
        super(Graph, self).__init__(nodes=nodes, g=from_graph,
                                    directed=directed, weighted=weighted)
        # take care of the weights and delays
        if weighted:
            self.new_edge_attribute('weight', 'double')
            self._w = _edge_prop(kwargs.get("weights", None))
        if "delays" in kwargs:
            self.new_edge_attribute('delay', 'double')
            self._d = _edge_prop(kwargs.get("delays", None))
        if 'inh_weight_factor' in kwargs:
            self._iwf = kwargs['inh_weight_factor']
        # update the counters
        self.__class__.__num_graphs += 1
        self.__class__.__max_id += 1

    def __del__(self):
        self.__class__.__num_graphs -= 1

    def __repr__(self):
        ''' Provide unambiguous informations regarding the object. '''
        d = "directed" if self._directed else "undirected"
        w = "weighted" if self._weighted else "binary"
        t = self.type
        n = self.node_nb()
        e = self.edge_nb()
        return "<{directed}/{weighted} {obj} object of type '{net_type}' " \
               "with {nodes} nodes and {edges} edges at 0x{obj_id}>".format(
                    directed=d, weighted=w, obj=type(self).__name__,
                    net_type=t, nodes=n, edges=e, obj_id=id(self))

    def __str__(self):
        '''
        Return the full string description of the object as would be stored
        inside a file when saving the graph.
        '''
        return _as_string(self)

    @property
    def graph_id(self):
        ''' Unique :class:`int` identifying the instance. '''
        return self.__id

    @property
    def name(self):
        ''' Name of the graph. '''
        return self._name

    @property
    def type(self):
        ''' Type of the graph. '''
        return self._graph_type

    #-------------------------------------------------------------------------#
    # Graph actions

    def copy(self):
        '''
        Returns a deepcopy of the current :class:`~nngt.Graph`
        instance
        '''
        gc_instance = Graph(name=self._name + '_copy',
                            weighted=self._weighted,
                            from_graph=self)
        if self.is_spatial():
            SpatialGraph.make_spatial(gc_instance)
        if self.is_network():
            Network.make_network(gc_instance, deepcopy(self.population))
        return gc_instance

    @graph_tool_check('2.22')
    def to_file(self, filename, fmt="auto", separator=" ", secondary=";",
                attributes=None, notifier="@"):
        '''
        Save graph to file; options detailed below.

        .. seealso::
            :py:func:`nngt.lib.save_to_file` function for options.
        '''
        save_to_file(self, filename, fmt=fmt, separator=separator,
                     secondary=secondary, attributes=attributes,
                     notifier=notifier)

    #~ def inhibitory_subgraph(self):
        #~ ''' Create a :class:`~nngt.Graph` instance which graph
        #~ contains only the inhibitory edges of the current instance's
        #~ :class:`graph_tool.Graph` '''
        #~ eprop_b_type = self._graph.new_edge_property(
                       #~ "bool",-self._graph.edge_properties[TYPE].a+1)
        #~ self._graph.set_edge_filter(eprop_b_type)
        #~ inhib_graph = Graph( name=self._name + '_inhib',
                             #~ weighted=self._weighted,
                             #~ from_graph=core.GraphObject(self._graph,prune=True) )
        #~ self.clear_filters()
        #~ return inhib_graph

    #~ def excitatory_subgraph(self):
        #~ '''
        #~ Create a :class:`~nngt.Graph` instance which graph contains only the
        #~ excitatory edges of the current instance's :class:`core.GraphObject`.
        #~ .. warning ::
            #~ Only works for graph_tool
        #~ .. todo ::
            #~ Make this method library independant!
        #~ '''
        #~ eprop_b_type = self._graph.new_edge_property(
                       #~ "bool",self._graph.edge_properties[TYPE].a+1)
        #~ self._graph.set_edge_filter(eprop_b_type)
        #~ exc_graph = Graph( name=self._name + '_exc',
                             #~ weighted=self._weighted,
                             #~ graph=core.GraphObject(self._graph,prune=True) )
        #~ self._graph.clear_filters()
        #~ return exc_graph

    #-------------------------------------------------------------------------#
    # Setters

    def set_name(self, name=""):
        ''' set graph name '''
        if name != "":
            self._name = name
        else:
            self._name = "Graph_" + str(self.__id)

    def new_edge_attribute(self, name, value_type, values=None, val=None):
        '''
        Create a new attribute for the edges.

        .. versionadded:: 0.7

        Parameters
        ----------
        name : str
            The name of the new attribute.
        value_type : str
            Type of the attribute, among 'int', 'double', 'string', or 'object'
        values : array, optional (default: None)
            Values with which the edge attribute should be initialized.
            (must have one entry per node in the graph)
        val : int, float or str , optional (default: None)
            Identical value for all edges.
        '''
        self._eattr.new_attribute(name, value_type, values=values, val=val)

    def new_node_attribute(self, name, value_type, values=None, val=None):
        '''
        Create a new attribute for the nodes.

        .. versionadded:: 0.7

        Parameters
        ----------
        name : str
            The name of the new attribute.
        value_type : str
            Type of the attribute, among 'int', 'double', 'string', or 'object'
        values : array, optional (default: None)
            Values with which the node attribute should be initialized.
            (must have one entry per node in the graph)
        val : int, float or str , optional (default: None)
            Identical value for all nodes.
        
        See also
        --------
        :func:`~nngt.Graph.new_edge_attribute`,
        :func:`~nngt.Graph.set_node_attribute`,
        :func:`~nngt.Graph.get_node_attributes`,
        :func:`~nngt.Graph.set_edge_attribute`,
        :func:`~nngt.Graph.get_edge_attributes`
        '''
        self._nattr.new_attribute(name, value_type, values=values, val=val)

    def set_edge_attribute(self, attribute, values=None, val=None,
                           value_type=None, edges=None):
        '''
        Set attributes to the connections between neurons.

        .. warning ::
            The special "type" attribute cannot be modified when using graphs
            that inherit from the :class:`~nngt.Network` class. This is because
            for biological networks, neurons make only one kind of synapse,
            which is determined by the :class:`nngt.NeuralGroup` they
            belong to.

        Parameters
        ----------
        attribute : str
            The name of the attribute.
        value_type : str
            Type of the attribute, among 'int', 'double', 'string'
        values : array, optional (default: None)
            Values with which the edge attribute should be initialized.
            (must have one entry per node in the graph)
        val : int, float or str , optional (default: None)
            Identical value for all edges.
        value_type : str, optional (default: None)
            Type of the attribute, among 'int', 'double', 'string'. Only used
            if the attribute does not exist and must be created.
        edges : list of edges or array of shape (E, 2), optional (default: all)
            Edges whose attributes should be set. Others will remain unchanged.
        
        See also
        --------
        :func:`~nngt.Graph.set_node_attribute`,
        :func:`~nngt.Graph.get_edge_attributes`,
        :func:`~nngt.Graph.new_edge_attribute`,
        :func:`~nngt.Graph.new_node_attribute`,
        :func:`~nngt.Graph.get_node_attributes`
        '''
        if attribute not in self.edges_attributes:
            assert value_type is not None, "`value_type` is necessary for " +\
                                           "new attributes."
            self.new_edge_attribute(name=attribute, value_type=value_type,
                                    values=values, val=val)
        else:
            num_edges = self.edge_nb() if edges is None else len(edges)
            if values is None:
                if val is not None:
                    # preserve list and avoid conversion to array
                    # if isinstance(val, list):
                    #     values    = np.full(num_edges, None, dtype=object)
                    #     for k in range(num_edges):
                    #         values[k] = val.copy()
                    values = [deepcopy(val) for _ in range(num_edges)]
                else:
                    raise InvalidArgument("At least one of the `values` and "
                        "`val` arguments should not be ``None``.")
            self._eattr.set_attribute(attribute, values, edges=edges)

    def set_node_attribute(self, attribute, values=None, val=None,
                           value_type=None, nodes=None):
        '''
        Set attributes to the connections between neurons.

        .. versionadded:: 0.9

        Parameters
        ----------
        attribute : str
            The name of the attribute.
        value_type : str
            Type of the attribute, among 'int', 'double', 'string'
        values : array, optional (default: None)
            Values with which the edge attribute should be initialized.
            (must have one entry per node in the graph)
        val : int, float or str , optional (default: None)
            Identical value for all edges.
        value_type : str, optional (default: None)
            Type of the attribute, among 'int', 'double', 'string'. Only used
            if the attribute does not exist and must be created.
        nodes : list of nodes, optional (default: all)
            Nodes whose attributes should be set. Others will remain unchanged.

        See also
        --------
        :func:`~nngt.Graph.set_edge_attribute`,
        :func:`~nngt.Graph.new_node_attribute`,
        :func:`~nngt.Graph.get_node_attributes`,
        :func:`~nngt.Graph.new_edge_attribute`,
        :func:`~nngt.Graph.get_edge_attributes`,
        '''
        if attribute not in self.nodes_attributes:
            assert value_type is not None, "`value_type` is necessary for " +\
                                           "new attributes."
            self.new_node_attribute(name=attribute, value_type=value_type,
                                    values=values, val=val)
        else:
            num_nodes = self.node_nb() if nodes is None else len(nodes)
            if values is None:
                if val is not None:
                    values = [deepcopy(val) for _ in  range(num_nodes)]
                else:
                    raise InvalidArgument("At least one of the `values` and "
                        "`val` arguments should not be ``None``.")
            self._nattr.set_attribute(attribute, values, nodes=nodes)

    def set_weights(self, weight=None, elist=None, distribution=None,
                    parameters=None, noise_scale=None):
        '''
        Set the synaptic weights.

        Parameters
        ----------
        weight : float or class:`numpy.array`, optional (default: None)
            Value or list of the weights (for user defined weights).
        elist : class:`numpy.array`, optional (default: None)
            List of the edges (for user defined weights).
        distribution : class:`string`, optional (default: None)
            Type of distribution (choose among "constant", "uniform",
            "gaussian", "lognormal", "lin_corr", "log_corr").
        parameters : dict, optional (default: {})
            Dictionary containing the properties of the weight distribution.
            Properties are as follow for the distributions

            - 'constant': 'value'
            - 'uniform': 'lower', 'upper'
            - 'gaussian': 'avg', 'std'
            - 'lognormal': 'position', 'scale'

        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.
        
        Note
        ----
        If `distribution` and `parameters` are provided and the weights are set
        for the whole graph (`elist` is None), then the distribution properties
        will be kept as the new default for subsequent edges. That is, if new
        edges are created without specifying their weights, then these new
        weights will automatically be drawn from this previous distribution.
        '''
        if isinstance(weight, float):
            size = self.edge_nb() if elist is None else len(elist)
            self._w = {"distribution": "constant", "value": weight}
            weight = np.repeat(weight, size)
        elif not nonstring_container(weight) and weight is not None:
            raise AttributeError("Invalid `weight` value: must be either "
                                 "float, array-like or None.")
        elif weight is not None:
            self._w = {"distribution": "custom"}
        elif None not in (distribution, parameters) and elist is None:
            self._w = {"distribution": distribution}
            self._w.update(parameters)

        if distribution is None:
            distribution = self._w.get("distribution", None)
        if parameters is None:
            parameters = self._w
        nngt.core.Connections.weights(
            self, elist=elist, wlist=weight, distribution=distribution,
            parameters=parameters, noise_scale=noise_scale)

    def set_types(self, syn_type, nodes=None, fraction=None):
        '''
        Set the synaptic/connection types.

        .. warning ::
          The special "type" attribute cannot be modified when using graphs
          that inherit from the :class:`~nngt.Network` class. This is because
          for biological networks, neurons make only one kind of synapse,
          which is determined by the :class:`nngt.NeuralGroup` they
          belong to.

        Parameters
        ----------
        syn_type : int or string
            Type of the connection among 'excitatory' (also `1`) or
            'inhibitory' (also `-1`).
        nodes : int, float or list, optional (default: `None`)
            If `nodes` is an int, number of nodes of the required type that
            will be created in the graph (all connections from inhibitory nodes
            are inhibitory); if it is a float, ratio of `syn_type` nodes in the
            graph; if it is a list, ids of the `syn_type` nodes.
        fraction : float, optional (default: `None`)
            Fraction of the selected edges that will be set as `syn_type` (if
            `nodes` is not `None`, it is the fraction of the specified nodes'
            edges, otherwise it is the fraction of all edges in the graph).

        Returns
        -------
        t_list : :class:`numpy.ndarray`
            List of the types in an order that matches the `edges` attribute of
            the graph.
        '''
        inhib_nodes = nodes
        if syn_type == 'excitatory' or syn_type == 1:
            if is_integer(nodes):
                inhib_nodes = self.node_nb() - nodes
            elif isinstance(nodes, np.float):
                inhib_nodes = 1. / nodes
            elif nonstring_container(nodes):
                inhib_nodes = list(range(self.node_nb()))
                nodes.sort()
                for node in nodes[::-1]:
                    del inhib_nodes[node]
        return nngt.core.Connections.types(self, inhib_nodes, fraction)

    def set_delays(self, delay=None, elist=None, distribution=None,
                   parameters=None, noise_scale=None):
        '''
        Set the delay for spike propagation between neurons.

        Parameters
        ----------
        delay : float or class:`numpy.array`, optional (default: None)
            Value or list of delays (for user defined delays).
        elist : class:`numpy.array`, optional (default: None)
            List of the edges (for user defined delays).
        distribution : class:`string`, optional (default: None)
            Type of distribution (choose among "constant", "uniform",
            "gaussian", "lognormal", "lin_corr", "log_corr").
        parameters : dict, optional (default: {})
            Dictionary containing the properties of the delay distribution.
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the delays.
        '''
        # check special cases and set self._d
        if isinstance(delay, float):
            size = self.edge_nb() if elist is None else len(elist)
            self._d = {"distribution": "constant", "value": delay}
            delay = np.repeat(delay, size)
        elif not nonstring_container(delay) and delay is not None:
            raise AttributeError("Invalid `delay` value: must be either "
                                 "float, array-like or None")
        elif delay is not None:
            self._d = {"distribution": "custom"}
        elif None not in (distribution, parameters):
            self._d = {"distribution": distribution}
            self._d.update(parameters)

        if distribution is None:
            if hasattr(self, "_d"):
                distribution = self._d["distribution"]
            else:
                raise AttributeError(
                    "Invalid `distribution` value: cannot be None if "
                    "default delays were not set at graph creation.")
        if parameters is None:
            if hasattr(self, "_d"):
                parameters = self._d
            else:
                raise AttributeError(
                    "Invalid `parameters` value: cannot be None if default"
                    " delays were not set at graph creation.")
        return nngt.core.Connections.delays(
            self, elist=elist, dlist=delay, distribution=distribution,
            parameters=parameters, noise_scale=noise_scale)

    #-------------------------------------------------------------------------#
    # Getters

    @property
    def nodes_attributes(self):
        '''
        Access node attributes

        .. versionadded:: 0.7
        
        See also
        --------
        :attr:`~nngt.Graph.edge_attributes`,
        :attr:`~nngt.Graph.get_node_attributes`,
        :attr:`~nngt.Graph.new_node_attribute`,
        :attr:`~nngt.Graph.set_node_attribute`,
        '''
        return self._nattr

    @property
    def edges_attributes(self):
        '''
        Access edge attributes

        .. versionadded:: 0.7
        '''
        return self._eattr

    def get_nodes(self, attribute=None, value=None):
        '''
        Return the nodes in the network fulfilling a given condition.

        .. versionadded:: 1.2

        Parameters
        ----------
        attribute : str, optional (default: all nodes)
            Whether the `attribute` of the returned nodes should have a specific
            value.
        value : object, optional (default : None)
            If an `attribute` name is passed, then only nodes with `attribute`
            being equal to `value` will be returned.
        
        See also
        --------
        :func:`~nngt.Graph.get_edges`, :attr:`~nngt.Graph.nodes_attributes`
        '''
        if attribute is None:
            return [i for i in range(self.node_nb())]

        if value is None and self._nattr.value_type(attribute) != "object":
            raise ValueError("`value` cannot be None for attribute '" +
                             attribute + "'.")

        return np.where(
            self.get_node_attributes(name=attribute) == value)[0]

    def get_edges(self, attribute=None, value=None, source_node=None,
                  target_node=None):
        '''
        Return the edges in the network fulfilling a given condition.

        .. versionadded:: 1.2

        Parameters
        ----------
        attribute : str, optional (default: all nodes)
            Whether the `attribute` of the returned edges should have a specific
            value.
        value : object, optional (default : None)
            If an `attribute` name is passed, then only edges with `attribute`
            being equal to `value` will be returned.
        source_node : int or list of ints, optional (default: all nodes)
            Retrict the edges to those stemming from `source_node`.
        target_node : int or list of ints, optional (default: all nodes)
            Retrict the edges to those arriving at `target_node`.
        
        See also
        --------
        :func:`~nngt.Graph.get_nodes`, :attr:`~nngt.Graph.edges_attributes`
        '''
        edges = None

        if source_node is None and target_node is None:
            edges = self.edges_array
        elif is_integer(source_node) and is_integer(target_node):
            # check that the edge exists, throw error otherwise
            self.edge_id((source_node, target_node))
            edges = np.array([[source_node, target_node]])
        else:
            # we need to use the adjacency matrix, get its subparts,
            # then use the list of nodes to get the original ids back
            # to do that we first convert source/target_node to lists
            # (note that this has no significant speed impact)
            mat = self.adjacency_matrix()

            if source_node is None:
                source_node = np.array(
                    [i for i in range(self.node_nb())], dtype=int)
            elif is_integer(source_node):
                source_node = np.array([source_node], dtype=int)
            else:
                source_node = np.sort(source_node)

            if target_node is None:
                target_node = np.array(
                    [i for i in range(self.node_nb())], dtype=int)
            elif is_integer(target_node):
                target_node = np.array([target_node], dtype=int)
            else:
                target_node = np.sort(target_node)

            nnz = mat[source_node].tocsc()[:, target_node].nonzero()
            
            edges = np.array(
                [source_node[nnz[0]], target_node[nnz[1]]], dtype=int).T

        # check attributes
        if attribute is None:
            return edges

        if value is None and self._eattr.value_type(attribute) != "object":
            raise ValueError("`value` cannot be None for attribute '" +
                             attribute + "'.")

        desired = (self.get_edge_attributes(edges, attribute) == value)
        
        return self.edges_array[desired]

    def get_edge_attributes(self, edges=None, name=None):
        '''
        Attributes of the graph's edges.

        .. versionchanged:: 1.0
            Returns the full dict of edges attributes if called without
            arguments.

        .. versionadded:: 0.8

        Parameters
        ----------
        edge : tuple or list of tuples, optional (default: ``None``)
            Edge whose attribute should be displayed.
        name : str, optional (default: ``None``)
            Name of the desired attribute.

        Returns
        -------
        Dict containing all graph's attributes (synaptic weights, delays...)
        by default. If `edge` is specified, returns only the values for these
        edges. If `name` is specified, returns value of the attribute for each
        edge.

        Note
        ----
        The attributes values are ordered as the edges in
        :func:`~nngt.Graph.edges_array` if `edges` is None.
        
        See also
        --------
        :func:`~nngt.Graph.get_node_attributes`,
        :func:`~nngt.Graph.new_edge_attribute`,
        :func:`~nngt.Graph.set_edge_attribute`,
        :func:`~nngt.Graph.new_node_attribute`,
        :func:`~nngt.Graph.set_node_attribute`
        '''
        if name is not None and edges is not None:
            if isinstance(edges, slice):
                return self._eattr[name][edges]
            else:
                return self._eattr[edges][name]
        elif name is None and edges is None:
            return {k: self._eattr[k] for k in self._eattr.keys()}
        elif name is None:
            return self._eattr[edges]
        else:
            return self._eattr[name]

    def get_node_attributes(self, nodes=None, name=None):
        '''
        Attributes of the graph's edges.

        .. versionchanged:: 1.0.1
            Corrected default behavior and made it the same as
            :func:`~nngt.Graph.get_edge_attributes`.

        .. versionadded:: 0.9

        Parameters
        ----------
        nodes : list of ints, optional (default: ``None``)
            Nodes whose attribute should be displayed.
        name : str, optional (default: ``None``)
            Name of the desired attribute.

        Returns
        -------
        Dict containing all nodes attributes by default. If `nodes` is
        specified, returns a ``dict`` containing only the attributes of these
        nodes. If `name` is specified, returns a list containing the values of
        the specific attribute for the required nodes (or all nodes if
        unspecified).

        See also
        --------
        :func:`~nngt.Graph.get_edge_attributes`,
        :func:`~nngt.Graph.new_node_attribute`,
        :func:`~nngt.Graph.set_node_attribute`,
        :func:`~nngt.Graph.new_edge_attributes`,
        :func:`~nngt.Graph.set_edge_attribute`
        '''
        res = None

        if name is None:
            res = {k: self._nattr[k] for k in self._nattr.keys()}
        else:
            res = self._nattr[name]

        if nodes is None:
            return res

        if isinstance(nodes, (slice, int)) or nonstring_container(nodes):
            if isinstance(res, dict):
                return {k: v[nodes] for k, v in res.items()}
            return res[nodes]
        else:
            raise ValueError("Invalid `nodes`: "
                             "{}, use slice, int, or list".format(nodes))

    def get_attribute_type(self, attribute_name, attribute_class=None):
        '''
        Return the type of an attribute (e.g. string, double, int).

        .. versionchanged:: 1.0
            Added `attribute_class` parameter.

        Parameters
        ----------
        attribute_name : str
            Name of the attribute.
        attribute_class : str, optional (default: both)
            Whether `attribute_name` is a "node" or an "edge" attribute.

        Returns
        -------
        type : str
            Type of the attribute.
        '''
        if attribute_class is None:
            if attribute_name in self._eattr and attribute_name in self._nattr:
                raise RuntimeError("Both edge and node attributes with name '"
                                   + attribute_name + "' exist, please "
                                   "specify `attribute_class`")
            elif attribute_name in self._eattr:
                return self._eattr.value_type(attribute_name)
            elif attribute_name in self._nattr:
                return self._nattr.value_type(attribute_name)
            else:
                raise KeyError("No '{}' attribute.".format(attribute_name))
        else:
            if attribute_class == "edge":
                return self._eattr.value_type(attribute_name)
            elif attribute_class == "node":
                return self._nattr.value_type(attribute_name)
            else:
                raise InvalidArgument(
                    "Unknown attribute class '{}'.".format(attribute_class))

    def get_name(self):
        ''' Get the name of the graph '''
        return self._name

    def get_graph_type(self):
        ''' Return the type of the graph (see nngt.generation) '''
        return self._graph_type

    def get_density(self):
        '''
        Density of the graph: :math:`\\frac{E}{N^2}`, where `E` is the number of
        edges and `N` the number of nodes.
        '''
        return self.edge_nb()/float(self.node_nb()**2)

    def is_weighted(self):
        ''' Whether the edges have weights '''
        return "weight" in self.edges_attributes

    def is_directed(self):
        ''' Whether the graph is directed or not '''
        return self._directed

    def get_degrees(self, deg_type="total", node_list=None, use_weights=False,
                    syn_type="all"):
        '''
        Degree sequence of all the nodes.

        .. versionchanged:: 0.9
            Added `syn_type` keyword.

        Parameters
        ----------
        deg_type : string, optional (default: "total")
            Degree type (among 'in', 'out' or 'total').
        node_list : list, optional (default: None)
            List of the nodes which degree should be returned
        use_weights : bool, optional (default: False)
            Whether to use weighted (True) or simple degrees (False).
        syn_type : int or str, optional (default: all)
            Restrict to a given synaptic type ("excitatory", 1, or
            "inhibitory", -1).

        Returns
        -------
        :class:`numpy.array` or None (if an invalid type is asked).
        '''
        valid_types = ("in", "out", "total")
        if deg_type in valid_types:
            if syn_type in ("excitatory", 1):
                e_neurons = []
                if isinstance(self, Network):
                    for g in self.population.values():
                        if g.neuron_type == 1:
                            e_neurons.extend(g.ids)
                else:
                    e_neurons = np.where(
                        self.get_node_attributes(name="type") == 1)[0]
                return self.adjacency_matrix(
                    weights=use_weights,
                    types=False)[e_neurons, :].sum(axis=0).A1
            elif syn_type in ("inhibitory", -1):
                i_neurons = []
                if isinstance(self, Network):
                    for g in self.population.values():
                        if g.neuron_type == -1:
                            i_neurons.extend(g.ids)
                else:
                    i_neurons = np.where(
                        self.get_node_attributes(name="type") == -1)[0]
                return self.adjacency_matrix(
                    weights=use_weights,
                    types=False)[i_neurons, :].sum(axis=0).A1
            elif syn_type == "all":
                return self.degree_list(node_list, deg_type, use_weights)
            else:
                raise InvalidArgument(
                    "Invalid synaptic type '{}'".format(syn_type))
        else:
            raise InvalidArgument("Invalid degree type '{}'".format(deg_type))

    def get_betweenness(self, btype="both", use_weights=False):
        '''
        Betweenness centrality sequence of all nodes and edges.

        Parameters
        ----------
        btype : str, optional (default: ``"both"``)
            Type of betweenness to return (``"edge"``, ``"node"``-betweenness,
            or ``"both"``).
        use_weights : bool, optional (default: False)
            Whether to use weighted (True) or simple degrees (False).

        Returns
        -------
        node_betweenness : :class:`numpy.array`
            Betweenness of the nodes (if `btype` is ``"node"`` or ``"both"``).
        edge_betweenness : :class:`numpy.array`
            Betweenness of the edges (if `btype` is ``"edge"`` or ``"both"``).
        '''
        return self.betweenness_list(btype=btype, use_weights=use_weights)

    def get_edge_types(self, edges=None):
        '''
        Return the type of all or a subset of the edges.

        .. versionchanged:: 1.0.1
            Added the possibility to ask for a subset of edges.

        Parameters
        ----------
        edges : (E, 2) array, optional (default: all edges)
            Edges for which the type should be returned.

        Returns
        -------
        the list of types (1 for excitatory, -1 for inhibitory)
        '''
        if TYPE in self.edges_attributes:
            return self.get_edge_attributes(name=TYPE, edges=edges)
        else:
            size = self.edge_nb() if edges is None else len(edges)
            return np.ones(size)

    def get_weights(self, edges=None):
        '''
        Returns the weights of all or a subset of the edges.

        .. versionchanged:: 1.0.1
            Added the possibility to ask for a subset of edges.

        Parameters
        ----------
        edges : (E, 2) array, optional (default: all edges)
            Edges for which the type should be returned.

        Returns
        -------
        the list of weights
        '''
        if self.is_weighted():
            if edges is None:
                return self._eattr["weight"]
            else:
                return self._eattr[edges]["weight"]
        else:
            size = self.edge_nb() if edges is None else len(edges)
            return np.ones(size)

    def get_delays(self, edges=None):
        '''
        Returns the delays of all or a subset of the edges.

        .. versionchanged:: 1.0.1
            Added the possibility to ask for a subset of edges.

        Parameters
        ----------
        edges : (E, 2) array, optional (default: all edges)
            Edges for which the type should be returned.

        Returns
        -------
        the list of delays
        '''
        if edges is None:
            return self._eattr["delay"]
        else:
            return self._eattr[edges]["delay"]

    def is_spatial(self):
        '''
        Whether the graph is embedded in space (i.e. if it has a
        :class:`~nngt.geometry.Shape` attribute).
        Returns ``True`` is the graph is a subclass of
        :class:`~nngt.SpatialGraph`.
        '''
        return True if issubclass(self.__class__, SpatialGraph) else False

    def is_network(self):
        '''
        Whether the graph is a subclass of :class:`~nngt.Network` (i.e. if it
        has a :class:`~nngt.NeuralPop` attribute).
        '''
        return True if issubclass(self.__class__, Network) else False


# ------------ #
# SpatialGraph #
# ------------ #

class SpatialGraph(Graph):

    """
    The detailed class that inherits from :class:`~nngt.Graph` and implements
    additional properties to describe spatial graphs (i.e. graph where the
    structure is embedded in space.
    """

    #-------------------------------------------------------------------------#
    # Class properties

    __num_graphs = 0
    __max_id = 0

    #-------------------------------------------------------------------------#
    # Constructor, destructor, attributes

    def __init__(self, nodes=0, name="SpatialGraph", weighted=True,
                 directed=True, from_graph=None, shape=None, positions=None,
                 **kwargs):
        '''
        Initialize SpatialClass instance.

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`Graph` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        shape : :class:`~nngt.geometry.Shape`, optional (default: None)
            Shape of the neurons' environment (None leads to a square of
            side 1 cm)
        positions : :class:`numpy.array` (N, 2), optional (default: None)
            Positions of the neurons; if not specified and `nodes` is not 0,
            then neurons will be reparted at random inside the
            :class:`~nngt.geometry.Shape` object of the instance.
        **kwargs : keyword arguments for :class:`~nngt.Graph` or
            :class:`~nngt.geometry.Shape` if no shape was given.

        Returns
        -------
        self : :class:`~nggt.SpatialGraph`
        '''
        self.__id = self.__class__.__max_id
        self.__class__.__num_graphs += 1
        self.__class__.__max_id += 1
        self._shape = None
        self._pos   = None
        super(SpatialGraph, self).__init__(nodes, name, weighted, directed,
                                           from_graph, **kwargs)
        self._init_spatial_properties(shape, positions, **kwargs)
        if "population" in kwargs:
            self.make_network(self, kwargs["population"])

    def __del__(self):
        if hasattr(self, '_shape'):
            if self._shape is not None:
                self._shape._parent = None
            self._shape = None
        super(SpatialGraph, self).__del__()
        self.__class__.__num_graphs -= 1

    @property
    def shape(self):
        return self._shape

    #-------------------------------------------------------------------------#
    # Init tool

    def _init_spatial_properties(self, shape, positions=None, **kwargs):
        '''
        Create the positions of the neurons from the graph `shape` attribute
        and computes the connections distances.
        '''
        self.new_edge_attribute('distance', 'double')
        if positions is not None and positions.shape[0] != self.node_nb():
            raise InvalidArgument("Wrong number of neurons in `positions`.")
        if shape is not None:
            shape.set_parent(self)
            self._shape = shape
        else:
            if positions is None or not np.any(positions):
                if 'height' in kwargs and 'width' in kwargs:
                    self._shape = nngt.geometry.Shape.rectangle(
                        kwargs['height'], kwargs['width'], parent=self)
                elif 'radius' in kwargs:
                    self._shape = nngt.geometry.Shape.disk(
                        kwargs['radius'], parent=self)
                elif 'radii' in kwargs:
                    self._shape = nngt.geometry.Shape.ellipse(
                        kwargs['radii'], parent=self)
                elif 'polygon' in kwargs:
                    self._shape = nngt.geometry.Shape.from_polygon(
                        kwargs['polygon'], min_x=kwargs.get('min_x', -5000.),
                        max_x=kwargs.get('max_x', 5000.),
                        unit=kwargs.get('unit', 'um'), parent=self)
                else:
                    raise RuntimeError('SpatialGraph needs a `shape` or '
                                       'keywords arguments to build one, or '
                                       'at least `positions` so it can create '
                                       'a square containing them')
            else:
                minx, maxx = np.min(positions[:, 0]), np.max(positions[:, 0])
                miny, maxy = np.min(positions[:, 1]), np.max(positions[:, 1])
                height, width = 1.01*(maxy - miny), 1.01*(maxx - minx)
                centroid = (0.5*(maxx + minx), 0.5*(maxy + miny))
                self._shape = nngt.geometry.Shape.rectangle(
                    height, width, centroid=centroid, parent=self)
        b_rnd_pos = True if not self.node_nb() or positions is None else False
        self._pos = self._shape.seed_neurons() if b_rnd_pos else positions
        nngt.core.Connections.distances(self)

    #-------------------------------------------------------------------------#
    # Getters

    def get_positions(self, neurons=None):
        '''
        Returns the neurons' positions as a (N, 2) array.

        Parameters
        ----------
        neurons : int or array-like, optional (default: all neurons)
            List of the neurons for which the position should be returned.
        '''
        if neurons is not None:
            return np.array(self._pos[neurons])
        return np.array(self._pos)


# ------- #
# Network #
# ------- #

class Network(Graph):

    """
    The detailed class that inherits from :class:`~nngt.Graph` and implements
    additional properties to describe various biological functions
    and interact with the NEST simulator.
    """

    #-------------------------------------------------------------------------#
    # Class attributes and methods

    __num_networks = 0
    __max_id       = 0

    @classmethod
    def num_networks(cls):
        ''' Returns the number of alive instances. '''
        return cls.__num_networks

    @classmethod
    def from_gids(cls, gids, get_connections=True, get_params=False,
                  neuron_model=default_neuron, neuron_param=None,
                  syn_model=default_synapse, syn_param=None, **kwargs):
        '''
        Generate a network from gids.

        Warning
        -------
        Unless `get_connections` and `get_params` is True, or if your
        population is homogeneous and you provide the required information, the
        information contained by the network and its `population` attribute
        will be erroneous!
        To prevent conflicts the :func:`~nngt.Network.to_nest` function is not
        available. If you know what you are doing, you should be able to find a
        workaround...

        Parameters
        ----------
        gids : array-like
            Ids of the neurons in NEST or simply user specified ids.
        get_params : bool, optional (default: True)
            Whether the parameters should be obtained from NEST (can be very
            slow).
        neuron_model : string, optional (default: None)
            Name of the NEST neural model to use when simulating the activity.
        neuron_param : dict, optional (default: {})
            Dictionary containing the neural parameters; the default value will
            make NEST use the default parameters of the model.
        syn_model : string, optional (default: 'static_synapse')
            NEST synaptic model to use when simulating the activity.
        syn_param : dict, optional (default: {})
            Dictionary containing the synaptic parameters; the default value
            will make NEST use the default parameters of the model.

        Returns
        -------
        net : :class:`~nngt.Network` or subclass
            Uniform network of disconnected neurons.
        '''
        from nngt.lib.errors import not_implemented
        if neuron_param is None:
            neuron_param = {}
        if syn_param is None:
            syn_param = {}
        # create the population
        size = len(gids)
        nodes = [i for i in range(size)]
        group = nngt.NeuralGroup(
            nodes, neuron_type=1, neuron_model=neuron_model, neuron_param=neuron_param)
        pop = nngt.NeuralPop.from_groups([group])
        # create the network
        net = cls(population=pop, **kwargs)
        net.nest_gid = np.array(gids)
        net._id_from_nest_gid = {gid: i for i, gid in enumerate(gids)}
        net.to_nest = not_implemented
        if get_connections:
            from nngt.simulation import get_nest_adjacency
            converter = {gid: i for i, gid in enumerate(gids)}
            mat = get_nest_adjacency(converter)
            edges = np.array(mat.nonzero()).T
            w = mat.data
            net.new_edges(edges, {'weight': w}, check_edges=False)
        if get_params:
            raise NotImplementedError('`get_params` not implemented yet.')
        return net

    @classmethod
    @deprecated("1.0", reason="of a redondant name", alternative="uniform")
    def uniform_network(cls, *args, **kwargs):
        return cls.uniform(*args, **kwargs)

    @classmethod
    @deprecated("1.0", reason="redondant name", alternative="exc_and_inhib")
    def ei_network(cls, *args, **kwargs):
        return cls.exc_and_inhib(*args, **kwargs)

    @classmethod
    def uniform(cls, size, neuron_model=default_neuron,
                        neuron_param=None, syn_model=default_synapse,
                        syn_param=None, **kwargs):
        '''
        Generate a network containing only one type of neurons.

        .. versionadded:: 1.0

        Parameters
        ----------
        size : int
            Number of neurons in the network.
        neuron_model : string, optional (default: 'aief_cond_alpha')
            Name of the NEST neural model to use when simulating the activity.
        neuron_param : dict, optional (default: {})
            Dictionary containing the neural parameters; the default value will
            make NEST use the default parameters of the model.
        syn_model : string, optional (default: 'static_synapse')
            NEST synaptic model to use when simulating the activity.
        syn_param : dict, optional (default: {})
            Dictionary containing the synaptic parameters; the default value
            will make NEST use the default parameters of the model.

        Returns
        -------
        net : :class:`~nngt.Network` or subclass
            Uniform network of disconnected neurons.
        '''
        if neuron_param is None:
            neuron_param = {}
        if syn_param is None:
            syn_param = {}
        pop = nngt.NeuralPop.uniform(
            size, neuron_model=neuron_model, neuron_param=neuron_param,
            syn_model=syn_model, syn_param=syn_param, parent=None)
        net = cls(population=pop, **kwargs)
        return net

    @classmethod
    def exc_and_inhib(cls, size, iratio=0.2, en_model=default_neuron,
            en_param=None, in_model=default_neuron, in_param=None,
            syn_spec=None, **kwargs):
        '''
        Generate a network containing a population of two neural groups:
        inhibitory and excitatory neurons.

        .. versionadded:: 1.0

        .. versionchanged:: 0.8
            Removed `es_{model, param}` and `is_{model, param}` in favour of
            `syn_spec` parameter.
            Renamed `ei_ratio` to `iratio` to match
            :func:`~nngt.NeuralPop.exc_and_inhib`.

        Parameters
        ----------
        size : int
            Number of neurons in the network.
        i_ratio : double, optional (default: 0.2)
            Ratio of inhibitory neurons: :math:`\\frac{N_i}{N_e+N_i}`.
        en_model : string, optional (default: 'aeif_cond_alpha')
           Nest model for the excitatory neuron.
        en_param : dict, optional (default: {})
            Dictionary of parameters for the the excitatory neuron.
        in_model : string, optional (default: 'aeif_cond_alpha')
           Nest model for the inhibitory neuron.
        in_param : dict, optional (default: {})
            Dictionary of parameters for the the inhibitory neuron.
        syn_spec : dict, optional (default: static synapse)
            Dictionary containg a directed edge between groups as key and the
            associated synaptic parameters for the post-synaptic neurons (i.e.
            those of the second group) as value. If provided, all connections
            between groups will be set according to the values contained in
            `syn_spec`. Valid keys are:

            - `('excitatory', 'excitatory')`
            - `('excitatory', 'inhibitory')`
            - `('inhibitory', 'excitatory')`
            - `('inhibitory', 'inhibitory')`

        Returns
        -------
        net : :class:`~nngt.Network` or subclass
            Network of disconnected excitatory and inhibitory neurons.

        See also
        --------
        :func:`~nngt.NeuralPop.exc_and_inhib`
        '''
        pop = nngt.NeuralPop.exc_and_inhib(
            size, iratio, en_model, en_param, in_model, in_param,
            syn_spec=syn_spec)
        net = cls(population=pop, **kwargs)
        return net

    #-------------------------------------------------------------------------#
    # Constructor, destructor and attributes

    def __init__(self, name="Network", weighted=True, directed=True,
                 from_graph=None, population=None, inh_weight_factor=1.,
                 **kwargs):
        '''
        Initializes :class:`~nngt.Network` instance.

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`Graph` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        from_graph : :class:`~nngt.core.GraphObject`, optional (default: None)
            An optional :class:`~nngt.core.GraphObject` to serve as base.
        population : :class:`nngt.NeuralPop`, (default: None)
            An object containing the neural groups and their properties:
            model(s) to use in NEST to simulate the neurons as well as their
            parameters.
        inh_weight_factor : float, optional (default: 1.)
            Factor to apply to inhibitory synapses, to compensate for example
            the strength difference due to timescales between excitatory and
            inhibitory synapses.

        Returns
        -------
        self : :class:`~nggt.Network`
        '''
        self.__id = self.__class__.__max_id
        self.__class__.__num_networks += 1
        self.__class__.__max_id += 1
        if population is None:
            raise InvalidArgument("Network needs a NeuralPop to be created")
        nodes = population.size
        if "nodes" in kwargs.keys():
            assert kwargs["nodes"] == nodes, "Incompatible values for " +\
                "`nodes` = {} with a `population` of size {}.".format(
                    kwargs["nodes"], nodes)
            del kwargs["nodes"]
        if "delays" not in kwargs:  # set default delay to 1.
            kwargs["delays"] = 1.
        super(Network, self).__init__(
            nodes=nodes, name=name, weighted=weighted, directed=directed,
            from_graph=from_graph, inh_weight_factor=inh_weight_factor,
            **kwargs)
        self._init_bioproperties(population)
        if "shape" in kwargs or "positions" in kwargs:
            self.make_spatial(self, shape=kwargs.get("shape", None),
                              positions=kwargs.get("positions", None))

    def __del__(self):
        super(Network, self).__del__()
        self.__class__.__num_networks -= 1

    @property
    def population(self):
        '''
        :class:`~nngt.NeuralPop` that divides the neurons into groups with
        specific properties.
        '''
        return self._population

    @population.setter
    def population(self, population):
        if issubclass(population.__class__, nngt.NeuralPop):
            if self.node_nb() == population.size:
                if population.is_valid:
                    self._population = population
                else:
                    raise AttributeError("NeuralPop is not valid (not all \
                    neurons are associated to a group).")
            else:
                raise AttributeError("{} and NeuralPop must have same number \
                    of neurons".format(self.__class__.__name__))
        else:
            raise AttributeError("Expecting NeuralPop but received \
                    {}".format(pop.__class__.__name__))

    @property
    def nest_gid(self):
        return self._nest_gid

    @nest_gid.setter
    def nest_gid(self, gids):
        self._nest_gid = gids
        for group in self.population.values():
            group._nest_gids = gids[group.ids]

    def get_edge_types(self):
        inhib_neurons = {}
        types         = np.ones(self.edge_nb())

        for g in self._population.values():
            if g.neuron_type == -1:
                for n in g.ids:
                    inhib_neurons[n] = None

        for i, e in enumerate(self.edges_array):
            if e[0] in inhib_neurons:
                types[i] = -1

        return types

    def id_from_nest_gid(self, gids):
        '''
        Return the ids of the nodes in the :class:`nngt.Network` instance from
        the corresponding NEST gids.

        Parameters
        ----------
        gids : int or tuple
            NEST gids.

        Returns
        -------
        ids : int or tuple
            Ids in the network. Same type as the requested `gids` type.
        '''
        if nonstring_container(gids):
            return np.array([self._id_from_nest_gid[gid] for gid in gids],
                            dtype=int)
        else:
            return self._id_from_nest_gid[gids]

    def to_nest(self, send_only=None, use_weights=True):
        '''
        Send the network to NEST.

        .. seealso::
            :func:`~nngt.simulation.make_nest_network` for parameters
        '''
        from nngt.simulation import make_nest_network
        if nngt._config['with_nest']:
            return make_nest_network(
                self, send_only=send_only, use_weights=use_weights)
        else:
            raise RuntimeError("NEST is not present.")

    #-------------------------------------------------------------------------#
    # Init tool

    def _init_bioproperties(self, population):
        ''' Set the population attribute and link each neuron to its group. '''
        self._population = None
        self._nest_gid = None
        self._id_from_nest_gid = None
        if not hasattr(self, '_iwf'):
            self._iwf = 1.
        if issubclass(population.__class__, nngt.NeuralPop):
            if population.is_valid or not self.node_nb():
                self._population = population
                nodes = population.size
                # create the delay attribute if necessary
                if "delay" not in self.edges_attributes:
                    self.set_delays()
            else:
                raise AttributeError("NeuralPop is not valid (not all neurons "
                                     "are associated to a group).")
        else:
            raise AttributeError("Expected NeuralPop but received "
                                 "{}".format(pop.__class__.__name__))

    #-------------------------------------------------------------------------#
    # Setter

    def set_types(self, syn_type, nodes=None, fraction=None):
        raise NotImplementedError("Cannot be used on :class:`~nngt.Network`.")

    def get_neuron_type(self, neuron_ids):
        '''
        Return the type of the neurons (+1 for excitatory, -1 for inhibitory).

        Parameters
        ----------
        neuron_ids : int or tuple
            NEST gids.

        Returns
        -------
        ids : int or tuple
            Ids in the network. Same type as the requested `gids` type.
        '''
        if is_integer(neuron_ids):
            group_name = self._population._neuron_group[neuron_ids]
            neuron_type = self._population[group_name].neuron_type
            return neuron_type
        else:
            groups = (self._population._neuron_group[i] for i in neuron_ids)
            types = tuple(self._population[gn].neuron_type for gn in groups)
            return types

    #-------------------------------------------------------------------------#
    # Getter

    def neuron_properties(self, idx_neuron):
        '''
        Properties of a neuron in the graph.

        Parameters
        ----------
        idx_neuron : int
            Index of a neuron in the graph.

        Returns
        -------
        dict of the neuron's properties.
        '''
        group_name = self._population._neuron_group[idx_neuron]
        return self._population[group_name].properties()


# -------------- #
# SpatialNetwork #
# -------------- #

class SpatialNetwork(Network, SpatialGraph):

    """
    Class that inherits from :class:`~nngt.Network` and
    :class:`~nngt.SpatialGraph` to provide a detailed description of a real
    neural network in space, i.e. with positions and biological properties to
    interact with NEST.
    """

    #-------------------------------------------------------------------------#
    # Class attributes

    __num_networks = 0
    __max_id = 0

    #-------------------------------------------------------------------------#
    # Constructor, destructor, and attributes

    def __init__(self, population, name="SpatialNetwork", weighted=True,
                 directed=True, shape=None, from_graph=None, positions=None,
                 **kwargs):
        '''
        Initialize SpatialNetwork instance

        Parameters
        ----------
        name : string, optional (default: "Graph")
            The name of this :class:`Graph` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        shape : :class:`~nngt.geometry.Shape`, optional (default: None)
            Shape of the neurons' environment (None leads to a square of side
            1 cm)
        positions : :class:`numpy.array`, optional (default: None)
            Positions of the neurons; if not specified and `nodes` != 0, then
            neurons will be reparted at random inside the
            :class:`~nngt.geometry.Shape` object of the instance.
        population : class:`~nngt.NeuralPop`, optional (default: None)
            Population from which the network will be built.

        Returns
        -------
        self : :class:`~nngt.SpatialNetwork`
        '''
        self.__id = self.__class__.__max_id
        self.__class__.__num_networks += 1
        self.__class__.__max_id += 1
        if population is None:
            raise InvalidArgument("Network needs a NeuralPop to be created")
        nodes = population.size
        super(SpatialNetwork, self).__init__(
            nodes=nodes, name=name, weighted=weighted, directed=directed,
            shape=shape, positions=positions, population=population,
            from_graph=from_graph, **kwargs)

    def __del__ (self):
        super(SpatialNetwork, self).__del__()
        self.__class__.__num_networks -= 1

    #-------------------------------------------------------------------------#
    # Setter

    def set_types(self, syn_type, nodes=None, fraction=None):
        raise NotImplementedError("Cannot be used on "
                                  ":class:`~nngt.SpatialNetwork`.")

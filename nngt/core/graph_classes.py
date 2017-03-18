#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Graph classes for graph generation and management """

from copy import deepcopy
import warnings

import numpy as np
import scipy.sparse as ssp

import nngt
import nngt.analysis as na
from nngt.lib import (InvalidArgument, as_string, save_to_file, load_from_file,
                      nonstring_container)
from nngt.lib.graph_helpers import _edge_prop
from nngt.globals import (default_neuron, default_synapse, POS, WEIGHT, DELAY,
                          DIST, TYPE)
if nngt.config['with_nest']:
    from nngt.simulation import make_nest_network


__all__ = [
    'Graph', 'SpatialGraph', 'Network', 'SpatialNetwork'
]


#-----------------------------------------------------------------------------#
# Graph
#------------------------
#

class Graph(nngt.core.GraphObject):
    
    """
    The basic class that contains a :class:`graph_tool.Graph` and some
    of is properties or methods to easily access them.
    """

    #-------------------------------------------------------------------------#
    # Class properties

    __num_graphs = 0
    __max_id = 0
    #~ __di_property_func = {
            #~ "reciprocity": reciprocity, "clustering": clustering,
            #~ "assortativity": assortativity, "diameter": diameter,
            #~ "scc": num_scc, "wcc": num_wcc, "radius": spectral_radius, 
            #~ "num_iedges": num_iedges }
    #~ __properties = __di_property_func.keys()
    
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
            library_graph._w = _edge_prop("weights", kwargs)
        library_graph._d = _edge_prop("delays", kwargs)
        library_graph.__id = cls.__max_id
        library_graph._name = "Graph" + str(cls.__num_graphs)
        cls.__max_id += 1
        cls.__num_graphs += 1
        return library_graph
        
    
    @classmethod
    def from_matrix(cls, matrix, weighted=True, directed=True):
        '''
        Creates a :class:`~nngt.Graph` from a :class:`scipy.sparse` matrix or
        a dense matrix.
        
        Parameters
        ----------
        matrix : :class:`scipy.sparse` matrix or :class:`numpy.array`
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
        if shape[0] != shape[1]:
            raise InvalidArgument('A square matrix is required')
        nodes = shape[0]
        if issubclass(matrix.__class__, ssp.spmatrix):
            graph_name = graph_name.replace('Y', 'Sparse')
            if not directed:
                if not (matrix.T != matrix).nnz == 0:
                    raise InvalidArgument('Incompatible directed=False option \
with non symmetric matrix provided.')
        else:
            graph_name = graph_name.replace('Y', 'Dense')
            if not directed:
                if not (matrix.T == matrix).all():
                    raise InvalidArgument('Incompatible directed=False option \
with non symmetric matrix provided.')
        edges = np.array(matrix.nonzero()).T
        graph = cls(nodes, name=graph_name.replace("Z", str(cls.__num_graphs)),
                    weighted=weighted, directed=directed)
        weights = None
        if weighted:
            if issubclass(matrix.__class__, ssp.spmatrix):
                weights = np.array(matrix[edges[:,0],edges[:,1]])[0]
            else:
                weights = matrix[edges[:,0], edges[:,1]]
        graph.new_edges(edges, {"weight": weights})
        return graph
    
    @staticmethod
    def from_file(filename, format="auto", delimiter=" ", secondary=";",
                 attributes=None, notifier="@", ignore="#", from_string=False):
        '''
        Import a saved graph from a file.
        @todo: implement population and shape loading, implement gml, dot, xml, gt

        Parameters
        ----------
        filename: str
            The path to the file.
        format : str, optional (default: "neighbour")
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
        delimiter : str, optional (default " ")
            Delimiter used to separate inputs in the case of custom formats 
            (namely "neighbour" and "edge_list")
        secondary : str, optional (default: ";")
            Secondary delimiter used to separate attributes in the case of
            custom formats.
        attributes : list, optional (default: [])
            List of names for the attributes present in the file. If a
            `notifier` is present in the file, names will be deduced from it;
            otherwise the attributes will be numbered.
        notifier : str, optional (default: "@")
            Symbol specifying the following as meaningfull information.
            Relevant information is formatted ``@info_name=info_value``, where
            ``info_name`` is in ("attributes", "directed", "name", "size") and
            associated ``info_value``s are of type (``list``, ``bool``,
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
        if attributes is None:
            attributes = []
        info, edges, attr, pop, shape = load_from_file(filename=filename,
                    format=format, delimiter=delimiter, secondary=secondary,
                    attributes=attributes, notifier=notifier)
        graph = Graph( nodes=info["size"], name=info["name"],
                       directed=info["directed"] )
        di_attr = {}
        if info["attributes"]: # their are attributes to add to the graph
            di_attr["names"] = info["attributes"]
            di_attr["types"] = info["attr_types"]
            di_attr["values"] = [ attr[name] for name in info["attributes"] ]
        graph.new_edges(edges, di_attr)
        if pop is not None:
            Network.make_network(graph, pop)
        if shape is not None:
            SpatialGraph.make_spatial(graph, shape)
        return graph

    @staticmethod
    def make_spatial(graph, shape=nngt.Shape(), positions=None, copy=False):
        '''
        Turn a :class:`~nngt.Graph` object into a :class:`~nngt.SpatialGraph`,
        or a :class:`~nngt.Network` into a :class:`~nngt.SpatialNetwork`.

        Parameters
        ----------
        graph : :class:`~nngt.Graph` or :class:`~nngt.SpatialGraph`
            Graph to convert.
        shape : :class:`~nngt.Shape`
            Shape to associate to the new :class:`~nngt.SpatialGraph`.
        positions : (2,N) array
            Positions, in a 2D space, of the N neurons.
        copy : bool, optional (default: ``False``)
            Whether the operation should be made in-place on the object or if a
            new object should be returned.

        Notes
        -----
        In-place operation that directly converts the original graph if `copy`
        is ``False``, else returns the copied :class:`~nngt.Graph` turned into
        a :class:`~nngt.SpatialGraph`.
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
    def make_network(graph, neural_pop, copy=False):
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
        # take care of the weights and delays
        # @todo: use those of the from_graph
        if weighted:
            self._w = _edge_prop("weights", kwargs)
        if "delays" in kwargs:
            self._d = _edge_prop("delays", kwargs)
        # Init the core.GraphObject
        super(Graph, self).__init__(nodes=nodes, g=from_graph,
                                    directed=directed, weighted=weighted)
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
        return as_string(self)

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
        gc_instance = Graph(name=self._name+'_copy',
                            weighted=self._weighted,
                            from_graph=self)
        if self.is_spatial():
            SpatialGraph.make_spatial(gc_instance)
        if self.is_network():
            Network.make_network(gc_instance, deepcopy(self.population))
        return gc_instance

    def to_file(self, filename, format="auto", delimiter=" ", secondary=";",
                attributes=None, notifier="@"):
        '''
        Save graph to file; options detailed below.

        .. seealso::
            :py:func:`nngt.lib.save_to_file` function for options.
        '''
        save_to_file(self, filename, format=format, delimiter=delimiter,
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
#~ 
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
        '''
        #~ print("sea", attribute, values, val, value_type, edges)
        if attribute not in self.attributes():
            self._eattr.new_ea(name=attribute, value_type=value_type,
                               values=values, val=val)
        else:
            #~ print("sea2", values, val, edges)
            num_edges = self.edge_nb() if edges is None else len(edges)
            if values is None:
                if val is not None:
                    values = np.repeat(val, num_edges)
                else:
                    raise InvalidArgument("At least one of the `values` and "
                        "`val` arguments should not be ``None``.")
            if num_edges == self.edge_nb():
                assert num_edges == len(values), "One value per edge required."
                self._eattr[attribute] = values
            else:
                raise NotImplementedError("Currently, it is only possible to "
                    "change the attribute of all the edges at the same time, "
                    "not of only a subset.")
                
    
    def set_weights(self, weight=None, elist=None, distribution=None,
                    parameters=None, noise_scale=None):
        '''
        Set the synaptic weights.
        ..todo ::
            take elist into account in Connections.weights
        
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
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.
        '''
        if isinstance(weight, float):
            size = self.edge_nb() if elist is None else len(elist)
            weight = np.repeat(weight, size)
        elif not hasattr(weight, "__len__") and weight is not None:
            raise AttributeError('''Invalid `weight` value: must be either
                                 float, array-like or None.''')
        if distribution is None:
            distribution = self._w["distribution"]
        if parameters is None:
            parameters = self._w
        nngt.Connections.weights(
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
            if issubclass(nodes.__class__, int):
                inhib_nodes = graph.node_nb() - nodes
            elif issubclass(nodes.__class__, float):
                inhib_nodes = 1./nodes
            elif hasattr(nodes, '__iter__'):
                inhib_nodes = list(range(graph.node_nb()))
                nodes.sort()
                for node in nodes[::-1]:
                    del inhib_nodes[node]
        return nngt.Connections.types(self, inhib_nodes, fraction)
        
    def set_delays(self, delay=None, elist=None, distribution=None,
                   parameters=None, noise_scale=None):
        '''
        Set the delay for spike propagation between neurons.
        ..todo ::
            take elist into account in Connections.delays
        
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
        if isinstance(delay, float):
            size = self.edge_nb() if elist is None else len(elist)
            delay = np.repeat(delay, size)
        elif not hasattr(delay, "__len__") and delay is not None:
            raise AttributeError("Invalid `delay` value: must be either "
                                 "float, array-like or None")
        if distribution is None:
            if hasattr(self, "_d"):
                distribution = self._d["distribution"]
            else:
                raise AttributeError("Invalid `distribution` value: cannot be "
                                     "None if default delays were not set at "
                                     "graph creation.")
        if parameters is None:
            if hasattr(self, "_d"):
                parameters = self._d
            else:
                raise AttributeError("Invalid `parameters` value: cannot be "
                                     "None if default delays were not set at "
                                     "graph creation.")
        return nngt.Connections.delays(
            self, elist=elist, dlist=delay, distribution=distribution,
            parameters=parameters, noise_scale=noise_scale)
        

    #-------------------------------------------------------------------------#
    # Getters
    
    @property
    def node_attributes(self):
        ''' Access node attributes '''
        return self._nattr
    
    @property
    def edge_attribute(self):
        ''' Access edge attributes '''
        return self._eattr

    def attributes(self, edge=None, name=None):
        '''
        Attributes of the graph's edges.

        Parameters
        ----------
        edge : tuple, optional (default: ``None``)
            Edge whose attribute should be displayed.
        name : str, optional (default: ``None``)
            Name of the desired attribute.

        Returns
        -------
        List containing the names of the graph's attributes (synaptic weights,
        delays...) if `edge` is ``None``, else a ``dict`` containing the
        attributes of the edge (or the value of attribute `name` if it is not
        ``None``).
        '''
        if name is None and edge is None:
            return self._eattr.keys()
        elif name is None:
            return self._eattr[edge]
        elif edge is None:
            return self._eattr[name]
        else:
            return self._eattr[edge][name]
    
    def get_attribute_type(self, attribute_name):
        ''' Return the type of an attribute '''
        return self._eattr.value_type(attribute_name)
    
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
        return "weight" in self.attributes()

    def is_directed(self):
        ''' Whether the graph is directed or not '''
        return self._directed

    #~ def get_property(self, s_property):
        #~ ''' Return the desired property or None for an incorrect one. '''
        #~ if s_property in Graph.__properties:
            #~ return Graph.__di_property_func[s_property](self._graph)
        #~ else:
            #~ warnings.warn("Ignoring request for unknown property \
                          #~ '{}'".format(s_property))
            #~ return None

    #~ def get_properties(self, a_properties):
        #~ '''
        #~ Return a dictionary containing the desired properties
#~ 
        #~ Parameters
        #~ ----------
        #~ a_properties : sequence
            #~ List or tuple of strings of the property names.
#~ 
        #~ Returns
        #~ -------
        #~ di_result : dict
            #~ A dictionary of values with the property names as keys.
        #~ '''
        #~ di_result = { prop: self.get_property(prop) for prop in a_properties }
        #~ return di_result

    def get_degrees(self, deg_type="total", node_list=None, use_weights=False):
        '''
        Degree sequence of all the nodes.
        
        Parameters
        ----------
        deg_type : string, optional (default: "total")
            Degree type (among 'in', 'out' or 'total').
        node_list : list, optional (default: None)
            List of the nodes which degree should be returned
        use_weights : bool, optional (default: False)
            Whether to use weighted (True) or simple degrees (False).
        
        Returns
        -------
        :class:`numpy.array` or None (if an invalid type is asked).
        '''
        valid_types = ("in", "out", "total")
        if deg_type in valid_types:
            return self.degree_list(node_list, deg_type, use_weights)
        else:
            raise InvalidArgument("Invalid degree type '{}'".format(strType))

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

    def get_edge_types(self):
        if TYPE in self.edge_properties.keys():
            return self.edge_properties[TYPE].a
        else:
            return repeat(1, self.edge_nb())
    
    def get_weights(self):
        ''' Returns the weighted adjacency matrix as a
        :class:`scipy.sparse.lil_matrix`.
        '''
        return self.eproperties["weight"]
    
    def get_delays(self):
        ''' Returns the delay adjacency matrix as a
        :class:`scipy.sparse.lil_matrix` if delays are present; else raises
        an error.
        '''
        return self.eproperties["delay"]

    def is_spatial(self):
        '''
        Whether the graph is embedded in space (i.e. if it has a
        :class:`~nngt.Shape` attribute).
        Returns ``True`` is the graph is a subclass of
        :class:`~nngt.SpatialGraph`.
        '''
        return True if issubclass(self.__class__, SpatialGraph) else False

    def is_network(self):
        '''
        Whether the graph is a subclass of :class:`~nngt.Network` (i.e. if it
        has a :class:`~nngt.Shape` attribute).
        '''
        return True if issubclass(self.__class__, Network) else False



#-----------------------------------------------------------------------------#
# SpatialGraph
#------------------------
#

class SpatialGraph(Graph):
    
    """
    The detailed class that inherits from :class:`Graph` and implements
    additional properties to describe spatial graphs (i.e. graph where the
    structure is embedded in space.
    """

    #-------------------------------------------------------------------------#
    # Class properties

    __num_graphs = 0
    __max_id = 0

    #-------------------------------------------------------------------------#
    # Constructor, destructor, attributes    
    
    def __init__(self, nodes=0, name="Graph", weighted=True, directed=True,
                  from_graph=None, shape=None, positions=None, **kwargs):
        '''
        Initialize SpatialClass instance.
        .. todo::
            see what we do with the from_graph argument

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
        shape : :class:`~nngt.Shape`, optional (default: None)
            Shape of the neurons' environment (None leads to Shape())
        positions : :class:`numpy.array`, optional (default: None)
            Positions of the neurons; if not specified and `nodes` is not 0,
            then neurons will be reparted at random inside the
            :class:`~nngt.Shape` object of the instance.
        
        Returns
        -------
        self : :class:`~nggt.SpatialGraph`
        '''
        self.__id = self.__class__.__max_id
        self.__class__.__num_graphs += 1
        self.__class__.__max_id += 1
        self._shape = None
        super(SpatialGraph, self).__init__(nodes, name, weighted, directed,
                                           from_graph, **kwargs)
        self._init_spatial_properties(shape, positions, **kwargs)
        
    def __del__(self):
        if self._shape is not None:
            self._shape._parent = None
        self._shape = None
        super(SpatialGraph, self).__del__()
        self.__class__.__num_graphs -= 1

    @property
    def shape(self):
        return self._shape

    @property
    def position(self):
        return self._pos

    #-------------------------------------------------------------------------#
    # Init tool
    
    def _init_spatial_properties(self, shape, positions=None, **kwargs):
        '''
        Create the positions of the neurons from the graph `shape` attribute
        and computes the connections distances.
        '''
        if shape is not None:
            shape.set_parent(self)
            self._shape = shape
        else:
            self._shape = nngt.Shape.rectangle(self,1,1)
        if positions is not None and positions.shape[1] != self.node_nb():
            raise InvalidArgument("Wrong number of neurons in `positions`.")
        b_rnd_pos = True if not self.node_nb() or positions is None else False
        self._pos = self._shape.rnd_distrib() if b_rnd_pos else positions
        nngt.Connections.distances(self)


#-----------------------------------------------------------------------------#
# Network
#------------------------
#

class Network(Graph):
    
    """
    The detailed class that inherits from :class:`Graph` and implements
    additional properties to describe various biological functions
    and interact with the NEST simulator.
    """

    #-------------------------------------------------------------------------#
    # Class attributes and methods

    __num_networks = 0
    __max_id = 0
        
    @classmethod
    def num_networks(cls):
        ''' Returns the number of alive instances. '''
        return cls.__num_networks

    @classmethod
    def uniform_network(cls, size, neuron_model=default_neuron,
                        neuron_param=None, syn_model=default_synapse,
                        syn_param=None):
        '''
        Generate a network containing only one type of neurons.
        
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
        pop = nngt.NeuralPop.uniform_population(
            size, None, neuron_model, neuron_param, syn_model, syn_param)
        net = cls(population=pop)
        return net

    @classmethod
    def ei_network(cls, size, ei_ratio=0.2, en_model=default_neuron,
            en_param=None, es_model=default_synapse, es_param=None,
            in_model=default_neuron, in_param=None, is_model=default_synapse,
            is_param=None):
        '''
        Generate a network containing a population of two neural groups:
        inhibitory and excitatory neurons.
        
        Parameters
        ----------
        size : int
            Number of neurons in the network.
        ei_ratio : double, optional (default: 0.2)
            Ratio of inhibitory neurons: :math:`\\frac{N_i}{N_e+N_i}`.
        en_model : string, optional (default: 'aeif_cond_alpha')
           Nest model for the excitatory neuron.
        en_param : dict, optional (default: {})
            Dictionary of parameters for the the excitatory neuron.
        es_model : string, optional (default: 'static_synapse')
            NEST model for the excitatory synapse.
        es_param : dict, optional (default: {})
            Dictionary containing the excitatory synaptic parameters.
        in_model : string, optional (default: 'aeif_cond_alpha')
           Nest model for the inhibitory neuron.
        in_param : dict, optional (default: {})
            Dictionary of parameters for the the inhibitory neuron.
        is_model : string, optional (default: 'static_synapse')
            NEST model for the inhibitory synapse.
        is_param : dict, optional (default: {})
            Dictionary containing the inhibitory synaptic parameters.
        
        Returns
        -------
        net : :class:`~nngt.Network` or subclass
            Network of disconnected excitatory and inhibitory neurons.
        '''
        if en_param is None:
            en_param = {}
        if es_param is None:
            es_param = {}
        if in_param is None:
            in_param = {}
        if is_param is None:
            is_param = {}
        pop = nngt.NeuralPop.exc_and_inhib(
            size, ei_ratio, None, en_model, en_param, es_model, es_param,
            in_model, in_param, is_model, is_param)
        net = cls(population=pop)
        return net

    #-------------------------------------------------------------------------#
    # Constructor, destructor and attributes
    
    def __init__(self, name="Network", weighted=True, directed=True,
                 from_graph=None, population=None, **kwargs):
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
            del kwargs["nodes"]
        super(Network, self).__init__(nodes=nodes, name=name,
                                      weighted=weighted, directed=directed,
                                      from_graph=from_graph, **kwargs)
        self._init_bioproperties(population)
    
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
        for group in iter(self.population.values()):
            group._nest_gids = gids[group.id_list]

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
        if isinstance(gids, int):
            return self._id_from_nest_gid[gids]
        else:
            return np.array([self._id_from_nest_gid[gid] for gid in gids],
                            dtype=int)

    def to_nest(self, use_weights=True):
        '''
        Send the network to NEST.
        
        .. seealso::
            :func:`~nngt.simulation.make_nest_network` for parameters
        '''
        from nngt.simulation import make_nest_network
        if nngt.config['with_nest']:
            return make_nest_network(self, use_weights)
        else:
            raise RuntimeError("NEST is not present.")

    #-------------------------------------------------------------------------#
    # Init tool
    
    def _init_bioproperties(self, population):
        ''' Set the population attribute and link each neuron to its group. '''
        self._population = None
        self._nest_gid = None
        self._id_from_nest_gid = None
        if issubclass(population.__class__, nngt.NeuralPop):
            if population.is_valid:
                self._population = population
                nodes = population.size
                # create the delay attribute if necessary
                if "delay" not in self.attributes():
                    nngt.Connections.delays(self)
            else:
                raise AttributeError("NeuralPop is not valid (not all \
                neurons are associated to a group).")
        else:
            raise AttributeError("Expected NeuralPop but received \
                    {}".format(pop.__class__.__name__))

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
        if isinstance(neuron_ids, int):
            group_name = self._population._neuron_group[neuron_ids]
            ntype = self._population[group_name].neuron_type
            return ntype
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



#-----------------------------------------------------------------------------#
# SpatialNetwork
#------------------------
#

class SpatialNetwork(Network,SpatialGraph):
    
    """
    Class that inherits from :class:`~nngt.Network` and :class:`SpatialGraph`
    to provide a detailed description of a real neural network in space, i.e.
    with positions and biological properties to interact with NEST.
    """

    #-------------------------------------------------------------------------#
    # Class attributes

    __num_networks = 0
    __max_id = 0

    #-------------------------------------------------------------------------#
    # Constructor, destructor, and attributes
    
    def __init__(self, population, name="Graph", weighted=True, directed=True,
                 shape=None, from_graph=None, positions=None, **kwargs):
        '''
        Initialize Graph instance

        Parameters
        ----------
        name : string, optional (default: "Graph")
            The name of this :class:`Graph` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        shape : :class:`~nngt.Shape`, optional (default: None)
            Shape of the neurons' environment (None leads to Shape())
        positions : :class:`numpy.array`, optional (default: None)
            Positions of the neurons; if not specified and `nodes` != 0, then
            neurons will be reparted at random inside the
            :class:`~nngt.Shape` object of the instance.
        population : class:`~nngt.NeuralPop`, optional (default: None)
        
        Returns
        -------
        self : :class:`~nggt.SpatialNetwork`
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
        raise NotImplementedError("Cannot be used on \
:class:`~nngt.SpatialNetwork`.")

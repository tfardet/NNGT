#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Graph classes for graph generation and management """

import warnings
from copy import deepcopy
from numpy import multiply
from ssp.sparse import lil_matrix

from ..constants import *
from .graph_measures import * #@todo: get only degrees betw and adjacency
from .graph_objects import GraphLib, GraphObject
from .graph_datastruct import NeuralPop, Shape, Connections



#-----------------------------------------------------------------------------#
# Graph
#------------------------
#

class Graph(object):
    
    """
    The basic class that contains a :class:`graph_tool.Graph` and some
    of is properties or methods to easily access them.

    :ivar id: :class:`int`
        unique id that identifies the instance.
    :ivar graph: :class:`~nngt.core.GraphObject`
        main attribute of the class instance.
    """

    #-------------------------------------------------------------------------#
    # Class properties

    __num_graphs = 0
    __max_id = 0
    __di_property_func = {
            "reciprocity": reciprocity, "clustering": clustering,
            "assortativity": assortativity, "diameter": diameter,
            "scc": num_scc, "wcc": num_wcc, "radius": spectral_radius, 
            "num_iedges": num_iedges }
    __properties = __di_property_func.keys()
    
    @classmethod
    def num_graphs(cls):
        ''' Returns the number of alive instances. '''
        return cls.__num_graphs

    #-------------------------------------------------------------------------#
    # Constructor/destructor and properties
    
    def __init__(self, nodes=0, name="Graph",
                  weighted=True, directed=True, libgraph=None, **kwargs):
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
        libgraph : :class:`~nngt.core.GraphObject`, optional
            An optional :class:`~nngt.core.GraphObject` to serve as base.
        
        Returns
        -------
        self : :class:`~nggt.core.Graph`
        '''
        self.__id = self.__class__.__max_id
        self._name = name
        self.__di_prop = {
            "id": self.__id,
            "name": name,
            "weighted": weighted,
            "directed": directed,
        }
        # dictionary containing the attributes
        self._data = {}
        if "data" in kwargs.keys():
            di_data = kwargs["data"]
            for key, value in di_data.iteritems():
                self._data[key] = value
        # create the graphlib graph
        if libgraph != None:
            self._graph = libgraph
        else:
            self._graph = GraphObject(nodes=nodes, directed=directed)
        # take care of the weights @todo: use those of the libgraph
        if weighted:
            self._data["weights"] = lil_matrix((nodes,nodes))
            self._graph.new_edge_attribute("weights", float, val=0)
        # update the counters
        self.__class__.__num_graphs += 1
        self.__class__.__max_id += 1

    def __del__(self):
        self.__class__.__num_graphs -= 1

    @property
    def id(self):
        ''' unique :class:`int` identifying the instance '''
        return self.__id
    
    @property
    def graph(self):
        ''' :class:`graph_tool.Graph` attribute of the instance '''
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        if isinstance(new_graph, GraphLib):
            self._graph = GraphObject.to_graph_object(new_graph)
        elif isinstance(new_graph, GraphObject):
            self._graph = new_graph
        else:
            raise TypeError("The object passed is not a \
                GraphObject but a {}".format(new_graph.__class__.__name__))
    
    @property
    def name(self):
        return self._name

    #-------------------------------------------------------------------------#
    # Graph actions
    
    def copy(self):
        '''
        Returns a deepcopy of the current :class:`~nngt.core.Graph`
        instance
        '''
        gc_instance = Graph(name=self.__di_prop["name"]+'_copy',
                            weighted=self.__di_prop["weighted"],
                            graph=self._graph.copy())
        return gc_instance
    
    def add_edges(self, lst_edges):
        '''
        Add a list of edges to the graph.
        
        Parameters
        ----------
        lst_edges : list of 2-tuples or np.array of shape (edge_nb, 2)
            List of the edges that should be added as tuples (source, target)
            
        @todo: add example, check the edges for self-loops and multiple edges
        '''
        self._graph.new_edges(lst_edges)

    def inhibitory_subgraph(self):
        ''' Create a :class:`~nngt.core.Graph` instance which graph
        contains only the inhibitory edges of the current instance's
        :class:`graph_tool.Graph` '''
        eprop_b_type = self._graph.new_edge_property(
                       "bool",-self._graph.edge_properties["type"].a+1)
        self._graph.set_edge_filter(eprop_b_type)
        inhib_graph = Graph(
                            name=self.__di_prop["name"]+'_inhib',
                            weighted=self.__di_prop["weighted"],
                            graph=GraphObject(self._graph,prune=True))
        self._graph.clear_filters()
        return inhib_graph

    def excitatory_subgraph(self):
        ''' create a :class:`~nngt.core.Graph` instance which graph
        contains only the excitatory edges of the current instance's
        :class:`GraphObject` '''
        eprop_b_type = self._graph.new_edge_property(
                       "bool",self._graph.edge_properties["type"].a+1)
        self._graph.set_edge_filter(eprop_b_type)
        exc_graph = Graph(
                            name=self.__di_prop["name"]+'_exc',
                            weighted=self.__di_prop["weighted"],
                            graph=GraphObject(self._graph,prune=True))
        self._graph.clear_filters()
        return exc_graph

    def adjacency_matrix(self):
        return self._graph.adjacency()

    #-------------------------------------------------------------------------#
    # Setters
        
    def set_name(self, name=""):
        ''' set graph name '''
        if name != "":
            self.__di_prop["name"] = name
        else:
            strName = self.__di_prop["type"]
            tplIgnore = ("type", "name", "weighted")
            for key,value in self.__di_prop.items():
                if key not in tplIgnore and (value.__class__ != dict):
                    strName += '_' + key[0] + str(value)
            self.__di_prop["name"] = strName
    
    def set_weights(self, elist=None, wlist=None, distrib="gaussian",
                    correl=None, noise_scale=0.):
        '''
        Set the synaptic weights.
        
        Parameters
        ----------
        elist : class:`numpy.array`, optional (default: None)
            List of the edges (for user defined weights).
        wlist : class:`numpy.array`, optional (default: None)
            List of the weights (for user defined weights).
        distrib : class:`string`, optional (default: None)
            Type of distribution (choose among "uniform", "lognormal",
            "gaussian", "user_def", "lin_corr", "log_corr", "user_correl").
        correl : class:`string`, optional (default: None)
            Property to which the weights should be correlated.
        noise_scale : class:`int`, optional (default: 0.)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.
        '''
        Connections.weights(self, elist, wlist, distrib, correl, noise_scale)
        

    #-------------------------------------------------------------------------#
    # Getters
    
    def __getitem__(self, key):
        return self._data[key]
    
    def attributes(self):
        return self._data.keys()
    
    def get_name(self):
        return self.__di_prop["name"]
    
    def node_nb(self):
        return self._graph.node_nb()
    
    def edge_nb(self):
        return self._graph.edge_nb()

    def get_density(self):
        return self._graph.edge_nb()/float(self._graph.node_nb()**2)

    def is_weighted(self):
        return self.__di_prop["weighted"]

    def is_directed(self):
        return self.__di_prop["directed"]

    def get_property(self, s_property):
        ''' Return the desired property or None for an incorrect one. '''
        if s_property in Graph.__properties:
            return Graph.__di_property_func[s_property](self._graph)
        else:
            warnings.warn("Ignoring request for unknown property \
                          '{}'".format(s_property))
            return None

    def get_properties(self, a_properties):
        '''
        Return a dictionary containing the desired properties

        Parameters
        ----------
        a_properties : sequence
            List or tuple of strings of the property names.

        Returns
        -------
        di_result : dict
            A dictionary of values with the property names as keys.
        '''
        di_result = { prop: self.get_property(prop) for prop in a_properties }
        return di_result

    def get_degrees(self, strType="total", bWeights=True):
        lstValidTypes = ["in", "out", "total"]
        if strType in lstValidTypes:
            return degree_list(self._graph, strType, bWeights)
        else:
            warnings.warn("Ignoring invalid degree type '{}'".format(strType))
            return None

    def get_betweenness(self, bWeights=True):
        if bWeights:
            if not self.bWBetwToDate:
                self.wBetweeness = betweenness_list(self._graph, bWeights)
                self.wBetweeness = True
            return self.wBetweeness
        if not self.bBetwToDate and not bWeights:
            self.betweenness = betweenness_list(self._graph, bWeights)
            self.bBetwToDate = True
            return self.betweenness

    def get_edge_types(self):
        if "type" in self._graph.edge_properties.keys():
            return self._graph.edge_properties["type"].a
        else:
            return repeat(1, self._graph.edge_nb())
    
    def get_weights(self):
        if self.is_weighted():
            epropW = self._graph.edge_properties["weight"].copy()
            epropW.a = multiply(epropW.a,
                                self._graph.edge_properties["type"].a)
            return epropW
        else:
            return self._graph.edge_properties["type"].copy()



#-----------------------------------------------------------------------------#
# SpatialGraph
#------------------------
#

class SpatialGraph(Graph):
    
    """
    The detailed class that inherits from :class:`Graph` and implements
    additional properties to describe various biological functions
    and interact with the NEST simulator.

    :ivar shape: :class:`~nngt.Shape`
        Shape of the neurons environment.
    :ivar positions: :class:`numpy.array`
        Positions of the neurons.
    :ivar graph: :class:`~nngt.GraphObject`
        Main attribute of the class instance.
    """

    #-------------------------------------------------------------------------#
    # Class properties

    __num_graphs = 0
    __max_id = 0
    __di_property_func = {}
    __properties = __di_property_func.keys()

    #-------------------------------------------------------------------------#
    # Constructor, destructor, attributes    
    
    def __init__(self, nodes=0, name="Graph", weighted=True, directed=True,
                  libgraph=None, shape=None, positions=None, **kwargs):
        '''
        Initialize SpatialClass instance.
        @todo: see what we do with the libgraph argument

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
        shape : :class:`~nngt.core.Shape`, optional (default: None)
            Shape of the neurons' environment (None leads to Shape())
        positions : :class:`numpy.array`, optional (default: None)
            Positions of the neurons; if not specified and `nodes` is not 0,
            then neurons will be reparted at random inside the
            :class:`~nngt.core.Shape` object of the instance.
        
        Returns
        -------
        self : :class:`~nggt.Graph`
        '''
        super(SpatialGraph, self).__init__(nodes, name, weighted, directed,
                                           libgraph, **kwargs)
        self.__id = self.__class__.__max_id
        
        self._init_spatial_properties(shape, positions, **kwargs)
        
        self.__class__.__num_graphs += 1
        self.__class__.__max_id += 1
        self.__b_valid_properties = True
        
    def __del__(self):
        super(SpatialGraph, self).__del__()
        self.__class__.__num_graphs -= 1

    @property
    def shape(self):
        return self._shape

    #-------------------------------------------------------------------------#
    # Init tool
    
    def _init_spatial_properties(self, shape, positions, **kwargs):
        self._shape = shape if shape is not None else Shape(self)
        b_rnd_pos = ( True if not self.node_nb() or positions is None
                      else len(positions) != self.node_nb() )
        pos = self._shape.rnd_distrib() if b_rnd_pos else positions
        self._data["positions"] = pos
        if "data" in kwargs.keys():
            if "distances" not in self._data.keys():
                self._data["distances"] = Connections.distances(self, pos=pos)
        else:
            self._data["distances"] = Connections.distances(self, pos=pos)


#-----------------------------------------------------------------------------#
# Network
#------------------------
#

class Network(Graph):
    
    """
    The detailed class that inherits from :class:`Graph` and implements
    additional properties to describe various biological functions
    and interact with the NEST simulator.
    
    :ivar neural_model: :class:`list`
        List of the NEST neural models for each neuron.
    :ivar syn_model: :class:`list`
        List of the NEST synaptic models for each edge.
    :ivar graph: :class:`~nngt.core.GraphObject`
        Main attribute of the class instance
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
    def uniform_network(cls, size, neuron_model="iaf_neuron", neuron_param={},
                        syn_model="static_synapse", syn_param={}):
        pop = NeuralPop.uniform_population(size, None, neuron_model,
           neuron_param, syn_model, syn_param)
        net = cls(population=pop)
        pop.parent = net
        return net

    @classmethod
    def ei_network(cls, size, ei_ratio=0.2, en_model="aeif_neuron",
            en_param={}, es_model="static_synapse", es_param={},
            in_model="aeif_neuron", in_param={}, is_model="static_synapse",
            is_param={}):
        pop = NeuralPop.ei_population(size, ei_ratio, None, en_model, en_param,
                    es_model, es_param, in_model, in_param, is_model, is_param)
        net = cls(population=pop)
        pop.parent = net
        return net

    #-------------------------------------------------------------------------#
    # Constructor, destructor and attributes
    
    def __init__(self, name="Graph", weighted=True, directed=True,
                 libgraph=None, population=None, **kwargs):
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
        libgraph : :class:`~nngt.core.GraphObject`, optional (default: None)
            An optional :class:`~nngt.core.GraphObject` to serve as base.
        @todo:
        population : :class:`NeuralPop`, (default: None)
            A tuple containing the model(s) to use in NEST to simulate the 
            neurons as well as a dictionary containing the parameters for the
            neuron.
        
        Returns
        -------
        self : :class:`~nggt.core.Graph`
        '''
        if population is None:
            raise ArgumentError("Network needs a NeuralPop to be created")
        nodes = population.size
        if "nodes" in kwargs.keys():
            del kwargs["nodes"]
        super(Network, self).__init__(nodes=nodes, name=name,
                                      weighted=weighted, directed=directed,
                                      libgraph=libgraph, **kwargs)
        self.__id = self.__class__.__max_id
        self._init_bioproperties(population)
        
        self.__class__.__num_networks += 1
        self.__class__.__max_id += 1
        self.__b_valid_properties = True
    
    def __del__(self):
        super(Network, self).__del__()
        self.__class__.__num_networks -= 1

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        if issubclass(NeuralPop, population.__class__):
            if self._graph.node_nb() == population.size:
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

    #-------------------------------------------------------------------------#
    # Init tool
    
    def _init_bioproperties(self, population):
        ''' Set the population attribute and link each neuron to its group. '''
        if issubclass(NeuralPop, population.__class__):
            if population.is_valid:
                self._population = population
            else:
                raise AttributeError("NeuralPop is not valid (not all \
                neurons are associated to a group).")
        else:
            raise AttributeError("Expected NeuralPop but received \
                    {}".format(pop.__class__.__name__))

    #-------------------------------------------------------------------------#
    # Getter

    def neuron_properties(self, idx_neuron):
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

    :ivar shape: :class:`nngt.core.Shape`
        Shape of the neurons environment.
    :ivar positions: :class:`numpy.array`
        Positions of the neurons.
    :ivar neural_model: :class:`list`
        List of the NEST neural models for each neuron.
    :ivar syn_model: :class:`list`
        List of the NEST synaptic models for each edge.
    :ivar graph: :class:`~nngt.core.GraphObject`
        Main attribute of the class instance.
    """

    #-------------------------------------------------------------------------#
    # Class attributes

    __num_networks = 0
    __max_id = 0

    #-------------------------------------------------------------------------#
    # Constructor, destructor, and attributes
    
    def __init__(self, population, name="Graph", weighted=True, directed=True,
                 shape=None, graph=None, positions=None, **kwargs):
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
        shape : :class:`~nngt.core.Shape`, optional (default: None)
            Shape of the neurons' environment (None leads to Shape())
        positions : :class:`numpy.array`, optional (default: None)
            Positions of the neurons; if not specified and `nodes` != 0, then
            neurons will be reparted at random inside the
            :class:`~nngt.core.Shape` object of the instance.
        population : class:`~nngt.NeuralPop`, optional (default: None)
        
        Returns
        -------
        self : :class:`~nggt.core.Graph`
        '''
        if population is None:
            raise ArgumentError("Network needs a NeuralPop to be created")
        nodes = population.size
        super(SpatialNetwork, self).__init__(
            nodes=nodes, name=name, weighted=weighted, directed=directed,
            shape=shape, positions=positions, population=population, **kwargs)
        
        self.__id = self.__class__.__max_id
        
        self.__class__.__num_networks += 1
        self.__class__.__max_id += 1
        self.__b_valid_properties = True

    def __del__ (self):
        super(SpatialNetwork, self).__del__()
        self.__class__.__num_networks -= 1

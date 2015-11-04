#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Graph classes for graph generation and management """

import warnings
from copy import deepcopy
from numpy import multiply

from graph_tool.spectral import adjacency

from ..constants import *
from .graph_measures import *
from .graph_objects import GraphLib, GraphObject
from .Shape import Shape



#
#---
# GraphClass
#------------------------

class GraphClass(object):
    
    """
    .. py:currentmodule:: nggt.core
    
    The basic class that contains a :class:`graph_tool.Graph` and some
    of is properties or methods to easily access them.

    :ivar id: :class:`int`
        unique id that identifies the instance.
    :ivar graph: :class:`~nngt.core.GraphObject`
        main attribute of the class instance.
    """

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


    def __init__(self, nodes=0, name="Graph",
                  weighted=True, directed=True, graph=None):
        '''
        Initialize GraphClass instance

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`GraphClass` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        graph : :class:`~nngt.core.GraphObject`, optional
            An optional :class:`~nngt.core.GraphObject` to serve as base.
        
        Returns
        -------
        self : :class:`~nggt.core.GraphClass`
        '''
        self.__id = self.__class__.__max_id
        self.__di_prop = {
            "id": self.__id,
            "name": name,
            "weighted": weighted,
            "directed": directed,
        }
        if graph != None:
            self._graph = graph
        else:
            self._graph = GraphObject(nodes=nodes, directed=directed)
        self.__class__.__num_graphs += 1
        self.__class__.__max_id += 1

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
            raise TypeError("The object passed is not a\
            < class 'graph_tool.Graph' > but a {}".format(new_graph.__class__))

    def copy(self):
        '''
        Returns a deepcopy of the current :class:`~nngt.core.GraphClass`
        instance
        '''
        gc_instance = GraphClass(
                            name=self.__di_prop["name"]+'_copy',
                            weighted=self.__di_prop["weighted"],
                            graph=self._graph.copy())
        return gc_instance


    def inhibitory_subgraph(self):
        ''' Create a :class:`~nngt.core.GraphClass` instance which graph
        contains only the inhibitory edges of the current instance's
        :class:`graph_tool.Graph` '''
        eprop_b_type = self._graph.new_edge_property(
                       "bool",-self._graph.edge_properties["type"].a+1)
        self._graph.set_edge_filter(eprop_b_type)
        inhib_graph = GraphClass(
                            name=self.__di_prop["name"]+'_inhib',
                            weighted=self.__di_prop["weighted"],
                            graph=GraphObject(self._graph,prune=True))
        self._graph.clear_filters()
        return inhib_graph

    def excitatory_subgraph(self):
        ''' create a :class:`~nngt.core.GraphClass` instance which graph
        contains only the excitatory edges of the current instance's
        :class:`GraphObject` '''
        eprop_b_type = self._graph.new_edge_property(
                       "bool",self._graph.edge_properties["type"].a+1)
        self._graph.set_edge_filter(eprop_b_type)
        exc_graph = GraphClass(
                            name=self.__di_prop["name"]+'_exc',
                            weighted=self.__di_prop["weighted"],
                            graph=GraphObject(self._graph,prune=True))
        self._graph.clear_filters()
        return exc_graph

        
    def set_name(self,name=""):
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
        if s_property in GraphClass.__properties:
            return GraphClass.__di_property_func[s_property](self._graph)
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


    def __del__(self):
        self.__class__.__num_graphs -= 1



#
#---
# SpatialGraph
#------------------------

class SpatialGraph(GraphClass):
    
    """
    .. py:currentmodule:: nngt.core
    
    The detailed class that inherits from :class:`GraphClass` and implements
    additional properties to describe various biological functions
    and interact with the NEST simulator.

    :ivar shape: :class:`~nngt.Shape`
        Shape of the neurons environment.
    :ivar positions: :class:`numpy.array`
        Positions of the neurons.
    :ivar graph: :class:`~nngt.GraphObject`
        Main attribute of the class instance.
    """

    __num_graphs = 0
    __max_id = 0
    __di_property_func = {}
    __properties = __di_property_func.keys()
    
    
    def __init__(self, nodes=0, name="Graph", weighted=True, directed=True,
                  graph=None, shape=None, positions=None, **kwargs):
        '''
        Initialize SpatialClass instance

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`GraphClass` instance.
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
        self : :class:`~nggt.GraphClass`
        '''
        super(SpatialGraph, self).__init__(nodes, name,
                                         weighted, directed, graph)
        self.__id = self.__class__.__max_id
        
        self._init_spatial_properties(shape, positions)
        
        self.__class__.__num_graphs += 1
        self.__class__.__max_id += 1
        self.__b_valid_properties = True
    
    def _init_spatial_properties(self, shape, positions):
        self._shape = shape if shape is not None else Shape()
        b_rnd_pos = ( True if not self.node_nb() or positions is None
                      else len(positions) != self.node_nb() )
        self._pos = self._shape.rnd_distrib() if b_rnd_pos else positions

    @property
    def shape(self):
        return self._shape
    
    @GraphClass.graph.getter
    def graph(self):
        self.__b_valid_properties = False
        warnings.warn("The 'graph' attribute should not be modified!")
        return self._graph

    @GraphClass.graph.setter
    def graph(self, val):
        raise RuntimeError("The 'graph' attribute cannot be substituted after \
                            creation.")

    
    def __del__(self):
        super(SpatialGraph, self).__del__()
        self.__class__.__num_graphs -= 1
        


#
#---
# Network
#------------------------

class Network(GraphClass):
    
    """
    .. py:currentmodule:: nngt.core
    
    The detailed class that inherits from :class:`GraphClass` and implements
    additional properties to describe various biological functions
    and interact with the NEST simulator.
    
    :ivar neural_model: :class:`list`
        List of the NEST neural models for each neuron.
    :ivar syn_model: :class:`list`
        List of the NEST synaptic models for each edge.
    :ivar graph: :class:`~nngt.core.GraphObject`
        Main attribute of the class instance
    """

    __num_networks = 0
    __max_id = 0
        
    @classmethod
    def num_networks(cls):
        ''' Returns the number of alive instances. '''
        return cls.__num_networks

    
    def __init__(self, nodes=0, name="Graph",
                 weighted=True, directed=True, graph=None,
                 neuron_type=1, neural_model=default_neuron, **kwargs):
        '''
        Initializes :class:`~nngt.Network` instance.

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`GraphClass` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        graph : :class:`~nngt.core.GraphObject`, optional
            An optional :class:`~nngt.core.GraphObject` to serve as base.
        neuron_type : +/-1 or array (default: 1)
            The type of the neurons, either 1 for "excitatory" or -1 
            "inhibitory".
        @todo:
        neural_model : :class:`NeuralModel`, optional (default: `(default_neuron,default_dict)`)
            A tuple containing the model(s) to use in NEST to simulate the 
            neurons as well as a dictionary containing the parameters for the
            neuron.
        
        Returns
        -------
        self : :class:`~nggt.core.GraphClass`
        '''
        super(Network, self).__init__(nodes, name, weighted, directed, graph)
        self.__id = self.__class__.__max_id
        
        self.__class__.__num_networks += 1
        self.__class__.__max_id += 1
        self.__b_valid_properties = True
    
    def _init_bioproperties(self, neuron_type=1, neural_model=default_neuron,
                            syn_model=default_synapse):
        ''' initialize the NEST models' properties inside PropertyMaps of the
        graph attribute '''
        self._node_types = self._graph.new_vertex_property("int", neuron_type)
        self._neural_model = neural_model
        self._syn_model = syn_model

    @GraphClass.graph.getter
    def graph(self):
        self.__b_valid_properties = False
        warnings.warn("The 'graph' attribute should not be modified!")
        return self._graph

    @GraphClass.graph.setter
    def graph(self, val):
        raise RuntimeError("The 'graph' attribute cannot be substituted after \
                            creation.")
    
    
    def __del__(self):
        super(Network, self).__del__()
        self.__class__.__num_networks -= 1



#
#---
# SpatialNetwork
#------------------------

class SpatialNetwork(Network,SpatialGraph):
    
    """
    .. py:currentmodule:: nngt.core
    
    The detailed class that inherits from :class:`GraphClass` and implements
    additional properties to describe spatially embedded networks with various
    biological functions and interact with the NEST simulator.

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

    __num_networks = 0
    __max_id = 0
    
    @classmethod
    def make_network(cls, graph):
        graph.__class__ = cls
        graph._init_bioproperties()
        
    
    def __init__(self, nodes=0, name="Graph", weighted=True, directed=True,
                 shape=None, graph=None, positions=None,
                 neuron_type=1, neural_model=default_neuron):
        '''
        Initialize GraphClass instance

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`GraphClass` instance.
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
        neuron_type : +/-1 or array (default: 1)
            The type of the neurons, either 1 for "excitatory" or -1 
            "inhibitory".
        @todo:
        neural_model : :class:`NeuralModel`, optional (default: `(default_neuron,default_dict)`)
            A tuple containing the model(s) to use in NEST to simulate the 
            neurons as well as a dictionary containing the parameters for the
            neuron.
        
        Returns
        -------
        self : :class:`~nggt.core.GraphClass`
        '''
        super(SpatialNetwork, self).__init__(
            nodes=nodes, name=name, weighted=weighted, directed=directed,
            shape=shape, positions=positions, neuron_type=neuron_type,
            neural_model=neural_model)
        self.__id = self.__class__.__max_id
        self._shape = shape
        
        self.__class__.__num_networks += 1
        self.__class__.__max_id += 1
        self.__b_valid_properties = True

    @property
    def shape(self):
        return self._shape

    @Network.graph.getter
    def graph(self):
        self.__b_valid_properties = False
        warnings.warn("The 'graph' attribute should not be modified!")
        return self._graph

    @Network.graph.setter
    def graph(self, val):
        raise RuntimeError("The 'graph' attribute cannot be substituted after \
                            creation.")


    def __del__ (self):
        super(SpatialNetwork, self).__del__()
        self.__class__.__num_networks -= 1

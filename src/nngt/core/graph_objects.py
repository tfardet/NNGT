#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" GraphObject for subclassing the libraries graphs """

graph_lib,GraphLib,TNEANet = None,object,object
try:
    from graph_tool import Graph as GraphLib
    graph_lib = "graph_tool"
except:
    from snap import TNEANet as GraphLib
    graph_lib = "snap"



#
#---
# GtGraph
#------------------------

class GtGraph(GraphLib):
    
    '''
    Subclass of :class:`graph_tool.Graph` that (with 
    :class:`~nngt.core.SnapGraph`) unifies the methods to work with either
    `graph_tool` or `SNAP`.
    '''

    #-------------------------------------------------------------------------#
    # Class properties

    @classmethod
    def to_graph_object(cls, obj):
        obj.__class__ = cls
        return obj

    #-------------------------------------------------------------------------#
    # Constructor and instance properties        

    def __init__ (self, nodes=0, g=None, directed=True, prune=False,
                  vorder=None):
        '''
        @todo: document that
        see :class:`graph_tool.Graph`'s constructor '''
        super(GtGraph,self).__init__(g,directed,prune,vorder)
        self.add_vertex(nodes)
        self._node_attributes = _GtNProperty(self)
        self._edge_attributes = _GtEProperty(self)

    @property
    def node_attributes(self):
        return self._node_attribute

    @property
    def edge_attributes(self):
        return self._edge_attributes

    #-------------------------------------------------------------------------#
    # Graph manipulation

    def new_node_attribute(self, name, value_type, values=None, val=None):
         self._node_attributes._new_na(self, name, value_type, values, val)

    def new_edge_attribute(self, name, value_type, values=None, val=None):
         self._edge_attributes._new_ea(self, name, value_type, values, val)
    
    def new_node(self, n=1, ntype=1):
        '''
        Adding a node to the graph, with optional properties.
        
        Parameters
        ----------
        n : int, optional (default: 1)
            Number of nodes to add.
        ntype : int, optional (default: 1)
            Type of neuron (1 for excitatory, -1 for inhibitory)
            
        Returns
        -------
        The node or an iterator over the nodes created.
        '''
        node = self.add_vertex(n)
        return node

    def new_edge(source, target, add_missing=True, weight=1.):
        '''
        Adding a connection to the graph, with optional properties.
        
        Parameters
        ----------
        source : :class:`int/node`
            Source node.
        target : :class:`int/node`
            Target node.
        add_missing : :class:`bool`, optional (default: None)
            Add the nodes if they do not exist.
        weight : :class:`double`, optional (default: 1.)
            Weight of the connection (synaptic strength with NEST).
            
        Returns
        -------
        The new connection.
        '''
        connection = self.add_edge(source, target, add_missing)
        return connection

    def new_edges(self, edge_list, hashed=False, string_vals=False,
                  eprops=None):
        '''
        Adds a list of connections to the graph
        @todo: see how the eprops work
        '''
        self.add_edge_list(edge_list, hashed, string_vals, eprops)

    def remove_edge(self, edge):
        raise NotImplementedError("This function has been removed because it \
            makes using edge properties too complicated")

    def remove_vertex(self, node, fast=False):
        raise NotImplementedError("This function has been removed because it \
            makes using node properties too complicated")

    def rm_all_edges(self):
        '''
        @todo: this should be implemented in GraphClass
        Remove all connections in the graph
        '''
        self.clear_edges()

    def set_node_property(self):
        #@todo: do it...
        pass

    #-------------------------------------------------------------------------#
    # Getters

    def node_nb(self):
        return self.num_vertices()

    def edge_nb(self):
        return self.num_edges()


#
#---
# SnapGraph
#------------------------

class SnapGraph(GraphLib):
    
    '''
    Subclass of :class:`SNAP.TNEANet` that (with :class:`~nngt.core.GtGraph`)
     unifies the methods to work with either `graph_tool` or `SNAP`.
     
     @todo: do it!
    '''

    @classmethod
    def to_graph_object(cls, obj):
        obj.__class__ = cls
        

    def __init__ (self, nodes=0, g=None, directed=True):
        ''' see :class:`graph_tool.Graph` constructor '''
        super(SnapGraph,self).__init__(g)
        for _ in range(nodes):
            self.AddNode()
        self._directed = directed
        if not self._directed:
            snap.MakeUnDir(self)

    def new_node(self, n=1,
                 position=None, neuron_type=None, neural_model=None):
        '''
        Adding a node to the graph, with optional properties.
        
        Parameters
        ----------
        n : int, optional (default: 1)
            Number of nodes to add.
        position : tuple of int or 2D array, optional (default: None)
            The position of the node.
        neuron_type : :class:`string` or array of :class:`string`s (default: None)
            The type of the node.
        neural_model : :class:`string` (array), optional (default: None)
            NEST neural model to use on that nodes.
            
        Returns
        -------
        The node or an iterator over the nodes created.
        '''
        pass

    def new_edge(source, target, add_missing=True,
                       sweight=1., syn_model=None, syn_delay=None):
        '''
        Adding a connection to the graph, with optional properties.
        
        Parameters
        ----------
        source : :class:`int/node`
            Source node.
        target : :class:`int/node`
            Target node.
        add_missing : :class:`bool`, optional (default: None)
            Add the nodes if they do not exist.
        weight : :class:`double`, optional (default: 1.)
            Weight of the connection (synaptic strength with NEST).
        syn_model : :class:`string`, optional (default: None)
            Synaptic model to implement in NEST.
        syn_delay : :class:`string`, optional (default: None)
            Synaptic delay in NEST.
            
        Returns
        -------
        The new connection.
        '''
        if self._directed:
            pass
        else:
            pass

    def new_edges(lst_connections, hashed=False,
                            string_vals=False, eprops=None):
        '''
        Adds a list of connections to the graph
        
        @todo: see how the eprops work
        '''
        if self._directed:
            pass
        else:
            # check how SNAP deals with multiple edges
            pass


#
#---
# Graph properties
#------------------------

class _GtNProperty:

    ''' Class for generic interactions with nodes properties (graph_tool)  '''

    def __init__ (self, parent):
        self.parent = parent

    def __getitem__(self, name):
        return self.parent.vertex_properties[name]

    def __setitem__(self, name, value):
        size = self.parent.node_nb()
        if len(value) == size:
            self.parent.vertex_properties[name].a = np.array(value)
        else:
            raise ValueError("A list or a np.array with one entry per node in \
                             the graph is required")

    def _new_na(self, name, value_type, values=None, val=None):
        vprop = self.parent.new_vertex_property(value_type, values, val)
        self.parent.vertex_properties[name] = vprop

class _GtEProperty:

    ''' Class for generic interactions with nodes properties (graph_tool)  '''

    def __init__ (self, parent):
        self.parent = parent

    def __getitem__(self, name):
        return self.parent.edge_properties[name].a

    def __setitem__(self, name, value):
        size = self.parent.edge_nb()
        if len(value) == size:
            self.parent.edge_properties[name].a = np.array(value)
        else:
            raise ValueError("A list or a np.array with one entry per edge in \
                             the graph is required")

    def _new_ea(self, name, value_type, values=None, val=None):
        vprop = self.parent.new_edge_property(value_type, values, val)
        self.parent.edge_properties[name] = vprop


#
#---
# GraphObject
#------------------------

GraphObject = SnapGraph if graph_lib == "snap" else GtGraph

if __name__ == "__main__":
    graph = GraphObject()

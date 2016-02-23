#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" GraphObject for subclassing the libraries graphs """

import numpy as np
import scipy.sparse as ssp

from nngt.globals import glib_data, glib_func, BWEIGHT



#-----------------------------------------------------------------------------#
# GtGraph
#------------------------
#

class GtGraph(glib_data["graph"]):
    
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
        obj._nattr = _GtNProperty(obj)
        obj._eattr = _GtEProperty(obj)
        return obj

    #-------------------------------------------------------------------------#
    # Constructor and instance properties        

    def __init__ (self, nodes=0, g=None, directed=True, prune=False,
                  vorder=None):
        '''
        @todo: document that
        see :class:`graph_tool.Graph`'s constructor '''
        super(GtGraph,self).__init__(g,directed,prune,vorder)
        if g is None:
            self.add_vertex(nodes)
        self._nattr = _GtNProperty(self)
        self._eattr = _GtEProperty(self)

    @property
    def nproperties(self):
        return self._nattr

    @property
    def eproperties(self):
        return self._eattr

    #-------------------------------------------------------------------------#
    # Graph manipulation
    
    def new_node_attribute(self, name, value_type, values=None, val=None):
         self._nattr.new_na(name, value_type, values, val)

    def new_edge_attribute(self, name, value_type, values=None, val=None):
         self._eattr.new_ea(name, value_type, values, val)
    
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

    def new_edge(source, target, weight=1.):
        '''
        Adding a connection to the graph, with optional properties.
        
        Parameters
        ----------
        source : :class:`int/node`
            Source node.
        target : :class:`int/node`
            Target node.
        weight : :class:`double`, optional (default: 1.)
            Weight of the connection (synaptic strength with NEST).
            
        Returns
        -------
        The new connection.
        '''
        connection = self.add_edge(source, target, add_missing=True)
        if self.is_weighted():
            self.edge_properties['weight'][connection] = weight
        return connection

    def new_edges(self, edge_list, eprops=None):
        '''
        Adds a list of connections to the graph
        @todo: see how the eprops work
        '''
        self.add_edge_list(edge_list, eprops=eprops)
        return edge_list

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
    
    #-------------------------------------------------------------------------#
    # Getters
    
    def node_nb(self):
        return self.num_vertices()

    def edge_nb(self):
        return self.num_edges()

    def edges(self):
        return super(GtGraph, self).edges()
    
    def degree_list(self, node_list=None, deg_type="total", use_weights=True):
        if node_list is None:
            node_list = slice(0,-1)
        if "weight" in self.edge_properties.keys() and use_weights:
            return self.degree_property_map(deg_type,
                            self.edge_properties["weight"]).a[node_list]
        else:
            return self.degree_property_map(deg_type).a[node_list]

    def betweenness_list(self, use_weights=True, as_prop=False, norm=True):
        if self.edge_nb():
            if "weight" in self.edge_properties.keys() and use_weights:
                w_pmap = self.edge_properties[BWEIGHT]
                tpl = glib_func["betweenness"](self, weight=w_pmap, norm=norm)
                if as_prop:
                    return tpl[0], tpl[1]
                else:
                    return tpl[0].a, tpl[1].a
            else:
                tpl = betweenness(self)
                if as_prop:
                    return tpl[0], tpl[1]
                else:
                    return tpl[0].a, tpl[1].a
        else:
            if as_prop:
                return None, None
            else:
                return np.array([]), np.array([])


#-----------------------------------------------------------------------------#
# igraph
#------------------------
#

class IGraph(glib_data["graph"]):

    '''
    Subclass of :class:`igraph.Graph`.
    '''

    #-------------------------------------------------------------------------#
    # Class properties
    
    @classmethod
    def to_graph_object(cls, obj):
        obj.__class__ = cls
        return obj

    #-------------------------------------------------------------------------#
    # Constructor and instance properties
    
    def __init__(self, nodes=0, g=None, directed=True):
        if g is None:
            super(IGraph,self).__init__(n=nodes, directed=directed)
        else:
            nodes = g.vcount()
            edge_list = g.get_edge_list()
            di_node_attr = {}
            di_edge_attr = {}
            nattr = g.vs[0].attributes().keys()
            eattr = g.es[0].attributes().keys()
            for attr in nattr:
                di_node_attr[attr] = g.vs[:][attr]
            for attr in eattr:
                di_edge_attr[attr] = g.es[:][attr]
            super(IGraph,self).__init__(n=nodes, vertex_attrs=di_node_attr,
                                        edge_attrs=di_edge_attr)
        self.directed = directed
        self._nattr = _IgNProperty(self)
        self._eattr = _IgEProperty(self)

    @property
    def nproperties(self):
        return self._node_attribute

    @property
    def eproperties(self):
        return self._eattr

    #-------------------------------------------------------------------------#
    # Graph manipulation
    
    def new_node_attribute(self, name, value_type, values=None, val=None):
        self._nattr.new_na(name, value_type, values, val)

    def new_edge_attribute(self, name, value_type, values=None, val=None):
         self._eattr.new_ea(name, value_type, values, val)
    
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
        node_list = []
        first_node_idx = self.vcount()
        for v in range(n):
            node = self.add_vertex(type=ntype)
            node_list.append(first_node_idx+v)
        return node_list

    def new_edge(source, target, weight=1.):
        '''
        Adding a connection to the graph, with optional properties.
        
        Parameters
        ----------
        source : :class:`int/node`
            Source node.
        target : :class:`int/node`
            Target node.
        weight : :class:`double`, optional (default: 1.)
            Weight of the connection (synaptic strength with NEST).
            
        Returns
        -------
        The new connection.
        '''
        self.add_edge(source,target,weight=weight)
        return (source, target)

    def new_edges(self, edge_list, eprops=None):
        ''' Adds a list of connections to the graph '''
        first_eid = self.ecount()
        self.add_edges(edge_list)
        last_eid = self.ecount()
        if eprops is not None:
            for attr,lst in eprops.iteritems():
                self.es[first_eid:last_eid][attr] = lst
        return edge_list

    def remove_edge(self, edge):
        raise NotImplementedError("This function has been removed because it \
            makes using edge properties too complicated")

    def remove_vertex(self, node, fast=False):
        raise NotImplementedError("This function has been removed because it \
            makes using node properties too complicated")

    def clear_edges(self):
        '''
        @todo: this should be implemented in GraphClass
        Remove all connections in the graph
        '''
        self.delete_edges(None)
    
    #-------------------------------------------------------------------------#
    # Getters
    
    def node_nb(self):
        return self.vcount()

    def edge_nb(self):
        return self.ecount()

    def edges(self):
        return self.get_edgelist()
    
    def degree_list(self, node_list=None, deg_type="total", use_weights=True):
        deg_type = 'all' if deg_type == 'total' else deg_type
        if use_weights:
            return np.array(self.strength(node_list,mode=deg_type))
        else:
            return np.array(self.degree(node_list,mode=deg_type))

    def betweenness_list(self, use_weights=True, as_prop=False, norm=True):
        w = None
        if use_weights:
            w = self.es['bweight']
        node_betweenness = np.array(self.betweenness(weights=w))
        edge_betweenness = np.array(self.edge_betweenness(weights=w))
        if norm:
            n = self.vcount()
            e = self.ecount()
            ncoeff_norm = (n-1)*(n-2)
            ecoeff_norm = (e-1)*(e-2)/2.
            node_betweenness /= ncoeff_norm
            edge_betweenness /= ecoeff_norm
        return node_betweenness, edge_betweenness
    

#-----------------------------------------------------------------------------#
# Networkx
#------------------------
#

class NxGraph(glib_data["graph"]):

    '''
    Subclass of networkx Graph
    '''

    #-------------------------------------------------------------------------#
    # Class properties
    
    @classmethod
    def to_graph_object(cls, obj):
        obj.__class__ = cls
        return obj

    di_value = { "string": "", "double": 0., "int": int(0) }

    #-------------------------------------------------------------------------#
    # Constructor and instance properties
    
    def __init__(self, nodes=0, g=None, directed=True):
        super(NxGraph,self).__init__(g)
        if g is None and nodes:
            self.add_nodes_from(range(nodes))
        self.directed = directed
        self._nattr = _NxNProperty(self)
        self._eattr = _NxEProperty(self)

    @property
    def nproperties(self):
        return self._node_attribute

    @property
    def eproperties(self):
        return self._eattr

    #-------------------------------------------------------------------------#
    # Graph manipulation
    
    def new_node_attribute(self, name, value_type, values=None, val=None):
        num_nodes = self.node_nb()
        if values is None:
            if val is not None:
                values = np.repeat(val,num_nodes)
            else:
                if vector in value_type:
                    values = [ [] for _ in range(num_nodes) ]
                else:
                    values = np.repeat(self.di_value[value_type], num_nodes)
        elif len(values) != num_nodes:
            raise InvalidArgument("'values' list must contain one element per \
node in the graph.")
        for n, val in enumerate(values):
            self.node[n][name] = val

    def new_edge_attribute(self, name, value_type, values=None, val=None):
        num_edges = self.edge_nb()
        if values is None:
            if val is not None:
                values = np.repeat(val,num_edges)
            else:
                if "vector" in value_type:
                    values = [ [] for _ in range(num_edges) ]
                else:
                    values = np.repeat(self.di_value[value_type], num_edges)
        elif len(values) != num_edges:
            raise InvalidArgument("'values' list must contain one element per \
edge in the graph.")
        for e, val in zip(self.edges(),values):
            self.edge[e[0]][e[1]][name] = val
    
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
        tpl_new_nodes = tuple(range(len(self),len(self)+n))
        for v in tpl_new_nodes:
            self.add_node(v)
        return node

    def new_edge(source, target, weight=1.):
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
        self.add_edge(source, target)
        self[source][target]['weight'] = weight
        if not self.directed:
            self.add_edge(target,source)
            self[target][source]['weight'] = weight
        return (source, target)

    def new_edges(self, edge_list, eprops=None):
        ''' Adds a list of connections to the graph '''
        if eprops is not None:
            for attr in eprops.iterkeys():
                arr = eprops[attr]
                edges = ( (tpl[0],tpl[i][1], arr[i])
                          for i, tpl in enumerate(edge_list) )
                self.add_weighted_edges_from(edges, weight=attr)
                if not self.directed:
                    self.add_weighted_edges_from(
                        np.array(edge_list)[:,[1,0,2]], weight=attr)
        else:
            self.add_edges_from(edge_list, eprops)
            if not self.directed:
                self.add_edges_from(np.array(edge_list)[:,::-1], eprops)
        return edge_list

    def remove_edge(self, edge):
        raise NotImplementedError("This function has been removed because it \
            makes using edge properties too complicated")

    def remove_vertex(self, node, fast=False):
        raise NotImplementedError("This function has been removed because it \
            makes using node properties too complicated")

    def clear_edges(self):
        '''
        @todo: this should be implemented in GraphClass
        Remove all connections in the graph
        '''
        self.remove_edges_from(self.edges(self.nodes()))

    def set_node_property(self):
        #@todo: do it...
        pass
    
    #-------------------------------------------------------------------------#
    # Getters
    
    def node_nb(self):
        return self.number_of_nodes()

    def edge_nb(self):
        return self.size()

    def edges(self, **kwargs):
        return super(NxGraph, self).edges(**kwargs)
    
    def adjacency(self, weighted=True):
        if weighted:
            return adjacency(self)
        else:
            return adjacency(self, weight=None)
    
    def degree_list(self, node_list=None, deg_type="total", use_weights=True):
        weight = 'weight' if use_weights else None
        di_deg = None
        if deg_type == 'total':
            di_deg = self.degree(node_list, weight=weight)
        elif deg_type == 'in':
            di_deg = self.in_degree(node_list, weight=weight)
        else:
            di_deg = self.out_degree(node_list, weight=weight)
        return np.array(di_deg.values())

    def betweenness_list(self, use_weights=True, as_prop=False):
        di_nbetw, di_ebetw = None, None
        if use_weights:
            di_nbetw = glib_data["library"].betweenness_centrality(self,weight=BWEIGHT)
            di_ebetw = glib_data["library"].edge_betweenness_centrality(self,weight=BWEIGHT)
        else:
            di_nbetw = glib_data["library"].betweenness_centrality(self)
            di_ebetw = glib_data["library"].edge_betweenness_centrality(self)
        node_betweenness = np.array(di_nbetw.values())
        edge_betweenness = np.array(di_ebetw.values())
        return node_betweenness, edge_betweenness    


#-----------------------------------------------------------------------------#
# Snap graph
#------------------------
#

class SnapGraph(glib_data["graph"]):
    
    '''
    Subclass of :class:`SNAP.TNEANet` that (with :class:`~nngt.core.GtGraph`)
     unifies the methods to work with either `graph_tool` or `SNAP`.
     
     @todo: do it!
    '''

    @classmethod
    def to_graph_object(cls, obj):
        obj.__class__ = cls
        

    def __init__ (self, nodes=0, g=None, directed=True):
        ''' @todo '''
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
                 weight=1., syn_model=None, syn_delay=None):
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
        return self.parent.vertex_properties[name].a

    def __setitem__(self, name, value):
        size = self.parent.node_nb()
        if len(value) == size:
            self.parent.vertex_properties[name].a = np.array(value)
        else:
            raise ValueError("A list or a np.array with one entry per node in \
the graph is required")
    
    def keys(self):
        return self.parent.vertex_properties.keys()

    def new_na(self, name, value_type, values=None, val=None):
        vprop = self.parent.new_vertex_property(value_type, values, val)
        self.parent.vertex_properties[name] = vprop

class _GtEProperty:

    ''' Class for generic interactions with nodes properties (graph_tool)  '''

    def __init__ (self, parent):
        self.parent = parent
        self.stored = {}

    def __getitem__(self, name):
        '''
        Return the attributes of an edge or a list of edges.
        '''
        if isinstance(name, str):
            return self.parent.edge_properties[name].a
        elif hasattr(name[0], '__iter__'):
            di_eattr = {}
            for key in self.keys():
                di_eattr[key] = np.array( [ self.parent.edge_properties[key][e]
                                            for e in name ] )
            return di_eattr
        else:
            di_eattr = {}
            for key in self.keys():
                di_eattr[key] = self.parent.edge_properties[key][name]
            return di_eattr

    def __setitem__(self, name, value):
        size = self.parent.edge_nb()
        if len(value) == size:
            self.parent.edge_properties[name].a = np.array(value)
        else:
            raise ValueError("A list or a np.array with one entry per edge in \
the graph is required")

    def keys(self):
        return self.parent.edge_properties.keys()

    def new_ea(self, name, value_type, values=None, val=None):
        eprop = self.parent.new_edge_property(value_type, values, val)
        self.parent.edge_properties[name] = eprop
        self.stored[name] = value_type

class _IgNProperty:

    '''
    @todo
    Class for generic interactions with nodes properties (igraph)
    '''

    def __init__ (self, parent):
        self.parent = parent

    def __getitem__(self, name):
        return self.parent.vs[name]

    def __setitem__(self, name, value):
        if len(value) == size:
            self.parent.vs[name] = value
        else:
            raise ValueError("A list or a np.array with one entry per node in \
the graph is required")
    
    def keys(self):
        return self.parent.vs.attributes()

    def new_na(self, name, value_type, values=None, val=None):
        if val is None:
            if value_type == "int":
                val = int(0)
            elif value_type == "double":
                val = 0.
            elif value_type == "string":
                val = ""
            else:
                val = None
        if values is None:
            values = np.repeat(val, self.parent.vcount())
        self.parent.vs[name] = values

class _IgEProperty:

    ''' Class for generic interactions with nodes properties (networkx)  '''

    def __init__ (self, parent):
        self.parent = parent

    def __getitem__(self, name):
        return self.parent.es[name]

    def __setitem__(self, name, value):
        size = self.parent.edge_nb()
        if len(value) == size:
            self.parent.es[name] = value
        else:
            raise ValueError("A list or a np.array with one entry per edge in \
the graph is required")

    def keys(self):
        edge_seq = self.parent.es
        return edge_seq.attributes()

    def new_ea(self, name, value_type, values=None, val=None):
        if val is None:
            if value_type == "int":
                val = int(0)
            elif value_type == "double":
                val = 0.
            elif value_type == "string":
                val = ""
            else:
                val = None
        if values is None:
            values = np.repeat(val, self.parent.ecount())
        elif len(values) == self.parent.ecount():
            self.parent.es[name] = values
        else:
            raise ValueError("A list or a np.array with one entry per edge in \
the graph is required")

class _NxNProperty:

    '''
    Class for generic interactions with nodes properties (networkx)
    '''

    def __init__ (self, parent):
        self.parent = parent

    def __getitem__(self, name):
        lst = [self.parent.node[i][name] for i in range(self.parent.node_nb())]
        return np.array(lst)

    def __setitem__(self, name, value):
        if len(value) == size:
            for i in range(self.parent.node_nb()):
                self.parent.node[i][name] = value[i]
        else:
            raise ValueError("A list or a np.array with one entry per node in \
the graph is required")
    
    def keys(self):
        if self.parent.node_nb():
            return self.parent.node[-1].keys()
        else:
            return []

    def new_na(self, name, value_type, values=None, val=None):
        if val is None:
            if value_type == "int":
                val = int(0)
            elif value_type == "double":
                val = 0.
            elif value_type == "string":
                val = ""
            else:
                val = None
        if values is None:
            values = np.repeat(val, self.parent.node_nb())
        self[name] = values

class _NxEProperty:

    ''' Class for generic interactions with edge properties (networkx)  '''

    def __init__ (self, parent):
        self.parent = parent

    def __getitem__(self, name):
        lst = []
        for e in self.parent.edges():
            lst.append(self.parent.edge[e[0],e[1]][name])
        return np.array(lst)

    def __setitem__(self, name, value):
        size = self.parent.edge_nb()
        if len(value) == size:
            for i,e in enumerate(self.parent.edges()):
                self.parent.edge[e[0]][e[1]][name] = value[i]
        else:
            raise ValueError("A list or a np.array with one entry per edge in \
the graph is required")

    def keys(self):
        if self.parent.edge_nb():
            e = self.parent.edges()[-1]
            return self.parent.edge[e[0]][e[1]].keys()
        else:
            return []

    def new_ea(self, name, value_type, values=None, val=None):
        if val is None:
            if value_type == "int":
                val = int(0)
            elif value_type == "double":
                val = 0.
            elif value_type == "string":
                val = ""
            else:
                val = None
        if values is None:
            values = np.repeat(val, self.parent.ecount())
        elif len(values) == self.parent.ecount():
            self[name] = values
        else:
            raise ValueError("A list or a np.array with one entry per edge in \
the graph is required")


#-----------------------------------------------------------------------------#
# GraphObject
#-------
#

di_graphlib = {
    "graph_tool": GtGraph,
    "igraph": IGraph,
    "networkx": NxGraph,
    #~ "snap": SnapGraph
}

GraphLib = glib_data["graph"]

#: Graph object (reference to one of the main libraries' wrapper
GraphObject = di_graphlib[glib_data["name"]]


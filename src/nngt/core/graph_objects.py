#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" GraphObject for subclassing the libraries graphs """

from collections import OrderedDict
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import scipy.sparse as ssp

from nngt.globals import glib_data, glib_func, BWEIGHT, with_metaclass



adjacency = glib_func["adjacency"]

#-----------------------------------------------------------------------------#
# Library-dependent graph properties
#------------------------
#


@with_metaclass(ABCMeta)
class BaseProperty(dict):
    
    def __init__ (self, parent):
        self.parent = parent
        self.stored = {}
    
    @abstractmethod
    def __getitem__(self, name):
        pass
    
    @abstractmethod
    def __setitem__(self, name, value):
        pass

class _GtNProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph_tool)  '''

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
        return self.stored.keys()

    def new_na(self, name, value_type, values=None, val=None):
        vprop = self.parent.new_vertex_property(value_type, values, val)
        self.parent.vertex_properties[name] = vprop
        self.stored[name] = value_type

class _GtEProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph_tool)  '''

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

    def new_ea(self, name, value_type, values=None, val=None):
        eprop = self.parent.new_edge_property(value_type, values, val)
        self.parent.edge_properties[name] = eprop
        self.stored[name] = value_type

class _IgNProperty(BaseProperty):

    '''
    @todo
    Class for generic interactions with nodes properties (igraph)
    '''

    def __getitem__(self, name):
        return self.parent.vs[name]

    def __setitem__(self, name, value):
        if len(value) == size:
            self.parent.vs[name] = value
        else:
            raise ValueError("A list or a np.array with one entry per node in \
the graph is required")

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
        self.stored[name] = value_type

class _IgEProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (networkx)  '''

    def __getitem__(self, name):
        return self.parent.es[name]

    def __setitem__(self, name, value):
        size = self.parent.edge_nb()
        if len(value) == size:
            self.parent.es[name] = value
        else:
            raise ValueError("A list or a np.array with one entry per edge in \
the graph is required")

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
        self.stored[name] = value_type

class _NxNProperty(BaseProperty):

    '''
    Class for generic interactions with nodes properties (networkx)
    '''

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
        self.stored[name] = value_type

class _NxEProperty(BaseProperty):

    ''' Class for generic interactions with edge properties (networkx)  '''

    def __getitem__(self, name):
        lst = []
        for e in self.parent.edges:
            lst.append(self.parent.edge[e[0],e[1]][name])
        return np.array(lst)

    def __setitem__(self, name, value):
        size = self.parent.edge_nb()
        if len(value) == size:
            for i,e in enumerate(self.parent.edges):
                self.parent.edge[e[0]][e[1]][name] = value[i]
        else:
            raise ValueError("A list or a np.array with one entry per edge in \
the graph is required")

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
        self.stored[name] = value_type

# dictionary to chose

di_graphprop = {
    "graph_tool": { "node":_GtNProperty, "edge":_GtEProperty },
    "igraph": { "node":_IgNProperty, "edge":_IgEProperty },
    "networkx": { "node":_NxNProperty, "edge":_NxEProperty }
}


#-----------------------------------------------------------------------------#
# BaseGraph
#------------------------
#

@with_metaclass(ABCMeta)
class BaseGraph(object):
    
    #-------------------------------------------------------------------------#
    # Classmethod
    
    @classmethod
    def to_graph_object(cls, obj):
        obj.__class__ = cls
        edges = glib_func["get_edges"](obj)
        obj._nattr = di_graphprop[glib_data["name"]]["node"](obj)
        obj._eattr = di_graphprop[glib_data["name"]]["edge"](obj)
        obj._edges = OrderedDict()
        for i, edge in edges:
            obj._edges[tuple(edge)] = i
        return obj
        
    #-------------------------------------------------------------------------#
    # Shared properties methods
    
    @property
    def edges(self):
        ''' :class:`OrderedDict` containing the edges as keys (2-tuple) and
        their index at the time of their creation as value '''
        return self._edges
        
    @property
    def edges_array(self):
        ''' Edges of the graph, sorted by order of creation, as an array of
        2-tuple. '''
        return np.array(self._edges.keys(), copy=True)

    @property
    def nproperties(self):
        return self._nattr

    @property
    def eproperties(self):
        return self._eattr
    
    def new_node_attribute(self, name, value_type, values=None, val=None):
         self._nattr.new_na(name, value_type, values, val)

    def new_edge_attribute(self, name, value_type, values=None, val=None):
         self._eattr.new_ea(name, value_type, values, val)

    def remove_edge(self, edge):
        raise NotImplementedError("This function has been removed because it \
            makes using edge properties too complicated")

    def remove_vertex(self, node, fast=False):
        raise NotImplementedError("This function has been removed because it \
            makes using node properties too complicated")
        
    def adjacency_matrix(self, types=True, weights=True):
        weights = "weight" if weights is True else weights
        types = "type" if types is True else False
        mat = adjacency(self, weights)
        if types in self.attributes() and weights in (True, "weight"):
            mtype = adjacency(self, weight="type")
            return mat*mtype
        else:
            return mat
    
    #-------------------------------------------------------------------------#
    # Properties and methods to implement
    
    @abstractmethod
    def new_node(self, n=1, ntype=1):
        pass
    
    @abstractmethod
    def new_edge(source, target, weight=1.):
        pass
        
    @abstractmethod
    def new_edges(self, edge_list, eprops=None):
        pass
        
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
    def clear_all_edges(self):
        pass


#-----------------------------------------------------------------------------#
# GtGraph
#------------------------
#

class GtGraph(BaseGraph, glib_data["graph"]):
    
    '''
    Subclass of :class:`graph_tool.Graph` that (with 
    :class:`~nngt.core.SnapGraph`) unifies the methods to work with either
    `graph_tool` or `SNAP`.
    '''
    
    #-------------------------------------------------------------------------#
    # Constructor and instance properties        

    def __init__ (self, nodes=0, g=None, directed=True, prune=False,
                  vorder=None):
        '''
        @todo: document that
        see :class:`graph_tool.Graph`'s constructor '''
        self._edges = OrderedDict()
        self._nattr = _GtNProperty(self)
        self._eattr = _GtEProperty(self)
        super(GtGraph,self).__init__(g,directed,prune,vorder)
        if g is None:
            self.add_vertex(nodes)
        else:
            if g.__class__ is glib_data["graph"]:
                edges = glib_func["get_edges"](g)
                for i, edge in enumerate(edges):
                    self._edges[tuple(e)] = i
            else:
                self._edges = g._edges.copy()

    #-------------------------------------------------------------------------#
    # Graph manipulation
    
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
        Adding an edge to the graph, with optional properties.
        
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
        The new edge.
        '''
        self._edges[(source, target)] = self.node_nb()
        connection = self.add_edge(source, target, add_missing=True)
        if self.is_weighted():
            self.edge_properties['weight'][connection] = weight
        return connection

    def new_edges(self, edge_list, eprops=None):
        '''
        Adds a list of edges to the graph
        @todo: see how the eprops work
        '''
        n = self.node_nb()
        for i, edge in enumerate(edge_list):
            self._edges[tuple(edge)] = n + i
        self.add_edge_list(edge_list, eprops=eprops)
        return edge_list
    
    def clear_all_edges(self):
        self.clear_edges()
        self._edges = OrderedDict()
        self._eattr.clear()
    
    #-------------------------------------------------------------------------#
    # Getters
    
    def node_nb(self):
        return self.num_vertices()

    def edge_nb(self):
        return self.num_edges()
    
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

class IGraph(BaseGraph, glib_data["graph"]):

    '''
    Subclass of :class:`igraph.Graph`.
    '''

    #-------------------------------------------------------------------------#
    # Constructor and instance properties
    
    def __init__(self, nodes=0, g=None, directed=True, parent=None):
        self._edges = OrderedDict()
        self.directed = directed
        self._nattr = _IgNProperty(self)
        self._eattr = _IgEProperty(self)
        if g is None:
            super(IGraph,self).__init__(n=nodes, directed=directed)
        else:
            nodes = g.vcount()
            edges = g.ecount()
            di_node_attr = {}
            di_edge_attr = {}
            if nodes:
                nattr = g.vs[0].attributes().keys()
            if edges:
                eattr = g.es[0].attributes().keys()
            for attr in nattr:
                di_node_attr[attr] = g.vs[:][attr]
            for attr in eattr:
                di_edge_attr[attr] = g.es[:][attr]
            super(IGraph,self).__init__(n=nodes, vertex_attrs=di_node_attr)
            lst_edges = glib_func["get_edges"](g)
            self.new_edges(lst_edges, eprops=di_edge_attr)

    #-------------------------------------------------------------------------#
    # Graph manipulation
    
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
        self._edges[(source, target)] = self.node_nb()
        self.add_edge(source,target,weight=weight)
        return (source, target)

    def new_edges(self, edge_list, eprops=None):
        ''' Adds a list of connections to the graph '''
        n = self.node_nb()
        for i, edge in enumerate(edge_list):
            self._edges[tuple(edge)] = n + i
        first_eid = self.ecount()
        super(IGraph, self).add_edges(edge_list)
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

    def clear_all_edges(self):
        ''' Remove all connections in the graph. '''
        self.delete_edges(None)
        self._edges = OrderedDict()
        self._eattr.clear()
    
    #-------------------------------------------------------------------------#
    # Getters
    
    def node_nb(self):
        return self.vcount()

    def edge_nb(self):
        return self.ecount()
    
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

class NxGraph(BaseGraph, glib_data["graph"]):

    '''
    Subclass of networkx Graph
    '''

    #-------------------------------------------------------------------------#
    # Class properties
    
    di_value = { "string": "", "double": 0., "int": int(0) }

    #-------------------------------------------------------------------------#
    # Constructor and instance properties
    
    def __init__(self, nodes=0, g=None, directed=True):
        self._edges = OrderedDict()
        self.directed = directed
        self._nattr = _NxNProperty(self)
        self._eattr = _NxEProperty(self)
        super(NxGraph,self).__init__(g)
        if g is not None:
            edges = glib_func["get_edges"](g)
            for i, edge in enumerate(edges):
                self._edges[tuple(edge)] = i
        elif nodes:
            self.add_nodes_from(range(nodes))

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
                if "vec" in value_type:
                    values = [ [] for _ in range(num_edges) ]
                else:
                    values = np.repeat(self.di_value[value_type], num_edges)
        elif len(values) != num_edges:
            raise InvalidArgument("'values' list must contain one element per \
edge in the graph.")
        for e, val in zip(self.edges_array,values):
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
        self._edges[(source, target)] = self.node_nb()
        self[source][target]['weight'] = weight
        if not self.directed:
            self.add_edge(target,source)
            self[target][source]['weight'] = weight
        return (source, target)

    def new_edges(self, edge_list, eprops=None):
        ''' Adds a list of connections to the graph '''
        n = self.node_nb()
        for i, edge in enumerate(edge_list):
            self._edges[tuple(edge)] = n + i
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

    def clear_all_edges(self):
        ''' Remove all connections in the graph '''
        self.remove_edges_from(self.edges_array)
        self._edges = OrderedDict()
        self._eattr.clear()

    def set_node_property(self):
        #@todo: do it...
        pass
    
    #-------------------------------------------------------------------------#
    # Getters
    
    def node_nb(self):
        return self.number_of_nodes()

    def edge_nb(self):
        return self.size()
    
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


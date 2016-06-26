#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" iGraph subclassing """

from collections import OrderedDict

import numpy as np
import scipy.sparse as ssp

import nngt.globals
from nngt.globals import BWEIGHT
from nngt.lib import InvalidArgument
from .base_graph import BaseGraph, BaseProperty



#-----------------------------------------------------------------------------#
# Properties
#------------------------
#

class _GtNProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __getitem__(self, name):
        return np.array(self.parent.vertex_properties[name].a)

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

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __getitem__(self, name):
        '''
        Return the attributes of an edge or a list of edges.
        '''
        if isinstance(name, str):
            return np.array(self.parent.edge_properties[name].a)
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

    def edge_prop(self, name, values, edges=None):
        if edges is None or len(edges) == self.parent.edge_nb():
            self[name] = values
        else:
            edges = self.parent._edges.keys()
            if len(values) == len(edges):
                for e, val in zip(edges, values):
                    self.parent.edge_properties[name][e] = val
            else:
                raise ValueError("A list or a np.array with one entry per \
edge in `edges` is required")

    def new_ea(self, name, value_type, values=None, val=None):
        eprop = self.parent.new_edge_property(value_type, values, val)
        self.parent.edge_properties[name] = eprop
        self.stored[name] = value_type


#-----------------------------------------------------------------------------#
# Graph
#------------------------
#

class _GtGraph(BaseGraph):
    
    '''
    Subclass of :class:`graph_tool.Graph` that (with 
    :class:`~nngt.core._SnapGraph`) unifies the methods to work with either
    `graph-tool` or `SNAP`.
    '''

    nattr_class = _GtNProperty
    eattr_class = _GtEProperty
    
    #-------------------------------------------------------------------------#
    # Constructor and instance properties        

    def __init__(self, nodes=0, g=None, weighted=True, directed=True,
                 prune=False, vorder=None, **kwargs):
        '''
        @todo: document that
        see :class:`graph_tool.Graph`'s constructor '''
        super(_GtGraph, self).__init__(g=g, directed=True, prune=prune,
                                       vorder=vorder)
        self._edges = OrderedDict()
        self._nattr = _GtNProperty(self)
        self._eattr = _GtEProperty(self)
        self._directed = directed
        self._weighted = weighted
        if weighted:
            self.new_edge_attribute("weight", "double")
        if g is None:
            self.add_vertex(nodes)
        else:
            if g.__class__ is nngt.globals.config["graph"]:
                edges = nngt.globals.analyze_graph["get_edges"](g)
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
        if n == 1:
            return node
        else:
            return tuple(node)

    def new_edge(self, source, target, attributes={}):
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
            
        Returns
        -------
        The new connection.
        '''
        if self._weighted and "weight" not in attributes:
            attributes["weight"] = 1.
        self._edges[(source, target)] = self.edge_nb()
        connection = super(_GtGraph, self).add_edge(source, target,
                                                   add_missing=True)
        for key, val in attributes:
            if key in self.edge_properties:
                self.edge_properties[key][connection] = val
            else:
                raise InvalidArgument("Unknown edge property `" + key + "'.")
        if not self._directed:
            self._edges[(source, target)] = self.edge_nb() + 1
            c2 = super(_GtGraph, self).add_edge(target, source)
            for key, val in attributes:
                if key in self.edge_properties:
                    self.edge_properties[key][c2] = val
                else:
                    raise InvalidArgument("Unknown edge property `"+ key +"'.")
        return connection

    def new_edges(self, edge_list, attributes={}):
        '''
        Add a list of edges to the graph.
        
        Parameters
        ----------
        edge_list : list of 2-tuples or np.array of shape (edge_nb, 2)
            List of the edges that should be added as tuples (source, target)
        attributes : :class:`dict`, optional (default: ``{}``)
            Dictionary containing optional edge properties. If the graph is
            weighted, defaults to ``{"weight": ones}``, where ``ones`` is an
            array the same length as the `edge_list` containing a unit weight
            for each connection (synaptic strength in NEST).
        
        warning ::
            For now attributes works only when adding edges for the first time
            (i.e. adding edges to an empty graph).
            
        @todo: add example, check the edges for self-loops and multiple edges
        '''
        e = self.edge_nb()
        edge_generator = ( e for e in edge_list )
        edge_list = np.array(edge_list)
        if self._weighted and "weight" not in attributes:
            attributes["weight"] = np.repeat(1., len(edge_list))
        if not self._directed:
            edge_list = np.concatenate((edge_list, edge_list[:,::-1]))
            for key, val in attributes.items():
                attributes[key] = np.concatenate((val, val))
        for i, edge in enumerate(edge_list):
            self._edges[tuple(edge)] = e + i
        super(_GtGraph, self).add_edge_list(edge_list)
        for key, val in attributes.items():
            self._eattr.edge_prop(key, val, edge_list)
        return edge_generator
    
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
    
    def degree_list(self, node_list=None, deg_type="total", use_weights=False):
        w = 1.
        if not self._directed:
            deg_type = "total"
            w = 0.5
        if node_list is None:
            node_list = slice(0,self.num_vertices()+1)
        if "weight" in self.edge_properties.keys() and use_weights:
            print("plop")
            return w*self.degree_property_map(deg_type,
                            self.edge_properties["weight"]).a[node_list]
        else:
            print("yeah")
            return w*np.array(self.degree_property_map(deg_type).a[node_list])

    def betweenness_list(self, btype="both", use_weights=False, as_prop=False,
                         norm=True):
        if self.edge_nb():
            w_p = None
            if "weight" in self.edge_properties.keys() and use_weights:
                w_p = self.edge_properties[BWEIGHT]
            tpl = nngt.globals.analyze_graph["betweenness"](self, weight=w_p, norm=norm)
            lst_return = []
            if btype == "node":
                return tpl[0] if as_prop else np.array(tpl[0].a)
            elif btype == "edge":
                return tpl[1] if as_prop else np.array(tpl[1].a)
            else:
                return ( tpl[0], tpl[1] if as_prop
                         else np.array(tpl[0].a), np.array(tpl[1].a) )
        else:
            if as_prop:
                return None, None
            else:
                return np.array([]), np.array([])

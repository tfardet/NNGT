#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Networkx subclassing """

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
        for e in iter(self.parent._edges.keys()):
            lst.append(self.parent.edge[e[0]][e[1]][name])
        return np.array(lst)

    def __setitem__(self, name, value):
        size = self.parent.edge_nb()
        if len(value) == size:
            for i,e in enumerate(self.parent._edges.keys()):
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


#-----------------------------------------------------------------------------#
# Graph
#------------------------
#

class _NxGraph(BaseGraph):

    '''
    Subclass of networkx Graph
    '''

    nattr_class = _NxNProperty
    eattr_class = _NxEProperty

    #-------------------------------------------------------------------------#
    # Class properties
    
    di_value = { "string": "", "double": 0., "int": int(0) }

    #-------------------------------------------------------------------------#
    # Constructor and instance properties
    
    def __init__(self, nodes=0, g=None, directed=True, weighted=False):
        self._edges = OrderedDict()
        self._directed = directed
        self._weighted = _weighted
        self._nattr = _NxNProperty(self)
        self._eattr = _NxEProperty(self)
        super(_NxGraph,self).__init__(g)
        if g is not None:
            edges = nngt.globals.analyze_graph["get_edges"](g)
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
        num_edges = self.number_of_edges()
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
        if len(tpl_new_nodes) == 1:
            return tpl_new_nodes[0]
        else:
            return tpl_new_nodes

    def new_edge(self, source, target, weight=1.):
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
        if not self._directed:
            self.add_edge(target,source)
            self[target][source]['weight'] = weight
        return (source, target)

    def new_edges(self, edge_list, eprops=None):
        ''' Adds a list of connections to the graph '''
        n = self.edge_nb()
        for i, edge in enumerate(edge_list):
            self._edges[tuple(edge)] = n + i
        if eprops is not None:
            for attr in iter(eprops.keys()):
                arr = eprops[attr]
                edges = ( (tpl[0],tpl[i][1], arr[i])
                          for i, tpl in enumerate(edge_list) )
                self.add_weighted_edges_from(edges, weight=attr)
                if not self._directed:
                    self.add_weighted_edges_from(
                        np.array(edge_list)[:,[1,0,2]], weight=attr)
        else:
            self.add_edges_from(edge_list, eprops)
            if not self._directed:
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
    
    def degree_list(self, node_list=None, deg_type="total", use_weights=False):
        weight = 'weight' if use_weights else None
        di_deg = None
        if deg_type == 'total':
            di_deg = self.degree(node_list, weight=weight)
        elif deg_type == 'in':
            di_deg = self.in_degree(node_list, weight=weight)
        else:
            di_deg = self.out_degree(node_list, weight=weight)
        return np.array(tuple(di_deg.values()))

    def betweenness_list(self, btype="both", use_weights=False, **kwargs):
        di_nbetw, di_ebetw = None, None
        w = BWEIGHT if use_weights else None
        if btype in ("both", "node"):
            di_nbetw = nngt.globals.config["library"].betweenness_centrality(self,
                                                                weight=BWEIGHT)
        if btype in ("both", "edge"):
            di_ebetw = nngt.globals.config["library"].edge_betweenness_centrality(self,
                                                                weight=BWEIGHT)
        else:
            di_nbetw = nngt.globals.config["library"].betweenness_centrality(self)
            di_ebetw = nngt.globals.config["library"].edge_betweenness_centrality(self)
        if btype == "node":
            return np.array(tuple(di_nbetw.values()))
        elif btype == "edge":
            return np.array(tuple(di_ebetw.values()))
        else:
            return ( np.array(tuple(di_nbetw.values())),
                     np.array(tuple(di_ebetw.values())) )


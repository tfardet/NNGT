#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Networkx subclassing """

from collections import OrderedDict

import numpy as np
import scipy.sparse as ssp

import nngt.globals
from nngt.globals import BWEIGHT
from nngt.lib import InvalidArgument
from nngt.lib.graph_helpers import _set_edge_attr
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
        lst = [self.parent().node[i][name]
               for i in range(self.parent().node_nb())]
        return np.array(lst)

    def __setitem__(self, name, value):
        if name in self:
            if len(value) == size:
                for i in range(self.parent().number_of_nodes()):
                    self.parent().node[i][name] = value[i]
            else:
                raise ValueError("A list or a np.array with one entry per \
node in the graph is required")
        else:
            raise InvalidArgument("Attribute does not exist yet, use \
set_attribute to create it.")

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
            values = np.repeat(val, self.parent().number_of_nodes())
        # store name and value type in the dict
        super(_NxNProperty,self).__setitem__(name, value_type)
        # store the real values in the attribute
        self[name] = values
        
class _NxEProperty(BaseProperty):

    ''' Class for generic interactions with edge properties (networkx)  '''

    def __getitem__(self, name):
        lst = []
        for e in iter(self.parent().edges()):
            lst.append(self.parent().edge[e[0]][e[1]][name])
        return np.array(lst)

    def __setitem__(self, name, value):
        if name in self:
            size = self.parent().number_of_edges()
            if len(value) == size:
                for i, e in enumerate(self.parent().edges()):
                    self.parent().edge[e[0]][e[1]][name] = value[i]
            else:
                raise ValueError("A list or a np.array with one entry per \
edge in the graph is required")
        else:
            raise InvalidArgument("Attribute does not exist yet, use \
set_attribute to create it.")

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
            values = np.repeat(val, self.parent().number_of_edges())
        # store name and value type in the dict
        super(_NxEProperty,self).__setitem__(name, value_type)
        # store the real values in the attribute
        self[name] = values


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
        self._directed = directed
        self._weighted = weighted
        self._nattr = _NxNProperty(self)
        self._eattr = _NxEProperty(self)
        super(_NxGraph, self).__init__(g)
        if g is not None:
            edges = nngt.globals.analyze_graph["get_edges"](g)
        elif nodes:
            self.add_nodes_from(range(nodes))

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
        if isinstance(edge[0], int):
            return self[edge[0]][edge[1]]["eid"]
        elif hasattr(edge[0], "__len__"):
            return [self[e[0]][e[1]]["eid"] for e in edge]
        else:
            raise AttributeError("`edge` must be either a 2-tuple of ints or\
an array of 2-tuples of ints.")
    
    @property
    def edges_array(self):
        ''' Edges of the graph, sorted by order of creation, as an array of
        2-tuple. '''
        return np.array(self.edges())
    
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
                values = np.repeat(val, num_edges)
            else:
                if "vec" in value_type:
                    values = [ [] for _ in range(num_edges) ]
                else:
                    values = np.repeat(self.di_value[value_type], num_edges)
        elif len(values) != num_edges:
            raise InvalidArgument("'values' list must contain one element per \
edge in the graph.")
        for e, val in zip(self.edges(), values):
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

    def new_edge(self, source, target, attributes=None):
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
        if attributes is None:
            attributes = {}
        _set_edge_attr(self, [(source, target)], attributes)
        for attr in attributes:
            if "_corr" in attr:
                raise NotImplementedError("Correlated attributes are not "
                                          "available with networkx.")
        if self._weighted and "weight" not in attributes:
            attributes["weight"] = 1.
        self.add_edge(source, target)
        self[source][target]["eid"] = self.number_of_edges()
        for key, val in attributes.items:
            self[source][target][key] = val
        if not self._directed:
            self.add_edge(target,source)
            self[source][target]["eid"] = self.number_of_edges()
            for key, val in attributes.items:
                self[target][source][key] = val
        return (source, target)

    def new_edges(self, edge_list, attributes=None):
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
        if attributes is None:
            attributes = {}
        _set_edge_attr(self, edge_list, attributes)
        for attr in attributes:
            if "_corr" in attr:
                raise NotImplementedError("Correlated attributes are not "
                                          "available with networkx.")
        initial_edges = self.number_of_edges()
        if not self._directed:
            elist_tmp = np.zeros((2*len(edge_list), 2), dtype=np.uint)
            i = 0
            for e, e_reversed in zip(edge_list, edge_list[:,::-1]):
                elist_tmp[i:i+2,:] = (e, e_reversed)
                i += 1
            edge_list = elist_tmp
            for key, val in attributes.items():
                attributes[key] = np.repeat(val, 2)
        edge_generator = ( e for e in edge_list )
        edge_list = np.array(edge_list)
        if self._weighted and "weight" not in attributes:
            attributes["weight"] = np.repeat(1., edge_list.shape[0])
        attributes["eid"] = np.arange(
            initial_edges, initial_edges + len(edge_list))
        for i, (u,v) in enumerate(edge_list):
            if u not in self.succ:
                self.succ[u] = self.adjlist_dict_factory()
                self.pred[u] = self.adjlist_dict_factory()
                self.node[u] = {}
            if v not in self.succ:
                self.succ[v] = self.adjlist_dict_factory()
                self.pred[v] = self.adjlist_dict_factory()
                self.node[v] = {}
            datadict = self.adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update({key: val[i] for key, val in attributes.items()})
            self.succ[u][v] = datadict
            self.pred[v][u] = datadict
        return edge_generator

    def clear_all_edges(self):
        ''' Remove all connections in the graph '''
        self.remove_edges_from(self.edges())
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
        if mode == "all":
            neighbours = self.successors(node)
            neighbours.extend(self.predecessors(node))
            return neighbours
        elif mode == "in":
            return self.predecessors(node)
        elif mode == "out":
            return self.successors(node)
        else:
            raise ArgumentError('''Invalid `mode` argument {}; possible values
                                are "all", "out" or "in".'''.format(mode))

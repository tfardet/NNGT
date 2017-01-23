#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" iGraph subclassing """

from collections import OrderedDict

import numpy as np
import scipy.sparse as ssp

import nngt.globals
from nngt.globals import BWEIGHT
from nngt.lib import InvalidArgument, nonstring_container
from .base_graph import BaseGraph, BaseProperty



#-----------------------------------------------------------------------------#
# Properties
#------------------------
#

class _IgNProperty(BaseProperty):

    '''
    Class for generic interactions with nodes properties (igraph)
    '''

    def __getitem__(self, name):
        return np.array(self.parent()().vs[name])

    def __setitem__(self, name, value):
        if name in self:
            if len(value) == size:
                self.parent()().vs[name] = value
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
            values = np.repeat(val, self.parent()().vcount())
        # store name and value type in the dict
        super(_IgNProperty,self).__setitem__(name, value_type)
        # store the real values in the attribute
        self[name] = values

class _IgEProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (igraph)  '''

    def __getitem__(self, name):
        return np.array(self.parent().es[name])

    def __setitem__(self, name, value):
        if name in self:
            size = self.parent().edge_nb()
            if len(value) == size:
                self.parent().es[name] = value
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
            values = np.repeat(val, self.parent().ecount())
        # store name and value type in the dict
        super(_IgEProperty,self).__setitem__(name, value_type)
        # store the real values in the attribute
        self[name] = values


#-----------------------------------------------------------------------------#
# Graph
#------------------------
#

class _IGraph(BaseGraph):

    '''
    Subclass of :class:`igraph.Graph`.
    '''
    
    nattr_class = _IgNProperty
    eattr_class = _IgEProperty

    #-------------------------------------------------------------------------#
    # Constructor and instance properties
    
    def __init__(self, nodes=0, g=None, directed=True, weighted=False):
        self._nattr = _IgNProperty(self)
        self._eattr = _IgEProperty(self)
        self._weighted = weighted
        self._directed = directed
        if g is None:
            super(_IGraph,self).__init__(n=nodes, directed=True)
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
            super(_IGraph, self).__init__(n=nodes, vertex_attrs=di_node_attr)
            lst_edges = nngt.globals.analyze_graph["get_edges"](g)
            self.new_edges(lst_edges, attributes=di_edge_attr)

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
            return self.get_eid(*edge)
        elif hasattr(edge[0], "__len__"):
            return self.get_eids(edge)
        else:
            raise AttributeError("`edge` must be either a 2-tuple of ints or\
an array of 2-tuples of ints.")
    
    @property
    def edges_array(self):
        ''' Edges of the graph, sorted by order of creation, as an array of
        2-tuple. '''
        return np.array([(e.source, e.target) for e in self.es])
    
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
        if self._weighted and "weight" not in attributes:
            attributes["weight"] = 1.
        super(_IGraph, self).add_edge(source, target, **attributes)
        if not self._directed:
            super(_IGraph, self).add_edge(target, source, **attributes)
        return (source, target)

    def new_edges(self, edge_list, attributes=None):
        '''
        Add a list of edges to the graph.
        
        Parameters
        ----------
        edge_list : list of 2-tuples or np.array of shape (edge_nb, 2)
            List of the edges that should be added as tuples (source, target)
        attributes : dict, optional (default: ``None``)
            Dictionary of the form ``{ "name": [], "values": [],
            "type": [] }``, containing the attributes of the new edges.
        
        warning ::
            For now attributes works only when adding edges for the first time
            (i.e. adding edges to an empty graph).
            
        @todo: add example, check the edges for self-loops and multiple edges
        
        Returns
        -------
        Returns new edges only.
        '''
        if attributes is None:
            attributes = {}
        initial_ecount = self.ecount()
        if not self._directed:
            edge_list = np.concatenate((edge_list, edge_list[:,::-1]))
            for key, val in attributes.items():
                attributes[key] = np.concatenate((val, val))
        first_eid = self.ecount()
        super(_IGraph, self).add_edges(edge_list)
        num_edges = self.ecount()
        # attributes
        if self._weighted and "weight" not in attributes:
            attributes["weight"] = np.repeat(1., num_edges)
        if attributes:
            elist0 = None #@todo: make elist supported and remove this
            # take care of classic attributes
            if "weight" in attributes:
                self.set_weights(weight=attributes["weight"], elist=elist0)
            if "delay" in attributes:
                self.set_delays(delay=attributes["delay"], elist=elist0)
            if "distance" in attributes:
                raise NotImplementedError("distance not implemented yet")
                #~ self.set_distances(elist=edge_list,
                                   #~ dlist=attributes["distance"])
            # take care of potential additional attributes
            if "names" in attributes:
                num_attr = len(attributes["names"])
                for i in range(num_attr):
                    v = attributes["values"]
                    if not nonstring_container(v):
                        v = np.repeat(v, self.ecount())
                    self._eattr.new_ea(attributes["names"][i],
                                       attributes["types"][i], values=v)
        return edge_list

    def new_edge_attribute(self, name, value_type, values=None, val=None):
        num_edges = self.ecount()
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
        self.es[name] = values
    
    def new_node_attribute(self, name, value_type, values=None, val=None):
        num_nodes = self.vcount()
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
        self.vs[name] = values

    def remove_edge(self, edge):
        raise NotImplementedError("This function has been removed because it \
            makes using edge properties too complicated")

    def remove_vertex(self, node, fast=False):
        raise NotImplementedError("This function has been removed because it \
            makes using node properties too complicated")

    def clear_all_edges(self):
        ''' Remove all connections in the graph. '''
        self.delete_edges(None)
        self._eattr.clear()
    
    #-------------------------------------------------------------------------#
    # Getters
    
    def node_nb(self):
        return self.vcount()

    def edge_nb(self):
        return self.ecount()
    
    def degree_list(self, node_list=None, deg_type="total", use_weights=False):
        deg_type = 'all' if deg_type == 'total' else deg_type
        if use_weights:
            return np.array(self.strength(node_list,mode=deg_type))
        else:
            return np.array(self.degree(node_list,mode=deg_type))

    def betweenness_list(self, btype="both", use_weights=False, norm=True,
                         **kwargs):
        n = self.vcount()
        e = self.ecount()
        ncoeff_norm = (n-1)*(n-2)
        ecoeff_norm = (e-1)*(e-2)/2.
        w, nbetw, ebetw = None, None, None
        if use_weights:
            w = self.es['bweight']
        if btype in ("both", "node"):
            nbetw = np.array(self.betweenness(weights=w))
        if btype in ("both", "edge"):
            ebetw = np.array(self.edge_betweenness(weights=w))
        if btype == "node":
            return nbetw/ncoeff_norm if norm else nbetw
        elif btype == "edge":
            return ebetw/ecoeff_norm if norm else ebetw
        else:
            return ( nbetw/ncoeff_norm, ebetw/ecoeff_norm if norm
                     else nbetw, ebetw )

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
            return self.neighbors(node, mode=3)
        elif mode == "in":
            return self.neighbors(node, mode=1)
        elif mode == "out":
            return self.neighbors(node, mode=2)
        else:
            raise ArgumentError('''Invalid `mode` argument {}; possible values
                                are "all", "out" or "in".'''.format(mode))

#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" iGraph subclassing """

from collections import OrderedDict

import numpy as np
import scipy.sparse as ssp

import nngt.globals
from nngt.globals import BWEIGHT
from nngt.lib import InvalidArgument
from nngt.lib.graph_helpers import _set_edge_attr
from nngt.lib.connect_tools import _unique_rows
from .base_graph import BaseGraph, BaseProperty



#-----------------------------------------------------------------------------#
# Properties
#------------------------
#

class _GtNProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __getitem__(self, name):
        return np.array(self.parent().vertex_properties[name].a)

    def __setitem__(self, name, value):
        if name in self:
            size = self.parent().num_vertices()
            if len(value) == size:
                self.parent().vertex_properties[name].a = np.array(value)
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
            values = np.repeat(val, self.parent().num_edges())
        if len(values) != self.parent().num_vertices():
            raise ValueError("A list or a np.array with one entry per \
edge in the graph is required")
        # store name and value type in the dict
        super(_GtNProperty, self).__setitem__(name, value_type)
        # store the real values in the attribute
        nprop = self.parent().new_node_property(value_type, values)
        self.parent().node_properties[name] = nprop

class _GtEProperty(BaseProperty):

    ''' Class for generic interactions with nodes properties (graph-tool)  '''

    def __getitem__(self, name):
        '''
        Return the attributes of an edge or a list of edges.
        '''
        if isinstance(name, str):
            return np.array(self.parent().edge_properties[name].a)
        elif hasattr(name[0], '__iter__'):
            di_eattr = {}
            for key in self.keys():
                di_eattr[key] = np.array(
                    [self.parent().edge_properties[key][e] for e in name])
            return di_eattr
        else:
            di_eattr = {}
            for key in self.keys():
                di_eattr[key] = self.parent().edge_properties[key][name]
            return di_eattr

    def __setitem__(self, name, value):
        if name in self:
            size = self.parent().num_edges()
            if len(value) == size:
                self.parent().edge_properties[name].a = np.array(value)
            else:
                raise ValueError("A list or a np.array with one entry per \
edge in the graph is required")
        else:
            raise InvalidArgument("Attribute does not exist yet, use \
set_attribute to create it.")
        self._num_values_set[name] = len(value)

    def set_property(self, name, values, edges=None):
        '''
        Set the edge property.
        
        Parameters
        ----------
        name : str
            Name of the edge property.
        values : array
            Values that should be set.
        edges : array-like, optional (default: None)
            Edges for which the value of the property should be set. If `edges`
            is not None, it must be an array of shape `(len(values), 2)`.
        '''
        num_edges = self.parent().num_edges()
        num_e = len(edges) if edges is not None else num_edges
        if num_e == num_edges:
            self[name] = values
        else:
            if num_e != len(values):
                raise ValueError("`edges` and `values` must have the same "
                                 "size; got respectively " + str(num_e) + \
                                 " and " + str(len(values)) + "entries.")
            if self._num_values_set[name] == num_edges - num_e:
                self.parent().edge_properties[name].a[-num_e:] = values
            else:
                for e, val in zip(edges, values):
                    gt_e = self.parent().edge(*e)
                    self.parent().edge_properties[name][gt_e] = val
        self._num_values_set[name] = num_edges

    def new_property(self, name, value_type, values=None, val=None):
        if values is None and val is None:
            self._num_values_set[name] = self.parent().num_edges()
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
            values = np.repeat(val, self.parent().num_edges())
        
        if len(values) != self.parent().num_edges():
            self._num_values_set[name] = 0
            raise ValueError("A list or a np.array with one entry per "
                             "edge in the graph is required")
        # store name and value type in the dict
        super(_GtEProperty, self).__setitem__(name, value_type)
        # store the real values in the attribute
        eprop = self.parent().new_edge_property(value_type, values)
        self.parent().edge_properties[name] = eprop


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
        self._nattr = _GtNProperty(self)
        self._eattr = _GtEProperty(self)
        self._directed = directed
        self._weighted = weighted
        super(_GtGraph, self).__init__(g=g, directed=True, prune=prune,
                                       vorder=vorder)
        #~ if weighted:
            #~ self.new_edge_attribute("weight", "double")
        if g is None:
            super(_GtGraph, self).add_vertex(nodes)
        else:
            if g.__class__ is nngt._config["graph"]:
                edges = nngt.analyze_graph["get_edges"](g)

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
            return self.edge_index[edge]
        elif hasattr(edge[0], "__len__"):
            idx = [self.edge_index[e] for e in edge]
            return idx
        else:
            raise AttributeError("`edge` must be either a 2-tuple of ints or\
an array of 2-tuples of ints.")
    
    @property
    def edges_array(self):
        '''
        Edges of the graph, sorted by order of creation, as an array of
        2-tuple.
        '''
        return np.array(
            [(int(e.source()), int(e.target())) for e in self.edges()])
    
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

    def new_edge(self, source, target, attributes=None, ignore=False):
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
        ignore : bool, optional (default: False)
            If set to True, ignore attempts to add an existing edge, otherwise
            raises an error.
            
        Returns
        -------
        The new connection.
        '''
        if attributes is None:
            attributes = {}
        # check that the edge does not already exist
        edge = self.edge(source, target)
        if edge is None:
            connection = super(_GtGraph, self).add_edge(source, target,
                                                        add_missing=True)
            _set_edge_attr(self, [(source, target)], attributes)
            for key, val in attributes:
                if key in self.edge_properties:
                    self.edge_properties[key][connection] = val[0]
                else:
                    raise InvalidArgument(
                        "Unknown edge property `" + key + "'.")
            if not self._directed:
                c2 = super(_GtGraph, self).add_edge(target, source)
                for key, val in attributes:
                    if key in self.edge_properties:
                        self.edge_properties[key][c2] = val[0]
                    else:
                        raise InvalidArgument(
                            "Unknown edge property `" + key + "'.")
        else:
            if not ignore:
                raise InvalidArgument("Trying to add existing edge.")
        return connection

    def new_edges(self, edge_list, attributes=None):
        '''
        Add a list of edges to the graph.
        
        .. warning ::
            This function currently does not check for duplicate edges!
        
        Parameters
        ----------
        edge_list : list of 2-tuples or np.array of shape (edge_nb, 2)
            List of the edges that should be added as tuples (source, target)
        attributes : :class:`dict`, optional (default: ``{}``)
            Dictionary containing optional edge properties. If the graph is
            weighted, defaults to ``{"weight": ones}``, where ``ones`` is an
            array the same length as the `edge_list` containing a unit weight
            for each connection (synaptic strength in NEST).
            
        @todo: add example, check the edges for self-loops and multiple edges
        '''
        if attributes is None:
            attributes = {}
        initial_edges = self.num_edges()
        if not isinstance(edge_list, np.ndarray):
            edge_list = np.array(edge_list)
        if not self._directed:
            recip_edges = edge_list[:,::-1]
            # slow but works
            unique = ~(recip_edges[..., np.newaxis]
                      == edge_list[..., np.newaxis].T).all(1).any(1)
            edge_list = np.concatenate((edge_list, recip_edges[unique]))
            for key, val in attributes.items():
                attributes[key] = np.concatenate((val, val[unique]))
        # create the edges
        super(_GtGraph, self).add_edge_list(edge_list)
        # prepare the default values for weight and delays if necessary
        _set_edge_attr(self, edge_list, attributes)
        # call parent function to set the attributes
        self.attr_new_edges(edge_list, attributes=attributes)
        return edge_list
    
    def clear_all_edges(self):
        super(_GtGraph, self).clear_edges()
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
            return w*self.degree_property_map(deg_type,
                            self.edge_properties["weight"]).a[node_list]
        else:
            return w*np.array(self.degree_property_map(deg_type).a[node_list])

    def betweenness_list(self, btype="both", use_weights=False, as_prop=False,
                         norm=True):
        if self.num_edges():
            w_p = None
            if "weight" in self.edge_properties.keys() and use_weights:
                w_p = self.edge_properties[BWEIGHT]
            tpl = nngt.analyze_graph["betweenness"](
                self, weight=w_p, norm=norm)
            lst_return = []
            if btype == "node":
                return tpl[0] if as_prop else np.array(tpl[0].a)
            elif btype == "edge":
                return tpl[1] if as_prop else np.array(tpl[1].a)
            else:
                return ( np.array(tpl[0], tpl[1]) if as_prop
                         else np.array(tpl[0].a), np.array(tpl[1].a) )
        else:
            if as_prop:
                return (None, None) if btype == "both" else None
            else:
                if btype == "both":
                    return np.array([]), np.array([])
                else:
                    return np.array([])

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
        v = self.vertex(node)
        if mode == "all":
            return tuple(v.all_neighbours())
        elif mode == "in":
            return tuple(v.in_neighbours())
        elif mode == "out":
            return tuple(v.out_neighbours())
        else:
            raise ArgumentError('''Invalid `mode` argument {}; possible values
                                are "all", "out" or "in".'''.format(mode))

    #-------------------------------------------------------------------------
    # Prevent users from calling graph_tool functions

    def add_vertex(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool functions for vertex "
                           "creation have been disabled.")

    def add_edge(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool functions for edge "
                           "creation have been disabled.")

    def add_edge_list(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool functions for edge "
                           "creation have been disabled.")

    def remove_vertex(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool functions for vertex "
                           "deletion have been disabled.")

    def clear_vertex(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool functions for vertex "
                           "deletion have been disabled.")

    def purge_vertices(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool functions for vertex "
                           "deletion have been disabled.")

    def remove_edge(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool functions for edge "
                           "deletion have been disabled.")

    def clear_edges(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool functions for edge "
                           "deletion have been disabled.")

    def purge_edges(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool functions for edge "
                           "deletion have been disabled.")

    def clear(self, *args, **kwargs):
        raise RuntimeError("Intrinsic graph_tool `clear` function has been "
                           "disabled.")

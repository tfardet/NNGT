#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" IO tools for NNGT """

import numpy as np
import scipy.sparse as ssp

from nngt.globals import glib_data, glib_func
from nngt.lib.errors import InvalidArgument


__all__ = [
    "load_from_file",
    "save_to_file"
]


#-----------------------------------------------------------------------------#
# Saving tools
#------------------------
#

def _neighbour_list(graph, delimiter, secondary, attributes):
    '''
    Generate a string containing the neighbour list of the graph as well as a
    dict containing the notifiers as key and the associated values.
    @todo: speed this up!
    '''
    lst_neighbours = list( graph.adjacency_matrix().tolil().rows)
    for v1 in range(graph.node_nb()):
        for i,v2 in enumerate(lst_neighbours[v1]):
            str_edge = str(v2)
            for attr in attributes:
                str_edge += "{}{}".format( secondary,
                                           graph.attributes((v1,v2), attr))
            lst_neighbours[v1][i] = str_edge
        lst_neighbours[v1] = "{}{}{}".format( v1, delimiter, delimiter.join(
                                              lst_neighbours[v1]) )
    str_neighbours = "\n".join(lst_neighbours)
    return str_neighbours

def _edge_list(graph, delimiter, secondary, attributes):
    ''' Generate a string containing the edge list and their properties. '''
    edges = graph.edges_array
    lst_edges = []
    for e in edges:
        str_edge = "{}{}{}".format(e[0], secondary, e[1])
        edge = tuple(e)
        for attr in attributes:
            str_edge += "{}{}".format(delimiter, graph.attributes(edge, attr))
        lst_edges.append(str_edge)
    str_edges = "\n".join(lst_edges)
    return str_edges

def _dot(graph, attributes, **kwargs):
    pass

def _gml(graph, attributes, **kwargs):
    pass

def _xml(graph, attributes, **kwargs):
    pass

def _gt(graph, attributes, **kwargs):
    pass


#-----------------------------------------------------------------------------#
# Loading tools
#------------------------
#

def _format_notif(notif_name, notif_val):
    if notif_name == "attributes":
        return notif_val[1:-1].split(", ")
    elif notif_name == "size":
        return int(notif_val)
    elif notif_name == "directed":
        return False if notif_val == "False" else True
    else:
        return notif_val    

def _get_notif(filegraph, notifier):
    di_notif = {}
    b_readnotif = True
    while b_readnotif:
        notif = filegraph.readline()
        b_readnotif = True if notif.startswith(notifier) else False
        if b_readnotif:
            idx_eq = notif.find("=")
            notif_name = notif[:idx_eq]
            notif_val = notif[idx_eq+1:]
            di_notif[notif_name] = _format_notif(notif_name, notif_val)
    return di_notif

def _gen_convert(atributes, attr_type):
    '''
    Generate a conversion dictionary that associates the right type to each
    attribute
    '''
    
    

def _get_edge_neighbour(line, attributes, delimiter, secondary, edges,
                        di_attributes, di_convert):
    source = int(line[0])
    neighbours = line[2:].split(delimiter)
    for stub in neighbours:
        target = int(stub[0])
        edges.append((source, target)
        attr_val = stub.split(secondary)[1:]
        for name,val in zip(attributes, attr_val):
            if val[0].isdigit() and '.' in val:
                di_attributes[name].append(di_convert[name](val))
            elif
            
    
    


#-----------------------------------------------------------------------------#
# Formatting
#------------------------
#

di_get_edge = {
    "neighbour": _get_edge_neighbour
}

di_format = {
    "neighbour":_neighbour_list,
    "edge_list": _edge_list
}


#-----------------------------------------------------------------------------#
# Import
#------------------------
#

def load_from_file(filename, format="neighbour", delimiter=" ", secondary=";",
                   attributes=[], notifier="@"):
    '''
    Import a saved graph as a (set of) :class:`~scipy.sparse.csr_matrix` from
    a file.
    @todo: implement population and shape loading, implement gml, dot, xml, gt

    Parameters
    ----------
    filename: str
        The path to the file.
    format : str, optional (default: "neighbour")
        The format used to save the graph. Supported formats are: "neighbour"
        (neighbour list, default if format cannot be deduced automatically),
        "ssp" (scipy.sparse), "edge_list" (list of all the edges in the graph,
        one edge per line, represented by a ``source target``-pair), "gml"
        (gml format, default if `filename` ends with '.gml'), "graphml"
        (graphml format, default if `filename` ends with '.graphml' or '.xml'),
        "dot" (dot format, default if `filename` ends with '.dot'), "gt" (only
        when using `graph_tool`<http://graph-tool.skewed.de/>_ as library,
        detected if `filename` ends with '.gt').
    delimiter : str, optional (default " ")
        Delimiter used to separate inputs in the case of custom formats (namely
        "neighbour" and "edge_list")
    secondary : str, optional (default: ";")
        Secondary delimiter used to separate attributes in the case of custom
        formats.
    attributes : list, optional (default: [])
        List of names for the attributes present in the file. If a `notifier`
        is present in the file, names will be deduced from it; otherwise the
        attributes will be numbered.
    notifier : str, optional (default: "@")
        Symbol specifying the following as meaningfull information. Relevant
        information are formatted ``@info_name=info_value``, where
        ``info_name`` is in ("attributes", "directed", "name", "size") and
        associated ``info_value``s are of type (``list``, ``bool``, ``str``,
        ``int``).
        Additional notifiers are ``@type=SpatialGraph/Network/SpatialNetwork``,
        which must be followed by the relevant notifiers among ``@shape``,
        ``@population``, and ``@graph``.

    Returns
    -------
    edges : list of 2-tuple
        Edges of the graph.
    di_attributes : dict
        Dictionary containing the attribute name as key and its value as a
        list sorted in the same order as `edges`.
    '''
    lst_lines, di_notif = None, None
    with open(filename, "r") as filegraph:
        lst_lines = [ line for line in filegraph.readlines() ]
        # notifier lines
        di_notif = _get_notif(lst_lines, notifier)
        # data
        lst_lines = lst_lines[len(di_notif)::-1]
    # make edges and attributes
    edges = []
    di_attributes = { name: [] for name in di_notif["attributes"] }
    di_convert = _gen_convert(lst_lines, di_notif["attributes"])
    line = None
    while lst_lines:
        line = lst_lines.pop()
        if line and not line.startswith(notifier):
            di_get_edge[format]( line, di_notif["attributes"], delimiter,
                                 secondary, edges, di_attributes, di_convert )
        else:
            break
    return edges, di_attributes
        


#-----------------------------------------------------------------------------#
# Save graph
#------------------------
#

def save_to_file(graph, filename, format="auto", delimiter=" ",
                 secondary=";", attributes=None, notifier="@"):
    '''
    Import a saved graph as a (set of) :class:`~scipy.sparse.csr_matrix` from
    a file.
    @todo: implement population and shape saving, implement gml, dot, xml, gt

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph to save.
    filename: str
        The path to the file.
    format : str, optional (default: "auto")
        The format used to save the graph. Supported formats are: "neighbour"
        (neighbour list, default if format cannot be deduced automatically),
        "ssp" (scipy.sparse), "edge_list" (list of all the edges in the graph,
        one edge per line, represented by a ``source target``-pair), "gml"
        (gml format, default if `filename` ends with '.gml'), "graphml"
        (graphml format, default if `filename` ends with '.graphml' or '.xml'),
        "dot" (dot format, default if `filename` ends with '.dot'), "gt" (only
        when using `graph_tool`<http://graph-tool.skewed.de/>_ as library,
        detected if `filename` ends with '.gt').
    delimiter : str, optional (default " ")
        Delimiter used to separate inputs in the case of custom formats (namely
        "neighbour" and "edge_list")
    secondary : str, optional (default: ";")
        Secondary delimiter used to separate attributes in the case of custom
        formats.
    attributes : list, optional (default: ``None``)
        List of names for the edge attributes present in the graph that will be
        saved to disk; by default (``None``), all attributes will be saved.
    notifier : str, optional (default: "@")
        Symbol specifying the following as meaningfull information. Relevant
        information are formatted ``@info_name=info_value``, with
        ``info_name`` in ("attributes", "attr_types", "directed", "name",
        "size").
        Additional notifiers are ``@type=SpatialGraph/Network/SpatialNetwork``,
        which are followed by the relevant notifiers among ``@shape``,
        ``@population``, and ``@graph`` to separate the sections.

    warning ::
        All formats except 'neighbour', 'edge_list' and 'ssp' lead to dataloss
        if your graph is a subclass of :class:`~nngt.SpatialGraph` or
        :class:`~nngt.Network` (the :class:`~nngt.Shape` and
        :class:`~nngt.NeuralPop` attributes will not be saved).
    '''
    # checks
    if delimiter == secondary:
        raise InvalidArgument("`delimiter` and `secondary` strings must be \
different.")
    if notifier == delimiter or notifier == secondary:
        raise InvalidArgument("`notifier` string should differ from \
`delimiter` and `secondary`.")
    # data
    if attributes is None:
        attributes = [ a for a in graph.attributes() if a != "bweight" ]
    di_notifiers = {
        "directed": graph.is_directed(),
        "attributes": attributes,
        "attr_types": raise Error,
        "name": graph.get_name(),
        "size": graph.node_nb()
    }
    if format == "auto":
        if filename.endswith('.gml'):
            format = 'gml'
        if filename.endswith('.graphml') or filename.endswith('.xml'):
            format = 'graphml'
        elif filename.endswith('.dot'):
            format = 'dot'
        elif filename.endswith('gt') and glib_data["library"] == "graph_tool":
            format = 'gt'
        else:
            format = 'neighbour'
    # generate string and write to file
    str_graph = di_format[format](graph, delimiter=delimiter,
                                  secondary=secondary, attributes=attributes)
    with open(filename, "w") as f_graph:
        for key,val in di_notifiers.iteritems():
            f_graph.write("{}{}={}\n".format(notifier, key,val))
        f_graph.write("\n")
        f_graph.write(str_graph)
    

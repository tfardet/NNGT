#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" IO tools for NNGT """

import numpy as np
import scipy.sparse as ssp

import nngt
from nngt.lib import InvalidArgument


__all__ = [
    "as_string",
    "load_from_file",
    "save_to_file"
]


#-----------------------------------------------------------------------------#
# Saving tools
#------------------------
#

def _get_format(format, filename):
    if format == "auto":
        if filename.endswith('.gml'):
            format = 'gml'
        if filename.endswith('.graphml') or filename.endswith('.xml'):
            format = 'graphml'
        elif filename.endswith('.dot'):
            format = 'dot'
        elif ( filename.endswith('gt') and
               nngt._config["graph_library"] == "graph_tool" ):
            format = 'gt'
        else:
            format = 'neighbour'
    return format


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
    if notif_name in ("attributes", "attr_types"):
        lst = notif_val[1:-1].split(", ")
        return [ val.strip("'\"") for val in lst ]
    elif notif_name == "size":
        return int(notif_val)
    elif notif_name == "directed":
        return False if notif_val == "False" else True
    else:
        return notif_val    


def _get_notif(lines, notifier):
    di_notif = { "attributes": [], "attr_types": [], "name": "LoadedGraph"}
    for line in lines:
        if line.startswith(notifier):
            idx_eq = line.find("=")
            notif_name = line[len(notifier):idx_eq]
            notif_val = line[idx_eq+1:]
            di_notif[notif_name] = _format_notif(notif_name, notif_val)
        else:
            break
    return di_notif


def _to_list(string):
    delimiters = (';', ',', ' ', '\t')
    count = -np.Inf
    chosen = -1
    for i, delim in enumerate(delimiters):
        current = string.count(delim)
        if count < current:
            count = current
            chosen = delimiters[i]
    return string.split(chosen)


def _gen_convert(attributes, attr_types):
    '''
    Generate a conversion dictionary that associates the right type to each
    attribute
    '''
    di_convert = {}
    for attr,attr_type in zip(attributes, attr_types):
        if attr_type in ("double", "float", "real"):
            di_convert[attr] = float
        elif attr_type in ("str", "string"):
            di_convert[attr] = str
        elif attr_type in ("int", "integer"):
            di_convert[attr] = int
        elif attr_type in ("lst", "list", "tuple", "array"):
            di_convert[attr] = _to_list
        else:
            raise TypeError("Invalid attribute type.")
    return di_convert


def _get_edges_neighbour(line, attributes, delimiter, secondary, edges,
                         di_attributes, di_convert):
    '''
    Add edges and attributes to `edges` and `di_attributes` for the "neighbour"
    format.
    '''
    source = int(line[0])
    len_first_delim = line.find(delimiter)+1
    if len_first_delim:
        neighbours = line[len_first_delim:].split(delimiter)
        for stub in neighbours:
            target = int(stub[0])
            edges.append((source, target))
            attr_val = stub.split(secondary)[1:]
            for name,val in zip(attributes, attr_val):
                di_attributes[name].append(di_convert[name](val))


def _get_edges_elist(line, attributes, delimiter, secondary, edges,
                     di_attributes, di_convert):
    '''
    Add edges and attributes to `edges` and `di_attributes` for the "neighbour"
    format.
    '''
    data = line.split(delimiter)
    source, target = int(data[0]), int(data[1])
    edges.append((source, target))
    if len(data) > 2:
        for name,val in zip(attributes, data[2:]):
            di_attributes[name].append(di_convert[name](val))
    

#-----------------------------------------------------------------------------#
# Formatting
#------------------------
#

di_get_edges = {
    "neighbour": _get_edges_neighbour,
    "edge_list": _get_edges_elist
}

di_format = {
    "neighbour": _neighbour_list,
    "edge_list": _edge_list
}


#-----------------------------------------------------------------------------#
# Import
#------------------------
#

def load_from_file(filename, format="neighbour", delimiter=" ", secondary=";",
                   attributes=[], notifier="@", ignore="#"):
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
    ignore : str, optional (default: "#")
        Ignore lines starting with the `ignore` string.

    Returns
    -------
    edges : list of 2-tuple
        Edges of the graph.
    di_attributes : dict
        Dictionary containing the attribute name as key and its value as a
        list sorted in the same order as `edges`.
    pop : :class:`~nngt.NeuralPop`
        Population (``None`` if not present in the file).
    shape : :class:`~nngt.geometry.Shape`
        Shape of the graph (``None`` if not present in the file).
    '''
    lst_lines, di_notif, pop, shape = None, None, None, None
    format = _get_format(format, filename)
    with open(filename, "r") as filegraph:
        lst_lines = [ line.strip() for line in filegraph.readlines() ]
    # notifier lines
    di_notif = _get_notif(lst_lines, notifier)
    # data
    lst_lines = lst_lines[::-1][:-len(di_notif)]
    while not lst_lines[-1] or lst_lines[-1].startswith(ignore):
        lst_lines.pop()
    # make edges and attributes
    edges = []
    di_attributes = { name: [] for name in di_notif["attributes"] }
    di_convert = _gen_convert(di_notif["attributes"], di_notif["attr_types"])
    line = None
    while lst_lines:
        line = lst_lines.pop()
        if line and not line.startswith(notifier):
            di_get_edges[format]( line, di_notif["attributes"], delimiter,
                                  secondary, edges, di_attributes, di_convert )
        else:
            break
    return di_notif, edges, di_attributes, pop, shape


#-----------------------------------------------------------------------------#
# Save graph
#------------------------
#

def as_string(graph, format="neighbour", delimiter=" ", secondary=";",
              attributes=None, notifier="@", return_info=False):
    '''
    Full string representation of the graph.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph to save.
    format : str, optional (default: "auto")
        The format used to save the graph. Supported formats are: "neighbour"
        (neighbour list, default if format cannot be deduced automatically),
        "ssp" (:mod:`scipy.sparse`), "edge_list" (list of all the edges in the 
        graph, one edge per line, represented by a ``source target``-pair), 
        "gml" (gml format, default if `filename` ends with '.gml'), "graphml"
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

    Returns
    -------
    str_graph : string
        The full graph representation as a string.
    '''
    # checks
    if delimiter == secondary:
        raise InvalidArgument("`delimiter` and `secondary` strings must be "
                              "different.")
    if notifier == delimiter or notifier == secondary:
        raise InvalidArgument("`notifier` string should differ from "
                              "`delimiter` and `secondary`.")
    # data
    if attributes is None:
        attributes = [ a for a in graph.attributes() if a != "bweight" ]
    di_notifiers = {
        "directed": graph._directed,
        "attributes": attributes,
        "attr_types": [graph.get_attribute_type(attr) for attr in attributes],
        "name": graph.get_name(),
        "size": graph.node_nb()
    }
    str_graph = di_format[format](graph, delimiter=delimiter,
                                  secondary=secondary, attributes=attributes)
    if return_info:
        return str_graph, di_notifiers
    else:
        return str_graph


def save_to_file(graph, filename, format="auto", delimiter=" ",
                 secondary=";", attributes=None, notifier="@"):
    '''
    Save a graph to file.
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
        For now, all formats lead to
        dataloss if your graph is a subclass of :class:`~nngt.SpatialGraph` or
        :class:`~nngt.Network` (the :class:`~nngt.geometry.Shape` and
        :class:`~nngt.NeuralPop` attributes will not be saved).
    '''
    format = _get_format(format, filename)
    str_graph, di_notif = as_string(graph, delimiter=delimiter, format=format,
                          secondary=secondary, attributes=attributes,
                          notifier=notifier,  return_info=True)
    with open(filename, "w") as f_graph:
        for key,val in iter(di_notif.items()):
            f_graph.write("{}{}={}\n".format(notifier, key,val))
        f_graph.write("\n")
        f_graph.write(str_graph)
    

#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" IO tools for NNGT """

import logging

import numpy as np
import scipy.sparse as ssp

import nngt
from nngt.lib import InvalidArgument
from nngt.lib.logger import _log_message
from .test_functions import graph_tool_check
from ..geometry import Shape, _shapely_support


logger = logging.getLogger(__name__)


# -- #
# IO #
# -- #

@graph_tool_check('2.22')
def load_from_file(filename, fmt="auto", separator=" ", secondary=";",
                   attributes=[], notifier="@", ignore="#"):
    '''
    Load the main properties (edges, attributes...) from a file.

    .. warning::
        To import a graph directly from a file, use the
        :func:`~nngt.Graph.from_file` classmethod.

    Parameters
    ----------
    filename: str
        The path to the file.
    fmt : str, optional (default: "neighbour")
        The format used to save the graph. Supported formats are: "neighbour"
        (neighbour list, default if format cannot be deduced automatically),
        "ssp" (scipy.sparse), "edge_list" (list of all the edges in the graph,
        one edge per line, represented by a ``source target``-pair), "gml"
        (gml format, default if `filename` ends with '.gml'), "graphml"
        (graphml format, default if `filename` ends with '.graphml' or '.xml'),
        "dot" (dot format, default if `filename` ends with '.dot'), "gt" (only
        when using `graph_tool`<http://graph-tool.skewed.de/>_ as library,
        detected if `filename` ends with '.gt').
    separator : str, optional (default " ")
        separator used to separate inputs in the case of custom formats (namely
        "neighbour" and "edge_list")
    secondary : str, optional (default: ";")
        Secondary separator used to separate attributes in the case of custom
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
    edges : list of 2-tuples
        Edges of the graph.
    di_attributes : dict
        Dictionary containing the attribute name as key and its value as a
        list sorted in the same order as `edges`.
    pop : :class:`~nngt.NeuralPop`
        Population (``None`` if not present in the file).
    shape : :class:`~nngt.geometry.Shape`
        Shape of the graph (``None`` if not present in the file).
    positions : array-like of shape (N, d)
        The positions of the neurons (``None`` if not present in the file).
    '''
    lst_lines, di_notif, pop, shape, positions = None, None, None, None, None
    fmt = _get_format(fmt, filename)
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
    di_attributes = {name: [] for name in di_notif["attributes"]}
    di_convert = _gen_convert(di_notif["attributes"], di_notif["attr_types"])
    line = None
    while lst_lines:
        line = lst_lines.pop()
        if line and not line.startswith(notifier):
            di_get_edges[fmt](line, di_notif["attributes"], separator,
                              secondary, edges, di_attributes, di_convert)
        else:
            break
    # check whether a shape is present
    if 'shape' in di_notif:
        if _shapely_support:
            min_x, max_x = float(di_notif['min_x']), float(di_notif['max_x'])
            unit = di_notif['unit']
            shape = Shape.from_wtk(
                di_notif['shape'], min_x=min_x, max_x=max_x, unit=unit)
        else:
            _log_message(logger, "WARNING",
                         'A Shape object was present in the file but could '
                         'not be loaded because Shapely is not installed.')
    if 'x' in di_notif:
        x = np.fromstring(di_notif['x'], sep=separator)
        y = np.fromstring(di_notif['y'], sep=separator)
        if 'z' in di_notif:
            z = np.fromstring(di_notif['z'], sep=separator)
            positions = np.array((x, y, z)).T
        else:
            positions = np.array((x, y)).T
    return di_notif, edges, di_attributes, pop, shape, positions


@graph_tool_check('2.22')
def save_to_file(graph, filename, fmt="auto", separator=" ",
                 secondary=";", attributes=None, notifier="@"):
    '''
    Save a graph to file.

    .. versionchanged:: 0.7
        Added support to write position and Shape when saving
        :class:`~nngt.SpatialGraph`. Note that saving Shape requires shapely.

    @todo: implement population and shape saving, implement gml, dot, xml, gt

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph to save.
    filename: str
        The path to the file.
    fmt : str, optional (default: "auto")
        The format used to save the graph. Supported formats are: "neighbour"
        (neighbour list, default if format cannot be deduced automatically),
        "ssp" (scipy.sparse), "edge_list" (list of all the edges in the graph,
        one edge per line, represented by a ``source target``-pair), "gml"
        (gml format, default if `filename` ends with '.gml'), "graphml"
        (graphml format, default if `filename` ends with '.graphml' or '.xml'),
        "dot" (dot format, default if `filename` ends with '.dot'), "gt" (only
        when using `graph_tool`<http://graph-tool.skewed.de/>_ as library,
        detected if `filename` ends with '.gt').
    separator : str, optional (default " ")
        separator used to separate inputs in the case of custom formats (namely
        "neighbour" and "edge_list")
    secondary : str, optional (default: ";")
        Secondary separator used to separate attributes in the case of custom
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

    .. warning ::
        For now, all formats lead to
        dataloss if your graph is a subclass of :class:`~nngt.SpatialGraph` or
        :class:`~nngt.Network` (the :class:`~nngt.geometry.Shape` and
        :class:`~nngt.NeuralPop` attributes will not be saved).

    .. note ::
        Positions are saved as bytes by :func:`numpy.nparray.tostring`
    '''
    fmt = _get_format(fmt, filename)
    str_graph, di_notif = _as_string(graph, separator=separator, fmt=fmt,
                          secondary=secondary, attributes=attributes,
                          notifier=notifier,  return_info=True)
    with open(filename, "w") as f_graph:
        for key, val in iter(di_notif.items()):
            f_graph.write("{}{}={}\n".format(notifier, key, val))
        f_graph.write("\n")
        f_graph.write(str_graph)


# --------------------- #
# String representation #
# --------------------- #

def _as_string(graph, fmt="neighbour", separator=" ", secondary=";",
              attributes=None, notifier="@", return_info=False):
    '''
    Full string representation of the graph.

    .. versionchanged:: 0.7
        Added support to write position and Shape when saving
        :class:`~nngt.SpatialGraph`. Note that saving Shape requires shapely.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph to save.
    fmt : str, optional (default: "auto")
        The format used to save the graph. Supported formats are: "neighbour"
        (neighbour list, default if format cannot be deduced automatically),
        "ssp" (:mod:`scipy.sparse`), "edge_list" (list of all the edges in the 
        graph, one edge per line, represented by a ``source target``-pair), 
        "gml" (gml format, default if `filename` ends with '.gml'), "graphml"
        (graphml format, default if `filename` ends with '.graphml' or '.xml'),
        "dot" (dot format, default if `filename` ends with '.dot'), "gt" (only
        when using `graph_tool`<http://graph-tool.skewed.de/>_ as library,
        detected if `filename` ends with '.gt').
    separator : str, optional (default " ")
        separator used to separate inputs in the case of custom formats (namely
        "neighbour" and "edge_list")
    secondary : str, optional (default: ";")
        Secondary separator used to separate attributes in the case of custom
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
        which are followed by the relevant notifiers among ``@shape``, ``@x``,
        ``@y``, ``@z``, ``@population``, and ``@graph`` to separate the
        sections.

    Returns
    -------
    str_graph : string
        The full graph representation as a string.
    '''
    # checks
    if separator == secondary:
        raise InvalidArgument("`separator` and `secondary` strings must be "
                              "different.")
    if notifier == separator or notifier == secondary:
        raise InvalidArgument("`notifier` string should differ from "
                              "`separator` and `secondary`.")
    # data
    if attributes is None:
        attributes = [a for a in graph.edges_attributes if a != "bweight"]
    additional_notif = {
        "directed": graph._directed,
        "attributes": attributes,
        "attr_types": [graph.get_attribute_type(attr) for attr in attributes],
        "name": graph.get_name(),
        "size": graph.node_nb()
    }
    # save positions for SpatialGraph (and shape if Shapely is available)
    if graph.is_spatial():
        if _shapely_support:
            additional_notif['shape'] = graph.shape.wkt
            additional_notif['unit'] = graph.shape.unit
            min_x, min_y, max_x, max_y = graph.shape.bounds
            additional_notif['min_x'] = min_x
            additional_notif['max_x'] = max_x
        else:
            _log_message(logger, "WARNING",
                         'The `shape` attribute of the graph could not be '
                         'saved to file because Shapely is not installed.')
        pos = graph.get_positions()
        # temporarily disable numpy cut threshold to save string
        old_threshold = np.get_printoptions()['threshold']
        #~ np.set_printoptions(threshold='nan')
        np.set_printoptions(threshold=np.NaN)
        additional_notif['x'] = np.array2string(
            pos[:, 0], max_line_width=np.NaN, separator=separator)[1:-1]
        additional_notif['y'] = np.array2string(
            pos[:, 1], max_line_width=np.NaN, separator=separator)[1:-1]
        if pos.shape[1] == 3:
            additional_notif['z'] = np.array2string(
                pos[:, 2], max_line_width=np.NaN, separator=separator)[1:-1]
        # set numpy cut threshold back on
        np.set_printoptions(threshold=old_threshold)

    str_graph = di_format[fmt](graph, separator=separator,
                               secondary=secondary, attributes=attributes)

    if return_info:
        return str_graph, additional_notif
    else:
        return str_graph


# ------------ #
# Saving tools #
# ------------ #

def _get_format(fmt, filename):
    if fmt == "auto":
        if filename.endswith('.gml'):
            fmt = 'gml'
        elif filename.endswith('.graphml') or filename.endswith('.xml'):
            fmt = 'graphml'
        elif filename.endswith('.dot'):
            fmt = 'dot'
        elif (filename.endswith('.gt') and
              nngt._config["graph_library"] == "graph-tool"):
            fmt = 'gt'
        elif filename.endswith('.nn'):
            fmt = 'neighbour'
        elif filename.endswith('.el'):
            fmt = 'edge_list'
        else:
            raise InvalidArgument('Could not determine format from filename '
                                  'please specify `fmt`.')
    return fmt


def _neighbour_list(graph, separator, secondary, attributes):
    '''
    Generate a string containing the neighbour list of the graph as well as a
    dict containing the notifiers as key and the associated values.
    @todo: speed this up!
    '''
    lst_neighbours = list(graph.adjacency_matrix().tolil().rows)
    attributes = graph._eattr
    for v1 in range(graph.node_nb()):
        for i, v2 in enumerate(lst_neighbours[v1]):
            str_edge = str(v2)
            for attr in attributes:
                str_edge += "{}{}".format(secondary,
                                          attributes[(v1, v2)][attr])
            lst_neighbours[v1][i] = str_edge
        lst_neighbours[v1] = "{}{}{}".format(
            v1, separator, separator.join(lst_neighbours[v1]))
    str_neighbours = "\n".join(lst_neighbours)
    return str_neighbours


def _edge_list(graph, separator, secondary, attributes):
    ''' Generate a string containing the edge list and their properties. '''
    edges = graph.edges_array
    attributes = {k: v for k, v in graph.edges_attributes.items()}
    end_strings = [secondary for _ in range(len(attributes) - 1)]
    end_strings.append('')
    lst_edges = []
    for i, e in enumerate(edges):
        str_edge = "{}{}{}".format(e[0], separator, e[1])
        if attributes:
            str_edge += separator
        for end, attr in zip(end_strings, attributes):
            str_edge += "{}{}".format(attributes[attr][i], end)
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


# ------------- #
# Loading tools #
# ------------- #

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
    separators = (';', ',', ' ', '\t')
    count = -np.Inf
    chosen = -1
    for i, delim in enumerate(separators):
        current = string.count(delim)
        if count < current:
            count = current
            chosen = separators[i]
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


def _get_edges_neighbour(line, attributes, separator, secondary, edges,
                         di_attributes, di_convert):
    '''
    Add edges and attributes to `edges` and `di_attributes` for the "neighbour"
    format.
    '''
    len_first_delim = line.find(separator)
    source = int(line[:len_first_delim])
    len_first_delim += 1
    if len_first_delim:
        neighbours = line[len_first_delim:].split(separator)
        for stub in neighbours:
            content = stub.split(secondary)
            target = int(content[0])
            edges.append((source, target))
            attr_val = content[1:] if len(content) > 1 else []
            for name, val in zip(attributes, attr_val):
                di_attributes[name].append(di_convert[name](val))


def _get_edges_elist(line, attributes, separator, secondary, edges,
                     di_attributes, di_convert):
    '''
    Add edges and attributes to `edges` and `di_attributes` for the "neighbour"
    format.
    '''
    data = line.split(separator)
    source, target = int(data[0]), int(data[1])
    edges.append((source, target))
    if len(data) == 3:
        attr_data = data[2].split(secondary)
        for name, val in zip(attributes, attr_data):
            di_attributes[name].append(di_convert[name](val))


# ---------- #
# Formatting #
# ---------- #

di_get_edges = {
    "neighbour": _get_edges_neighbour,
    "edge_list": _get_edges_elist
}

di_format = {
    "neighbour": _neighbour_list,
    "edge_list": _edge_list
}


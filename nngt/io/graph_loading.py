#-*- coding:utf-8 -*-
#
# io/graph_loading.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Loading functions """

import ast
import codecs
import logging
import pickle
import types

import numpy as np

import nngt
from nngt.lib import InvalidArgument
from nngt.lib.logger import _log_message
from ..geometry import Shape, _shapely_support
from .io_helpers import _get_format
from .loading_helpers import *


logger = logging.getLogger(__name__)


# ---------- #
# Formatting #
# ---------- #

di_get_edges = {
    "neighbour": _get_edges_neighbour,
    "edge_list": _get_edges_elist,
    "gml": _get_edges_gml,
    "graphml": _get_edges_graphml,
    "xml": _get_edges_graphml,
}


# ------------- #
# Load function #
# ------------- #

def load_from_file(filename, fmt="auto", separator=" ", secondary=";",
                   attributes=None, attributes_types=None, notifier="@",
                   ignore="#", name="LoadedGraph", directed=True,
                   cleanup=False):
    '''
    Load a Graph from a file.

    .. versionchanged :: 2.0
        Added optional `attributes_types` and `cleanup` arguments.

    .. warning ::
       Support for GraphML and DOT formats are currently limited and require
       one of the non-default backends (DOT requires graph-tool).

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
        For "edge_list", attributes may also be present as additional columns
        after the source and the target.
    attributes_types : dict, optional (default: str)
        Backup information if the type of the attributes is not specified
        in the file. Values must be callables (types or functions) that will
        take the argument value as a string input and convert it to the proper
        type.
    notifier : str, optional (default: "@")
        Symbol specifying the following as meaningfull information. Relevant
        information are formatted ``@info_name=info_value``, where
        ``info_name`` is in ("attributes", "directed", "name", "size") and
        associated ``info_value`` are of type (``list``, ``bool``, ``str``,
        ``int``).
        Additional notifiers are ``@type=SpatialGraph/Network/SpatialNetwork``,
        which must be followed by the relevant notifiers among ``@shape``,
        ``@structure``, and ``@graph``.
    ignore : str, optional (default: "#")
        Ignore lines starting with the `ignore` string.
    name : str, optional (default: from file information or 'LoadedGraph')
        The name of the graph.
    directed : bool, optional (default: from file information or True)
        Whether the graph is directed or not.
    cleanup : bool, optional (default: False)
       If true, removes nodes before the first one that appears in the
       edges and after the last one and renumber the nodes from 0.

    Returns
    -------
    graph : :class:`~nngt.Graph` or subclass
        Loaded graph.
    '''
    return nngt.Graph.from_file(
        filename, fmt=fmt, separator=separator, secondary=secondary,
        attributes=attributes, attributes_types=attributes_types,
        notifier=notifier, ignore=ignore, name=name, directed=directed,
        cleanup=cleanup) 


def _load_from_file(filename, fmt="auto", separator=" ", secondary=";",
                    attributes=None, attributes_types=None,
                    notifier="@", ignore="#", cleanup=False):
    '''
    Load the main properties (edges, attributes...) from a file.

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
        when using `graph_tool <http://graph-tool.skewed.de/>`_ as library,
        detected if `filename` ends with '.gt').
    separator : str, optional (default " ")
        separator used to separate inputs in the case of custom formats (namely
        "neighbour" and "edge_list")
    secondary : str, optional (default: ";")
        Secondary separator used to separate attributes in the case of custom
        formats.
    attributes : list, optional (default: [])
        List of names for the edge attributes present in the file. If a
        `notifier` is present in the file, names will be deduced from it;
        otherwise the attributes will be numbered.
    attributes_types : dict, optional (default: str)
        Backup information if the type of the attributes is not specified
        in the file. Values must be callables (types or functions) that will
        take the argument value as a string input and convert it to the proper
        type.
    notifier : str, optional (default: "@")
        Symbol specifying the following as meaningfull information. Relevant
        information are formatted ``@info_name=info_value``, where
        ``info_name`` is in ("attributes", "directed", "name", "size") and
        associated ``info_value`` are of type (``list``, ``bool``, ``str``,
        ``int``).
        Additional notifiers are ``@type=SpatialGraph/Network/SpatialNetwork``,
        which must be followed by the relevant notifiers among ``@shape``,
        ``@structure``, and ``@graph``.
    ignore : str, optional (default: "#")
        Ignore lines starting with the `ignore` string.
    cleanup : bool, optional (default: False)
       If true, removes nodes before the first one that appears in the
       edges and after the last one and renumber the nodes from 0. 

    Returns
    -------
    di_notif : dict
        Dictionary containing the main graph arguments.
    edges : list of 2-tuples
        Edges of the graph.
    di_nattributes : dict
        Dictionary containing the node attributes.
    di_eattributes : dict
        Dictionary containing the edge attributes (name as key, value as a
        list sorted in the same order as `edges`).
    struct : :class:`~nngt.NeuralPop`
        Population (``None`` if not present in the file).
    shape : :class:`~nngt.geometry.Shape`
        Shape of the graph (``None`` if not present in the file).
    positions : array-like of shape (N, d)
        The positions of the neurons (``None`` if not present in the file).
    '''
    # check for mpi
    if nngt.get_config("mpi"):
        raise NotImplementedError("This function is not ready for MPI yet.")

    # load
    lst_lines, struct, shape, positions = None, None, None, None
    fmt = _get_format(fmt, filename)

    if fmt not in di_get_edges:
        raise ValueError("Unsupported format: '{}'".format(fmt))

    with open(filename, "r") as filegraph:
        lst_lines = _process_file(filegraph, fmt, separator)

    # notifier lines
    di_notif = _get_notif(filename, lst_lines, notifier, attributes, fmt=fmt,
                          atypes=attributes_types)

    # get nodes attributes
    nattr_convertor = _gen_convert(di_notif["node_attributes"],
                                   di_notif["node_attr_types"],
                                   attributes_types=attributes_types)
    di_nattributes = _get_node_attr(di_notif, separator, fmt=fmt,
                                    lines=lst_lines, convertor=nattr_convertor)

    # make edges and attributes
    eattributes     = di_notif["edge_attributes"]
    di_eattributes  = {name: [] for name in eattributes}
    eattr_convertor = _gen_convert(di_notif["edge_attributes"],
                                   di_notif["edge_attr_types"],
                                   attributes_types=attributes_types)

    # process file
    edges = di_get_edges[fmt](
        lst_lines, eattributes, ignore, notifier, separator, secondary,
        di_attributes=di_eattributes, convertor=eattr_convertor,
        di_notif=di_notif)

    if cleanup:
        edges = np.array(edges) - np.min(edges)

    # add missing size information if necessary
    if "size" not in di_notif:
        di_notif["size"] = int(np.max(edges)) + 1

    # check whether a shape is present
    if 'shape' in di_notif:
        if _shapely_support:
            min_x, max_x = float(di_notif['min_x']), float(di_notif['max_x'])
            unit = di_notif['unit']
            shape = Shape.from_wkt(
                di_notif['shape'], min_x=min_x, max_x=max_x, unit=unit)
            # load areas
            try:
                def_areas      = ast.literal_eval(di_notif['default_areas'])
                def_areas_prop = ast.literal_eval(
                    di_notif['default_areas_prop'])

                for k in def_areas:
                    p = {key: float(v) for key, v in def_areas_prop[k].items()}
                    if "default_area" in k:
                        shape._areas["default_area"]._prop.update(p)
                        shape._areas["default_area"].height = p["height"]
                    else:
                        a = Shape.from_wkt(def_areas[k], unit=unit)
                        shape.add_area(a, height=p["height"], name=k,
                                       properties=p)

                ndef_areas      = ast.literal_eval(
                                      di_notif['non_default_areas'])
                ndef_areas_prop = ast.literal_eval(
                                      di_notif['non_default_areas_prop'])
                for i in ndef_areas:
                    p = {k: float(v) for k, v in ndef_areas_prop[i].items()}
                    a = Shape.from_wkt(ndef_areas[i], unit=unit)
                    shape.add_area(a, height=p["height"], name=i, properties=p)
            except KeyError:
                # backup compatibility with older versions
                pass
        else:
            _log_message(logger, "WARNING",
                         'A Shape object was present in the file but could '
                         'not be loaded because Shapely is not installed.')

    # check whether a structure is present
    if 'structure' in di_notif:
        str_enc = di_notif['structure'].replace('~', '\n').encode()
        str_dec = codecs.decode(str_enc, "base64")
        try:
            struct = pickle.loads(str_dec)
        except UnicodeError:
            struct = pickle.loads(str_dec, encoding="latin1")

    if 'x' in di_notif:
        x = np.fromstring(di_notif['x'], sep=separator)
        y = np.fromstring(di_notif['y'], sep=separator)
        if 'z' in di_notif:
            z = np.fromstring(di_notif['z'], sep=separator)
            positions = np.array((x, y, z)).T
        else:
            positions = np.array((x, y)).T

    return (di_notif, edges, di_nattributes, di_eattributes, struct, shape,
            positions)


def _library_load(filename, fmt):
    ''' Load the file using the library functions '''
    if nngt.get_config("backend") == "networkx":
        import networkx as nx

        if fmt == "graphml":
            return nx.read_graphml(filename)
        else:
            raise NotImplementedError
    elif nngt.get_config("backend") == "igraph":
        import igraph as ig

        if fmt == "graphml":
            return ig.Graph.Read_GraphML(filename)
        else:
            raise NotImplementedError
    elif nngt.get_config("backend") == "graph-tool":
        import graph_tool as gt

        return gt.load_graph(filename, fmt=fmt)
    else:
        raise NotImplementedError

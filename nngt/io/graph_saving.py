#-*- coding:utf-8 -*-
#
# io/graph_saving.py
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

""" IO tools for NNGT """

import codecs
import logging
import pickle
import sys
import weakref

from collections import defaultdict

import numpy as np
import scipy.sparse as ssp

import nngt
from nngt.lib import InvalidArgument, on_master_process
from nngt.lib.logger import _log_message

from ..geometry import Shape, _shapely_support
from .io_helpers import _get_format
from .saving_helpers import (_neighbour_list, _edge_list, _gml, _xml,
                             _custom_info, _gml_info, _xml_info,
                             _str_bytes_len)


logger = logging.getLogger(__name__)


# ---------- #
# Formatting #
# ---------- #

di_format = {
    "neighbour": _neighbour_list,
    "edge_list": _edge_list,
    "gml": _gml,
    "graphml": _xml,
    "xml": _xml,
}

format_graph_info = defaultdict(lambda: _custom_info)
format_graph_info["gml"] = _gml_info
format_graph_info["xml"] = _xml_info
format_graph_info["graphml"] = _xml_info


# --------------- #
# Saving function #
# --------------- #

def save_to_file(graph, filename, fmt="auto", separator=" ",
                 secondary=";", attributes=None, notifier="@"):
    '''
    Save a graph to file.

    @todo: implement dot, xml/graphml, and gt formats

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
        when using `graph_tool <http://graph-tool.skewed.de/>`_ as library,
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
        ``@structure``, and ``@graph`` to separate the sections.

    Note
    ----
    Positions are saved as bytes by :func:`numpy.nparray.tostring`
    '''
    fmt = _get_format(fmt, filename)

    # check for mpi
    if nngt.get_config("mpi"):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        # get the
        str_local, di_notif = _as_string(
            graph, separator=separator, fmt=fmt, secondary=secondary,
            attributes=attributes, notifier=notifier, return_info=True)
        # make notification string only on master thread
        str_notif = ""
        if on_master_process():
            for key, val in iter(di_notif.items()):
                str_notif += "{}{}={}\n".format(notifier, key, val)
        # strings need to start with a newline because MPI strips last
        str_local = "\n" + str_local
        # gather all strings sizes
        sizes = comm.allgather(
            _str_bytes_len(str_local) + _str_bytes_len(str_notif))
        # get rank-based offset
        offset = [_str_bytes_len(str_notif)]
        offset.extend(np.cumsum(sizes)[:-1])
        # open file and write
        if on_master_process():
            with open(filename, "w") as f_graph:
                f_graph.write(str_notif)
        # parallel write
        amode = MPI.MODE_WRONLY
        fh = MPI.File.Open(comm, filename, amode)
        fh.Write_at_all(offset[rank], str_local.encode('utf-8'))
        fh.Close()
    else:
        str_graph = _as_string(
            graph, separator=separator, fmt=fmt, secondary=secondary,
            attributes=attributes, notifier=notifier)
        with open(filename, "w") as f_graph:
            f_graph.write(str_graph)


# --------------------- #
# String representation #
# --------------------- #

def _as_string(graph, fmt="neighbour", separator=" ", secondary=";",
              attributes=None, notifier="@", return_info=False):
    '''
    Full string representation of the graph.

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
    attributes : list, optional (default: all)
        List of names for the edge attributes present in the graph that will be
        saved to disk; by default, all attributes will be saved.
    notifier : str, optional (default: "@")
        Symbol specifying the following as meaningfull information. Relevant
        information are formatted ``@info_name=info_value``, with
        ``info_name`` in ("attributes", "attr_types", "directed", "name",
        "size").
        Additional notifiers are ``@type=SpatialGraph/Network/SpatialNetwork``,
        which are followed by the relevant notifiers among ``@shape``, ``@x``,
        ``@y``, ``@z``, ``@structure``, and ``@graph`` to separate the
        sections.

    Returns
    -------
    str_graph : string
        The full graph representation as a string.
    '''
    # checks
    if separator == secondary and fmt != "edge_list":
        raise InvalidArgument("`separator` and `secondary` strings must be "
                              "different.")

    if notifier == separator or notifier == secondary:
        raise InvalidArgument("`notifier` string should differ from "
                              "`separator` and `secondary`.")

    # temporarily disable numpy cut threshold to save string
    old_threshold = np.get_printoptions()['threshold']
    np.set_printoptions(threshold=sys.maxsize)

    # data
    if attributes is None:
        attributes = [a for a in graph.edge_attributes if a != "bweight"]

    nattributes = [a for a in graph.node_attributes]

    additional_notif = {
        "directed": graph.is_directed(),
        "node_attributes": nattributes,
        "node_attr_types": [
            graph.get_attribute_type(nattr, "node") for nattr in nattributes
        ],
        "edge_attributes": attributes,
        "edge_attr_types": [
            graph.get_attribute_type(attr, "edge") for attr in attributes
        ],
        "name": graph.name,
        "size": graph.node_nb()
    }

    # add node attributes to the notifications
    if fmt != "graphml":
        for nattr in additional_notif["node_attributes"]:
            key = "na_" + nattr

            tmp = np.array2string(
                graph.get_node_attributes(name=nattr), max_line_width=np.NaN,
                separator=separator)[1:-1].replace("'" + separator + "'",
                                                   '"' + separator + '"')

            # replace possible variants
            tmp = tmp.replace("'" + separator + '"', '"' + separator + '"')
            tmp = tmp.replace('"' + separator + "'", '"' + separator + '"')

            if tmp.startswith("'"):
                tmp = '"' + tmp[1:]

            if tmp.endswith("'"):
                 tmp = tmp[:-1] + '"'

            # make and store final string
            additional_notif[key] = tmp

    # save positions for SpatialGraph (and shape if Shapely is available)
    if graph.is_spatial():
        if _shapely_support:
            additional_notif['shape'] = graph.shape.wkt
            additional_notif['default_areas'] = \
                {k: v.wkt for k, v in graph.shape.default_areas.items()}
            additional_notif['default_areas_prop'] = \
                {k: v.properties for k, v in graph.shape.default_areas.items()}
            additional_notif['non_default_areas'] = \
                {k: v.wkt for k, v in graph.shape.non_default_areas.items()}
            additional_notif['non_default_areas_prop'] = \
                {k: v.properties
                 for k, v in graph.shape.non_default_areas.items()}
            additional_notif['unit'] = graph.shape.unit
            min_x, min_y, max_x, max_y = graph.shape.bounds
            additional_notif['min_x'] = min_x
            additional_notif['max_x'] = max_x
        else:
            _log_message(logger, "WARNING",
                         'The `shape` attribute of the graph could not be '
                         'saved to file because Shapely is not installed.')

        pos = graph.get_positions()
        additional_notif['x'] = np.array2string(
            pos[:, 0], max_line_width=np.NaN, separator=separator)[1:-1]
        additional_notif['y'] = np.array2string(
            pos[:, 1], max_line_width=np.NaN, separator=separator)[1:-1]
        if pos.shape[1] == 3:
            additional_notif['z'] = np.array2string(
                pos[:, 2], max_line_width=np.NaN, separator=separator)[1:-1]

    if graph.structure is not None:
        # temporarily remove weakrefs
        graph.structure._parent = None
        for g in graph.structure.values():
            g._struct = None
            g._net    = None
        # save as string
        if nngt.get_config("mpi"):
            if nngt.get_config("mpi_comm").Get_rank() == 0:
                additional_notif["structure"] = codecs.encode(
                    pickle.dumps(graph.structure, protocol=2),
                                 "base64").decode().replace('\n', '~')
        else:
            additional_notif["structure"] = codecs.encode(
                pickle.dumps(graph.structure, protocol=2),
                             "base64").decode().replace('\n', '~')
        # restore weakrefs
        graph.structure._parent = weakref.ref(graph)
        for g in graph.structure.values():
            g._struct = weakref.ref(graph.structure)
            g._net    = weakref.ref(graph)

    str_graph = di_format[fmt](graph, separator=separator,
                               secondary=secondary, attributes=attributes,
                               additional_notif=additional_notif)

    # set numpy cut threshold back on
    np.set_printoptions(threshold=old_threshold)

    if return_info:
        return str_graph, additional_notif

    # format the info into the string
    info_str = format_graph_info[fmt](additional_notif, notifier, graph=graph)

    return info_str + str_graph

#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
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

""" Loading helpers """

import re
import types

import numpy as np

from ..lib.converters import (_np_dtype, _to_int, _to_string, _to_list,
                              _string_from_object)


__all__ = [
    "_cleanup_line",
    "_gen_convert",
    "_get_edges_elist",
    "_get_edges_gml",
    "_get_edges_neighbour",
    "_get_node_attr",
    "_get_notif",
    "_process_file",
]


# ----------------------- #
# Initial file processing #
# ----------------------- #

def _process_file(f, fmt, separator):
    if fmt == "gml":
        # format gml input to expected one
        lines = []

        for l in f.readlines():
            clean_line = _cleanup_line(l, separator)

            if clean_line.endswith("[") and len(clean_line) > 1:
                lines.append(clean_line[:-1].strip())
                lines.append("[")
            elif clean_line.endswith("]") and len(clean_line) > 1:
                lines.append(clean_line[:-1].strip())
                lines.append("]")
            else:
                lines.append(clean_line)

        return lines

    # otherwise just cleanup the lines
    return [_cleanup_line(line, separator) for line in f.readlines()]


# ---------------- #
# Graph properties #
# ---------------- #

def _format_notif(notif_name, notif_val):
    attr = (
        "node_attributes", "edge_attributes", "node_attr_types",
        "edge_attr_types"
    )
    if notif_name in attr:
        lst = notif_val[1:-1].split(", ")
        if lst != ['']:  # check empty string
            return [val.strip("'\"") for val in lst]
        else:
            return []
    elif notif_name == "size":
        return int(notif_val)
    elif notif_name == "directed":
        return False if notif_val in ("False", "0") else True
    else:
        return notif_val


def _get_notif(lines, notifier, attributes, fmt=None, atypes=None):
    di_notif = {
        "node_attributes": [], "edge_attributes": [], "node_attr_types": [],
        "edge_attr_types": [],
    }

    # special case for GML
    if fmt == "gml":
        start = 0

        for i, l in enumerate(lines):
            if l == "graph":
                start = i
                break

        # nodes
        nodes = [i for i, l in enumerate(lines) if l == "node" and i > start]
        num_nodes = len(nodes)

        di_notif["size"]  = num_nodes
        di_notif["nodes"] = nodes

        # node attributes
        diff = np.diff(nodes) - 4  # number of lines other than node spec

        num_nattr = diff[0]

        if not np.all(diff == num_nattr):
            raise RuntimeError("All nodes should have the same attributes.")

        if num_nattr > len(di_notif["node_attributes"]):
            for i in range(nodes[0] + 3, nodes[0] + num_nattr + 3):
                name = lines[i].split(" ")[0]
                di_notif["node_attributes"].append(name)
                # default type is object
                if atypes is not None:
                    di_notif["node_attr_types"].append(
                        _string_from_object(atypes.get(name, object)))
                else:
                    di_notif["node_attr_types"].append("object")

        # graph attributes
        for line in lines[:nodes[0]]:
            first_space = line.find(" ")
            key, val = line[:first_space], line[first_space + 1:]

            di_notif[key] = _format_notif(key, val)

        # edges
        edges = [i for i, l in enumerate(lines) if l == "edge" and i > start]

        di_notif["edges"] = edges

        diff = np.diff(edges) - 5  # number of lines other than edge spec

        num_eattr = diff[0]

        if not np.all(diff == num_eattr):
            raise RuntimeError("All edges should have the same attributes.")

        if num_eattr > len(di_notif["edge_attributes"]):
            for i in range(edges[0] + 4, edges[0] + num_eattr + 4):
                name = lines[i].split(" ")[0]
                di_notif["edge_attributes"].append(name)
                # default type is object
                if atypes is not None:
                    di_notif["edge_attr_types"].append(
                        _string_from_object(atypes.get(name, object)))
                else:
                    di_notif["edge_attr_types"].append("object")
    else:
        for line in lines:
            if line.startswith(notifier):
                idx_eq = line.find("=")
                notif_name = line[len(notifier):idx_eq]
                notif_val = line[idx_eq+1:]
                di_notif[notif_name] = _format_notif(notif_name, notif_val)
            else:
                break

        if attributes is not None:
            di_notif["edge_attributes"] = attributes

            if atypes is not None:
                for attr in attributes:
                    di_notif["edge_attr_types"].append(
                        _string_from_object(atypes.get(attr, object)))
            else:
                di_notif["edge_attributes"] = ["object"]*len(attributes)

    return di_notif


# ----- #
# Edges #
# ----- #

def _get_edges_neighbour(lst_lines, attributes, ignore, notifier, separator,
                         secondary, di_attributes, di_convert, **kwargs):
    '''
    Add edges and attributes to `edges` and `di_attributes` for the "neighbour"
    format.
    '''
    edges = []

    lst_lines = lst_lines[::-1]

    while lst_lines:
        line = lst_lines.pop()

        if line and not (line.startswith(notifier) or line.startswith(ignore)):
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

    return edges


def _get_edges_elist(lst_lines, attributes, ignore, notifier, separator,
                     secondary, di_attributes, di_convert, **kwargs):
    '''
    Add edges and attributes to `edges` and `di_attributes` for the "neighbour"
    format.
    '''
    edges = []

    lst_lines = lst_lines[::-1]

    while lst_lines:
        line = lst_lines.pop()

        if line and not (line.startswith(notifier) or line.startswith(ignore)):
            data = line.split(separator)
            source, target = int(data[0]), int(data[1])
            edges.append((source, target))

            # different ways of loading the attributes
            if len(data) == 3 and secondary in data[2]:  # secondary notifier
                attr_data = data[2].split(secondary)
                for name, val in zip(attributes, attr_data):
                    di_attributes[name].append(di_convert[name](val))
            elif len(data) == len(attributes) + 2:  # regular columns
                for name, val in zip(attributes, data[2:]):
                    di_attributes[name].append(di_convert[name](val))

    return edges


def _get_edges_gml(lst_lines, attributes, *args, di_attributes=None,
                   di_convert=None, di_notif=None):
    '''
    Add edges and attributes to `edges` and `di_attributes` for the "neighbour"
    format.
    '''
    edges = []

    edge_lines = di_notif["edges"]
    num_eattr  = len(di_attributes)

    for line_num in edge_lines:
        source = int(lst_lines[line_num + 2][7:])
        target = int(lst_lines[line_num + 3][7:])

        edges.append((source, target))

        for i, name in zip(range(num_eattr), attributes):
            lnum  = line_num + 4 + i
            start = lst_lines[lnum].find(" ") + 1
            attr  = lst_lines[lnum][start:]
            di_attributes[name].append(di_convert[name](attr))

    return edges


def _get_node_attr(di_notif, separator, fmt=None, lines=None, atypes=None):
    '''
    Return node attributes.

    For custom formats, attributes are stored under @na_{attr_name} in the
    file, so they are stored under the coresponding key in `di_notif`.

    For GML, need to get them from the nodes.
    '''
    di_nattr   = {}

    if fmt == "gml":
        node_lines = di_notif["nodes"]
        num_nattr  = node_lines[1] - node_lines[0] - 3  # lines other than attr

        has_types = len(di_notif["node_attr_types"]) == num_nattr

        if num_nattr:
            for line_num in node_lines:
                for i in range(num_nattr):
                    l = lines[line_num + i + 2]
                    first_space = l.find(" ")

                    # get attribute name and value
                    name = l[:first_space]
                    val  = l[first_space + 1:]

                    if name not in di_nattr:
                        di_nattr[name] = []

                    dtype = str if atypes is None else atypes.get(name, str)

                    if has_types:
                        dtype = _type_converter(di_notif["node_attr_types"][i])

                    di_nattr[name].append(dtype(val))
    else:
        nattr_name = {str("na_" + k): k for k in di_notif["node_attributes"]}
        nattr_type = di_notif["node_attr_types"]

        for k, s in di_notif.items():
            if k in nattr_name:
                attr           = nattr_name[k]
                idx            = di_notif["node_attributes"].index(attr)
                dtype          = _np_dtype(nattr_type[idx])

                if dtype == object:
                    di_nattr[attr] = np.array(s.split(separator), dtype=dtype)
                else:
                    di_nattr[attr] = np.fromstring(s, sep=separator,
                                                   dtype=dtype)
                

    return di_nattr


# ---------- #
# Converters #
# ---------- #

def _gen_convert(attributes, attr_types, attributes_types=None):
    '''
    Generate a conversion dictionary that associates the right type to each
    attribute
    '''
    di_convert = {}

    if attributes and not attr_types:
        attr_types.extend(("string" for _ in attributes))

    for attr, attr_type in zip(attributes, attr_types):
        if attributes_types is not None and attr in attributes_types:
            # user defined converter
            di_convert[attr] = attributes_types[attr]
        elif attr_type in ("double", "float", "real"):
            di_convert[attr] = float
        elif attr_type in ("str", "string"):
            di_convert[attr] = lambda x: str(x).strip("\"'")
        elif attr_type in ("int", "integer"):
            di_convert[attr] = _to_int
        elif attr_type in ("lst", "list", "tuple", "array"):
            di_convert[attr] = _to_list
        elif attr_type == "object":
            di_convert[attr] = lambda x: x
        else:
            raise TypeError("Invalid attribute type: '{}'.".format(attr_type))

    return di_convert


def _cleanup_line(string, char):
    ''' Replace multiple occurrences of a separator and remove line ends '''
    pattern = char + '+'
    string = re.sub(pattern, char, string)
    return string.strip()

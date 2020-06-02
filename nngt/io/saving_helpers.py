#!/usr/bin/env python
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

""" IO tools for NNGT """


def _neighbour_list(graph, separator, secondary, attributes):
    '''
    Generate a string containing the neighbour list of the graph as well as a
    dict containing the notifiers as key and the associated values.
    @todo: speed this up!
    '''
    lst_neighbours = list(graph.adjacency_matrix(mformat="lil").rows)

    for v1 in range(graph.node_nb()):
        for i, v2 in enumerate(lst_neighbours[v1]):
            str_edge = str(v2)
            eattr    = graph.get_edge_attributes((v1, v2))

            for attr in attributes:
                str_edge += "{}{}".format(secondary, eattr[attr])

            lst_neighbours[v1][i] = str_edge

        lst_neighbours[v1] = "{}{}{}".format(
            v1, separator, separator.join(lst_neighbours[v1]))

    str_neighbours = "\n".join(lst_neighbours)

    return str_neighbours


def _edge_list(graph, separator, secondary, attributes):
    ''' Generate a string containing the edge list and their properties. '''
    edges = graph.edges_array

    di_attributes = {
        k: v for k, v in graph.edge_attributes.items()
        if k != 'bweight'
    }

    end_strings = (len(attributes) - 1)*[secondary]

    end_strings.append('')

    lst_edges = []

    for i, e in enumerate(edges):
        str_edge = "{}{}{}".format(e[0], separator, e[1])

        if attributes:
            str_edge += separator

        for end, attr in zip(end_strings, attributes):
            str_edge += "{}{}".format(di_attributes[attr][i], end)

        lst_edges.append(str_edge)

    str_edges = "\n".join(lst_edges)

    return str_edges


def _dot(graph, attributes, **kwargs):
    pass


def _gml(graph, *args, **kwargs):
    ''' Generate a string containing the edge list and their properties. '''
    node_str = "  node\n  [\n    id {id}{attr}\n  ]"
    edge_str = "  edge\n  [\n    source {s}\n    target {t}\n{attr}\n  ]"
    attr_str = "    {key} {val}"

    indent = "    "

    # set nodes
    lst_elements = []

    for i in range(graph.node_nb()):
        lst_attr = []

        for k, v in graph.node_attributes.items():
            lst_attr.append(attr_str.format(key=k, val=v[i]))

        nattr = "\n" + "\n".join(lst_attr)

        lst_elements.append(node_str.format(id=i, attr=nattr))

    # set edges
    edges = graph.edges_array

    for i, e in enumerate(edges):
        lst_attr = []

        for k, v in graph.edge_attributes.items():
            lst_attr.append(attr_str.format(key=k, val=v[i]))

        eattr = "\n".join(lst_attr)

        lst_elements.append(edge_str.format(s=e[0], t=e[1], attr=eattr))

    str_gml = "\n".join(lst_elements)

    str_gml += "\n]"

    return str_gml


def _xml(graph, attributes, **kwargs):
    pass


def _gt(graph, attributes, **kwargs):
    pass


def _custom_info(graph_info, notifier, *args, **kwargs):
    ''' Format the graph information for custom formats '''
    info_str = ""

    for key, val in iter(graph_info.items()):
        info_str += "{}{}={}\n".format(notifier, key, val)

    return info_str


def _gml_info(graph_info, *args, **kwargs):
    ''' Format the graph information for the GML format '''
    info_str = "graph\n[\n"

    for key, val in iter(graph_info.items()):
        if not key.startswith("na_"):
            val = 1 if val is True else (0 if val is False else val)

            info_str += "  {} {}\n".format(key, val)

    return info_str


def _str_bytes_len(s):
    return len(s.encode('utf-8'))

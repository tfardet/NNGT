#-*- coding:utf-8 -*-
#
# io/saving_helpers.py
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

import logging

from nngt.lib.logger import _log_message

logger = logging.getLogger(__name__)


def _neighbour_list(graph, separator, secondary, attributes, **kwargs):
    '''
    Generate a string containing the neighbour list of the graph as well as a
    dict containing the notifiers as key and the associated values.
    @todo: speed this up!
    '''
    lst_neighbours = None

    if graph.is_directed():
        lst_neighbours = list(graph.adjacency_matrix(mformat="lil").rows)
    else:
        import scipy.sparse as ssp
        lst_neighbours = list(
            ssp.tril(graph.adjacency_matrix(), format="lil").rows)

    for v1 in range(graph.node_nb()):
        for i, v2 in enumerate(lst_neighbours[v1]):
            str_edge = str(v2)

            eattr = graph.get_edge_attributes((v1, v2))

            for attr in attributes:
                str_edge += "{}{}".format(secondary, eattr[attr])

            lst_neighbours[v1][i] = str_edge

        lst_neighbours[v1] = "{}{}{}".format(
            v1, separator, separator.join(lst_neighbours[v1]))

    str_neighbours = "\n".join(lst_neighbours)

    return str_neighbours


def _edge_list(graph, separator, secondary, attributes, **kwargs):
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

        attrs = graph.get_edge_attributes(e)

        for k, v in attrs.items():
            lst_attr.append(attr_str.format(key=k, val=v))

        eattr = "\n".join(lst_attr)

        lst_elements.append(edge_str.format(s=e[0], t=e[1], attr=eattr))

    str_gml = "\n".join(lst_elements)

    str_gml += "\n]"

    return str_gml


def _xml(graph, attributes=None, additional_notif=None, **kwargs):
    try:
        from lxml import etree as ET
        lxml = True
    except:
        lxml = False
        import xml.etree.ElementTree as ET
        _log_message(logger, "WARNING",
                     "LXML is not installed, using Python XML for export. "
                     "Some apps like Gephi <= 0.9.2 will not read attributes "
                     "from the generated GraphML file due to elements' order.")

    NS_GRAPHML = "http://graphml.graphdrawing.org/xmlns"
    NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
    NS_Y = "http://www.yworks.com/xml/graphml"
    NSMAP = {
        "xsi": NS_XSI
    }
    SCHEMALOCATION = " ".join(
        [
            "http://graphml.graphdrawing.org/xmlns",
            "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
        ]
    )

    doc = ET.Element(
        "graphml",
        {
            "xmlns": NS_GRAPHML,
        },
        nsmap=NSMAP
    )

    n = doc.set("{{}}xsi".format(NS_GRAPHML), NS_XSI)
    n = doc.set("{{}}schemaLocation".format(NS_XSI), SCHEMALOCATION)

    # make graph element
    directedness = "directed" if graph.is_directed() else "undirected"

    eg = ET.SubElement(doc, "graph", edgedefault=directedness, id=graph.name)

    # prepare graph data
    del additional_notif["directed"]
    del additional_notif["name"]

    nattrs = additional_notif.pop("node_attributes")
    ntypes = additional_notif.pop("node_attr_types")

    for attr, atype in zip(nattrs, ntypes):
        kw = {"for": "node", "attr.name": attr, "attr.type": atype}
        if lxml:
            key = ET.Element("key", id=attr, **kw)
            eg.addprevious(key)
        else:
            ET.SubElement(doc, "key", id=attr, **kw)

    eattrs = additional_notif.pop("edge_attributes")
    etypes = additional_notif.pop("edge_attr_types")

    for attr, atype in zip(eattrs, etypes):
        kw = {"for": "edge", "attr.name": attr, "attr.type": atype}
        if lxml:
            key = ET.Element("key", id=attr, **kw)
            eg.addprevious(key)
        else:
            ET.SubElement(doc, "key", id=attr, **kw)

    # add remaining information as data to the graph
    for k, v in additional_notif.items():
        elt = ET.SubElement(doc, "data", key=k)
        elt.text = str(v)

    # add node information
    nattr = graph.get_node_attributes()

    for n in graph.get_nodes():
        nelt = ET.SubElement(eg, "node", id=str(n))

        for k, v in nattr.items():
            elt = ET.SubElement(nelt, "data", key=k)
            elt.text = str(v[n])

    # add edge information
    for e in graph.get_edges():
        nelt = ET.SubElement(eg, "edge", id="e{}".format(graph.edge_id(e)),
                             source=str(e[0]), target=str(e[1]))
        for k in eattrs:
            elt = ET.SubElement(nelt, "data", key=k)
            elt.text = str(graph.get_edge_attributes(e, name=k))

    kw = {"pretty_print": True} if lxml else {}

    return ET.tostring(doc, encoding="unicode", **kw)


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


def _xml_info(*args, **kwargs):
    ''' Return empty string '''
    return ""


def _str_bytes_len(s):
    return len(s.encode('utf-8'))

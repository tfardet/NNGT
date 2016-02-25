#!/usr/bin/env python
#-*- coding:utf-8 -*-

# base_test.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

import xml.etree.ElementTree as xmlet



#-----------------------------------------------------------------------------#
# Xml tools
#------------------------
#   

def _bool_from_string(string):
    return True if (string.lower() == "true") else False

def _list_from_string(string, elt_type, di_convert):
    lst = string[1:-1].split(", ")
    return [ di_convert[elt_type](elt) for elt in lst ]

def _xml_to_dict(xml_elt, di_types):
    di_result = {}
    for child in xml_elt:
        str_type = child.tag
        name = child.attrib["name"]
        if len(child):
            elt = child.find("start")
            if elt is not None:
                start = di_types[str_type](child.find("start").text)
                stop = di_types[str_type](child.find("stop").text)
                step = di_types[str_type](child.find("step").text)
                di_result[name] = np.arange(start,stop,step)
            else:
                di_result[child.tag] = _xml_to_dict(child, di_types)
        else:
            if child.tag in ("string", "str"):
                text = child.text.replace("\\t","\t")
                di_result[name] = di_types[str_type](text)
            elif child.tag == "list":
                elt_type = child.attrib["type"]
                di_result[name] = _list_from_string( child.text,
                                                     elt_type, di_types )
            else:
                di_result[name] = di_types[str_type](child.text)
    return di_result


#-----------------------------------------------------------------------------#
# Decorator: repeat test for each graph
#------------------------
#

def foreach_graph(func):
    '''
    Decorator that automatically does the test for all graph instructions
    provided in the argument.
    '''       
    def wrapper(*args, **kwargs):
        self = args[0]
        for graph in self.__class__.graphs:
            func(self, graph, **kwargs)
    return wrapper

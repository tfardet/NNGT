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

def _xml_to_dict(xml_elt, di_types):
	di_result = {}
	for child in xml_elt:
		str_type = child.tag
		if len(child):
			elt = child.find("start")
			if elt is not None:
				start = di_types[str_type](child.find("start").text)
				stop = di_types[str_type](child.find("stop").text)
				step = di_types[str_type](child.find("step").text)
				di_result[child.attrib["name"]] = np.arange(start,stop,step)
			else:
				di_result[child.tag] = _xml_to_dict(child, di_types)
		else:
			di_result[child.attrib["name"]] = di_types[child.tag](child.text)
	return di_result

def _make_graph_list(string):
    pass


#-----------------------------------------------------------------------------#
# Decorators
#------------------------
#

def with_metaclass(mcls):
    ''' Python 2/3 compatible metaclass declaration. '''
    def decorator(cls):
        body = vars(cls).copy()
        # clean out class body
        body.pop('__dict__', None)
        body.pop('__weakref__', None)
        return mcls(cls.__name__, cls.__bases__, body)
    return decorator

def foreach_graph(graphs):
    '''
    Decorator that automatically does the test for all graph instructions
    provided in the argument.
    '''
    def decorator(func):          
        def wrapper(*args, **kwargs):
            self = args[0]
            for graph in graphs:
                func(self, graph, **kwargs)
        return wrapper
    return decorator

def set_path(path):
    '''
    Decorator that automatically sets the right path to network files.
    '''
    def decorator(func):          
        def wrapper(*args, **kwargs):
            kwargs["path"] = path + kwargs["path"]
            return func(*args, **kwargs)
        return wrapper
    return decorator

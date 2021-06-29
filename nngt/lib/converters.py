#-*- coding:utf-8 -*-
#
# lib/converters.py
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

""" Loading helpers """

import types

import numpy as np


def _to_np_array(data, dtype):
    '''
    Transform data to a numpy array, preserving nested lists as lists if dtype
    is "object".
    '''
    dtype = object if dtype == "string" else dtype

    if dtype in (object, "object"):
        # preserve potential list entries
        arr    = np.empty(len(data), dtype=object)
        arr[:] = data
        return arr

    return np.array(data, dtype=dtype)


def _to_list(string):
    separators = (';', ',', ' ', '\t')
    count = -np.Inf
    chosen = -1
    for i, delim in enumerate(separators):
        current = string.count(delim)
        if count < current:
            count = current
            chosen = separators[i]
            break
    return string.split(chosen)


def _to_int(string):
    try:
        return int(string)
    except ValueError:
        return int(float(string))


def _to_string(byte_string):
    ''' Convert bytes to string '''
    if isinstance(byte_string, bytes):
        return str(byte_string.decode())

    return byte_string


def _string_from_object(obj):
    ''' Return a type string from an object (usually a class) '''
    if obj.__class__ == type:
        # dealing with a class
        if issubclass(obj, float):
            return "double"

        if issubclass(obj, (int, np.integer)):
            return "int"

        if issubclass(obj, str):
            return "string"

        return "object"
    elif issubclass(obj.__class__, (types.FunctionType, types.MethodType)):
        # dealing with a function
        return "object"

    raise ValueError("Cannot deduce class string from '{}'.".format(obj))


def _np_dtype(attribute_type):
    '''
    Return a relevant numpy dtype entry.
    '''
    if attribute_type in ("double", "float", "real"):
        return float
    elif attribute_type in ("int", "integer"):
        return int

    return object


def _python_type(attribute_type):
    '''
    Return a relevant numpy dtype entry.
    '''
    if attribute_type in ("double", "float", "real"):
        return float
    elif attribute_type in ("int", "integer"):
        return int

    return str


def _type_converter(attribute_type):
    if not isinstance(attribute_type, str):
        return attribute_type

    return _np_dtype(attribute_type)

# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/lib/converters.py

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


def _default_value(attribute_type):
    if attribute_type in ("double", "float", "real"):
        return np.NaN
    elif attribute_type in ("int", "integer"):
        return 0

    return ""

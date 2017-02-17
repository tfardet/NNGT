#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Test functions for the NNGT """

import collections


def valid_gen_arguments(func):
    def wrapper(*args, **kwargs):
        return func(*args,**kwargs)
    return wrapper


def nonstring_container(obj):
    '''
    Returns true for any iterable which is not a string or byte sequence.
    '''
    if not isinstance(obj, collection.abc.Container):
        return False
    try:
        if isinstance(obj, unicode):
            return False
    except NameError:
        pass
    if isinstance(obj, bytes):
        return False
    if isinstance(obj, str):
        return False
    return True

#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Test decorators for the NNGT """


def valid_gen_arguments(func):
    def wrapper(*args, **kwargs):
        return func(*args,**kwargs)
    return wrapper

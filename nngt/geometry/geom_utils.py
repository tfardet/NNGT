#!/usr/bin/env python
#-*- coding:utf-8 -*-

'''
Geometry utility functions.
'''


_di_mag = {
    'um': 1e-6,
    'mm': 1e-3,
    'cm': 1e-2,
    'dm': 0.1,
    'm': 1.
}


def conversion_magnitude(source_unit, target_unit):
    '''
    Returns the magnitude necessary to convert from values in `source_unit` to
    values in `target_unit`.
    '''
    return _di_mag[source_unit] / _di_mag[target_unit]

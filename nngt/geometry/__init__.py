#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
Module dedicated to the spatial embedding of the graphs.

It uses the `shapely<http://toblerity.org/shapely/index.html>`_ library to
generate and deal with the spatial environment of the nodes.


Content
=======
"""

import importlib

import shapely
from shapely import speedups
if speedups.available:
    speedups.enable()


try:
    from .shape import Shape
except ImportError:
    from .backup_shape import Shape


__all__ = ["Shape"]

try:
    from .svgtools import from_svg
    __all__.append("from_svg")
except:
    pass

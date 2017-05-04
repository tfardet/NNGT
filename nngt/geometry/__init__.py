#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
Module dedicated to the spatial embedding of the graphs.

It uses the `shapely<http://toblerity.org/shapely/index.html>`_ library to
generate and deal with the spatial environment of the nodes.


Content
=======
"""

try:
    import shapely
    from shapely import speedups
    if speedups.available:
        speedups.enable()
    from .shape import Shape
except ImportError:
    from .backup_shape import Shape


__all__ = ["Shape"]

from . import svgtools
try:
    from . import svgtools
    from .svgtools import *
    __all__.extend(svgtools.__all__)
except Exception as e:
    print("Could not import svgtools: {}".format(e))

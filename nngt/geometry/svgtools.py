#!/usr/bin/env python
#-*- coding:utf-8 -*-

from xml.dom.minidom import parse
from svg.path import parse_path, CubicBezier, QuadraticBezier, Arc
from itertools import chain

import shapely
from shapely.affinity import scale
from shapely.geometry import Point, Polygon

import numpy as np

from nngt.geometry import Shape


'''
Shape generation from SVG files.
'''


__all__ = [
    "shapes_from_svg",
    "culture_from_svg",
]


# predefined svg shapes and their parameters
_predefined = {
    'path': None,
    'ellipse': ("cx", "cy", "rx", "ry"),
    'circle': ("cx", "cy", "r"),
    'rect': ("x", "y", "width", "height")
}


def shapes_from_svg(filename, interpolate_curve=50, parent=None,
                    return_points=False):
    '''
    Generate :class:`shapely.geometry.Polygon` objects from an SVG file.
    '''
    svg = parse(filename)
    elt_structs = {k: [] for k in _predefined.keys()}
    elt_points = {k: [] for k in _predefined.keys()}

    # get the properties of all predefined elements
    for elt_type, elt_prop in _predefined.items():
        _build_struct(svg, elt_structs[elt_type], elt_type, elt_prop)

    # build all shapes
    shapes = []
    for elt_type, instructions in elt_structs.items():
        for struct in instructions:
            polygon, points = _make_shape(
                elt_type, struct, parent=parent, return_points=True)
            shapes.append(polygon)
            elt_points[elt_type].append(points)

    if return_points:
        return shapes, elt_points

    return shapes


def culture_from_svg(filename, min_x=-5000., max_x=5000., unit='um',
                     parent=None, interpolate_curve=50):
    '''
    Generate a culture from an SVG file.
    
    Valid file needs to contain only closed objects among:
    rectangles, circles, ellipses, polygons, and closed curves.
    The objects do not have to be simply connected.
    '''
    shapes, points = shapes_from_svg(
        filename, parent=parent, interpolate_curve=interpolate_curve,
        return_points=True)
    idx_main_container = 0
    idx_local = 0
    type_main_container = ''
    count = 0
    min_x_val = np.inf
    
    # the main container must own the smallest x value
    for elt_type, elements in points.items():
        for i, elt_points in enumerate(elements):
            min_x_tmp = elt_points[:, 0].min()
            if min_x_tmp < min_x_val:
                min_x_val = min_x_tmp
                idx_main_container = count
                idx_local = i
                type_main_container = elt_type
            count += 1
    
    # make sure that the main container contains all other shapes
    main_container = shapes.pop(idx_main_container)
    exterior = points[type_main_container].pop(idx_local)
    for shape in shapes:
        assert main_container.contains(shape), "Some shapes are not " +\
            "contained in the main container."
    
    # all remaining shapes are considered as boundaries for the interior
    interiors = [item.coords for item in main_container.interiors]
    for _, elements in points.items():
        for elt_points in elements:
            interiors.append(elt_points)

    # scale the shape
    if None not in (min_x, max_x):
        exterior = np.array(main_container.exterior.coords)
        leftmost = np.min(exterior[:, 0])
        rightmost = np.max(exterior[:, 0])
        scaling = (max_x - min_x) / (rightmost - leftmost)
        exterior *= scaling
        interiors = [np.multiply(l, scaling) for l in interiors]

    culture = Shape(exterior, interiors, unit=unit, parent=parent)
    return culture


# ----- #
# Tools #
# ----- #

def _build_struct(svg, container, elt_type, elt_properties):
    for elt in svg.getElementsByTagName(elt_type):
        if elt_type == 'path':
            #~ for s in elt.getAttribute('d').split('z'):
                #~ if s:
                    #~ container.append(s.lstrip() + 'z')
            container.append(elt.getAttribute('d'))
        else:
            struct = {}
            for item in elt_properties:
                struct[item] = float(elt.getAttribute(item))
            container.append(struct)


def _make_shape(elt_type, instructions, parent=None, interpolate_curve=50,
                return_points=False):
    container = None
    shell = []  # outer points defining the polygon's outer shell
    holes = []  # inner points defining holes

    if elt_type == "path":  # build polygons from custom paths
        path_data = parse_path(instructions)
        num_data = len(path_data)
        if not path_data.closed:
            raise RuntimeError("Only closed shapes accepted.")
        start = path_data[0].start
        points = shell  # the first path is the outer shell?
        for j, item in enumerate(path_data):
            if isinstance(item, (Arc, CubicBezier, QuadraticBezier)):
                for frac in np.linspace(0, 1, interpolate_curve):
                    points.append(
                        (item.point(frac).real, -item.point(frac).imag))
            else:
                points.append((item.start.real, -item.start.imag))
            # if the shell is closed, the rest defines holes
            if item.end == start and j < len(path_data) - 1:
                holes.append([])
                points = holes[-1]
                start = path_data[j+1].start
        container = Shape(shell, holes=holes)
        shell = np.array(shell)
    elif elt_type == "ellipse":  # build ellipses
        circle = Point((instructions["cx"], -instructions["cy"])).buffer(1)
        rx, ry = instructions["rx"], instructions["ry"]
        container = Shape.from_polygon(scale(circle, rx, ry), min_x=None)
    elif elt_type == "circle":  # build circles
        container = Shape.from_polygon(Point((instructions["cx"],
            -instructions["cy"])).buffer(instructions["r"]), min_x=None)
    elif elt_type == "rect":  # build rectangles
        x, y = instructions["x"], -instructions["y"]
        w, h = instructions["width"], -instructions["height"]
        shell = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        container = Shape(shell)
    else:
        raise RuntimeError("Unexpected element type: '{}'.".format(elt_type))

    if return_points:
        if len(shell) == 0:
            shell = np.array(container.exterior.coords)
        return container, shell

    return container

#!/usr/bin/env python
#-*- coding:utf-8 -*-

from xml.dom.minidom import parse
from svg.path import parse_path
from itertools import chain

import shapely
from shapely.affinity import scale
from shapely.geometry import Point, Polygon


'''
Shape generation from SVG files.
'''


def shapes_from_svg(filename):
    svg = parse(filename)

    # custom path instructions
    path_strings = [path.getAttribute('d')
                    for path in svg.getElementsByTagName('path')]
    # ellipse instructions
    ellipse_structs = []
    _build_struct(svg, ellipse_structs, 'ellipse', ("cx", "cy", "rx", "ry"))
    # circle instructions
    circle_structs = []
    _build_struct(svg, circle_structs, 'circle', ("cx", "cy", "r"))
    # rectangle instructions
    rectangle_structs = []
    _build_struct(
        svg, rectangle_structs, 'rect', ("x", "y", "width", "height"))

    # get points custom polygons
    polygons = []
    for i, path_string in enumerate(path_strings):
        path_data = parse_path(path_string)
        num_data = len(path_data)
        if not path_data.closed:
            raise RuntimeError("Only closed shapes accepted")
        points = np.zeros((num_data, 2))
        for j, item in enumerate(path_data):
            points[j, 0] = item.start.real
            points[j, 1] = -item.start.imag
        polygons.append(points)

    # build all shapes
    shapes = []
    for points in polygons:
        shapes.append(Polygon(points))
    for s in ellipse_structs:
        shapes.append(_make_shapely('ellipse', s))
    for s in circle_structs:
        shapes.append(_make_shapely('circle', s))
    for s in rectangle_structs:
        shapes.append(_make_shapely('rect', s))

    return shapes


def culture_from_svg(filename):
    pass


# ----- #
# Tools #
# ----- #

def _build_struct(svg, container, elt_type, elt_properties):
    for elt in svg.getElementsByTagName(elt_type):
        struct = {}
        for item in elt_properties:
            struct[item] = float(elt.getAttribute(item))
        container.append(struct)


def _test_min_x(biggest, min_x, structs, elt_type, elt_properties):
    for i, struct in enumerate(structs):
        min_x_tmp = struct[elt_properties[0]]
        if len(elt_properties) == 2:
            min_x_tmp -= struct[elt_properties[1]]
        if min_x_tmp < min_x:
            min_x = min_x_tmp
            biggest[0] = elt_type
            biggest[1] = i
    return min_x


def _make_shapely(elt_type, instructions):
    container = None
    if elt_type == "polygon":
        container = Polygon(instructions)
    elif elt_type == "ellipse":
        circle = Point((instructions["cx"], -instructions["cy"])).buffer(1)
        rx, ry = instructions["rx"], instructions["ry"]
        container = scale(circle, rx, ry)
    elif elt_type == "circle":
        container = Point((instructions["cx"],
            -instructions["cy"])).buffer(instructions["r"])
    elif elt_type == "rect":
        x, y = instructions["x"], -instructions["y"]
        w, h = instructions["width"], -instructions["height"]
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        container = Polygon(points)
    else:
        raise RuntimeError("Unexpected element type: '{}'.".format(elt_type))
    return container

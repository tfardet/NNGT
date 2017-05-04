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

# predefined svg shapes and their parameters
_predefined = {
    'path': None,
    'ellipse': ("cx", "cy", "rx", "ry"),
    'circle': ("cx", "cy", "r"),
    'rect': ("x", "y", "width", "height")
}


def shapes_from_svg(filename, return_points=False):
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
            polygon, points = _make_shapely(elt_type, struct)
            shapes.append(polygon)
            elt_points[elt_type].append(points)

    if return_points:
        return shapes, elt_points
    return shapes


def culture_from_svg(filename):
    '''
    Generate a culture from an SVG file.
    
    Valid file needs to contain only filled objects (no holes inside) among:
    rectangles, circles, ellipses, and closed paths (polygons).
    '''
    shapes, points = shapes_from_svg(filename, return_points=True)
    idx_main_container = 0
    idx_local = 0
    type_main_container = ''
    count = 0
    min_x = np.inf
    
    # the main container must own the smallest x value
    for elt_type, elements in points.items():
        for i, elt_points in enumerate(elements):
            min_x_tmp = elt_points[:, 0].min()
            if min_x_tmp < min_x:
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
    interiors = []
    for _, elements in points.items():
        for elt_points in elements:
            interiors.append(elt_points)
    culture = Polygon(exterior, interiors)
    return culture


# ----- #
# Tools #
# ----- #

def _build_struct(svg, container, elt_type, elt_properties):
    for elt in svg.getElementsByTagName(elt_type):
        struct = {}
        if elt_type == 'path':
            struct = path.getAttribute('d')
        else:
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


def _make_shapely(elt_type, instructions, return_points=False):
    container = None
    points = []
    if elt_type == "path":
        path_data = parse_path(instructions)
        num_data = len(path_data)
        if not path_data.closed:
            raise RuntimeError("Only closed shapes accepted")
        points = np.zeros((num_data, 2))
        for j, item in enumerate(path_data):
            points[j, 0] = item.start.real
            points[j, 1] = -item.start.imag
        container = Polygon(points)
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
        points = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        container = Polygon(points)
    else:
        raise RuntimeError("Unexpected element type: '{}'.".format(elt_type))
    if return_points:
        if len(points) == 0:
            points = np.array(container.exterior.coords)
        return container, points
    return container

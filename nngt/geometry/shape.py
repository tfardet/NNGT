#!/usr/bin/env python
#-*- coding:utf-8 -*-

'''
Shape implementation using the
`shapely<http://toblerity.org/shapely/index.html>`_ library.
'''

import weakref

import shapely
from shapely.affinity import scale
from shapely.geometry import Point, Polygon

import numpy as np
from numpy.random import uniform

from .geom_utils import conversion_magnitude
from nngt.lib import InvalidArgument


__all__ = ["Shape"]


class Shape(Polygon):
    """
    Class containing the shape of the area where neurons will be distributed to
    form a network.

    Attributes
    ----------
    area : double
        Area of the shape in the :class:`~nngt.geometry.Shape`'s
        :func:`~nngt.geometry.Shape.unit` squared (:math:`\mu m^2`,
        :math:`mm^2`, :math:`cm^2`, :math:`dm^2` or :math:`m^2`).
    centroid : tuple of doubles
        Position of the center of mass of the current shape in `unit`.

    See also
    --------
    Parent class: :class:`shapely.geometry.Polygon`
    """

    @staticmethod
    def from_svg(filename, min_x=-5000., max_x=5000., unit='um', parent=None,
                 interpolate_curve=50):
        '''
        Create a shape from an SVG file.
        
        Parameters
        ----------
        filename : str
            Path to the file that should be loaded.
        min_x : float, optional (default: -5000.)
            Absolute horizontal position of the leftmost point in the
            environment in `unit` (default: 'um'). If None, no rescaling
            occurs.
        max_x : float, optional (default: 5000.)
            Absolute horizontal position of the rightmost point in the
            environment in `unit`. If None, no rescaling occurs.
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'.
        parent : :class:`~nngt.Graph` object
            The parent which will become a :class:`~nngt.SpatialGraph`.
        interpolate_curve : int, optional (default: 50)
            Number of points that should be used to interpolate a curve.
        '''
        try:
            from .svgtools import culture_from_svg
            return culture_from_svg(
                filename,  min_x=min_x, max_x=max_x, unit=unit, parent=parent,
                interpolate_curve=interpolate_curve)
        except ImportError:
            raise ImportError("Install 'svg.path' to use this feature.")

    @classmethod
    def from_polygon(cls, polygon, min_x=-5000., max_x=5000., unit='um',
                     parent=None):
        '''
        Create a shape from a :class:`shapely.geometry.Polygon`.
        
        Parameters
        ----------
        polygon : :class:`shapely.geometry.Polygon`
            The initial polygon.
        min_x : float, optional (default: -5000.)
            Absolute horizontal position of the leftmost point in the
            environment in `unit` If None, no rescaling occurs.
        max_x : float, optional (default: 5000.)
            Absolute horizontal position of the rightmost point in the
            environment in `unit` If None, no rescaling occurs.
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'
        '''
        assert isinstance(polygon, Polygon), "Expected a Polygon object."
        # find the scaling factor
        scaling = 1.
        if None not in (min_x, max_x):
            ext = np.array(polygon.exterior.coords)
            leftmost = np.min(ext[:, 0])
            rightmost = np.max(ext[:, 0])
            scaling = (max_x - min_x) / (rightmost - leftmost)
        # create the newly scaled polygon and convert it to Shape
        p2 = scale(polygon, scaling, scaling)
        p2.__class__ = cls
        p2._parent = parent
        p2._unit = unit
        p2._geom_type = 'Polygon'
        return p2

    @classmethod
    def rectangle(cls, height, width, centroid=(0., 0.), unit='um',
                  parent=None):
        '''
        Generate a rectangle of given height, width and center of mass.

        Parameters
        ----------
        height : float
            Height of the rectangle in `unit`
        width : float
            Width of the rectangle in `unit`
        centroid : tuple of floats, optional (default: (0., 0.))
            Position of the rectangle's center of mass in `unit`
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'
        parent : :class:`~nngt.Graph` or subclass, optional (default: None)
            The parent container.

        Returns
        -------
        shape : :class:`~nngt.geometry.Shape`
            Rectangle shape.
        '''
        half_w = 0.5 * width
        half_h = 0.5 * height
        centroid = np.array(centroid)
        points = [centroid + [half_w, half_h],
                  centroid + [half_w, -half_h],
                  centroid - [half_w, half_h],
                  centroid - [half_w, -half_h]]
        shape = cls(points, unit=unit, parent=parent)
        shape._geom_type = "Rectangle"
        return shape

    @classmethod
    def disk(cls, radius, centroid=(0.,0.), unit='um', parent=None):
        '''
        Generate a disk of given radius and center (`centroid`).

        Parameters
        ----------
        radius : float
            Radius of the disk in `unit`
        centroid : tuple of floats, optional (default: (0., 0.))
            Position of the rectangle's center of mass in `unit`
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'
        parent : :class:`~nngt.Graph` or subclass, optional (default: None)
            The parent container.

        Returns
        -------
        shape : :class:`~nngt.geometry.Shape`
            Rectangle shape.
        '''
        centroid = np.array(centroid)
        shape = Shape.from_polygon(
            Point(centroid).buffer(radius), unit=unit, parent=parent)
        shape._geom_type = "Disk"
        shape.radius = radius
        return shape

    @classmethod
    def ellipse(cls, radii, centroid=(0.,0.), unit='um', parent=None):
        '''
        Generate a disk of given radius and center (`centroid`).

        Parameters
        ----------
        radii : tuple of floats
            Couple (rx, ry) containing the radii of the two axes in `unit`
        centroid : tuple of floats, optional (default: (0., 0.))
            Position of the rectangle's center of mass in `unit`
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'
        parent : :class:`~nngt.Graph` or subclass, optional (default: None)
            The parent container.

        Returns
        -------
        shape : :class:`~nngt.geometry.Shape`
            Rectangle shape.
        '''
        centroid = np.array(centroid)
        rx, ry = radii
        ellipse = cls.from_polygon(
            scale(Point(centroid).buffer(1.), rx, ry), unit=unit,
            parent=parent)
        shape._geom_type = "Ellipse"
        shape.radii = radii
        return shape

    def __init__(self, shell, holes=None, unit='um', parent=None):
        '''
        Initialize the :class:`~nngt.geometry.Shape` object and the underlying
        :class:`shapely.geometry.Polygon`.

        Parameters
        ----------
        exterior : array-like object of shape (N, 2)
            List of points defining the external border of the shape.
        interiors : array-like, optional (default: None)
            List of array-like objects of shape (M, 2), defining empty regions
            inside the shape.
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'.
        parent : :class:`~nngt.Graph` or subclass
            The graph which is associated to this Shape.
        '''
        self._parent = weakref.proxy(parent) if parent is not None else None
        self._unit = unit
        super(Shape, self).__init__(shell, holes=holes)
    
    @property
    def parent(self):
        ''' Return the parent of the :class:`~nngt.geometry.Shape`. '''
        return self._parent
    
    @property
    def unit(self):
        '''
        Return the unit for the :class:`~nngt.geometry.Shape` coordinates.
        '''
        return self._unit

    def set_parent(self, parent):
        ''' Set the parent :class:`~nngt.Graph`. '''
        self._parent = weakref.proxy(parent) if parent is not None else None

    def add_subshape(self, subshape, position, unit='um'):
        """
        Add a :class:`~nngt.geometry.Shape` to the current one.

        Parameters
        ----------
        subshape : :class:`~nngt.geometry.Shape`
            Subshape to add.
        position : tuple of doubles
            Position of the subshape's center of gravity in space.
        unit : string (default: 'um')
            Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'

        Returns
        -------
        None
        """
        raise NotImplementedError("To be implemented.")

    def seed_neurons(self, unit=None, neurons=None):
        '''
        Return the positions of the neurons inside the
        :class:`~nngt.geometry.Shape`.
        
        Parameters
        ----------
        unit : string (default: None)
            Unit in which the positions of the neurons will be returned, among
            'um', 'mm', 'cm', 'dm', 'm'.
        neurons : int, optional (default: None)
            Number of neurons to seed. This argument is considered only if the
            :class:`~nngt.geometry.Shape` has no `parent`, otherwise, a
            position is generated for each neuron in `parent`.
        
        Returns
        -------
        positions : array of double with shape (N, 2)
        '''
        positions = None
        if self._parent is not None:
            neurons = self._parent.node_nb()
        if neurons is None:
            raise InvalidArgument("`neurons` cannot be None if `parent` is.")
        if self.geom_type == "Rectangle":
            points = self._convex_hull.points
            min_x, max_x = points[:,0].min(), points[:,0].max()
            min_y, max_y = points[:,1].min(), points[:,1].max()
            ra_x = uniform(min_x, max_x, size=neurons)
            ra_y = uniform(min_y, max_y, size=neurons)
            positions = np.vstack((ra_x, ra_y))
        elif self.geom_type == "Disk":
            theta = uniform(0, 2*np.pi, size=neurons)
            r = uniform(0, self.radius, size=neurons)
            positions = np.vstack((r*np.cos(theta), r*np.sin(theta)))
        elif self.geom_type == "Ellipse":
            theta = uniform(0, 2*np.pi, size=neurons)
            r = uniform(0, 1, size=neurons)
            rx, ry = self.radii
            positions = np.vstack((rx*r*np.cos(theta), ry*r*np.sin(theta)))
        else:
            points = []
            min_x, min_y, max_x, max_y = self.bounds
            p = Point()
            while len(points) < neurons:
                new_x = uniform(min_x, max_x, neurons-len(points))
                new_y = uniform(min_y, max_y, neurons-len(points))
                for x, y in zip(new_x, new_y):
                    p.coords = (x, y)
                    if self.contains(p):
                        points.append((x, y))
            positions = np.array(points)

        if unit is not None and unit != self._unit:
            positions *= conversion_magnitude(unit, self._unit)

        return positions

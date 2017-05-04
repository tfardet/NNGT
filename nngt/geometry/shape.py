#!/usr/bin/env python
#-*- coding:utf-8 -*-

'''
Shape implementation using the
`shapely<http://toblerity.org/shapely/index.html>`_ library.
'''

import shapely
from shapely.affinity import scale
from shapely.geometry import MultiPoint, Point, Polygon

import numpy as np
from numpy.random import uniform


__all__ = ["Shape"]


class Shape(Polygon):
    """
    Class containing the shape of the area where neurons will be distributed to
    form a network.

    Attributes
    ----------
    area: double
        Area of the shape in mm^2.
    com: tuple of doubles
        Position of the center of mass of the current shape.

    Methods
    -------
    add_subshape: void
        @todo
        Add a :class:`~nngt.Shape` to a preexisting one.

    See also
    --------
    Parent class: :class:`shapely.geometry.Polygon`
    """

    @classmethod
    def from_polygon(cls, polygon, parent=None):
        '''
        Create a shape from a :class:`shapely.geometry.Polygon`.
        '''
        assert isinstance(polygon, Polygon), "Expected a Polygon object."
        polygon.__class__ = cls
        polygon.parent = parent
        polygon._geom_type = 'Polygon'
        return polygon

    @classmethod
    def rectangle(cls, parent, height, width, centroid=(0., 0.)):
        '''
        Generate a rectangle of given height, width and center of mass.

        Parameters
        ----------
        parent : :class:`~nngt.SpatialGraph` or subclass
            The parent container.
        height : float
            Height of the rectangle.
        width : float
            Width of the rectangle.
        pos_com : tuple of floats, optional (default: (0., 0.))
            Position of the rectangle's center of mass

        Returns
        -------
        shape : :class:`~nngt.Shape`
            Rectangle shape.
        '''
        half_w = 0.5 * width
        half_h = 0.5 * height
        centroid = np.array(centroid)
        points = [centroid + [half_w, half_h],
                  centroid + [half_w, -half_h],
                  centroid - [half_w, half_h],
                  centroid - [half_w, -half_h]]
        shape = cls(points, parent=parent)
        shape._geom_type = "Rectangle"
        return shape

    @classmethod
    def disk(cls, parent, radius, centroid=(0.,0.)):
        '''
        Generate a disk of given radius and center (`centroid`).

        Parameters
        ----------
        parent : :class:`~nngt.SpatialGraph` or subclass
            The parent container.
        height : float
            Height of the rectangle.
        width : float
            Width of the rectangle.
        centroid : tuple of floats, optional (default: (0., 0.))
            Position of the rectangle's center of mass

        Returns
        -------
        shape : :class:`~nngt.Shape`
            Rectangle shape.
        '''
        centroid = np.array(centroid)
        circle = Point(centroid).buffer(radius)
        points = [(c[0], c[1]) for c in circle.coords]
        shape(points, parent=parent)
        shape._geom_type = "Disk"
        shape.radius = radius
        return shape

    def __init__(self, shell, holes=None, parent=None):
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
        parent : :class:`~nngt.SpatialGraph` or subclass
            The network which is associated to this Shape.
        '''
        self.parent = weakref.proxy(parent) if parent is not None else None
        super(Shape, self).__init__(shell, holes=holes)

    def set_parent(self, parent):
        self.parent = weakref.proxy(parent) if parent is not None else None

    def add_subshape(self, subshape, position, unit='mm'):
        """
        Add a :class:`~nngt.core.Shape` to the current one.

        Parameters
        ----------
        subshape: :class:`~nngt.Shape`
            Subshape to add.
        position: tuple of doubles
            Position of the subshape's center of gravity in space.
        unit: string (default 'mm')
            Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'

        Returns
        -------
        None
        """
        pass

    def rnd_distrib(self, nodes=None):
        if self.parent is not None:
            nodes = self.parent.node_nb()
        if self.geom_type == "Rectangle":
            points = self._convex_hull.points
            min_x, max_x = points[:,0].min(), points[:,0].max()
            min_y, max_y = points[:,1].min(), points[:,1].max()
            ra_x = uniform(min_x, max_x, size=nodes)
            ra_y = uniform(min_y, max_y, size=nodes)
            return np.vstack((ra_x, ra_y))
        elif self.geom_type == "Disk":
            theta = uniform(0, 2*np.pi, size=nodes)
            r = uniform(0, self.radius, size=nodes)
            return np.vstack((r*np.cos(theta), r*np.sin(theta)))
        else:
            points = []
            min_x, min_y, max_x, max_y = self.bounds
            p = Point()
            while len(points) < nodes:
                new_x = uniform(min_x, max_x, nodes-len(points))
                new_y = uniform(min_y, max_y, nodes-len(points))
                #~ points = MultiPoint(np.vstack((new_x, new_y)).T)
                for x, y in zip(new_x, new_y):
                    p.coords = (x, y)
                    if self.contains(p):
                        points.append((x, y))
            return np.array(points)

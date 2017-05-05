#!/usr/bin/env python
#-*- coding:utf-8 -*-

'''
Backup Shape implementation using scipy.
'''

from nngt.lib import InvalidArgument


class Shape:
    """
    Class containing the shape of the area where neurons will be distributed to
    form a network.

    ..warning :
        so far, only a rectangle or a disk can be created.

    Attributes
    ----------
    area: double
        Area of the shape in mm^2.
    centroid: tuple of doubles
        Position of the center of mass of the current shape.

    Methods
    -------
    add_subshape: void
        @todo
        Add a :class:`~nngt.geometry.Shape` to a preexisting one.
    """

    @classmethod
    def rectangle(cls, parent, height, width, centroid=(0.,0.)):
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
        shape : :class:`~nngt.geometry.Shape`
            Rectangle shape.
        '''
        shape = cls(parent)
        half_w = 0.5 * width
        half_h = 0.5 * height
        centroid = np.array(centroid)
        points = [centroid + [half_w, half_h],
                  centroid + [half_w, -half_h],
                  centroid - [half_w, half_h],
                  centroid - [half_w, -half_h]]
        shape._convex_hull = sptl.Delaunay(points)
        shape._com = centroid
        shape._area = height * width
        shape._bounds = (points[2][0], points[2][1],
                         points[0][0], points[0][1])
        shape._length = 2*width + 2*height
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
        shape : :class:`~nngt.geometry.Shape`
            Rectangle shape.
        '''
        shape = cls(parent)
        centroid = np.array(centroid)
        points = [(centroid + radius*np.cos(theta),
                  centroid + radius*np.sin(theta))
                  for theta in np.linspace(0, 2*np.pi, 100)]
        shape._convex_hull = sptl.Delaunay(points)
        shape._com = centroid
        shape._area = np.pi * np.square(radius)
        shape._bounds = (centroid[0] - radius, centroid[1] - radius,
                         centroid[0] + radius, centroid[1] + radius)
        shape._length = 2 * np.pi * radius
        shape._geom_type = "Disk"
        shape.radius = radius
        return shape

    def __init__(self, parent=None):
        self.parent = weakref.proxy(parent) if parent is not None else None
        self._area = 0.
        self._com = (0., 0.)
        self._convex_hull = None

    @property
    def area(self):
        ''' Area of the shape. '''
        return self._area

    @property
    def centroid(self):
        ''' Centroid of the shape. '''
        return self._com

    @property
    def centroid(self):
        ''' Centroid of the shape. '''
        return self._com

    @property
    def coords(self):
        return self._convex_hull.points

    @property
    def geom_type(self):
        return self._geom_type

    def set_parent(self, parent):
        self.parent = weakref.proxy(parent) if parent is not None else None

    def add_subshape(self, subshape, position, unit='mm'):
        """
        Add a :class:`~nngt.geometry.Shape` to the current one.

        Parameters
        ----------
        subshape: :class:`~nngt.geometry.Shape`
            Subshape to add.
        position: tuple of doubles
            Position of the subshape's center of gravity in space.
        unit: string (default 'mm')
            Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'

        Returns
        -------
        None
        """
        raise NotImplementedError

    def seed_neurons(self, nodes=None):
        #@todo: make it general
        if self.parent is not None:
            nodes = self.parent.node_nb()
        if self._geom_type == "Rectangle":
            points = self._convex_hull.points
            min_x, max_x = points[:,0].min(), points[:,0].max()
            min_y, max_y = points[:,1].min(), points[:,1].max()
            ra_x = uniform(min_x, max_x, size=nodes)
            ra_y = uniform(min_y, max_y, size=nodes)
            return np.vstack((ra_x, ra_y))
        elif self._geom_type == "Disk":
            theta = uniform(0, 2*np.pi, size=nodes)
            r = uniform(0, self.radius, size=nodes)
            return np.vstack((r*np.cos(theta), r*np.sin(theta)))
        else:
            raise RuntimeError("Unsupported: '{}'.".format(self._geom_type))

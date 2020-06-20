#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# spatial_graph.py
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" SpatialGraph class for spatial graph generation and management """

import numpy as np

import nngt
from .graph import Graph


# ------------ #
# SpatialGraph #
# ------------ #

class SpatialGraph(Graph):

    """
    The detailed class that inherits from :class:`~nngt.Graph` and implements
    additional properties to describe spatial graphs (i.e. graph where the
    structure is embedded in space.
    """

    #-------------------------------------------------------------------------#
    # Class properties

    __num_graphs = 0
    __max_id = 0

    #-------------------------------------------------------------------------#
    # Constructor, destructor, attributes

    def __init__(self, nodes=0, name="SpatialGraph", weighted=True,
                 directed=True, from_graph=None, shape=None, positions=None,
                 **kwargs):
        '''
        Initialize SpatialClass instance.

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`Graph` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        shape : :class:`~nngt.geometry.Shape`, optional (default: None)
            Shape of the neurons' environment (None leads to a square of
            side 1 cm)
        positions : :class:`numpy.array` (N, 2), optional (default: None)
            Positions of the neurons; if not specified and `nodes` is not 0,
            then neurons will be reparted at random inside the
            :class:`~nngt.geometry.Shape` object of the instance.
        **kwargs : keyword arguments for :class:`~nngt.Graph` or
            :class:`~nngt.geometry.Shape` if no shape was given.

        Returns
        -------
        self : :class:`~nggt.SpatialGraph`
        '''
        self.__id = self.__class__.__max_id
        self.__class__.__num_graphs += 1
        self.__class__.__max_id += 1

        self._shape = None
        self._pos   = None

        super().__init__(nodes, name, weighted, directed, from_graph, **kwargs)

        self._init_spatial_properties(shape, positions, **kwargs)

        if "population" in kwargs:
            self.make_network(self, kwargs["population"])

    def __del__(self):
        if hasattr(self, '_shape'):
            if self._shape is not None:
                self._shape._parent = None
            self._shape = None

        super().__del__()

        self.__class__.__num_graphs -= 1

    @property
    def shape(self):
        ''' The environment's spatial structure. '''
        return self._shape

    #-------------------------------------------------------------------------#
    # Init tool

    def _init_spatial_properties(self, shape, positions=None, **kwargs):
        '''
        Create the positions of the neurons from the graph `shape` attribute
        and computes the connections distances.
        '''
        positions = None if positions is None else np.asarray(positions)

        self.new_edge_attribute('distance', 'double')

        if positions is not None and len(positions) != self.node_nb():
            raise InvalidArgument("Wrong number of neurons in `positions`.")

        if shape is not None:
            shape.set_parent(self)
            self._shape = shape
        else:
            if positions is None or not np.any(positions):
                if 'height' in kwargs and 'width' in kwargs:
                    self._shape = nngt.geometry.Shape.rectangle(
                        kwargs['height'], kwargs['width'], parent=self)
                elif 'radius' in kwargs:
                    self._shape = nngt.geometry.Shape.disk(
                        kwargs['radius'], parent=self)
                elif 'radii' in kwargs:
                    self._shape = nngt.geometry.Shape.ellipse(
                        kwargs['radii'], parent=self)
                elif 'polygon' in kwargs:
                    self._shape = nngt.geometry.Shape.from_polygon(
                        kwargs['polygon'], min_x=kwargs.get('min_x', -5000.),
                        max_x=kwargs.get('max_x', 5000.),
                        unit=kwargs.get('unit', 'um'), parent=self)
                else:
                    raise RuntimeError('SpatialGraph needs a `shape` or '
                                       'keywords arguments to build one, or '
                                       'at least `positions` so it can create '
                                       'a square containing them')
            else:
                minx, maxx = np.min(positions[:, 0]), np.max(positions[:, 0])
                miny, maxy = np.min(positions[:, 1]), np.max(positions[:, 1])

                height, width = 1.01*(maxy - miny), 1.01*(maxx - minx)

                centroid = (0.5*(maxx + minx), 0.5*(maxy + miny))

                self._shape = nngt.geometry.Shape.rectangle(
                    height, width, centroid=centroid, parent=self)

        b_rnd_pos = True if not self.node_nb() or positions is None else False
        self._pos = self._shape.seed_neurons() if b_rnd_pos else positions

        nngt.core.Connections.distances(self)

    #-------------------------------------------------------------------------#
    # Getters

    def get_positions(self, neurons=None):
        '''
        Returns a copy of the neurons' positions as a (N, 2) array.

        Parameters
        ----------
        neurons : int or array-like, optional (default: all neurons)
            List of the neurons for which the position should be returned.
        '''
        if neurons is not None:
            if isinstance(neurons, tuple):
                # numpy mulitple slicing does not work with tuples
                neurons = list(neurons)

            return np.array(self._pos[neurons])

        return np.array(self._pos)

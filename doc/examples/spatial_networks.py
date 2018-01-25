#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
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

''' SpatialNetwork generation, complex shapes '''

import os
import time

import numpy as np

import nngt
# ~ from nngt.geometry import Shape
from PyNCulture import Shape

# nngt.seed(0)

nngt.set_config({"omp": 8, "palette": 'RdYlBu'})


''' Create a circular shape and add obstacles inside '''

shape = Shape.disk(radius=2500.)

params           = {"height": 250., "width": 250.}
filling_fraction = 0.4

shape.random_obstacles(filling_fraction, form="rectangle", params=params,
                       heights=30., etching=20.)


''' Create a Spatial network and seed neurons on top/bottom areas '''

pop = nngt.NeuralPop(100)
pop.set_model(None)
num_top    = 100
num_bottom = 100
pop.create_group("top", num_top)
pop.create_group("bottom", num_bottom)

net = nngt.SpatialGraph(shape=shape, population=pop)

# seed neurons
bottom_pos     = shape.seed_neurons(num_bottom,
                                    on_area=shape.default_areas,
                                    soma_radius=15)
bottom_neurons = np.array(net.new_node(num_bottom, positions=bottom_pos,
                                       groups="bottom"), dtype=int)
top_pos        = shape.seed_neurons(num_top, on_area=shape.non_default_areas,
                                    soma_radius=15)
top_neurons    = np.array(net.new_node(num_top, positions=top_pos,
                                       groups="top"), dtype=int)


''' Make the connectivity '''

scale = 100.

# connect bottom area
avg_deg = 0.1 * len(bottom_neurons)
nngt.generation.connect_nodes(net, bottom_neurons, bottom_neurons,
                              "distance_rule", scale=scale, avg_deg=avg_deg)
print("bottom-bottom done")

# connect top areas
for name, area in shape.non_default_areas.items():
    print(name)
    contained = area.contains_neurons(top_pos)
    neurons   = top_neurons[contained]
    if np.any(neurons):
        # connect intra-area
        avg_deg   = 0.4*len(neurons)
        nngt.generation.connect_nodes(net, neurons, neurons, "distance_rule",
                                      scale=scale, avg_deg=avg_deg)
        print("top-top done")
        # connect top to bottom and bottom to top
        edges = 0.08 * len(bottom_neurons) * len(neurons)
        nngt.generation.connect_nodes(net, neurons, bottom_neurons,
                                      "distance_rule", scale=3*scale,
                                      edges=edges)
        print("top-down done")
        avg_deg = 0.02 * len(bottom_neurons) * len(neurons)
        nngt.generation.connect_nodes(net, bottom_neurons, neurons,
                                      "distance_rule", scale=3*scale,
                                      edges=edges)
        print("bottom-up done")

nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                       restrict_sources="top", restrict_targets="top",
                       show=False)

nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                       restrict_sources="top", restrict_targets="bottom",
                       show=False)

nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                       restrict_sources="bottom", restrict_targets="top",
                       show=False)

nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                       restrict_sources="bottom", restrict_targets="bottom",
                       show=False)

nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5, show=True)

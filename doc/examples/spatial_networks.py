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

import numpy as np

import nngt
from nngt.geometry import Shape


nngt.set_config({"omp": 8, "palette": 'RdYlBu'})

# ~ nngt.seed(0)


''' Create a circular shape and add obstacles inside '''

shape = Shape.disk(radius=2500.)

params           = {"height": 250., "width": 250.}
filling_fraction = 0.4

shape.random_obstacles(filling_fraction, form="rectangle", params=params,
                       heights=30., etching=20.)


''' Create a Spatial network and seed neurons on top/bottom areas '''

num_neurons = 500

# neurons are reparted proportionaly to the area
total_area  = shape.area
bottom_area = np.sum([a.area for a in shape.default_areas.values()])

num_bottom  = int(num_neurons * bottom_area / total_area)
num_top     = num_neurons - num_bottom

pop = nngt.NeuralPop(num_neurons)
pop.set_model(None)
pop.create_group("top", num_top)
pop.create_group("bottom", num_bottom)

# make the graph
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

top_scale    = 200.
bottom_scale = 100.
mixed_scale  = 150.

base_proba   = 3.
p_up         = 0.6
p_down       = 0.9
p_other_up   = p_down**2

# connect bottom area
for name, area in shape.default_areas.items():
    contained = area.contains_neurons(bottom_pos)
    neurons   = bottom_neurons[contained]
    nngt.generation.connect_nodes(net, bottom_neurons, bottom_neurons,
                                  "distance_rule", scale=bottom_scale,
                                  max_proba=base_proba)

# connect top areas
for name, area in shape.non_default_areas.items():
    contained = area.contains_neurons(top_pos)
    neurons   = top_neurons[contained]
    other_top = [n for n in top_neurons if n not in neurons]
    if np.any(neurons):
        # connect intra-area
        nngt.generation.connect_nodes(net, neurons, neurons, "distance_rule",
                                      scale=top_scale, max_proba=base_proba)
        # connect between top areas (do it?)
        nngt.generation.connect_nodes(net, neurons, other_top, "distance_rule",
                                      scale=mixed_scale,
                                      max_proba=base_proba*p_other_up)
        # connect top to bottom
        nngt.generation.connect_nodes(net, neurons, bottom_neurons,
                                      "distance_rule", scale=mixed_scale,
                                      max_proba=base_proba*p_down)
        # connect bottom to top
        nngt.generation.connect_nodes(net, bottom_neurons, neurons,
                                      "distance_rule", scale=mixed_scale,
                                      max_proba=base_proba*p_up)


''' Plot if available '''

if nngt.get_config("with_plot"):

    ''' Check the degree distribution '''

    nngt.plot.degree_distribution(
        net, ["in", "out"], nodes=top_neurons, show=False)
    nngt.plot.degree_distribution(
        net, ["in", "out"], nodes=bottom_neurons, show=True)


    ''' Plot the resulting network and subnetworks '''

    restrict = [
        ("bottom", "bottom"), ("top", "bottom"), ("top", "top"),
        ("bottom", "top")
    ]

    for r_source, r_target in restrict:
        nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                               restrict_sources=r_source,
                               restrict_targets=r_target, show=False)

    fig, axis = plt.subplots()
    count = 0
    for r_source, r_target in restrict:
        show_env = (count == 0)
        nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                               restrict_sources=r_source,
                               restrict_targets=r_target,
                               show_environment=show_env, axis=axis, show=False)
        count += 1

    plt.show()
# ~ nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5, show=True)

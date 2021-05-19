#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2020  Tanguy Fardet
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

"""
Chord diagram
=============
"""

import matplotlib.pyplot as plt

import nngt

plt.rcParams["figure.facecolor"] = (0, 0, 0, 0)

nngt.seed(0)


# create a structured graph

room1 = nngt.Group(25)
room2 = nngt.Group(50)
room3 = nngt.Group(40)
room4 = nngt.Group(35)

names = ["R1", "R2", "R3", "R4"]

struct = nngt.Structure.from_groups((room1, room2, room3, room4), names)

g = nngt.Graph(structure=struct)

for room in struct:
    nngt.generation.connect_groups(g, room, room, "all_to_all")

nngt.generation.connect_groups(g, (room1, room2), struct, "erdos_renyi",
                               avg_deg=10, ignore_invalid=True)

nngt.generation.connect_groups(g, room3, room1, "erdos_renyi", avg_deg=20)

nngt.generation.connect_groups(g, room4, room3, "erdos_renyi", avg_deg=10)


# get the structure graph and plot

sg = g.get_structure_graph()

nngt.plot.chord_diagram(sg, names="name", sort="distance", fontcolor="grey",
                        use_gradient=True, show=True)

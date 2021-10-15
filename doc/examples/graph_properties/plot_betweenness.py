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
Plot the betweenness distributions of a graph
=============================================
"""

import nngt
import nngt.plot as nplt
from nngt.geometry import Shape

import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.edgecolor': 'grey', 'xtick.color': 'grey', 'ytick.color': 'grey',
    "figure.facecolor": (0, 0, 0, 0), "axes.facecolor": (0, 0, 0, 0),
    "axes.labelcolor": "grey", "text.color": "grey", "legend.facecolor": "none"
})


nngt.seed(0)


# %%
# Let's start by making a random exponential graph

shape = Shape.disk(100)

g = nngt.generation.distance_rule(5, shape=shape, nodes=1000, avg_deg=3)


# %%
# then we can plot the betweenness

nplt.betweenness_distribution(g, logx=True, show=True, legend_location='left')

# %%
# we can of course change various parameters and plot only the nodes

nplt.betweenness_distribution(g, logx=False, show=True)

nplt.betweenness_distribution(g, btype="node", num_nbins="auto", alpha=0.5,
                              show=True)

# %%
# By the way, this is the graph we're looking at

nplt.draw_network(g, max_nsize=5, max_esize=4, ecolor="grey", eborder_color="w",
                  curved_edges=True, show_environment=False, show=True)

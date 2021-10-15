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
Plot the degree distributions of a graph
========================================
"""

import nngt
import nngt.plot as nplt

import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.edgecolor': 'grey', 'xtick.color': 'grey', 'ytick.color': 'grey',
    "figure.facecolor": (0, 0, 0, 0), "axes.facecolor": (0, 0, 0, 0),
    "axes.labelcolor": "grey", "text.color": "grey", "legend.facecolor": "none"
})


nngt.seed(0)


# %%
# First, let's create a scale-free network

g = nngt.generation.random_scale_free(2.1, 3.2, nodes=1000, avg_deg=100)


# %%
# Plot the degree distribution

nplt.degree_distribution(g, deg_type=["in", "out"], show=True)

# %%
# It's not bad... but we don't see much! Let's move a more relevant scale

nplt.degree_distribution(g, deg_type=["in", "out"], logy=True, show=True)


# %%
# Or we can use Bayesian binning

nplt.degree_distribution(g, deg_type=["in", "out"], num_bins="bayes",
                         show=True)

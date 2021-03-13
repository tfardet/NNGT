#-*- coding:utf-8 -*-
#
# plot/custom_plt.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Matplotlib customization """

import itertools
import logging

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as clrs
from matplotlib.markers import MarkerStyle as MS

import nngt
from nngt.lib.logger import _log_message


logger = logging.getLogger(__name__)

# ---------------- #
# Customize PyPlot #
# ---------------- #

with_seaborn = False

def palette_continuous(numbers=None):
    pal = cm.get_cmap(nngt._config["palette_continuous"])
    if numbers is None:
        return pal
    else:
        return pal(numbers)

def palette_discrete(numbers=None):
    pal = cm.get_cmap(nngt._config["palette_discrete"])
    if numbers is None:
        return pal
    else:
        return pal(numbers)

# markers list
markers = [m for m in MS().filled_markers if m != '.']

if nngt._config["color_lib"] == "seaborn":
    try:
        import seaborn as sns
        with_seaborn = True
        sns.set_style("whitegrid")

        def sns_palette(c):
            if isinstance(c, float):
                pal = sns.color_palette(nngt._config["palette"], 100)
                return pal[int(c*100)]
            else:
                return sns.color_palette(nngt._config["palette"], len(c))

        palette_continuous = sns_palette
    except ImportError as e:
        _log_message(logger, "WARNING",
                     "`seaborn` requested but could not set it: {}.".format(e))


if not with_seaborn:
    try:
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['font.family'] = 'serif'
        if nngt._config['use_tex']:
            mpl.rc('text', usetex=True)
        mpl.rcParams['axes.labelsize'] = mpl.rcParams['font.size']
        mpl.rcParams['axes.titlesize'] = 1.2*mpl.rcParams['font.size']
        mpl.rcParams['legend.fontsize'] = mpl.rcParams['font.size']
        mpl.rcParams['xtick.labelsize'] = mpl.rcParams['font.size']
        mpl.rcParams['ytick.labelsize'] = mpl.rcParams['font.size']
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['savefig.format'] = 'pdf'
        mpl.rcParams['xtick.major.size'] = 3
        mpl.rcParams['xtick.minor.size'] = 3
        mpl.rcParams['xtick.major.width'] = 1
        mpl.rcParams['xtick.minor.width'] = 1
        mpl.rcParams['ytick.major.size'] = 3
        mpl.rcParams['ytick.minor.size'] = 3
        mpl.rcParams['ytick.major.width'] = 1
        mpl.rcParams['ytick.minor.width'] = 1
        mpl.rcParams['legend.frameon'] = False
        mpl.rcParams['legend.numpoints'] = 1
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['grid.linestyle'] = ':'
        mpl.rcParams['path.simplify'] = True
    except Exception as e:
        _log_message(logger, "WARNING",
                     "Error configuring `matplotlib`: {}.".format(e))


def format_exponent(ax, axis='y', pos=(1.,0.), valign="top", halign="right"):
    import matplotlib.pyplot as plt
    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-3, 2))
    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
    else:
        ax_axis = ax.xaxis
    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    ##### THIS IS A BUG 
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of 
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()
    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' %expo
        # Turn off the offset text that's calculated automatically
        ax_axis.offsetText.set_visible(False)
        ax.text(pos[0], pos[1], offset_text, transform=ax.transAxes,
               horizontalalignment=halign,
               verticalalignment=valign)
    return ax

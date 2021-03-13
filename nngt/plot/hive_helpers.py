#-*- coding:utf-8 -*-
#
# plot/hive_helpers.py
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

from matplotlib import cm
from matplotlib.colors import ColorConverter
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import numpy as np

from ..lib.test_functions import nonstring_container
from .chord_diag.gradient import linear_gradient


__all__ = [
    "_get_ax_angles",
    "_get_axes_nodes",
    "_get_axes_radial_coord",
    "_get_colors",
    "_get_radial_values",
    "_get_size",
    "_plot_bezier",
    "_plot_nodes",
    "_set_names_lims",
    "RMIN"
]


RMIN = 0.1


def _get_axes_radial_coord(radial, axes, axes_bins, network):
    ''' Compute the number of axes and radial coordinates '''
    num_axes, num_radial = None, None

    # find the number of axes and radial coordinates
    if nonstring_container(radial) and len(radial) > 0:
        is_string = [isinstance(elt, str) for elt in radial]

        all_string = np.all(is_string)
        any_string = np.any(is_string)
        any_contnr = np.any([nonstring_container(elt) for elt in radial])

        if all_string:
            num_axes = num_radial = len(radial)
        elif any_string or any_contnr:
            raise ValueError("`radial` must be either a (list of) string or "
                             "a custom array of numbers.")
        else:
            num_radial = 1
    elif isinstance(radial, str):
        num_radial = 1
    else:
        raise ValueError("`radial` must be a str, list of str, or an array.")

    if nonstring_container(axes) and len(axes) > 0:
        for elt in axes:
            assert isinstance(elt, str), \
                "An `axes` list can only contain attribute names."

        assert axes_bins is None, \
            "If `axes` are given by different attributes, `axes_bins` must " \
            "be None."

        num_axes = len(axes)
    elif axes == "groups":
        assert network.structure is not None, \
            "A structure or population is required if `axes` is 'groups'."

        assert axes_bins is None, "No bins can be used if `axes` is 'groups'."

        num_axes = len(network.structure)
    elif isinstance(axes, str):
        assert axes in network.node_attributes, "Unknown `axes` attribute."

        if nonstring_container(axes_bins):
            num_axes = len(axes_bins) - 1
        elif isinstance(axes_bins, (int, np.integer)):
            num_axes = axes_bins
        else:
            raise ValueError("If `axes` is a str, then `axes_bins` must be a"
                             "valid list/number of bins.")
    else:
        raise ValueError("`axes` must be a str or list of str.")

    if num_radial > 1:
        assert num_radial == num_axes, \
            "If there is more than one radial coordinate, there must be one " \
            "per axis."

    assert num_axes >= 2, "There must be at least two axes."

    return num_axes, num_radial


def _get_axes_nodes(network, radial, axes, axes_bins, num_axes, num_radial):
    ''' Get the axes names and associated nodes and coordinates '''
    ax_names = []
    ax_nodes = []
    ax_radco = []

    if num_radial == 1:
        if axes_bins is None:
            struct = network.structure

            # each axis is a node attribute or a group
            if axes == 'groups':
                ax_names = list(network.structure)
                ax_nodes = [g.ids for g in network.structure.values()]
                ax_radco = (radial if nonstring_container(radial)
                            else network.node_attributes[radial]
                            for _ in range(num_axes))
            elif struct is not None and axes[0] in struct:
                for elt in axes:
                    assert elt in struct, \
                        "'{}' is not in the graph Structure.".format(elt)

                ax_names = list(axes)
                ax_nodes = [struct[n].ids for n in axes]
                ax_radco = (radial if nonstring_container(radial)
                            else network.node_attributes[radial]
                            for _ in range(num_axes))
            else:
                ax_names = list(axes)
                ax_nodes = [range(network.node_nb())]*num_axes
                ax_radco = [network.node_attributes[attr] for attr in axes]
        else:
            ax_radco = (radial if nonstring_container(radial)
                        else network.node_attributes[radial]
                        for _ in range(num_axes))

            axcoord = network.node_attributes[axes]

            # find node bin
            bins = axes_bins

            if isinstance(axes_bins, (int, np.integer)):
                bins = np.linspace(axcoord.min(), axcoord.max(), axes_bins + 1)

                bins[-1] += 0.01 * (bins[-1] - bins[0])

            nbin = np.digitize(axcoord, bins)

            for i in range(1, len(bins)):
                ax_nodes.append(np.where(nbin == i)[0])

            # each axis corresponds to a range of the node attribute
            for start, stop in zip(bins[:], bins[1:]):
                ax_names.append("{name}\nin [{start:.3g}, {stop:.3g}]".format(
                    name=axes, start=start, stop=stop))
    else:
        # each axis is a different radial coordinate
        ax_names = list(radial)
        ax_nodes = [range(network.node_nb())]*num_axes
        ax_radco = (network.node_attributes[attr] for attr in radial)

    return ax_names, ax_nodes, ax_radco


def _get_radial_values(ax_radco, axes_units, network):
    # r should vary between RMIN and 1 + RMIN
    radial_values = []

    if axes_units == "normed":
        for i, val in enumerate(ax_radco):
            vmin = val.min()
            vmax = val.max()
            nval = RMIN + (val - vmin) / (vmax - vmin)

            radial_values.append(nval)
    elif axes_units == "rank":
        num_nodes = network.node_nb()
        rmin = int(RMIN*num_nodes)

        for i, val in enumerate(ax_radco):
            ranks = np.argsort(np.argsort(val))
            radial_values.append((ranks + rmin) / num_nodes)
    elif axes_units == "native":
        vmin, vmax = np.inf, -np.inf

        for i, val in enumerate(ax_radco):
            vmin = min(vmin, val.min())
            vmax = max(vmax, val.max())

            # store the values as ax_radco may be a single use generator
            radial_values.append(val)

        for i, val in enumerate(radial_values):
            radial_values[i] = RMIN + (val - vmin) / (vmax - vmin)
    else:
        raise ValueError("Invalid `axes_units`: '{}'.".format(axes_units))

    return radial_values


def _smallest_angle(a1, a2):
    dtheta = np.abs(a1 - a2)

    if dtheta > np.pi:
        return 2*np.pi - dtheta

    return dtheta


def _get_ax_angles(angles, i, j, intra_connections):
    if intra_connections:
        # also works for the intra connections (i = j)
        as1 = angles[2*i]
        as2 = angles[2*i + 1]
        at1 = angles[2*j]
        at2 = angles[2*j + 1]

        if _smallest_angle(as1, at2) <= _smallest_angle(as2, at1):
            return 2*i, 2*j + 1

        return 2*i + 1, 2*j

    return i, j


def _get_size(node_size, max_nsize, ax_nodes, network):
    if node_size is None:
        max_nodes = np.max([len(nn) for nn in ax_nodes])
        return np.repeat(max(400/max_nodes, 4), network.node_nb())
    elif nonstring_container(node_size):
        assert len(node_size) == network.node_nb(), \
            "One size per node is required for array-like `node_size`."

        return np.array(node_size) / np.max(node_size) * max_nsize
    elif node_size in network.node_attributes:
        node_size = network.node_attributes[node_size]

        return node_size * (max_nsize / node_size.max())
    elif isinstance(node_size, float):
        return np.repeat(node_size, network.node_nb())

    raise ValueError("`nsize` must be float, attribute name, or array-like")


def _get_colors(axes_colors, edge_colors, angles, num_axes, intra_connections,
                network):
    ecolors = ["k"]*len(angles)
    ncolors = None

    if axes_colors is None or isinstance(axes_colors, str):
        named_cmap = "Set1" if axes_colors is None else axes_colors

        cmap = cm.get_cmap(named_cmap)

        values = list(range(num_axes))

        qualitative_cmaps = [
            "Pastel1", "Pastel2", "Accent", "Dark2", "Set1", "Set2",
            "Set3", "tab10"
        ]

        if named_cmap not in qualitative_cmaps:
            values = np.array(values) / (num_axes - 1)

        ncolors = cmap(values)
        ecolors = {}

        for i in range(num_axes):
            for j in range(num_axes):
                if i == j:
                    ecolors[(i, i)] = ncolors[i]
                else:
                    num_colors = 4 if network.is_directed() else 3

                    grad = linear_gradient(ncolors[i], ncolors[j], num_colors)

                    ecolors[(i, j)] = grad[1]
    else:
        if nonstring_container(axes_colors):
            assert len(axes_colors) == num_axes

    return ncolors, ecolors


def _set_names_lims(names, angles, max_radii, xs, ys, intra_connections,
                    show_names, axis, show_circles):
    # add names if necessary
    if show_names:
        prop = {
            "fontsize": 16*0.8,
            "ha": "center",
            "va": "center"
        }

        max_rmax = max(max_radii)

        for i, name in enumerate(names):
            angle = angles[i]
            rmax  = max_radii[i]

            if intra_connections:
                angle = 0.5*(angles[2*i] + angles[2*i + 1])
                rmax  = max_radii[2*i]

            rmax += 0.07 * (1 + name.count("\n")) * max_rmax

            x, y = rmax*np.cos(angle), rmax*np.sin(angle)

            # move to degrees
            angle *= 180 / np.pi

            if -30 <= angle <= 210:
                angle -= 90
            else:
                angle -= 270

            axis.text(x, y, name, rotation=angle, **prop)

    if not show_circles:
        for angle, rmax in zip(angles, max_radii):
            x, y = rmax*np.cos(angle), rmax*np.sin(angle)

            xs.append(x)
            ys.append(y)

        xmin = np.nanmin(xs)
        xmax = np.nanmax(xs)

        ymin = np.nanmin(ys)
        ymax = np.nanmax(ys)

        factor = 1.1

        axis.set_xlim(factor*xmin, factor*xmax)
        axis.set_ylim(factor*ymin, factor*ymax)


def _plot_nodes(nn, node_size, xx, yy, color, nborder_width, nborder_color,
                axis, zorder=3):
    if len(nn):
        ss = node_size[nn]

        axis.scatter(xx[nn], yy[nn], ss, color=color, linewidth=nborder_width,
                     edgecolors=nborder_color, zorder=zorder)


def _test_clockwise(i, j, num_axes):
    delta_max = int(0.5*num_axes)

    if num_axes == 2:
        return i != j

    for target in range(delta_max):
        for d in range(delta_max):
            if j == ((num_axes - 1 + target - d) % num_axes) and i == target:
                return True

    return False


def _plot_bezier(pstart, pstop, astart, astop, rstart, rstop, i, j, num_axes,
                 xs, ys):
    dtheta = np.abs(astart - astop)

    if dtheta > np.pi:
        dtheta = 2*np.pi - dtheta

    dtheta *= 0.3

    lstart = rstart*np.sin(dtheta)
    lstop  = rstop*np.sin(dtheta)

    dist = np.abs(i - j)

    if dist > 0.5*num_axes:
        dist -= int(0.5*num_axes)

    if _test_clockwise(i, j , num_axes):
        lstop *= -1
    elif _test_clockwise(j, i, num_axes) and dist == 1:
        lstart *= -1
    elif i > j:
        lstop *= -1
    else:
        lstart *= -1

    dp1 = np.array((lstart*np.cos(astart - 0.5*np.pi), lstart*np.sin(astart - 0.5*np.pi)))
    dp2 = np.array((lstop*np.cos(astop - 0.5*np.pi), lstop*np.sin(astop - 0.5*np.pi)))

    p1 = pstart + dp1
    p2 = pstop + dp2

    xs.append(p1[0])
    xs.append(p2[0])

    ys.append(p1[1])
    ys.append(p2[1])

    return Path([pstart, p1, p2, pstop],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])

#-*- coding:utf-8 -*-
#
# geospatial/plot.py
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

import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs

from ..plot import draw_network
from .countries import maps, convertors, country_points, country_codes


def draw_map(graph, node_names, geodata=None, geodata_names=None,
             points=None, show_points=False, linecolor=None, hue=None,
             proj=None, all_geodata=True, axis=None, show=False, **kwargs):
    '''
    Draw a network on a map.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph to plot.
    node_names : str
        Node attribute containing the nodes' names or A3 codes. This attribute
        will be used to place each node on the map. By default (if no `geodata`
        is provided), the world map is used and each node must therefore be
        associated to a country name or (better) an A3 ISO code.
    geodata : :class:`~geopandas.GeoDataFrame`, optional (default: world map)
        Optional dataframe containing the geospatial information.
        Predefined geodatas are "110m", "50m", and "10m" for world maps with
        respectively 110, 50, and 10 meter resolutions, or "adaptive" (default)
        for a world map with adaptive resolution depending on the country size.
    geodata_names : str, optional (default: "NAME_LONG" or "SU_A3")
        Column in `geodata` corresponding to the `node_names` (respectively
        for full country names or A3 codes).
    points : str, optional (default: capitals and representative points)
        Whether a precise point should be associated to each node.
        It can be either an entry in `geodata`, the "centroid" of each geometry
        entry, or a "representative" point.
        By default, if the world map is used, each country will be associated
        to its capital (contained in the module's ``cities`` object); if
        another `geodata` element is provided, it defaults to "representative".
    show_points : bool, optional (default: False)
        Wether the points should be displayed.
    linecolor : str, char, float or array, optional (default: current palette)
        Color of the map lines.
    esize : float, str, or array of floats, optional (default: 0.5)
        Width of the edges in percent of canvas length. Available string values
        are "betweenness" and "weight".
    ecolor : str, char, float or array, optional (default: "k")
        Edge color. If ecolor="groups", edges color will depend on the source
        and target groups, i.e. only edges from and toward same groups will
        have the same color.
    max_esize : float, optional (default: 5.)
        If a custom property is entered as `esize`, this normalizes the edge
        width between 0. and `max_esize`.
    threshold : float, optional (default: 0.5)
        Size under which edges are not plotted.
    proj : :mod:`cartopy.crs` object, optional (default: cartesian plane)
        Projection that will be used to draw the map.
    all_geodata : bool, optional (default: True)
        Whether all the data contained in `geodata` should be plotted, even if
        `graph` contains only a subset of it.
    axis : matplotlib axis, optional (default: a new axis)
        Axis that will be used to plot the graph.
    **kwargs : dict
        All possible arguments from :func:`~nngt.plot.draw_network`.
    '''
    names = graph.node_attributes[node_names]

    # check whether the names are full names or A3 codes
    is_a3_codes = True

    for n in names:
        if len(n) != 3:
            is_a3_codes = False
            break

    convert = None

    if geodata_names is None:
        if is_a3_codes:
            geodata_names = "SU_A3"
        else:
            geodata_names = "NAME_LONG"

    # set map
    world_map = True
    dataframe = None

    geodata = "adaptive" if geodata is None else geodata

    if isinstance(geodata, str):
        dataframe = maps[geodata]

        # update names
        if not is_a3_codes:
            names = [convertors.get(name, name) for name in names]
    else:
        world_map = False

    # projection
    if proj is None:
        proj = ccrs.PlateCarree()
    else:
        try:
            crs_proj4 = proj.proj4_init
            dataframe = dataframe.to_crs(crs_proj4)
        except:
            # PlateCarree
            pass

    if axis is None:
        fig = plt.figure()
        axis = plt.axes(projection=proj)

    # underlying map (optional)
    lw = kwargs.get("linewidth", 1)

    if all_geodata:
        dataframe.boundary.plot(ax=axis, color=linecolor, alpha=0.2, zorder=0,
                                linewidth=lw)

    # get existing elements
    mapping = None

    if is_a3_codes and isinstance(geodata, str):
        cc = country_codes[geodata]
        mapping = {n: cc[n] for n in names}
    else:
        mapping = {s[geodata_names]: i for i, s in dataframe.iterrows()}

    elements = [mapping[n] for n in names]

    if hue is None:
        dataframe.iloc[elements].boundary.plot(ax=axis, color=linecolor,
                                               zorder=1, linewidth=lw)
    else:
        if hue not in dataframe:
            dataframe[hue] = np.full(len(dataframe), np.NaN)

            dataframe.loc[elements, hue] = graph.node_attributes[hue]

        dataframe.iloc[elements].plot(column=hue, ax=axis, zorder=1,
                                      linewidth=lw)

    # get positions
    pos = []

    if points in dataframe:
        pos = [(p.xy[0][0], p.xy[1][0])
               for p in dataframe.iloc[elements, points]]
    elif points == "centroid":
        df = dataframe.loc[elements, "geometry"].centroid
        pos = [(p.xy[0][0], p.xy[1][0]) for p in df]
    elif points == "representative":
        df = dataframe.loc[elements, "geometry"].representative_point()
        pos = [(p.xy[0][0], p.xy[1][0]) for p in df]
    elif points is None and geodata in (None, "adaptive"):
        try:
            crs_proj4 = proj.proj4_init
            cpoints = country_points.to_crs(crs_proj4)
        except:
            # PlateCarree
            cpoints = country_points
        df = cpoints.loc[elements, "geometry"]
        pos = [(p.xy[0][0], p.xy[1][0]) for p in df]
    elif points is None:
        df  = dataframe.loc[elements, "geometry"].representative_point()
        pos = [(p.xy[0][0], p.xy[1][0]) for p in df]
    else:
        raise ValueError("Invalid value for `points`: {}".format(points))

    pos = np.array(pos)

    if "restrict_nodes" in kwargs:
        pos = pos[kwargs["restrict_nodes"]]

    rm_kw = [
        "show_environment", "positions", "axis", "tight", "fast", "spatial"
    ]

    for k in rm_kw:
        if k in kwargs:
            del kwargs[k]

    # make plot
    draw_network(graph, layout=pos, axis=axis, show_environment=False,
                 fast=True, tight=False, proj=proj, spatial=False, show=False,
                 **kwargs)

    # restore full map
    if all_geodata:
        axis.set_global()

    if show:
        plt.show()

    return axis

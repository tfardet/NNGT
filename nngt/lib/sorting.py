#-*- coding:utf-8 -*-
#
# lib/sorting.py
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

""" Sorting tools """

import numpy as np

from .errors import InvalidArgument
from .test_functions import nonstring_container, is_integer


def find_idx_nearest(array, values):
    '''
    Find the indices of the nearest elements of `values` in a sorted `array`.

    .. warning::
        Both ``array`` and ``values`` should be `numpy.array` objects and
        `array` MUST be sorted in increasing order.

    Parameters
    ----------
    array : reference list or np.ndarray
    values : double, list or array of values to find in `array`

    Returns
    -------
    idx : int or array representing the index of the closest value in `array`
    '''
    idx = np.searchsorted(array, values, side="left") # get the interval
    # return the index of the closest
    if isinstance(values, np.float) or is_integer(values):
        if idx == len(array):
            return idx-1
        else:
            idx -= (np.abs(values-array[idx-1]) < np.abs(values-array[idx]))
            return idx
    else:
        # find where it is idx_max+1
        overflow = (idx == len(array))
        idx[overflow] -= 1
        # for the others, find the nearest
        tmp = idx[~overflow]
        idx[~overflow] = tmp - (np.abs(values[~overflow] - array[tmp-1])
                                < np.abs(values[~overflow] - array[tmp]))
        return idx


def _sort_neurons(sort, gids, network, data=None, return_attr=False):
    '''
    Sort the neurons according to the `sort` property.

    If `sort` is "firing_rate" or "B2", then data must contain the `senders`
    and `times` list given by a NEST ``spike_recorder``.

    Parameters
    ----------
    sort : str or array
        Sorting method or indices
    gids : array-like
        NEST gids
    network : the network
    data : numpy.array of shape (N, 2)
        Senders on column 1, times on column 2.

    Returns
    -------
    For N neurons, labeled from ``GID_MIN`` to ``GID_MAX``, returns a`sorting`
    array of size ``GID_MAX``, where ``sorting[gids]`` gives the sorted ids of
    the neurons, i.e. an integer between 1 and N.
    '''
    from nngt.analysis import node_attributes, get_b2
    from nngt.simulation.nest_utils import nest_version

    min_nest_gid = network.nest_gids.min()
    max_nest_gid = network.nest_gids.max()

    sorting = np.zeros(max_nest_gid + 1)

    attribute = None
    sorted_ids = None

    if isinstance(sort, str):
        if sort == "firing_rate":
            # compute number of spikes per neuron
            spikes = np.bincount(data[:, 0].astype(int))

            # shift from 1 in NEST 3+
            max_entry = max_nest_gid + 1 if nest_version == 3 else max_nest_gid

            if spikes.shape[0] < max_entry: # one entry per neuron
                spikes.resize(max_entry)

            # sort them (neuron with least spikes arrives at min_nest_gid)
            sorted_ids = np.argsort(spikes)[min_nest_gid:] - min_nest_gid

            # get attribute
            if len(data[:, 0]):
                idx_min = int(np.min(data[:, 0]))
                attribute = spikes[idx_min:] \
                            / (np.max(data[:, 1]) - np.min(data[:, 1]))
            else:
                attribute = []
        elif sort.lower() == "b2":
            attribute = get_b2(network, data=data, nodes=gids)
            sorted_ids = np.argsort(attribute)
            # check for non-spiking neurons
            num_b2 = attribute.shape[0]
            if num_b2 < network.node_nb():
                spikes = np.bincount(data[:, 0])
                non_spiking = np.where(spikes[min_nest_gid:] == 0)[0]
                sorted_ids.resize(network.node_nb())
                for i, n in enumerate(non_spiking):
                    sorted_ids[sorted_ids >= n] += 1
                    sorted_ids[num_b2 + i] = n
        elif sort == "space":
            xs, ys = network.get_positions().T
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)

            num_boxes = 10

            xbins  = np.linspace(x_min, x_max, num_boxes + 1)
            ybins  = np.linspace(y_min, y_max, num_boxes + 1)
            xboxes = np.digitize(xs, xbins)
            yboxes = np.digitize(ys, ybins)

            attribute  = xboxes*num_boxes + yboxes
            sorted_ids = np.argsort(attribute)
        elif sort == "x":
            xs, _ = network.get_positions().T
            x_min, x_max = np.min(xs), np.max(xs)

            num_boxes = 10

            xbins  = np.linspace(x_min, x_max, num_boxes + 1)
            attribute = np.digitize(xs, xbins)
            sorted_ids = np.argsort(attribute)
        elif sort == "y":
            _, ys = network.get_positions().T
            y_min, y_max = np.min(ys), np.max(ys)

            num_boxes = 10

            ybins  = np.linspace(y_min, y_max, num_boxes + 1)
            attribute = np.digitize(ys, ybins)
            sorted_ids = np.argsort(attribute)
        else:
            attribute = node_attributes(network, sort)
            sorted_ids = np.argsort(attribute)
    else:
        sorted_ids = np.argsort(sort)
        attribute  = sort[sorted_ids]

    if network.is_network():
        num_sorted = 1
        for group in network.population.values():
            gids = network.nest_gids[group.ids]
            order = np.argsort(np.argsort(np.argsort(sorted_ids)[group.ids]))
            sorting[gids] = num_sorted + order
            num_sorted += len(group.ids)

    if return_attr:
        return sorting.astype(int), attribute

    return sorting.astype(int)


def _sort_groups(pop):
    '''
    Sort the groups of a NeuralPop by decreasing size.
    '''
    names, groups = [], []
    for name, group in pop.items():
        names.append(name)
        groups.append(group)
    sizes = [len(g.ids) for g in groups]
    order = np.argsort(sizes)[::-1]
    return [names[i] for i in order], [groups[i] for i in order]

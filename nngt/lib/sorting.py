#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Sorting tools """

import numpy as np

from .errors import InvalidArgument


def _sort_neurons(sort, gids, network, data=None):
    '''
    Sort the neurons according to the `sort` property.

    If `sort` is "firing_rate", then data contains the `senders` list given by
    a NEST ``spike_recorder``.
    If `sort` is "B2", then data contains both the senders and the spike time
    associated.

    Returns
    -------
    For N neurons, labeled from ``GID_MIN`` to ``GID_MAX``, returns a`sorting`
    array of size ``GID_MAX``, where ``sorting[gids]`` gives the sorted ids of
    the neurons, i.e. an integer between 1 and N.
    '''
    min_nest_gid = network.nest_gid.min()
    max_nest_gid = network.nest_gid.max()
    sorting = np.zeros(max_nest_gid + 1)
    if isinstance(sort, str):
        sorted_ids = None
        if "degree" in sort:
            deg_type = sort[:sort.find("-")]
            degrees = network.get_degrees(deg_type)
            sorted_ids = np.argsort(degrees)
        elif sort == "betweenness":
            betw = network.get_betweenness(btype="node")
            sorted_ids = np.argsort(betw)
        elif sort == "firing_rate" and data is not None:
            # compute number of spikes per neuron
            spikes = np.bincount(data)
            # sort them (neuron with least spikes arrives at min_nest_gid)
            sorted_ids = np.argsort(spikes)[min_nest_gid:] - min_nest_gid
        elif sort.lower() == "b2":
            b2 = _b2(data)
            sorted_ids = np.argsort(spikes)[min_nest_gid:] - min_nest_gid
        else:
            raise InvalidArgument(
                'Unknown sorting parameter {}; choose among "in-degree" ' + \
                'out-degree", "total-degree" or "betweenness".'.format(sort))
        num_sorted = 1
        _, sorted_groups = _sort_groups(network.population)
        for group in sorted_groups:
            gids = network.nest_gid[group.id_list]
            order = np.argsort(sorted_ids[group.id_list])
            sorting[gids] = num_sorted + order
            num_sorted += len(group.id_list)
    else:
        sorting[network.nest_gid[sort]] = sort
    return sorting


def _sort_groups(pop):
    '''
    Sort the groups of a NeuralPop by decreasing size.
    '''
    names, groups = [], []
    for name, group in iter(pop.items()):
        names.append(name)
        groups.append(group)
    sizes = [len(g.id_list) for g in groups]
    order = np.argsort(sizes)[::-1]
    return [names[i] for i in order], [groups[i] for i in order]


def _b2(data):
    ''' Compute the b2 coefficient for the neurons. '''
    senders = data[0,:]
    times = data[1,:]
    gid_start, gid_stop = senders.min(), senders.max()
    b2 = np.zeros(gid_stop + 1 - gid_start)
    for i in range(gid_start, gid_stop+1):
        ids = np.where(senders == i)[0]
        dt1 = np.diff(times[ids])
        dt2 = np.diff(dt1)
        b2[i - gid_start] = (2*np.var(dt1) - np.var(dt2)) / (2*np.average(dt1))
    return b2

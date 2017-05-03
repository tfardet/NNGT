#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Sorting tools """

from nngt.analysis import node_attributes
import numpy as np

from .errors import InvalidArgument


def _sort_neurons(sort, gids, network, data=None):
    '''
    Sort the neurons according to the `sort` property.

    If `sort` is "firing_rate" or "B2", then data must contain the `senders`
    and `times` list given by a NEST ``spike_recorder``.

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
        if sort == "firing_rate":
            # compute number of spikes per neuron
            spikes = np.bincount(data[0])
            if spikes.shape[0] < max_nest_gid: # one entry per neuron
                spikes.resize(max_nest_gid)
            # sort them (neuron with least spikes arrives at min_nest_gid)
            sorted_ids = np.argsort(spikes)[min_nest_gid:] - min_nest_gid
        elif sort.lower() == "b2":
            b2 = _b2(data)
            sorted_ids = np.argsort(b2)
            # check for non-spiking neurons
            num_b2 = b2.shape[0]
            if num_b2 < network.node_nb():
                spikes = np.bincount(data[0])
                non_spiking = np.where(spikes[min_nest_gid] == 0)[0]
                sorted_ids.resize(network.node_nb())
                for i, n in enumerate(non_spiking):
                    sorted_ids[sorted_ids >= n] += 1
                    sorted_ids[num_b2 + i] = n
        else:
            attr = node_attributes(network, sort)
            sorted_ids = np.argsort(attr)
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
    senders = data[0]
    times = np.array(data[1])
    gid_start = np.min(senders)
    num_active = np.max(senders) - gid_start + 1
    b2 = np.zeros(num_active)
    for i in range(num_active):
        ids = np.where(senders == gid_start + i)[0]
        dt1 = np.diff(times[ids])
        dt2 = dt1[1:] + dt1[:-1]
        avg_isi = np.mean(dt1)
        if avg_isi != 0.:
            b2[i] = (2*np.var(dt1) - np.var(dt2)) / (2*avg_isi**2)
        else:
            b2[i] = np.inf
    return b2

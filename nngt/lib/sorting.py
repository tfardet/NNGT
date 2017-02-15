#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Sorting tools """

import numpy as np


def _sort_neurons(sort, gids, network):
    max_nest_gid = network.nest_gid.max() + 1
    sorting = np.zeros(max_nest_gid)
    if isinstance(sort, str):
        sorted_ids = None
        if "degree" in sort:
            deg_type = sort[:sort.find("-")]
            degrees = network.get_degrees(deg_type)
            sorted_ids = np.argsort(degrees)
        elif sort == "betweenness":
            betw = network.get_betweenness(btype="node")
            sorted_ids = np.argsort(betw)
        else:
            raise InvalidArgument(
                '''Unknown sorting parameter {}; choose among "in-degree",
                "out-degree", "total-degree" or "betweenness".'''.format(sort))
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

#!/usr/bin/env python
#-*- coding:utf-8 -*-

# nest_utils.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Tools for the nngt.simulation module """

import numpy as np


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


def _generate_random(number, instructions):
    name = instructions[0]
    dbl1, dbl2 = instructions[1:]
    if name == "uniform":
        return np.random.uniform(dbl1, dbl2, number)
    elif name == "normal" or name == "gaussian":
        return np.random.normal(dbl1, dbl2, number)
    else:
        raise NotImplementedError('''Only "uniform" and "normal"/"gaussian"
                                  distributions are implemented so far.''')

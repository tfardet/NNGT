#!/usr/bin/env python
#-*- coding:utf-8 -*-

# nest_utils.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Utility functions to monitor NEST simulated activity """

import nest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from nngt.lib import InvalidArgument


__all__ = [
    'set_noise',
    'set_poisson_input',
    'set_step_currents',
    'monitor_groups',
    'monitor_nodes',
]


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


#-----------------------------------------------------------------------------#
# Inducing activity
#------------------------
#

def set_noise(gids, mean, std):
    '''
    Submit neurons to a current white noise.
    @todo: check how NEST handles the :math:`\\sqrt{t}` in the standard dev.
    
    Parameters
    ----------
    gids : tuple
        NEST gids of the target neurons.
    mean : float
        Mean current value.
    std : float
        Standard deviation of the current
    
    Returns
    -------
    noise : tuple
        The NEST gid of the noise_generator.
    '''
    noise = nest.Create("noise_generator")
    nest.SetStatus(noise, {"mean": mean, "std": std })
    nest.Connect(noise,gids)
    return noise
    
def set_poisson_input(gids, rate):
    '''
    Submit neurons to a Poissonian rate of spikes.
    
    Parameters
    ----------
    gids : tuple
        NEST gids of the target neurons.
    rate : float
        Rate of the spike train.
    
    Returns
    -------
    poisson_input : tuple
        The NEST gid of the poisson_generator.
    '''
    poisson_input = nest.Create("poisson_generator")
    nest.SetStatus(poisson_input,{"rate": rate})
    nest.Connect(poisson_input, gids)
    return poisson_input

def set_step_currents(gids, times, currents):
    '''
    Set step-current excitations
    
    Parameters
    ----------
    gids : tuple
        NEST gids of the target neurons.
    times : list or :class:`numpy.ndarray`
        List of the times where the current will change (by default the current
        generator is initiated at I=0. for t=0.)
    currents : list or :class:`numpy.ndarray`
        List of the new current value after the associated time value in 
        `times`.
    
    Returns
    -------
    noise : tuple
        The NEST gid of the noise_generator.
    '''
    if len(times) != len(currents):
        raise InvalidArgument('Length of `times` and `currents` must be the \
same')
    params = { "amplitude_times": times, "amplitude_values":currents }
    scg = nest.Create("step_current_generator", 1, params)
    nest.Connect(scg, gids)
    return scg


#-----------------------------------------------------------------------------#
# Monitoring the activity
#------------------------
#

def _monitor(gids, nest_recorder, params):
    new_record = []
    recorders = []
    for i,rec in enumerate(nest_recorder):
        # multi/volt/conductancemeter
        if "meter" in rec:
            device = None
            di_spec = {"rule": "all_to_all"}
            if not params[i].get("to_accumulator", False):
                device = nest.Create(rec, len(gids))
                di_spec["rule"] = "one_to_one"
            else:
                device = nest.Create(rec)
            recorders.append(device)
            new_record.append(params[i]["record_from"])
            nest.SetStatus(device, params[i])
            nest.Connect(device, gids, conn_spec=di_spec)
        # event detectors
        elif "detector" in rec:
            device = nest.Create(rec)
            recorders.append(device)
            new_record.append("spikes")
            nest.SetStatus(device,params[i])
            nest.Connect(gids, device)
        else:
            raise InvalidArgument('''Invalid recorder item in `nest_recorder`:
                                  {} is unknown.'''.format(nest_recorder))
    return tuple(recorders), new_record


def monitor_groups(group_names, network, nest_recorder=["spike_detector"],
                   params=[{}]):
    '''
    Monitoring the activity of nodes in the network.

    Parameters
    ----------
    group_name : list of strings
        Names of the groups that should be recorded.
    network : :class:`~nngt.Network` or subclass
        Network which population will be used to differentiate groups.
    nest_recorder : list of strings, optional (default: ["spike_detector"])
        List of devices to monitor the network.
    params : list of dict, optional (default: [{}])
        List of dictionaries containing the parameters for each recorder (see 
        `NEST documentation <http://www.nest-simulator.org/quickref/#nodes>`_ 
        for details).

    Returns
    -------
    recorders : tuple
        Tuple of the recorders' gids
    recordables : tuple
        Typle of the recordables' names.
    '''
    recorders, recordables = [], []
    # sort and monitor
    nodes_gids = []
    sorted_names, sorted_groups =_sort_groups(network.population)
    sort_input = np.argsort([sorted_names.index(name) for name in group_names])
    sorted_input = [group_names[i] for i in sort_input]
    for name in sorted_input:
        gids = tuple(network.population[name].nest_gids)
        recdr, recdbls = _monitor(gids, nest_recorder, params)
        recorders.extend(recdr)
        recordables.extend(recdbls)
    return recorders, recordables


def monitor_nodes(gids, nest_recorder=["spike_detector"], params=[{}],
                  network=None):
    '''
    Monitoring the activity of nodes in the network.

    Parameters
    ----------
    gids : tuple of ints or list of tuples
        GIDs of the neurons in the NEST subnetwork; either one list per
        recorder if they should monitor different neurons or a unique list
        which will be monitored by all devices.
    nest_recorder : list of strings, optional (default: ["spike_detector"])
        List of devices to monitor the network.
    params : list of dict, optional (default: [{}])
        List of dictionaries containing the parameters for each recorder (see
        `NEST documentation <http://www.nest-simulator.org/quickref/#nodes>`_
        for details).
    network : :class:`~nngt.Network` or subclass, optional (default: None)
        Network which population will be used to differentiate groups.

    Returns
    -------
    recorders : tuple
        Tuple of the recorders' gids
    recordables : tuple
        Typle of the recordables' names.
    '''
    return _monitor(gids, nest_recorder, params)

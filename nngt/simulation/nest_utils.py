#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
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

# nest_utils.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Utility functions to monitor NEST simulated activity """

import nest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from nngt.lib import InvalidArgument, nonstring_container
from nngt.lib.rng_tools import _generate_random
from nngt.lib.sorting import _sort_groups


__all__ = [
    'monitor_groups',
    'monitor_nodes',
    'randomize_neural_states',
    'set_minis',
    'set_noise',
    'set_poisson_input',
    'set_step_currents',
]


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


def set_minis(network, base_rate, weight_fraction=0.05, gids=None):
    '''
    Mimick spontaneous release of neurotransmitters, called spontaneous PSCs or
    "minis".
    These minis consists in only a fraction of the usual strength of a spike-
    triggered PSC and can be modeled by a Poisson process.
    This Poisson process occurs independently at every synapse of a neuron, so
    a neuron receiving :math:`k` inputs will be subjected to these events with
    a rate :math:`k*\\lambda`, where :math:`\\lambda` is the base rate.

    Parameters
    ----------
    network : :class:`~nngt.Network` object
        Network on which the minis should be simulated.
    base_rate : float
        Rate for the Poisson process on one synapse (:math:`\\lambda`).
    weight_fraction : float, optional (default: 0.05)
        Fraction of a spike-triggered PSC that will be released by a mini.
    gids : array-like container of NEST gids, optional (default = all neurons)
        Neurons that should be subjected to minis.
    '''
    assert (weight_fraction >= 0. and weight_fraction <= 1.), \
           "`weight_fraction` must be between 0 and 1."
    assert network.nest_gid is not None, "Create the NEST network first."
    degrees = network.get_degrees("in")
    # mean returns a np.matrix, getA1 converts to 1D array
    weights = np.transpose(network.adjacency_matrix().mean(1)).getA1()
    deg_set = set(degrees)
    map_deg_pg = {d: i for i, d in enumerate(deg_set)}
    pgs = nest.Create("poisson_generator", len(deg_set))
    for d, pg in zip(deg_set, pgs):
        nest.SetStatus([pg], {"rate": d*base_rate})
    if gids is None:
        gids = network.nest_gid
    else:
        raise NotImplementedError(
            "Choosing only specific nodes is not yet implemented.")
    for i in range(len(gids)):
        gid, d, w = [gids[i]], degrees[i], weights[i]*weight_fraction
        pg = [pgs[map_deg_pg[d]]]
        nest.Connect(pg, gid, syn_spec={"weight": w})


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
        raise InvalidArgument('Length of `times` and `currents` must be the '
                              'same')
    params = { "amplitude_times": times, "amplitude_values":currents }
    scg = nest.Create("step_current_generator", 1, params)
    nest.Connect(scg, gids)
    return scg


def randomize_neural_states(network, instructions, groups=None,
                            make_nest=False):
    '''
    Randomize the neural states according to the instructions.

    Parameters
    ----------
    network : :class:`~nngt.Network` subclass instance
        Network that will be simulated.
    instructions : dict
        Variables to initialize. Allowed keys are "V_m" and "w". Values are
        3-tuples of type ``("distrib_name", double, double)``.
    groups : list of :class:`~nngt.NeuralGroup`, optional (default: None)
        If provided, only the neurons belonging to these groups will have their
        properties randomized.
    make_nest : bool, optional (default: False)
        If ``True`` and network has not been converted to NEST, automatically
        generate the network, else raises an exception.

    Example
    -------
    python::
        instructions = {
            "V_m": ("uniform", -80., -60.),
            "w": ("normal", 50., 5.)
        }
    '''
    # check whether network is in NEST
    if network._nest_gid is None:
        if make_nest:
            network.to_nest()
        else:
            raise AttributeError(
                '`network` has not been converted to NEST yet.')
    num_neurons = 0
    gids = list(network._nest_gid)
    if groups is not None:
        for group in groups:
            gids.extend(group.nest_gids)
        gids = list(set(gids))
        num_neurons = len(gids)
    else:
        num_neurons = network.node_nb()
    for key, val in instructions.items():
        state = _generate_random(num_neurons, val)
        nest.SetStatus(gids, key, state)


#-----------------------------------------------------------------------------#
# Monitoring the activity
#------------------------
#

def monitor_groups(group_names, network, nest_recorder=None, params=None):
    '''
    Monitoring the activity of nodes in the network.

    Parameters
    ----------
    group_name : list of strings
        Names of the groups that should be recorded.
    network : :class:`~nngt.Network` or subclass
        Network which population will be used to differentiate groups.
    nest_recorder : strings or list, optional (default: "spike_detector"0)
        Device(s) to monitor the network.
    params : dict or list of, optional (default: `{}`)
        Dictionarie(s) containing the parameters for each recorder (see
        `NEST documentation <http://www.nest-simulator.org/quickref/#nodes>`_
        for details).

    Returns
    -------
    recorders : tuple
        Tuple of the recorders' gids
    recordables : tuple
        Tuple of the recordables' names.
    '''
    if nest_recorder is None:
        nest_recorder = ["spike_detector"]
    elif not nonstring_container(nest_recorder):
        nest_recorder = [nest_recorder]
    if params is None:
        params = [{}]
    elif isinstance(params, dict):
        params = [param]
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


def monitor_nodes(gids, nest_recorder=None, params=None, network=None):
    '''
    Monitoring the activity of nodes in the network.

    Parameters
    ----------
    gids : tuple of ints or list of tuples
        GIDs of the neurons in the NEST subnetwork; either one list per
        recorder if they should monitor different neurons or a unique list
        which will be monitored by all devices.
    nest_recorder : strings or list, optional (default: "spike_detector")
        Device(s) to monitor the network.
    params : dict or list of, optional (default: `{}`)
        Dictionarie(s) containing the parameters for each recorder (see
        `NEST documentation <http://www.nest-simulator.org/quickref/#nodes>`_
        for details).
    network : :class:`~nngt.Network` or subclass, optional (default: None)
        Network which population will be used to differentiate groups.

    Returns
    -------
    recorders : tuple
        Tuple of the recorders' gids
    recordables : tuple
        Tuple of the recordables' names.
    '''
    if nest_recorder is None:
        nest_recorder = ["spike_detector"]
    elif not nonstring_container(nest_recorder):
        nest_recorder = [nest_recorder]
    if params is None:
        params = [{}]
    elif isinstance(params, dict):
        params = [param]
    return _monitor(gids, nest_recorder, params)


def _monitor(gids, nest_recorder, params):
    new_record = []
    recorders = []
    for i, rec in enumerate(nest_recorder):
        # multi/volt/conductancemeter
        if "meter" in rec:
            device = None
            di_spec = {"rule": "all_to_all"}
            if not params[i].get("to_accumulator", True):
                device = nest.Create(rec, len(gids))
                di_spec["rule"] = "one_to_one"
            else:
                device = nest.Create(rec)
            recorders.append(device)
            device_params = nest.GetDefaults(rec)
            device_params.update(params[i])
            new_record.append(device_params["record_from"])
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
            raise InvalidArgument('Invalid recorder item in `nest_recorder`: '
                                  '{} is unknown.'.format(nest_recorder))
    return tuple(recorders), new_record

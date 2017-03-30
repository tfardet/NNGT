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

from nngt.lib import InvalidArgument, nonstring_container
from nngt.lib.rng_tools import _generate_random
from nngt.lib.sorting import _sort_groups


__all__ = [
    'set_noise',
    'set_poisson_input',
    'set_step_currents',
    'monitor_groups',
    'monitor_nodes',
    'randomize_neural_states',
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
    gids = []
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
    nest_recorder : strings or list, optional (default: None)
        Device(s) to monitor the network. Defaults to "spike_detector".
    params : dict or list of, optional (default: None)
        Dictionarie(s) containing the parameters for each recorder (see
        `NEST documentation <http://www.nest-simulator.org/quickref/#nodes>`_
        for details). Defaults to ``{}``.

    Returns
    -------
    recorders : tuple
        Tuple of the recorders' gids
    recordables : tuple
        Typle of the recordables' names.
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
    nest_recorder : strings or list, optional (default: None)
        Device(s) to monitor the network. Defaults to "spike_detector".
    params : dict or list of, optional (default: None)
        Dictionarie(s) containing the parameters for each recorder (see
        `NEST documentation <http://www.nest-simulator.org/quickref/#nodes>`_
        for details). Defaults to ``{}``.
    network : :class:`~nngt.Network` or subclass, optional (default: None)
        Network which population will be used to differentiate groups.

    Returns
    -------
    recorders : tuple
        Tuple of the recorders' gids
    recordables : tuple
        Typle of the recordables' names.
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
            raise InvalidArgument('''Invalid recorder item in `nest_recorder`:
                                  {} is unknown.'''.format(nest_recorder))
    return tuple(recorders), new_record

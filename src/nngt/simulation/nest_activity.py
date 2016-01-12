#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Analyze the activity of a network """

import nest
import numpy as np

import matplotlib.pyplot as plt

from nngt.lib import InvalidArgument

__all__ = [ 'get_activity_types' ]


#-----------------------------------------------------------------------------#
# Finding the various activities
#------------------------
#

def get_activity_types(network, spike_detector, simtime, phase_coeff=(0.2,5.),
                       min_fraction=0.1, return_steps=False):
    '''
    Analyze the spiking pattern of a neural network.

    Parameters
    ----------
    network : :class:`~nngt.Network`
        Neural network that was analyzed
    spike_detector : NEST node(s), (tuple or list of tuples)
        The recording device that monitored the network's spikes
    simtime : float
        Duration of the simulation in ms.
    phase_coeff : tuple of floats, optional (default: (0.2, 5.))
        A phase is considered `bursting' when the interspike between all spikes
        that compose it is smaller than ``phase_coeff[0]*avg_rate`` (where
        ``avg_rate`` is the average firing rate), `quiescent' when it is greater
        that ``phase_coeff[1]*avg_rate``, `mixed' otherwise.
    min_fraction : float, optional (default: 0.1)
        Minimal fraction of the neurons that should participate for a burst to
        be validated (i.e. if the interspike is smaller that the limit BUT the
        number of participating neurons is too small, the phase will be
        considered as mixed).
    return_steps : bool, optional (default: False)
        If ``True``, a second dictionary, `phases_steps` will also be returned.
    
    Returns
    -------
    phases : dict
        Dictionary containing the time intervals (in ms) for all three phase types as
        lists.
        E.g: ``phases["bursting"]`` could give ``[[123.5,334.2],[857.1,1000.6]]``.
    phases_steps : dict, optional
        Dictionary containing the timesteps in NEST.
    '''
    # check if there are several recorders
    times,senders = [], []
    if len(spike_detector) > 1:
        for sd in spike_detector:
            data = nest.GetStatus(sd)[0]["events"]
            times.extend(data["times"])
            senders.extend(data["senders"])
        idx_sorted = np.argsort(times)
        times = np.array(times)[idx_sorted]
        senders = np.array(senders)[idx_sorted]
    else:
        data = nest.GetStatus(spike_detector)[0]["events"]
        times = data["times"]
        senders = data["senders"]
    plt.plot(times,senders)
    # get the average firing rate to differenciate the phases
    avg_rate = len(times)/float(simtime)
    lim_burst = phase_coeff[0]/avg_rate
    lim_quiet = phase_coeff[1]/avg_rate
    # find the phases
    phases = { "bursting":[[]], "mixed":[[]], "quiescent":[[]] }
    phases_steps = { "bursting":[[]], "mixed":[[]], "quiescent":[[]] }
    diff = np.diff(times).tolist()[::-1]
    i = 0
    while diff:
        tau = diff.pop()
        while True:
            phase = phases["bursting"][-1]
            phase_step = phases_steps["bursting"][-1]
            if tau < lim_burst: # bursting phase
                if phase:
                    phase[1] = times[i]
                    phase_step[1] = i
                else:
                    phases["bursting"][-1] = [times[i],times[i+1]]
                    phases_steps["bursting"][-1] = [i,i+1]
                i+=1
                break
            elif phase:
                # make sure a new empty list is prepared for the next burst
                phases["bursting"].append([])
                phases_steps["bursting"].append([])
            phase = phases["quiescent"][-1]
            phase_step = phases_steps["quiescent"][-1]
            if tau > lim_quiet:
                if phase:
                    phase[1] = times[i]
                    phase_step[1] = i
                else:
                    phases["quiescent"][-1] = [times[i],times[i+1]]
                    phases_steps["quiescent"][-1] = [i,i+1]
                i+=1
                break
            elif phase:
                phases["quiescent"].append([])
                phases_steps["quiescent"].append([])
            phase = phases["mixed"][-1]
            phase_step = phases_steps["mixed"][-1]
            if tau <= lim_quiet or tau >= lim_burst :
                if phase:
                    phase[1] = times[i]
                    phase_step[1] = i
                else:
                    phases["mixed"][-1] = [times[i],times[i+1]]
                    phases_steps["mixed"][-1] = [i,i+1]
                i+=1
                break
            elif phase:
                phases["mixed"].append([])
                phases_steps["mixed"].append([])
    # get rid of trailing empty lists
    for key, val in phases.iteritems():
        if not val[-1]:
            val.pop()
            phases_steps[key].pop()
    # check that bursting periods involve at least min_fraction of the neurons
    lst_transfer = []
    n = network.node_nb()
    for i,burst in enumerate(phases_steps["bursting"]):
        participating_frac = len(set(senders[burst[0]:burst[1]]))/float(n)
        if participating_frac < min_fraction:
            lst_transfer.append(i)
    for i in lst_transfer[::-1]:
        phase = phases["bursting"].pop(i)
        phase_steps = phases_steps["bursting"].pop(i)
        phases["mixed"].append(phase)
        phases_steps["mixed"].append(phase_steps)
    if return_steps:
        return phases, phases_steps
    else:
        return phases

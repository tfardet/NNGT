#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Analyze the activity of a network """

import weakref
from collections import namedtuple
from copy import deepcopy

import nest
import numpy as np

from nngt.lib import InvalidArgument



__all__ = [ "ActivityRecord", 'activity_types' ]


#-----------------------------------------------------------------------------#
# Finding the various activities
#------------------------
#

class ActivityRecord:
    '''
    Class to record the properties of the simulated activity.
    '''
    
    def __init__(self, network, limits=None, spike_detector=None, phases=None,
                 multimeter=None, **kwargs):
        '''
        Initialize the instance (store proxy to the `network` and optional
        `spike_detector`) and compute the properties if data was provided.
        '''
        self.network = weakref.proxy(network)
        self.limits = deepcopy(limits)
        self.data_in_files = False
        self.phases = {
            "bursting":[],
            "mixed":[],
            "quiescent":[],
            "localized": [],
        }
        self.data = {
            "spike_files": [],
            "spike_times": [],
            "spike_senders": [],
        }
        self._prop = None
        if spike_detector is not None:
            self.sd = spike_detector
            if True in nest.GetStatus(self.sd, "to_file"):
                self.data_in_files = True
                for fpath in nest.GetStatus(self.sd, "record_to"):
                    self.data["spike_files"].extend(fpath)
            else:
                for events in nest.GetStatus(self.sd, "events"):
                    self.data["spike_times"].extend(events["times"])
                    self.data["spike_senders"].extend(events["senders"])
                self.data["spike_times"] = np.array(self.data["spike_times"])
                self.data["spike_senders"] = np.array(self.data["spike_senders"])
            if phases is None:
                self.phases = self.get_phases(**kwargs)
        if phases is not None:
            self.phases = phases.copy()
        self.compute_properties()
    
    @property
    def properties(self):
        return self._prop
    
    def get_phases(self, **kwargs):
        self.phases = activity_types(self.network, self.sd, self.limits,
                                     **kwargs)
    
    def compute_properties(self):
        if self.data_in_files:
            raise NotImplementedError()
        else:
            prop = { "phases": self.phases }
            times = self.data["spike_times"]
            prop["rate"] = len(self.data["spike_times"])/np.diff(self.limits)
            num_bursts = len(self.phases["bursting"])
            if num_bursts:
                prop["bursting"] = True
                prop.update({"T_burst": 0., "IBI": 0. })
            else:
                prop["bursting"] = False
            if times.any():
                prop["avg_spb"] = 0.
            for i, burst in enumerate(self.phases["bursting"]):
                # T_burst
                prop["T_burst"] += burst[1]-burst[0]
                # IBI
                if i > 0:
                    end_older_burst = self.phases["bursting"][i-1][1]
                    prop["IBI"] += burst[0]-end_older_burst
                # get num_spikes inside the burst, divide by num_neurons
                if times.any():
                    idxs = np.where((times >= burst[0])*(times <= burst[1]))[0]
                    spikes = set(times[idxs])
                    neurons = set(self.data["spike_senders"][idxs])
                    prop["avg_spb"] += len(spikes)/float(len(neurons))
            for key in iter(prop.keys()):
                if key != "phases":
                    prop[key] /= float(num_bursts)
            # generate properties as a namedtuple
            Properties = namedtuple('Properties', prop.keys())
            self._prop = Properties(**prop)


#-----------------------------------------------------------------------------#
# Finding the various activities
#------------------------
#

def activity_types(network, spike_detector, limits, phase_coeff=(0.5,10.),
                   mbis=0.5, mfb=0.2, mflb=0.05, simplify=False, fignums=[],
                   show=False):
    '''
    Analyze the spiking pattern of a neural network.
    .. todo ::
        think about inserting t=0. and t=simtime at the beginning and at the 
        end of ``times''.

    Parameters
    ----------
    network : :class:`~nngt.Network`
        Neural network that was analyzed
    spike_detector : NEST node(s), (tuple or list of tuples)
        The recording device that monitored the network's spikes
    limits : tuple of floats
        Time limits of the simulation regrion which should be studied (in ms).
    phase_coeff : tuple of floats, optional (default: (0.2, 5.))
        A phase is considered `bursting' when the interspike between all spikes
        that compose it is smaller than ``phase_coeff[0]*avg_rate`` (where
        ``avg_rate`` is the average firing rate), `quiescent' when it is greater
        that ``phase_coeff[1]*avg_rate``, `mixed' otherwise.
    mbis : float, optional (default: 0.5)
        Maximum interspike interval allowed for two spikes to be considered in
        the same burst (in ms).
    mfb : float, optional (default: 0.2)
        Minimal fraction of the neurons that should participate for a burst to
        be validated (i.e. if the interspike is smaller that the limit BUT the
        number of participating neurons is too small, the phase will be
        considered as `localized`).
    mflb : float, optional (default: 0.05)
        Minimal fraction of the neurons that should participate for a local 
        burst to be validated (i.e. if the interspike is smaller that the limit
        BUT the number of participating neurons is too small, the phase will be
        considered as `mixed`).
    simplify: bool, optional (default: False)
        If ``True``, `mixed` phases that are contiguous to a burst are
        incorporated to it.
    return_steps : bool, optional (default: False)
        If ``True``, a second dictionary, `phases_steps` will also be returned.
        @todo: not implemented yet
    fignums : list, optional (default: [])
        Indices of figures on which the periods can be drawn.
    show : bool, optional (default: False)
        Whether the figures should be displayed.
    
    Returns
    -------
    phases : dict
        Dictionary containing the time intervals (in ms) for all four phases
        (`bursting', `quiescent', `mixed', and `localized`) as lists.
        E.g: ``phases["bursting"]`` could give ``[[123.5,334.2],[857.1,1000.6]]``.
    phases_steps : dict, optional (not implemented yet)
        Dictionary containing the timesteps in NEST.
    '''
    # check if there are several recorders
    data, senders, times = [], [], []
    
    if True in nest.GetStatus(spike_detector, "to_file"):
        for fpath in nest.GetStatus(spike_detector, "record_to"):
            data.extend(fpath)
    else:
        for events in nest.GetStatus(spike_detector, "events"):
            times.extend(events["times"])
            senders.extend(events["senders"])
    idx_sorted = np.argsort(times)
    times = np.array(times)[idx_sorted]
    senders = np.array(senders)[idx_sorted]
    # set the studied region
    num_spikes = len(times)
    if limits[0] >= times[0]:
        idx_start = np.where(times>=limits[0])[0][0]
        times = times[idx_start:]
        senders = senders[idx_start:]
        num_spikes = len(times)
    else:
        times = np.concatenate(([limits[0]],times))
        senders = np.concatenate(([-1],senders))
    if limits[1] <= times[-1]:
        idx_end = np.where(times<=limits[0])[0][-1]
        times = times[:idx_end]
        senders = senders[:idx_end]
        num_spikes = len(times)
    else:
        times = np.concatenate((times,[limits[1]]))
    # get the average firing rate to differenciate the phases
    simtime = limits[1]-limits[0]
    avg_rate = num_spikes/float(simtime)
    lim_burst = max(phase_coeff[0]/avg_rate,mbis)
    lim_quiet = min(phase_coeff[1]/avg_rate,10.)
    # find the phases
    phases = { "bursting":[], "mixed":[], "quiescent":[], "localized": [] }
    diff = np.diff(times).tolist()[::-1]
    i = 0
    previous = { "bursting": -2, "mixed": -2, "quiescent": -2 }
    while diff:
        tau = diff.pop()
        while True:
            if tau < lim_burst: # bursting phase
                if previous["bursting"] == i-1:
                    phases["bursting"][-1][1] = times[i+1]
                else:
                    if simplify and previous["mixed"] == i-1:
                        start_mixed = phases["mixed"][-1][0]
                        phases["bursting"].append([start_mixed,times[i+1]])
                        del phases["mixed"][-1]
                    else:
                        phases["bursting"].append([times[i],times[i+1]])
                previous["bursting"] = i
                i+=1
                break
            elif tau > lim_quiet:
                if previous["quiescent"] == i-1:
                    phases["quiescent"][-1][1] = times[i+1]
                else:
                    phases["quiescent"].append([times[i],times[i+1]])
                previous["quiescent"] = i
                i+=1
                break
            else:
                if previous["mixed"] == i-1:
                    phases["mixed"][-1][1] = times[i+1]
                    previous["mixed"] = i
                else:
                    if simplify and previous["bursting"] == i-1:
                        phases["bursting"][-1][1] = times[i+1]
                        previous["bursting"] = i
                    else:
                        phases["mixed"].append([times[i],times[i+1]])
                        previous["mixed"] = i
                i+=1
                break
    # check that bursting periods involve at least mfb of the neurons
    transfer, destination = [], {}
    n = network.node_nb()
    for i,burst in enumerate(phases["bursting"]):
        idx_start = np.where(times==burst[0])[0][0]
        idx_end = np.where(times==burst[1])[0][0]
        participating_frac = len(set(senders[idx_start:idx_end]))/float(n)
        if participating_frac < mflb:
            transfer.append(i)
            destination[i] = "mixed"
        elif participating_frac < mfb:
            transfer.append(i)
            destination[i] = "localized"
    for i in transfer[::-1]:
        phase = phases["bursting"].pop(i)
        phases[destination[i]].insert(0,phase)
    remove = []
    i = 0
    while i < len(phases['mixed']):
        mixed = phases['mixed'][i]
        j=i+1
        for span in phases['mixed'][i+1:]:
            if span[0] == mixed[1]:
                mixed[1] = span[1]
                remove.append(j)
                i=-1
            elif span[1] == mixed[0]:
                mixed[0] = span[0]
                remove.append(j)
                i=-1
            j+=1
        i+=1
    remove = list(set(remove))
    remove.sort()
    for i in remove[::-1]:
        del phases["mixed"][i]
    colors = ('r', 'orange', 'g', 'b')
    names = ('bursting', 'mixed', 'localized', 'quiescent')
    if nngt.config['with_plot']:
        import matplotlib.pyplot as plt
        for fignum in fignums:
            fig = plt.figure(fignum)
            for ax in fig.axes:
                for phase,color in zip(names,colors):
                    for span in phases[phase]:
                        ax.axvspan(span[0],span[1], facecolor=color, alpha=0.2)
        if fignums and show:
            plt.show()
    return ActivityRecord(network, limits, spike_detector, phases)

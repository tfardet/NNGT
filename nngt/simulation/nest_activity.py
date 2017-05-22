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

""" Analyze the activity of a network """

from collections import namedtuple
from copy import deepcopy
import weakref

import matplotlib.pyplot as plt
import nest
import numpy as np
import scipy.sparse as ssp

from nngt.lib import InvalidArgument
from nngt.analysis.activity_analysis import _b2_from_data, _fr_from_data


__all__ = [
    "ActivityRecord",
    "activity_types",
    "analyze_raster",
    "get_b2",
    "get_firing_rate",
    "get_spikes",
]


#-----------------------------------------------------------------------------#
# Finding the various activities
#------------------------
#

class ActivityRecord:
    '''
    Class to record the properties of the simulated activity.
    '''
    
    def __init__(self, spike_data, phases, properties, parameters=None):
        '''
        Initialize the instance using `spike_data` (store proxy to an optional
        `network`) and compute the properties of provided data.

        Parameters
        ----------
        spike_data : 2D array
            Array of shape (num_spikes, 2), containing the senders on the 1st
            row and the times on the 2nd row.
        phases : dict
            Limits of the different phases in the simulated period.
        properties : dict
            Values of the different properties of the activity (e.g.
            "firing_rate", "IBI"...).
        parameters : dict, optional (default: None)
            Parameters used to compute the phases.
        
        Note
        ----
        The firing rate is computed as num_spikes / total simulation time,
        the period is the sum of an IBI and a bursting period.
        '''
        self._data = spike_data
        self._phases = phases.copy()
        self._properties = properties.copy()
        self.parameters = parameters

    def simplify():
        raise NotImplementedError("Will be implemented soon.")

    @property
    def data(self):
        '''
        Returns the (N, 2) array of (senders, spike times).
        '''
        return self._data

    @property
    def phases(self):
        '''
        Returns the phases detected:
            - "bursting" for periods of high activity where a large fraction
            of the network is recruited.
            - "quiescent" for periods of low activity
            - "mixed" for firing rate in between "quiescent" and "bursting".
            - "localized" for periods of high activity but where only a small
            fraction of the network is recruited.
        
        Note
        ----
        See `parameters` for details on the conditions used to
        differenciate these phases.
        '''
        return self._phases

    @property
    def properties(self):
        '''
        Returns the properties of the activity.
        Contains the following entries:
            - "firing_rate": average value in Hz for 1 neuron in the network.
            - "bursting": True if there were bursts of activity detected.
            - "burst_duration", "IBI", "ISI", and "period" in ms, if
              "bursting" is True.
            - "SpB" (Spikes per Burst): average number of spikes per neuron
              during a burst.
        '''
        return self._properties


#-----------------------------------------------------------------------------#
# analyse acitivity
#------------------------
#

def get_spikes(recorder=None, spike_times=None, senders=None):
    '''
    Return a 2D sparse matrix, where:
        - each row i contains the spikes of neuron i
        - each column j contains the times of the jth spike for all neurons

    Parameters
    ----------
    recorder : tuple, optional (default: None)
        Tuple of NEST gids, where the first one should point to the
        spike_detector which recorded the spikes.
    spike_times : array-like, optional (default: None)
        If `recorder` is not provided, the spikes' data can be passed directly
        through their `spike_times` and the associated `senders`.
    senders : array-like, optional (default: None)
        `senders[i]` corresponds to the neuron which fired at `spike_times[i]`.

    Example
    -------
    >> get_spikes()

    >> get_spikes(recorder)

    >> times = [1.5, 2.68, 125.6]
    >> neuron_ids = [12, 0, 65]
    >> get_spikes(spike_times=times, senders=neuron_ids)

    Note
    ----
    This function supposes that neuron GIDs for a continuous set of integers.
    If no arguments are passed to the function, the first spike_recorder
    available in NEST will be used.

    Returns
    -------
    CSR matrix containing the spikes sorted by neuron (rows) and time
    (columns).
    '''
    # get spikes
    if recorder is not None:
        data = nest.GetStatus(recorder[0])[0]["events"]
        spike_times = data["times"]
        senders = data["senders"]
        senders -= np.min(senders)
    elif spike_times is None and senders is None:
        nodes = nest.GetNodes(
            (0,), properties={'model': 'spike_detector'})
        data = nest.GetStatus(nodes[0])[0]["events"]
        spike_times = data["times"]
        senders = data["senders"]
        senders -= np.min(senders)
    # create the sparse matrix
    data = [0 for _ in range(len(set(senders)))]
    row_idx = []
    col_idx = []
    for time, neuron in zip(spike_times, senders):
        row_idx.append(neuron)
        col_idx.append(data[neuron])
        data[neuron] += 1
    return ssp.csr_matrix((spike_times, (row_idx, col_idx)))


def get_b2(network, spike_detector=None, data=None, nodes=None):
    '''
    Return the B2 coefficient for the neurons.

    Parameters
    ----------
    network : :class:`nngt.Network`
        Network for which the activity was simulated.
    spike_detector : tuple of ints, optional (default: spike detectors)
        GID of the "spike_detector" objects recording the network activity.
    data : array-like of shape (2, N), optionale (default: None)
        Array containing the spikes data (first line must contain the NEST GID
        of the neuron that fired, second line must contain the associated spike
        time).
    nodes : array-like, optional (default: all neurons)
        NNGT ids of the nodes for which the B2 should be computed.

    Returns
    -------
    b2 : array-like
        B2 coefficient for each neuron in `nodes`.
    '''
    if data is None:
        data = [[], []]
    if nodes is None:
        nodes = network.nest_gid
    else:
        nodes = network.nest_gid[nodes]
    if not len(data[0]):
        if spike_detector is None:
            spike_detector = nest.GetNodes(
                (0,), properties={'model': 'spike_detector'})[0]
        events = nest.GetStatus(spike_detector, "events")
        for ev_dict in events:
            data[0].extend(ev_dict["senders"])
            data[1].extend(ev_dict["times"])
    data[0] = np.array(data[0])
    data[1] = np.array(data[1])
    return _b2_from_data(nodes, data)


def get_firing_rate(network, spike_detector=None, data=None, nodes=None):
    '''
    Return the average firing rate for the neurons.

    Parameters
    ----------
    network : :class:`nngt.Network`
        Network for which the activity was simulated.
    spike_detector : tuple of ints, optional (default: spike detectors)
        GID of the "spike_detector" objects recording the network activity.
    data : array-like of shape (2, N), optionale (default: None)
        Array containing the spikes data (first line must contain the NEST GID
        of the neuron that fired, second line must contain the associated spike
        time).
    nodes : array-like, optional (default: all nodes)
        NNGT ids of the nodes for which the B2 should be computed.

    Returns
    -------
    fr : array-like
        Firing rate for each neuron in `nodes`.
    '''
    if data is None:
        data = [[], []]
    if nodes is None:
        nodes = network.nest_gid
    else:
        nodes = network.nest_gid[nodes]
    if not len(data[0]):
        if spike_detector is None:
            spike_detector = nest.GetNodes(
                (0,), properties={'model': 'spike_detector'})[0]
        events = nest.GetStatus(spike_detector, "events")
        for ev_dict in events:
            data[0].extend(ev_dict["senders"])
            data[1].extend(ev_dict["times"])
    data[0] = np.array(data[0])
    data[1] = np.array(data[1])
    return _fr_from_data(nodes, data)


def activity_types(spike_detector, limits, network=None,
                   phase_coeff=(0.5, 10.), mbis=0.5, mfb=0.2, mflb=0.05,
                   skip_bursts=0, simplify=False, fignums=[], show=False):
    '''
    Analyze the spiking pattern of a neural network.

    @todo:
      think about inserting t=0. and t=simtime at the beginning and at the 
      end of ``times``.

    Parameters
    ----------
    spike_detector : NEST node(s), (tuple or list of tuples)
        The recording device that monitored the network's spikes
    limits : tuple of floats
        Time limits of the simulation regrion which should be studied (in ms).
    network : :class:`~nngt.Network`, optional (default: None)
        Neural network that was analyzed
    phase_coeff : tuple of floats, optional (default: (0.2, 5.))
        A phase is considered `bursting' when the interspike between all spikes
        that compose it is smaller than ``phase_coeff[0] / avg_rate`` (where
        ``avg_rate`` is the average firing rate), `quiescent' when it is
        greater that ``phase_coeff[1] / avg_rate``, `mixed' otherwise.
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
    skip_bursts : int, optional (default: 0)
        Skip the `skip_bursts` first bursts to consider only the permanent
        regime.
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

    Note
    ----
    Effects of `skip_bursts` and `limits[0]` are cumulative: the `limits[0]`
    first milliseconds are ignored, then the `skip_bursts` first bursts of the
    remaining activity are ignored.
    
    Returns
    -------
    phases : dict
        Dictionary containing the time intervals (in ms) for all four phases
        (`bursting', `quiescent', `mixed', and `localized`) as lists.
        E.g: ``phases["bursting"]`` could give ``[[123.5,334.2],
        [857.1,1000.6]]``.
    '''
    # check if there are several recorders
    senders, times = [], []
    if True in nest.GetStatus(spike_detector, "to_file"):
        for fpath in nest.GetStatus(spike_detector, "record_to"):
            data = _get_data(fpath)
            times.extend(data[:, 1])
            senders.extend(data[:, 0])
    else:
        for events in nest.GetStatus(spike_detector, "events"):
            times.extend(events["times"])
            senders.extend(events["senders"])
        idx_sort = np.argsort(times)
        times = np.array(times)[idx_sort]
        senders = np.array(senders)[idx_sort]
    # compute phases and properties
    data = np.array((senders, times))
    phases, fr = _analysis(times, senders, limits, network=network,
              phase_coeff=phase_coeff, mbis=mbis, mfb=mfb, mflb=mflb,
              simplify=simplify)
    properties = _compute_properties(data, phases, fr, skip_bursts)
    kwargs = {
        "limits": limits,
        "phase_coeff": phase_coeff,
        "mbis": mbis,
        "mfb": mfb,
        "mflb": mflb,
        "simplify": simplify
    }
    # plot if required
    if show:
        _plot_phases(phases, fignums)
    return ActivityRecord(data, phases, properties, kwargs)


def analyze_raster(raster=None, limits=None, network=None,
                   phase_coeff=(0.5, 10.), mbis=0.5, mfb=0.2, mflb=0.05,
                   skip_bursts=0, skip_ms=0., simplify=False, fignums=[],
                   show=False):
    '''
    Return the activity types for a given raster.

    Parameters
    ----------
    raster : array-like or str
        Either an array containing the ids of the spiking neurons and the
        corresponding time, or the path to a NEST .gdf recording.
    limits : tuple of floats
        Time limits of the simulation regrion which should be studied (in ms).
    network : :class:`~nngt.Network`, optional (default: None)
        Network on which the recorded activity was simulated.
    phase_coeff : tuple of floats, optional (default: (0.2, 5.))
        A phase is considered `bursting' when the interspike between all spikes
        that compose it is smaller than ``phase_coeff[0] / avg_rate`` (where
        ``avg_rate`` is the average firing rate), `quiescent' when it is
        greater that ``phase_coeff[1] / avg_rate``, `mixed' otherwise.
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
    skip_bursts : int, optional (default: 0)
        Skip the `skip_bursts` first bursts to consider only the permanent
        regime.
    simplify: bool, optional (default: False)
        If ``True``, `mixed` phases that are contiguous to a burst are
        incorporated to it.
    fignums : list, optional (default: [])
        Indices of figures on which the periods can be drawn.
    show : bool, optional (default: False)
        Whether the figures should be displayed.
    
    Note
    ----
    Effects of `skip_bursts` and `limits[0]` are cumulative: the 
    `limits[0]` first milliseconds are ignored, then the `skip_bursts`
    first bursts of the remaining activity are ignored.

    Returns
    -------
    activity : ActivityRecord
        Object containing the phases and the properties of the activity
        from these phases.
    '''
    data = _get_data(raster) if isinstance(raster, str) else raster
    if limits is None:
        limits = [np.min(data[:, 1]), np.max(data[:, 1])]
    kwargs = {
        "limits": limits,
        "phase_coeff": phase_coeff,
        "mbis": mbis,
        "mfb": mfb,
        "mflb": mflb,
        "simplify": simplify
    }
    # compute phases and properties
    phases, fr = _analysis(data[:, 1], data[:, 0], limits, network=network,
              phase_coeff=phase_coeff, mbis=mbis, mfb=mfb, mflb=mflb,
              simplify=simplify)
    properties = _compute_properties(data.T, phases, fr, skip_bursts)
    # plot if required
    if show:
        if fignums:
            _plot_phases(phases, fignums)
        else:
            fig, ax = plt.subplots()
            ax.scatter(data[:, 1], data[:, 0])
            _plot_phases(phases, [fig.number])
    return ActivityRecord(data, phases, properties, kwargs)


def _get_data(source):
    '''
    Returns the (times, senders) array.

    Parameters
    ----------
    source : tuple or str
        Index of a spike detector or path to the .gdf file.
    
    Returns
    -------
    data : 2D array of shape (N, 2)
    '''
    data = None
    if isinstance(source, str):
        data = np.loadtxt(source)
        idx_sort = np.argsort(data[:, 1])
        data = data[idx_sort, :]
    else:
        events = nest.GetStatus(source, "events")
        idx_sort = np.argsort(events["times"])
        data = np.zeros((len(events["times"]), 2))
        data[:, 1] = events["times"][idx_sort]
        data[:, 0] = events["senders"][idx_sort]
    return data


def _find_phases(times, phases, lim_burst, lim_quiet, simplify):
    '''
    Find the time limits of the different phases.
    '''
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
                        phases["bursting"].append([start_mixed, times[i+1]])
                        del phases["mixed"][-1]
                    else:
                        phases["bursting"].append([times[i], times[i+1]])
                previous["bursting"] = i
                i+=1
                break
            elif tau > lim_quiet:
                if previous["quiescent"] == i-1:
                    phases["quiescent"][-1][1] = times[i+1]
                else:
                    phases["quiescent"].append([times[i], times[i+1]])
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
                        phases["mixed"].append([times[i], times[i+1]])
                        previous["mixed"] = i
                i+=1
                break


def _check_burst_size(phases, senders, times, network, mflb, mfb):
    '''
    Check that bursting periods involve at least a fraction mfb of the neurons.
    '''
    transfer, destination = [], {}
    n = len(set(senders)) if network is None else network.node_nb()
    for i,burst in enumerate(phases["bursting"]):
        idx_start = np.where(times==burst[0])[0][0]
        idx_end = np.where(times==burst[1])[0][0]
        participating_frac = len(set(senders[idx_start:idx_end])) / float(n)
        if participating_frac < mflb:
            transfer.append(i)
            destination[i] = "mixed"
        elif participating_frac < mfb:
            transfer.append(i)
            destination[i] = "localized"
    for i in transfer[::-1]:
        phase = phases["bursting"].pop(i)
        phases[destination[i]].insert(0, phase)
    remove = []
    i = 0
    while i < len(phases['mixed']):
        mixed = phases['mixed'][i]
        j=i+1
        for span in phases['mixed'][i+1:]:
            if span[0] == mixed[1]:
                mixed[1] = span[1]
                remove.append(j)
            elif span[1] == mixed[0]:
                mixed[0] = span[0]
                remove.append(j)
            j+=1
        i+=1
    remove = list(set(remove))
    remove.sort()
    for i in remove[::-1]:
        del phases["mixed"][i]


def _analysis(times, senders, limits, network=None,
              phase_coeff=(0.5, 10.), mbis=0.5, mfb=0.2, mflb=0.05,
              simplify=False):
    # prepare the phases and check the validity of the data
    phases = {
        "bursting": [],
        "mixed": [],
        "quiescent": [],
        "localized": []
    }
    num_spikes, avg_rate = len(times), 0.
    if num_spikes:
        num_neurons = (len(np.unique(senders)) if network is None
                       else network.node_nb())
        # set the studied region
        if limits[0] >= times[0]:
            idx_start = np.where(times >= limits[0])[0][0]
            times = times[idx_start:]
            senders = senders[idx_start:]
        if limits[1] <= times[-1]:
            idx_end = np.where(times <= limits[1])[0][-1]
            times = times[:idx_end]
            senders = senders[:idx_end]
        # get the average firing rate to differenciate the phases
        simtime = limits[1] - limits[0]
        lim_burst, lim_quiet = 0., 0.
        avg_rate = num_spikes / float(simtime)
        lim_burst = max(phase_coeff[0] / avg_rate, mbis)
        lim_quiet = min(phase_coeff[1] / avg_rate, 10.)
        # find the phases
        _find_phases(times, phases, lim_burst, lim_quiet, simplify)
        _check_burst_size(phases, senders, times, network, mflb, mfb)
        avg_rate *= 1000. / float(num_neurons)
    return phases, avg_rate


def _compute_properties(data, phases, fr, skip_bursts):
    '''
    Compute the properties from the spike times and phases.
    
    Parameters
    ----------
    data : 2D array, shape (N, 2)
        Spike times and senders.
    phases : dict
        The phases.
    fr : double
        Firing rate.

    Returns
    -------
    prop : dict
        Properties of the activity. Contains the following pairs:
            - "firing_rate": average value in Hz for 1 neuron in the network.
            - "bursting": True if there were bursts of activity detected.
            - "burst_duration", "ISI", and "IBI" in ms, if "bursting" is True.
            - "SpB": average number of spikes per burst for one neuron.
    '''
    prop = {}
    times = data[1, :]
    # firing rate (in Hz, normalized for 1 neuron)
    prop["firing_rate"] = fr
    num_bursts = len(phases["bursting"])
    init_val = 0. if num_bursts > skip_bursts else np.NaN
    if num_bursts:
        prop["bursting"] = True
        prop.update({
            "burst_duration": init_val,
            "IBI": init_val,
            "ISI": init_val,
            "SpB": init_val,
            "period": init_val})
    else:
        prop["bursting"] = False
    for i, burst in enumerate(phases["bursting"]):
        if i >= skip_bursts:
            # burst_duration
            prop["burst_duration"] += burst[1] - burst[0]
            # IBI
            if i > 0:
                end_older_burst = phases["bursting"][i-1][1]
                prop["IBI"] += burst[0]-end_older_burst
            # get num_spikes inside the burst, divide by num_neurons
            idxs = np.where((times >= burst[0])*(times <= burst[1]))[0]
            num_spikes = len(times[idxs])
            num_neurons = len(set(data[0, :][idxs]))
            prop["SpB"] += num_spikes / float(num_neurons)
            # ISI
            prop["ISI"] += num_neurons * (burst[1] - burst[0])\
                           / float(num_spikes)
    for key in iter(prop.keys()):
        if key not in ("bursting", "firing_rate") and num_bursts > skip_bursts:
            prop[key] /= float(num_bursts - skip_bursts)
    if num_bursts > skip_bursts:
        prop["period"] = prop["IBI"] + prop["burst_duration"]
    if num_bursts and prop["SpB"] < 2.:
        prop["ISI"] = np.NaN
    return prop


def _plot_phases(phases, fignums):
    colors = ('r', 'orange', 'g', 'b')
    names = ('bursting', 'mixed', 'localized', 'quiescent')
    for fignum in fignums:
        fig = plt.figure(fignum)
        for ax in fig.axes:
            for phase, color in zip(names, colors):
                for span in phases[phase]:
                    ax.axvspan(span[0], span[1], facecolor=color,
                               alpha=0.2)
    plt.show()

#-*- coding:utf-8 -*-
#
# analysis/activity_analysis.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Tools for activity analysis from data """

import logging

import numpy as np
import scipy.signal as sps
import scipy.sparse as ssp

from nngt.lib import nonstring_container, find_idx_nearest
from nngt.lib.logger import _log_message


__all__ = [
    "get_b2",
    "get_firing_rate",
    "get_spikes",
    "total_firing_rate",
]


logger = logging.getLogger(__name__)


# ----------------------- #
# Get activity properties #
# ----------------------- #

def get_b2(network=None, spike_detector=None, data=None, nodes=None):
    '''
    Return the B2 coefficient for the neurons.

    Parameters
    ----------
    network : :class:`nngt.Network`, optional (default: None)
        Network for which the activity was simulated.
    spike_detector : tuple of ints, optional (default: spike detectors)
        GID of the "spike_detector" objects recording the network activity.
    data : array-like of shape (N, 2), optionale (default: None)
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
        data, nodes = _set_data_nodes(network, data, nodes)
        data = _set_spike_data(data, spike_detector)
    else:
        if nodes is None:
            nodes = np.unique(data[:, 0])

    return _b2_from_data(nodes, data)


def get_firing_rate(network=None, spike_detector=None, data=None, nodes=None):
    '''
    Return the average firing rate for the neurons.

    Parameters
    ----------
    network : :class:`nngt.Network`, optional (default: None)
        Network for which the activity was simulated.
    spike_detector : tuple of ints, optional (default: spike detectors)
        GID of the "spike_detector" objects recording the network activity.
    data : :class:`numpy.array` of shape (N, 2), optionale (default: None)
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
        data, nodes = _set_data_nodes(network, data, nodes)
        data = _set_spike_data(data, spike_detector)
    else:
        if nodes is None:
            nodes = np.unique(data[:, 0])

    return _fr_from_data(nodes, data)


def total_firing_rate(network=None, spike_detector=None, nodes=None, data=None,
                      kernel_center=0., kernel_std=30., resolution=None,
                      cut_gaussian=5.):
    '''
    Computes the total firing rate of the network from the spike times.
    Firing rate is obtained as the convolution of the spikes with a Gaussian
    kernel characterized by a standard deviation and a temporal shift.

    .. versionadded:: 0.7

    Parameters
    ----------
    network : :class:`nngt.Network`, optional (default: None)
        Network for which the activity was simulated.
    spike_detector : tuple of ints, optional (default: spike detectors)
        GID of the "spike_detector" objects recording the network activity.
    data : :class:`numpy.array` of shape (N, 2), optionale (default: None)
        Array containing the spikes data (first line must contain the NEST GID
        of the neuron that fired, second line must contain the associated spike
        time).
    kernel_center : float, optional (default: 0.)
        Temporal shift of the Gaussian kernel, in ms.
    kernel_std : float, optional (default: 30.)
        Characteristic width of the Gaussian kernel (standard deviation) in ms.
    resolution : float or array, optional (default: `0.1*kernel_std`)
        The resolution at which the firing rate values will be computed.
        Choosing a value smaller than `kernel_std` is strongly advised.
        If resolution is an array, it will be considered as the times were the
        firing rate should be computed.
    cut_gaussian : float, optional (default: 5.)
        Range over which the Gaussian will be computed. By default, we consider
        the 5-sigma range. Decreasing this value will increase speed at the
        cost of lower fidelity; increasing it with increase the fidelity at the
        cost of speed.

    Returns
    -------
    fr : array-like
        The firing rate in Hz.
    times : array-like
        The times associated to the firing rate values.
    '''
    times, kernel_size = None, None

    if data is None:
        data, _ = _set_data_nodes(network, data, nodes)
        data    = _set_spike_data(data, spike_detector)

    # set resolution and kernel properties + generate the times
    if resolution is None:
        resolution = 0.1*kernel_std

    if nonstring_container(resolution):
        dt = np.diff(resolution)
        assert np.allclose(dt - dt[0], 0.), 'If `resolution` is an array, ' +\
                                            'it must contain evenly spaced ' +\
                                            'times.'
        times = np.array(resolution)
        resolution = dt[0]

    bin_std = int(kernel_std / float(resolution))
    kernel_size = int(2. * cut_gaussian * bin_std)

    if times is None:
        delta_T = resolution * 0.5 * kernel_size
        times = np.arange(np.min(data[:, 1]) - delta_T,
                          np.max(data[:, 1]) + delta_T, resolution)

    rate = np.zeros(len(times))

    # counts the spikes at each time
    pos = find_idx_nearest(times, data[:, 1])
    bins = np.linspace(0, len(times), len(times)+1)
    counts, _ = np.histogram(pos, bins=bins)

    # initialize with delta rate in Hz
    rate += 1000. * counts / (kernel_std*np.sqrt(np.pi))
    fr = _smooth(rate, kernel_size, bin_std, mode='same')

    # translate times
    times += kernel_center

    return fr, times


def get_spikes(recorder=None, spike_times=None, senders=None, astype="ssp"):
    '''
    Return a 2D sparse matrix, where:

    - each row i contains the spikes of neuron i (in NEST),
    - each column j contains the times of the jth spike for all neurons.

    .. versionchanged:: 1.0
        Neurons are now located in the row corresponding to their NEST GID.

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
    astype : str, optional (default: "ssp")
        Format of the returned data. Default is sparse lil_matrix ("ssp")
        with one row per neuron, otherwise "np" returns a (T, 2) array, with
        T the number of spikes (the first row being the NEST gid, the second
        the spike time).

    Example
    -------
    >>> get_spikes()

    >>> get_spikes(recorder)

    >>> times = [1.5, 2.68, 125.6]
    >>> neuron_ids = [12, 0, 65]
    >>> get_spikes(spike_times=times, senders=neuron_ids)

    Note
    ----
    If no arguments are passed to the function, the first spike_recorder
    available in NEST will be used.
    Neuron positions correspond to their GIDs in NEST.

    Returns
    -------
    CSR matrix containing the spikes sorted by neuron GIDs (rows) and time
    (columns).
    '''
    if recorder is not None:
        from ..simulation.nest_utils import _get_nest_gids
        import nest
        data = nest.GetStatus(_get_nest_gids(recorder))[0]["events"]
        spike_times = data["times"]
        senders = data["senders"]
    elif spike_times is None and senders is None:
        from ..simulation.nest_utils import _get_nest_gids, spike_rec
        import nest
        nodes = nest.GetNodes(properties={'model': spike_rec})
        data = nest.GetStatus(nodes)[0]["events"]
        spike_times = data["times"]
        senders = data["senders"]

    if astype == "np":
        return np.array([senders, spike_times]).T
    elif astype == "ssp":
        if np.any(senders):
            max_sender = np.max(senders)
            # create the sparse matrix
            data = [0 for _ in range(max_sender + 1)]
            row_idx = []
            col_idx = []
            for time, neuron in zip(spike_times, senders):
                row_idx.append(neuron)
                col_idx.append(data[neuron])
                data[neuron] += 1
            return ssp.csr_matrix((spike_times, (row_idx, col_idx)))
        else:
            return ssp.csr_matrix([])


# ----- #
# Tools #
# ----- #

def _b2_from_data(ids, data):
    b2 = np.full(len(ids), np.NaN)
    if len(data[:, 0]) > 0:
        for i, neuron in enumerate(ids):
            ids = np.where(data[:, 0] == neuron)[0]
            dt1 = np.diff(data[ids, 1])
            dt2 = dt1[1:] + dt1[:-1]
            avg_isi = np.mean(dt1)
            if avg_isi != 0.:
                b2[i] = (2*np.var(dt1) - np.var(dt2)) / (2*avg_isi**2)
            else:
                b2[i] = np.inf
    else:
        _log_message(logger, "WARNING", 'No spikes in the data.')
    return b2


def _fr_from_data(ids, data):
    fr = np.zeros(len(ids))

    if len(data[:, 0]):
        T = float(np.max(data[:, 1]) - np.min(data[:, 1]))

        for i, neuron in enumerate(ids):
            ids = np.where(data[:, 0] == neuron)[0]
            fr[i] = len(ids) / T

    return fr


def _set_data_nodes(network, data, nodes):
    from ..simulation.nest_utils import _get_nest_gids

    if data is None:
        data = [[], []]

    if nodes is None:
        nodes = network.nest_gids
    else:
        nodes = network.nest_gids[nodes]

    return data, _get_nest_gids(nodes)


def _set_spike_data(data, spike_detector):
    '''
    Data must be [[], []]
    '''
    import nest
    from ..simulation.nest_utils import _get_nest_gids, spike_rec, nest_version

    if not len(data[0]):
        if spike_detector is None:
            prop = {'model': spike_rec}
            if nest_version == 3:
                spike_detector = nest.GetNodes(properties=prop)
            else:
                spike_detector = nest.GetNodes((0,), properties=prop)[0]

        events = nest.GetStatus(spike_detector, "events")

        for ev_dict in events:
            data[0].extend(ev_dict["senders"])
            data[1].extend(ev_dict["times"])

    sorter = np.argsort(data[1])

    return np.array(data)[:, sorter].T


def _smooth(data, kernel_size, std, mode='same'):
    '''
    Convolve an array by a Gaussian kernel.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel array in bins.
    std : float
        Width of the Gaussian (also in bins).

    Returns
    -------
    convolved array.
    '''
    kernel = sps.gaussian(kernel_size, std)
    kernel /= np.sum(kernel)
    return sps.convolve(data, kernel, mode=mode)

#-*- coding:utf-8 -*-
#
# simulation/nest_plot.py
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

# nest_plot.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Utility functions to plot NEST simulated activity """

import itertools
import logging

from matplotlib.colors import ColorConverter
import numpy as np
import nest

import nngt

from nngt.analysis import total_firing_rate
from nngt.lib import InvalidArgument, nonstring_container, is_integer
from nngt.lib.sorting import _sort_groups, _sort_neurons
from nngt.lib.logger import _log_message
from nngt.plot import palette_discrete, markers
from nngt.plot.plt_properties import _set_new_plot, _set_ax_lims

from .nest_utils import nest_version, spike_rec, _get_nest_gids

logger = logging.getLogger(__name__)


# --------------------- #
# Plotting the activity #
# --------------------- #

def plot_activity(gid_recorder=None, record=None, network=None, gids=None,
                  axis=None, show=False, limits=None, histogram=False,
                  title=None, fignum=None, label=None, sort=None,
                  average=False, normalize=1., decimate=None, transparent=True,
                  kernel_center=0., kernel_std=None, resolution=None,
                  cut_gaussian=5., **kwargs):
    '''
    Plot the monitored activity.

    .. versionchanged:: 1.2
        Switched `hist` to `histogram` and default value to False.

    .. versionchanged:: 1.0.1
        Added `axis` parameter, restored missing `fignum` parameter.

    Parameters
    ----------
    gid_recorder : tuple or list of tuples, optional (default: None)
        The gids of the recording devices. If None, then all existing
        spike_recs are used.
    record : tuple or list, optional (default: None)
        List of the monitored variables for each device. If `gid_recorder` is
        None, record can also be None and only spikes are considered.
    network : :class:`~nngt.Network` or subclass, optional (default: None)
        Network which activity will be monitored.
    gids : tuple, optional (default: None)
        NEST gids of the neurons which should be monitored.
    axis : matplotlib axis object, optional (default: new one)
        Axis that should be use to plot the activity. This takes precedence
        over `fignum`.
    show : bool, optional (default: False)
        Whether to show the plot right away or to wait for the next plt.show().
    histogram : bool, optional (default: False)
        Whether to display the histogram when plotting spikes rasters.
    limits : tuple, optional (default: None)
        Time limits of the plot (if not specified, times of first and last
        spike for raster plots).
    title : str, optional (default: None)
        Title of the plot.
    fignum : int, or dict, optional (default: None)
        Plot the activity on an existing figure (from ``figure.number``). This
        parameter is ignored if `axis` is provided.
    label : str or list, optional (default: None)
        Add labels to the plot (one per recorder).
    sort : str or list, optional (default: None)
        Sort neurons using a topological property ("in-degree", "out-degree",
        "total-degree" or "betweenness"), an activity-related property
        ("firing_rate" or neuronal property) or a user-defined list of sorted
        neuron ids. Sorting is performed by increasing value of the `sort`
        property from bottom to top inside each group.
    normalize : float or list, optional (default: None)
        Normalize the recorded results by a given float. If a list is provided,
        there should be one entry per voltmeter or multimeter in the recorders.
        If the recording was done through `monitor_groups`, the population can
        be passed to normalize the data by the nuber of nodes in each group.
    decimate : int or list of ints, optional (default: None)
        Represent only a fraction of the spiking neurons; only one neuron in
        `decimate` will be represented (e.g. setting `decimate` to 5 will lead
        to only 20% of the neurons being represented). If a list is provided,
        it must have one entry per NeuralGroup in the population.
    kernel_center : float, optional (default: 0.)
        Temporal shift of the Gaussian kernel, in ms (for the histogram).
    kernel_std : float, optional (default: 0.5% of simulation time)
        Characteristic width of the Gaussian kernel (standard deviation) in ms
        (for the histogram).
    resolution : float or array, optional (default: `0.1*kernel_std`)
        The resolution at which the firing rate values will be computed.
        Choosing a value smaller than `kernel_std` is strongly advised.
        If resolution is an array, it will be considered as the times were the
        firing rate should be computed (for the histogram).
    cut_gaussian : float, optional (default: 5.)
        Range over which the Gaussian will be computed (for the histogram).
        By default, we consider the 5-sigma range. Decreasing this value will
        increase speed at the cost of lower fidelity; increasing it with
        increase the fidelity at the cost of speed.
    **kwargs : dict
        "color" and "alpha" values can be overriden here.

    Warning
    -------
    Sorting with "firing_rate" only works if NEST gids form a continuous
    integer range.

    Returns
    -------
    lines : list of lists of :class:`matplotlib.lines.Line2D`
        Lines containing the data that was plotted, grouped by figure.
    '''
    import matplotlib.pyplot as plt
    recorders = _get_nest_gids([])
    lst_labels, lines, axes, labels = [], {}, {}, {}

    # normalize recorders and recordables
    if gid_recorder is not None:
        assert record is not None, "`record` must also be provided."
        if len(record) != len(gid_recorder):
            raise InvalidArgument('`record` must either be the same for all '
                                  'recorders, or contain one entry per '
                                  'recorder in `gid_recorder`')
        for rec in gid_recorder:
            if nest_version == 3:
                recorders = _get_nest_gids(gid_recorder)
            else:
                if isinstance(gid_recorder[0], tuple):
                    recorders.append(rec)
                else:
                    recorders.append((rec,))
    else:
        prop = {'model': spike_rec}
        if nest_version == 3:
            recorders = nest.GetNodes(properties=prop)
        else:
            recorders = [
                (gid,) for gid in nest.GetNodes((0,), properties=prop)[0]
            ]

        record = tuple("spikes" for _ in range(len(recorders)))

    # get gids and groups
    gids = network.nest_gids if (gids is None and network is not None) \
           else gids

    if gids is None:
        gids = []

        for rec in recorders:
            gids.extend(nest.GetStatus(rec)[0]["events"]["senders"])

        gids = np.unique(gids)

    num_group = 1 if network is None else len(network.population)
    num_lines = max(num_group, len(recorders))

    # sorting
    sorted_neurons = np.array([])

    if len(gids):
        sorted_neurons = np.arange(
            np.max(gids) + 1).astype(int) - np.min(gids) + 1

    attr = None

    if sort is not None:
        assert network is not None, "`network` is required for sorting."
        if nonstring_container(sort):
            attr = sort
            sorted_neurons = _sort_neurons(attr, gids, network)
            sort = "user defined sort"
        else:
            data = None
            if sort.lower() in ("firing_rate", "b2"):  # get senders
                data = [[], []]
                for rec in recorders:
                    info = nest.GetStatus(rec)[0]
                    if str(info["model"]) == spike_rec:
                        data[0].extend(info["events"]["senders"])
                        data[1].extend(info["events"]["times"])
                data = np.array(data).T
            sorted_neurons, attr = _sort_neurons(
                sort, gids, network, data=data, return_attr=True)
    elif network is not None and network.is_spatial():
        sorted_neurons, attr = _sort_neurons(
            "space", gids, network, data=None, return_attr=True)

    # spikes plotting
    colors = palette_discrete(np.linspace(0, 1, num_lines))
    num_raster, num_detec, num_meter = 0, 0, 0
    fignums = fignum if isinstance(fignum, dict) else {}
    decim = []
    if decimate is None:
        decim = [None for _ in range(num_lines)]
    elif is_integer(decimate):
        decim = [decimate for _ in range(num_lines)]
    elif nonstring_container(decimate):
        assert len(decimate) == num_lines, "`decimate` should have one " +\
                                           "entry per plot."
        decim = decimate
    else:
        raise AttributeError(
            "`decimate` must be either an int or a list of `int`.")

    # set labels
    if label is None:
        lst_labels = [None for _ in range(len(recorders))]
    else:
        if isinstance(label, str):
            lst_labels = [label]
        else:
            lst_labels = label
        if len(label) != len(recorders):
            _log_message(logger, "WARNING",
                         'Incorrect length for `label`: expecting {} but got '
                         '{}.\nIgnoring.'.format(len(recorders), len(label)))
            lst_labels = [None for _ in range(len(recorders))]

    datasets = []
    max_time = 0.

    for rec in recorders:
        info = nest.GetStatus(rec)[0]

        if len(info["events"]["times"]):
            max_time = max(max_time, np.max(info["events"]["times"]))

        datasets.append(info)

    if kernel_std is None:
        kernel_std = max_time*0.005

    if resolution is None:
        resolution = 0.5*kernel_std

    # plot
    for info, var, lbl in zip(datasets, record, lst_labels):
        fnum = fignums.get(info["model"], fignum)
        if info["model"] not in labels:
            labels[info["model"]] = []
            lines[info["model"]] = []

        if str(info["model"]) == spike_rec:
            if spike_rec in axes:
                axis = axes[spike_rec]
            c = colors[num_raster]
            times, senders = info["events"]["times"], info["events"]["senders"]
            sorted_ids = sorted_neurons[senders]
            l = raster_plot(times, sorted_ids, color=c, show=False,
                            limits=limits, sort=sort, fignum=fnum, axis=axis,
                            decimate=decim[num_raster], sort_attribute=attr,
                            network=network, histogram=histogram,
                            transparent=transparent,
                            hist_ax=axes.get('histogram', None),
                            kernel_center=kernel_center,
                            kernel_std=kernel_std, resolution=resolution,
                            cut_gaussian=cut_gaussian)
            num_raster += 1
            if l:
                fig_raster = l[0].figure.number
                fignums[spike_rec] = fig_raster
                axes[spike_rec] = l[0].axes
                labels[spike_rec].append(lbl)
                lines[spike_rec].extend(l)
                if histogram:
                    axes['histogram'] = l[1].axes
        elif "detector" in str(info["model"]):
            c = colors[num_detec]
            times, senders = info["events"]["times"], info["events"]["senders"]
            sorted_ids = sorted_neurons[senders]
            l = raster_plot(times, sorted_ids, fignum=fnum, color=c, axis=axis,
                            show=False, histogram=histogram, limits=limits,
                            kernel_center=kernel_center,
                            kernel_std=kernel_std, resolution=resolution,
                            cut_gaussian=cut_gaussian)
            if l:
                fig_detect = l[0].figure.number
                num_detec += 1
                fignums[info["model"]] = fig_detect
                labels[info["model"]].append(lbl)
                lines[info["model"]].extend(l)
                if histogram:
                    axes['histogram'] = l[1].axes
        else:
            da_time  = info["events"]["times"]
            # prepare axis setup
            fig = None
            if axis is None:
                fig = plt.figure(fnum)
                fignums[info["model"]] = fig.number
            else:
                fig = axis.get_figure()
            lines_tmp, labels_tmp = [], []
            if nonstring_container(var):
                m_colors = palette_discrete(np.linspace(0, 1, len(var)))
                axes = fig.axes
                if axis is not None:
                    # multiple y axes on a single subplot, adapted from
                    # https://matplotlib.org/examples/pylab_examples/
                    # multiple_yaxis_with_spines.html
                    axes = [axis]
                    axis.name = var[0]
                    if len(var) > 1:
                        axes.append(axis.twinx())
                        axes[-1].name = var[1]
                    if len(var) > 2:
                        fig.subplots_adjust(right=0.75)
                        for i, name in zip(range(len(var)-2), var[2:]):
                            new_ax = axis.twinx()
                            new_ax.spines["right"].set_position(
                                ("axes", 1.2*(i+1)))
                            axes.append(new_ax)
                            _make_patch_spines_invisible(new_ax)
                            new_ax.spines["right"].set_visible(True)
                            axes[-1].name = name
                if not axes:
                    axes = _set_new_plot(fig.number, names=var)[1]
                labels_tmp = [lbl for _ in range(len(var))]
                for subvar, c in zip(var, m_colors):
                    c = kwargs.get('color', c)
                    alpha = kwargs.get('alpha', 1)
                    for ax in axes:
                        if ax.name == subvar:
                            da_subvar = info["events"][subvar]
                            if isinstance(normalize, nngt.NeuralPop):
                                da_subvar /= normalize[num_meter].size
                            elif nonstring_container(normalize):
                                da_subvar /= normalize[num_meter]
                            elif normalize is not None:
                                da_subvar /= normalize
                            lines_tmp.extend(
                                ax.plot(da_time, da_subvar, color=c,
                                        alpha=alpha))
                            ax.set_ylabel(subvar)
                            ax.set_xlabel("time")
                            if limits is not None:
                                ax.set_xlim(limits[0], limits[1])
            else:
                num_axes, ax = len(fig.axes), axis
                if axis is None:
                    ax = fig.add_subplot(num_axes + 1, 1, num_axes + 1)
                da_var = info["events"][var]
                c = kwargs.get('color', None)
                alpha = kwargs.get('alpha', 1)
                lines_tmp.extend(ax.plot(da_time, da_var/normalize, color=c,
                                         alpha=alpha))
                labels_tmp.append(lbl)
                ax.set_ylabel(var)
                ax.set_xlabel("time")
            labels[info["model"]].extend(labels_tmp)
            lines[info["model"]].extend(lines_tmp)
            num_meter += 1

    if spike_rec in axes:
        ax = axes[spike_rec]

        if limits is not None:
            ax.set_xlim(limits[0], limits[1])
        else:
            t_min, t_max, idx_min, idx_max = np.inf, -np.inf, np.inf, -np.inf

            for l in ax.lines:
                t_max = max(np.max(l.get_xdata()), t_max)
                t_min = min(np.min(l.get_xdata()), t_max)
                idx_min = min(np.min(l.get_ydata()), idx_min)
                idx_max = max(np.max(l.get_ydata()), idx_max)

            dt   = t_max - t_min
            didx = idx_max - idx_min
            pc   = 0.02

            if not np.any(np.isinf((t_max, t_min))):
                ax.set_xlim([t_min - pc*dt, t_max + pc*dt])

            if not np.any(np.isinf((idx_min, idx_max))):
              ax.set_ylim([idx_min - pc*didx, idx_max + pc*didx])

    for recorder in fignums:
        fig = plt.figure(fignums[recorder])
        if title is not None:
            fig.suptitle(title)
        if label is not None:
            fig.legend(lines[recorder], labels[recorder])

    if show:
        plt.show()

    return lines


def raster_plot(times, senders, limits=None, title="Spike raster",
                histogram=False, num_bins=1000, color="b", decimate=None,
                axis=None, fignum=None, label=None, show=True, sort=None,
                sort_attribute=None, network=None, transparent=True,
                kernel_center=0., kernel_std=30., resolution=None,
                cut_gaussian=5., **kwargs):
    """
    Plotting routine that constructs a raster plot along with
    an optional histogram.

    .. versionchanged:: 1.2
        Switched `hist` to `histogram`.

    .. versionchanged:: 1.0.1
        Added `axis` parameter.

    Parameters
    ----------
    times : list or :class:`numpy.ndarray`
        Spike times.
    senders : list or :class:`numpy.ndarray`
        Index for the spiking neuron for each time in `times`.
    limits : tuple, optional (default: None)
        Time limits of the plot (if not specified, times of first and last
        spike).
    title : string, optional (default: 'Spike raster')
        Title of the raster plot.
    histogram : bool, optional (default: True)
        Whether to plot the raster's histogram.
    num_bins : int, optional (default: 1000)
        Number of bins for the histogram.
    color : string or float, optional (default: 'b')
        Color of the plot lines and markers.
    decimate : int, optional (default: None)
        Represent only a fraction of the spiking neurons; only one neuron in
        `decimate` will be represented (e.g. setting `decimate` to 10 will lead
        to only 10% of the neurons being represented).
    axis : matplotlib axis object, optional (default: new one)
        Axis that should be use to plot the activity.
    fignum : int, optional (default: None)
        Id of another raster plot to which the new data should be added.
    label : str, optional (default: None)
        Label the current data.
    show : bool, optional (default: True)
        Whether to show the plot right away or to wait for the next plt.show().
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
        Range over which the Gaussian will be computed (for the histogram).
        By default, we consider the 5-sigma range. Decreasing this value will
        increase speed at the cost of lower fidelity; increasing it with
        increase the fidelity at the cost of speed.

    Returns
    -------
    lines : list of :class:`matplotlib.lines.Line2D`
        Lines containing the data that was plotted.
    """
    import matplotlib.pyplot as plt

    lines = []

    mpl_kwargs = {k: v for k, v in kwargs.items() if k != 'hist_ax'}

    if label is None:
        mpl_kwargs['label'] = label

    # decimate if necessary
    if decimate is not None:
        idx_keep = np.where(np.mod(senders, decimate) == 0)[0]
        senders = senders[idx_keep]
        times = times[idx_keep]

    if len(times):
        if axis is not None:
            fig = axis.get_figure()
        else:
            fig = plt.figure(fignum)
        if transparent:
            fig.patch.set_visible(False)
        ylabel = "Neuron ID"
        xlabel = "Time (ms)"

        delta_t = 0.01*(times[-1]-times[0])

        if histogram:
            ax1, ax2 = None, None
            if kwargs.get("hist_ax", None) is None:
                num_axes = len(fig.axes)
                for i, old_ax in enumerate(fig.axes):
                    old_ax.change_geometry(num_axes + 2, 1, i+1)
                ax1 = fig.add_subplot(num_axes + 2, 1, num_axes + 1)
                ax2 = fig.add_subplot(num_axes + 2, 1, num_axes + 2,
                                      sharex=ax1)
            else:
                ax1 = axis
                ax2 = kwargs["hist_ax"]

            if limits is not None:
                start, stop = limits

                keep    = (times >= start)&(times <= stop)
                times   = times[keep]
                senders = senders[keep]

            lines.extend(ax1.plot(
                times, senders, c=color, marker="o", linestyle='None',
                mec="k", mew=0.5, ms=4, **mpl_kwargs))

            ax1_lines = ax1.lines

            if len(ax1_lines) > 1:
                t_max = max(ax1_lines[0].get_xdata().max(),times[-1])
                ax1.set_xlim([-delta_t, t_max+delta_t])

            ax1.set_ylabel(ylabel)

            if limits is not None:
                ax1.set_xlim(*limits)

            fr, fr_times = total_firing_rate(
                data=np.array([senders, times]).T, kernel_center=kernel_center,
                kernel_std=kernel_std, resolution=resolution,
                cut_gaussian=cut_gaussian)

            hist_lines = ax2.get_lines()

            if hist_lines:
                data = hist_lines[-1].get_data()
                bottom = data[1]
                if limits is None:
                    dt = fr_times[1] - fr_times[0]
                    old_times = data[0]
                    old_start = int(old_times[0] / dt)
                    new_start = int(fr_times[0] / dt)
                    old_end = int(old_times[-1] / dt)
                    new_end = int(fr_times[-1] / dt)
                    diff_start = new_start-old_start
                    diff_end = new_end-old_end
                    if diff_start > 0:
                        bottom = bottom[diff_start:]
                    else:
                        bottom = np.concatenate(
                            (np.zeros(-diff_start), bottom))
                    if diff_end > 0:
                        bottom = np.concatenate((bottom, np.zeros(diff_end)))
                    else:
                        bottom = bottom[:diff_end-1]
                    b_len, h_len = len(bottom), len(fr)
                    if  b_len > h_len:
                        bottom = bottom[:h_len]
                    elif b_len < h_len:
                        bottom = np.concatenate(
                            (bottom, np.zeros(h_len-b_len)))
                else:
                    bottom = bottom[:-1]

                ax2.fill_between(fr_times, fr + bottom, bottom, color=color)
                lines.extend(ax2.plot(fr_times, fr + bottom, ls="", marker=""))
            else:
                ax2.fill_between(fr_times, fr, 0., color=color)
                lines.extend(ax2.plot(fr_times, fr, ls="", marker=""))

            ax2.set_ylabel("Rate (Hz)")
            ax2.set_xlabel(xlabel)
            ax2.set_xlim(ax1.get_xlim())
            _second_axis(sort, sort_attribute, ax1)
        else:
            if axis is not None:
                ax = axis
            else:
                num_axes = len(fig.axes)
                for i, old_ax in enumerate(fig.axes):
                    old_ax.change_geometry(num_axes + 1, 1, i+1)
                ax = fig.add_subplot(num_axes + 1, 1, num_axes + 1)

            if limits is not None:
                start, stop = limits

                keep    = (times >= start)&(times <= stop)
                times   = times[keep]
                senders = senders[keep]

            if network is not None:
                pop = network.population
                colors = palette_discrete(np.linspace(0, 1, len(pop)))
                mm = itertools.cycle(markers)
                for m, (k, v), c in zip(mm, pop.items(), colors):
                    keep = np.where(
                        np.in1d(senders, network.nest_gids[v.ids]))[0]
                    if len(keep):
                        if label is None:
                            mpl_kwargs['label'] = k
                        lines.extend(ax.plot(
                            times[keep], senders[keep], c=c, marker=m,
                            ls='None', mec='k', mew=0.5, ms=4, **mpl_kwargs))
            else:
                lines.extend(ax.plot(
                    times, senders, c=color, marker="o", linestyle='None',
                    mec="k", mew=0.5, ms=4, **mpl_kwargs))

            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            if limits is not None:
                ax.set_xlim(limits)
            else:
                _set_ax_lims(ax, np.max(times), np.min(times), np.max(senders),
                             np.min(senders))

            if label is not None:
                ax.legend(bbox_to_anchor=(1.1, 1.2))
            _second_axis(sort, sort_attribute, ax)

        fig.suptitle(title)

        if show:
            plt.show()
    else:
        _log_message(logger, "WARNING",
                     "No activity was detected during the simulation.")

    return lines


#-----------------------------------------------------------------------------
# Tools
#------------------------
#

def _fill_between_steps(x, y1, y2=0, h_align='mid'):
    '''
    Fills a hole in matplotlib: fill_between for step plots.

    Parameters :
    ------------
    x : array-like
        Array/vector of index values. These are assumed to be equally-spaced.
        If not, the result will probably look weird...
    y1 : array-like
        Array/vector of values to be filled under.
    y2 : array-Like
        Array/vector or bottom values for filled area. Default is 0.
    '''
    # First, duplicate the x values
    xx = np.repeat(x,2)
    # Now: the average x binwidth
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    #~ xx = np.append(xx, xx.max() + xstep[-1])

    # Make it possible to change step alignment.
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = np.repeat(y1,2)#[:-1]
    if type(y2) == np.ndarray:
        y2 = np.repeat(y2,2)#[:-1]

    return xx, y1, y2


def _moving_average (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'same')
    return sma


def _second_axis(sort, sort_attribute, ax):
    import matplotlib.pyplot as plt
    if sort is not None:
        fig = ax.get_figure()
        twin = None
        for axis in fig.axes:
            if axis.get_ylabel() == sort:
                twin = axis
                break
        if twin is None:
            asort = np.argsort(sort_attribute)
            twin = ax.twinx()
            twin.grid(False)
            twin.set_ylabel(sort)
            plt.draw()
            old_ticks = ax.get_yticks()
            twin.set_yticks(old_ticks)
            twin.set_ylim(ax.get_ylim())
            labels = ['' for _ in range(len(old_ticks))]
            idx_max = len(sort_attribute) - 1
            for i, t in enumerate(old_ticks):
                if t >= 0:
                    idx = min(int(t), idx_max)
                    labels[i] = _sci_format(sort_attribute[asort[idx]])
            twin.set_yticklabels(labels)


def _sci_format(n):
    label = ''
    if np.abs(n) < 0.01 or np.abs(n) >= 1000:
        a = '{:.1E}'.format(n)
        label = '$' + a.split('E')[0].rstrip('0').rstrip('.') + '\\cdot 10^{'
        exponent = a.split('E')[1].lstrip('0')
        if exponent[0] == '-':
            exponent = exponent[0] + exponent[1:].lstrip('0')
        elif exponent[0] == '+':
            exponent = exponent[1:].lstrip('0')
        label += exponent + '}$'
    elif np.abs(n) >= 100:
       label = '{:.0f}'.format(n)
    elif np.abs(n) >= 10:
       label = '{:.1f}'.format(n)
    else:
       label = '{:.2f}'.format(n)
    return label

def _make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

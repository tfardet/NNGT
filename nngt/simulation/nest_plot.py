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

# nest_plot.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Utility functions to plot NEST simulated activity """

import logging

from matplotlib.colors import ColorConverter
import numpy as np
import nest

import nngt
from nngt.plot import palette, markers
from nngt.plot.plt_properties import _set_new_plot, _set_ax_lims
from nngt.lib import InvalidArgument, nonstring_container, is_integer
from nngt.lib.sorting import _sort_groups, _sort_neurons
from nngt.lib.logger import _log_message


logger = logging.getLogger(__name__)


# --------------------- #
# Plotting the activity #
# --------------------- #

def plot_activity(gid_recorder=None, record=None, network=None, gids=None,
                  show=False, limits=None, hist=True, title=None, label=None,
                  sort=None, average=False, normalize=1., decimate=None,
                  transparent=True):
    '''
    Plot the monitored activity.
    
    Parameters
    ----------
    gid_recorder : tuple or list of tuples, optional (default: None)
        The gids of the recording devices. If None, then all existing
        "spike_detector"s are used.
    record : tuple or list, optional (default: None)
        List of the monitored variables for each device. If `gid_recorder` is
        None, record can also be None and only spikes are considered.
    network : :class:`~nngt.Network` or subclass, optional (default: None)
        Network which activity will be monitored.
    gids : tuple, optional (default: None)
        NEST gids of the neurons which should be monitored.
    show : bool, optional (default: False)
        Whether to show the plot right away or to wait for the next plt.show().
    hist : bool, optional (default: True)
        Whether to display the histogram when plotting spikes rasters.
    limits : tuple, optional (default: None)
        Time limits of the plot (if not specified, times of first and last
        spike for raster plots).
    title : str, optional (default: None)
        Title of the plot.
    fignum : int, optional (default: None)
        Plot the activity on an existing figure (from ``figure.number``).
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
    lst_rec, lst_labels, lines, labels = [], [], {}, {}
    num_fig = np.max(plt.get_fignums()) if plt.get_fignums() else 0
    # normalize recorders and recordables
    if gid_recorder is not None:
        if len(record) != len(gid_recorder):
            raise InvalidArgument('`record` must either be the same for all '
                                  'recorders, or contain one entry per '
                                  'recorder in `gid_recorder`')
        for rec in gid_recorder:
            if isinstance(gid_recorder[0], tuple):
                lst_rec.append(rec[0])
            else:
                lst_rec.append(rec)
    else:
        lst_rec = nest.GetNodes(
            (0,), properties={'model': 'spike_detector'})[0]
        record = tuple("spikes" for _ in range(len(lst_rec)))
    # get gids and groups
    gids = network.nest_gid if (gids is None and network is not None) else gids
    if gids is None:
        gids = []
        for rec in lst_rec:
            gids.extend(nest.GetStatus([rec])[0]["events"]["senders"])
        gids = np.unique(gids)
    num_group = len(network.population) if network is not None else 1
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
                for rec in lst_rec:
                    info = nest.GetStatus([rec])[0]
                    if str(info["model"]) == "spike_detector":
                        data[0].extend(info["events"]["senders"])
                        data[1].extend(info["events"]["times"])
                data = np.array(data).T
            sorted_neurons, attr = _sort_neurons(
                sort, gids, network, data=data, return_attr=True)
    # spikes plotting
    colors = palette(np.linspace(0, 1, num_group))
    num_raster, num_detec, num_meter = 0, 0, 0
    fignums = {}
    decim = []
    if decimate is None:
        decim = [None for _ in range(num_group)]
    elif is_integer(decimate):
        decim = [decimate for _ in range(num_group)]
    elif nonstring_container(decimate):
        assert len(decimate) == num_group, "`decimate` should have one " +\
                                           "entry per group in the population."
        decim = decimate
    else:
        raise AttributeError(
            "`decimate` must be either an int or a list of `int`.")

    # set labels
    if label is None:
        lst_labels = [None for _ in range(len(lst_rec))]
    else:
        if isinstance(label, str):
            lst_labels = [label]
        else:
            lst_labels = label
        if len(label) != len(lst_rec):
            _log_message(logger, "WARNING",
                         'Incorrect length for `label`: expecting {} but got '
                         '{}.\nIgnoring.'.format(len(lst_rec), len(label)))
            lst_labels = [None for _ in range(len(lst_rec))]

    # plot
    for rec, var, lbl in zip(lst_rec, record, lst_labels):
        info = nest.GetStatus([rec])[0]
        fnum = fignums[info["model"]] if info["model"] in fignums else None
        if info["model"] not in labels:
            labels[info["model"]] = []
            lines[info["model"]] = []
        if str(info["model"]) == "spike_detector":
            c = colors[num_raster]
            times, senders = info["events"]["times"], info["events"]["senders"]
            sorted_ids = sorted_neurons[senders]
            l = raster_plot(times, sorted_ids, color=c, show=False,
                            limits=limits, sort=sort, fignum=fnum,
                            decimate=decim[num_raster], sort_attribute=attr,
                            network=network, transparent=transparent)
            num_raster += 1
            if l:
                fig_raster = l[0].figure.number
                fignums['spike_detector'] = fig_raster
                labels["spike_detector"].append(lbl)
                lines["spike_detector"].extend(l)
        elif "detector" in str(info["model"]):
            c = colors[num_detec]
            times, senders = info["events"]["times"], info["events"]["senders"]
            sorted_ids = sorted_neurons[senders]
            l = raster_plot(times, sorted_ids, fignum=fnum, color=c,
                            show=False, hist=hist, limits=limits)
            if l:
                fig_detect = l[0].figure.number
                num_detec += 1
                fignums[info["model"]] = fig_detect
                labels[info["model"]].append(lbl)
                lines[info["model"]].extend(l)
        else:
            da_time = info["events"]["times"]
            fig = plt.figure(fnum)
            fignums[info["model"]] = fig.number
            lines_tmp, labels_tmp = [], []
            if nonstring_container(var):
                axes = fig.axes
                if not axes:
                    axes = _set_new_plot(fig.number, names=var)[1]
                labels_tmp = [lbl for _ in range(len(var))]
                for subvar in var:
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
                                ax.plot(da_time, da_subvar))
                            ax.set_ylabel(subvar)
                            ax.set_xlabel("time")
                            if limits is not None:
                                ax.set_xlim(limits[0], limits[1])
            else:
                ax = fig.add_subplot(111)
                da_var = info["events"][var]
                lines_tmp.extend(ax.plot(da_time, da_var/normalize))
                labels_tmp.append(lbl)
                ax.set_ylabel(var)
                ax.set_xlabel("time")
            labels[info["model"]].extend(labels_tmp)
            lines[info["model"]].extend(lines_tmp)
            num_meter += 1
    for recorder in fignums:
        fig = plt.figure(fignums[recorder])
        if title is not None:
            fig.suptitle(title)
        if label is not None:
            fig.legend(lines[recorder], labels[recorder])
    if show:
        plt.show()
    return lines


def raster_plot(times, senders, limits=None, title="Spike raster", hist=False,
                num_bins=1000, color="b", decimate=None, fignum=None,
                label=None, show=True, sort=None, sort_attribute=None,
                network=None, transparent=True):
    """
    Plotting routine that constructs a raster plot along with
    an optional histogram.
    
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
    hist : bool, optional (default: True)
        Whether to plot the raster's histogram.
    num_bins : int, optional (default: 1000)
        Number of bins for the histogram.
    color : string or float, optional (default: 'b')
        Color of the plot lines and markers.
    decimate : int, optional (default: None)
        Represent only a fraction of the spiking neurons; only one neuron in
        `decimate` will be represented (e.g. setting `decimate` to 10 will lead
        to only 10% of the neurons being represented).
    fignum : int, optional (default: None)
        Id of another raster plot to which the new data should be added.
    label : str, optional (default: None)
        Label the current data.
    show : bool, optional (default: True)
        Whether to show the plot right away or to wait for the next plt.show().
    
    Returns
    -------
    lines : list of :class:`matplotlib.lines.Line2D`
        Lines containing the data that was plotted.
    """
    import matplotlib.pyplot as plt
    num_neurons = len(np.unique(senders))
    lines = []
    kwargs = {} if label is None else {'label': label}

    # decimate if necessary
    if decimate is not None:
        idx_keep = np.where(np.mod(senders, decimate) == 0)[0]
        senders = senders[idx_keep]
        times = times[idx_keep]

    if len(times):
        fig = plt.figure(fignum)
        if transparent:
            fig.patch.set_visible(False)
        ylabel = "Neuron ID"
        xlabel = "Time (ms)"

        delta_t = 0.01*(times[-1]-times[0])

        if hist:
            ax1, ax2 = None, None
            if len(fig.axes) == 2:
                ax1 = fig.axes[0]
                ax2 = fig.axes[1]
            else:
                ax1 = fig.add_axes([0.1, 0.3, 0.85, 0.6])
                ax2 = fig.add_axes([0.1, 0.08, 0.85, 0.17], sharex=ax1)
            lines.extend(ax1.plot(
                times, senders, c=color, marker="o", linestyle='None',
                mec="k", mew=0.5, ms=4, **kwargs))
            ax1_lines = ax1.lines
            if len(ax1_lines) > 1:
                t_max = max(ax1_lines[0].get_xdata().max(),times[-1])
                ax1.set_xlim([-delta_t, t_max+delta_t])
            ax1.set_ylabel(ylabel)
            if limits is not None:
                ax1.set_xlim(*limits)
            #~ ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

            bin_width = ( np.amax(times) - np.amin(times) ) / float(num_bins)
            t_bins = np.linspace(np.amin(times), np.amax(times), num_bins)
            if limits is not None:
                t_bins = np.linspace(limits[0], limits[1], num_bins)
            n, bins = np.histogram(times, bins=t_bins)
            #~ n = _moving_average(n,5)
            t_bins = np.concatenate(([t_bins[0]], t_bins))
            #~ heights = 1000 * n / (hist_binwidth * num_neurons)
            # height = rate in Hz, knowing that t is in ms
            heights = 1000*np.concatenate(([0],n,[0]))/(num_neurons*bin_width)
            height = np.repeat(0, len(heights)) if bin_width == 0. else heights
            lines = ax2.patches
            if lines:
                data = lines[-1].get_xy()
                bottom = data[:,1]
                if limits is None:
                    old_bins = data[:,0]
                    old_start = int(old_bins[0] / (old_bins[2]-old_bins[0]))
                    new_start = int(t_bins[0] / (t_bins[2]-t_bins[0]))
                    old_end = int(old_bins[-2] / (old_bins[-2]-old_bins[-3]))
                    new_end = int(t_bins[-1] / (t_bins[-1]-t_bins[-2]))
                    diff_start = new_start-old_start
                    diff_end = new_end-old_end
                    if diff_start > 0:
                        bottom = bottom[diff_start:]
                    else:
                        bottom = np.concatenate((np.zeros(-diff_start),bottom))
                    if diff_end > 0:
                        bottom = np.concatenate((bottom,np.zeros(diff_end)))
                    else:
                        bottom = bottom[:diff_end-1]
                    b_len, h_len = len(bottom), len(heights)
                    if  b_len > h_len:
                        bottom = bottom[:h_len]
                    elif b_len < h_len:
                        bottom = np.concatenate((bottom,np.zeros(h_len-b_len)))
                else:
                    bottom = bottom[:-1]
                #~ x,y1,y2 = _fill_between_steps(t_bins,heights,bottom[::2], h_align='left')
                #~ x,y1,y2 = _fill_between_steps(t_bins[:-1],heights+bottom[::2], bottom[::2], h_align='left')
                ax2.fill_between(t_bins,heights+bottom, bottom, color=color)
            else:
                #~ x,y1,_ = _fill_between_steps(t_bins,heights, h_align='left')
                #~ x,y1,_ = _fill_between_steps(t_bins[:-1],heights)
                ax2.fill(t_bins,heights, color=color)
            yticks = [int(x) for x in np.linspace(0,int(max(heights)*1.1)+5,4)]
            ax2.set_yticks(yticks)
            ax2.set_ylabel("Rate (Hz)")
            ax2.set_xlabel(xlabel)
            ax2.set_xlim(ax1.get_xlim())
            _second_axis(sort, sort_attribute, ax1)
        else:
            ax = fig.axes[0] if fig.axes else fig.add_subplot(111)
            if network is not None:
                for m, (k, v) in zip(markers, network.population.items()):
                    keep = np.where(
                        np.in1d(senders, network.nest_gid[v.ids]))[0]
                    if len(keep):
                        if label is None:
                            kwargs['label'] = k
                        lines.extend(ax.plot(
                            times[keep], senders[keep], c=color, marker=m,
                            ls='None', mec='k', mew=0.5, ms=4, **kwargs))
                        if 'inh' in k:
                            c_rgba = ColorConverter().to_rgba(color, alpha=0.5)
                            lines[-1].set_markerfacecolor(c_rgba)
            else:
                lines.extend(ax.plot(
                    times, senders, c=color, marker="o", linestyle='None',
                    mec="k", mew=0.5, ms=4, **kwargs))
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
        asort = np.argsort(sort_attribute)
        ax3 = ax.twinx()
        ax3.grid(False)
        ax3.set_ylabel(sort)
        plt.draw()
        old_ticks = ax.get_yticks()
        ax3.set_yticks(old_ticks)
        ax3.set_ylim(ax.get_ylim())
        labels = ['' for _ in range(len(old_ticks))]
        idx_max = len(sort_attribute) - 1
        for i, t in enumerate(old_ticks):
            if t >= 0:
                idx = min(int(t), idx_max)
                labels[i] = _sci_format(sort_attribute[asort[idx]])
        ax3.set_yticklabels(labels)


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

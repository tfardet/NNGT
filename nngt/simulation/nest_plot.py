#!/usr/bin/env python
#-*- coding:utf-8 -*-

# nest_plot.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Utility functions to plot NEST simulated activity """

import matplotlib.pyplot as plt
import numpy as np
import nest

from nngt.plot import palette
from nngt.plot.plt_properties import _set_new_plot
from nngt.lib import InvalidArgument
from .nest_utils import _sort_groups


#-----------------------------------------------------------------------------#
# Plotting the activity
#------------------------
#

def plot_activity(gid_recorder, record, network=None, gids=None, show=True,
                  limits=None, hist=True, title=None, fignum=None, label=None,
                  sort=None, normalize=1., decimate=None):
    '''
    Plot the monitored activity.
    
    Parameters
    ----------
    gid_recorder : tuple or list
        The gids of the recording devices.
    record : tuple or list
        List of the monitored variables for each device.
    network : :class:`~nngt.Network` or subclass, optional (default: None)
        Network which activity will be monitored.
    gids : tuple, optional (default: None)
        NEST gids of the neurons which should be monitored.
    show : bool, optional (default: True)
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
    label : str, optional (default: None)
        Add a label to the plot.
    sort : str or list, optional (default: None)
        Node property among ("in-degree", "out-degree", "total-degree" or
        "betweenness") or list of sorted neuron ids. Neurons are sorted
        by increasing value of the `sort` property from bottom to top inside
        each group.
    normalize : float, optional (default: None)
        Normalize the recorded results by a given float.
    decimate : int or list of ints, optional (default: None)
        Represent only a fraction of the spiking neurons; only one neuron in
        `decimate` will be represented (e.g. setting `decimate` to 10 will lead
        to only 10% of the neurons being represented). If a list is provided,
        it must have one entry per NeuralGroup in the population.

    Returns
    -------
    fignums : list
        List of the figure numbers.
    '''
    lst_rec = []
    for rec in gid_recorder:
        if isinstance(gid_recorder[0],tuple):
            lst_rec.append(rec)
        else:
            lst_rec.append((rec,))
    # get gids and groups
    gids = network.nest_gid if (gids is None and network is not None) else gids
    if gids is None:
        gids = nest.GetStatus(lst_rec[0])[0]["events"]["senders"]
    num_group = len(network.population) if network is not None else 1
    # sorting
    sorted_neurons = np.arange(np.max(gids)+1).astype(int) - np.min(gids) + 1
    if sort and network is not None:
        sorted_neurons = _sort_neurons(sort, gids, network)
    # spikes plotting
    colors = palette(np.linspace(0, 1, num_group))
    num_raster, num_detec = 0, 0
    fig_raster, fig_detec = None, None
    fignums = []
    decim = []
    if decimate is None:
        decim = [None for _ in range(num_group)]
    elif isinstance(decimate, int):
        decim = [decimate for _ in range(num_group)]
    elif hasattr(decimate, "__len__"):
        assert len(decimate) == num_group, "`decimate` should have one \
entry per group in the population"
        decim = decimate
    else:
        raise AttributeError(
            "`decimate must` be either an int or a list of ints")

    # plot
    for rec, var in zip(lst_rec, record):
        info = nest.GetStatus(rec)[0]
        if str(info["model"]) == "spike_detector":
            c = colors[num_raster]
            times, senders = info["events"]["times"], info["events"]["senders"]
            sorted_ids = sorted_neurons[senders]
            fig_raster = raster_plot(times, sorted_ids, fignum=fig_raster,
                                    color=c, show=False, label=info["label"],
                                    limits=limits, decimate=decim[num_raster])
            num_raster += 1
            fignums.append(fig_raster)
        elif "detector" in str(info["model"]):
            c = colors[num_detec]
            times, senders = info["events"]["times"], info["events"]["senders"]
            sorted_ids = sorted_neurons[senders]
            fig_detec = raster_plot(times, sorted_ids, fignum=fig_detec,
                                    color=c, show=False, hist=hist,
                                    label=info["label"], limits=limits)
            num_detec += 1
            fignums.append(fig_detect)
        else:
            da_time = info["events"]["times"]
            fig = plt.figure(fignum)
            if isinstance(var,list) or isinstance(var,tuple):
                axes = fig.axes
                if not axes:
                    axes = _set_new_plot(fig.number, names=var)[1]
                for subvar in var:
                    for ax in axes:
                        if ax.name == subvar:
                            da_subvar = info["events"][subvar]
                            ax.plot(da_time, da_subvar/normalize, label=label)
                            ax.set_ylabel(subvar)
                            ax.set_xlabel("time")
                            if limits is not None:
                                ax.set_xlim(limits[0], limits[1])
                            if label is not None:
                                ax.legend()
            else:
                ax = fig.add_subplot(111)
                da_var = info["events"][var]
                ax.plot(da_time, da_var/normalize, label=label)
                ax.set_ylabel(var)
                ax.set_xlabel("time")
                if label is not None:
                    ax.legend()
            fignums.append(fig.number)
    if title is not None:
        for n in fignums:
            fig = plt.figure(n)
            fig.suptitle(title)
    if show:
        plt.show()
    return list(set(fignums))


def raster_plot(times, senders, limits=None, title="Spike raster", hist=False,
                num_bins=1000, color="b", decimate=None, fignum=None,
                label=None, show=True):
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
    fig.number : int
        Id of the :class:`matplotlib.Figure` on which the raster is plotted.
    """
    num_neurons = len(np.unique(senders))

    # decimate if necessary
    if decimate is not None:
        idx_keep = np.where(np.mod(senders, decimate) == 0)[0]
        senders = senders[idx_keep]
        times = times[idx_keep]

    if len(times):
        fig = plt.figure(fignum)
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
            ax1.plot(times, senders, c=color, marker="o", linestyle='None',
                mec="k", mew=0.5, ms=4, label=label)
            ax1_lines = ax1.lines
            if len(ax1_lines) > 1:
                t_max = max(ax1_lines[0].get_xdata().max(),times[-1])
                ax1.set_xlim([-delta_t, t_max+delta_t])
            ax1.set_ylabel(ylabel)
            if limits is not None:
                ax1.set_xlim(*limits)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

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
        else:
            ax = fig.axes[0] if fig.axes else fig.add_subplot(111)
            ax.plot(times, senders, c=color, marker="o", linestyle='None',
                mec="k", mew=0.5, ms=4, label=label)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if limits is not None:
                ax.set_xlim(limits)
            else:
                ax.set_xlim([times[0]-delta_t, times[-1]+delta_t])
            ax.legend(bbox_to_anchor=(1.1, 1.2))

        fig.suptitle(title)
        if show:
            plt.show()
        return fig.number


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

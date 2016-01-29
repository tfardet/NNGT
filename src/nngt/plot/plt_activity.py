#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Tools to plot graph properties """

import matplotlib.pyplot as plt
import numpy as np

from nngt import NeuralPop
from nngt.plot.custom_plt import palette, format_exponent
from nngt.analysis import degree_distrib, betweenness_distrib


__all__ = [ 'spike_raster' ]


def spike_raster(spike_data, limits=None, title="Spike raster", hist=True,
                 num_bins=1000, neural_groups=None, fignum=None, show=True):
    """
    Plotting routine that constructs a raster plot along with
    an optional histogram.
    
    Parameters
    ----------
    spike_data : 2D-array (:class:`numpy.array` or list)
        An 2-column array containing the neuron ids in the first row and the
        spike times in the second.
    limits : tuple, optional (default: None)
        Time limits of the plot (if not specified, times of first and last
        spike).
    title : string, optional (default: 'Spike raster')
        Title of the raster plot.
    hist : bool, optional (default: True)
        Whether to plot the raster's histogram.
    num_bins : int, optional (default: 1000)
        Number of bins for the histogram.
    neural_groups : :class:`~nngt.NeuralPop` or list of neuron ids
        An object that defines the different neural groups to plot their spikes
        in different colors.
    fignum : int, optional (default: None)
        Id of another raster plot to which the new data should be added.
    show : bool, optional (default: True)
        Whether to show the plot right away or to wait for the next plt.show().
    
    Returns
    -------
    fig.number : int
        Id of the :class:`matplotlib.Figure` on which the raster is plotted.
    """
    senders, ts = spike_data[:,0], spike_data[:,1]
    num_neurons = len(np.unique(senders))
    # make the groups
    di_groups = {}
    lst_names = []
    if issubclass(neural_groups.__class__, NeuralPop):
        for i,(name,group) in enumerate(neural_groups.iteritems()):
            lst_names.append(name)
            for neuron in group.id_list:
                di_groups[neuron] = i
    elif neural_groups is not None:
        for i,group in enumerate(neural_groups):
            for neuron in group:
                di_groups[neuron] = i
        lst_names = [ "spikes group {}".format(i+1)
                      for i in range(len(neural_groups)) ]
    else:
        lst_names = [ "spikes" ]
    # sort the spikes into their groups
    lst_spikes = []
    gid_min = np.min(senders)-1
    if di_groups:
        lst_spikes = [ [[],[]] for group in range(len(neural_groups)) ]
        for time, sender in zip(ts,senders):
            glist = lst_spikes[di_groups[sender]]
            glist[0].append(sender-gid_min)
            glist[1].append(time)
    else:
        lst_spikes.append((senders, ts))
    # plot
    fig = plt.figure()
    ax1, ax2 = None, None
    if hist:
        ax1 = fig.add_axes([0.1, 0.3, 0.85, 0.6])
        ax2 = fig.add_axes([0.1, 0.08, 0.85, 0.17], sharex=ax1)
    else:
        ax1 = fig.add_subplot(111)
    n = len(lst_spikes)
    for i,spikes in enumerate(lst_spikes):
        _spike_raster(spikes, lst_names[i], title, limits, hist, num_bins,
                      palette(float(i)/n), fig.number, show=False)
    if show:
        plt.show()
    return fig.number
    
        
def _spike_raster(spike_data, legend, title, limits=None, hist=True,
                  num_bins=1000, color="b", fignum=None, show=True):
    """
    Plotting routine that constructs a raster plot along with
    an optional histogram.
    
    Parameters
    ----------
    detec : tuple
        Gid of the NEST detector from which the data should be recovered.
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
    fignum : int, optional (default: None)
        Id of another raster plot to which the new data should be added.
    show : bool, optional (default: True)
        Whether to show the plot right away or to wait for the next plt.show().
    
    Returns
    -------
    fig.number : int
        Id of the :class:`matplotlib.Figure` on which the raster is plotted.
    """
    senders, ts = spike_data[0], spike_data[1]
    num_neurons = len(np.unique(senders))

    if len(ts):
        fig = plt.figure(fignum)

        ylabel = "Neuron ID"
        xlabel = "Time (ms)"

        delta_t = 0.01*(ts[-1]-ts[0])

        if hist:
            ax1, ax2 = None, None
            if len(fig.axes) == 2:
                ax1 = fig.axes[0]
                ax2 = fig.axes[1]
            else:
                ax1 = fig.add_axes([0.1, 0.3, 0.85, 0.6])
                ax2 = fig.add_axes([0.1, 0.08, 0.85, 0.17], sharex=ax1)
            ax1.plot(ts, senders, c=color, marker="o", linestyle='None',
                mec="k", mew=0.5, ms=4, label=legend)
            ax1_lines = ax1.lines
            if len(ax1_lines) > 1:
                t_max = max(ax1_lines[0].get_xdata().max(),ts[-1])
                ax1.set_xlim([-delta_t, t_max+delta_t])
            ax1.set_ylabel(ylabel)
            if limits is not None:
                ax1.set_xlim(*limits)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

            bin_width = ( np.amax(ts) - np.amin(ts) ) / float(num_bins)
            t_bins = np.linspace(np.amin(ts), np.amax(ts), num_bins)
            if limits is not None:
                t_bins = np.linspace(limits[0], limits[1], num_bins)
            n, bins = np.histogram(ts, bins=t_bins)
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
                #~ x,y1,y2 = fill_between_steps(t_bins,heights,bottom[::2], h_align='left')
                #~ x,y1,y2 = fill_between_steps(t_bins[:-1],heights+bottom[::2], bottom[::2], h_align='left')
                ax2.fill_between(t_bins,heights+bottom, bottom, color=color)
            else:
                #~ x,y1,_ = fill_between_steps(t_bins,heights, h_align='left')
                #~ x,y1,_ = fill_between_steps(t_bins[:-1],heights)
                ax2.fill(t_bins,heights, color=color)
            yticks = [int(x) for x in np.linspace(0,int(max(heights)*1.1)+5,4)]
            ax2.set_yticks(yticks)
            ax2.set_ylabel("Rate (Hz)")
            ax2.set_xlabel(xlabel)
            ax2.set_xlim(ax1.get_xlim())
        else:
            ax = fig.axes[0] if fig.axes else fig.subplots(111)
            ax.plot(ts, senders, c=color, marker="o", linestyle='None',
                mec="k", mew=0.5, ms=4, label=legend)
            ax.set_ylabel(ylabel)
            ax.set_ylim([np.min(senders),np.max(senders)])
            ax.set_xlim([ts[0]-delta_t, ts[-1]+delta_t])
            ax.legend(bbox_to_anchor=(1.1, 1.2))

        fig.suptitle(title)
        if show:
            plt.show()
        return fig.number

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

from ..lib import InvalidArgument
from ..plot.custom_plt import palette
from ..plot.plt_properties import _set_new_plot


__all__ = [
            'set_noise',
            'set_poisson_input',
            'set_step_currents',
            'monitor_nodes',
            'plot_activity'
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


#-----------------------------------------------------------------------------#
# Monitoring the activity
#------------------------
#

def monitor_nodes(gids, nest_recorder=["spike_detector"], params=[{}],
                  network=None):
    '''
    Monitoring the activity of nodes in the network.
    
    Parameters
    ----------
    gids : tuple of ints or list of tuples
        GIDs of the neurons in the NEST subnetwork; either one list per
        recorder if they should monitor different neurons or a unique list
        which will be monitored by all devices.
    nest_recorder : list of strings, optional (default: ["spike_detector"])
        List of devices to monitor the network.
    params : list of dict, optional (default: [{}])
        List of dictionaries containing the parameters for each recorder (see 
        `NEST documentation <http://www.nest-simulator.org/quickref/#nodes>`_ 
        for details).
    network : :class:`~nngt.Network` or subclass, optional (default: "")
        Network which population will be used to differentiate inhibitory and 
        excitatory spikes.
    
    Returns
    -------
    recorders : tuple
        Tuple of the recorders' gids
    '''
    
    new_record = []
    recorders = []
    for i,rec in enumerate(nest_recorder):
        # multi/volt/conductancemeter
        if "meter" in rec:
            device = nest.Create(rec)
            recorders.append(device)
            new_record.append(params[i]["record_from"])
            nest.SetStatus(device, params[i])
            nest.Connect(device, gids)
        # event detectors
        elif "detector" in rec:
            if network is not None:
                for name, group in iter(network.population.items()):
                    device = nest.Create(rec)
                    recorders.append(device)
                    new_record.append(["spikes"])
                    nest.SetStatus(device,params[i])
                    nest.Connect(tuple(network.nest_gid[group.id_list]), device)
            else:
                device = nest.Create(rec)
                recorders.append(device)
                nest.SetStatus(device,params[i])
                nest.Connect(gids, device)
        else:
            raise InvalidArgument("Invalid recorder item in 'nest_recorder'.")
    return tuple(recorders), new_record


#-----------------------------------------------------------------------------#
# Plotting the activity
#------------------------
#

def plot_activity(gid_recorder, record, network=None, gids=None, show=True,
                  limits=None, hist=True):
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

    Returns
    -------
    fignums : list
        List of the figure numbers.
    '''
    gids = network.nest_gid if (gids is None and network is not None) else gids
    lst_rec = []
    for rec in gid_recorder:
        if isinstance(gid_recorder[0],tuple):
            lst_rec.append(rec)
        else:
            lst_rec.append((rec,))
            
    # spikes plotting
    num_group = len(network.population) if network is not None else 1
    colors = palette(np.linspace(0, 1, num_group))
    num_spike, num_detec = 0, 0
    fig_spike, fig_detec = None, None

    fignums = []
    for rec, var in zip(lst_rec, record):
        info = nest.GetStatus(rec)[0]
        if str(info["model"]) == "spike_detector":
            c = colors[num_spike]
            fig_spike = raster_plot(rec, fignum=fig_spike, color=c, show=False,
                                    limits=limits)
            num_spike += 1
            fignums.append(fig_spike)
        elif "detector" in str(info["model"]):
            c = colors[num_detec]
            fig_detec = raster_plot(rec, fignum=fig_detec, color=c, show=False,
                                    hist=hist, limits=limits)
            num_detec += 1
            fignums.append(fig_detect)
        else:
            da_time = info["events"]["times"]
            fig = plt.figure()
            if isinstance(var,list) or isinstance(var,tuple):
                axes = _set_new_plot(fig.number, len(var))[1]
                for subvar, ax in zip(var, axes):
                    da_subvar = info["events"][subvar]
                    ax.plot(da_time,da_subvar,'k')
                    ax.set_ylabel(subvar)
                    ax.set_xlabel("time")
            else:
                ax = fig.add_subplot(111)
                da_var = info["events"][var]
                ax.plot(da_time,da_var,'k')
                ax.set_ylabel(var)
                ax.set_xlabel("time")
            fignums.append(fig.number)
    if show:
        plt.show()
    return fignums

def _moving_average (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'same')
    return sma

def raster_plot(detec, limits=None, title="Spike raster", hist=True,
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
    info = nest.GetStatus(detec)[0]
    ev = info["events"]
    ts, senders = ev["times"], ev["senders"]
    num_neurons = len(np.unique(senders))

    if len(ts):
        fig = plt.figure(fignum)

        legend = info["label"]
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

def fill_between_steps(x, y1, y2=0, h_align='mid'):
    ''' Fills a hole in matplotlib: fill_between for step plots.
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

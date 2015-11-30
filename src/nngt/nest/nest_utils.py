#!/usr/bin/env python
#-*- coding:utf-8 -*-

# nest_utils.py
# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Utility functions to monitor NEST simulated activity """

import nest
import nest.raster_plot
import numpy as np
import matplotlib.pyplot as plt

from ..lib import InvalidArgument
from ..plot.custom_plt import palette
from ..plot.plt_properties import _set_new_plot


__all__ = [
            'set_noise',
            'set_poisson_input',
            'monitor_nodes',
            'plot_activity'
          ]



#-----------------------------------------------------------------------------#
# Inducing activity
#------------------------
#

def set_noise(gids, mean, std):
    bg_noise = nest.Create("noise_generator")
    nest.SetStatus(bg_noise, {"mean": mean, "std": std })
    nest.Connect(bg_noise,gids)
    
def set_poisson_input(gids, rate):
    poisson_input = nest.Create("poisson_generator")
    nest.SetStatus(poisson_input,{"rate": rate})
    nest.Connect(poisson_input, gids)


#-----------------------------------------------------------------------------#
# Monitoring the activity
#------------------------
#

def monitor_nodes(gids, nest_recorder=["spike_detector"], record=[["spikes"]],
                  accumulator=True, interval=1., to_file="", network=None):
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
    record : list of lists of strings, optional (default: (["spikes"],))
        List of the variables to record; one list per recording device.
    accumulator : bool, optional (default: True)
        Whether multi/volt/conductancemeters should sum the records of all the
        nodes they are conencted to.
    interval : float, optional (default: 1.)
        Interval of time at which multimeter-like devices sample data.
    to_file : string, optional (default: "")
        File where the recorded data should be stored; if "", the data will not
        be saved in a file.
    
    Returns
    -------
    recorders : tuple
        Tuple of the recorders' gids
    '''
    
    recorders = []
    for i,rec in enumerate(nest_recorder):
        # multi/volt/conductancemeter
        if "meter" in rec:
            device = nest.Create(rec)
            recorders.append(device)
            nest.SetStatus(device, {"withtime": True, "record_from": record[i],
                        "to_accumulator": accumulator, "interval": interval})
            nest.Connect(device, gids)
        # event detectors
        elif "detector" in rec:
            if network is not None:
                for name, group in network.population.iteritems():
                    device = nest.Create(rec)
                    recorders.append(device)
                    nest.SetStatus(device,{"label": record[i][0] + " " + name,
                                           "withtime": True, "withgid": True})
                    print(network.nest_id[group.id_list], device)
                    nest.Connect(tuple(network.nest_id[group.id_list]), device)
            else:
                device = nest.Create(rec)
                recorders.append(device)
                nest.SetStatus(device,{"label": record[i][0], "withtime": True,
                                    "withgid": True})
                nest.Connect(gids, device)
        else:
            raise InvalidArgument("Invalid recorder item in 'nest_recorder'.")
    return tuple(recorders), record


#-----------------------------------------------------------------------------#
# Plotting the activity
#------------------------
#

def plot_activity(network, gid_recorder, record, gids=None):
    '''
    Plot the monitored activity.
    
    Parameters
    ----------
    
    gid_recorder : tuple or list
        The gids of the recording devices.
    record : tuple or list
        List of the monitored variables for each device.
    '''
    gids = network.nest_id if gids is None else gids
    lst_rec = []
    for rec in gid_recorder:
        if isinstance(gid_recorder[0],tuple):
            lst_rec.append(rec)
        else:
            lst_rec.append((rec,))
            
    # spikes plotting
    num_group = len(network.population)
    colors = palette(np.linspace(0, 1, num_group))
    print(colors, num_group)
    num_spike, num_detec = 0, 0
    fig_spike, fig_detec = None, None
    
    for rec, var in zip(lst_rec, record):
        info = nest.GetStatus(rec)[0]
        if str(info["model"]) == "spike_detector":
            c = colors[num_spike]
            fig_spike = raster_plot(rec, fignum=fig_spike, color=c,
                                    show=False)
        elif "detector" in str(info["model"]):
            c = colors[num_detec]
            fig_detec = raster_plot(rec, fignum=fig_detec, color=c, show=False)
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
    plt.show()


def raster_plot(detec, xlabel=None, title="Spike raster", hist=True,
                hist_binwidth=5., color="b", fignum=None, show=True):
    """
    Generic plotting routine that constructs a raster plot along with
    an optional histogram (common part in all routines above)
    """
    ev = nest.GetStatus(detec, "events")[0]
    ts, senders = ev["times"], ev["senders"]

    fig = plt.figure(fignum)

    ylabel = "Neuron ID"
    if xlabel is None:
        xlabel = "Time (ms)"

    if hist:
        ax1, ax2 = None, None
        if len(fig.axes) == 2:
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
        else:
            ax1 = fig.add_axes([0.1, 0.3, 0.85, 0.6])
            ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.17])
        ax1.scatter(ts, senders, c=color)
        ax1.set_ylabel(ylabel)
        ax1.set_xlim([0.95*ts[0], 1.05*ts[-1]])

        t_bins = np.arange(np.amin(ts), np.amax(ts), float(hist_binwidth))
        n, bins = np.histogram(ts, bins=t_bins)
        num_neurons = len(np.unique(senders))
        heights = 1000 * n / (hist_binwidth * num_neurons)
        ax2.bar(t_bins[:-1], heights, hist_binwidth, color=color, edgecolor="k")
        lines = ax2.get_lines()
        if lines:
            ax2.bar(t_bins, heights, hist_binwidth,
                    color=c, bottom=lines[-1].get_ydata())
        yticks = [int(x) for x in np.linspace(0., int(max(heights)*1.1)+5, 4)]
        ax2.set_yticks(yticks)
        ax2.set_ylabel("Rate (Hz)")
        ax2.set_xlabel(xlabel)
        ax2.set_xlim(ax1.get_xlim())
    else:
        ax = fig.axes[0] if fig.axes else fig.subplots(111)
        ax1.scatter(ts, senders, c=color)
        ax1.set_ylabel(ylabel)
        ax1.set_xlim([0.95*ts[0], 1.05*ts[-1]])

    fig.suptitle(title)
    if show:
        plt.show()
    return fig.number

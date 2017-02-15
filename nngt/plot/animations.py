#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Animation tools """

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as anim

from nngt.lib.sorting import _sort_neurons


# ----------------- #
# Animation classes #
# ----------------- #

class Animation2d(anim.FuncAnimation):
    
    '''
    Class to plot the raster plot, firing-rate, and average trajectory in
    a 2D phase-space for a network activity.
    '''

    def __init__(self, multimeter, spike_detector, start=0., timewindow=None,
                 trace=5., x='time', y='V_m', sort_neurons=False,
                 network=None, interval=50):
        '''
        Generate a SubplotAnimation instance to plot a network activity.
        
        Parameters
        ----------
        multimeter : tuple
            NEST gid of the ``multimeter``(s) which recorded the network.
        spike_detector : tuple
            NEST gid of the ``spike_detector``(s) which recorded the network.
        timewindow : double, optional (default: None)
            Time window which will be shown for the spikes and self.rate.
        trace : double, optional (default: 5.)
            Interval of time (ms) over which the data is overlayed in red.
        x : str, optional (default: "time")
            Name of the `x`-axis variable (must be either "time" or the name
            of a NEST recordable in the `multimeter`).
        y : str, optional (default: "V_m")
            Name of the `y`-axis variable (must be either "time" or the name
            of a NEST recordable in the `multimeter`).
        '''
        import nest

        x = "times" if x == "time" else x
        y = "times" if y == "time" else y

        # get data
        data_s = nest.GetStatus(spike_detector)[0]["events"]
        spikes = np.where(data_s["times"] >= start)[0]
        data_mm = nest.GetStatus(multimeter)[0]["events"]
        self.times = data_mm["times"]

        if np.any(spikes):
            idx_start = spikes[0]
            self.spikes = data_s["times"][idx_start:]
            self.senders = data_s["senders"][idx_start:]

            self.num_neurons = np.max(self.senders) - np.min(self.senders)
            # sorting
            if sort_neurons and network is not None:
                sorted_neurons = _sort_neurons(
                    sort_neurons, self.senders, network)
                self.senders = sorted_neurons[self.senders]

            idx_start = np.where(self.times >= start)[0][0]
            self.idx_start = idx_start
            self.times = self.times[idx_start:]
            self.x = data_mm[x][idx_start:] / self.num_neurons
            self.y = data_mm[y][idx_start:] / self.num_neurons

            dt = self.times[1] - self.times[0]
            self.simtime = self.times[-1]

            # generate the spike rate
            self.firing_rate = np.zeros(len(self.times))
            for i, t in enumerate(self.times):
                gauss = np.exp(-np.square((t - self.spikes) / trace))
                self.firing_rate[i] += np.sum(gauss)
            self.firing_rate *= 1000. \
                                / (trace * np.sqrt(np.pi) * self.num_neurons)

            self.start = start
            self.trace = trace
            self.timewindow = timewindow
            if timewindow is None:
                self.timewindow = timewindow = 0.3 * self.times.max()
        else:
            raise RuntimeError("No spikes between {} and {}.".format(
                start, self.times[-1]))

        # figure/canvas: pause/resume and step by step interactions
        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('key_press_event', self.on_keyboard_press)
        fig.canvas.mpl_connect('key_release_event', self.on_keyboard_release)

        # Axes for phase-space, spikes and spike rate representations
        self.ps = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        self.spks = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=2)
        self.rate = plt.subplot2grid((2, 4), (1, 2), rowspan=1, colspan=2,
                                     sharex=self.spks)

        # Phase-space trajectory
        self.ps.set_xlabel(_convert_axis(x))
        self.ps.set_ylabel(_convert_axis(y))
        self.line_ps_ = Line2D([], [], color='black')
        self.line_ps_a = Line2D([], [], color='red', linewidth=2)
        self.line_ps_e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        self.ps.add_line(self.line_ps_)
        self.ps.add_line(self.line_ps_a)
        self.ps.add_line(self.line_ps_e)
        min_V, max_V = np.min(self.x), np.max(self.x)
        min_w, max_w = np.min(self.y), np.max(self.y)
        self.ps.set_xlim(_min_axis(min_V), _max_axis(max_V))
        self.ps.set_ylim(_min_axis(min_w), _max_axis(max_w))

        # Spikes raster plot
        self.spks.set_ylabel('Neuron')
        self.spks.set_xlabel('Time (ms)')
        self.line_spks_ = Line2D(
            [], [], ls='None', marker='o', color='black', ms=2, mew=0)
        self.line_spks_a = Line2D(
            [], [], ls='None', marker='o', color='red', ms=2, mew=0)
        self.spks.add_line(self.line_spks_)
        self.spks.add_line(self.line_spks_a)
        self.spks.set_xlim(start, min(self.simtime, timewindow + start))
        self.spks.set_ylim(0, self.num_neurons)

        # Rate plot
        self.rate.set_ylabel('Rate (Hz)')
        self.rate.set_xlabel('Time (ms)')
        self.line_rate_ = Line2D([], [], color='black')
        self.line_rate_a = Line2D([], [], color='red', linewidth=2)
        self.line_rate_e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        self.rate.add_line(self.line_rate_)
        self.rate.add_line(self.line_rate_a)
        self.rate.add_line(self.line_rate_e)
        self.rate.set_xlim(start, min(self.simtime, timewindow + start))
        fr_min, fr_max = self.firing_rate.min(), self.firing_rate.max()
        fr_min -= 0.02 * fr_max / 1.02
        self.rate.set_ylim(_min_axis(fr_min), _max_axis(fr_max))

        plt.tight_layout()

        # user interaction
        self.pause = False
        self.event = None
        self.increment = 1

        # init parent class
        super(Animation2d, self).__init__(
            fig, self._draw, self._gen_data, interval=interval, blit=True)

    #-------------------------------------------------------------------------
    # Animation instructions

    def _gen_data(self):
        i = -1
        imax = len(self.x) - 1
        while i < imax - self.increment:
            if not self.pause:
                i += self.increment
            elif self.event is not None:
                if self.event.key in ('right', 'n'):
                    i += self.increment
                elif self.event.key in ('left', 'p'):
                    i -= self.increment
                if self.event.key in ('n', 'p'):
                    self.event = None
            yield i

    def _draw(self, framedata):
        i = framedata
        head = i - 1
        head_slice = ((self.times > self.times[i] - self.trace)
                      & (self.times < self.times[i]))
        spike_slice = ((self.spikes > self.times[i] - self.trace)
                       & (self.spikes <= self.times[i]))
        spike_cum = self.spikes < self.times[i]

        self.line_ps_.set_data(self.x[:i], self.y[:i])
        self.line_ps_a.set_data(self.x[head_slice], self.y[head_slice])
        self.line_ps_e.set_data(self.x[head], self.y[head])
        
        self.line_spks_.set_data(
            self.spikes[spike_cum], self.senders[spike_cum])
        if np.any(spike_slice):
            self.line_spks_a.set_data(
                self.spikes[spike_slice], self.senders[spike_slice])
        else:
            self.line_spks_a.set_data([], [])
        # set axis limits
        # 1. check user-defined
        current_window = np.diff(self.spks.get_xlim())
        default_window = (np.isclose(current_window, self.timewindow)
            or np.isclose(current_window, self.simtime - self.start))[0]
        if default_window:
            if self.times[i] > self.start + self.timewindow:
                self.spks.set_xlim(
                    self.times[i] - self.timewindow, self.times[i])
                self.rate.set_xlim(
                    self.times[i] - self.timewindow, self.times[i])
            else:
                self.spks.set_xlim(
                    self.start,
                    min(self.simtime, self.timewindow + self.start))
                self.rate.set_xlim(
                    self.start,
                    min(self.simtime, self.timewindow + self.start))

        self.line_rate_.set_data(self.times[:i], self.firing_rate[:i])
        self.line_rate_a.set_data(
            self.times[head_slice], self.firing_rate[head_slice])
        self.line_rate_e.set_data(self.times[head], self.firing_rate[head])

        return [self.line_ps_, self.line_ps_a, self.line_ps_e, self.line_spks_,
                self.line_spks_a, self.line_rate_, self.line_rate_a,
                self.line_rate_e]

    def _init_draw(self):
        lines = [self.line_ps_, self.line_ps_a, self.line_ps_e,
                 self.line_spks_, self.line_spks_a,
                 self.line_rate_, self.line_rate_a, self.line_rate_e]
        for l in lines:
            l.set_data([], [])

    #-------------------------------------------------------------------------
    # User interaction

    def on_click(self, event):
        #~ self.pause ^= True
        pass

    def on_keyboard_press(self, kb_event):
        if kb_event.key == ' ':
            self.pause ^= True
        elif kb_event.key == 'F':
            self.increment *= 2
        elif kb_event.key == 'B':
            self.increment = max(1, int(self.increment / 2))
        self.event = kb_event

    def on_keyboard_release(self, kb_event):
        self.event = None


# ----- #
# Tools #
# ----- #

def _max_axis(value):
    if np.sign(value) > 0.:
        return 1.02*value
    else:
        return 0.98*value


def _min_axis(value):
    if np.sign(value) < 0.:
        return 1.02*value
    else:
        return 0.98*value


def _convert_axis(axis_name):
    lowercase = axis_name.lower()
    if lowercase == "times":
        return "Time (ms)"
    new_name = "$"
    i = axis_name.find("_")
    if i != -1:
        start = lowercase[:i]
        if start in ("tau", "alpha", "beta", "gamma", "delta"):
            new_name += "\\" + axis_name[:i] + "_{" + axis_name[i+1:] + "}$"
        elif start in ("v", "e"):
            new_name += axis_name[:i] + "_{" + axis_name[i+1:] + "}$ (mV)"
        elif start == "i":
            new_name += axis_name[:i] + "_{" + axis_name[i+1:] + "}$ (pA)"
        else:
            new_name += axis_name[:i] + "_{" + axis_name[i+1:] + "}$"
    else:
        if lowercase in ("tau", "alpha", "beta", "gamma", "delta"):
            new_name += "\\" + lowercase + "$"
        elif lowercase == "w":
            new_name = "$w$ (pA)"
        else:
            new_name += lowercase + "$"
    return new_name

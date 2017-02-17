#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Animation tools """

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as anim

from nngt.lib.sorting import _sort_neurons


# ----------------- #
# Animation classes #
# ----------------- #

class _SpikeAnimator:
    
    '''
    Generic class to plot raster plot and firing-rate in time for a given
    network.
    '''
    
    steps = [
        1, 5, 10, 20, 25, 50, 100, 200, 250,
        500, 1000, 2000, 2500, 5000, 10000
    ]

    def __init__(self, spike_detector=None, sort_neurons=False,
                 network=None, grid=(2, 4), pos_raster=(0, 2),
                 span_raster=(1, 2), pos_rate=(1, 2),
                 span_rate=(1, 2), make_rate=True, **kwargs):
        '''
        Generate a SubplotAnimation instance to plot a network activity.
        
        Parameters
        ----------
        spike_detector : tuple
            NEST gid of the ``spike_detector``(s) which recorded the network.
        times : array-like, optional (default: None)
            List of times to run the animation.

        Note
        ----
        Calling class is supposed to have defined `self.times`, `self.start`,
        `self.duration`, `self.trace`, and `self.timewindow`.
        '''
        import nest
        
        # organization
        self.grid = grid
        self.has_rate = make_rate

        # get data
        data_s = nest.GetStatus(spike_detector)[0]["events"]
        spikes = np.where(data_s["times"] >= self.times[0])[0]
        print(self.times[0], spikes[0], self.times[-1])

        if np.any(spikes):
            idx_start = spikes[0]
            self.spikes = data_s["times"][idx_start:]
            self.senders = data_s["senders"][idx_start:]

            self.num_neurons = np.max(self.senders) - np.min(self.senders)
            # sorting
            if sort_neurons:
                if network is not None:
                    sorted_neurons = _sort_neurons(
                        sort_neurons, self.senders, network)
                    self.senders = sorted_neurons[self.senders]
                else:
                    warnings.warn("Could not sort neurons because no " \
                                  + "`network` was provided.")

            dt = self.times[1] - self.times[0]
            self.simtime = self.times[-1] - self.times[0]

            # generate the spike-rate
            if make_rate:
                self.firing_rate = np.zeros(len(self.times))
                for i, t in enumerate(self.times):
                    gauss = np.exp(-np.square((t - self.spikes) / self.trace))
                    self.firing_rate[i] += np.sum(gauss)
                self.firing_rate *= 1000. / (self.trace * np.sqrt(np.pi) \
                                             * self.num_neurons)
        else:
            raise RuntimeError("No spikes between {} and {}.".format(
                start, self.times[-1]))

        # figure/canvas: pause/resume and step by step interactions
        self.fig = plt.figure()
        self.pause = False
        self.event = None
        self.increment = 1
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_keyboard_press)
        self.fig.canvas.mpl_connect(
            'key_release_event', self.on_keyboard_release)

        # Axes for spikes and spike-rate/other representations
        self.spks = plt.subplot2grid(
            grid, pos_raster, rowspan=span_raster[0], colspan=span_raster[1])
        self.second = plt.subplot2grid(
            grid, pos_rate, rowspan=span_rate[0], colspan=span_rate[1],
            sharex=self.spks)
        
        # lines
        self.line_spks_ = Line2D(
            [], [], ls='None', marker='o', color='black', ms=2, mew=0)
        self.line_spks_a = Line2D(
            [], [], ls='None', marker='o', color='red', ms=2, mew=0)
        self.line_second_ = Line2D([], [], color='black')
        self.line_second_a = Line2D([], [], color='red', linewidth=2)
        self.line_second_e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')

        # Spikes raster plot
        xlim = (self.start, min(self.simtime, self.timewindow + self.start))
        ylim = (0, self.num_neurons)
        self.lines_raster = [self.line_spks_, self.line_spks_a]
        self.set_axis(self.spks, xlabel='Time (ms)', ylabel='Neuron',
            lines=self.lines_raster, xlim=xlim, ylim=ylim, set_xticks=True)
        self.lines_second = [
            self.line_second_, self.line_second_a, self.line_second_e]

        # Rate plot
        if make_rate:
            self.set_axis(
                self.second, xlabel='Time (ms)', ylabel='Rate (Hz)',
                lines=self.lines_second, ydata=self.firing_rate, xlim=xlim)

    #-------------------------------------------------------------------------
    # Axis definition
    
    def set_axis(self, axis, xlabel, ylabel, lines, xdata=None, ydata=None,
                 **kwargs):
        '''
        Setup an axis.
        
        Parameters
        ----------
        axis : :class:`matplotlib.axes.Axes` object
        xlabel : str
        ylabel : str
        lines : list of :class:`matplotlib.lines.Line2D` objects
        xdata : 1D array-like, optional (default: None)
        ydata : 1D array-like, optional (default: None)
        **kwargs : dict, optional (default: {})
            Optional arguments ("xlim" or "ylim", 2-tuples; "set_xticks",
            bool).
        '''
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        for line2d in lines:
            axis.add_line(line2d)
        if 'xlim' in kwargs:
            axis.set_xlim(*kwargs['xlim'])
        else:
            xmin, xmax = np.min(xdata), np.max(xdata)
            axis.set_xlim(_min_axis(xmin, xmax), _max_axis(xmax, xmin))
        if 'ylim' in kwargs:
            axis.set_ylim(*kwargs['ylim'])
        else:
            ymin, ymax = np.min(ydata), np.max(ydata)
            axis.set_ylim(_min_axis(ymin, ymax), _max_axis(ymax, ymin))
        if kwargs.get('set_xticks', False):
            self._make_ticks(self.timewindow)
    
    def _draw(self, i, head, head_slice, spike_cum, spike_slice):
        self.line_spks_.set_data(
            self.spikes[spike_cum], self.senders[spike_cum])
        if np.any(spike_slice):
            self.line_spks_a.set_data(
                self.spikes[spike_slice], self.senders[spike_slice])
        else:
            self.line_spks_a.set_data([], [])
        if self.has_rate:
            self.line_second_.set_data(self.times[:i], self.firing_rate[:i])
            self.line_second_a.set_data(
                self.times[head_slice], self.firing_rate[head_slice])
            self.line_second_e.set_data(
                self.times[head], self.firing_rate[head])
    
    def _make_ticks(self, timewindow):
        target_num_ticks = np.ceil(self.duration / timewindow * 5)
        target_step = self.duration / target_num_ticks
        idx_step = np.abs(self.steps-target_step).argmin()
        step = self.steps[idx_step]
        num_steps = int(self.duration / step) + 2
        self.xticks = [self.start + i*step for i in range(num_steps)]
        self.xlabels = [str(i) for i in self.xticks]

    #-------------------------------------------------------------------------
    # User interaction

    def on_click(self, event):
        if event.button == '2':
            self.pause ^= True

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


class Animation2d(_SpikeAnimator, anim.FuncAnimation):
    
    '''
    Class to plot the raster plot, firing-rate, and average trajectory in
    a 2D phase-space for a network activity.
    '''

    def __init__(self, multimeter, spike_detector, start=0., timewindow=None,
                 trace=5., x='time', y='V_m', sort_neurons=False,
                 network=None, interval=50, **kwargs):
        '''
        Generate a SubplotAnimation instance to plot a network activity.
        
        Parameters
        ----------
        multimeter : tuple
            NEST gid of the ``multimeter``(s) which recorded the network.
        spike_detector : tuple
            NEST gid of the ``spike_detector``(s) which recorded the network.
        timewindow : double, optional (default: None)
            Time window which will be shown for the spikes and self.second.
        trace : double, optional (default: 5.)
            Interval of time (ms) over which the data is overlayed in red.
        x : str, optional (default: "time")
            Name of the `x`-axis variable (must be either "time" or the name
            of a NEST recordable in the `multimeter`).
        y : str, optional (default: "V_m")
            Name of the `y`-axis variable (must be either "time" or the name
            of a NEST recordable in the `multimeter`).
        **kwargs : dict, optional (default: {})
            Optional arguments such as 'make_rate'.
        '''
        import nest

        x = "times" if x == "time" else x
        y = "times" if y == "time" else y

        # get data
        data_s = nest.GetStatus(spike_detector)[0]["events"]
        spikes = np.where(data_s["times"] >= start)[0]
        data_mm = nest.GetStatus(multimeter)[0]["events"]
        self.times = data_mm["times"]

        idx_start = np.where(self.times >= start)[0][0]
        self.idx_start = idx_start
        self.times = self.times[idx_start:]

        dt = self.times[1] - self.times[0]
        self.simtime = self.times[-1]
        self.start = start
        self.duration = self.simtime - start
        self.trace = trace
        if timewindow is None:
            self.timewindow = 0.3 * self.simtime
        else:
            self.timewindow = min(timewindow, self.simtime - self.start)

        # init _SpikeAnimator parent class (create figure and right axes)
        make_rate = kwargs.get('make_rate', True)
        super(Animation2d, self).__init__(
            spike_detector, sort_neurons=sort_neurons, network=network,
            make_rate=make_rate)

        # Data and axis for phase-space
        self.x = data_mm[x][idx_start:] / self.num_neurons
        self.y = data_mm[y][idx_start:] / self.num_neurons
        
        self.ps = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        self.line_ps_ = Line2D([], [], color='black')
        self.line_ps_a = Line2D([], [], color='red', linewidth=2)
        self.line_ps_e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        lines = [self.line_ps_, self.line_ps_a, self.line_ps_e]
        self.set_axis(
            self.ps, xlabel=_convert_axis(x), ylabel=_convert_axis(y),
            lines=lines, xdata=self.x, ydata=self.y)

        plt.tight_layout()
        
        anim.FuncAnimation.__init__(self, self.fig, self._draw, self._gen_data,
                                    interval=interval, blit=True)

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
        # restore and init all
        if i == 0:
            self.fig.canvas.restore_region(self.bg)
            self.spks.draw_artist(self.spks.get_xaxis())
            self.second.draw_artist(self.spks.get_xaxis())
            self.fig.canvas.blit(self.spks.clipbox)

        head = i - 1
        head_slice = ((self.times > self.times[i] - self.trace)
                      & (self.times < self.times[i]))
        spike_slice = ((self.spikes > self.times[i] - self.trace)
                       & (self.spikes <= self.times[i]))
        spike_cum = self.spikes < self.times[i]

        self.line_ps_.set_data(self.x[:i], self.y[:i])
        self.line_ps_a.set_data(self.x[head_slice], self.y[head_slice])
        self.line_ps_e.set_data(self.x[head], self.y[head])
        
        super(Animation2d, self)._draw(
            i, head, head_slice, spike_cum, spike_slice)
        
        # set axis limits: 1. check user-defined
        current_window = np.diff(self.spks.get_xlim())
        default_window = (np.isclose(current_window, self.timewindow)
            or np.isclose(current_window, self.simtime - self.start))[0]
        # 3. change if necessary
        if default_window:
            self.fig.canvas.restore_region(self.bg)
            #~ self.fig.canvas.restore_region(self.bg_second)
            xlims = self.spks.get_xlim()
            if self.times[i] >= xlims[1]:
                #~ print(self.times[i] - self.timewindow, self.times[i])
                self.spks.set_xlim(
                    self.times[i] - self.timewindow, self.times[i])
                self.second.set_xlim(
                    self.times[i] - self.timewindow, self.times[i])
                self.spks.draw_artist(self.spks.get_xaxis())
                self.second.draw_artist(self.second.get_xaxis())
                #~ for a in self.spks.get_xaxis().get_majorticklabels():
                    #~ self.spks.draw_artist(a)
                self.fig.canvas.blit(self.spks.clipbox)
                self.fig.canvas.blit(self.second.clipbox)
                #~ self.spks.draw_artist(self.spks.get_xaxis().get_label())
                #~ self.fig.canvas.blit(self.spks.clipbox)
                #~ self.spks.set_xticks(xticks)
                #~ self.spks.set_xticklabels([str(i) for i in xticks])
                #~ self.second.set_xticks(xticks)
                #~ self.second.set_xlim(
                    #~ self.times[i] - self.timewindow, self.times[i])
            elif self.times[i] <= xlims[0]:
                self.spks.set_xlim(self.start, self.timewindow + self.start)
                #~ self.second.set_xlim(self.start, self.timewindow + self.start)

        return [self.line_ps_, self.line_ps_a, self.line_ps_e, self.line_spks_,
                self.line_spks_a, self.line_second_, self.line_second_a,
                self.line_second_e]

    def _init_draw(self):
        '''
        Remove ticks from spks/second axes, save background,
        then restore state to allow for moveable axes and labels.
        '''
        xlim = self.spks.get_xlim()
        xlabel = self.spks.get_xlabel()
        # remove
        self.spks.set_xticks([])
        self.spks.set_xticklabels([])
        self.spks.set_xlabel("")
        self.second.set_xticks([])
        self.second.set_xticklabels([])
        self.second.set_xlabel("")
        # background
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # restore
        self.spks.set_xticks(self.xticks)
        self.spks.set_xticklabels(self.xlabels)
        self.spks.set_xlim(*xlim)
        self.spks.set_xlabel(xlabel)
        self.second.set_xticks(self.xticks)
        self.second.set_xticklabels(self.xlabels)
        self.second.set_xlim(*xlim)
        self.second.set_xlabel(xlabel)
        # initialize empty lines
        lines = [self.line_ps_, self.line_ps_a, self.line_ps_e,
                 self.line_spks_, self.line_spks_a,
                 self.line_second_, self.line_second_a, self.line_second_e]
        for l in lines:
            l.set_data([], [])
        self.spks.set_animated(True)
        #~ self.spks.get_xaxis().set_animated(True)
        #~ self.spks.draw_artist(self.spks.get_xaxis())
        self.spks.draw_artist(self.spks)
        self.second.set_animated(True)
        #~ self.second.get_xaxis().set_animated(True)
        #~ self.second.draw_artist(self.second.get_xaxis())
        self.second.draw_artist(self.second)


# ----- #
# Tools #
# ----- #

def _max_axis(value, min_val=0.):
    if np.isclose(value, 0.):
        return -0.02*min_val
    elif np.sign(value) > 0.:
        return 1.02*value
    else:
        return 0.98*value


def _min_axis(value, max_val=0.):
    if np.isclose(value, 0.):
        return -0.02*max_val
    elif np.sign(value) < 0.:
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

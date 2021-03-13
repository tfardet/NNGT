#-*- coding:utf-8 -*-
#
# plot/animations.py
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

""" Animation tools """

import warnings
import weakref
import subprocess

import numpy as np

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.animation as anim

from nngt.lib import InvalidArgument
from nngt.lib.sorting import _sort_neurons
from nngt.analysis import total_firing_rate
from .plt_networks import draw_network


# ----------------- #
# Animation classes #
# ----------------- #

class _SpikeAnimator(anim.TimedAnimation):

    '''
    Generic class to plot raster plot and firing-rate in time for a given
    network.

    .. warning::
        This class is not supposed to be instantiated directly, but only
        through Animation2d or AnimationNetwork.
    '''

    steps = [
        1, 5, 10, 20, 25, 50, 100, 200, 250, 500,
        1000, 2000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 250000
    ]

    def __init__(self, source, sort_neurons=None,
                 network=None, grid=(2, 4), pos_raster=(0, 2),
                 span_raster=(1, 2), pos_rate=(1, 2),
                 span_rate=(1, 2), make_rate=True, **kwargs):
        '''
        Generate a SubplotAnimation instance to plot a network activity.

        Parameters
        ----------
        source : NEST gid tuple or str
            NEST gid of the `spike_detector`(s) which recorded the network or
            path to a file containing the recorded spikes.

        Note
        ----
        Calling class is supposed to have defined `self.times`, `self.start`,
        `self.duration`, `self.trace`, and `self.timewindow`.
        '''
        import matplotlib.pyplot as plt
        import nest
        from nngt.simulation.nest_activity import _get_data

        # organization
        self.grid = grid
        self.has_rate = make_rate

        # get data
        data_s = _get_data(source)
        spikes = np.where(data_s[:, 1] >= self.times[0])[0]

        if np.any(spikes):
            idx_start = spikes[0]
            self.spikes = data_s[:, 1][idx_start:]
            self.senders = data_s[:, 0][idx_start:].astype(int)
            self._ymax = np.max(self.senders)
            self._ymin = np.min(self.senders)

            if network is None:
                self.num_neurons = int(self._ymax - self._ymin)
            else:
                self.num_neurons = network.node_nb()
            # sorting
            if sort_neurons is not None:
                if network is not None:
                    sorted_neurons = _sort_neurons(
                        sort_neurons, self.senders, network, data=data_s)
                    self.senders = sorted_neurons[self.senders]
                else:
                    warnings.warn("Could not sort neurons because no " \
                                  + "`network` was provided.")

            dt = self.times[1] - self.times[0]
            self.simtime = self.times[-1] - self.times[0]
            self.dt = dt

            # generate the spike-rate
            if make_rate:
                self.firing_rate, _ = total_firing_rate(
                    network, data=data_s, resolution=self.times)
        else:
            raise RuntimeError("No spikes between {} and {}.".format(
                self.start, self.times[-1]))

        # figure/canvas: pause/resume and step by step interactions
        self.fig = plt.figure(
            figsize=kwargs.get("figsize", (8, 6)), dpi=kwargs.get("dpi", 75))
        self.pause = False
        self.pause_after = False
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
        kw_args = {}
        if self.timewindow != self.duration:
            kw_args['xlim'] = (self.start,
                min(self.simtime, self.timewindow + self.start))
        ylim = (self._ymin, self._ymax)
        self.lines_raster = [self.line_spks_, self.line_spks_a]
        self.set_axis(self.spks, xlabel='Time (ms)', ylabel='Neuron',
            lines=self.lines_raster, ylim=ylim, set_xticks=True, **kw_args)
        self.lines_second = [
            self.line_second_, self.line_second_a, self.line_second_e]

        # Rate plot
        if make_rate:
            self.set_axis(
                self.second, xlabel='Time (ms)', ylabel='Rate (Hz)',
                lines=self.lines_second, ydata=self.firing_rate, **kw_args)

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
        if kwargs.get('set_xticks', False):
            self._make_ticks(self.timewindow)
        for line2d in lines:
            axis.add_line(line2d)
        if 'xlim' in kwargs:
            axis.set_xlim(*kwargs['xlim'])
        else:
            xmin, xmax = self.xticks[0], self.xticks[-1]
            axis.set_xlim(_min_axis(xmin, xmax), _max_axis(xmax, xmin))
        if 'ylim' in kwargs:
            axis.set_ylim(*kwargs['ylim'])
        else:
            ymin, ymax = np.min(ydata), np.max(ydata)
            axis.set_ylim(_min_axis(ymin, ymax), _max_axis(ymax, ymin))

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

        # set axis limits: 1. check user-defined
        current_window = np.diff(self.spks.get_xlim())
        default_window = (np.isclose(current_window, self.timewindow)
            or np.isclose(current_window, self.simtime - self.start))[0]
        # 3. change if necessary
        if default_window:
            xlims = self.spks.get_xlim()
            if self.times[i] >= xlims[1]:
                self.spks.set_xlim(
                    self.times[i] - self.timewindow, self.times[i])
                self.second.set_xlim(
                    self.times[i] - self.timewindow, self.times[i])
            elif self.times[i] <= xlims[0]:
                self.spks.set_xlim(self.start, self.timewindow + self.start)

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
            if self.pause:
                self.pause = False
                self.event_source.start()
            else:
                self.pause = True
                self.event_source.stop()

    def on_keyboard_press(self, kb_event):
        if kb_event.key == ' ':
            if self.pause:
                self.pause = False
                self.event_source.start()
            else:
                self.pause = True
                self.event_source.stop()
        else:
            if kb_event.key in ('B', 'F', 'N', 'P'):
                if self.pause:
                    self.pause = False
                    self.pause_after = True    # stop at next iteration
                    self.event_source.start()  # restart temporarily
                if kb_event.key == 'F':
                    self.increment *= 2
                elif kb_event.key == 'B':
                    self.increment = max(1, int(self.increment / 2))
        self.event = kb_event

    def on_keyboard_release(self, kb_event):
        if kb_event.key in (' ', 'B', 'F', 'N', 'P'):
            if self.pause_after:
                self.pause = True
                self.pause_after = False
                self.event_source.stop()  # pause again
            self.event = None

    def save_movie(self, filename, fps=30, video_encoder='html5', codec="h264",
                   bitrate=-1, start=None, stop=None, interval=None,
                   num_frames=None, metadata=None):
        '''
        Save the animation to a movie file.

        Parameters
        ----------
        filename : :obj:`str`
            Name of the file where the movie will be saved.
        fps : int, optional (default: 30)
            Frame per second.
        video_encoder : :obj:`str`, optional (default 'html5')
            Movie encoding format; either 'ffmpeg', 'html5', or 'imagemagick'.
        codec : :obj:`str`, optional (default: "h264")
            Codec to use for writing movie; if None, default `animation.codec`
            from `matplotlib` will be used.
        bitrate : int, optional (default: -1)
            Controls size/quality tradeoff for movie. Default (-1) lets utility
            auto-determine.
        start : float, optional (default: initial time)
            Start time, corresponding to the first spike time that will appear
            on the video.
        stop : float, optional (default: final time)
            Stop time, corresponding to the last spike time that will appear
            on the video.
        interval : int, optional (default: None)
            Timestep increment for each new frame. Default saves all
            timesteps (often heavy). E.g. setting `interval` to 10 will make
            the file 10 times lighter.
        num_frames : int, optional (default: None)
            Total number of frames that should be saved.
        metadata : :obj:`dict`, optional (default: None)
            Metadata for the video (e.g. 'title', 'artist', 'comment',
            'copyright')

        Notes
        -----
        * ``ffmpeg`` is required for 'ffmpeg' and 'html5' encoders.
          To get available formats, type ``ffmpeg -formats`` in a terminal;
          type ``ffmpeg -codecs | grep EV`` for available codecs.
        * Imagemagick is required for 'imagemagick' encoder.
        '''
        if interval is not None and num_frames is not None:
            raise InvalidArgument("Incompatible arguments `interval` and "
                                  "`num_frames` provided. Choose one.")
        elif interval is None and num_frames is None:
            self.increment  = 1
            self.save_count = self.num_frames
        elif interval is None:
            self.increment = max(1, int(self.num_frames / num_frames))
            self.save_count = num_frames
        else:
            self.increment = interval
            self.save_count = int(self.num_frames / interval)
        start_frame = 0
        stop_frame  = self.save_count
        if start is not None:
            start_frame = int(start / self.dt / self.increment) + 1
            self.spks.set_xlim(left=start)
            self.second.set_xlim(left=start)
        if stop is not None:
            stop_frame = int(stop / self.dt / self.increment) + 1
            self.spks.set_xlim(right=stop)
            self.second.set_xlim(right=stop)
        _save_movie(
            self, filename, fps, video_encoder, codec, bitrate, metadata,
            self.fig.dpi, start_frame, stop_frame)


class Animation2d(_SpikeAnimator, anim.FuncAnimation):

    '''
    Class to plot the raster plot, firing-rate, and average trajectory in
    a 2D phase-space for a network activity.
    '''

    def __init__(self, source, multimeter, start=0., timewindow=None,
                 trace=5., x='time', y='V_m', sort_neurons=None,
                 network=None, interval=50, vector_field=False, **kwargs):
        '''
        Generate a SubplotAnimation instance to plot a network activity.

        Parameters
        ----------
        source : tuple
            NEST gid of the ``spike_detector``(s) which recorded the network.
        multimeter : tuple
            NEST gid of the ``multimeter``(s) which recorded the network.
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
        vector_field : bool, optional (default: False)
            Whether the :math:`\dot{x}` and :math:`\dot{y}` arrows should be
            added to phase space. Requires additional 'dotx' and 'doty'
            arguments which are user defined functions to compute the
            derivatives of `x` and `x` in time. These functions take 3
            parameters, which are `x`, `y`, and `time_dependent`, where the
            last parameter is a list of doubles associated to recordables
            from the neuron model (see example for details). These recordables
            must be declared in a `time_dependent` parameter.
        sort_neurons : str or list, optional (default: None)
            Sort neurons using a topological property ("in-degree",
            "out-degree", "total-degree" or "betweenness"), an activity-related
            property ("firing_rate", 'B2') or a user-defined list of sorted
            neuron ids. Sorting is performed by increasing value of the
            `sort_neurons` property from bottom to top inside each group.
        **kwargs : dict, optional (default: {})
            Optional arguments such as 'make_rate', 'num_xarrows',
            'num_yarrows', 'dotx', 'doty', 'time_dependent', 'recordables',
            'arrow_scale'.
        '''
        import matplotlib.pyplot as plt
        import nest

        x = "times" if x == "time" else x
        y = "times" if y == "time" else y

        # get data
        data_mm = nest.GetStatus(multimeter)[0]["events"]
        self.times = data_mm["times"]

        self.num_frames = len(self.times)

        idx_start = np.where(self.times >= start)[0][0]
        self.idx_start = idx_start
        self.times = self.times[idx_start:]

        dt = self.times[1] - self.times[0]
        self.simtime = self.times[-1]
        self.start = start
        self.duration = self.simtime - start
        self.trace = trace
        self.vector_field = vector_field
        if timewindow is None:
            self.timewindow = self.duration
        else:
            self.timewindow = min(timewindow, self.duration)

        # init _SpikeAnimator parent class (create figure and right axes)
        if 'make_rate' not in kwargs:
            kwargs['make_rate'] = True
        super(Animation2d, self).__init__(
            source, sort_neurons=sort_neurons, network=network,
            **kwargs)

        # Data and axis for phase-space
        self.x = data_mm[x][idx_start:] / self.num_neurons
        self.y = data_mm[y][idx_start:] / self.num_neurons

        self.ps = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        self.ps.grid(False)

        # lines
        self.line_ps_ = Line2D([], [], color='black')
        self.line_ps_a = Line2D([], [], color='red', linewidth=2)
        self.line_ps_e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        lines = [self.line_ps_, self.line_ps_a, self.line_ps_e]
        xlim = (_min_axis(self.x.min()), _max_axis(self.x.max()))
        self.set_axis(
            self.ps, xlabel=_convert_axis(x), ylabel=_convert_axis(y),
            lines=lines, xdata=self.x, ydata=self.y, xlim=xlim)

        # For quiver plot (vector field)
        nx = kwargs.get('num_xarrows', 20)
        ny = kwargs.get('num_yarrows', 20)
        scale = kwargs.get('arrow_scale', 30.)
        if self.vector_field:
            time_dependent_rec = kwargs.get('time_dependent', [])
            self.time_dependent = [
                data_mm[key][idx_start:] / self.num_neurons
                for key in time_dependent_rec
            ]
            self.dotx, self.doty = kwargs['dotx'], kwargs['doty']
            xx = np.repeat(np.linspace(xlim[0], xlim[1], nx), ny)
            yy = np.tile(np.linspace(self.y.min(), self.y.max(), ny), nx)
            self.q = self.ps.quiver(xx, yy, [], [], scale=scale, color='grey')

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
                if self.event is not None:
                    if self.event.key == 'N':
                        i += self.increment
                    elif self.event.key == 'P':
                        i -= self.increment
                    self.event = None
                else:
                    i += self.increment
            yield i

    def _draw(self, framedata):
        i = int(framedata)

        head = i - 1
        head_slice = ((self.times > (self.times[i] - self.trace))
                      & (self.times < self.times[i]))
        spike_slice = ((self.spikes > (self.times[i] - self.trace))
                       & (self.spikes <= self.times[i]))
        spike_cum = self.spikes < self.times[i]

        lines = []
        if self.vector_field:
            time_dep = [arr[i] for arr in self.time_dependent]
            u = self.dotx(self.q.X, self.q.Y, time_dep)
            v = self.doty(self.q.X, self.q.Y, time_dep)
            self.q.set_UVC(u, v)
            lines.append(self.q)

        self.line_ps_.set_data(self.x[:i], self.y[:i])
        self.line_ps_a.set_data(self.x[head_slice], self.y[head_slice])
        self.line_ps_e.set_data(self.x[i], self.y[i])

        lines.extend([self.line_ps_, self.line_ps_a, self.line_ps_e,
             self.line_spks_, self.line_spks_a, self.line_second_,
             self.line_second_a, self.line_second_e])

        super(Animation2d, self)._draw(
            i, head, head_slice, spike_cum, spike_slice)

        return lines

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
        if self.vector_field:
            self.q.set_UVC([], [])
        # initialize empty lines
        lines = [self.line_ps_, self.line_ps_a, self.line_ps_e,
                 self.line_spks_, self.line_spks_a,
                 self.line_second_, self.line_second_a, self.line_second_e]
        for l in lines:
            l.set_data([], [])


class AnimationNetwork(_SpikeAnimator, anim.FuncAnimation):

    '''
    Class to plot the raster plot, firing-rate, and space-embedded spiking
    activity (neurons on the graph representation flash when spiking) in time.
    '''

    def __init__(self, source, network, resolution=1., start=0.,
                 timewindow=None, trace=5., show_spikes=False,
                 sort_neurons=None, decimate_connections=False,
                 interval=50, repeat=True, resting_size=None, active_size=None,
                 **kwargs):
        '''
        Generate a SubplotAnimation instance to plot a network activity.

        Parameters
        ----------
        source : tuple
            NEST gid of the ``spike_detector``(s) which recorded the network.
        network : :class:`~nngt.SpatialNetwork`
            Network embedded in space to plot the actvity of the neurons in
            space.
        resolution : double, optional (default: None)
            Time resolution of the animation.
        timewindow : double, optional (default: None)
            Time window which will be shown for the spikes and self.second.
        trace : double, optional (default: 5.)
            Interval of time (ms) over which the data is overlayed in red.
        show_spikes : bool, optional (default: True)
            Whether a spike trajectory should be displayed on the network.
        sort_neurons : str or list, optional (default: None)
            Sort neurons using a topological property ("in-degree",
            "out-degree", "total-degree" or "betweenness"), an activity-related
            property ("firing_rate", 'B2') or a user-defined list of sorted
            neuron ids. Sorting is performed by increasing value of the
            `sort_neurons` property from bottom to top inside each group.
        **kwargs : dict, optional (default: {})
            Optional arguments such as 'make_rate', or all arguments for the
            :func:`nngt.plot.draw_network`.
        '''
        import matplotlib.pyplot as plt
        import nest
        from nngt.simulation.nest_activity import _get_data

        self.network = weakref.ref(network)
        self.simtime = _get_data(source)[-1, 1]
        self.times = np.arange(start, self.simtime + resolution, resolution)

        self.num_frames = len(self.times)
        self.start = start
        self.duration = self.simtime - start
        self.trace = trace
        self.show_spikes = show_spikes
        if timewindow is None:
            self.timewindow = self.duration
        else:
            self.timewindow = min(timewindow, self.duration)

        # init _SpikeAnimator parent class (create figure and right axes)
        #~ self.decim_conn = 1 if decimate is not None else decimate
        self.kwargs = kwargs
        cs = kwargs.get('chunksize', 10000)
        mpl.rcParams['agg.path.chunksize'] = cs
        if 'make_rate' not in kwargs:
            kwargs['make_rate'] = True
        super(AnimationNetwork, self).__init__(
            source, sort_neurons=sort_neurons, network=network,
            **kwargs)

        self.env = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)

        # Data and axis for network representation
        bbox = self.env.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted())
        area_px = bbox.width * bbox.height * self.fig.dpi**2
        # neuron size
        n_size = (resting_size if resting_size is not None
                  else max(2, 0.5*np.sqrt(area_px/self.num_neurons)))
        if active_size is None:
            active_size = n_size + 2
        pos = network.get_positions()  # positions of the neurons
        self.x = pos[:, 0]
        self.y = pos[:, 1]

        # neurons
        self.line_neurons = Line2D(
            [], [], ls='None', marker='o', color='black', ms=n_size, mew=0)
        self.line_neurons_a = Line2D(
            [], [], ls='None', marker='o', color='red', ms=active_size, mew=0)
        self.lines_env = [self.line_neurons, self.line_neurons_a]
        xlim = (_min_axis(self.x.min()), _max_axis(self.x.max()))
        self.set_axis(self.env, xlabel='Network', ylabel='',
            lines=self.lines_env, xdata=self.x, ydata=self.y, xlim=xlim)
        # spike trajectory
        if show_spikes:
            self.line_st_a = Line2D([], [], color='red', linewidth=1)
            self.line_st_e = Line2D(
                [], [], color='red', marker='d', ms=2, markeredgecolor='r')
            self.lines_env.extend((self.line_st_a, self.line_st_e))
        # remove the axes and grid from env
        self.env.set_xticks([])
        self.env.set_yticks([])
        self.env.set_xticklabels([])
        self.env.set_yticklabels([])
        self.env.grid(None)

        plt.tight_layout()

        anim.FuncAnimation.__init__(
            self, self.fig, self._draw, self._gen_data, repeat=repeat,
            interval=interval, blit=True)

    #-------------------------------------------------------------------------
    # Animation instructions

    def _gen_data(self):
        i = -1
        imax = len(self.times) - 1
        while i < imax - self.increment:
            if not self.pause:
                if self.event is not None:
                    if self.event.key == 'N':
                        i += self.increment
                    elif self.event.key == 'P':
                        i -= self.increment
                else:
                    i += self.increment
            yield i

    def _draw(self, framedata):
        i = int(framedata)
        if i == 0:  # initialize neurons and connections
            self.line_neurons.set_data(self.x, self.y)
            #~ self.line_connections.set_data(self.x_conn, self.y_conn)

        head = i - 1
        head_slice = ((self.times > self.times[i] - self.trace)
                      & (self.times < self.times[i]))
        spike_slice = ((self.spikes > self.times[i] - self.trace)
                       & (self.spikes <= self.times[i]))
        spike_cum = self.spikes < self.times[i]

        pos_ids = self.network().id_from_nest_gid(self.senders[spike_slice])
        self.line_neurons_a.set_data(self.x[pos_ids], self.y[pos_ids])

        if self.show_spikes:
            # @todo: make this work for heterogeneous delays
            time = self.times[i]
            delays = np.average(self.network().get_delays())
            departures = self.spikes[spikes_slice]
            arrivals = departures + delays
            # get the spikers
            ids_dep = self.nids[self.senders[spikes_slice]]
            degrees = network.get_degrees('out', nodes=ids_dep)
            ids_dep = np.repeat(ids_dep, degrees)  # repeat based on out-degree
            x_dep = self.x[ids_dep]
            y_dep = self.y[ids_dep]
            # get their out-neighbours
            #~ for d, a in zip(departures, arrivals):

        super(AnimationNetwork, self)._draw(
            i, head, head_slice, spike_cum, spike_slice)

        return [self.line_neurons, self.line_neurons_a, self.line_spks_,
                self.line_spks_a, self.line_second_, self.line_second_a,
                self.line_second_e]

    def _init_draw(self):
        '''
        Remove ticks from spks/second axes, save background,
        then restore state to allow for moveable axes and labels.
        '''
        # remove
        xlim = self.spks.get_xlim()
        xlabel = self.spks.get_xlabel()
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
        lines = [self.line_spks_, self.line_spks_a, self.line_neurons_a,
                 self.line_second_, self.line_second_a, self.line_second_e,
                 self.line_neurons]
        for l in lines:
            l.set_data([], [])
        # initialize the neurons and connections between neurons
        draw_network(self.network(), ncolor='k', axis=self.env, show=False,
                     simple_nodes=True, decimate_connections=-1, tight=False,
                     **self.kwargs)
        if self.network().is_spatial():
            shape = self.network().shape
            xmin, ymin, xmax, ymax = shape.bounds
            dx = 0.02*(xmax-xmin)
            dy = 0.02*(ymax-ymin)
            self.env.set_xlim(xmin-dx, xmax+dx)
            self.env.set_ylim(ymin-dy, ymax+dy)
        self.line_neurons = self.env.lines[0]
        #~ self.line_neurons.set_data(self.x, self.y)
        #~ num_edges = self.network().edge_nb()
        #~ self.x_conn = np.zeros(3*num_edges)
        #~ self.y_conn = np.zeros(3*num_edges)
        #~ adj_mat = self.network().adjacency_matrix()
        #~ edges = adj_mat.nonzero()
        #~ self.x_conn[::3] = self.x[edges[0]]   # x position of source nodes
        #~ self.x_conn[1::3] = self.x[edges[1]]  # x position of target nodes
        #~ self.x_conn[2::3] = np.NaN            # NaN to separate
        #~ self.y_conn[::3] = self.y[edges[0]]   # y position of source nodes
        #~ self.y_conn[1::3] = self.y[edges[1]]  # y position of target nodes
        #~ self.y_conn[2::3] = np.NaN            # NaN to separate
        #~ self.env.plot(
            #~ self.x_conn[::self.decim_conn], self.y_conn[::self.decim_conn],
            #~ color='k', alpha=0.3, lw=1)


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


def _save_movie(animation, filename, fps, video_encoder, codec, bitrate,
                metadata, dpi, start, stop):
    if filename.endswith('.mp4') or filename.endswith('.avi'):
        ffcodec = 'h264' if filename.endswith('.mp4') else 'xvid'
        fig = animation.fig
        canvas_width, canvas_height = fig.get_size_inches()*fig.dpi
        # Open an ffmpeg process
        cmdstring = ('ffmpeg',
                     '-y', '-r', str(fps), # overwrite, 1fps
                     '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
                     '-pix_fmt', 'argb', # format
                     '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                     '-vcodec', ffcodec, filename) # output encoding
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

        # Draw frames and write to the pipe
        for i in range(start, stop):
            frame = int(i*animation.increment)
            # draw the frame
            animation._draw(frame)
            fig.canvas.draw()

            # extract the image as an ARGB string
            string = fig.canvas.tostring_argb()

            # write to pipe
            p.stdin.write(string)

        # Finish up
        p.communicate()
        animation._init_draw()
    else:
        if metadata is None:
            metadata = {"artist": "NNGT"}
        encoder = 'ffmpeg' if video_encoder == 'html5' else video_encoder
        Writer = anim.writers[encoder]
        if video_encoder == 'html5':
            codec = 'libx264'
        writer = Writer(codec=codec, fps=fps, bitrate=bitrate, metadata=metadata)
        animation.save(filename, writer=writer, dpi=dpi)


def _vector_field(q, dotx_func, doty_func, x, y, Is):
    '''
    Add the vector field of the x and y derivatives in phase space.

    Parameters
    ----------
    q : :class:`matplotlib.quiver.Quiver`
        Phase space quiver object.
    dotx_func : function
        User provided function giving :math:`\dot{x} = f(x, y, Is(t))`.
    doty_func : function
        User provided function giving :math:`\dot{y} = g(x, y, Is(t))`.
    x : :class:`numpy.ndarray`.
    y : :class:`numpy.ndarray`.
    Is : float
        Current (time dependent data).
    '''
    q.set_UVC(dotx_func(x, y, Is), doty_func(x, y, Is))

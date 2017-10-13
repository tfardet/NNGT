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

""" Configuration tools for NNGT """

import sys
import logging

import nngt
from .logger import _configure_logger, _init_logger, _log_message
from .reloading import reload_module
from .test_functions import mpi_checker, num_mpi_processes
from .errors import InvalidArgument


logger = logging.getLogger(__name__)


# ----------------- #
# Getter and setter #
# ----------------- #

def get_config(key=None):
    ''' Get the NNGT configuration as a dictionary. '''
    if key is None:
        return {key: val for key, val in nngt._config.items()}
    else:
        res = nngt._config[key]
        return res


def set_config(config, value=None, silent=False):
    '''
    Set NNGT's configuration.

    Parameters
    ----------
    config : dict or str
        Either a full configuration dictionary or one key to be set together
        with its associated value.
    value : object, optional (default: None)
        Value associated to `config` if `config` is a key.

    Examples
    --------

    >>> nngt.set_config({'multithreading': True, 'omp': 4})
    >>> nngt.set_config('multithreading', False)

    Note
    ----
    See the config file `nngt/nngt.conf.default` or `~/.nngt/nngt.conf` for
    details about your configuration.

    See also
    --------
    :func:`~nngt.get_config`
    '''
    old_mt     = nngt._config["multithreading"]
    old_mpi    = nngt._config["mpi"]
    old_gl     = nngt._config["graph_library"]
    new_config = None
    if not isinstance(config, dict):
        new_config = {config: value}
    else:
        new_config = config.copy()
    for key in new_config:
        if key not in nngt._config:
            raise KeyError("Unknown configuration property: {}".format(key))
        if key == "log_level":
            new_config[key] = _convert(new_config[key])
        if key == "graph_library" and new_config[key] != old_gl:
            nngt.use_library(new_config[key])
            del new_config[key]
    # check multithreading status and number of threads
    mt = "multithreading"
    if "omp" in new_config:
        has_mt = new_config.get(mt, old_mt)
        if new_config["omp"] > 1:
            if mt in new_config and not new_config[mt]:
                _log_message(logger, "WARNING",
                             "Updating to 'multithreading' == False with "
                             "'omp' greater than one.")
            elif mt not in new_config and not old_mt:
                new_config[mt] = True
                _log_message(logger, "WARNING",
                             "'multithreading' was set to False but new "
                             "'omp' is greater than one. Updating "
                             "'multithreading' to True.")
    if new_config.get('mpi', False) and new_config.get(mt, False):
        raise InvalidArgument('Cannot set both "mpi" and "multithreading" to '
                              'True simultaneously, choose one or the other.')
    elif new_config.get(mt, False):
        new_config['mpi'] = False
    elif new_config.get('mpi', False):
        if old_mt:
            new_config[mt] = False
            _log_message(logger, "WARNING",
                         '"mpi" set to True but previous configuration was '
                         'using OpenMP; setting "multithreading" to False '
                         'to switch to mpi algorithms.')
    # reset seeds if necessary
    reset_seeds  = (new_config.get("omp", 1) != nngt._config["omp"])
    reset_seeds += (nngt._config["mpi"] and new_config.get(mt, False))
    reset_seeds += (nngt._config[mt] and new_config.get('mpi', False))
    if reset_seeds:
        nngt._config['seeds'] = None
    # update
    nngt._config.update(new_config)
    # apply multithreading parameters
    new_multithreading = new_config.get("multithreading", old_mt)
    if new_multithreading != old_mt:
        reload_module(sys.modules["nngt"].generation.graph_connectivity)
    # if multithreading loading failed, set omp back to 1
    if not nngt._config['multithreading']:
        nngt._config['omp'] = 1
        nngt._config['seeds'] = None
    # reload for mpi
    if new_config.get('mpi', old_mpi) != old_mpi:
        reload_module(sys.modules["nngt"].generation.graph_connectivity)
    # set graph-tool config
    _set_gt_config(old_gl, new_config)
    # update matplotlib
    if nngt._config['use_tex']:
        import matplotlib
        matplotlib.rc('text', usetex=True)
    # log changes
    _configure_logger(nngt._logger)
    glib = (nngt._config["library"] if nngt._config["library"] is not None
            else nngt)
    num_mpi = num_mpi_processes()
    s_mpi = False if not nngt._config["mpi"] else "True ({} process{})".format(
                num_mpi, "es" if num_mpi > 1 else "")
    conf_info = config_info.format(
        gl     = nngt._config["graph_library"] + " " + glib.__version__[:5],
        thread = nngt._config["multithreading"],
        plot   = nngt._config["with_plot"],
        nest   = nngt._config["with_nest"],
        db     = nngt._config["use_database"],
        omp    = nngt._config["omp"],
        s      = "s" if nngt._config["omp"] > 1 else "",
        mpi    = s_mpi
    )
    if not silent:
        _log_conf_changed(conf_info)


# ----- #
# Tools #
# ----- #

def _convert(value):
    value = str(value)
    if value.isdigit():
        return int(value)
    elif value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.upper() == "CRITICAL":
        return logging.CRTICAL
    elif value.upper() == "DEBUG":
        return logging.DEBUG
    elif value.upper() == "ERROR":
        return logging.ERROR
    elif value.upper() == "INFO":
        return logging.INFO
    elif value.upper() == "WARNING":
        return logging.WARNING
    else:
        return value


def _load_config(path_config):
    ''' Load `~/.nngt.conf` and parse it, return the settings '''
    with open(path_config, 'r') as fconfig:
        options = [l.strip() for l in fconfig if l.strip() and l[0] != "#"]
        for opt in options:
            sep = opt.find("=")
            opt_name = opt[:sep].strip()
            nngt._config[opt_name] = _convert(opt[sep+1:].strip())
    _init_logger(nngt._logger)


@mpi_checker
def _log_conf_changed(conf_info):
    logger.info(conf_info)


def _set_gt_config(old_gl, new_config):
    using_gt  = old_gl == "graph-tool"
    using_gt *= new_config.get("graph_library", old_gl) == "graph-tool"
    using_gt *= nngt._config["library"] is not None
    if "omp" in new_config and using_gt:
        omp_nest = new_config["omp"]
        if nngt._config['with_nest']:
            import nest
            omp_nest = nest.GetKernelStatus("local_num_threads")
        if omp_nest == new_config["omp"]:
            nngt._config["library"].openmp_set_num_threads(nngt._config["omp"])
        else:
            _log_message(logger, "WARNING",
                         "Using NEST and graph_tool, OpenMP number must be "
                         "consistent throughout the code. Current NEST "
                         "config states omp = " + str(omp_nest) + ", hence "
                         "`graph_tool` configuration was not changed.")
    


config_info = '''
# -------------- #
# Config changed #
# -------------- #
Graph library:  {gl}
Multithreading: {thread} ({omp} thread{s})
Plotting:       {plot}
NEST support:   {nest}
Database:       {db}
MPI:            {mpi}
'''

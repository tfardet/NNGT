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
from .reloading import reload_module


logger = logging.getLogger(__name__)


# ----------------- #
# Getter and setter #
# ----------------- #

def get_config(key=None):
    if key is None:
        return {key: val for key, val in nngt._config.items()}
    else:
        res = nngt._config[key]
        return res


def set_config(config, value=None):
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
    old_multithreading = nngt._config["multithreading"]
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
    # check multithreading status and number of threads 
    if "omp" in new_config:
        has_mt = new_config.get("multithreading", old_multithreading)
        if new_config["omp"] > 1 and not has_mt:
             print("Warning: 'multithreading' is set to False but 'omp' is "
                   "greater than one.")
    # update
    nngt._config.update(new_config)
    # apply multithreading parameters
    new_multithreading = new_config.get("multithreading", old_multithreading)
    if new_multithreading != old_multithreading:
        reload_module(sys.modules["nngt"].generation.graph_connectivity)
    if "omp" in new_config and nngt._config["graph_library"] == "graph-tool":
        omp_nest = new_config["omp"]
        if nngt._config['with_nest']:
            import nest
            omp_nest = nest.GetKernelStatus("local_num_threads")
        if omp_nest == new_config["omp"]:
            nngt._config["library"].openmp_set_num_threads(nngt._config["omp"])
        else:
            logger.warning("Using NEST and graph_tool, OpenMP number must be "
                           "consistent throughout the code. Current NEST "
                           "config states omp = " + str(omp_nest) + ", hence "
                           "`graph_tool` configuration was not changed.")
    # log changes
    logger.setLevel(nngt._config["log_level"])
    conf_info = config_info.format(
        gl=nngt._config["graph_library"],
        thread=nngt._config["multithreading"],
        plot=nngt._config["with_plot"],
        nest=nngt._config["with_nest"],
        db=nngt._config["use_database"],
        omp=nngt._config["omp"],
        s="s" if nngt._config["omp"] > 1 else ""
    )
    logger.info(conf_info)


# ----- #
# Tools #
# ----- #

def _convert(value):
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
    config = {
        'color_lib': 'matplotlib',
        'db_folder': "~/.nngt/database",
        'db_to_file': False,
        'db_url': "mysql:///nngt_db",
        'graph': object,
        'graph_library': "",
        'library': None,
        'load_nest': False,
        'log_folder': "~/.nngt/log",
        'log_level': logging.INFO,
        'log_to_file': False,
        'mpl_backend': None,
        'multithreading': False,
        'omp': 1,
        'palette': 'Set1',
        'seed': None,
        'use_database': False,
        'set_omp_graph_tool': False,
        'with_nest': False,
        'with_plot': False,
    }
    with open(path_config, 'r') as fconfig:
        options = [l.strip() for l in fconfig if l.strip() and l[0] != "#"]
        for opt in options:
            sep = opt.find("=")
            opt_name = opt[:sep].strip()
            config[opt_name] = _convert(opt[sep+1:].strip())
    return config


config_info = '''
    --------------
    Config changed
    --------------
Graph library:  {gl}
Multithreading: {thread} ({omp} thread{s})
Plotting:       {plot}
NEST support:   {nest}
Database:       {db}
    --------------
'''

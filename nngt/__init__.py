#-*- coding:utf-8 -*-
#
# __init__.py
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

"""
NNGT
====

Package aimed at facilitating the analysis of Neural Networks and Graphs'
Topologies in Python by providing a unified interface for network generation
and analysis.

The library mainly provides algorithms for

1. generating networks
2. studying their topological properties
3. doing some basic spatial, topological, and statistical visualizations
4. interacting with neuronal simulators and analyzing neuronal activity


Available modules
-----------------

analysis
  Tools to study graph topology and neuronal activity.
core
  Where the main classes are coded; however, most useful classes and methods
  for users are loaded at the main level (`nngt`) when the library is imported,
  so `nngt.core` should generally not be used.
generation
  Functions to generate specific networks.
geometry
  Tools to work on metric graphs (see
  `PyNCulture <https://github.com/SENeC-Initiative/PyNCulture>`_).
io
  Tools for input/output operations.
lib
  Basic functions used by several most other modules.
simulation
  Tools to provide complex network generation with NEST and help analyze the
  influence of the network structure on neuronal activity.
plot
  Plot data or graphs using matplotlib.


Units
-----

Functions related to spatial embedding of networks are using micrometers
(um) as default unit; other units from the metric system can also be
provided:

- `mm` for milimeters
- `cm` centimeters
- `dm` for decimeters
- `m` for meters


Main classes and functions
==========================
"""

import os as _os
import errno as _errno
import importlib.util as _imputil
import shutil as _shutil
import sys as _sys
import logging as _logging

import numpy as _np


__version__ = '2.5.1'


# ----------------------- #
# Requirements and config #
# ----------------------- #

# IMPORTANT: configuration MUST come first
_config = {
    'color_lib': 'matplotlib',
    'db_folder': "~/.nngt/database",
    'db_name': "main",
    'db_to_file': False,
    'db_url': None,
    'graph': object,
    'backend': "nngt",
    'library': None,
    'log_folder': "~/.nngt/log",
    'log_level': 10,
    'log_to_file': False,
    'mpi': False,
    'mpi_comm': None,
    'mpl_backend': None,
    'msd': None,
    'multithreading': True,
    'omp': 1,
    'palette_continuous': 'magma',
    'palette_discrete': 'Set1',
    'use_database': False,
    'use_tex': False,
    'seeds': None,
    'load_nest': True,
    'load_gis': True,
    'with_nest': False,
    'with_plot': False,
}

# tools for nest interactions (can be used in config)

_old_nest_func = {}

# random generator for numpy

_rng = _np.random.default_rng()

# state of master seed (already seeded or not)
_seeded = False

# state of local seeds for multithreading or MPI (already used or not)
_seeded_local = False
_used_local   = False

# database (predeclare here, can be used in config)

_db      = None
_main_db = None

# configuration folders and files

_lib_folder = _os.path.expanduser('~') + '/.nngt'
_new_config = _os.path.expanduser('~') + '/.nngt/nngt.conf'
_default_config = _os.path.dirname(_os.path.realpath(__file__)) + \
                  '/nngt.conf.default'

# check that library config folder exists
if not _os.path.isdir(_lib_folder):
    try:
        _os.mkdir(_lib_folder)
    except OSError as e:
        if e.errno != _errno.EEXIST:
            raise

# IMPORTANT: first create logger
from .lib.logger import _init_logger, _log_message

_logger = _logging.getLogger(__name__)
_init_logger(_logger)

# IMPORTANT: afterwards, import config
from .lib.nngt_config import (get_config, set_config, _load_config, _convert,
                              _log_conf_changed, _lazy_load, _config_info)

# check that config file exists
if not _os.path.isfile(_new_config):  # if it does not, create it
    _shutil.copy(_default_config, _new_config)
else:                                 # if it does check it is up-to-date
    with open(_new_config, 'r+') as fconfig:
        _options = [l.strip() for l in fconfig if l.strip() and l[0] != "#"]
        config_version = ""
        for _opt in _options:
            sep = _opt.find("=")
            _opt_name = _opt[:sep].strip()
            _opt_val = _convert(_opt[sep+1:].strip())
            if _opt_name == "version":
                config_version = _opt_val
        if config_version != __version__:
            fconfig.seek(0)
            data = []
            with open(_default_config) as fdefault:
                data = [l for l in fdefault]
            i = 0
            for line in data:
                if '{version}' in line:
                    fconfig.write(line.format(version=__version__))
                    i += 1
                    break
                else:
                    fconfig.write(line)
                    i += 1
            for line in data[i:]:
                fconfig.write(line)
            fconfig.truncate()
            _log_message(_logger, "WARNING",
                         "Updating the configuration file, your previous "
                         "settings have be overwritten.")

_load_config(_new_config)

# multithreading
_config["omp"] = int(_os.environ.get("OMP", 1))

if _config["omp"] > 1:
    _config["multithreading"] = True


# --------------------- #
# Loading graph library #
#---------------------- #

from .lib.graph_backends import use_backend, analyze_graph

_libs = ['graph-tool', 'igraph', 'networkx']
_glib = _config['backend']

assert _glib in _libs or _glib == 'nngt', \
	   "Internal error for graph library loading, please report " +\
	   "this on GitHub."

try:
    use_backend(_config['backend'], False, silent=True)
except ImportError:
    idx = _libs.index(_config['backend'])
    del _libs[idx]
    keep_trying = True
    while _libs and keep_trying:
        try:
            use_backend(_libs[-1], False, silent=True)
            keep_trying = False
        except ImportError:
            _libs.pop()

if not _libs:
    use_backend('nngt', False, silent=True)
    _log_message(_logger, "WARNING",
                 "This module needs one of the following graph libraries to "
                 "study networks: `graph_tool`, `igraph`, or `networkx`.")


# ------- #
# Modules #
# ------- #

# import some tools into main namespace

from .io.graph_loading import load_from_file
from .io.graph_saving import save_to_file
from .lib.rng_tools import seed
from .lib.test_functions import on_master_process, num_mpi_processes

from .core.group_structure import Group, MetaGroup, Structure
from .core.neural_pop_group import (GroupProperty, MetaNeuralGroup,
                                    NeuralGroup, NeuralPop)
from .core.graph import Graph
from .core.spatial_graph import SpatialGraph
from .core.networks import Network, SpatialNetwork
from .generation.graph_connectivity import generate


# import modules

from . import analysis
from . import core
from . import generation
from . import geometry
from . import io
from . import lib


__all__ = [
    "analysis",
    "analyze_graph",
    "core",
    "generate",
    "generation",
    "geometry",
    "get_config",
    "Graph",
    "GroupProperty",
    "lib",
    "load_from_file",
    "Network",
    "NeuralGroup",
    "NeuralPop",
    "num_mpi_processes",
    "on_master_process",
    "save_to_file",
    "seed",
    "set_config",
    "SpatialGraph",
    "SpatialNetwork",
    "use_backend",
    "__version__"
]


# test geometry supports

try:
    import svg.path as _svg
    _has_svg = True
except:
    _has_svg = False
try:
    import dxfgrabber as _dxf
    _has_dxf = True
except:
    _has_dxf = False
try:
    import shapely as _shapely
    _has_shapely = _shapely.__version__
except:
    _has_shapely = False


# test if plot module is supported

try:
    from . import plot
    _config['with_plot'] = True
    __all__.append('plot')
except ImportError as e:
    _log_message(_logger, "DEBUG",
                 "An error occured, plot module will not be loaded: " + str(e))
    _config['with_plot'] = False


# lazy load for simulation module

if _config['load_nest'] and _imputil.find_spec("nest") is not None:
    _config['with_nest'] = True
    simulation = _lazy_load("nngt.simulation")
    __all__.append("simulation")


# lazy load for geospatial module

_has_geospatial = False
_has_geopandas = _imputil.find_spec("geopandas")

if _config["load_gis"] is not None and _has_shapely:
    geospatial = _lazy_load("nngt.geospatial")
    __all__.append("geospatial")
    _has_geospatial = True


# load database module if required

if _config["use_database"]:
    try:
        from . import database
        __all__.append('database')
    except ImportError as e:
        _log_message(_logger, "DEBUG",
                     "Could not load database module: " + str(e))


# ------------------------ #
# Print config information #
# ------------------------ #

_glib_version = (_config["library"].__version__[:5]
                 if _config["library"] is not None else __version__)


_log_info = _config_info.format(
    gl      = _config["backend"] + ' ' + _glib_version,
    thread  = _config["multithreading"],
    plot    = _config["with_plot"],
    nest    = _config["with_nest"],
    db      =_config["use_database"],
    omp     = _config["omp"],
    s       = "s" if _config["omp"] > 1 else "",
    mpi     = _config["mpi"],
    shapely = _has_shapely,
    svg     = _has_svg,
    dxf     = _has_dxf,
    geotool = _has_geospatial,
)

_log_conf_changed(_log_info)

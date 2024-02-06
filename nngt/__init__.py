# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/__init__.py

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
import sys as _sys
import logging as _logging

import numpy as _np


__version__ = '2.7.3'


# ----------------------- #
# Requirements and config #
# ----------------------- #

# IMPORTANT: configuration MUST come first
_config = {
    'color_lib': 'matplotlib',
    'db_folder': "",
    'db_name': "main",
    'db_to_file': False,
    'db_url': None,
    'graph': object,
    'backend': "nngt",
    'library': None,
    'log_folder': "",
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


# IMPORTANT: first create logger
from .lib.logger import _init_logger, _log_message

_logger = _logging.getLogger(__name__)
_init_logger(_logger)

# IMPORTANT: afterwards, import config
from .lib.nngt_config import (
    _config_info, _init_config, _lazy_load, _log_conf_changed, get_config,
    save_config, set_config, reset_config)

from .lib.graph_backends import use_backend, analyze_graph

_init_config()


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
    "save_config",
    "set_config",
    "SpatialGraph",
    "SpatialNetwork",
    "reset_config",
    "use_backend",
    "__version__"
]


# test geometry supports

try:
    import svg.path as _svg
    _has_svg = True
except ImportError:
    _has_svg = False
try:
    import dxfgrabber as _dxf
    _has_dxf = True
except ImportError:
    _has_dxf = False
try:
    import shapely as _shapely
    _has_shapely = _shapely.__version__
except ImportError:
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

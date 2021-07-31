#-*- coding:utf-8 -*-
#
# lib/nngt_config.py
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

""" Configuration tools for NNGT """

import os
import sys
import logging
import importlib.util as imputil
from copy import deepcopy
from importlib import reload

import numpy as np

import nngt
from .errors import InvalidArgument
from .logger import _configure_logger, _init_logger, _log_message
from .rng_tools import seed as nngt_seed
from .test_functions import mpi_checker, num_mpi_processes, mpi_barrier


logger = logging.getLogger(__name__)


# ----------------- #
# Getter and setter #
# ----------------- #

def get_config(key=None, detailed=False):
    '''
    Get the NNGT configuration as a dictionary.

    Note
    ----
    This function has no MPI barrier on it.
    '''
    if key is None:
        cfg = {
            key: val.copy() if isinstance(val, list) else val
            for key, val in nngt._config.items()
        }

        if detailed:
            return cfg
        else:
            # hide mpi conf if not used
            if not nngt._config["mpi"]:
                del cfg['mpi_comm']

            # hide technical stuff
            del cfg["graph"]
            del cfg["library"]
            del cfg["palette_continuous"]
            del cfg["palette_discrete"]
            del cfg["use_tex"]
            del cfg["mpl_backend"]
            del cfg["color_lib"]
            del cfg["load_nest"]
            del cfg["load_gis"]

            # hide database config if not used
            rm = []
            if not nngt._config["use_database"]:
                for k in cfg:
                    if k.startswith('db_'):
                        rm.append(k)

            # hide log config
            for k in cfg:
                if k.startswith('log_'):
                    rm.append(k)

            for k in rm:
                del cfg[k]

        return cfg

    res = nngt._config[key]

    return res


@mpi_barrier
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

    Notes
    -----
    See the config file `nngt/nngt.conf.default` or `~/.nngt/nngt.conf` for
    details about your configuration.

    This function has an MPI barrier on it, so it must always be called on all
    processes.

    See also
    --------
    :func:`~nngt.get_config`
    '''
    old_mt     = nngt._config["multithreading"]
    old_mpi    = nngt._config["mpi"]
    old_omp    = nngt._config["omp"]
    old_gl     = nngt._config["backend"]
    old_msd    = nngt._config["msd"]

    old_config = nngt._config.copy()
    new_config = None

    if not isinstance(config, dict):
        new_config = {config: value}
    else:
        new_config = config.copy()

    for key, val in new_config.items():
        # support for previous "palette" keyword
        if key not in nngt._config:
            if key == "palette":
                _log_message(logger, "WARNING",
                             "`palette` argument is deprecated and will be "
                             "removed in version 3.")
            else:
                raise KeyError(
                    "Unknown configuration property: {}".format(key))

        if key == "log_level":
            new_config[key] = _convert(val)
        if key == "backend" and val != old_gl:
            nngt.use_backend(val)
        if key == "log_folder":
            new_config["log_folder"] = os.path.abspath(
                os.path.expanduser(val))
        if key == "db_folder":
            new_config["db_folder"] = os.path.abspath(
                os.path.expanduser(val))

    # support for previous "palette" keyword
    if "palette" in new_config:
        new_config["palette_continuous"] = new_config["palette"]
        new_config["palette_discrete"]   = new_config["palette"]

        del new_config["palette"]

    # check multithreading status and number of threads
    _pre_update_parallelism(new_config, old_mt, old_omp, old_mpi)

    # update
    nngt._config.update(new_config)

    # apply multithreading parameters
    _post_update_parallelism(new_config, old_gl, old_msd, old_mt, old_mpi)

    # update matplotlib
    if nngt._config['use_tex']:
        import matplotlib
        matplotlib.rc('text', usetex=True)

    # update database
    if nngt._config["use_database"] and not hasattr(nngt, "db"):
        from .. import database
        sys.modules["nngt.database"] = database
        if nngt._config["db_to_file"]:
            _log_message(logger, "WARNING",
                        "This functionality is not available")

    # update nest
    if nngt._config["load_nest"] and imputil.find_spec("nest") is not None:
        _lazy_load("nngt.simulation")
        nngt._config["with_nest"] = True
    else:
        nngt._config["with_nest"] = False

    # check geometry
    try:
        import svg.path
        has_svg = True
    except:
        has_svg = False
    try:
        import dxfgrabber
        has_dxf = True
    except:
        has_dxf = False
    try:
        import shapely
        has_shapely = shapely.__version__
    except:
        has_shapely = False

    # check geospatial
    has_geospatial = False
    has_geopandas = imputil.find_spec("geopandas") is not None

    if nngt._config["load_gis"] and has_geopandas and has_shapely:
        _lazy_load("nngt.geospatial")
        has_geospatial = True

    # log changes
    _configure_logger(nngt._logger)

    glib = (nngt._config["library"] if nngt._config["library"] is not None
            else nngt)

    num_mpi = num_mpi_processes()

    s_mpi = False if not nngt._config["mpi"] \
            else "True ({} process{})".format(
                num_mpi, "es" if num_mpi > 1 else "")

    conf_info = _config_info.format(
        gl      = nngt._config["backend"] + " " + glib.__version__[:5],
        thread  = nngt._config["multithreading"],
        plot    = nngt._config["with_plot"],
        nest    = nngt._config["with_nest"],
        db      = nngt._config["use_database"],
        omp     = nngt._config["omp"],
        s       = "s" if nngt._config["omp"] > 1 else "",
        mpi     = s_mpi,
        shapely = has_shapely,
        svg     = has_svg,
        dxf     = has_dxf,
        geotool = has_geospatial,
    )

    if not silent and old_config != nngt._config:
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


@mpi_checker(logging=True)
def _log_conf_changed(conf_info):
    logger.info(conf_info)


def _set_gt_config(old_gl, new_config):
    using_gt  = old_gl == "graph-tool"
    using_gt *= new_config.get("backend", old_gl) == "graph-tool"
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


def _pre_update_parallelism(new_config, old_mt, old_omp, old_mpi):
    mt = "multithreading"

    if "omp" in new_config:
        if new_config["omp"] > 1:
            if mt in new_config and not new_config[mt]:
                _log_message(logger, "WARNING",
                             "Updating to 'multithreading' == False with "
                             "'omp' greater than one.")
            elif mt not in new_config and not old_mt:
                new_config[mt] = True
                _log_message(logger, "INFO",
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
            _log_message(logger, "INFO",
                         '"mpi" set to True but previous configuration was '
                         'using OpenMP; setting "multithreading" to False '
                         'to switch to mpi algorithms.')

    with_mt  = new_config.get(mt, old_mt)
    with_mpi = new_config.get('mpi', old_mpi)

    # check that seeds are correct
    if new_config.get('seeds', None) is not None:
        seeds = new_config['seeds']
        err  = 'Expected {} seeds.'
        err2 = 'All seeds must be different.'
        if with_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            assert size == len(seeds), err.format(size)
            assert len(set(seeds)) == len(seeds), err2
        elif with_mt:
            num_omp = new_config.get("omp", old_omp)
            assert num_omp == len(seeds), err.format(num_omp)
            assert len(set(seeds)) == len(seeds), err2
    else:
        # reset seeds if necessary
        # - because the number of threads changed
        reset_seeds  = (new_config.get("omp", 1) != nngt._config["omp"])
        # - because we switched from OpenMP to MPI
        reset_seeds += (with_mpi and old_mt)
        # - because we switched from MPI to OpenMP
        reset_seeds += (with_mt and old_mpi)

        if reset_seeds:
            new_config['seeds'] = None
            new_config['msd']   = None
            nngt._seeded        = False


def _post_update_parallelism(new_config, old_gl, old_msd, old_mt, old_mpi):
    # reload for omp
    new_multithreading = new_config.get("multithreading", old_mt)

    if new_multithreading != old_mt:
        reload(sys.modules["nngt"].generation.graph_connectivity)
        reload(sys.modules["nngt"].generation.connectors)
        reload(sys.modules["nngt"].generation.rewiring)

    # if multithreading loading failed, set omp back to 1
    if not nngt._config['multithreading']:
        nngt._config['omp'] = 1
        nngt._config['seeds'] = None

    # if MPI is on, set mpi_comm and check random numbers
    if new_config.get('mpi', old_mpi):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nngt._config['mpi_comm'] = comm
        # check that master seed is the same everywhere
        msd = nngt._config['msd']
        msd = comm.gather(msd, root=0)
        if rank == 0:
            if None not in msd:
                msd = np.array(msd, dtype=int)
                if not np.alltrue(msd == msd[0]):
                    nngt._config["mpi"] = False
                    raise InvalidArgument("'msd' entry must be the same on "
                                          "all MPI processes.")
            else:
                differs = [seed != None for seed in msd]
                if np.any(differs):
                    raise InvalidArgument("'msd' entry must be the same on "
                                          "all MPI processes.")

    # reload for mpi
    if new_config.get('mpi', old_mpi) != old_mpi:
        reload(sys.modules["nngt"].generation.graph_connectivity)
        reload(sys.modules["nngt"].generation.connectors)
        reload(sys.modules["nngt"].generation.rewiring)

    # set graph-tool config
    _set_gt_config(old_gl, new_config)

    # seed python RNGs
    if old_msd != nngt._config['msd'] or not nngt._seeded:
        nngt_seed(msd=nngt._config['msd'])


def _lazy_load(fullname):
    '''
    Lazy loading for simulation.

    From: https://stackoverflow.com/a/51126745/5962321
    '''
    try:
        return sys.modules[fullname]
    except KeyError:
        spec   = imputil.find_spec(fullname)
        module = imputil.module_from_spec(spec)
        loader = imputil.LazyLoader(spec.loader)

        # setup module and insert into sys.modules
        loader.exec_module(module)

        return module


_config_info = '''
# ----------- #
# NNGT config #
# ----------- #
Graph library:  {gl}
Multithreading: {thread} ({omp} thread{s})
MPI:            {mpi}
Plotting:       {plot}
NEST support:   {nest}
Shapely:        {shapely}
SVG support:    {svg}
DXF support:    {dxf}
Database:       {db}
Geo support:    {geotool}
'''

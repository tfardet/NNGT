# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/lib/test_functions.py

""" Test functions for the NNGT """

import collections
import functools
import inspect
import warnings

from collections.abc import Container as _container
from collections.abc import Iterable as _iterable
from collections.abc import KeysView as _key_view
from collections.abc import ValuesView as _value_view

import numpy as np

import nngt

from .decorator import decorate


def deprecated(version, reason=None, alternative=None, removal=None):
    '''
    Decorator to mark deprecated functions.
    '''
    def decorator(func):
        def wrapper(func, *args, **kwargs):
            # turn off filter temporarily
            warnings.simplefilter('always', DeprecationWarning)
            message = "Function {} is deprecated since version {}"
            message = message.format(func.__name__, version)
            if reason is not None:
                message += " because " + reason + "."
            else:
                message += "."
            if removal is not None:
                message += " It will be removed in version {}.".format(removal)
            if alternative is not None:
                message += " Use " + alternative + " instead."
            warnings.warn(message, category=DeprecationWarning)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return decorate(func, wrapper)
    return decorator


def on_master_process():
    '''
    Check whether the current code is executing on the master process (rank 0)
    if MPI is used.

    Returns
    -------
    True if rank is 0, if mpi4py is not present or if MPI is not used,
    otherwise False.
    '''
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            return True
        else:
            return False
    except ImportError:
        return True


def num_mpi_processes():
    ''' Returns the number of MPI processes (1 if MPI is not used) '''
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm.Get_size()
    except ImportError:
        return 1


def mpi_barrier(func=None):
    def wrapper(func, *args, **kwargs):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            comm.Barrier()
        except ImportError:
            pass

        if func is not None:
            return func(*args, **kwargs)

    # act as a real decorator
    if func is not None:
        return decorate(func, wrapper)

    # otherwise just execute the barrier
    wrapper(None)


def mpi_checker(logging=False):
    '''
    Decorator used to check for mpi and make sure only rank zero is used
    to store and generate the graph if the mpi algorithms are activated.
    '''
    def decorator(func):
        def wrapper(func, *args, **kwargs):
            # when using MPI, make sure everyone waits for the others
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                comm.Barrier()
            except ImportError:
                pass
            # check backend ("nngt" is fully parallel, not the others)
            backend = False
            if not logging:
                backend = nngt.get_config("backend") == "nngt"
            if backend or on_master_process():
                return func(*args, **kwargs)
            else:
                return None
        return decorate(func, wrapper)
    return decorator


def mpi_random(func):
    '''
    Decorator asserting that all processes start with same random seed when
    using mpi.
    '''
    def wrapper(func, *args, **kwargs):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if rank == 0:
                state = np.random.get_state()
            else:
                state = None

            state = comm.bcast(state, root=0)
            np.random.set_state(state)
        except ImportError:
            pass

        return func(*args, **kwargs)

    return decorate(func, wrapper)


def nonstring_container(obj):
    '''
    Returns true for any iterable which is not a string or byte sequence.
    '''
    if isinstance(obj, (_key_view, _value_view)):
        return True

    if not isinstance(obj, _container):
        return False

    if isinstance(obj, (bytes, str)):
        return False

    return True


def is_integer(obj):
    ''' Return whether the object is an integer '''
    return isinstance(obj, (int, np.integer))


def is_iterable(obj):
    ''' Return whether the object is iterable '''
    return isinstance(obj, _iterable)


def graph_tool_check(version_min):
    '''
    Raise an error for function not working with old versions of graph-tool.
    '''
    def decorator(func):
        def wrapper(func, *args, **kwargs):
            old_graph_tool = _old_graph_tool(version_min)
            if old_graph_tool:
                raise NotImplementedError('This function is not working for '
                                          'graph-tool < ' + version_min + '.')
            else:
                return func(*args, **kwargs)
        return decorate(func, wrapper)  # to preserve the docstring info
    return decorator


def _old_graph_tool(version_min):
    '''
    Check for old versions of graph-tool for which some functions are not
    working.
    '''
    return (nngt.get_config('backend') == 'graph-tool'
            and nngt.get_config('library').__version__[:4] < version_min)

# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# testing/__init__.py

"""
==============
Testing module
==============

This module tests the various functionalities of NNGT to make sure that all
implementations remain compatible with the graph libraries and versions 3.X of
python.

note ::
    When adding new tests, filename should be of the form `test_xxx.py` and the
    code should contain a list of functions called `test_yyy`
"""

# std imports
import sys
import importlib
from os import listdir, environ
from os.path import abspath, dirname, isfile
import unittest

# personal library
import nngt


# ------------- #
# Get the tests #
# ------------- #

# get the arguments for the graph library
backend = environ.get("GL", None)

if backend == "gt":
    nngt.use_backend("graph-tool")
    assert nngt.get_config('backend') == "graph-tool", \
           "Loading graph-tool failed..."
elif backend == "ig":
    nngt.use_backend("igraph")
    assert nngt.get_config('backend') == "igraph", \
           "Loading igraph failed..."
elif backend == "nx":
    nngt.use_backend("networkx")
    assert nngt.get_config('backend') == "networkx", \
           "Loading networkx failed..."
elif backend == "nngt":
    nngt.use_backend("nngt")
    assert nngt.get_config('backend') == "nngt", \
           "Loading nngt failed..."


# get the arguments for MPI/OpenMP + hide log
omp = int(environ.get("OMP", 1))
mpi = bool(environ.get("MPI", False))

conf = {
    "multithreading": omp > 1,
    "omp": omp,
    "mpi": mpi,
    "log_level": "ERROR",
}

nngt.set_config(conf, silent=True)


# get the tests
current_dir = dirname(abspath(__file__))
dir_files = listdir(current_dir)
sys.path.insert(0, current_dir)
testfiles = [fname[:-3] for fname in dir_files if (fname.startswith("test_")
             and fname.endswith(".py"))]

# remove the MPI test unless we're using it, otherwise remove the examples
if not nngt.get_config("mpi"):
    idx = None
    for i, test in enumerate(testfiles):
        if "test_mpi" in test:
            idx = i
            break
    del testfiles[idx]
else:
    idx = None
    for i, test in enumerate(testfiles):
        if "test_examples" in test:
            idx = i
            break
    del testfiles[idx]


tests = [importlib.import_module(name) for name in testfiles]


# ----------- #
# Run if main #
# ----------- #

if __name__ == "__main__":
    for test in tests:
        if hasattr(test, "suite"):
            unittest.TextTestRunner(verbosity=2).run(test.suite)

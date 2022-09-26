#-*- coding:utf-8 -*-

# test_analysis.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Check that the examples work.
"""

import os
from os import environ
from os.path import dirname, abspath, join

import pytest
from scipy.special import lambertw
import matplotlib as mpl

import nngt


''' Set state, paths, and global variables '''

# prevent drawing
mpl.use("agg")

# set multithreading
nngt.set_config("omp", int(environ.get("OMP", 2)))

# set example dir
current_dir = dirname(abspath(__file__))
idx_testing = current_dir.find('testing')
example_dir = join(current_dir[:idx_testing], 'doc/examples/')

# set globals
glob = {"lambertw": lambertw}


# ---------- #
# Test class #
# ---------- #

@pytest.mark.mpi_skip
@pytest.mark.skipif(int(environ.get("OMP", 1)) == 1, reason='Run only with OMP')
def test_example_graph_struct():
    for root, _, files in os.walk(join(example_dir, "graph_structure")):
        for fname in files:
            if fname.endswith(".py"):
                fullname = join(root, fname)
                with open(fullname) as f:
                    code = compile(f.read(), fullname, 'exec')
                    try:
                        exec(code, {})
                    except Exception as e:
                        print(f"Running example file {fname} failed.")
                        raise e


@pytest.mark.mpi_skip
@pytest.mark.skipif(int(environ.get("OMP", 1)) == 1, reason='Run only with OMP')
def test_example_graph_prop():
    for root, _, files in os.walk(join(example_dir, "graph_properties")):
        for fname in files:
            if fname.endswith(".py"):
                fullname = join(root, fname)
                with open(fullname) as f:
                    code = compile(f.read(), fullname, 'exec')
                    try:
                        exec(code)
                    except Exception as e:
                        print(f"Running example file {fname} failed.")
                        raise e


@pytest.mark.mpi_skip
@pytest.mark.skipif(int(environ.get("OMP", 1)) == 1, reason='Run only with OMP')
def test_examples():
    for root, _, files in os.walk(example_dir):
        if root == example_dir:
            for fname in files:
                if fname.endswith(".py"):
                    fullname = join(root, fname)
                    with open(fullname) as f:
                        code = compile(f.read(), fullname, 'exec')
                        try:
                            exec(code, glob)
                        except Exception as e:
                            print(f"Running example file {fname} failed.")
                            raise e

if __name__ == "__main__":
    if not nngt.get_config("mpi"):
        test_example_graph_struct()
        test_example_graph_prop()
        test_examples()

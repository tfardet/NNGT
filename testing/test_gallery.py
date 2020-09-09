#-*- coding:utf-8 -*-

# test_gallery.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.


""" Testing main functions """

import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

import nngt

folder = os.path.dirname(__file__)
folder = folder if folder else "."

root   = os.path.abspath(folder + "/..")
struct = root + "/examples/graph_structure"
props  = root + "/examples/graph_properties"


def mock_show():
    pass


@pytest.mark.mpi_skip
def test_structure(monkeypatch):
    '''
    Test gallery for structure visualization.
    '''
    monkeypatch.setattr(plt, "show", mock_show)

    for _, _, files in os.walk(struct):
        for fname in files:
            if fname.endswith(".py"):
                exec(open(fname).read())


@pytest.mark.mpi_skip
def test_properties(monkeypatch):
    '''
    Test gallery for graph properties.
    '''
    monkeypatch.setattr(plt, "show", mock_show)

    for _, _, files in os.walk(props):
        for fname in files:
            if fname.endswith(".py"):
                exec(open(fname).read())


if __name__ == "__main__":
    class mptch:
        def setattr(*args):
            pass

    mp = mptch()

    test_structure(mp)
    test_properties(mp)

#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_attributes.py

"""
Test the validity of the plotting functions.
"""

import os

import numpy as np
import pytest

import nngt
import nngt.generation as ng
import nngt.plot as nplt


# absolute directory path

dirpath = os.path.abspath(os.path.dirname(__file__))


# tests

@pytest.mark.mpi_skip
def test_plot_prop():
    net = nngt.generation.erdos_renyi(nodes=100, avg_deg=10)
    net.set_weights(distribution="gaussian",
                    parameters={"avg": 5, "std": 0.5})
    net.new_node_attribute("attr", "int",
                           values=np.random.randint(-10, 20, 100))

    nplt.degree_distribution(net, ["in", "out", "total"], show=False)

    nplt.edge_attributes_distribution(net, "weight", show=False)

    nplt.node_attributes_distribution(net, "attr", show=False)

    if nngt.get_config("backend") != "nngt":
        nplt.betweenness_distribution(net, show=False)


@pytest.mark.mpi_skip
def test_plot_net():
    fname = dirpath + "/Networks/p2p-Gnutella04.txt"

    net = nngt.load_from_file(fname, fmt="edge_list", separator="\t")

    nplt.draw_network(net, show=False)


@pytest.mark.mpi_skip
def test_draw_network_options():
    net = nngt.generation.erdos_renyi(nodes=100, avg_deg=10)

    nplt.draw_network(net, ncolor="in-degree", nsize=3, esize=2, show=False)

    nplt.draw_network(
        net, ncolor="betweenness", nsize="total-degree",
        decimate_connections=3, curved_edges=True, show=False)

    nplt.draw_network(net, ncolor="g", nshape='s', ecolor="b",
                      restrict_targets=[1, 2, 3], show=False)

    nplt.draw_network(net, restrict_nodes=[i for i in range(10)],
                      fast=True, show=False)

    nplt.draw_network(net, restrict_targets=[4, 5, 6, 7, 8],
                      show=False)

    nplt.draw_network(net, restrict_sources=[4, 5, 6, 7, 8],
                      show=False)


@pytest.mark.mpi_skip
def test_library_plot():
    ''' Check that plotting with the underlying backend library works '''
    nngt.use_backend("igraph")
    g = ng.newman_watts(4, 0.2, nodes=50)

    nplt.library_draw(g, show=True)


if __name__ == "__main__":
    # ~ test_plot_prop()
    # ~ test_plot_net()
    # ~ test_draw_network_options()
    test_library_plot()

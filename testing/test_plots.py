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
def test_plot_config():
    ''' Test the default plot configuration '''
    nngt.set_config("color_lib", "seaborn")
    nngt.set_config("palette", "viridis")
    nngt.set_config("palette_discrete", "Set3")
    nngt.set_config("palette_continuous", "magma")


@pytest.mark.mpi_skip
def test_plot_prop():
    net = ng.erdos_renyi(nodes=100, avg_deg=10)
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
def test_draw_network_options():
    net = nngt.generation.erdos_renyi(nodes=100, avg_deg=10)

    net.set_weights(np.random.randint(0, 20, net.edge_nb()))

    nplt.draw_network(net, ncolor="in-degree", nsize=3, esize=2,
                      colorbar=True, show=False)

    if nngt.get_config("backend") != "nngt":
        nplt.draw_network(net, ncolor="betweenness", nsize="total-degree",
                          decimate_connections=3, curved_edges=True,
                          show=False)

    nplt.draw_network(net, ncolor="g", nshape='s', ecolor="b",
                      restrict_targets=[1, 2, 3], show=False)

    nplt.draw_network(net, restrict_nodes=list(range(10)), fast=True,
                      show=False)

    nplt.draw_network(net, restrict_targets=[4, 5, 6, 7, 8], show=False)

    nplt.draw_network(net, restrict_sources=[4, 5, 6, 7, 8], show=False)

    # plot on a single axis
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    nplt.draw_network(net, simple_nodes=True, ncolor="k",
                      decimate_connections=-1, axis=ax, show=False)

    nplt.draw_network(net, simple_nodes=True, ncolor="r", nsize=2,
                      restrict_nodes=list(range(10)), esize='weight',
                      ecolor="b", fast=True, axis=ax, show=False)


@pytest.mark.mpi_skip
def test_library_plot():
    ''' Check that plotting with the underlying backend library works '''
    pop = nngt.NeuralPop.exc_and_inhib(50)
    g   = ng.newman_watts(4, 0.2, population=pop)

    g.set_weights(np.random.uniform(1, 5, g.edge_nb()))

    nplt.library_draw(g, show=False)

    nplt.library_draw(g, ncolor="total-degree", ecolor="k", show=False)

    if nngt.get_config("backend") != "nngt":
        nplt.library_draw(g, ncolor="in-degree", ecolor="betweenness",
                          esize='weight', max_esize=5, show=False)

    nplt.library_draw(g, nshape="s", esize="weight", layout="random",
                      show=False)

    nplt.library_draw(g, nshape="s", esize="weight", layout="random",
                      show=False)

    nplt.library_draw(g, ncolor="in-degree", esize="weight", layout="circular",
                      show=False)


if __name__ == "__main__":
    test_plot_config()
    test_plot_prop()
    test_draw_network_options()
    test_library_plot()

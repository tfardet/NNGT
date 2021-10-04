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
    nngt.set_config("palette_discrete", "Set3")
    nngt.set_config("palette_continuous", "magma")


@pytest.mark.mpi_skip
def test_plot_prop():
    num_nodes = 50
    net = ng.erdos_renyi(nodes=num_nodes, avg_deg=5)

    net.set_weights(distribution="gaussian",
                    parameters={"avg": 5, "std": 0.5})

    net.new_node_attribute("attr", "int",
                           values=np.random.randint(-10, 20, num_nodes))

    nplt.degree_distribution(net, ["in", "out", "total"], show=False)

    nplt.edge_attributes_distribution(
        net, "weight", colors="g", show=False)

    nplt.node_attributes_distribution(
        net, "out-degree", colors="r", show=False)

    if nngt.get_config("backend") != "nngt":
        nplt.edge_attributes_distribution(
            net, ["betweenness", "weight"], colors=["g", "b"],
            axtitles=["Edge betw.", "Weights"], show=False)

        nplt.node_attributes_distribution(
            net, ["betweenness", "attr", "out-degree"], colors=["r", "g", "b"],
            show=False)


@pytest.mark.mpi_skip
def test_draw_network_options():
    net = nngt.generation.erdos_renyi(nodes=100, avg_deg=10)

    net.set_weights(np.random.randint(0, 20, net.edge_nb()))
    
    net.new_node_attribute("attr", "int",
                           values=np.random.randint(-10, 20, 100))

    nplt.draw_network(net, ncolor="in-degree", nsize=3, esize=2,
                      colorbar=True, show=False)

    if nngt.get_config("backend") != "nngt":
        nplt.draw_network(net, ncolor="betweenness", nsize="total-degree",
                          decimate_connections=3, curved_edges=True,
                          show=False)

    # restrict nodes

    nplt.draw_network(net, ncolor="g", nshape='s', ecolor="b",
                      restrict_targets=[1, 2, 3], curved_edges=True, show=False)

    nplt.draw_network(net, restrict_nodes=list(range(10)), fast=True,
                      show=False)

    nplt.draw_network(net, restrict_targets=[4, 5, 6, 7, 8], show=False)

    nplt.draw_network(net, restrict_sources=[4, 5, 6, 7, 8], simple_nodes=True,
                      show=False)

    # colors and sizes
    for fast in (True, False):
        maxns = 100 if fast else 20
        minns = 10 if fast else 2
        maxes = 2 if fast else 20
        mines = 0.2 if fast else 2

        nplt.draw_network(net, ncolor="r", nalpha=0.5, ecolor="#999999",
                          ealpha=0.5, nsize="in-degree", max_nsize=maxns,
                          min_nsize=minns, esize="weight", max_esize=maxes,
                          min_esize=mines, fast=fast, show=False)

        nplt.draw_network(net, ncolor="attr", nalpha=1, ecolor="slateblue",
                          ealpha=0.2, nsize=maxns, esize=maxes, fast=fast,
                          show=False)

    # plot on a single axis
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    nplt.draw_network(net, simple_nodes=True, ncolor="k",
                      decimate_connections=-1, axis=ax, show=False)

    nplt.draw_network(net, simple_nodes=True, ncolor="r", nsize=20,
                      restrict_nodes=list(range(10)), esize='weight',
                      ecolor="b", fast=True, axis=ax, show=False)


@pytest.mark.mpi_skip
def test_group_plot():
    ''' Test plotting with a Network and group colors '''
    gsize = 5

    g1 = nngt.Group(gsize)
    g2 = nngt.Group(gsize)

    s = nngt.Structure.from_groups({"1": g1, "2": g2})

    positions = np.concatenate((
        nngt._rng.uniform(-5, -2, size=(gsize, 2)),
        nngt._rng.uniform(2, 5, size=(gsize, 2))))

    g = nngt.SpatialGraph(2*gsize, structure=s, positions=positions)

    nngt.generation.connect_groups(g, g1, g1, "erdos_renyi", edges=5)
    nngt.generation.connect_groups(g, g1, g2, "erdos_renyi", edges=5)
    nngt.generation.connect_groups(g, g2, g2, "erdos_renyi", edges=5)
    nngt.generation.connect_groups(g, g2, g1, "erdos_renyi", edges=5)

    g.new_edge(6, 6, self_loop=True)

    nplt.draw_network(g, ncolor="group", ecolor="group", show_environment=False,
                      fast=True, show=False)

    nplt.draw_network(g, ncolor="group", ecolor="group", max_nsize=0.4,
                      esize=0.3, show_environment=False, show=False)

    nplt.draw_network(g, ncolor="group", ecolor="group", max_nsize=0.4,
                      esize=0.3, show_environment=False, curved_edges=True,
                      show=False)


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

    edges = g.edges_array[::5]

    nplt.library_draw(g, ncolor="in-degree", esize="weight", layout="circular",
                      restrict_edges=edges, show=False)


@pytest.mark.mpi_skip
@pytest.mark.skipif(nngt.get_config("backend") == "nngt", reason="Lib needed.")
def test_hive_plot():
    g = nngt.load_from_file(dirpath + "/Networks/rat_brain.graphml",
                            attributes=["weight"], cleanup=True,
                            attributes_types={"weight": float})

    cc = nngt.analysis.local_clustering(g, weights="weight")

    g.new_node_attribute("cc", "double", values=cc)

    g.new_node_attribute("strength", "double",
                         values=g.get_degrees(weights="weight"))

    g.new_node_attribute(
        "SC", "double", values=nngt.analysis.subgraph_centrality(g))

    cc_bins = [0, 0.1, 0.25, 0.6]

    nplt.hive_plot(g, g.get_degrees(), axes="cc", axes_bins=cc_bins)

    nplt.hive_plot(g, "strength", axes="cc", axes_bins=cc_bins,
                   axes_units="rank")

    # set colormap and highlight nodes
    nodes = list(range(g.node_nb(), 3))

    nplt.hive_plot(g, "strength", axes="SC", axes_bins=4,
                   axes_colors="brg", highlight_nodes=nodes)

    # multiple radial axes and edge highlight
    rad_axes = ["cc", "strength", "SC"]
    edges    = g.get_edges(source_node=nodes)

    nplt.hive_plot(g, rad_axes, rad_axes,
                   nsize=g.get_degrees(), max_nsize=50, highlight_edges=edges)

    # check errors
    with pytest.raises(ValueError):
        nplt.hive_plot(g, [cc, "closeness", "strength"])
        nplt.hive_plot(g, 124)

    with pytest.raises(AssertionError):
        nplt.hive_plot(g, cc, axes=[1, 2, 3])
        nplt.hive_plot(g, cc, axes=1)
        nplt.hive_plot(g, cc, axes="groups")


@pytest.mark.mpi_skip
def test_plot_spatial_alpha():
    ''' Test positional layout and alpha parameters '''
    num_nodes = 4
    pos = [(1, 1), (0, -1), (-1, -1), (-1, 1)]

    g = nngt.SpatialGraph(num_nodes, positions=pos)
    g.new_edges([(0, 1), (0, 2), (1, 3), (3, 2)])

    for fast in (True, False):
        nplt.draw_network(g, nsize=0.02 + 30*fast, ealpha=1, esize=0.1 + fast,
                          fast=fast)

        nplt.draw_network(g, layout=[(y, x) for (x, y) in pos],
                          show_environment=False, nsize=0.02 + 30*fast,
                          nalpha=0.5, esize=0.1 + 3*fast, fast=fast)


@pytest.mark.mpi_skip
def test_annotations():
    num_nodes = 5
    positions = nngt._rng.uniform(-10, 10, (num_nodes, 2))

    g = nngt.generation.erdos_renyi(edges=10, nodes=num_nodes,
                                    positions=positions)

    g.new_node_attribute("name", "string", values=["a", "b", "c", "d", "e"])

    nplt.draw_network(g, annotate=False, show=False)
    nplt.draw_network(g, show=False)
    nplt.draw_network(g, annotations="name", show=False)


if __name__ == "__main__":
    test_plot_config()
    test_plot_prop()
    test_draw_network_options()
    test_library_plot()
    test_hive_plot()
    test_plot_spatial_alpha()
    test_group_plot()
    test_annotations()

#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_nest.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

import nngt
import nngt.generation as ng
import numpy as np
import pytest


def make_nest_net(size, w, deg):
    '''
    Create a network in NEST
    '''
    import nngt.simulation as ns

    net = nngt.Network.exc_and_inhib(size)

    pop = net.population

    ng.connect_groups(net, pop, pop, graph_model="fixed_degree",
                      degree=deg, degree_type="out", weights=w)

    gids = net.to_nest()

    return net, gids


@pytest.mark.skipif(nngt.get_config('mpi'), reason="Don't test for MPI")
def test_net_creation():
    '''
    Test the creation of a network in NEST.
    '''
    nest = pytest.importorskip("nest")

    nest.ResetKernel()

    w = 5.

    net, gids = make_nest_net(100, w, deg=10)

    # check nodes and connections
    assert len(gids) == net.node_nb()

    conn = nest.GetConnections()

    assert len(conn) == net.edge_nb()

    weights = np.array([d['weight'] for d in nest.GetStatus(conn)])

    # check inhibitory connections
    etypes = net.get_edge_types()

    num_i = np.sum(etypes == -1)

    assert num_i == int(0.2*net.edge_nb())

    assert np.sum(weights == -w) == num_i

    assert set(weights) == {-w, w}


@pytest.mark.skipif(nngt.get_config('mpi'), reason="Don't test for MPI")
def test_utils():
    '''
    Check NEST utility functions
    '''
    nest = pytest.importorskip("nest")

    import nngt.simulation as ns

    nest.ResetKernel()

    resol = nest.GetKernelStatus('resolution')

    w = 5.

    net, gids = make_nest_net(100, w, deg=10)

    # set inputs
    ns.set_noise(gids, 0., 20.)

    assert len(nest.GetConnections()) == net.edge_nb() + net.node_nb()

    ns.set_poisson_input(gids, 5.)

    assert len(nest.GetConnections()) == net.edge_nb() + 2*net.node_nb()

    ns.set_minis(net, base_rate=1.5, weight=2.)

    assert len(nest.GetConnections()) == net.edge_nb() + 3*net.node_nb()

    ns.set_step_currents(gids[::2], times=[40., 60.], currents=[800., 0.])

    min_conn = net.edge_nb() + 3*net.node_nb() + 1
    max_conn = net.edge_nb() + 4*net.node_nb()

    num_conn = len(nest.GetConnections())

    assert min_conn <= num_conn <= max_conn

    # check randomization of neuronal properties
    vms = {d['V_m'] for d in nest.GetStatus(gids)}

    assert len(vms) == 1

    ns.randomize_neural_states(net, {'V_m': ('uniform', -70., -60.)})

    vms = {d['V_m'] for d in nest.GetStatus(gids)}

    assert len(vms) == net.node_nb()

    # monitoring nodes
    sd, _ = ns.monitor_groups(net.population, net)

    assert len(nest.GetConnections()) == num_conn + net.node_nb()

    vm, rec = ns.monitor_nodes(gids[0], nest_recorder='voltmeter',
                               params={'interval': resol})

    assert len(nest.GetConnections()) == num_conn + net.node_nb() + 1

    nest.Simulate(100.)

    ns.plot_activity(show=False)

    ns.plot_activity(vm, rec, show=True)


if __name__ == "__main__":
    if not nngt.get_config("mpi"):
        test_net_creation()
        test_utils()

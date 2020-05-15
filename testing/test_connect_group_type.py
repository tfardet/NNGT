#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_generation.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

"""
Test the connect group and type methods.
"""

import pytest
import numpy as np

import nngt
import nngt.generation as ng


@pytest.mark.mpi_skip
def test_fixed():
    ''' Fixed degree with type '''
    pop = nngt.NeuralPop.exc_and_inhib(1000)

    igroup = pop["inhibitory"]
    egroup = pop["excitatory"]

    net = nngt.Network(population=pop)

    deg_e = 50
    ng.connect_neural_types(net, 1, [-1, 1], graph_model="fixed_degree",
                           degree=deg_e, degree_type="out-degree")

    deg_i = 100
    ng.connect_neural_types(net, -1, [-1, 1], graph_model="fixed_degree",
                            degree=deg_i, degree_type="out-degree")

    edeg = net.get_degrees("out", nodes=egroup.ids)
    ideg = net.get_degrees("out", nodes=igroup.ids)

    assert np.all(edeg == deg_e)
    assert np.all(ideg == deg_i)

    edeg = net.get_degrees("in", nodes=egroup.ids)
    ideg = net.get_degrees("in", nodes=igroup.ids)

    avg_deg = (deg_e*egroup.size + deg_i*igroup.size) / pop.size
    std_deg = np.sqrt(avg_deg)

    assert avg_deg - std_deg < np.average(edeg) < avg_deg + std_deg
    assert avg_deg - std_deg < np.average(ideg) < avg_deg + std_deg


@pytest.mark.mpi_skip
def test_gaussian():
    ''' Gaussian degree with groups '''
    pop = nngt.NeuralPop.exc_and_inhib(1000)

    igroup = pop["inhibitory"]
    egroup = pop["excitatory"]

    net = nngt.Network(population=pop)

    avg_e = 50
    std_e = 5
    ng.connect_groups(net, egroup, [igroup, egroup],
                      graph_model="gaussian_degree", avg=avg_e, std=std_e,
                      degree_type="out-degree")

    avg_i = 100
    std_i = 5
    ng.connect_groups(net, igroup, [igroup, egroup],
                      graph_model="gaussian_degree", avg=avg_i, std=std_i,
                      degree_type="out-degree")

    edeg = net.get_degrees("out", nodes=egroup.ids)
    ideg = net.get_degrees("out", nodes=igroup.ids)

    # call only on root process (for mpi) and not if using distributed backend
    if nngt.on_master_process() and nngt.get_config("backend") != "nngt":
        assert avg_e - std_e < np.average(edeg) < avg_e + std_e
        assert avg_i - std_i < np.average(ideg) < avg_i + std_i
    elif nngt.get_config("mpi") and nngt.get_config("backend") == "nngt":
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_mpi = comm.Get_size()

        edeg = comm.gather(edeg[rank::num_mpi], root=0)
        ideg = comm.gather(ideg[rank::num_mpi], root=0)

        if nngt.on_master_process():
            assert avg_e - std_e < np.average(edeg) < avg_e + std_e
            assert avg_i - std_i < np.average(ideg) < avg_i + std_i

    edeg = net.get_degrees("in", nodes=egroup.ids)
    ideg = net.get_degrees("in", nodes=igroup.ids)

    avg_deg = (avg_e*egroup.size + avg_i*igroup.size) / pop.size
    std_deg = np.sqrt(avg_deg)

    # call only on root process (for mpi) unless using distributed backend
    if nngt.on_master_process() and nngt.get_config("backend") != "nngt":
        assert avg_deg - std_deg < np.average(edeg) < avg_deg + std_deg
        assert avg_deg - std_deg < np.average(ideg) < avg_deg + std_deg
    elif nngt.get_config("mpi") and nngt.get_config("backend") == "nngt":
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_mpi = comm.Get_size()

        edeg = comm.gather(edeg, root=0)
        ideg = comm.gather(ideg, root=0)

        if nngt.on_master_process():
            edeg = np.sum(edeg, axis=0)
            ideg = np.sum(ideg, axis=0)

            assert avg_deg - std_deg < np.average(edeg) < avg_deg + std_deg
            assert avg_deg - std_deg < np.average(ideg) < avg_deg + std_deg


def test_group_vs_type():
    ''' Gaussian degree with groups and types '''
    # first with groups
    nngt.seed(0)

    pop = nngt.NeuralPop.exc_and_inhib(1000)

    igroup = pop["inhibitory"]
    egroup = pop["excitatory"]

    net1 = nngt.Network(population=pop)

    all_groups = list(pop.keys())  # necessary to have same order as types

    avg_e = 50
    std_e = 5
    ng.connect_groups(net1, egroup, all_groups, graph_model="gaussian_degree",
                      avg=avg_e, std=std_e, degree_type="out-degree")

    avg_i = 100
    std_i = 5
    ng.connect_groups(net1, igroup, all_groups, graph_model="gaussian_degree",
                      avg=avg_i, std=std_i, degree_type="out-degree")

    # then with types
    nngt.seed(0)

    pop = nngt.NeuralPop.exc_and_inhib(1000)

    igroup = pop["inhibitory"]
    egroup = pop["excitatory"]

    net2 = nngt.Network(population=pop)

    avg_e = 50
    std_e = 5
    ng.connect_neural_types(net2, 1, [-1, 1], graph_model="gaussian_degree",
                            avg=avg_e, std=std_e, degree_type="out-degree")

    avg_i = 100
    std_i = 5
    ng.connect_neural_types(net2, -1, [-1, 1], graph_model="gaussian_degree",
                            avg=avg_i, std=std_i, degree_type="out-degree")

    # call only on root process (for mpi) unless using distributed backend
    if nngt.on_master_process() or nngt.get_config("backend") == "nngt":
        # check that both networks are equals
        assert np.all(net1.get_degrees() == net2.get_degrees())


if __name__ == "__main__":
    import os

    if os.environ.get("MPI"):
        nngt.set_config("mpi", True)


    if os.environ.get("GL", ""):
        nngt.set_config("backend", os.environ["GL"])

    if not nngt.get_config("mpi"):
        test_fixed()

    test_gaussian()
    test_group_vs_type()

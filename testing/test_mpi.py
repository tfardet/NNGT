#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# test_mpi.py
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Test the main methods of the :mod:`~nngt.generation` module.
"""

import os
import unittest

import numpy as np

import nngt
from nngt.analysis import *
from nngt.lib.connect_tools import _compute_connections

from base_test import TestBasis, XmlHandler, network_dir
from test_generation import _distance_rule_theo, _distance_rule_exp
from tools_testing import foreach_graph


if os.environ.get("MPI"):
    nngt.set_config("mpi", True)


# -------- #
# Test MPI #
# -------- #

class TestMPI(TestBasis):

    '''
    Class testing the main methods of the :mod:`~nngt.generation` module.
    '''

    theo_prop = {
        "distance_rule": _distance_rule_theo,
    }

    exp_prop = {
        "distance_rule": _distance_rule_exp,
    }

    tolerance = 0.08

    @property
    def test_name(self):
        return "test_mpi"

    @unittest.skipIf(not nngt.get_config('mpi'), "Not using MPI.")
    def gen_graph(self, graph_name):
        di_instructions = self.parser.get_graph_options(graph_name)
        graph = nngt.generate(di_instructions)
        if nngt.on_master_process():
            graph.set_name(graph_name)
        return graph, di_instructions

    @foreach_graph
    def test_model_properties(self, graph, instructions, **kwargs):
        '''
        When generating graphs from on of the preconfigured models, check that
        the expected properties are indeed obtained.
        '''
        if nngt.get_config("backend") != "nngt" and nngt.on_master_process():
            graph_type = instructions["graph_type"]
            ref_result = self.theo_prop[graph_type](instructions)
            computed_result = self.exp_prop[graph_type](graph, instructions)

            if graph_type == 'distance_rule':
                # average degree
                self.assertTrue(
                    ref_result[0] == computed_result[0],
                    "Avg. deg. for graph {} failed:\nref = {} vs exp {}\
                    ".format(graph.name, ref_result[0], computed_result[0]))

                # average error on distance distribution
                sqd = np.square(
                    np.subtract(ref_result[1:], computed_result[1:]))
                avg_sqd = sqd / np.square(computed_result[1:])
                err = np.sqrt(avg_sqd).mean()
                tolerance = (self.tolerance if instructions['rule'] == 'lin'
                             else 0.25)

                self.assertTrue(err <= tolerance,
                    "Distance distribution for graph {} failed:\nerr = {} > {}\
                    ".format(graph.name, err, tolerance))
        elif nngt.get_config("backend") == "nngt":
            from mpi4py import MPI
            comm       = MPI.COMM_WORLD
            num_proc   = comm.Get_size()
            graph_type = instructions["graph_type"]
            ref_result = self.theo_prop[graph_type](instructions)
            computed_result = self.exp_prop[graph_type](graph, instructions)
            if graph_type == 'distance_rule':
                # average degree
                self.assertTrue(
                    ref_result[0] == computed_result[0] * num_proc,
                    "Avg. deg. for graph {} failed:\nref = {} vs exp {}\
                    ".format(graph.name, ref_result[0], computed_result[0]))
                # average error on distance distribution
                sqd = np.square(
                    np.subtract(ref_result[1:], computed_result[1:]))
                avg_sqd = sqd / np.square(computed_result[1:])
                err = np.sqrt(avg_sqd).mean()
                tolerance = (self.tolerance if instructions['rule'] == 'lin'
                             else 0.25)
                self.assertTrue(err <= tolerance,
                    "Distance distribution for graph {} failed:\nerr = {} > {}\
                    ".format(graph.name, err, tolerance))



# ---------- #
# Test suite #
# ---------- #

if nngt.get_config('mpi'):
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMPI)

if __name__ == "__main__":
    unittest.main()

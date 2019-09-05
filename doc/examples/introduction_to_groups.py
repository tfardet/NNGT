#!/usr/bin/env python
#-*- coding:utf-8 -*-
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

''' Introduction to neural groups '''

import numpy as np

import nngt


''' ------------- #
# Creating groups #
# ------------- '''

# the default group is empty, which is not very useful in general
empty_group = nngt.NeuralGroup()
print(empty_group)
print("Group is empty?", empty_group.size == 0, "\nIt is therefore invalid?",
      "No!" if empty_group.is_valid else "Yes!", "\n")

# to create a useful group, one can just say how many neurons it should contain
group1 = nngt.NeuralGroup(500)  # a group with 500 neurons
print("Ids are not created", group1.ids,
      "but the size is stored:", group1.size, "\n")

# if you want to set the ids directly, you can pass them directly, otherwise
# they will be determine automatically when a Network is created using the group
group2 = nngt.NeuralGroup(range(10, 20))  # 10 neurons with ids from 10 to 19
print("Neuron ids are:", group2.ids, "\n")


''' ------------------- #
# More group properties #
# ------------------- '''

# group can have names
named_group = nngt.NeuralGroup(500, name="named_group")
print("I'm a named group!", named_group, "\n")

# more importantly, they can store whether neurons are excitatory or inhibitory
exc   = nngt.NeuralGroup(800, neuron_type=1)   # excitatory group (optional)
exc2  = nngt.NeuralGroup(800)                  # also excitatory
inhib = nngt.NeuralGroup(200, neuron_type=-1)  # inhibitory group
print("'exc2' is an excitatory group:", exc2.neuron_type == 1,
      "/ 'inhib' is an inhibitory group:", inhib.neuron_type == -1, "\n")


''' ---------------------------------- #
# Complete groups for NEST simulations #
# ---------------------------------- '''

# to make a complete group, one must include a valid neuronal model and
# (optionally) associated parameters

pyr = nngt.NeuralGroup(800, neuron_type=1, neuron_model="iaf_psc_alpha",
                       neuron_param={"tau_m": 50.}, name="pyramidal_cells")

fsi = nngt.NeuralGroup(200, neuron_type=-1, neuron_model="iaf_psc_alpha",
                       neuron_param={"tau_m": 20.},
                       name="fast_spiking_interneurons")

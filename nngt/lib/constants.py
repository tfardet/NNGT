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

""" Constant values for NNGT """


__all__ = [
    "default_neuron",
    "default_synapse",
    "POS",
    "BWEIGHT",
    "DIST",
    "WEIGHT",
    "DELAY",
    "TYPE",
]


# ----- #
# Names #
# ----- #

POS = "position"
DIST = "distance"
WEIGHT = "weight"
BWEIGHT = "bweight"
DELAY = "delay"
TYPE = "type"


# ------------ #
# Basic values #
# ------------ #

default_neuron = "aeif_cond_alpha"
''' :class:`string`, the default NEST neuron model '''
default_synapse = "static_synapse"
''' :class:`string`, the default NEST synaptic model '''
default_delay = 1.
''' :class:`double`, the default synaptic delay in NEST '''

# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/lib/constants.py

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

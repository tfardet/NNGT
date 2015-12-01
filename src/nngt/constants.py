#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Constant values for NNGT """

cst__all__ = ["version",
                "default_neuron",
                "default_synapse"]

version = '0.3'
''' :class:`string`, the current version '''


#-----------------------------------------------------------------------------#
# Names
#------------------------
#

POS = "position"
DIST = "distance"
WEIGHT = "weight"
DELAY = "delay"
TYPE = "type"


#-----------------------------------------------------------------------------#
# Basic values
#------------------------
#

#~ default_neuron = "iaf_neuron"
default_neuron = "aeif_cond_alpha"
''' :class:`string`, the default NEST neuron model '''
default_synapse = "static_synapse"
''' :class:`string`, the default NEST synaptic model '''
default_delay = 1.
''' :class:`double`, the default synaptic delay in NEST '''

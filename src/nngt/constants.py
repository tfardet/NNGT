#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
Constant values for NNGT
"""

cst__all__ = ["version",
                "default_neuron",
                "default_synapse"]

version = '0.2a'
''' :class:`string`, the current version '''

default_neuron = "iaf_neuron"
''' :class:`string`, the default NEST neuron model '''
default_synapse = "static_synapse"
''' :class:`string`, the default NEST synaptic model '''

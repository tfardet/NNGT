#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generating the weights of the graph object's connections """

import numpy as np

def uniform_weights(graph_class, min_weight=-1., max_weight=1.):
    pass

def gaussian_weights(graph_class, avg_weight=0., std_dev=1.):
    pass

def lognormal_weights(graph_class, position=0., scale=1.):
    pass

def correlated_weights(graph_class, connection_property,
                       min_weight=0., max_weight=1.):
    pass

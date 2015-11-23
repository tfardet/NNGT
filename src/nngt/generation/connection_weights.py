#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generating the weights of the graph object's connections """

import numpy as np
import scipy.sparse as ssp



def uniform_weights(min_weight=0., max_weight=1.5, dense=False, **kwargs):
    pass

def gaussian_weights(avg_weight=1., std_dev=0.2, dense=False, **kwargs):
    pass

def lognormal_weights(position=1., scale=0.2, dense=False, **kwargs):
    pass

def lin_correlated_weights(correl_attribute, noise_scale=None,  min_weight=0., 
                          max_weight=2., dense=False, **kwargs):
    pass

def log_correlated_weights(correl_attribute, noise_scale=None, min_weight=0., 
                           max_weight=2., dense=False, **kwargs):
    pass

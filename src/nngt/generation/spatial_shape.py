#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generating the shape of the neurons' environment """

import numpy as np

		

#
#---
# Shape generation
#--------------------

def rectangle(length, width, unit='mm'):
	"""
	Generate a rectangular area to contain the neurons.
	
	Parameters
    ----------
    length: double
		Length of the rectangle (by default in mm).
	width: double
		Width of the rectangle (by default in mm).
	unit: string (default 'mm')
		Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'
	
	Returns
    -------
    AGNet.generation.Shape object
	"""

def ellipse(major_axis, minor_axis, unit='mm'):
	"""
	Generate a rectangular area to contain the neurons.
	
	Parameters
    ----------
    major_axis: double
		Length of the major axis (by default in mm).
	minor_axis: double
		Length of the minor axis (by default in mm).
	unit: string (default 'mm')
		Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'
	
	Returns
    -------
    AGNet.generation.Shape object
	"""

def custom_shape(shape, unit='mm'):
	"""
	Generate a custom-shaped area to contain the neurons.
	
	Parameters
    ----------
    shape: string (default 'mm')
		Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'
	
	Returns
    -------
    AGNet.generation.Shape object
	"""

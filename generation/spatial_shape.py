#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generating the shape of the neurons' environment """

import numpy as np



#
#---
# Shape class
#--------------------

class Shape:
	"""
	Class containing the shape of the area where neurons will be
	distributed to form a network.
	
	Attributes
    ----------
    area: double
		Area of the shape in mm^2.
	gravity_center: tuple of doubles
		Position of the center of gravity of the current shape.
	
	Methods
    -------
    add_subshape: void
		Add a AGNet.generation.Shape to a preexisting one.
	"""
	
	def __init__(self):
		self.__area = 0.
		self.__gravity_center = (0.,0.)
	
	def add_subshape(self,subshape,position,unit='mm'):
		"""
		Add a AGNet.generation.Shape to the current one.
		
		Parameters
		----------
		subshape: AGNet.generation.Shape
			Length of the rectangle (by default in mm).
		position: tuple of doubles
			Position of the subshape's center of gravity in space.
		unit: string (default 'mm')
			Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'
		
		Returns
		-------
		None
		"""
		

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

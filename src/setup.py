#!/usr/bin/env python
#-*- coding:utf-8 -*-

from setuptools import setup

setup(	name='nngt',
		version='0.2',
		description='Package to study growth and activity of neural networks',
		url='https://github.com/Silmathoron/NNGT',
		author='Tanguy Fardet',
		author_email='tanguy.fardet@univ-paris-diderot.fr',
		license='GNU',
		packages=[
		'nngt',
		'nngt.core',
		'nngt.lib',
		'nngt.generation',
		'nngt.nest'
		],
		zip_safe=False)

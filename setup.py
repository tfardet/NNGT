#!/usr/bin/env python
#-*- coding:utf-8 -*-

from setuptools import setup

setup(name='AGNet',
      version='0.1',
      description='Package to study growth and activity of neural networks',
      url='http://github.com/storborg/funniest',
      author='Tanguy Fardet',
      author_email='tanguy.fardet@univ-paris-diderot.fr',
      license='GNU',
      packages=['core', 'lib', 'nest'],
      zip_safe=False)

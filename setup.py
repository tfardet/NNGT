#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
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

import os
import sys
from setuptools import setup, Extension, find_packages
import numpy

try:
    from Cython.Build import cythonize
    import setuptools
    version = setuptools.__version__
    version = int(version[:version.index(".")])
    with_cython = True if version >= 18 else False
    from _cpp_header import clean_header
except ImportError:
    with_cython = False


# ----- #
# Paths #
# ----- #

omp_pos = sys.argv.index("--omp") if "--omp" in sys.argv else -1
omp_lib_dir = "/usr/lib" if omp_pos == -1 else sys.argv[omp_pos + 1]

dirname = os.path.abspath(__file__)[:-8]
dirname += ("/" if dirname[-1] != "/" else "") + "nngt/generation/"


# -------------------------------------- #
# Extension for multithreaded algorithms #
# -------------------------------------- #

ext = '.pyx' if with_cython else '.cpp'

extensions = Extension(
    "nngt.generation.cconnect", # name of extension
    sources = [dirname + "cconnect" + ext, dirname + "func_connect.cpp"],
    extra_compile_args = [
        "-O2", "-g", "-std=c++11", "-fopenmp", "-ftree-vectorize", "-msse",
        "-Wno-cpp", "-ffast-math", "-Wno-unused-function"
    ],
    language="c++",
    include_dirs=[dirname, numpy.get_include()],
    libraries = ['gomp'],
    library_dirs = [dirname, omp_lib_dir]
)

if with_cython:
    extensions = cythonize(extensions)
    clean_header(dirname + 'cconnect.cpp')
else:
    extensions = [extensions]


# ----- #
# Setup #
# ----- #

setup(
        name = 'nngt',
        version = '0.9.dev2',
        description = 'Package to study structure and activity in ' +\
                      'neuronal networks',

        package_dir = {'': '.'},
        packages = find_packages('.'),

        # Include the non python files:
        package_data = {'': [
            '*.txt', '*.rst', '*.md', '*.default', '*.pyx', '*.pxd', '*.cpp',
            '*.h', '*.pyxbld',
        ]},

        # Requirements
        install_requires = ['numpy', 'scipy>=0.11'],
        python_requires = '>=2.7, <4',
        extras_require = {
            'matplotlib': 'matplotlib',
            'PySide': ['PySide'],
            'PDF':  ["ReportLab>=1.2", "RXP"],
            'reST': ["docutils>=0.3"],
            'nx': ['networkx>=2.0'],
            'ig': ['python-igraph']
        },
        
        # Cython module
        ext_modules = extensions,

        # Metadata
        url = 'https://github.com/Silmathoron/NNGT',
        author = 'Tanguy Fardet',
        author_email = 'tanguy.fardet@univ-paris-diderot.fr',
        license = 'GPL3',
        keywords = 'neuronal network graph structure simulation NEST ' +\
                   'topology growth',
        long_description = 'NNGT provides a unified interface to use three ' +\
                           'of the main Python graph libraries ' +\
                           '(graph-tool, igraph, and networkx) in order ' +\
                           'to generate and study neuronal networks. It ' +\
                           'allows the user to easily send this graph to ' +\
                           'the NEST simulator, the analyze the resulting ' +\
                           'activity while taking structure into account.',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Natural Language :: English',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
            'Programming Language :: C++',
            'Programming Language :: Cython',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics'
        ],
)

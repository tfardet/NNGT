#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
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
import warnings
import traceback
import platform
import sysconfig

from setuptools import setup, Extension, find_packages
from distutils.command.build_ext import build_ext

import numpy

from nngt import __version__

try:
    from Cython.Build import cythonize
    import setuptools
    version = setuptools.__version__
    version = int(version[:version.index(".")])
    with_cython = (version >= 18)
except ImportError as e:
    with_cython = False


# ------------------ #
# Paths and platform #
# ------------------ #

# OS name: Linux/Darwin (Mac)/Windows
os_name = platform.system()

# OpenMP
omp_lib     = [] if os_name == "Windows" else ["gomp"]
omp_pos     = sys.argv.index("--omp") if "--omp" in sys.argv else -1
omp_lib_dir = "/usr/lib" if omp_pos == -1 else sys.argv[omp_pos + 1]

dirname = "."
dirname += ("/" if dirname[-1] != "/" else "") + "nngt/generation/"


# ------------------------ #
# Compiling OMP algorithms #
# ------------------------ #

# compiler options

copt =  {
    'msvc': ['/openmp', '/O2', '/fp:precise', '/permissive-', '/Zc:twoPhase-'],
    'unix': [
        '-std=c++11', '-Wno-cpp', '-Wno-unused-function', '-fopenmp',
        '-ffast-math', '-msse', '-ftree-vectorize', '-O2', '-g',
    ]
}

lopt =  {
    'unix': ['-fopenmp'],
    'clang': ['-fopenmp'],
}


class CustomBuildExt(build_ext):

    def build_extensions(self):
        c = os.environ.get('CC', None)
        if c is None:
            c = sysconfig.get_config_var('CC')
        if c is None:
            from distutils import ccompiler
            c = ccompiler.get_default_compiler()
        if "gcc" in c or "g++" in c or "mingw" in c:
            c = "unix"
        elif "msvc" in c:
            c = "msvc"

        for e in self.extensions:
            e.extra_link_args.extend(lopt.get(c, []))
            e.extra_compile_args.extend(copt.get(c, []))

        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
            self.compiler.compiler_so.remove("-O3")
        except:
            pass

        build_ext.build_extensions(self)


# cython extensions

ext = '.pyx' if with_cython else '.cpp'

extensions = Extension(
    "nngt.generation.cconnect", # name of extension
    sources = [dirname + "cconnect" + ext, dirname + "func_connect.cpp"],
    extra_compile_args = [],
    language="c++",
    include_dirs=[dirname, numpy.get_include()],
    libraries = omp_lib,
    library_dirs = [dirname, omp_lib_dir]
)


if with_cython:
    extensions = cythonize(extensions)
else:
    extensions = [extensions]


long_descr = '''
NNGT provides a unified interface to use three of the main Python graph
libraries (graph-tool, igraph, and networkx) in order to generate and study
neuronal networks. It allows the user to easily send this graph to the NEST
simulator, the analyze the resulting activity while taking structure into
account.
'''


# ----- #
# Setup #
# ----- #

setup_params = dict(
    name = 'nngt',
    version = __version__,
    description = 'Package to study structure and activity in ' +\
                  'neuronal networks',

    package_dir = {'': '.'},
    packages = find_packages('.'),
    include_package_data = True,

    cmdclass = {'build_ext': CustomBuildExt},

    # Include the non python files:
    package_data = {'': [
        '*.txt', '*.rst', '*.md', '*.default', '*.pyx', '*.pxd',
        'nngt/generation/func_connect.cpp',
        '*.h', '*.pyxbld',
    ]},

    # Requirements
    install_requires = ['numpy>=1.17', 'scipy>=0.11', 'cython'],
    python_requires = '>=3.5, <4',
    extras_require = {
        'matplotlib': 'matplotlib',
        'nx': ['networkx>=2.4'],
        'ig': ['python-igraph'],
        'geometry': ['matplotlib', 'shapely', 'dxfgrabber', 'svg.path'],
        'geospatial': ['matplotlib', 'geopandas', 'descartes', 'cartopy'],
        'full': ['networkx>=2.4', 'shapely', 'dxfgrabber', 'svg.path',
                 'matplotlib', 'geopandas', 'descartes', 'cartopy', 'lxml']
    },

    # Cython module
    ext_modules = extensions,

    # Metadata
    url = 'https://sr.ht/~tfardet/NNGT',
    author = 'Tanguy Fardet',
    author_email = 'tanguy.fardet@tuebingen.mpg.de',
    license = 'GPL3',
    keywords = 'network graph structure simulation neuron NEST DeNSE topology '
               'growth igraph graph-tool networkx geospatial',
    long_description = long_descr,
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Programming Language :: C++',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)


# Try to compile with multithreaded algorithms; if fails, pure python install

try:
    setup(**setup_params)
except (Exception, SystemExit) as e:
    sys.stderr.write(
        "Could not compile multithreading algorithms: {}\n".format(e))
    sys.stderr.write("Switching to pure python install.\n")
    sys.stderr.flush()

    setup_params["ext_modules"] = []
    setup(**setup_params)


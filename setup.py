#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
import setuptools
from setuptools import setup, Extension, find_packages
import numpy

try:
    from Cython.Build import cythonize
    with_cython = True if setuptools.__version__ > "18.0" else False
except ImportError:
    with_cython = False


#-----------------------------------------------------------------------------
# Paths
#------------------------
#

omp_pos = sys.argv.index("--omp") if "--omp" in sys.argv else -1
omp_lib_dir = "/usr/lib" if omp_pos == -1 else sys.argv[omp_pos + 1]

dirname = os.path.abspath(__file__)[:-8]
dirname += ("/" if dirname[-1] != "/" else "") + "nngt/generation/"


#-----------------------------------------------------------------------------
# Extension for multithreaded algorithms
#------------------------
#

ext = '.pyx' if with_cython else '.cpp'

extensions = Extension(
    "nngt.generation.cconnect", # name of extension
    sources = [dirname + "cconnect" + ext, dirname + "func_connect.cpp"],
    extra_compile_args=["-std=c++11", "-fopenmp"],
    extra_link_args=["-std=c++11"],
    language="c++",
    include_dirs=[dirname, numpy.get_include()],
    libraries = ['gomp'],
    library_dirs = [dirname, omp_lib_dir]
)

if with_cython:
    extensions = cythonize(extensions)


#-----------------------------------------------------------------------------
# Setup
#------------------------
#

setup(
        name='nngt',
        version = '0.5',
        description = 'Package to study growth and activity of neural networks',
        package_dir={'': '.'},
        packages = find_packages('.'),

        # Include the non python files:
        package_data = { '': ['*.txt', '*.rst', '*.md', '*.default'] },

        # Requirements
        install_requires = [ 'numpy', 'scipy>=0.11', 'matplotlib' ],
        extras_require = {
            'PySide': ['PySide'],
            'PDF':  ["ReportLab>=1.2", "RXP"],
            'reST': ["docutils>=0.3"],
            'nx': ['networkx'],
            'ig': ['python-igraph']
        },
        entry_points = {
            #@todo
            #~ 'console_scripts': [
                #~ 'rst2pdf = nngt.tools.pdfgen [PDF]',
                #~ 'rst2html = nngt.tools.htmlgen'
            #~ ],
            #~ 'gui_scripts': [ 'netgen = nngt.gui.main.__main__:main [PySide]' ]
        },
        
        # Cython module
        ext_modules = [extensions],

        # Metadata
        url = 'https://github.com/Silmathoron/NNGT',
        author = 'Tanguy Fardet',
        author_email = 'tanguy.fardet@univ-paris-diderot.fr',
        license = 'GNU',
        keywords = 'neural network graph simulation NEST topology growth'
)

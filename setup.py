#!/usr/bin/env python
#-*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
        name='nngt',
        version = '0.4a',
        description = 'Package to study growth and activity of neural networks',
        packages = find_packages('src'),
        
        # Include the non python files:
        package_data = { '': ['*.txt', '*.rst', '*.md', '*.plt'] },
        
        # Requirements
        install_requires = [ 'numpy', 'scipy', 'matplotlib' ],
        extras_require = {
            'pyside': ['PySide'],
            'PDF':  ["ReportLab>=1.2", "RXP"],
            'reST': ["docutils>=0.3"],
        },
        entry_points = {
            #@todo
            'console_scripts': [
                'rst2pdf = nngt.tools.pdfgen [PDF]',
                'rst2html = nngt.tools.htmlgen'
            ],
            'gui_scripts': [ 'netgen = nngt.gui.main.__main__:main [pyside]' ]
        },
        
        # Metadata
        url = 'https://github.com/Silmathoron/NNGT',
        author = 'Tanguy Fardet',
        author_email = 'tanguy.fardet@univ-paris-diderot.fr',
        license = 'GNU',
        keywords = 'neural network graph simulation NEST topology growth'
)

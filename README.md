# NNGT: a unified interface for networks in python

<img align="left" style="padding-top: 5px; margin-right: 15px;" src="doc/_static/nngt_logo.png" />

[![Build Status](https://travis-ci.org/Silmathoron/NNGT.svg?branch=master)](https://travis-ci.org/Silmathoron/NNGT) [![Documentation Status](https://readthedocs.org/projects/nngt/badge/?version=latest)](http://nngt.readthedocs.org/en/latest/?badge=latest) [![License](http://img.shields.io/:license-GPLv3+-yellow.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![Coverage Status](https://coveralls.io/repos/github/Silmathoron/NNGT/badge.svg?branch=master)](https://coveralls.io/github/Silmathoron/NNGT?branch=master)<br>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3402493.svg)](https://doi.org/10.5281/zenodo.3402493)  ![GitHub release (latest by date)](https://img.shields.io/github/v/release/Silmathoron/NNGT) ![PyPI](https://img.shields.io/pypi/v/nngt)

The Neural Network Growth and Topology (NNGT) module provides tools to grow and
study detailed biological networks by interfacing efficient graph libraries with
highly distributed activity simulators.


## Principle

NNGT provides a unified interface that acts as a wrapper for 3 major graph
libraries in Python: [networkx](https://networkx.github.io/),
[igraph](https://igraph.org/python/), and
[graph-tool](https://graph-tool.skewed.de/).

Use the same code, run it at home on the latest linux with graph-tool, then
on your collaborator's laptop with networkx on Windows, no changes required!

In addition to this common interface, NNGT provides additional tools and
methods to generate complex neuronal networks.
Once the networks are created, they can be seamlessly sent to the
[nest-simulator](https://nest-simulator.readthedocs.io/), which will generate
activity. This activity can then be analyzed together with the structure using
NNGT.

Eventually, NNGT is also able to import neuronal networks generated using the
[DeNSE](https://dense.readthedocs.io/) simulator for neuronal growth.


## Install and use the library

NNGT requires Python 3.5+ since version 2.0, and is directly available on Pypi.
To install it, make sure you have a valid Python installation, then do:

```
pip install --user nngt
```

To use it, once installed, open a Python terminal or script file and type

```python
import nngt
```

If you want to have the latest updates before they are released into a stable
version, you can install directly from ``master`` via:

```
pip install --user git+https://github.com:Silmathoron/NNGT.git@master
```


## Cloning/updating the repository

This repository includes the
[``PyNCultures``](https://github.com/SENeC-Initiative/PyNCulture) package from
the [SENeC](https://github.com/SENeC-Initiative/) initiative as its
``geometry`` module, using the
[``git submodule``](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
feature.
Thus, when cloning the repository, you must do:

    git clone https://github.com/Silmathoron/NNGT.git
    cd NNGT && git submodule init && git submodule update

To update your local repository, do:

    git pull
    git submodule update --remote --merge


## Features

**Compatibility**
* Currently supports graph-tool (> 2.22), igraph, and networkx (>= 2.4).
* Interactions with [NEST](https://nest-simulator.readthedocs.io/) and
  [DeNSE](https://dense.readthedocs.io/).

**Status**
* Standard functions and graph generation algorithms.
* Special methods for graph analysis on weighted directed networks.
* Full support for node and edge attributes.
* Extended I/O features as well as graphical representations.
* Advanced methods to design neuronal networks.
* Supports complex 2D structures with shapely.

See doc on ReadTheDocs: [![Documentation Status](https://readthedocs.org/projects/nngt/badge/?version=latest)](http://nngt.readthedocs.org/en/latest/?badge=latest)

# NNGT [![Build Status](https://travis-ci.org/Silmathoron/NNGT.svg?branch=master)](https://travis-ci.org/Silmathoron/NNGT) [![Coverage Status](https://coveralls.io/repos/github/Silmathoron/NNGT/badge.svg?branch=master)](https://coveralls.io/github/Silmathoron/NNGT?branch=master) [![Documentation Status](https://readthedocs.org/projects/nngt/badge/?version=latest)](http://nngt.readthedocs.org/en/latest/?badge=latest)

![NNGT logo](doc/_static/nngt_logo.png)

The Neural Network Growth and Topology (NNGT) module provides tools to grow and
study detailed biological networks by interfacing efficient graph libraries with
highly distributed activity simulators.


## Cloning the repository

This repository includes the
[``PyNCultures``](https://github.com/SENeC-Initiative/PyNCulture) package from
the [SENeC](https://github.com/SENeC-Initiative/) initiative as its
``geometry`` module, using the
[``git submodule``](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
feature.
Thus, when cloning the repository, you must do:

    git clone https://github.com/Silmathoron/NNGT.git
    cd NNGT && git submodule init && git submodule update


## Features

**Compatibility**
It currently supports NEST, graph-tool, igraph and networkx.

**Status**
Basic functions and graph generation algorithms implemented.

See doc on ReadTheDocs: [![Documentation Status](https://readthedocs.org/projects/nngt/badge/?version=latest)](http://nngt.readthedocs.org/en/latest/?badge=latest)

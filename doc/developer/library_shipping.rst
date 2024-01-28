..
    SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
    SPDX-License-Identifier: CC-BY-SA-4.0
    doc/developer/library_shipping.rst

================
Library shipping
================

Multiplatform usage:

* NNGT can be used in pure-python mode on any platform
* Compilation of the multithreading algorithms can also be done on all
  available platforms
* To simplify things, precompiled binaries for Linux (@todo MacOS/Windows)
  are provided directly on PyPi.


Moving to cibuildwheel
======================

Install https://cibuildwheel.readthedocs.io/en/stable/setup/


The manylinux wheels
====================

To prepare the ``manylinux`` wheels, one must use the docker containers
provided by https://github.com/pypa/manylinux ::

    podman pull quay.io/pypa/manylinux_2_28_x86_64

Then run ``extra/build_wheels.sh`` in it (you must be in the ``NNGT/`` folder)::

    podman run -v `pwd`:/io quay.io/pypa/manylinux_2_28_x86_64 /io/extra/build_wheels.sh

This links the current folder to the ``/io/`` folder in the container so a ``wheelhouse/``
folder will apear on the host system.


Pushing to PyPi
===============

https://twine.readthedocs.io/en/latest/

First test it: ::

    twine upload -r testpypi whhelhouse/NNGT*
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nngt

Then upload "for real" ::

    twine upload -r NNGT whhelhouse/NNGT*

with NNGT, testpypi, and the tokens configured in ~/.pypirc::

    [distutils]
    index-servers =
        NNGT
        testpypi

    [NNGT]
    repository = https://upload.pypi.org/legacy/
    username = __token__
    password =

    [testpypi]
    repository = https://test.pypi.org/legacy/
    username = __token__
    password =

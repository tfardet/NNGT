..
    SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
    SPDX-License-Identifier: CC-BY-SA-4.0
    doc/future/interoperability.rst

================
Interoperability
================

Graph generation
================

Look into CSA and the libraries they provide from Djurfeldt 2014.


Simulators
==========

Look into PyNN.


Synaptic weights
================

Make it so that synaptic weights always have the same effect: they lead to the
same peak value/integral in the membrane potential:

* for each neural population, compute a map with N bins (lin or logspace),
* make a function that computes the index in the map from the weight value,
* this means that before constructing the NEST network I have to build a fake
  network to compute the map, then reset the kernel.


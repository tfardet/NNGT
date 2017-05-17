==================
Speed optimisation
==================

Building NEST networks
======================

Currently, building networks via nngt is roughly 20 times slower than building
them with NEST because the generation of the network is done using NEST's
``DataConnect`` function, which is super slow...

In the future, I'll see whether:
    1. ``DataConnect`` can be sped up
    2. I can make use of the csa library to generate the network in NEST more
       efficiently


Building and studying graphs
============================

Use multigraph=True if you don't mind having multiple edges (which become
negligible for large graphs anyway): it leads to up to 10-fold speed increase.

When looping over edges, use a generator rather than a list.

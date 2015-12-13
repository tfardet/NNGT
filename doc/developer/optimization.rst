==================
Speed optimisation
==================

Currently, building networks via nngt is roughly 20 times slower than building them with nest because the generation of the network is done using nest's ``DataConnect`` function,which is super slow...

In the future, I'll see whether:
    1. ``DataConnect`` can be sped up
    2. I can make use of the csa library to generate the network in NEST more efficiently

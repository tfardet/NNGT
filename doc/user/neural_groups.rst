.. _neural_groups:

=============================
Neural groups and populations
=============================

One of the key features of NNGT is to enable users to group nodes (neurons) into
groups sharing common properties in order to facilitate the generation of
network, the analysis of its properties, or complex simulations with NEST.

.. contents::
   :local:


Creating groups
===============

Simple groups
-------------

Neural groups can be created easily through calls to :class:`nngt.NeuralGroup`.

>>> group = nngt.NeuralGroup()

creates a single empty group (nothing very interesting).

Minimally, any useful group requires at least neuron ids and a type (excitatory
or inhibitory) to be useful.

To create a useful group, one can therefore either just tell how many neurons it
should contain:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 38

or directly pass it a list of ids

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 44


More realistic groups
---------------------

When designing neuronal networks, one usually cares about their type (excitatory
or inhibitory for instance), their properties, etc.

By default, neural groups are created excitatory and the following lines are
therefore equivalent:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 57-58

To create an inhibotory group, the neural type must be set to -1:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 59

Moving towards really realistic groups to run simulation on NEST afterwards,
the last step is to associate a neuronal model and set the properties of these
neurons (and optionally give them names):

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 72-77


Populations
===========


Complex populations and metagroups
==================================
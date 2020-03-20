.. _neural_groups:

=============================
Neural groups and populations
=============================

One of the key features of NNGT is to enable users to group nodes (neurons)
into groups sharing common properties in order to facilitate the generation of
network, the analysis of its properties, or complex simulations with NEST_.

The complete example file containing the code discussed here, as well as
additional information on how to access :class:`~nngt.NeuralGroup` and
:class:`~nngt.NeuralPop` properties can be found there:
:source:`docs/examples/introduction_to_groups.py`.

**Contents**

.. contents::
   :local:


Neuronal groups
===============

Neuronal groups are entities containing neurons which share common properties.
Inside a population, a single neuron belongs to a single
:class:`~nngt.NeuralGroup` object. Conversely the union of all groups contains
all neurons in the network once and only once.

When creating a group, it is therefore important to make sure that it forms a
coherent set of neurons, as this will make network handling easier.

For more versatile grouping, where neurons can belong to multiple ensembles,
see the section about meta-groups below: `Complex populations and metagroups`_.


Creating simple groups
----------------------

Neural groups can be created easily through calls to :class:`nngt.NeuralGroup`.

>>> group = nngt.NeuralGroup()

creates a single empty group (nothing very interesting).

Minimally, any useful group requires at least neuron ids and a type (excitatory
or inhibitory) to be useful.

To create a useful group, one can therefore either just tell how many neurons
it should contain:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 40

or directly pass it a list of ids (to avoid typing ``nngt.`` all
the time, we do ``from nngt import NeuralGroup`` at the beginning)

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 46


More realistic groups
---------------------

When designing neuronal networks, one usually cares about their type
(excitatory or inhibitory for instance), their properties, etc.

By default, neural groups are created excitatory and the following lines are
therefore equivalent:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 59-60

To create an inhibitory group, the neural type must be set to -1:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 61

Moving towards really realistic groups to run simulation on NEST afterwards,
the last step is to associate a neuronal model and set the properties of these
neurons (and optionally give them names):

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 73-78


Populations
===========

Populations are ensembles of neuronal groups which describe all neurons in a
corresponding network.
They are usually created before the network and then used to generate
connections, but the can also be generated after the network creation, then
associated to it.


Simple populations
------------------

To create a population, you can start from scratch by creating an empty
population, then adding groups to it:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 86-88

NNGT also provides a two default routine to create simple populations:

* :func:`~nngt.NeuralPop.uniform`, to generate a single population where all
  neurons belong to the same group,
* :func:`~nngt.NeuralPop.exc_and_inhib`, to generate a mixed excitatory and
  inhibitory population.

As before, we do ``from nngt import NeuralPop`` to avoid typing ``nngt.`` all
the time.

To create such populations, just use:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 94-95

Eventually, a population can be created from exiting groups using
:func:`~nngt.NeuralPop.from_groups`:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 107-108

Note that, here, we pass ``with_models=False`` to the population because these
groups were created without the information necessary to create a network in
NEST_ (a valid neuron model).


NEST-enabled populations
------------------------

To create a NEST-enabled population, one can use one of the standard
classmethods (:func:`~nngt.NeuralPop.uniform` and
:func:`~nngt.NeuralPop.exc_and_inhib`) and pass it valid parameters for the
neuronal models (optionally also a synaptic model and neuronal/synaptic
parameters).

Otherwise, one can build the population from groups that already contain these
properties, e.g. the previous ``pyr`` and ``fsi`` groups:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 126-132

.. warning::
    `syn_spec` can contain any synaptic model and parameters associated to the
    NEST model; however, neither the synaptic weight nor the synaptic delay
    can be set there. For details on how to set synaptic weight and delays
    between groups, see :func:`~nngt.generation.connect_neural_groups`.

To see how to use a population to create a :class:`~nngt.Network` and send it
to NEST_, see :ref:`nest_net`.


Complex populations and metagroups
==================================

When building complex neuronal networks, it may be useful to have neurons
belong to multiple groups at the same time.
Because standard groups can contain a neuron only once, meta-groups were
introduced to provide this additional functionality.

Contrary to normal groups, a neuron can belong to any number of metagroups,
which allow to make various sub- or super-groups.
For instance, when modeling a part of cortex, neurons will belong to a layer,
and to a given cell class whithin that layer.
In that case, you may want to create specific groups for cell classes, like
``L3Py``, ``L5Py``, ``L3I``, ``L5I`` for layer 4 and 5 pyramidal cells as well
as interneurons, but you can then also group neurons in a same layer together,
and same with pyramidal neurons or interneurons.

First create the normal groups:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 144-158

Then make the metagroups for the layers:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 162-166

Create the population:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 169-170

Then add additional metagroups for cell types:

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 174-179

----

**Go to other tutorials:**

* :ref:`intro`
* :ref:`graph_gen`
* :ref:`multithread`
* :ref:`nest_int`
* :ref:`activ_analysis`
* :ref:`graph-prop`


.. links

.. _NEST: https://www.nest-simulator.org/

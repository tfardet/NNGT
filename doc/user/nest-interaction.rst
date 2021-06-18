.. _nest_int:

===================================
Interacting with the NEST simulator
===================================

This section details how to create detailed neuronal networks, then run
simulations on them using the NEST simulator.

Readers are supposed to have a good grasp of the way NEST handles neurons and
models, and how to create and setup NEST nodes.
If this is not the case, please see the `NEST user doc`_ and the
`PyNEST tutorials`_ first.

NNGT tools should work for NEST_ version 2 or 3; they can be separated into

* the structural tools (:class:`~nngt.Network`, :class:`~nngt.NeuralPop` ...)
  that are used to prepare the neuronal network and setup its properties and
  connectivity; these tools should be used **before**
* the :func:`~nngt.simulation.make_nest_network` and the associated,
  :func:`~nngt.Network.to_nest` functions that are used to send the previously
  prepared network to NEST;
* then, **after** using one of the previous functions, all the other functions
  contained in the :mod:`nngt.simulation` module can be used to add
  stimulations to the neurons or monitor them.


.. note::
    Calls to ``nest.ResetKernel`` will also reset all networks and populations,
    which means that after such a call, populations, parameters, etc, can again
    be changed until the next invocation of
    :func:`~nngt.simulation.make_nest_network` or :func:`~nngt.Network.to_nest`.


Example files associated to the interactions between NEST_ and NNGT can be
found here: :source:`docs/examples/nest_network.py` /
:source:`docs/examples/nest_receptor_ports.py`.


**Content:**

.. contents::
   :local:


Creating detailed neuronal networks
===================================

:class:`~nngt.NeuralPop` and :class:`~nngt.NeuralGroup`
-------------------------------------------------------

These two classes are the basic blocks to design neuronal networks: a
:class:`~nngt.NeuralGroup` is a set of neurons sharing common properties while
the :class:`~nngt.NeuralPop` is the main container that represents the whole
network as an ensemble of groups.

Depending on your perspective, you can either create the groups first, then
build the population from them, or create the population first, then split
it into various groups.

For more details on groups and populations, see :ref:`neural_groups`.

**Neuronal groups before the population**

Neural groups can be created as follow: ::

    # 100 inhibitory neurons
    basic_group = nngt.NeuralGroup(100, neuron_type=-1)
    # 10 excitatory (default) aeif neurons
    aeif_group  = nngt.NeuralGroup(10, neuron_model="aeif_psc_alpha")
    # an unspecified number of aeif neurons with specific parameters
    p = {"E_L": -58., "V_th": -54.}
    aeif_g2 = nngt.NeuralGroup(neuron_model="aeif_psc_alpha", neuron_param=p)

In the case where the number of neurons is specified upon creation, NNGT can
check that the number of neurons matches in the network and the associated
population and raise a warning if they don't. However, it is just a security
check and it does not prevent the network for being created if the numbers
don't match.

Once the groups are created, you can simply generate the population using ::

    pop = nngt.NeuralPop.from_groups([basic_group, aeif_group], ["b", "a"])

This created a population separated into "a" and "b" from the previously
created groups.

.. note :
    To be used in a populations, groups must be valid, i.e. have a size or be
    associated to neuron ids. E.g. ``aeif_g2`` is not valid yet and can
    therefore not be used to create ``pop``. It could however be associated
    to neuron ids using ``aeif_g2.ids``.

**Population before the groups**

A population with excitatory and inhibitory neurons ::

    pop = nngt.NeuralPop(1000)
    pop.create_group(800, "first")
    pop.create_group(200, "second", neuron_type=-1)

or, more compact ::

    pop = nngt.NeuralPop.exc_and_inhib(1000, iratio=0.2)


The :class:`~nngt.Network` class
--------------------------------

Besides connectivity, the main interest of the :class:`~nngt.NeuralGroup` is
that you can pass it the biological properties that the neurons belonging to
this group will share.

Since we are using NEST, these properties are:

* the model's name
* its non-default properties
* the synapses that the neurons have and their properties
* the type of the neurons (``1`` for excitatory or ``-1`` for inhibitory)

.. literalinclude:: ../examples/nest_network.py
   :lines: 29-53, 63-83

Once this network is created, it can simply be sent to nest through the
command: ``gids = net.to_nest()``, and the NEST gids are returned.

In order to access the gids from each group, you can do: ::

    oscill_gids = net.nest_gids[oscill.ids]

or directly::

    oscill_gids = oscill.nest_gids

As shown in ":ref:`nest_net`", synaptic strength from inhibitory neurons in
NNGT are positive (for compatibility with graph analysis tools) but they are
automatically converted to negative values when the network is created in NEST.


Changing the parameters of neurons
==================================

Before sending the network to NEST
----------------------------------

Once the :class:`~nngt.NeuralPop` has been created, you can change the
parameters of the neuron groups **before you send the network to NEST**.

To do this, you can use the :func:`~nngt.NeuralPop.set_param` function, to
which you pass the parameter dict and the name of the
:class:`~nngt.NeuralGroup` you want to modify.

If you are dealing directly with :class:`~nngt.NeuralGroup` objects, you can
access and modify their ``neuron_param`` attribute as long as the network has
not been sent to nest. Once sent, these parameters become unsettable and any
wourkaround to circumvent this will not change the values inside NEST anyway.

After sending the network to NEST, randomizing
----------------------------------------------

Once the network has been sent to NEST, neuronal parameters can still be
changed, but only for randomization purposes.
It is possible to randomize the neuronal parameters through the
:func:`~nngt.simulation.randomize_neural_states` function.
This sets the parameters using a specified distribution and stores their
values inside the network nodes' attributes.

----

**Go to other tutorials:**

* :ref:`intro`
* :ref:`graph_gen`
* :ref:`parallelism`
* :ref:`neural_groups`
* :ref:`activ_analysis`
* :ref:`graph-prop`


.. References

.. _`NEST user doc`: http://www.nest-simulator.org/documentation/
.. _`PyNEST tutorials`: http://www.nest-simulator.org/introduction-to-pynest/
.. _NEST: https://www.nest-simulator.org/

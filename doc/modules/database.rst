===============
Database module
===============

NNGT provides a database to store NEST simulations.
This database requires ``peewee>3`` to work and can be switched on using: ::

    nngt.set_config("use_database", True)

The commands are then used by calling ``nngt.database`` to access the database
tools.


.. contents::
   :local:


Functions
---------


.. autofunction:: nngt.database.get_results

.. autofunction:: nngt.database.is_clear

.. autofunction:: nngt.database.log_simulation_end

.. autofunction:: nngt.database.log_simulation_start

.. autofunction:: nngt.database.reset


Recording a simulation
----------------------

::

    nngt.database.log_simulation_start(net, "nest-2.14")
    nest.Simulate(1000.)
    nngt.database.log_simulation_end()


Checking results in the database
--------------------------------

The database contains the following tables, associated to their respective
fields:

* 'activity': :class:`~nngt.database.db_generation.Activity`,
* 'computer': :class:`~nngt.database.db_generation.Computer`,
* 'connection': :class:`~nngt.database.db_generation.Connection`,
* 'neuralnetwork': :class:`~nngt.database.db_generation.NeuralNetwork`,
* 'neuron': :class:`~nngt.database.db_generation.Neuron`,
* 'simulation': :class:`~nngt.database.db_generation.Simulation`,
* 'synapse': :class:`~nngt.database.db_generation.Synapse`.

These tables are the first keyword passed to :func:`~nngt.database.get_results`,
you can find the existing columns for each of the tables in the following
classes descriptions:

.. automodule:: nngt.database.db_generation

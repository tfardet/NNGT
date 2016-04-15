#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Store results into a database """

from collections import namedtuple

from peewee import *
from playhouse.csv_loader import load_csv, dump_csv
from playhouse.fields import PickledField, CompressedField
from playhouse.migrate import *
from playhouse.db_url import connect

from nngt import config


__all__ = [
    'Activity',
    'Computer',
    'Connection',
    'db_migrator',
    'ignore',
    'main_db',
    'migrate',
    'NeuralNetwork',
    'Neuron',
    'Simulation',
    'Synapse',
    'val_to_field',
]



#-----------------------------------------------------------------------------#
# Parse config file and generate database
#------------------------
#

main_db = connect(config['db_url'], fields={'longblob': 'longblob'}) #: Object refering to the database


#-----------------------------------------------------------------------------#
# Database classes
#------------------------
#

class LongCompressedField(CompressedField):
    db_field = 'longblob'

    
class BaseModel(Model):
    class Meta:
        database = main_db


class Computer(BaseModel):
    '''
    Class containing informations about the conputer.
    '''
    
    name = TextField()
    ''' : Name from ``platform.node()`` or ``"unknown"`` '''
    platform = TextField()
    ''' System information from ``platform.platform()`` '''
    python = TextField()
    ''' Python version given by ``platform.python_version()`` '''
    cores = IntegerField()
    ''' Number of cores returned by ``psutil.cpu_count()`` or ``-1`` '''
    ram = BigIntegerField()
    ''' Total memory given by ``psutil.virtual_memory().total`` (long) or
        ``-1`` '''


class NeuralNetwork(BaseModel):
    '''
    Class containing informations about the neural network.
    '''

    network_type = TextField()
    ''' Type of the network from Graph.type '''
    directed = BooleanField()
    ''' Whether the graph is directed or not '''
    nodes = IntegerField()
    ''' Number of nodes. '''
    edges = IntegerField()
    ''' Number of edges. '''
    weighted = BooleanField()
    ''' Whether the graph is weighted or not. '''
    weight_distribution = TextField(null=True)
    ''' Name of the weight_distribution used. '''
    compressed_file = LongCompressedField(null=True)
    ''' Compressed (bz2) string of the graph from ``str(graph)``; once
        uncompressed, can be loaded using ``Graph.from_file(name,
        from_string=True)``. '''


class Neuron(BaseModel):
    '''
    Base class that will be modified to contain all the properties of the
    neurons used during a simulation.
    '''
    pass


class Synapse(BaseModel):
    '''
    Base class that will be modified to contain all the properties of the
    synapses used during a simulation.
    '''
    pass
        

class Connection(BaseModel):
    '''
    Class detailing the existing connections in the network: a couple of pre-
    and post-synaptic neurons and a synapse.
    '''

    pre = ForeignKeyField(Neuron, null=True, related_name='out_connections')
    post = ForeignKeyField(Neuron, null=True, related_name='int_connections')
    synapse = ForeignKeyField(Synapse, null=True, related_name='connections')


class Activity(BaseModel):
    '''
    Class detailing the network's simulated activity.
    '''
    pass


class Simulation(BaseModel):
    '''
    Class containing all informations about the simulation properties.
    '''
    
    start_time = DateTimeField()
    ''' Date and time at which the simulation started. '''
    completion_time = DateTimeField()
    ''' Date and time at which the simulation ended. '''
    simulated_time = FloatField()
    ''' Virtual time that was simulated for the neural network. '''
    resolution = FloatField()
    ''' Timestep used to simulate the components of the neural network '''
    simulator = TextField()
    ''' Name of the neural simulator used (NEST, Brian...) '''
    grnd_seed = IntegerField(null=True)
    ''' Master seed of the simulation. '''
    local_seeds = PickledField(null=True)
    ''' List of the local threads seeds. '''
    computer = ForeignKeyField(Computer, related_name='simulations')
    ''' Computer table entry where the computer used is defined. '''
    network = ForeignKeyField(NeuralNetwork, related_name='simulations')
    ''' Network table entry where the simulated network is described. '''
    activity = ForeignKeyField(Activity, related_name='simulations')
    ''' Activity table entry where the simulated activity is described. '''
    population = PickledField()
    ''' Pickled list containing the neural group names. '''
    pop_sizes = PickledField()
    ''' Pickled list containing the group sizes. '''


#-----------------------------------------------------------------------------#
# Generate the custom Neuron and Synapse classes
#------------------------
#

ignore = {
    'global_id': True,
    'gsl_error_tol': True,
    'local_id': True,
    'recordables': True,
    'thread': True,
    'thread_local_id': True,
    'vp': True,
    'synaptic_elements': True,
    'sizeof': True,
    'source': True,
    'target': True,
}

val_to_field = {
    'int': IntegerField,
    'bigint': IntegerField,
    'tinyint': IntegerField,
    'long': PickledField,
    'blob': PickledField,
    'datetime': DateTimeField,
    'str': TextField,
    'longtext': TextField,
    'SLILiteral': TextField,
    'float': FloatField,
    'float64': FloatField,
    'float32': FloatField,
    'bool': BooleanField,
    'lst': PickledField,
    'dict': PickledField,
    'ndarray': PickledField,
    'compressed': LongCompressedField
}

db_migrator = {
    'SqliteDatabase': SqliteMigrator,
    'PostgresqlDatabase': PostgresqlMigrator,
    'MySQLDatabase': MySQLMigrator,
}

#-*- coding:utf-8 -*-
#
# core/neural_pop_group.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Graph data strctures in NNGT """

import logging
import weakref
from copy import deepcopy

import numpy as np

import nngt
from nngt.lib import (InvalidArgument, nonstring_container, is_integer,
                      default_neuron, default_synapse)
from nngt.lib._frozendict import _frozendict
from nngt.lib.logger import _log_message

from .group_structure import Structure, Group, MetaGroup


__all__ = [
    'GroupProperty',
    'MetaNeuralGroup',
    'NeuralGroup',
    'NeuralPop',
]

logger = logging.getLogger(__name__)

undefined = "undefined"


# --------- #
# NeuralPop #
# --------- #

class NeuralPop(Structure):

    """
    The basic class that contains groups of neurons and their properties.

    :ivar has_models: :obj:`bool`,
        ``True`` if every group has a ``model`` attribute.
    :ivar ~nngt.NeuralPop.size: :obj:`int`,
        Returns the number of neurons in the population.
    :ivar syn_spec: :obj:`dict`,
        Dictionary containing informations about the synapses between the
        different groups in the population.
    :ivar ~nngt.NeuralPop.is_valid: :obj:`bool`,
        Whether this population can be used to create a network in NEST.
    """

    # number of created populations
    __num_created = 0

    # store weakrefs to created populations
    __pops = weakref.WeakValueDictionary()

    #-------------------------------------------------------------------------#
    # Class attributes and methods

    @classmethod
    def from_network(cls, graph, *args):
        '''
        Make a NeuralPop object from a network. The groups of neurons are
        determined using instructions from an arbitrary number of
        :class:`~nngt.properties.GroupProperties`.
        '''
        return cls(parent=graph, graph=graph, group_prop=args)

    @classmethod
    def from_groups(cls, groups, names=None, syn_spec=None, parent=None,
                    meta_groups=None, with_models=True):
        '''
        Make a NeuralPop object from a (list of) :class:`~nngt.NeuralGroup`
        object(s).

        Parameters
        ----------
        groups : list of :class:`~nngt.NeuralGroup` objects
            Groups that will be used to form the population. Note that a given
            neuron can only belong to a single group, so the groups should form
            pairwise disjoints complementary sets.
        names : list of str, optional (default: None)
            Names that can be used as keys to retreive a specific group. If not
            provided, keys will be the group name (if not empty) or the position
            of the group in `groups`, stored as a string.
            In the latter case, the first group in a population named `pop`
            will be retreived by either `pop[0]` or `pop['0']`.
        parent : :class:`~nngt.Graph`, optional (default: None)
            Parent if the population is created from an exiting graph.
        syn_spec : dict, optional (default: static synapse)
            Dictionary containg a directed edge between groups as key and the
            associated synaptic parameters for the post-synaptic neurons (i.e.
            those of the second group) as value.
            If a 'default' entry is provided, all unspecified connections will
            be set to its value.
        meta_groups : list or dict of str/:class:`~nngt.NeuralGroup` items
            Additional set of groups which can overlap: a neuron can belong to
            several different meta groups. Contrary to the primary groups, meta
            groups do therefore no need to be disjoint.
            If all meta-groups have a name, they can be passed directly through
            a list; otherwise a dict is necessary.
        with_model : bool, optional (default: True)
            Whether the groups require models (set to False to use populations
            for graph theoretical purposes, without NEST interaction)

        Example
        -------
        For synaptic properties, if provided in `syn_spec`, all connections
        between groups will be set according to the values.
        Keys can be either group names or types (1 for excitatory, -1 for
        inhibitory). Because of this, several combination can be available for
        the connections between two groups. Because of this, priority is given
        to source (presynaptic properties), i.e. NNGT will look for the entry
        matching the first group name as source before looking for entries
        matching the second group name as target.

        .. code-block:: python

            # we created groups `g1`, `g2`, and `g3`
            prop = {
                ('g1', 'g2'): {'model': 'tsodyks2_synapse', 'tau_fac': 50.},
                ('g1', g3'): {'weight': 100.},
                ...
            }
            pop = NeuronalPop.from_groups(
                [g1, g2, g3], names=['g1', 'g2', 'g3'], syn_spec=prop)

        Note
        ----
        If the population is not generated from an existing
        :class:`~nngt.Graph` and the groups do not contain explicit ids, then
        the ids will be generated upon population creation: the first group, of
        size N0, will be associated the indices 0 to N0 - 1, the second group
        (size N1), will get N0 to N0 + N1 - 1, etc.
        '''
        if not nonstring_container(groups):
            groups = [groups]

        gsize = len(groups)
        names = [] if names is None else list(names)

        if not names:
            for i, g in enumerate(groups):
                if g.name:
                    names.append(g.name)
                else:
                    names.append(str(i))

        assert len(names) == gsize, "`names` and `groups` must have " +\
                                    "the same size."

        for n in names:
            assert isinstance(n, str), "Group names must be strings."

        if syn_spec:
            _check_syn_spec(syn_spec, names, groups)

        current_size = 0
        for g in groups:
            # generate the neuron ids if necessary
            ids = g.ids
            if len(ids) == 0:
                ids = list(range(current_size, current_size + g.size))
                g.ids = ids
            current_size += len(ids)

        pop = cls(current_size, parent=parent, meta_groups=meta_groups,
                  with_models=with_models)

        for name, g in zip(names, groups):
            pop[name] = g
            g._struct = weakref.ref(pop)
            g._net    = weakref.ref(parent) if parent is not None else None

        # take care of synaptic connections
        pop._syn_spec = deepcopy(syn_spec if syn_spec is not None else {})

        return pop

    @classmethod
    def uniform(cls, size, neuron_type=1, neuron_model=default_neuron,
                neuron_param=None, syn_model=default_synapse, syn_param=None,
                parent=None, meta_groups=None):
        '''
        Make a NeuralPop of identical neurons belonging to a single "default"
        group.

        Parameters
        ----------
        size : int
            Number of neurons in the population.
        neuron_type : int, optional (default: 1)
            Type of the neurons in the population: 1 for excitatory or -1 for
            inhibitory.
        neuron_model : str, optional (default: default neuron model)
            Neuronal model for the simulator.
        neuron_param : dict, optional (default: default neuron parameters)
            Parameters associated to `neuron_model`.
        syn_model : str, optional (default: default static synapse)
            Synapse model for the simulator.
        syn_param : dict, optional (default: default synaptic parameters)
            Parameters associated to `syn_model`.
        parent : :class:`~nngt.Graph` object, optional (default: None)
            Parent graph described by the population.
        meta_groups : list or dict of str/:class:`~nngt.NeuralGroup` items
            Set of groups which can overlap: a neuron can belong to
            several different meta groups, i.e. they do no need to be disjoint.
            If all meta-groups have a name, they can be passed directly through
            a list; otherwise a dict is necessary.
        '''
        neuron_param = {} if neuron_param is None else neuron_param.copy()

        if syn_param is not None:
            assert 'weight' not in syn_param, '`weight` cannot be set here.'
            assert 'delay' not in syn_param, '`delay` cannot be set here.'
            syn_param = syn_param.copy()
        else:
            syn_param = {}

        pop = cls(size, parent, meta_groups=meta_groups)
        pop.create_group(range(size), "default", neuron_type, neuron_model,
                         neuron_param)

        pop._syn_spec = {'model': syn_model}

        if syn_param is not None:
            pop._syn_spec.update(syn_param)

        return pop

    @classmethod
    def exc_and_inhib(cls, size, iratio=0.2, en_model=default_neuron,
                      en_param=None, in_model=default_neuron, in_param=None,
                      syn_spec=None, parent=None, meta_groups=None):
        '''
        Make a NeuralPop with a given ratio of inhibitory and excitatory
        neurons.

        Parameters
        ----------
        size : int
            Number of neurons contained by the population.
        iratio : float, optional (default: 0.2)
            Fraction of the neurons that will be inhibitory.
        en_model : str, optional (default: default_neuron)
            Name of the NEST model that will be used to describe excitatory
            neurons.
        en_param : dict, optional (default: default NEST parameters)
            Parameters of the excitatory neuron model.
        in_model : str, optional (default: default_neuron)
            Name of the NEST model that will be used to describe inhibitory
            neurons.
        in_param : dict, optional (default: default NEST parameters)
            Parameters of the inhibitory neuron model.
        syn_spec : dict, optional (default: static synapse)
            Dictionary containg a directed edge between groups as key and the
            associated synaptic parameters for the post-synaptic neurons (i.e.
            those of the second group) as value. If provided, all connections
            between groups will be set according to the values contained in
            `syn_spec`. Valid keys are:

            - `('excitatory', 'excitatory')`
            - `('excitatory', 'inhibitory')`
            - `('inhibitory', 'excitatory')`
            - `('inhibitory', 'inhibitory')`
        parent : :class:`~nngt.Network`, optional (default: None)
            Network associated to this population.
        meta_groups : list dict of str/:class:`~nngt.NeuralGroup` items
            Additional set of groups which can overlap: a neuron can belong to
            several different meta groups. Contrary to the primary 'excitatory'
            and 'inhibitory' groups, meta groups are therefore no necessarily
            disjoint.
            If all meta-groups have a name, they can be passed directly through
            a list; otherwise a dict is necessary.

        See also
        --------
        :func:`nest.Connect` for a description of the dict that can be passed
        as values for the `syn_spec` parameter.
        '''
        num_exc_neurons = int(size*(1-iratio))

        pop = cls(size, parent, meta_groups=meta_groups)

        pop.create_group(
            range(num_exc_neurons), "excitatory", neuron_type=1,
            neuron_model=en_model, neuron_param=en_param)
        pop.create_group(
            range(num_exc_neurons, size), "inhibitory", neuron_type=-1,
            neuron_model=in_model, neuron_param=in_param)

        if syn_spec:
            _check_syn_spec(
                syn_spec, ["excitatory", "inhibitory"], pop.values())
            pop._syn_spec = deepcopy(syn_spec)
        else:
            pop._syn_spec = {}

        return pop

    @classmethod
    def _nest_reset(cls):
        '''
        Reset the _to_nest bool and potential parent networks.
        '''
        for pop in cls.__pops.valuerefs():
            if pop() is not None:
                pop()._to_nest = False
                for g in pop().values():
                    g._to_nest = False
                if pop().parent is not None:
                    pop().parent._nest_gids = None

    #-------------------------------------------------------------------------#
    # Contructor and instance attributes

    def __init__(self, size=None, parent=None, meta_groups=None,
                 with_models=True, **kwargs):
        '''
        Initialize NeuralPop instance.

        Parameters
        ----------
        size : int, optional (default: 0)
            Number of neurons that the population will contain.
        parent : :class:`~nngt.Network`, optional (default: None)
            Network associated to this population.
        meta_groups : dict of str/:class:`~nngt.NeuralGroup` items
            Optional set of groups. Contrary to the primary groups which
            define the population and must be disjoint, meta groups can
            overlap: a neuron can belong to several different meta
            groups.
        with_models : :class:`bool`
            whether the population's groups contain models to use in NEST
        *args : items for OrderedDict parent
        **kwargs : :obj:`dict`

        Returns
        -------
        pop : :class:`~nngt.NeuralPop` object.
        '''
        super().__init__(size=size, parent=parent, meta_groups=meta_groups,
                         **kwargs)

        self._syn_spec = {}
        self._has_models = with_models

        # whether the network this population represents was sent to NEST
        self._to_nest = False

        # update class properties
        self.__id = self.__class__.__num_created
        self.__class__.__num_created += 1
        self.__class__.__pops[self.__id] = self

    def __reduce__(self):
        '''
        Overwrite this function to make NeuralPop pickable.
        OrderedDict.__reduce__ returns a 3 to 5 tuple:
        - the first is the class
        - the second is the init args in Py2, empty sequence in Py3
        - the third can be used to store attributes
        - the fourth is None and needs to stay None
        - the last must be kept unchanged: odict_iterator in Py3
        '''
        state    = super().__reduce__()
        newstate = (
            NeuralPop, state[1][:3] + (self._has_models,) + state[1][3:],
            state[2], state[3], state[4]
        )

        return newstate

    def __setitem__(self, key, value):
        if self._to_nest:
            raise RuntimeError("Populations items can no longer be modified "
                               "once the network has been sent to NEST!")
        super().__setitem__(key, value)

    def copy(self):
        '''
        Return a deep copy of the population.
        '''
        # copy groups and metagroups
        groups = {k: v.copy() for k, v in self.items()}
        metagroups = {k: v.copy() for k, v in self._meta_groups.items()}

        # generate new population
        copy = NeuralPop.from_groups(
            groups.values(), groups.keys(), syn_spec=self._syn_spec,
            parent=None, meta_groups=metagroups, with_models=self._has_models)

        return copy

    @property
    def nest_gids(self):
        '''
        Return the NEST gids of the nodes inside the population.
        '''
        gids = []

        for g in self.values():
            gids.extend(g.nest_gids)

        return gids

    @property
    def excitatory(self):
        '''
        Return the ids of all excitatory nodes inside the population.
        '''
        ids = []

        for g in self.values():
            if g.neuron_type == 1:
                ids.extend(g.ids)

        return ids

    @property
    def inhibitory(self):
        '''
        Return the ids of all inhibitory nodes inside the population.
        '''
        ids = []

        for g in self.values():
            if g.neuron_type == -1:
                ids.extend(g.ids)

        return ids

    @property
    def syn_spec(self):
        '''
        The properties of the synaptic connections between groups.
        Returns a :obj:`dict` containing tuples as keys and dicts of parameters
        as values.

        The keys are tuples containing the names of the groups in the
        population, with the projecting group first (presynaptic neurons) and
        the receiving group last (post-synaptic neurons).

        Example
        -------
        For a population of excitatory ("exc") and inhibitory ("inh") neurons.

        .. code-block:: python

            syn_spec = {
                ("exc", "exc"): {'model': 'stdp_synapse', 'weight': 2.5},
                ("exc", "inh"): {'model': 'static_synapse'},
                ("exc", "inh"): {'model': 'stdp_synapse', 'delay': 5.},
                ("inh", "inh"): {
                    'model': 'stdp_synapse', 'weight': 5.,
                    'delay': ('normal', 5., 2.)}
                }
            }
        '''
        return deepcopy(self._syn_spec)

    @syn_spec.setter
    def syn_spec(self, syn_spec):
        raise NotImplementedError('`syn_spec` is not settable yet.')

    @property
    def has_models(self):
        ''' Whether all groups have been assigned a neuronal model. '''
        return self._has_models

    #-------------------------------------------------------------------------#
    # Methods

    def create_group(self, neurons, name, neuron_type=1, neuron_model=None,
                     neuron_param=None, replace=False):
        '''
        Create a new group in the population.

        Parameters
        ----------
        neurons : int or array-like
            Desired number of neurons or list of the neurons indices.
        name : str
            Name of the group.
        neuron_type : int, optional (default: 1)
            Type of the neurons : 1 for excitatory, -1 for inhibitory.
        neuron_model : str, optional (default: None)
            Name of a neuron model in NEST.
        neuron_param : dict, optional (default: None)
            Parameters for `neuron_model` in the NEST simulator. If None,
            default parameters will be used.
        replace : bool, optional (default: False)
            Whether to override previous exiting meta group with same name.
        '''
        assert isinstance(name, str), "Group `name` must be a string."
        assert neuron_type in (-1, 1), "Valid neuron type must be -1 or 1."

        if self._to_nest:
            raise RuntimeError("Groups can no longer be created once the "
                               "network has been sent to NEST!")

        if name in self and not replace:
            raise KeyError("Group with name '" + name + "' already " +\
                           "exists. Use `replace=True` to overwrite it.")

        neuron_param = {} if neuron_param is None else neuron_param.copy()

        group = NeuralGroup(neurons, neuron_type=neuron_type,
                            neuron_model=neuron_model,
                            neuron_param=neuron_param, name=name)

        self[name] = group

    def create_meta_group(self, neurons, name, neuron_param=None,
                          replace=False):
        '''
        Create a new meta group and add it to the population.

        Parameters
        ----------
        neurons : int or array-like
            Desired number of neurons or list of the neurons indices.
        name : str
            Name of the group.
        neuron_type : int, optional (default: 1)
            Type of the neurons : 1 for excitatory, -1 for inhibitory.
        neuron_model : str, optional (default: None)
            Name of a neuron model in NEST.
        neuron_param : dict, optional (default: None)
            Parameters for `neuron_model` in the NEST simulator. If None,
            default parameters will be used.
        replace : bool, optional (default: False)
            Whether to override previous exiting meta group with same name.
        '''
        neuron_param = {} if neuron_param is None else neuron_param.copy()

        group = MetaNeuralGroup(neurons, name=name, neuron_param=neuron_param)

        self.add_meta_group(group, replace=replace)

        return group

    def set_model(self, model, group=None):
        '''
        Set the groups' models.

        Parameters
        ----------
        model : dict
            Dictionary containing the model type as key ("neuron" or "synapse")
            and the model name as value (e.g. {"neuron": "iaf_neuron"}).
        group : list of strings, optional (default: None)
            List of strings containing the names of the groups which models
            should be updated.

        Note
        ----
        By default, synapses are registered as "static_synapse"s in NEST;
        because of this, only the ``neuron_model`` attribute is checked by
        the ``has_models`` function: it will answer ``True`` if all groups
        have a 'non-None' ``neuron_model`` attribute.

        Warning
        -------
        No check is performed on the validity of the models, which means
        that errors will only be detected when building the graph in NEST.
        '''
        if self._to_nest:
            raise RuntimeError("Models cannot be changed after the network "
                               "has been sent to NEST!")
        if group is None:
            group = self.keys()
        try:
            for key, val in model.items():
                for name in group:
                    if key == "neuron":
                        self[name].neuron_model = val
                    elif key == "synapse":
                        self[name].syn_model = val
                    else:
                        raise ValueError(
                            "Model type {} is not valid; choose among 'neuron'"
                            " or 'synapse'.".format(key))
        except:
            if model is not None:
                raise InvalidArgument(
                    "Invalid model dict or group; see docstring.")

        b_has_models = True

        if model is None:
            b_has_models = False

        for group in self.values():
            b_has_models *= group.has_model

        self._has_models = b_has_models

    def set_neuron_param(self, params, neurons=None, group=None):
        '''
        Set the parameters of specific neurons or of a whole group.

        .. versionadded:: 1.0

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the neurons. Entries can be
            either a single number (same for all neurons) or a list (one entry
            per neuron).
        neurons : list of ints, optional (default: None)
            Ids of the neurons whose parameters should be modified.
        group : list of strings, optional (default: None)
            List of strings containing the names of the groups whose parameters
            should be updated. When modifying neurons from a single group, it
            is still usefull to specify the group name to speed up the pace.

        Note
        ----
        If both `neurons` and `group` are None, all neurons will be modified.

        Warning
        -------
        No check is performed on the validity of the parameters, which means
        that errors will only be detected when building the graph in NEST.
        '''
        if self._to_nest:
            raise RuntimeError("Parameters cannot be changed after the "
                               "network has been sent to NEST!")

        if neurons is not None:  # specific neuron ids
            groups = []
            # get the groups they could belong to
            if group is not None:
                if nonstring_container(group):
                    groups.extend((self[g] for g in group))
                else:
                    groups.append(self[group])
            else:
                groups.extend(self.values())
            # update the groups parameters
            for g in groups:
                idx = np.where(np.in1d(g.ids, neurons, assume_unique=True))[0]
                # set the properties of the nodes for each entry in params
                for k, v in params.items():
                    default = np.NaN
                    if k in g.neuron_param:
                        default = g.neuron_param[k]
                    elif nngt.get_config('with_nest'):
                        try:
                            import nest
                            try:
                                default = nest.GetDefaults(g.neuron_model, k)
                            except nest.NESTError:
                                pass
                        except ImportError:
                            pass
                    vv      = np.repeat(default, g.size)
                    vv[idx] = v
                    # update
                    g.neuron_param[k] = vv
        else:  # all neurons in one or several groups
            group = self.keys() if group is None else group
            if not nonstring_container(group):
                group = [group]
            start = 0
            for name in group:
                g = self[name]
                for k, v in params.items():
                    if nonstring_container(v):
                        g.neuron_param[k] = v[start:start+g.size]
                    else:
                        g.neuron_param[k] = v
                start += g.size

    def get_param(self, groups=None, neurons=None, element="neuron"):
        '''
        Return the `element` (neuron or synapse) parameters for neurons or
        groups of neurons in the population.

        Parameters
        ----------
        groups : ``str``, ``int`` or array-like, optional (default: ``None``)
            Names or numbers of the groups for which the neural properties
            should be returned.
        neurons : int or array-like, optional (default: ``None``)
            IDs of the neurons for which parameters should be returned.
        element : ``list`` of ``str``, optional (default: ``"neuron"``)
            Element for which the parameters should be returned (either
            ``"neuron"`` or ``"synapse"``).

        Returns
        -------
        param : ``list``
            List of all dictionaries with the elements' parameters.
        '''
        if neurons is not None:
            groups = self._neuron_group[neurons]
        elif groups is None:
            groups = tuple(self.keys())
        key = "neuron_param" if element == "neuron" else "syn_param"
        if isinstance(groups, (str, int, np.integer)):
            return self[groups].properties[key]
        else:
            param = []
            for group in groups:
                param.append(self[group].properties[key])
            return param

    def add_to_group(self, group_name, ids):
        '''
        Add neurons to a specific group.

        Parameters
        ----------
        group_name : str or int
            Name or index of the group.
        ids : list or 1D-array
            Neuron ids.
        '''
        if self._to_nest:
            raise RuntimeError("Groups cannot be changed after the "
                               "network has been sent to NEST!")
        super().add_to_group(group_name, ids)

    def _validity_check(self, name, group):
        if self._has_models and not group.has_model:
            raise AttributeError(
                "This NeuralPop requires group to have a model attribute that "
                "is not `None`; to disable this, use `set_model(None)` "
                "method on this NeuralPop instance or set `with_models` to "
                "False when creating it.")
        elif group.has_model and not self._has_models:
            _log_message(logger, "WARNING",
                         "This NeuralPop is not set to take models into "
                         "account; use the `set_model` method to change its "
                         "behaviour.")

        if group.neuron_type not in (-1, 1):
            raise AttributeError("Valid neuron type must be -1 or 1.")

        # check pairwise disjoint
        super()._validity_check(name, group)

    def _sent_to_nest(self):
        '''
        Signify to the population and its groups that the network was sent
        to NEST and that therefore properties and groups should no longer
        be modified.
        '''
        self._to_nest = True

        for g in self.values():
            g._to_nest = True


# ----------------------------- #
# NeuralGroup and GroupProperty #
# ----------------------------- #

class NeuralGroup(Group):

    """
    Class defining groups of neurons.

    Its main variables are:

    :ivar ~nngt.NeuralGroup.ids: :obj:`list` of :obj:`int`
        the ids of the neurons in this group.
    :ivar ~nngt.NeuralGroup.neuron_type: :obj:`int`
        the default is ``1`` for excitatory neurons; ``-1`` is for inhibitory
        neurons; meta-groups must have `neuron_type` set to ``None``
    :ivar ~nngt.NeuralGroup.neuron_model: str, optional (default: None)
        the name of the model to use when simulating the activity of this group
    :ivar ~nngt.NeuralGroup.neuron_param: dict, optional (default: {})
        the parameters to use (if they differ from the model's defaults)
    :ivar ~nngt.NeuralGroup.is_metagroup: :obj:`bool`
        whether the group is a meta-group or not (`neuron_type` is ``None``
        for meta-groups)

    Warning
    -------
    Equality between :class:`~nngt.properties.NeuralGroup`s only compares
    the  size and neuronal type, ``model`` and ``param`` attributes.
    This means that groups differing only by their ``ids`` will register as
    equal.
    """

    __num_created = 0

    def __new__(cls, nodes=None, neuron_type=undefined, neuron_model=None,
                neuron_param=None, name=None, **kwargs):
        # check neuron type for MetaGroup
        if neuron_type == undefined:
            neuron_type = 1 if cls == NeuralGroup else None

        metagroup = (neuron_type is None)

        kwargs["metagroup"] = metagroup

        obj = super().__new__(cls, nodes=nodes, name=name, **kwargs)

        if metagroup:
            obj.__class__ = nngt.MetaNeuralGroup

        return obj

    def __init__(self, nodes=None, neuron_type=1, neuron_model=None,
                 neuron_param=None, name=None, **kwargs):
        '''
        Calling the class creates a group of neurons.
        The default is an empty group but it is not a valid object for
        most use cases.

        Parameters
        ----------
        nodes : int or array-like, optional (default: None)
            Desired size of the group or, a posteriori, NNGT indices of the
            neurons in an existing graph.
        neuron_type : int, optional (default: 1)
            Type of the neurons (1 for excitatory, -1 for inhibitory) or None
            if not relevant (only allowed for metag roups).
        neuron_model : str, optional (default: None)
            NEST model for the neuron.
        neuron_param : dict, optional (default: model defaults)
            Dictionary containing the parameters associated to the NEST model.

        Returns
        -------
        A new :class:`~nngt.core.NeuralGroup` instance.
        '''
        super().__init__(nodes, **kwargs)

        assert neuron_type in (1, -1, None), \
            "`neuron_type` can either be 1 or -1."

        neuron_param = {} if neuron_param is None else neuron_param.copy()

        self._has_model = False if neuron_model is None else True
        self._neuron_model = neuron_model

        group_num  = NeuralGroup.__num_created + 1
        self._name = "Group {}".format(group_num) if name is None \
                                                  else name

        self._nest_gids = None
        self._neuron_param = neuron_param if self._has_model else {}
        self._neuron_type = neuron_type

        # whether the network this group belongs to was sent to NEST
        self._to_nest = False

        # parents
        self._struct = None
        self._net    = None

        NeuralGroup.__num_created += 1

    def __eq__ (self, other):
        if isinstance(other, NeuralGroup):
            same_size = self.size == other.size
            same_nmodel = ((self.neuron_model == other.neuron_model)
                           * (self.neuron_param == other.neuron_param))
            same_type = self.neuron_type == other.neuron_type

            return same_size*same_nmodel*same_type

        return False

    def __str__(self):
        return "NeuralGroup({}size={})".format(
            self._name + ": " if self._name else "", self.size)

    def _repr_pretty_(self, p, cycle):
        return p.text(str(self))

    def copy(self):
        '''
        Return a deep copy of the group.
        '''
        copy = NeuralGroup(nodes=self._ids, neuron_type=self._neuron_type,
                           neuron_model=self._neuron_model,
                           neuron_param=self._neuron_param, name=self._name)

        return copy

    @property
    def neuron_model(self):
        ''' Model that will be used to simulate the neurons of this group. '''
        return self._neuron_model

    @property
    def neuron_type(self):
        ''' Type of the neurons in the group (excitatory or inhibitory). '''
        return self._neuron_type

    @neuron_model.setter
    def neuron_model(self, value):
        if self._to_nest:
            raise RuntimeError("Models cannot be changed after the "
                               "network has been sent to NEST!")
        self._neuron_model = value
        self._has_model = False if value is None else True

    @property
    def neuron_param(self):
        ''' Parameters associated to the group's neurons. '''
        if self._to_nest:
            return _frozendict(self._neuron_param, message="Cannot set " +
                               "neuron params after the network has been " +
                               "sent to NEST!")
        else:
            return self._neuron_param

    @neuron_param.setter
    def neuron_param(self, value):
        if self._to_nest:
            raise RuntimeError("Parameters cannot be changed after the "
                               "network has been sent to NEST!")
        self._neuron_param = value

    @Group.ids.setter
    def ids(self, value):
        ''' Ids of the group's neurons. '''
        if self._to_nest:
            raise RuntimeError("Ids cannot be changed after the "
                               "network has been sent to NEST!")

        self._ids = value

    @property
    def nest_gids(self):
        ''' Global ids associated to the neurons in the NEST simulator. '''
        return self._nest_gids

    @property
    def has_model(self):
        ''' Whether this group have been given a model for the simulation. '''
        return self._has_model

    @property
    def properties(self):
        '''
        Properties of the neurons in this group, including `neuron_type`,
        `neuron_model` and `neuron_params`.
        '''
        dic = {
            "neuron_type": self.neuron_type,
            "neuron_model": self._neuron_model,
            "neuron_param": deepcopy(self._neuron_param)
        }

        return dic


class MetaNeuralGroup(MetaGroup, NeuralGroup):

    """
    Class defining a meta-group of neurons.

    Its main variables are:

    :ivar ~nngt.MetaGroup.ids: :obj:`list` of :obj:`int`
        the ids of the neurons in this group.
    :ivar ~nngt.MetaGroup.is_metagroup: :obj:`bool`
        whether the group is a meta-group or not (`neuron_type` is
        ``None`` for meta-groups)
    """

    def __init__(self, nodes=None, name=None, properties=None, **kwargs):
        '''
        Calling the class creates a group of neurons.
        The default is an empty group but it is not a valid object for
        most use cases.

        Parameters
        ----------
        nodes : int or array-like, optional (default: None)
            Desired size of the group or, a posteriori, NNGT indices of
            the neurons in an existing graph.
        name : str, optional (default: "Group N")
            Name of the meta-group.

        Returns
        -------
        A new :class:`~nngt.MetaNeuralGroup` object.
        '''
        kwargs["neuron_type"] = kwargs.get("neuron_type", None)

        super().__init__(nodes=nodes, name=name, properties=properties,
                         **kwargs)

    def __str__(self):
        return "MetaNeuralGroup({}size={})".format(
            self._name + ": " if self._name else "", self.size)

    @property
    def excitatory(self):
        '''
        Return the ids of all excitatory nodes inside the meta-group.
        '''
        if self.parent is not None:
            gtype = np.array(
                [g.neuron_type for g in self.parent.values()],
                dtype=int)

            ids = np.array(self.ids, dtype=int)

            parents = self.parent.get_group(ids, numbers=True)

            return ids[gtype[parents] == 1]

        return []

    @property
    def inhibitory(self):
        '''
        Return the ids of all inhibitory nodes inside the meta-group.
        '''
        if self.parent is not None:
            gtype = np.array(
                [g.neuron_type for g in self.parent.values()],
                dtype=int)

            ids = np.array(self.ids, dtype=int)

            parents = self.parent.get_group(ids, numbers=True)

            return ids[gtype[parents] == -1]

        return []

    @property
    def properties(self):
        return self._prop


class GroupProperty:

    """
    Class defining the properties needed to create groups of neurons from an
    existing :class:`~nngt.Graph` or one of its subclasses.

    :ivar ~nngt.GroupProperty.size: :obj:`int`
        Size of the group.
    :ivar constraints: :obj:`dict`, optional (default: {})
        Constraints to respect when building the
        :class:`~nngt.properties.NeuralGroup` .
    :ivar ~nngt.GroupProperty.neuron_model: str, optional (default: None)
        name of the model to use when simulating the activity of this group.
    :ivar ~nngt.GroupProperty.neuron_param: dict, optional (default: {})
        the parameters to use (if they differ from the model's defaults)
    """

    def __init__ (self, size, constraints={}, neuron_model=None,
                  neuron_param={}, syn_model=None, syn_param={}):
        '''
        Create a new instance of GroupProperties.

        Notes
        -----
        The constraints can be chosen among:
            - "avg_deg", "min_deg", "max_deg" (:class:`int`) to constrain the
              total degree of the nodes
            - "avg/min/max_in_deg", "avg/min/max_out_deg", to work with the
              in/out-degrees
            - "avg/min/max_betw" (:class:`double`) to constrain the betweenness
              centrality
            - "in_shape" (:class:`nngt.geometry.Shape`) to chose neurons inside
              a given spatial region

        Examples
        --------
        >>> di_constrain = { "avg_deg": 10, "min_betw": 0.001 }
        >>> group_prop = GroupProperties(200, constraints=di_constrain)
        '''
        self.size = size
        self.constraints = constraints
        self.neuron_model = neuron_model
        self.neuron_param = neuron_param
        self.syn_model = syn_model
        self.syn_param = syn_param


def _make_groups(graph, group_prop):
    '''
    Divide `graph` into groups using `group_prop`, a list of group properties
    @todo
    '''
    pass


# ----- #
# Tools #
# ----- #

def _check_syn_spec(syn_spec, group_names, groups):
    gsize = len(groups)
    # test if all types syn_spec are contained
    alltypes = set(((1, 1), (1, -1), (-1, 1), (-1, -1))).issubset(
        syn_spec.keys())
    # is there more than 1 type?
    types = list(set(g.neuron_type for g in groups))
    mt_type = len(types) > 1
    # check that only allowed entries are present
    edge_keys = []
    for k in syn_spec.keys():
        if isinstance(k, tuple):
            edge_keys.extend(k)
    edge_keys = set(edge_keys)
    allkeys = group_names + types
    assert edge_keys.issubset(allkeys), \
        '`syn_spec` edge entries can only be made from {}.'.format(allkeys)
    # warn if connections might be missing
    nspec = len(edge_keys)
    has_default = len(syn_spec) > nspec
    if mt_type and nspec < gsize**2 and not alltypes and not has_default:
        _log_message(
            logger, "WARNING",
            'There is not one synaptic specifier per inter-group'
            'connection in `syn_spec` and no default model was provided. '
            'Therefore, {} or 4 entries were expected but only {} were '
            'provided. It might be right, but make sure all cases are '
            'covered. Missing connections will be set as "static_'
            'synapse".'.format(gsize**2, nspec))
    for val in syn_spec.values():
        assert 'weight' not in val, '`weight` cannot be set here.'
        assert 'delay' not in val, '`delay` cannot be set here.'

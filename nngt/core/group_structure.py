#-*- coding:utf-8 -*-
#
# core/group_structure.py
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

from collections import OrderedDict
import logging
import weakref
from copy import deepcopy

import numpy as np

import nngt
from nngt.lib import InvalidArgument
from nngt.lib.test_functions import deprecated, is_integer, nonstring_container
from nngt.lib.logger import _log_message


__all__ = [
    'Group',
    'MetaGroup',
    'Structure',
]

logger = logging.getLogger(__name__)


# --------- #
# Structure #
# --------- #

class Structure(OrderedDict):

    """
    The basic class that contains groups of nodes and their properties.

    .. versionadded: 2.2

    :ivar ~nngt.Structure.ids: :obj:`lst`,
        Returns the ids of nodes in the structure.
    :ivar ~nngt.Structure.is_valid: :obj:`bool`,
        Whether the structure is consistent with its associated network.
    :ivar ~nngt.Structure.parent: :class:`~nngt.Network`,
        Parent network.
    :ivar ~nngt.Structure.size: :obj:`int`,
        Returns the number of nodes in the structure.
    """

    # number of created populations
    __num_created = 0

    # store weakrefs to created populations
    __structs = weakref.WeakValueDictionary()

    #-------------------------------------------------------------------------#
    # Class attributes and methods

    @classmethod
    def from_groups(cls, groups, names=None, parent=None, meta_groups=None):
        '''
        Make a :class:`~nngt.Structure` object from a (list of)
        :class:`~nngt.Group` object(s).

        Parameters
        ----------
        groups : dict or list of :class:`~nngt.Group` objects
            Groups that will be used to form the structure. Note that a given
            node can only belong to a single group, so the groups should form
            pairwise disjoints complementary sets.
        names : list of str, optional (default: None)
            Names that can be used as keys to retreive a specific group. If not
            provided, keys will be the group name (if not empty) or the position
            of the group in `groups`, stored as a string.
            In the latter case, the first group in a structure named `struct`
            will be retreived by either `struct[0]` or `struct['0']`.
        parent : :class:`~nngt.Graph`, optional (default: None)
            Parent if the structure is created from an exiting graph.
        meta_groups : list or dict of str/:class:`~nngt.Group` items
            Additional set of groups which can overlap: a node can belong to
            several different meta groups. Contrary to the primary groups, meta
            groups do therefore no need to be disjoint.
            If all meta-groups have a name, they can be passed directly through
            a list; otherwise a dict is necessary.

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

            # we already created groups `g1`, `g2`, and `g3`
            struct = Structure.from_groups([g1, g2, g3],
                                           names=['g1', 'g2', 'g3'])

        Note
        ----
        If the structure is not generated from an existing
        :class:`~nngt.Graph` and the groups do not contain explicit ids, then
        the ids will be generated upon structure creation: the first group, of
        size N0, will be associated the indices 0 to N0 - 1, the second group
        (size N1), will get N0 to N0 + N1 - 1, etc.
        '''
        if not nonstring_container(groups):
            groups = [groups]
        elif isinstance(groups, dict):
            names = list(groups) if names is None else names
            groups = list(groups.values())

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

        current_size = 0

        for g in groups:
            # generate the node ids if necessary
            ids = g.ids
            if len(ids) == 0:
                ids = list(range(current_size, current_size + g.size))
                g.ids = ids
            current_size += len(ids)

        struct = cls(current_size, parent=parent, meta_groups=meta_groups)

        for name, g in zip(names, groups):
            struct[name] = g
            g._struct    = weakref.ref(struct)
            g._net       = weakref.ref(parent) if parent is not None else None
    
        return struct

    #-------------------------------------------------------------------------#
    # Contructor and instance attributes

    def __init__(self, size=None, parent=None, meta_groups=None, **kwargs):
        '''
        Initialize Structure instance.

        Parameters
        ----------
        size : int, optional (default: 0)
            Number of nodes that the structure will contain.
        parent : :class:`~nngt.Network`, optional (default: None)
            Network associated to this structure.
        meta_groups : dict of str/:class:`~nngt.Group` items
            Optional set of groups. Contrary to the primary groups which
            define the structure and must be disjoint, meta groups can
            overlap: a neuron can belong to several different meta
            groups.
        **kwargs : :obj:`dict`

        Returns
        -------
        struct : :class:`~nngt.Structure` object.
        '''
        # check meta groups
        meta_groups = {} if meta_groups is None else meta_groups

        if not isinstance(meta_groups, dict):
            for g in meta_groups:
                if not g.name:
                    raise ValueError(
                        "When providing a list for `meta_groups`, "
                        "all meta groups should be named")
            meta_groups = {g.name: g for g in meta_groups}

        # set main properties
        self._is_valid = False
        self._desired_size = size if parent is None else parent.node_nb()
        self._size = 0
        self._parent = None if parent is None else weakref.ref(parent)
        self._meta_groups = {}

        # create `_groups`: an array containing the id of the group
        # associated to the index of each neuron, which 'maps' nodes to the
        # primary group they belong to
        if self._desired_size is None:
            self._groups = None
            self._max_id = 0
        else:
            self._groups = np.repeat(-1, self._desired_size)
            self._max_id = len(self._groups) - 1

        # add meta groups
        for nmg, mg in meta_groups.items():
            self.add_meta_group(mg, nmg)

        if parent is not None and 'group_prop' in kwargs:
            dic = _make_groups(parent, kwargs["group_prop"])
            self._is_valid = True
            self.update(dic)

        # init the OrderedDict
        super().__init__()

        # update class properties
        self.__id = self.__class__.__num_created
        self.__class__.__num_created += 1
        self.__class__.__structs[self.__id] = self

    def __reduce__(self):
        '''
        Overwrite this function to make Structure pickable.
        OrderedDict.__reduce__ returns a 3 to 5 tuple:
        - the first is the class
        - the second is the init args in Py2, empty sequence in Py3
        - the third can be used to store attributes
        - the fourth is None and needs to stay None
        - the last must be kept unchanged: odict_iterator in Py3
        '''
        state = super().__reduce__()
        last  = state[4] if len(state) == 5 else None
        dic   = state[2]
        args  = (dic.get("_size", None), dic.get("_parent", None),
                    dic.get("_meta_groups", {}))

        newstate = (Structure, args, dic, None, last)

        return newstate

    def __contains__(self, key):
        return super().__contains__(key) or key in self._meta_groups

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            assert key >= 0, "Index must be positive, not {}.".format(key)
            new_key = tuple(self.keys())[key]
            return OrderedDict.__getitem__(self, new_key)
        else:
            if key in self._meta_groups:
                return self._meta_groups[key]
            elif key in self:
                return OrderedDict.__getitem__(self, key)
            else:
                raise KeyError("Not (meta) group named '{}'.".format(key))

    def __setitem__(self, key, value):
        self._validity_check(key, value)

        int_key = None

        if is_integer(key):
            new_key = tuple(self.keys())[key]
            int_key = key
            OrderedDict.__setitem__(self, new_key, value)
        else:
            OrderedDict.__setitem__(self, key, value)
            int_key = list(super(Structure, self).keys()).index(key)

        # set name and parents
        value._name = key
        value._struct  = weakref.ref(self)
        value._net  = self._parent

        # update struct size/max_id
        group_size = len(value.ids)
        max_id     = np.max(value.ids) if group_size != 0 else 0

        _update_max_id_and_size(self, max_id)

        self._groups[value.ids] = int_key

        if -1 in list(self._groups):
            self._is_valid = False
        else:
            if self._desired_size is not None:
                self._is_valid = (self._desired_size == self._size)
            else:
                self._is_valid = True

    def copy(self):
        '''
        Return a deep copy of the structure.
        '''
        # copy groups and metagroups
        groups = {k: v.copy() for k, v in self.items()}
        metagroups = {k: v.copy() for k, v in self._meta_groups.items()}

        # generate new structure
        copy = Structure.from_groups(groups.values(), groups.keys(),
                                     parent=None, meta_groups=metagroups)

        return copy

    @property
    def size(self):
        '''
        Number of nodes in this structure.
        '''
        return self._size

    @property
    def ids(self):
        '''
        Return all the ids of the nodes inside the structure.
        '''
        ids = []

        for g in self.values():
            ids.extend(g.ids)

        return ids

    @property
    def parent(self):
        '''
        Parent :class:`~nngt.Network`, if it exists, otherwise ``None``.
        '''
        return None if self._parent is None else self._parent()

    @property
    def meta_groups(self):
        return self._meta_groups.copy()

    @property
    def is_valid(self):
        '''
        Whether the structure is consistent with the associated network.
        '''
        return self._is_valid

    #-------------------------------------------------------------------------#
    # Methods

    def create_group(self, nodes, name, properties=None, replace=False):
        '''
        Create a new group in the structure.

        Parameters
        ----------
        nodes : int or array-like
            Desired number of nodes or list of the nodes indices.
        name : str
            Name of the group.
        properties : dict, optional (default: None)
            Properties associated to the nodes in this group.
        replace : bool, optional (default: False)
            Whether to override previous exiting meta group with same name.
        '''
        assert isinstance(name, str), "Group `name` must be a string."

        if name in self and not replace:
            raise KeyError("Group with name '" + name + "' already " +\
                           "exists. Use `replace=True` to overwrite it.")

        properties = {} if properties is None else properties.copy()

        group = Group(nodes, properties=properties, name=name)

        self[name] = group

    def create_meta_group(self, nodes, name, properties=None, replace=False):
        '''
        Create a new meta group and add it to the structure.

        Parameters
        ----------
        nodes : int or array-like
            Desired number of nodes or list of the nodes indices.
        name : str
            Name of the group.
        properties : dict, optional (default: None)
            Properties associated to the nodes in this group.
        replace : bool, optional (default: False)
            Whether to override previous exiting meta group with same name.
        '''
        properties = {} if properties is None else properties.copy()

        group = MetaGroup(nodes, name=name, properties=properties)

        self.add_meta_group(group, replace=replace)

        return group

    def add_meta_group(self, group, name=None, replace=False):
        '''
        Add an existing meta group to the structure.

        Parameters
        ----------
        group : :class:`Group`
            Meta group.
        name : str, optional (default: group name)
            Name of the meta group.
        replace : bool, optional (default: False)
            Whether to override previous exiting meta group with same name.

        Note
        ----
        The name of the group is automatically updated to match the `name`
        argument.
        '''
        name = name if name else group.name

        if not name:
            raise ValueError("Group is not named, but no `name` entry was "
                             "provided.")

        if name in self._meta_groups and not replace:
            raise KeyError("Cannot add meta group with name '" + name +\
                           "': primary group with that name already exists.")

        if name in self._meta_groups and not replace:
            raise KeyError("Meta group with name '" + name + "' already " +\
                           "exists. Use `replace=True` to overwrite it.")

        if not group.is_metagroup:
            raise ValueError("`Group '" + group.name + "' is no meta-group.")

        # check that meta_groups are compatible with the structure size
        if group.ids:
            assert np.max(group.ids) <= self._max_id, \
                "The meta group contains ids larger than the structure size."

        group._name = name
        group._struct  = weakref.ref(self)
        group._net  = self._parent

        self._meta_groups[name] = group

    def set_properties(self, props, nodes=None, group=None):
        '''
        Set the parameters of specific nodes or of a whole group.

        .. versionadded:: 2.2

        Parameters
        ----------
        props : dict
            Dictionary containing parameters for the nodes. Entries can be
            either a single number (same for all nodes) or a list (one entry
            per nodes).
        nodes : list of ints, optional (default: None)
            Ids of the nodes whose parameters should be modified.
        group : list of strings, optional (default: None)
            List of strings containing the names of the groups whose parameters
            should be updated. When modifying nodes from a single group, it
            is still usefull to specify the group name to speed up the pace.

        Note
        ----
        If both `nodes` and `group` are None, all nodes will be modified.
        '''
        # specific neuron ids
        if nodes is not None:
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
                idx = np.where(np.in1d(g.ids, nodes, assume_unique=True))[0]

                # set the properties of the nodes for each entry in props
                for k, v in props.items():
                    default = np.NaN

                    if k in g.properties:
                        default = g.properties[k]

                    vv      = np.repeat(default, g.size)
                    vv[idx] = v

                    # update
                    g.properties[k] = vv
        else:  # all nodes in one or several groups
            group = self.keys() if group is None else group
            if not nonstring_container(group):
                group = [group]
            start = 0
            for name in group:
                g = self[name]
                for k, v in props.items():
                    if nonstring_container(v):
                        g.properties[k] = v[start:start+g.size]
                    else:
                        g.properties[k] = v
                start += g.size

    def get_properties(self, key=None, groups=None, nodes=None):
        '''
        Return the properties of nodes or groups of nodes in the structure.

        Parameters
        ----------
        groups : ``str``, ``int`` or array-like, optional (default: ``None``)
            Names or numbers of the groups for which the neural properties
            should be returned.
        nodes : int or array-like, optional (default: ``None``)
            IDs of the nodes for which parameters should be returned.

        Returns
        -------
        props : ``list``
            List of all dictionaries with properties.
        '''
        if nodes is not None:
            groups = self._groups[nodes]
        elif groups is None:
            groups = tuple(self.keys())

        if isinstance(groups, (str, int, np.integer)):
            if key is None:
                return self[groups].properties

            return self[groups].properties[key]
        else:
            props = []
            for group in groups:
                if key is None:
                    props.append(self[group].properties)
                else:
                    props.append(self[group].properties[key])
            return props

    def get_group(self, nodes, numbers=False):
        '''
        Return the group of the nodes.

        Parameters
        ----------
        nodes : int or array-like
            IDs of the nodes for which the group should be returned.
        numbers : bool, optional (default: False)
            Whether the group identifier should be returned as a number; if
            ``False``, the group names are returned.
        '''
        names = np.array(tuple(self.keys()), dtype=object)

        if numbers:
            return self._groups[nodes]
        else:
            if self._is_valid:
                return names[self._groups[nodes]]
            else:
                groups = []
                for i in self._groups[nodes]:
                    if i >= 0:
                        groups.append(names[i])
                    else:
                        groups.append(None)
                return groups

    def add_to_group(self, group_name, ids):
        '''
        Add nodes to a specific group.

        Parameters
        ----------
        group_name : str or int
            Name or index of the group.
        ids : list or 1D-array
            Node ids.
        '''
        idx = None

        if is_integer(group_name):
            assert 0 <= group_name < len(self), "Group index does not exist."
            idx = group_name
        else:
            idx = list(self.keys()).index(group_name)

        if ids:
            self[group_name]._ids.update(ids)

            # update number of nodes
            max_id = np.max(self[group_name].ids)
            _update_max_id_and_size(self, max_id)
            self._groups[np.array(ids)] = idx

        if -1 not in list(self._groups):
            self._is_valid = True

    def _validity_check(self, name, group):
        # check pairwise disjoint property for groups
        for n, g in self.items():
            assert set(g.ids).isdisjoint(group.ids), \
                "New group overlaps with existing group '{}'".format(n)


# ----------------------- #
# Group and GroupProperty #
# ----------------------- #

class Group:

    """
    Class defining groups of nodes.

    .. versionadded: 2.2

    Its main variables are:

    :ivar ~nngt.Group.ids: :obj:`list` of :obj:`int`
        the ids of the nodes in this group.
    :ivar ~nngt.Group.properties: dict, optional (default: {})
        properties associated to the nodes
    :ivar ~nngt.Group.is_metagroup: :obj:`bool`
        whether the group is a meta-group or not.

    Note
    ----
    A :class:`Group` contains a set of nodes that are unique;
    the size of the group is the number of unique nodes contained in the group.
    Passing non-unique nodes will automatically convert them to a unique set.

    Warning
    -------
    Equality between :class:`~nngt.properties.Group`s only compares
    the  size and ``properties`` attributes.
    This means that groups differing only by their ``ids`` will register as
    equal.
    """

    __num_created = 0

    def __new__(cls, nodes=None, properties=None, name=None, **kwargs):
        obj = super().__new__(cls)

        metagroup = \
            kwargs.get("metagroup", False) or issubclass(cls, MetaGroup)

        if metagroup:
            obj.__class__ = nngt.MetaGroup

        obj._metagroup = metagroup

        return obj

    def __init__(self, nodes=None, properties=None, name=None, **kwargs):
        '''
        Calling the class creates a group of nodes.
        The default is an empty group but it is not a valid object for
        most use cases.

        Parameters
        ----------
        nodes : int or array-like, optional (default: None)
            Desired size of the group or, a posteriori, NNGT indices of the
            nodes in an existing graph.
        properties : dict, optional (default: {})
            Dictionary containing the properties associated to the nodes.

        Returns
        -------
        A new :class:`~nngt.Group` instance.
        '''
        self._props = {} if properties is None else properties.copy()

        if nodes is None:
            self._desired_size = None
            self._ids = set()
        elif nonstring_container(nodes):
            self._desired_size = None
            self._ids = set(nodes)
        elif is_integer(nodes):
            self._desired_size = nodes
            self._ids = set()
        else:
            raise InvalidArgument('`nodes` must be either array-like or int.')

        group_num  = Group.__num_created + 1
        self._name = "Group {}".format(group_num) if name is None \
                                                  else name

        # parents
        self._struct = None
        self._net = None

        Group.__num_created += 1

    def __eq__(self, other):
        if isinstance(other, Group):
            same_size = self.size == other.size
            same_prop = (self.properties == other.properties)

            return same_size*same_prop

        return False

    def __len__(self):
        return len(self.ids)

    def __str__(self):
        return "Group({}size={})".format(
            self._name + ": " if self._name else "", self.size)

    def _repr_pretty_(self, p, cycle):
        return p.text(str(self))

    def copy(self):
        '''
        Return a deep copy of the group.
        '''
        copy = Group(nodes=self._ids, properties=self._props, name=self._name)

        return copy

    def add_nodes(self, nodes):
        '''
        Add nodes to the group.

        Parameters
        ----------
        nodes : list of ids
        '''
        if not nonstring_container(nodes):
            raise ValueError("`nodes` must be a list of ids.")

        parent = self.parent

        if parent is not None:
            parent.add_to_group(self.name, nodes)
        else:
            self._ids.update(nodes)

    @property
    def name(self):
        ''' The name of the group. '''
        return self._name

    @property
    def parent(self):
        '''
        Return the parent :class:`~nngt.Structure` of the group
        '''
        if self._struct is not None:
            return self._struct()

        return None

    @property
    def size(self):
        ''' The (desired) number of nodes in the group. '''
        if self._desired_size is not None:
            return self._desired_size

        return len(self._ids)

    @property
    def ids(self):
        ''' Ids of the nodes belonging to the group. '''
        return list(self._ids)

    @ids.setter
    def ids(self, value):
        data = set(value)

        if self._desired_size is not None and self._desired_size != len(data):
            _log_message(logger, "WARNING",
                         'The number of unique `ids` passed is not the same '
                         'as the initial size that was declared: {} before '
                         'vs {} now. Setting `ids` anyway, but check your '
                         'code!'.format(self._desired_size, len(value)))
        self._ids = data
        self._desired_size = None

    @property
    @deprecated("2.2", reason="it is not useful", removal="3.0")
    def is_valid(self):
        '''
        Whether the group can be used in a structure: i.e. if it has
        either a size or some ids associated to it.
        '''
        return True

    @property
    def is_metagroup(self):
        ''' Whether the group is a meta-group. '''
        return self._metagroup

    @property
    def properties(self):
        ''' Properties associated to the nodes in the group. '''
        return self._props


class MetaGroup(Group):

    """
    Class defining a meta-group of nodes.

    Its main variables are:

    :ivar ~nngt.MetaGroup.ids: :obj:`list` of :obj:`int`
        the ids of the nodes in this group.
    """

    __num_created = 0

    def __init__(self, nodes=None, name=None, **kwargs):
        '''
        Calling the class creates a group of nodes.
        The default is an empty group but it is not a valid object for
        most use cases.

        Parameters
        ----------
        nodes : int or array-like, optional (default: None)
            Desired size of the group or, a posteriori, NNGT indices of
            the nodes in an existing graph.
        name : str, optional (default: "Group N")
            Name of the meta-group.

        Returns
        -------
        A new :class:`~nngt.MetaGroup` object.
        '''
        group_num = MetaGroup.__num_created + 1
        name = "MetaGroup {}".format(group_num) if name is None \
                                                else name

        super().__init__(nodes=nodes, name=name, **kwargs)

        MetaGroup.__num_created += 1

    def __str__(self):
        return "MetaGroup({}size={})".format(
            self._name + ": " if self._name else "", self.size)


# ----- #
# Tools #
# ----- #

def _update_max_id_and_size(neural_pop, max_id):
    '''
    Update Structure after modification of a Group ids.
    '''
    old_max_id = neural_pop._max_id

    neural_pop._max_id = max(neural_pop._max_id, max_id)

    # update size
    neural_pop._size = 0

    for g in neural_pop.values():
        neural_pop._size += g.size

    # update the group node property
    if neural_pop._groups is None:
        neural_pop._groups = np.repeat(-1, neural_pop._max_id + 1)
    elif neural_pop._max_id >= len(neural_pop._groups):
        ngroup_tmp = np.repeat(-1, neural_pop._max_id + 1)
        ngroup_tmp[:old_max_id + 1] = neural_pop._groups
        neural_pop._groups = ngroup_tmp

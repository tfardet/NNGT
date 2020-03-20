#!/usr/bin/env python
#-*- coding:utf-8 -*-

# test_attributes.py

"""
Test the population and group functions.
"""

import numpy as np

import nngt


def test_groups():
    ids = [i for i in range(10)]
    g1 = nngt.NeuralGroup(ids, neuron_type=None)

    assert ids == g1.ids
    assert len(ids) == len(g1)
    assert g1.neuron_model is None
    assert g1.neuron_type is None
    assert not g1.has_model
    assert g1.is_valid
    assert g1.is_metagroup

    g2 = nngt.NeuralGroup(ids, neuron_type=None)

    assert g1 == g2
    assert g1.name.startswith("MetaGroup ")
    assert int(g1.name[9:]) + 1 == int(g2.name[9:])

    g3 = nngt.NeuralGroup(ids, neuron_type=1, name="test")
    assert g1 != g3
    assert g3.name == "test"


def test_population():
    # basic population
    ids1 = [i for i in range(10)]
    g1   = nngt.NeuralGroup(ids1, neuron_type=1)

    ids2 = [i for i in range(10, 20)]
    g2   = nngt.NeuralGroup(ids2, neuron_type=-1)

    pop = nngt.NeuralPop.from_groups((g1, g2), ("g1", "g2"), with_models=False)

    assert "g1" in pop and "g2" in pop
    assert set(ids1 + ids2) == set(pop.ids)
    assert pop.inhibitory == ids2
    assert pop.excitatory == ids1
    assert len(pop) == 2
    assert pop.size == len(ids1) + len(ids2)
    assert pop["g1"].ids == ids1

    #add meta groups
    g3  = nngt.NeuralGroup(ids1[::2] + ids2[::2], neuron_type=None)
    pop.add_meta_group(g3)

    g4 = pop.create_meta_group([i for i in range(pop.size) if i % 3],
                               "mg2")

    assert g3 in pop.meta_groups.values()
    assert g4 in pop.meta_groups.values()

    assert set(g2.ids).issuperset(g3.inhibitory)
    assert set(g1.ids).issuperset(g3.excitatory)

    assert set(g2.ids).issuperset(g4.inhibitory)
    assert set(g1.ids).issuperset(g4.excitatory)

    # build population from empty one
    pop = nngt.NeuralPop()
    g5  = pop.create_group([i for i in range(10)], "group5",
                           neuron_model="none")
    g2.neuron_model = "none"
    pop["group2"] = g2

    pop.set_model({"neuron": "random_model"})

    assert pop.is_valid

    # basic populations
    pop = nngt.NeuralPop.uniform(100)

    assert len(pop) == 1
    assert pop.size == 100

    pop = nngt.NeuralPop.exc_and_inhib(100)

    assert len(pop) == 2
    assert len(pop["excitatory"].ids) == 80
    assert pop["excitatory"].ids == pop.excitatory
    assert len(pop["inhibitory"].ids) == 20
    assert pop["inhibitory"].ids == pop.inhibitory


def test_failed_pop():
    ids1 = [i for i in range(10)]
    g1   = nngt.NeuralGroup(ids1, neuron_type=None)

    ids2 = [i for i in range(10, 20)]
    g2   = nngt.NeuralGroup(ids2, neuron_type=None)

    failed = True

    try:
        pop = nngt.NeuralPop.from_groups((g1, g2), ("g1", "g2"))
        failed = False
    except:
        pass

    try:
        pop = nngt.NeuralPop.from_groups((g1, g2), ("g1", "g2"),
                                         with_models=False)
        failed = False
    except:
        pass

    assert failed


if __name__ == "__main__":
    test_groups()
    test_population()
    test_failed_pop()

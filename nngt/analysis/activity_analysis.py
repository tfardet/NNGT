#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Tools for activity analysis from data """

import numpy as np


def _b2_from_data(ids, data):
    b2 = np.zeros(len(ids))
    for i, neuron in enumerate(ids):
        ids = np.where(data[0] == neuron)[0]
        dt1 = np.diff(data[1][ids])
        dt2 = dt1[1:] + dt1[:-1]
        avg_isi = np.mean(dt1)
        if avg_isi != 0.:
            b2[i] = (2*np.var(dt1) - np.var(dt2)) / (2*avg_isi**2)
        else:
            b2[i] = np.inf
    return b2


def _fr_from_data(ids, data):
    fr = np.zeros(len(ids))
    T = float(np.max(data[1]) - np.min(data[1]))
    for i, neuron in enumerate(ids):
        ids = np.where(data[0] == neuron)[0]
        fr[i] = len(ids) / T
    return fr

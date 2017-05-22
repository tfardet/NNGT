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

""" Fill autosummary entries """

import importlib
import inspect


def gen_autosum(source, target, module):
    mod = importlib.import_module(module)
    mod_dir = dir(mod)
    str_autosum = ''
    for member in mod_dir:
        if not member.startswith('_'):
            m = getattr(mod, member)
            if inspect.isclass(m) or inspect.isfunction(m):
                str_autosum += '   ' + module + '.' + member + '\n'
    with open(source, "r") as rst_input:
        with open(target, "w") as main_rst:
            for line in rst_input:
                if line.find("@autosum@") != -1:
                    main_rst.write(str_autosum)
                else:
                    main_rst.write(line)

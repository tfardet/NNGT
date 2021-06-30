#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
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

import types
import importlib
import inspect


def gen_autosum(source, target, module, autotype, dtype="all", ignore=None):
    '''
    Automatically write a sphinx-parsable file, adding a list of functions or
    classes to the autosummary method of sphinx, in place of the @autosum@
    keyword.

    Parameters
    ----------
    source : str
        Name of the input file, usually of the form "source.rst.in".
    target : str
        Name of the output file, usually "source.rst".
    module : str
        Name of the module on which autosummary should be performed.
    autotype : str
        Type of summary (normal for all, 'autofunction' or 'autoclass').
        Use 'autoall' for both functions and classes.
        Use 'full' to get both summary and detailed list.
    dtype : str, optional (default: all)
        Type of object that should be kept ('func' or 'class'), depending on
        `autotype`.
    ignore : list, optional (default: None)
        Names of the objects that should not be included in the summary.
    '''
    # load module and get content
    try:
        mod = importlib.import_module(module)
        mod_dir = dir(mod)
    except:
        # for the main classes
        import nngt

        lastdot = module.rfind(".")
        modname, objname = module[:lastdot], module[lastdot + 1:]
        mod = getattr(importlib.import_module(modname), objname)

        mod_dir = set(dir(mod))

        if objname != "Graph" and issubclass(mod, nngt.Graph):
            mod_dir = mod_dir.difference(dir(nngt.Graph))
        elif objname != "Structure" and issubclass(mod, nngt.Structure):
            mod_dir = mod_dir.difference(dir(nngt.Structure))
        elif objname != "Group" and issubclass(mod, nngt.Group):
            mod_dir = mod_dir.difference(dir(nngt.Group))

        mod_dir = sorted(mod_dir)

    # set ignored classes
    ignore = [] if ignore is None else ignore

    # list classes and functions
    str_autosum = ''
    str_autoref = ('\nDetails\n'
                   '=======\n\n'
                   '.. module:: ' + module + '\n\n')

    for member in mod_dir:
        if not (member.startswith('_') or member in ignore):
            m = getattr(mod, member)
            keep = 1

            isfunc   = inspect.isfunction(m)
            isclass  = inspect.isclass(m)
            ismember = isinstance(m, (property, types.MethodType))

            currenttype = None

            if autotype in ("autofunction", "autoclass"):
                currenttype = autotype
            elif isfunc:
                currenttype = 'autofunction'
            elif isclass:
                currenttype = 'autoclass'

            if dtype == "func":
                keep *= isfunc
            elif dtype == "class":
                keep *= isclass
            elif dtype == "classmembers":
                keep *= ismember + isfunc
            else:
                keep *= isfunc + isclass

            if keep:
                if autotype in ("summary", "full"):
                    str_autosum += '    ' + module + '.' + member + '\n'
                else:
                    str_autosum += '\n.. {}:: {}\n'.format(currenttype, member)

                    if currenttype == 'autoclass':
                        str_autosum += '    :members:\n'

                if autotype == "full":
                    str_autoref += '\n.. {}:: {}\n'.format(currenttype, member)

                    if currenttype == 'autoclass':
                        str_autosum += '    :members:\n'

    # write to file
    done = False

    str_final = ''

    with open(source, "r") as rst_input:
            for line in rst_input:
                if line.find("@autosum@") != -1 and not done:
                    str_final += str_autosum + "\n"

                    if autotype == "full":
                        str_final += str_autoref + '\n'

                    done = True
                else:
                    str_final += line

    with open(target, "w") as main_rst:
        main_rst.write(str_final)

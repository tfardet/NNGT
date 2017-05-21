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

""" Logging for the NGT module """

import os
from datetime import date
import logging

import nngt


# create logger

logger = logging.getLogger('nngt')
logger.setLevel(logging.DEBUG)


# check that log folder exists, otherwise create it

nngt._config["log_folder"] = os.path.expanduser(nngt._config["log_folder"])
if not os.path.isdir(nngt._config["log_folder"]):
    os.mkdir(nngt._config["log_folder"])


# configure logger

logFileFormatter = logging.Formatter(
    '[%(levelname)s @ %(name)s] %(asctime)s:\n\t%(message)s')
logConsoleFormatter = logging.Formatter(
    '[%(levelname)s @ %(name)s]: %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logConsoleFormatter)
consoleHandler.setLevel(nngt._config["log_level"])
logger.addHandler(consoleHandler)


# initialize file log

if nngt._config["log_to_file"]:
    today = date.today()
    fileName = "/nngt_{}-{}-{}".format(today.month, today.day, today.year)
    fileHandler = logging.FileHandler(
        "{}/{}.log".format(nngt._config["log_folder"], fileName))
    fileHandler.setFormatter(logFileFormatter)
    fileHandler.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)

# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# extra/check_headers.py

"""
Header generation for all python files

NB: this file must be run from inside the NNGT folder via
``python extra/format_headers.py``.
"""

import datetime
import os.path as op
import os
import re
import sys

from format_headers import ignore


# check current working directory
cwd = op.abspath(op.curdir)

assert cwd.endswith("NNGT"), \
    "Script must be called via `python extra/format_headers.py`."


def check_headers(filetype, header, year, ignore=None):
    ''' Check header files '''
    ignore = ignore or "$^"

    for (dirpath, dirnames, filenames) in os.walk(cwd):
        for f in filenames:
            abspath = op.join(dirpath, f)

            if f.endswith(filetype) and not re.search(ignore, abspath):
                with open(abspath, "r") as rf:
                    start = dirpath.find("NNGT")
                    start += 6 if dirpath.endswith("/") else 5

                    filename = op.join(dirpath[start:], f)

                    ref = header.format(filename=filename, year=year)
                    data = rf.read()

                    if data[:len(ref)] != ref:
                        print(f"Incorrect header for {op.join(dirpath, f)}.")
                        sys.exit(1)


''' Python headers '''


# load header string
headerfile = os.path.join(cwd, "extra/header.txt")

with open(headerfile, "r") as h:
    header = h.read()

# get year
year = datetime.datetime.now().year

# check_headers
check_headers(".py", header, year, ignore)


''' Documentation headers '''

# load header string
headerfile = os.path.join(cwd, "extra/header_rst.txt")

with open(headerfile, "r") as h:
    header = h.read()

check_headers(".rst.in", header, year)
check_headers(".rst", header, year)

sys.exit(0)

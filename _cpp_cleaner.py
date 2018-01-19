#!/usr/bin/env python
#-*- coding:utf-8 -*-

# Solution to clean C++ comments based on:
# https://stackoverflow.com/a/18234680/5962321 by Menno Rubingh, adapted from
# code byMarkus Jarderot and atikat.

''' Remove cython generated comments from _pygrowth.cpp '''

import re


def replacer(match):
    ''' Replace comment by single newline '''
    s = match.group(0)
    if s.startswith('/'):
        # Matched string is //...EOL or /*...*/  ==> keep only one newline
        return "\n"
    else:
        # Matched string is '...' or "..."  ==> Keep unchanged
        return s


def rm_cpp_comment(text):
    ''' Remove the C++ comments from a string '''
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def clean_cpp(filename):
    ''' Clean the cython generated code (header/comments) '''
    with open(filename, "r+") as f:
        output = rm_cpp_comment("".join([line for line in f]))
        f.seek(0)
        f.write(output)
        f.truncate()

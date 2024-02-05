# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: CC0-1.0
# setup.py

import os
import os.path as op
import platform
import re
import sys
import sysconfig
import tempfile

from distutils.errors import CompileError
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

import numpy as np

from Cython.Build import cythonize


# ------------------ #
# Paths and platform #
# ------------------ #

# OS name: Linux/Darwin (Mac)/Windows
os_name = platform.system()

# OpenMP
omp_lib = [] if os_name == "Windows" else ["gomp"]
omp_pos = sys.argv.index("--omp") if "--omp" in sys.argv else -1
omp_lib_dir = "/usr/lib" if omp_pos == -1 else sys.argv[omp_pos + 1]

dirname = os.path.join(".", "nngt/generation/")


# ------------------------ #
# Compiling OMP algorithms #
# ------------------------ #

# compiler options

copt =  {
    'msvc': ['/openmp', '/O2', '/fp:precise', '/permissive-', '/Zc:twoPhase-'],
    'gcc': [
        '-std=c++14', '-Wno-cpp', '-Wno-unused-function', '-ffast-math',
        '-msse', '-ftree-vectorize', '-O2', '-g', '-fopenmp'
    ],
    'clang': [
        '-std=c++14', '-Wno-cpp', '-Wno-unused-function', '-ffast-math',
        '-msse', '-ftree-vectorize', '-O2', '-g',
    ]
}

lopt =  {
    'gcc': ['-fopenmp'],
    'clang': [],
}

# check whether compiler supports a flag
def has_flag(compiler, flagname):
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except CompileError:
            return False
    return True


extensions = Extension(
    "nngt.generation.cconnect",
    sources=[
        os.path.join(dirname, "cconnect.pyx"),
        os.path.join(dirname, "func_connect.cpp")
    ],
    extra_compile_args=[],
    language="c++",
    include_dirs=[dirname, np.get_include()],
    libraries=omp_lib,
    library_dirs=[dirname, omp_lib_dir]
)


class build_ext(_build_ext):

    def initialize_options(self):
        super().initialize_options()

        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []

    def build_extensions(self):
        # only Unix compilers and their ports have `compiler_so`
        c = getattr(self.compiler, 'compiler_so', None)

        if c is None:
            c = os.environ.get('CC', os.environ.get('CXX'))
        else:
            c = c[0]

        if c is None:
            # if we really don't get anything, we're probably on windows
            c = sysconfig.get_config_var('CC') or self.compiler.compiler_type

        if re.match(r"gcc|g\+\+|mingw", c):
            c = "gcc"
        if "clang" in c:
            c = "clang"

            # macos clang compiler is bad
            if has_flag(self.compiler, "-fopenmp"):
                copt["clang"].append("-fopenmp")
                lopt["clang"].append("-fopenmp")
        elif "msvc" in c:
            c = "msvc"

        for e in self.distribution.ext_modules:
            e.extra_link_args.extend(lopt.get(c, []))
            e.extra_compile_args.extend(copt.get(c, []))

        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except Exception:
            pass

        try:
            self.compiler.compiler_so.remove("-O3")
        except Exception:
            pass

        super().build_extensions()


if __name__ == "__main__":
    try:
        extensions = cythonize(extensions)
    except Exception:
        extensions = []

    setup(cmdclass={"build_ext": build_ext}, ext_modules=extensions)

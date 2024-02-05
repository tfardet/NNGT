# SPDX-FileCopyrightText: 2024 Tanguy Fardet
# SPDX-License-Identifier: CC0-1.0
#!/bin/bash
set -e -x

# cleanup
rm -f /io/wheelhouse/NNGT*

# get NNGT
git clone https://git.sr.ht/~tfardet/NNGT
cd NNGT
git submodule init
git submodule update

cd ..

# Compile wheels
for PYBIN in /opt/python/cp3[^6]*/bin; do
    "${PYBIN}/pip" install -r NNGT/requirements.txt
    "${PYBIN}/pip" wheel NNGT/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/NNGT*.whl; do
    auditwheel repair "$whl" -w io/wheelhouse/
done

# Compile sdist
/opt/python/cp312-cp312/bin/pip install build
/opt/python/cp312-cp312/bin/python -m build --sdist NNGT/ -o /io/wheelhouse/

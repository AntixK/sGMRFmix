#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

/opt/python/cp37-cp37m/bin/pip install twine cmake
ln -s /opt/python/cp37-cp37m/bin/cmake /usr/bin/cmake


# Install a system package required by our library

# Compile wheels
for PYBIN in /opt/python/cp3*/bin; do
    VER=$(echo $PYBIN| cut -b 16)   # Get the python version
    if [ $VER -gt 6 ]               # Build only for python verion greater than 3.6
    then
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/pip" wheel /io/ -w wheelhouse/
        "${PYBIN}/python" /io/setup.py sdist -d /io/wheelhouse/
    fi
    
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/cp3*/bin/; do
    VER=$(echo $PYBIN| cut -b 16)   # Get the python version
    if [ $VER -gt 6 ]
    then
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/pip" install --no-index -f /io/wheelhouse sgmrfmix
        (cd "$PYHOME"; "${PYBIN}/unittest" /io/tests/)
    fi
done
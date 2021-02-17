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

# /opt/python/cp37-cp37m/bin/pip install twine cmake
# ln -s /opt/python/cp37-cp37m/bin/cmake /usr/bin/cmake


# Install a system package required by our library
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    apt-get update -qq
    apt-get install -qq libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev 
    apt-get install -qq libboost-all-dev build-essential

    # if [[ -f /etc/lsb-release ]];then
    #     apt-get update -qq
    #     apt-get install -qq libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev 
    #     apt-get install -qq libboost-all-dev build-essential
    # elif [[ -f /etc/centos-release ]] || [[ -f /etc/redhat-release ]]; then
    #     yum -y install blas-devel lapack-devel boost-devel  
    # fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install boost
fi

# Compile wheels
for PYBIN in /opt/python/cp3*/bin; do
    VER=$(echo $PYBIN| cut -b 16)   # Get the python version
    if [ $VER -gt 5 ]               # Build only for python verion greater than 3.6
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
    if [ $VER -gt 5 ]
    then
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/pip" install --no-index -f /io/wheelhouse sgmrfmix
        (cd "$PYHOME"; "${PYBIN}/unittest" /io/tests/)
    fi
done
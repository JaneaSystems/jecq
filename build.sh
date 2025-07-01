#!/bin/bash
# Copyright (c) 2025 Janea Systems
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


set -e 
set -o pipefail

echo "Building and installing JECQ from source on $(uname -a)..."

pushd .

# Determine build configuration
buildType=Release
useLTO=ON
if [ "$1" == "Debug" ]; then
    useLTO=OFF
    buildType=Debug
fi
echo "Inferred build type: $buildType"

# Clean up any previous builds
rm -rf ./build_linux
mkdir ./build_linux

cd build_linux
pushd .

# Setup virtual environment
echo "Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../jecq/python/requirements.txt
popd

# Set up Intel oneAPI environment
if [[ "$SETVARS_COMPLETED" != "1" ]]
then
    echo "Running oneapi/setvars.sh..."
    . /opt/intel/oneapi/setvars.sh
else
    echo "Not running oneapi/setvars.sh; already run"
fi

# Build
echo "Running CMake and make step..."

cmake ../ -DCMAKE_BUILD_TYPE=$buildType \
-DFAISS_USE_LTO=$useLTO -DFAISS_OPT_LEVEL=avx2 \
-DBLA_VENDOR=Intel10_64lp -DMKL_LINK=static \
"-DMKL_LIBRARIES=-Wl,--start-group;libiomp5.a;${MKLROOT}/lib/intel64/libmkl_intel_lp64.a;${MKLROOT}/lib/intel64/libmkl_gnu_thread.a;${MKLROOT}/lib/intel64/libmkl_core.a;-Wl,--end-group" \
-DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=ON -DJECQ_ENABLE_PYTHON=ON

make -j 8

# Run tests
echo "Running tests..."
if [ -d ./tests ]; then
    echo "Running C++ tests..."
    if [ -f ./tests/jecq_test ]; then
        echo "Running JECQ tests..."
        ./tests/jecq_test
    else
        echo "No JECQ tests found, skipping."
    fi
else
    echo "No tests directory found, skipping C++ tests."
fi

# Install Python package
echo "Installing Python package..."
pushd .
cd ./jecq/python
# Make sure wheel is installed
pip install wheel
python3 setup.py install
pip wheel --no-binary jecq .
python3 -c "import jecq;jecq.IndexJecq();jecq.IndexIVFJecq()"
python3 -c "import jecq;assert jecq.IndexJecq.__module__ == 'jecq.swigjecq_avx2'"
popd

# Update wheels
echo "Updating wheels..."
shopt -s globstar
# Create wheels directory if it doesn't exist
mkdir -p ../wheels/linux
cp ./**/jecq*.whl ../wheels/linux/

popd

#!/usr/bin/env bash
set -euo pipefail

# Create & enter build directory
mkdir -p build
cd build

# Configure CMake, pointing at your conda env & pybind11
cmake .. \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -Dpybind11_DIR="$CONDA_PREFIX/lib/python3.13/site-packages/pybind11/share/cmake/pybind11"

# Build with all available cores
cmake --build . -j$(nproc)


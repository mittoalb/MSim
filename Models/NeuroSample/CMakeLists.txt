cmake_minimum_required(VERSION 3.14)
project(brain LANGUAGES CXX)

# ——————————————————————————————————————————————————————————————————————
# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# default to Release if no build type was specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ——————————————————————————————————————————————————————————————————————
# Find pybind11 (pip-installed or system)
# Optionally override via -Dpybind11_DIR=<path-to-pybind11/share/cmake/pybind11>
find_package(pybind11 CONFIG REQUIRED)

# Find the Python interpreter & dev libraries
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# ——————————————————————————————————————————————————————————————————————
# Source files
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# ——————————————————————————————————————————————————————————————————————
# Build the Python extension; name must match your PYBIND11_MODULE(...)
pybind11_add_module(brain ${SOURCES})

# Ensure the .so is output into the project root (next to your scripts)
set_target_properties(brain PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Expose our public headers (vessels.h, cells.h, cball.h)
target_include_directories(brain
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Link explicitly against the Python library
target_link_libraries(brain
    PRIVATE
        Python::Python
)

# ——————————————————————————————————————————————————————————————————————
# Optional install step
include(GNUInstallDirs)
install(TARGETS brain
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)


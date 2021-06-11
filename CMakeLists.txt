cmake_minimum_required (VERSION 3.0 FATAL_ERROR)
project(uunet LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Choosing compiler (clang has preference).
if (CMAKE_CXX_COMPILER MATCHES ".*clang.*" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set (CMAKE_COMPILER_IS_CLANGXX 1)
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fno-inline -std=c++14")

if (APPLE)
    set(CMAKE_INSTALL_RPATH @loader_path)
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

add_subdirectory(ext/pybind11)
add_subdirectory(ext/uunet)

file (GLOB_RECURSE SOURCES C++/*cpp)

pybind11_add_module(_multinet MODULE ${SOURCES})
target_link_libraries(_multinet PRIVATE uunet)

target_include_directories (_multinet PRIVATE
ext/uunet/ext/eigen3
ext/uunet/ext/spectra
ext/uunet/ext/eclat/eclat/src
ext/uunet/ext/eclat/tract/src
ext/uunet/ext/eclat/math/src
ext/uunet/ext/eclat/util/src
ext/uunet/ext/eclat/apriori/src
ext/uunet/ext/infomap
ext/uunet/src
ext/uunet/libs)

add_definitions(-DNS_INFOMAP -DONLY_C_LOCALE=1)

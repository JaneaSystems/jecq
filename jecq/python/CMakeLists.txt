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

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(pyjecq
  DESCRIPTION "Python bindings for jecq."
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

macro(configure_swigjecq source)
  set_source_files_properties(${source} PROPERTIES
    CPLUSPLUS ON
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWORDSIZE64
    )
  endif()
  if(WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWIN
    )
  endif()
endmacro()

# CMake's SWIG wrappers only allow tweaking certain settings at source level, so
# we duplicate the source in order to override the module name.
configure_file(swigjecq.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigjecq_avx2.swig COPYONLY)
configure_file(swigjecq.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigjecq_avx512.swig COPYONLY)
configure_file(swigjecq.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigjecq_avx512_spr.swig COPYONLY)
configure_file(swigjecq.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigjecq_sve.swig COPYONLY)

configure_swigjecq(swigjecq.swig)
configure_swigjecq(swigjecq_avx2.swig)
configure_swigjecq(swigjecq_avx512.swig)
configure_swigjecq(swigjecq_avx512_spr.swig)
configure_swigjecq(swigjecq_sve.swig)

if(NOT TARGET faiss)
  find_package(faiss REQUIRED)
endif()

if(TARGET jecq)
  # Manually add headers as extra dependencies of swigjecq.
  set(SWIG_MODULE_swigjecq_EXTRA_DEPS)
  foreach(h ${FAISS_HEADERS})
    list(APPEND SWIG_MODULE_swigjecq_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigjecq_avx2_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigjecq_avx512_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigfjecq_avx512_spr_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigjecq_sve_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
  endforeach()
else()
  find_package(jecq REQUIRED)
endif()

foreach(h ${JECQ_HEADERS})
list(APPEND SWIG_MODULE_swigjecq_EXTRA_DEPS
  "${h}")
endforeach()

if (${CMAKE_SYSTEM_NAME} MATCHES "AIX")
swig_add_library(swigjecq
  TYPE MODULE
  LANGUAGE python
  SOURCES swigjecq.swig
)
else ()
swig_add_library(swigjecq
  TYPE SHARED
  LANGUAGE python
  SOURCES swigjecq.swig
)
endif()

set_property(TARGET swigjecq PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

set_property(SOURCE swigjecq_avx2.swig
  PROPERTY SWIG_MODULE_NAME swigjecq_avx2)
swig_add_library(swigjecq_avx2
  TYPE SHARED
  LANGUAGE python
  SOURCES swigjecq_avx2.swig
)
set_property(TARGET swigjecq_avx2 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2")
  set_target_properties(swigjecq_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swigjecq_avx512.swig
  PROPERTY SWIG_MODULE_NAME swigjecq_avx512)
swig_add_library(swigjecq_avx512
  TYPE SHARED
  LANGUAGE python
  SOURCES swigjecq_avx512.swig
)
set_property(TARGET swigjecq_avx512 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512")
  set_target_properties(swigjecq_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swigjecq_avx512_spr.swig
  PROPERTY SWIG_MODULE_NAME swigjecq_avx512_spr)
swig_add_library(swigjecq_avx512_spr
  TYPE SHARED
  LANGUAGE python
  SOURCES swigjecq_avx512_spr.swig
)
set_property(TARGET swigjecq_avx512_spr PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(swigjecq_avx512_spr PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swigjecq_sve.swig
  PROPERTY SWIG_MODULE_NAME swigjecq_sve)
swig_add_library(swigjecq_sve
  TYPE SHARED
  LANGUAGE python
  SOURCES swigjecq_sve.swig
)
set_property(TARGET swigjecq_sve PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "sve")
  set_target_properties(swigjecq_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

if(NOT WIN32)
  # NOTE: Python does not recognize the dylib extension.
  set_target_properties(swigjecq PROPERTIES SUFFIX .so)
  set_target_properties(swigjecq_avx2 PROPERTIES SUFFIX .so)
  set_target_properties(swigjecq_avx512 PROPERTIES SUFFIX .so)
  set_target_properties(swigjecq_avx512_spr PROPERTIES SUFFIX .so)
  set_target_properties(swigjecq_sve PROPERTIES SUFFIX .so)
else()
  # we need bigobj for the swig wrapper
  target_compile_options(swigjecq PRIVATE /bigobj)
  target_compile_options(swigjecq_avx2 PRIVATE /bigobj)
  target_compile_options(swigjecq_avx512 PRIVATE /bigobj)
  target_compile_options(swigjecq_avx512_spr PRIVATE /bigobj)
  target_compile_options(swigjecq_sve PRIVATE /bigobj)
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(swigjecq PRIVATE
  jecq
  faiss
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigjecq_avx2 PRIVATE
  faiss_avx2
  jecq_avx2
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigjecq_avx512 PRIVATE
  faiss_avx512
  jecq_avx512
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigjecq_avx512_spr PRIVATE
  faiss_avx512_spr
  jecq_avx512_spr
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigjecq_sve PRIVATE
  faiss_sve
  jecq_sve
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(swigjecq PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigjecq PRIVATE ${PROJECT_SOURCE_DIR}/../../faiss)
target_include_directories(swigjecq_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigjecq_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/../../faiss)
target_include_directories(swigjecq_avx512 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigjecq_avx512 PRIVATE ${PROJECT_SOURCE_DIR}/../../faiss)
target_include_directories(swigjecq_avx512_spr PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigjecq_avx512_spr PRIVATE ${PROJECT_SOURCE_DIR}/../../faiss)
target_include_directories(swigjecq_sve PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigjecq_sve PRIVATE ${PROJECT_SOURCE_DIR}/../../faiss)

find_package(Python REQUIRED
  COMPONENTS Development NumPy
)

target_link_libraries(swigjecq PRIVATE faiss_python_callbacks)
target_link_libraries(swigjecq_avx2 PRIVATE faiss_python_callbacks)
target_link_libraries(swigjecq_avx512 PRIVATE faiss_python_callbacks)
target_link_libraries(swigjecq_avx512_spr PRIVATE faiss_python_callbacks)
target_link_libraries(swigjecq_sve PRIVATE faiss_python_callbacks)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
configure_file(loader.py loader.py COPYONLY)
configure_file(class_wrappers.py class_wrappers.py COPYONLY)
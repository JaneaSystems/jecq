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

file(
  GLOB
  JECQ_SRC
  *.cpp)
set(JECQ_SRC ${JECQ_SRC})

file(
  GLOB
  JECQ_HEADERS
  *.h)
set(JECQ_HEADERS ${JECQ_HEADERS})

# Export JECQ_HEADERS variable to parent scope.
set(JECQ_HEADERS ${JECQ_HEADERS} PARENT_SCOPE)
add_library(jecq ${JECQ_SRC})

add_library(jecq_avx2 ${JECQ_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(jecq_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  target_compile_options(jecq_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)
else()
  # MSVC enables FMA with /arch:AVX2; no separate flags for F16C, POPCNT
  # Ref. FMA (under /arch:AVX2): https://docs.microsoft.com/en-us/cpp/build/reference/arch-x64
  # Ref. F16C (2nd paragraph): https://walbourn.github.io/directxmath-avx2/
  # Ref. POPCNT: https://docs.microsoft.com/en-us/cpp/intrinsics/popcnt16-popcnt-popcnt64
  target_compile_options(jecq_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  # we need bigobj for the swig wrapper
  add_compile_options(/bigobj)
endif()

add_library(jecq_avx512 ${JECQ_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512")
  set_target_properties(jecq_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  # All modern CPUs support F, CD, VL, DQ, BW extensions.
  # Ref: https://en.wikipedia.org/wiki/AVX512
  target_compile_options(jecq_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)
else()
  target_compile_options(jecq_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  # we need bigobj for the swig wrapper
  add_compile_options(/bigobj)
endif()

add_library(jecq_avx512_spr ${JECQ_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(jecq_avx512_spr PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  # Architecture mode to support AVX512 extensions available since Intel(R) Sapphire Rapids.
  # Ref: https://networkbuilders.intel.com/solutionslibrary/intel-avx-512-fp16-instruction-set-for-intel-xeon-processor-based-products-technology-guide
  target_compile_options(jecq_avx512_spr PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=sapphirerapids -mtune=sapphirerapids>)
else()
  target_compile_options(jecq_avx512_spr PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  # we need bigobj for the swig wrapper
  add_compile_options(/bigobj)
endif()

add_library(jecq_sve ${JECQ_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "sve")
  set_target_properties(jecq_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
    # Do nothing, expect SVE to be enabled by -march=native
  elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
    # Add +sve
    target_compile_options(jecq_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
  elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
    # No valid -march, so specify -march=armv8-a+sve as the default
    target_compile_options(jecq_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
  endif()
  if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
    # Do nothing, expect SVE to be enabled by -march=native
  elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
    # Add +sve
    target_compile_options(jecq_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
  elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
    # No valid -march, so specify -march=armv8-a+sve as the default
    target_compile_options(jecq_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
  endif()
endif()

target_include_directories(jecq PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <jecq/foo.h>`.
target_include_directories(jecq_avx2 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <jecq/foo.h>`.
target_include_directories(jecq_avx512 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <jecq/foo.h>`.
target_include_directories(jecq_avx512_spr PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <jecq/foo.h>`.
target_include_directories(jecq_sve PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)  

set_target_properties(jecq jecq_avx2 jecq_avx512 jecq_avx512_spr jecq_sve PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(WIN32)
  target_compile_definitions(jecq PRIVATE JECQ_MAIN_LIB)
  target_compile_definitions(jecq_avx2 PRIVATE JECQ_MAIN_LIB)
  target_compile_definitions(jecq_avx512 PRIVATE JECQ_MAIN_LIB)
  target_compile_definitions(jecq_avx512_spr PRIVATE JECQ_MAIN_LIB)
  target_compile_definitions(jecq_sve PRIVATE JECQ_MAIN_LIB)
endif()

if(WIN32)
  set_target_properties(jecq PROPERTIES LINK_FLAGS "-Wl,--export-all-symbols")
endif()

string(FIND "${CMAKE_CXX_FLAGS}" "FINTEGER" finteger_idx)
if (${finteger_idx} EQUAL -1)
  target_compile_definitions(jecq PRIVATE FINTEGER=int)
endif()
target_compile_definitions(jecq_avx2 PRIVATE FINTEGER=int)
target_compile_definitions(jecq_avx512 PRIVATE FINTEGER=int)
target_compile_definitions(jecq_avx512_spr PRIVATE FINTEGER=int)
target_compile_definitions(jecq_sve PRIVATE FINTEGER=int)

if(FAISS_USE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error LANGUAGES CXX)

  if (ipo_supported)
    message(STATUS "Jecq: LTO enabled")
    set_property(TARGET jecq PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET jecq_avx2 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET jecq_avx512 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET jecq_avx512_spr PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "Jecq: LTO not supported: <${ipo_error}>")
  endif()
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(jecq PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(jecq PRIVATE faiss)
target_link_libraries(jecq_avx2 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(jecq_avx2 PRIVATE faiss_avx2)
target_link_libraries(jecq_avx512 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(jecq_avx512 PRIVATE faiss_avx512)
target_link_libraries(jecq_avx512_spr PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(jecq_avx512_spr PRIVATE faiss_avx512_spr)
target_link_libraries(jecq_sve PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(jecq_sve PRIVATE faiss_sve)

find_path(FAISS_INCLUDE_DIR
  NAMES   faiss/impl/ProductQuantizer.h
  HINTS   "${CMAKE_SOURCE_DIR}/faiss"
  NO_DEFAULT_PATH
  CACHE   PATH  "Path to Faiss headers")

if (NOT FAISS_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find faiss headers based on hint path impl/ProductQuantizer.h")
endif()

target_include_directories(jecq
  PRIVATE
    ${FAISS_INCLUDE_DIR}
)

target_include_directories(jecq_avx2
  PRIVATE
    ${FAISS_INCLUDE_DIR}
)

target_include_directories(jecq_avx512
  PRIVATE
    ${FAISS_INCLUDE_DIR}
)

target_include_directories(faiss_sve
  PRIVATE
    ${FAISS_INCLUDE_DIR}
)

get_target_property(_incs jecq INCLUDE_DIRECTORIES)
message(STATUS "Jecq: Include dirs: ${_incs}")

if(FAISS_ENABLE_MKL)
  message(STATUS "Jecq: Using MKL for BLAS")
  # find_package(MKL)
  # Don't call find_package(MKL); it seems to not be idempotent and causes issues
endif()

message(STATUS "Jecq: FAISS_ENABLE_MKL = [${FAISS_ENABLE_MKL}]")
message(STATUS "Jecq: MKL_LIBRARIES = [${MKL_LIBRARIES}]")

if(MKL_FOUND)
  target_link_libraries(jecq PRIVATE ${MKL_LIBRARIES})
  target_link_libraries(jecq_avx2 PRIVATE ${MKL_LIBRARIES})
  target_link_libraries(jecq_avx512 PRIVATE ${MKL_LIBRARIES})
  target_link_libraries(jecq_avx512_spr PRIVATE ${MKL_LIBRARIES})
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(jecq PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(jecq_avx2 PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(jecq_avx512 PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(jecq_avx512_spr PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(jecq_sve PRIVATE ${BLAS_LIBRARIES})

  find_package(LAPACK REQUIRED)
  target_link_libraries(jecq PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(jecq_avx2 PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(jecq_avx512 PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(jecq_avx512_spr PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(jecq_sve PRIVATE ${LAPACK_LIBRARIES})
endif()

install(TARGETS jecq
  EXPORT jecq-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(FAISS_OPT_LEVEL STREQUAL "avx2")
  install(TARGETS jecq_avx2
    EXPORT jecq-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
if(FAISS_OPT_LEVEL STREQUAL "avx512")
  install(TARGETS jecq_avx2 jecq_avx512
    EXPORT jecq-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
if(FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  install(TARGETS jecq_avx2 jecq_avx512_spr
    EXPORT jecq-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
if(FAISS_OPT_LEVEL STREQUAL "sve")
  install(TARGETS jecq_sve
    EXPORT jecq-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()


foreach(header ${JECQ_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jecq/${dir}
  )
endforeach()

target_sources(jecq PRIVATE ${JECQ_HEADERS})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/jecq-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_file(${PROJECT_SOURCE_DIR}/cmake/jecq-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/jecq-config.cmake
  COPYONLY
)
install(FILES ${PROJECT_BINARY_DIR}/cmake/jecq-config.cmake
  ${PROJECT_BINARY_DIR}/cmake/jecq-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/jecq
)

install(EXPORT jecq-targets
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/jecq
)
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

# @lint-ignore-every LICENSELINT

function(link_to_jecq_lib target)
  if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr" AND NOT FAISS_OPT_LEVEL STREQUAL "sve")
    target_link_libraries(${target} PRIVATE jecq)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx2")
    if(NOT WIN32)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
    target_link_libraries(${target} PRIVATE jecq_avx2)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx512")
    if(NOT WIN32)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx512f -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    endif()
    target_link_libraries(${target} PRIVATE jecq_avx512)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx512_spr")
    if(NOT WIN32)
      # Architecture mode to support AVX512 extensions available since Intel (R) Sapphire Rapids.
      # Ref: https://networkbuilders.intel.com/solutionslibrary/intel-avx-512-fp16-instruction-set-for-intel-xeon-processor-based-products-technology-guide
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=sapphirerapids -mtune=sapphirerapids>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    endif()
    target_link_libraries(${target} PRIVATE jecq_avx512_spr)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "sve")
    if(NOT WIN32)
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${target}  PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
      endif()
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${target}  PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
      endif()
    else()
      # TODO: support Windows
    endif()
    target_link_libraries(${target} PRIVATE jecq_sve)
  endif()
endfunction()

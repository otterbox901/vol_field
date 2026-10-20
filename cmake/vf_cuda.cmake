# cmake/vf_cuda.cmake
#
# Finds a working nvcc + host compiler pair, enables the CUDA language and
# enforces the sm_75 floor. CUDA is required: every frame source in this
# project feeds a GPU kernel, so there is no build without it.
#
# Finding a *usable* nvcc takes more than a bare check_language(CUDA):
#  * nvcc rejects host compilers newer than it knows, and distro GCC routinely
#    runs ahead of CUDA. check_language() probes in a separate configure that
#    does not receive CMAKE_CUDA_HOST_COMPILER, so it always tried the default
#    g++ and failed. The probe does honour the CUDAHOSTCXX environment variable,
#    so each candidate host compiler is passed that way.
#  * nvcc is often installed outside PATH (/usr/local/cuda/bin).
#  * A failed probe is cached as NOTFOUND and never retried. A verified result
#    is recorded separately, so anything unverified is probed again on the next
#    configure and you never need to delete a build directory to recover.

include(CheckLanguage)

# Fast path: a previous configure already proved this compiler pair works.
set(_vf_pair "${CMAKE_CUDA_COMPILER}|${CMAKE_CUDA_HOST_COMPILER}")
if(NOT (CMAKE_CUDA_COMPILER AND VF_CUDA_VERIFIED_PAIR STREQUAL _vf_pair))
    # --- locate nvcc: explicit setting, then CUDACXX, then PATH and usual prefixes
    if(CMAKE_CUDA_COMPILER)
        set(_vf_nvcc "${CMAKE_CUDA_COMPILER}")
    elseif(DEFINED ENV{CUDACXX})
        set(_vf_nvcc "$ENV{CUDACXX}")
    else()
        find_program(_vf_nvcc NAMES nvcc
                     PATHS ENV CUDA_PATH ENV CUDA_HOME /usr/local/cuda /opt/cuda
                     PATH_SUFFIXES bin
                     NO_CACHE)
    endif()

    if(NOT _vf_nvcc OR NOT EXISTS "${_vf_nvcc}")
        unset(CMAKE_CUDA_COMPILER CACHE)
        message(FATAL_ERROR
            "vf: no CUDA compiler found, and CUDA is required.\n"
            "    Install the CUDA Toolkit, or point CUDACXX / CMAKE_CUDA_COMPILER\n"
            "    at nvcc, and reconfigure. On WATcloud: source scripts/watcloud/env.sh")
    endif()

    # --- host compiler candidates. An explicit choice is the only candidate;
    # otherwise the default g++ first, then versioned g++ newest to oldest.
    if(CMAKE_CUDA_HOST_COMPILER)
        set(_vf_hosts "${CMAKE_CUDA_HOST_COMPILER}")
    elseif(DEFINED ENV{CUDAHOSTCXX})
        set(_vf_hosts "$ENV{CUDAHOSTCXX}")
    else()
        set(_vf_hosts "")
        foreach(_ver RANGE 9 20)
            find_program(_vf_cxx NAMES "g++-${_ver}" "g++${_ver}" NO_CACHE)
            if(_vf_cxx)
                list(PREPEND _vf_hosts "${_vf_cxx}")
            endif()
            unset(_vf_cxx)
        endforeach()
        list(PREPEND _vf_hosts "default")
    endif()

    # --- probe each pairing until nvcc accepts one
    set(ENV{CUDACXX} "${_vf_nvcc}")
    set(_vf_ok OFF)
    foreach(_host IN LISTS _vf_hosts)
        unset(CMAKE_CUDA_COMPILER CACHE)
        unset(CMAKE_CUDA_COMPILER)
        if(_host STREQUAL "default")
            unset(ENV{CUDAHOSTCXX})
            message(STATUS "vf: probing ${_vf_nvcc} with the default host compiler")
        else()
            set(ENV{CUDAHOSTCXX} "${_host}")
            message(STATUS "vf: probing ${_vf_nvcc} with host compiler ${_host}")
        endif()

        check_language(CUDA)

        if(CMAKE_CUDA_COMPILER)
            if(NOT _host STREQUAL "default")
                set(CMAKE_CUDA_HOST_COMPILER "${_host}" CACHE FILEPATH "nvcc host compiler" FORCE)
            endif()
            set(_vf_ok ON)
            break()
        endif()
    endforeach()

    if(NOT _vf_ok)
        unset(VF_CUDA_VERIFIED_PAIR CACHE)
        string(REPLACE ";" ", " _vf_tried "${_vf_hosts}")
        message(FATAL_ERROR
            "vf: found ${_vf_nvcc}, but it accepted none of these host compilers:\n"
            "    ${_vf_tried}\n"
            "    Install a g++ this CUDA release supports (e.g. sudo dnf install gcc14-c++)\n"
            "    or pass -DCMAKE_CUDA_HOST_COMPILER=/path/to/g++ and reconfigure.\n"
            "    Details: CMakeFiles/CMakeConfigureLog.yaml")
    endif()

    set(VF_CUDA_VERIFIED_PAIR "${CMAKE_CUDA_COMPILER}|${CMAKE_CUDA_HOST_COMPILER}" CACHE INTERNAL
        "nvcc|host compiler pair that passed the vf probe")
endif()

if(CMAKE_CUDA_HOST_COMPILER)
    message(STATUS "vf: using ${CMAKE_CUDA_HOST_COMPILER} as the nvcc host compiler")
endif()

# Must be set before enable_language(CUDA): that is when the default is captured.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

# --- sm_75 floor ---------------------------------------------------------
# The per-option seqlock is taken with a CAS spin that can run in two lanes of
# one warp at once. That only makes progress with independent thread
# scheduling (sm_70+); before Volta the spinning lane can starve the lane that
# holds the lock. sm_75 is the floor rather than sm_70 because it is also what
# CUDA 13 still builds for, and it keeps WATcloud's GTX 1080 (sm_61) out.
set(_vf_archs "${CMAKE_CUDA_ARCHITECTURES}")
if(_vf_archs STREQUAL "native")
    if(CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(_vf_archs "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    else()
        set(_vf_archs "")
        message(WARNING "vf: cannot see which GPU 'native' resolved to; "
                        "the binaries re-check compute capability >= 7.5 at startup")
    endif()
elseif(_vf_archs MATCHES "^all")
    message(FATAL_ERROR
        "vf: CMAKE_CUDA_ARCHITECTURES=${_vf_archs} includes pre-Turing targets, and the\n"
        "    field needs sm_75 or newer. List the targets instead, e.g. \"75;86;89\".")
endif()
foreach(_arch IN LISTS _vf_archs)
    string(REGEX MATCH "^[0-9]+" _num "${_arch}")
    if(_num AND _num LESS 75)
        message(FATAL_ERROR
            "vf: CUDA architecture ${_arch} is below sm_75. The field's seqlock needs\n"
            "    independent thread scheduling; build for 75 or newer.")
    endif()
endforeach()

message(STATUS "vf: CUDA enabled (arch=${CMAKE_CUDA_ARCHITECTURES} -> ${_vf_archs})")

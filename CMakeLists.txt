cmake_minimum_required(VERSION 3.24)
project(vol_field LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(VF_BUILD_SIM        "Build vf_sim: pcap / UDP frames fed to the GPU through a VRAM ring" ON)
option(VF_BUILD_TESTS      "Build test_bono, test_black and test_field, as ctest tests" ON)
option(VF_ENABLE_GPUNETIO  "Build vf_gpunetio when the DOCA SDK exists" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(vf_cuda)   # CUDA is required, sm_75 or newer; fatal otherwise
include(vf_doca)   # -> VF_WITH_GPUNETIO

# Warnings on our own C++ only: nvcc takes host flags through -Xcompiler, and
# the CUDA headers are not -Wextra clean. Not -Werror, so a newer compiler
# cannot break the build.
function(vf_set_warnings target)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endfunction()

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

# vf_field: everything both frame sources share - the device tables and the
# surface kernel, the shared-memory snapshot, the CLI, the run loop and the
# report. The per-frame BONO logic is header-only (vf/field_device.cuh) so each
# receive kernel compiles it inline, without relocatable device code.
add_library(vf_field STATIC
    src/field/common.cpp
    src/field/cli.cpp
    src/field/field.cu
    src/field/shm.cpp
    src/field/loop.cpp
    src/field/report.cpp)
target_include_directories(vf_field PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# Link cudart explicitly: the executables' own sources may be plain C++, and
# are linked by the C++ driver, which would not pick the runtime up implicitly.
target_link_libraries(vf_field PUBLIC Threads::Threads CUDA::cudart rt)
set_target_properties(vf_field PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
# -lineinfo keeps the receive and surface kernels attributable in Nsight Compute.
target_compile_options(vf_field PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
vf_set_warnings(vf_field)

# vf_sim: the simulated NIC. Frames from a pcap (or a UDP socket) are paced
# into a pinned staging ring, copied to VRAM, and a persistent kernel with the
# same block shape as the DOCA receive kernel processes them.
if(VF_BUILD_SIM)
    add_executable(vf_sim
        src/sim/main_sim.cpp
        src/sim/pcap_reader.cpp
        src/sim/sim_source.cpp
        src/sim/udp_source.cpp
        src/sim/sim_kernel.cu)
    target_include_directories(vf_sim PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/sim")
    target_link_libraries(vf_sim PRIVATE vf_field)
    target_compile_options(vf_sim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    vf_set_warnings(vf_sim)
endif()

if(VF_BUILD_TESTS)
    enable_testing()

    add_executable(test_bono tests/test_bono.cpp)
    target_link_libraries(test_bono PRIVATE vf_field)
    vf_set_warnings(test_bono)
    add_test(NAME bono COMMAND test_bono)

    add_executable(test_black tests/test_black.cpp)
    target_link_libraries(test_black PRIVATE vf_field)
    vf_set_warnings(test_black)
    add_test(NAME black COMMAND test_black)

    # Needs a GPU: it runs the real receive path on hand-built batches.
    add_executable(test_field tests/test_field.cu)
    target_link_libraries(test_field PRIVATE vf_field)
    vf_set_warnings(test_field)
    add_test(NAME field COMMAND test_field)
endif()

# vf_gpunetio: NIC -> VRAM through DOCA GPUNetIO. Compile-verified only until a
# ConnectX-6 Dx (or newer) turns up; see docs/running.md "On real hardware".
if(VF_WITH_GPUNETIO)
    add_executable(vf_gpunetio
        src/gpunetio/main_gpunetio.cpp
        src/gpunetio/setup.cpp
        src/gpunetio/receive_kernel.cu)
    target_include_directories(vf_gpunetio PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/gpunetio")
    target_link_libraries(vf_gpunetio PRIVATE
        vf_field
        PkgConfig::VF_DOCA_GPUNETIO
        PkgConfig::VF_DOCA_ETH
        PkgConfig::VF_DOCA_FLOW
        PkgConfig::VF_DOCA_COMMON)
    # The DOCA device-side receive functions live in a static device library,
    # which only links into relocatable device code.
    set_target_properties(vf_gpunetio PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    vf_set_warnings(vf_gpunetio)
    # DOCA marks its experimental API with a deprecation attribute, so every
    # GPUNetIO call would warn. Silencing it here keeps the rest of -Wall
    # -Wextra meaningful on this target.
    target_compile_options(vf_gpunetio PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
endif()

message(STATUS "vf: sim=${VF_BUILD_SIM} tests=${VF_BUILD_TESTS} gpunetio=${VF_WITH_GPUNETIO}")

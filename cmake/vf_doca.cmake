# cmake/vf_doca.cmake - find DOCA GPUNetIO; sets VF_WITH_GPUNETIO.
#
# The GPUNetIO path needs the DOCA SDK, which ships only for Ubuntu, RHEL/Rocky,
# Debian, SLES, Oracle, Azure and Amazon Linux - not Fedora. Build it inside
# NVIDIA's DOCA container instead: scripts/doca-build.sh.
#
#   doca-gpunetio  GPU handle, GPU memory, the device-side receive headers
#   doca-eth       doca_eth_rxq: the receive queue whose buffer lives in VRAM
#   doca-flow      hardware steering of the feed groups into that queue
#   doca-common    doca_dev, doca_mmap, doca_ctx, error strings
#
# Missing pieces leave VF_WITH_GPUNETIO OFF and skip the target.

set(VF_WITH_GPUNETIO OFF)

if(NOT VF_ENABLE_GPUNETIO)
    message(STATUS "vf: GPUNetIO path disabled (VF_ENABLE_GPUNETIO=OFF)")
    return()
endif()

set(_vf_doca_hint
    "    The GPUNetIO path needs the DOCA SDK. It has no Fedora build; run\n"
    "    scripts/doca-build.sh to build it inside NVIDIA's DOCA container.")

find_package(PkgConfig QUIET)
if(NOT PkgConfig_FOUND)
    message(WARNING "vf: pkg-config not found, cannot look for DOCA.\n" ${_vf_doca_hint})
    return()
endif()

pkg_check_modules(VF_DOCA_GPUNETIO IMPORTED_TARGET doca-gpunetio)
pkg_check_modules(VF_DOCA_ETH      IMPORTED_TARGET doca-eth)
pkg_check_modules(VF_DOCA_FLOW     IMPORTED_TARGET doca-flow)
pkg_check_modules(VF_DOCA_COMMON   IMPORTED_TARGET doca-common)

if(NOT (VF_DOCA_GPUNETIO_FOUND AND VF_DOCA_ETH_FOUND
        AND VF_DOCA_FLOW_FOUND AND VF_DOCA_COMMON_FOUND))
    message(WARNING "vf: DOCA not found; vf_gpunetio is not built.\n" ${_vf_doca_hint})
    return()
endif()

set(VF_WITH_GPUNETIO ON)
message(STATUS "vf: GPUNetIO path enabled (DOCA ${VF_DOCA_GPUNETIO_VERSION})")

#pragma once
//
// src/gpunetio/kernel.h - what setup.cpp needs from receive_kernel.cu.
//
// Kept private to src/gpunetio/ so no public header mentions CUDA or DOCA.
//

#include <cstdint>

#include "vf/field.hpp"

struct CUstream_st;
struct doca_gpu_eth_rxq;

namespace vf {
namespace gpunetio {

/// Launch the persistent receive kernel: one block per queue, `rxqs` holding
/// `n_queues` handles in GPU-visible memory. The kernel runs until
/// `*exit_flag` becomes non-zero; a block that hits a receive error sets it
/// to 2 itself, which stops the others too.
bool launch_receive_kernel(CUstream_st* stream, const FieldDev& f, doca_gpu_eth_rxq** rxqs,
                           uint32_t n_queues, uint32_t* exit_flag);

/// How many receive blocks can be resident at once (a persistent block that
/// is not resident never starts).
uint32_t max_resident_queues();

}  // namespace gpunetio
}  // namespace vf

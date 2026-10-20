//
// src/gpunetio/receive_kernel.cu - the persistent GPU-direct receive loop.
//
// The NIC has already written whole Ethernet frames into GPU memory, so the
// kernel reads packet bytes locally and never asks the CPU for anything: the
// only host interaction is the exit flag. One block per receive queue; DOCA
// hands back a batch of packets that landed in parallel, and the whole block
// runs vf::dev::process_batch over it - the code vf_sim's kernel runs on its
// VRAM ring, so everything after "here is a batch" is tested on the laptop.
//

#include <doca_gpunetio_dev_eth_rxq.cuh>

#include <cuda_runtime.h>

#include "vf/common.hpp"
#include "vf/field_device.cuh"

#include "kernel.h"

namespace vf {
namespace gpunetio {
namespace {

/// Packets per receive call: one process_batch, i.e. two frames per thread.
/// Each entry costs 16 bytes of shared memory for its attributes.
constexpr uint32_t kMaxRxPkts = dev::kMaxBatch;
static_assert(kMaxRxPkts % dev::kBlockThreads == 0,
              "DOCA wants a multiple of the block size: every thread receives at least once");

/// How long one receive call waits before returning empty-handed: the
/// granularity at which the exit flag is noticed.
constexpr uint64_t kRxTimeoutNs = 500000;  // 500 us

/// Frame i of a batch that starts at packet index `first`.
struct RxqFrames {
    doca_gpu_eth_rxq*                       rxq;
    uint64_t                                first;
    const doca_gpu_dev_eth_rxq_attr*        attr;
    __device__ dev::FrameRef operator()(uint32_t i) const {
        const uint64_t addr = doca_gpu_dev_eth_rxq_get_pkt_addr(rxq, first + i);
        return dev::FrameRef{reinterpret_cast<const uint8_t*>(addr), attr[i].bytes};
    }
};

}  // namespace

/// Block b drains receive queue b until the host sets the exit flag.
__global__ void __launch_bounds__(dev::kBlockThreads)
    vf_receive_kernel(FieldDev f, doca_gpu_eth_rxq** rxqs, uint32_t* exit_flag) {
    doca_gpu_eth_rxq* rxq = rxqs[blockIdx.x];

    __shared__ uint64_t first_pkt_idx;
    __shared__ uint32_t pkt_num;
    __shared__ doca_gpu_dev_eth_rxq_attr attr[kMaxRxPkts];
    __shared__ int s_quit;

    for (;;) {
        // One read for the whole block: threads that disagreed about the flag
        // would split between leaving and the receive's internal barriers.
        if (threadIdx.x == 0) s_quit = DOCA_GPUNETIO_VOLATILE(*exit_flag) != 0;
        __syncthreads();
        if (s_quit) break;

        const doca_error_t ret =
            doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,
                                      DOCA_GPUNETIO_ETH_MCST_AUTO,
                                      DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
                                      DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
                rxq, kMaxRxPkts, kRxTimeoutNs, &first_pkt_idx, &pkt_num, attr);

        if (ret != DOCA_SUCCESS) {
            // Not recoverable for this queue, and spinning on it would wedge
            // the run: stop every block and let the host report it.
            if (threadIdx.x == 0) DOCA_GPUNETIO_VOLATILE(*exit_flag) = 2;
            break;
        }
        if (pkt_num == 0) continue;

        // process_batch ends with __syncthreads, so attr[] and the batch
        // bounds are free for the next receive when it returns.
        dev::process_batch(f, pkt_num, RxqFrames{rxq, first_pkt_idx, attr});
    }
}

bool launch_receive_kernel(CUstream_st* stream, const FieldDev& f, doca_gpu_eth_rxq** rxqs,
                           uint32_t n_queues, uint32_t* exit_flag) {
    if (cudaGetLastError() != cudaSuccess) return false;
    vf_receive_kernel<<<n_queues, dev::kBlockThreads, 0, stream>>>(f, rxqs, exit_flag);
    return cudaGetLastError() == cudaSuccess;
}

uint32_t max_resident_queues() {
    int device = 0, sms = 0, per_sm = 0;
    VF_CUDA_CHECK(cudaGetDevice(&device));
    VF_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    VF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, vf_receive_kernel,
                                                                dev::kBlockThreads, 0));
    return static_cast<uint32_t>(per_sm * sms);
}

}  // namespace gpunetio
}  // namespace vf

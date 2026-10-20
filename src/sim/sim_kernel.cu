//
// src/sim/sim_kernel.cu - the simulator's persistent receive kernel.
//
// The GPUNetIO receive kernel with doca_gpu_dev_eth_rxq_recv() swapped for a
// poll of the VRAM ring's produced counter. Everything after "here is a batch
// of frames" is the shared vf::dev::process_batch().
//

#include <cuda_runtime.h>

#include "vf/common.hpp"
#include "vf/field_device.cuh"

#include "sim.hpp"

namespace vf {
namespace sim {
namespace {

/// Frame i of a batch that starts at ring index `first`.
struct RingFrames {
    const uint8_t*     ring;
    unsigned long long first;
    __device__ dev::FrameRef operator()(uint32_t i) const {
        const uint8_t* s = ring + ((first + i) & (kRingSlots - 1)) * kSlotBytes;
        return dev::FrameRef{s + kSlotHeader, *reinterpret_cast<const uint32_t*>(s)};
    }
};

__device__ __forceinline__ unsigned long long load_acquire(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

}  // namespace

/// Block b drains queue b. Thread 0 polls for frames and cuts a batch of up
/// to kMaxBatch; the whole block processes it; thread 0 publishes progress.
__global__ void __launch_bounds__(dev::kBlockThreads)
    vf_sim_kernel(FieldDev f, const QueueDev* queues, const unsigned int* exit_flag,
                  unsigned long long max_run_ns, unsigned int backoff_ns) {
    const QueueDev q = queues[blockIdx.x];

    __shared__ unsigned long long s_first;
    __shared__ unsigned int s_n;
    __shared__ int s_quit;

    const unsigned long long t0 = device_now_ns();
    unsigned long long consumed = 0;

    for (;;) {
        if (threadIdx.x == 0) {
            unsigned int n = 0;
            int quit = 0;
            for (unsigned long long spins = 1;; ++spins) {
                // Acquire: the copy engine wrote the frames before this counter,
                // and an acquire at GPU scope keeps the block's L1 from serving
                // bytes left over from the ring's previous lap.
                const unsigned long long produced = load_acquire(q.produced);
                if (produced > consumed) {
                    const unsigned long long avail = produced - consumed;
                    n = avail < dev::kMaxBatch ? static_cast<unsigned int>(avail) : dev::kMaxBatch;
                    break;
                }
                // The host raises the flag only after its last copy landed, so
                // an empty ring here really is the end.
                if (dev::ld(exit_flag)) {
                    quit = 1;
                    break;
                }
                if ((spins & 1023u) == 0 && device_now_ns() - t0 > max_run_ns) {
                    quit = 1;
                    break;
                }
                if (backoff_ns) __nanosleep(backoff_ns);
            }
            s_first = consumed;
            s_n = n;
            s_quit = quit;
        }
        __syncthreads();
        if (s_quit) break;

        const unsigned int n = s_n;
        dev::process_batch(f, n, RingFrames{q.ring, s_first});
        consumed += n;

        if (threadIdx.x == 0) {
            dev::st(q.consumed, consumed);
            __threadfence_system();
        }
    }
}

bool launch_sim_kernel(CUstream_st* stream, const FieldDev& f, const QueueDev* queues,
                       uint32_t n_queues, const unsigned int* exit_flag, uint64_t max_run_ns,
                       uint32_t backoff_ns) {
    vf_sim_kernel<<<n_queues, dev::kBlockThreads, 0, stream>>>(f, queues, exit_flag, max_run_ns,
                                                               backoff_ns);
    return cudaGetLastError() == cudaSuccess;
}

uint32_t max_resident_queues() {
    int device = 0, sms = 0, per_sm = 0;
    VF_CUDA_CHECK(cudaGetDevice(&device));
    VF_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    VF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, vf_sim_kernel,
                                                                dev::kBlockThreads, 0));
    return static_cast<uint32_t>(per_sm * sms);
}

}  // namespace sim
}  // namespace vf

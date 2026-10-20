//
// src/field/common.cpp - the host clock and CUDA error reporting.
//

#include "vf/common.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace vf {

uint64_t host_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

void cuda_fail(int err, const char* file, int line, const char* expr) {
    std::fprintf(stderr, "[vf] CUDA error %s:%d: %s -> %s\n", file, line, expr,
                 cudaGetErrorString(static_cast<cudaError_t>(err)));
    std::abort();
}

}  // namespace vf

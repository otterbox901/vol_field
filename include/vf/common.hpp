#pragma once
//
// vf/common.hpp - definitions shared by host (.cpp) and device (.cu) code.
//
// This header must stay compilable by a plain C++17 compiler: everything that
// needs nvcc is guarded by __CUDACC__.
//

#include <cstddef>
#include <cstdint>

// Mark functions that must exist on both sides. Expands to nothing when the
// translation unit is built by the host compiler rather than nvcc.
#if defined(__CUDACC__)
#  define VF_HD __host__ __device__
#  define VF_FORCEINLINE __forceinline__
#else
#  define VF_HD
#  define VF_FORCEINLINE inline __attribute__((always_inline))
#endif

namespace vf {

/// Monotonic clock in nanoseconds (std::chrono::steady_clock). The host's time
/// base for pacing, the run loop and the snapshot's host_ns stamp. Never
/// compared with the GPU's %globaltimer.
uint64_t host_now_ns();

/// Report a failed CUDA runtime call and abort. Takes the error as an int so
/// this header does not need the CUDA headers; see VF_CUDA_CHECK.
[[noreturn]] void cuda_fail(int err, const char* file, int line, const char* expr);

/// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's
/// days_from_civil). Used for --date on the host and for option expiries on
/// the device, so both sides count days the same way.
VF_HD constexpr int32_t days_from_civil(int32_t y, uint32_t m, uint32_t d) {
    y -= m <= 2 ? 1 : 0;
    const int32_t era = (y >= 0 ? y : y - 399) / 400;
    const uint32_t yoe = static_cast<uint32_t>(y - era * 400);
    const uint32_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const uint32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int32_t>(doe) - 719468;
}

}  // namespace vf

/// Abort loudly on a failed CUDA runtime call. Works in .cpp and .cu files
/// alike: cudaError_t converts to int, and cudaSuccess is 0.
#define VF_CUDA_CHECK(expr)                                              \
    do {                                                                 \
        const int _vf_e = static_cast<int>(expr);                        \
        if (_vf_e != 0) ::vf::cuda_fail(_vf_e, __FILE__, __LINE__, #expr); \
    } while (0)

#if defined(__CUDACC__)
namespace vf {

/// GPU-side nanosecond counter.
///
/// %globaltimer is a free-running ns counter shared by all SMs. It shares
/// neither an epoch nor a reliable rate with host_now_ns(), so it is used only
/// for device-side intervals (when a persistent kernel must give up), never
/// compared against a host timestamp.
__device__ __forceinline__ uint64_t device_now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

}  // namespace vf
#endif  // __CUDACC__

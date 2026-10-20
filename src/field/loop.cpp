//
// src/field/loop.cpp - the surface pass loop, the status line, Ctrl-C, and
// the device check both drivers start with.
//

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>

#include <cuda_runtime.h>

#include "vf/run.hpp"

namespace vf {
namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true); }

}  // namespace

void install_stop_signals() {
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

bool stop_requested() { return g_stop.load(); }

std::string clock_string(uint32_t feed_sec) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%02u:%02u:%02u", (feed_sec / 3600) % 100,
                  (feed_sec / 60) % 60, feed_sec % 60);
    return buf;
}

std::string si(double v) {
    char buf[32];
    if (v >= 1e9) std::snprintf(buf, sizeof buf, "%.2fG", v / 1e9);
    else if (v >= 1e6) std::snprintf(buf, sizeof buf, "%.2fM", v / 1e6);
    else if (v >= 1e4) std::snprintf(buf, sizeof buf, "%.1fk", v / 1e3);
    else std::snprintf(buf, sizeof buf, "%.0f", v);
    return buf;
}

int check_device(bool verbose) {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::fprintf(stderr, "[vf] no usable CUDA device\n");
        return 0;
    }
    const int cc = prop.major * 10 + prop.minor;
    std::printf("[vf] GPU %d: %s, sm_%d, %d SMs\n", device, prop.name, cc,
                prop.multiProcessorCount);
    if (cc < 75) {
        std::fprintf(stderr,
                     "[vf] %s is sm_%d; the field needs sm_75 or newer (its per-option seqlock "
                     "relies on independent thread scheduling)\n",
                     prop.name, cc);
        return 0;
    }
    int timeout = 0;
    cudaDeviceGetAttribute(&timeout, cudaDevAttrKernelExecTimeout, device);
    if (timeout) {
        std::printf("[vf] note: this GPU has a kernel run-time limit (it drives a display). A "
                    "persistent kernel may stall the desktop or be killed; keep --duration "
                    "short.\n");
    }
    if (verbose) {
        int fp64_ratio = 0;
        cudaDeviceGetAttribute(&fp64_ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device);
        std::printf("[vf] %.1f GiB VRAM, FP64 at 1/%d of FP32, concurrent kernels %s\n",
                    static_cast<double>(prop.totalGlobalMem) / (1u << 30), fp64_ratio,
                    prop.concurrentKernels ? "yes" : "no");
    }
    return prop.multiProcessorCount;
}

SurfaceRunner::SurfaceRunner(Field* field, const Snapshot& snap)
    : field_(field), snap_dev_(snap.dev) {
    cudaStream_t s = nullptr;
    VF_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    stream_ = s;
    cudaEvent_t a = nullptr, b = nullptr;
    VF_CUDA_CHECK(cudaEventCreate(&a));
    VF_CUDA_CHECK(cudaEventCreate(&b));
    start_ = a;
    stop_ = b;
}

SurfaceRunner::~SurfaceRunner() {
    if (stream_) cudaStreamSynchronize(stream_);
    if (start_) cudaEventDestroy(start_);
    if (stop_) cudaEventDestroy(stop_);
    if (stream_) cudaStreamDestroy(stream_);
}

void SurfaceRunner::collect(bool wait) {
    if (!in_flight_) return;
    if (wait) VF_CUDA_CHECK(cudaEventSynchronize(stop_));
    else if (cudaEventQuery(stop_) == cudaErrorNotReady) return;
    float ms = 0.f;
    VF_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    in_flight_ = false;
    ++stats_.passes;
    stats_.gpu_ms_sum += ms;
    if (ms > stats_.gpu_ms_max) stats_.gpu_ms_max = ms;
}

void SurfaceRunner::tick() {
    collect(false);
    if (in_flight_) {
        ++stats_.skipped;
        return;
    }
    VF_CUDA_CHECK(cudaEventRecord(start_, stream_));
    field_launch_surface(field_, stream_, snap_dev_, host_now_ns(), kSnapRunning);
    VF_CUDA_CHECK(cudaEventRecord(stop_, stream_));
    in_flight_ = true;
}

void SurfaceRunner::finish() {
    collect(true);
    VF_CUDA_CHECK(cudaEventRecord(start_, stream_));
    field_launch_surface(field_, stream_, snap_dev_, host_now_ns(), kSnapFinal);
    VF_CUDA_CHECK(cudaEventRecord(stop_, stream_));
    in_flight_ = true;
    collect(true);
}

uint64_t run_loop(const RunConfig& cfg, SurfaceRunner& surface, const Snapshot& snap,
                  const std::function<bool()>& source_done,
                  const std::function<std::string()>& source_status) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const auto period = std::chrono::milliseconds(cfg.snapshot_ms);
    auto next_pass = t0;
    auto next_status = t0 + std::chrono::seconds(1);

    SnapHeader prev{};
    bool have_prev = false;

    for (;;) {
        const auto now = clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
        if (stop_requested()) break;
        if (cfg.duration_ms && ms >= cfg.duration_ms) break;
        if (source_done && source_done()) break;

        if (now >= next_pass) {
            surface.tick();
            next_pass += period;
            if (next_pass < now) next_pass = now + period;  // fell behind: do not burst
        }

        if (now >= next_status) {
            next_status += std::chrono::seconds(1);
            SnapHeader h{};
            if (snapshot_header(snap, h) && h.snapshot_id > 0) {
                const FeedCounters& c = h.counters;
                double rate = 0.0;
                if (have_prev && h.host_ns > prev.host_ns)
                    rate = static_cast<double>(c.messages - prev.counters.messages) * 1e9 /
                           static_cast<double>(h.host_ns - prev.host_ns);
                std::printf("[vf] %6.1fs  %s  msgs %s (%s/s)  quotes %s  stale %s  unmapped %s  "
                            "iv %s ok / %s fail  options %u%s\n",
                            static_cast<double>(ms) / 1e3, clock_string(h.feed_sec).c_str(),
                            si(static_cast<double>(c.messages)).c_str(), si(rate).c_str(),
                            si(static_cast<double>(c.quotes)).c_str(),
                            si(static_cast<double>(c.stale)).c_str(),
                            si(static_cast<double>(c.unmapped)).c_str(),
                            si(static_cast<double>(c.iv_solved)).c_str(),
                            si(static_cast<double>(c.iv_failed)).c_str(), h.n_options,
                            source_status ? source_status().c_str() : "");
                std::fflush(stdout);
                prev = h;
                have_prev = true;
            }
        }

        const auto wake = next_pass < next_status ? next_pass : next_status;
        std::this_thread::sleep_until(wake < now + std::chrono::milliseconds(5)
                                          ? wake
                                          : now + std::chrono::milliseconds(5));
    }
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
}

}  // namespace vf

//
// src/sim/sim_source.cpp - the simulated NIC's rings, and pcap replay.
//

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "vf/moldudp64.hpp"
#include "vf/net.hpp"
#include "vf/run.hpp"

#include "source.hpp"

namespace vf {
namespace sim {

// ---------------------------------------------------------------------------
// Rings
// ---------------------------------------------------------------------------

struct Rings::Queue {
    uint8_t*            staging = nullptr;       ///< pinned host, kRingSlots x kSlotBytes
    uint8_t*            ring_dev = nullptr;      ///< VRAM, same layout
    unsigned long long* produced_dev = nullptr;  ///< VRAM
    unsigned long long* cells = nullptr;         ///< pinned: source of each produced-counter copy
    cudaStream_t        stream = nullptr;        ///< this queue's copies, in order
    uint64_t            produced = 0;            ///< frames committed
    uint64_t            flushed = 0;             ///< frames whose copy is queued
    uint32_t            max_len = 0;             ///< longest frame since the last flush
};

bool Rings::create(uint32_t n_queues, std::atomic<bool>* abort) {
    n_ = n_queues;
    abort_ = abort;
    q_ = new Queue[n_];

    VF_CUDA_CHECK(cudaHostAlloc(&consumed_host_, sizeof(unsigned long long) * n_,
                                cudaHostAllocMapped));
    std::memset(consumed_host_, 0, sizeof(unsigned long long) * n_);
    unsigned long long* consumed_dev = nullptr;
    VF_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&consumed_dev),
                                           consumed_host_, 0));
    VF_CUDA_CHECK(cudaHostAlloc(&exit_host_, sizeof(unsigned int), cudaHostAllocMapped));
    *exit_host_ = 0;
    VF_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&exit_dev_), exit_host_, 0));

    std::vector<QueueDev> desc(n_);
    const size_t bytes = size_t{kRingSlots} * kSlotBytes;
    for (uint32_t i = 0; i < n_; ++i) {
        Queue& q = q_[i];
        VF_CUDA_CHECK(cudaHostAlloc(&q.staging, bytes, cudaHostAllocDefault));
        VF_CUDA_CHECK(cudaMalloc(&q.ring_dev, bytes));
        VF_CUDA_CHECK(cudaMalloc(&q.produced_dev, sizeof(unsigned long long)));
        VF_CUDA_CHECK(cudaMemset(q.produced_dev, 0, sizeof(unsigned long long)));
        VF_CUDA_CHECK(cudaHostAlloc(&q.cells, sizeof(unsigned long long) * kRingSlots,
                                    cudaHostAllocDefault));
        // Created before the kernel launches: streams made lazily mid-run were
        // unreliable alongside a persistent kernel (kernel_bypass design.md).
        VF_CUDA_CHECK(cudaStreamCreateWithFlags(&q.stream, cudaStreamNonBlocking));
        desc[i] = QueueDev{q.ring_dev, q.produced_dev, consumed_dev + i};
    }
    VF_CUDA_CHECK(cudaMalloc(&queues_dev_, sizeof(QueueDev) * n_));
    VF_CUDA_CHECK(cudaMemcpy(queues_dev_, desc.data(), sizeof(QueueDev) * n_,
                             cudaMemcpyHostToDevice));
    VF_CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

void Rings::destroy() {
    if (!q_) return;
    for (uint32_t i = 0; i < n_; ++i) {
        Queue& q = q_[i];
        if (q.stream) cudaStreamSynchronize(q.stream);
        if (q.stream) cudaStreamDestroy(q.stream);
        cudaFreeHost(q.staging);
        cudaFree(q.ring_dev);
        cudaFree(q.produced_dev);
        cudaFreeHost(q.cells);
    }
    delete[] q_;
    q_ = nullptr;
    cudaFree(queues_dev_);
    cudaFreeHost(consumed_host_);
    cudaFreeHost(exit_host_);
    queues_dev_ = nullptr;
    consumed_host_ = nullptr;
    exit_host_ = nullptr;
}

uint64_t Rings::produced(uint32_t q) const { return q_[q].produced; }

uint64_t Rings::consumed(uint32_t q) const {
    return *const_cast<const volatile unsigned long long*>(&consumed_host_[q]);
}

uint32_t Rings::pending(uint32_t q) const {
    return static_cast<uint32_t>(q_[q].produced - q_[q].flushed);
}

uint8_t* Rings::slot(uint32_t qi) {
    Queue& q = q_[qi];
    // A slot is free once the kernel has consumed it - which also proves its
    // copy (and the counter copy after it) has run, so the staging bytes and
    // the produced cell can be reused too.
    if (q.produced - consumed(qi) >= kRingSlots) {
        flush(qi);  // the kernel can only free slots it has been given
        ++full_waits_;
        while (q.produced - consumed(qi) >= kRingSlots) {
            if (abort_ && abort_->load()) return nullptr;
            std::this_thread::yield();
        }
    }
    return q.staging + (q.produced & (kRingSlots - 1)) * size_t{kSlotBytes};
}

void Rings::commit(uint32_t qi, uint32_t len) {
    Queue& q = q_[qi];
    uint8_t* s = q.staging + (q.produced & (kRingSlots - 1)) * size_t{kSlotBytes};
    std::memcpy(s, &len, sizeof len);
    if (len > q.max_len) q.max_len = len;
    ++q.produced;
}

void Rings::flush(uint32_t qi) {
    Queue& q = q_[qi];
    if (q.flushed == q.produced) return;

    // Copy only as much of each slot as the longest frame needs: a 2D copy
    // with the slot as pitch. Two pieces when the batch wraps the ring.
    const size_t width = std::min<size_t>(kSlotBytes, (kSlotHeader + q.max_len + 15) & ~size_t{15});
    const uint64_t first = q.flushed & (kRingSlots - 1);
    const uint64_t count = q.produced - q.flushed;
    const uint64_t head = std::min<uint64_t>(count, kRingSlots - first);
    VF_CUDA_CHECK(cudaMemcpy2DAsync(q.ring_dev + first * kSlotBytes, kSlotBytes,
                                    q.staging + first * kSlotBytes, kSlotBytes, width, head,
                                    cudaMemcpyHostToDevice, q.stream));
    if (count > head)
        VF_CUDA_CHECK(cudaMemcpy2DAsync(q.ring_dev, kSlotBytes, q.staging, kSlotBytes, width,
                                        count - head, cudaMemcpyHostToDevice, q.stream));

    // Then the counter, from a cell tied to the batch's last slot: that slot,
    // and so the cell, is not reused before the kernel has seen this value.
    unsigned long long* cell = &q.cells[(q.produced - 1) & (kRingSlots - 1)];
    *cell = q.produced;
    VF_CUDA_CHECK(cudaMemcpyAsync(q.produced_dev, cell, sizeof *cell, cudaMemcpyHostToDevice,
                                  q.stream));
    q.flushed = q.produced;
    q.max_len = 0;
}

void Rings::flush_all() {
    for (uint32_t i = 0; i < n_; ++i) flush(i);
}

void Rings::sync() {
    for (uint32_t i = 0; i < n_; ++i) VF_CUDA_CHECK(cudaStreamSynchronize(q_[i].stream));
}

bool Rings::drained() const {
    for (uint32_t i = 0; i < n_; ++i)
        if (q_[i].flushed != q_[i].produced || consumed(i) != q_[i].produced) return false;
    return true;
}

void Rings::request_exit() {
    *const_cast<volatile unsigned int*>(exit_host_) = 1;
}

// ---------------------------------------------------------------------------
// PcapSource
// ---------------------------------------------------------------------------

bool PcapSource::load(const PcapOptions& opt, uint32_t n_queues, std::string& err) {
    opt_ = opt;
    if (!pcap_load(opt.path.c_str(), pcap_, err)) return false;

    uint64_t seq_min = UINT64_MAX, seq_end = 0;
    for (const PcapFrame& pf : pcap_.frames) {
        const uint8_t* f = pcap_.data.data() + pf.offset;
        UdpView u{};
        if (pf.len > kMaxFrameBytes || !parse_udp(f, pf.len, u)) {
            ++skipped_;
            continue;
        }
        if (!opt.feeds.empty()) {
            const auto to_feed = [&](const Endpoint& e) {
                return e.port == u.dst_port && (e.ip == 0 || e.ip == u.dst_ip);
            };
            if (std::none_of(opt.feeds.begin(), opt.feeds.end(), to_feed)) {
                ++skipped_;
                continue;
            }
        }
        Frame fr{pf.ts_ns, pf.offset, pf.len, queue_of(u.dst_ip, u.dst_port, n_queues), -1};
        mold::Header h{};
        if (mold::parse_header(u.payload, u.len, h)) {
            fr.seq_off = static_cast<int32_t>((u.payload - f) + mold::kSessionBytes);
            const uint64_t n = (h.count == mold::kEndOfSession) ? 0 : h.count;
            seq_min = std::min(seq_min, h.seq);
            seq_end = std::max(seq_end, h.seq + n);
        }
        frames_.push_back(fr);
    }
    if (frames_.empty()) {
        err = opt.path + ": no IPv4/UDP frames" +
              (opt.feeds.empty() ? "" : " to the --feed destinations");
        return false;
    }
    // One lap's worth of sequence numbers, so lap k+1 is entirely newer than lap k.
    lap_seq_span_ = seq_end > seq_min && seq_min != UINT64_MAX ? seq_end - seq_min : 0;
    lap_ns_ = frames_.back().ts_ns - frames_.front().ts_ns + 1000000;  // + 1 ms between laps
    return true;
}

void PcapSource::start(Rings& rings) {
    rings_ = &rings;
    thread_ = std::thread([this] { run(); });
}

void PcapSource::stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

void PcapSource::run() {
    using clock = std::chrono::steady_clock;
    Rings& rings = *rings_;
    const auto t0 = clock::now();
    const uint64_t ts0 = frames_.front().ts_ns;
    const bool paced = opt_.pps != 0.0;
    // A gap longer than this is worth flushing for, so frames do not sit in
    // staging while the producer sleeps; shorter gaps are batched.
    const auto slack = std::chrono::microseconds(50);

    uint64_t i = 0;
    for (uint64_t lap = 0; !stop_.load(); ++lap) {
        lap_.store(lap);
        for (const Frame& fr : frames_) {
            if (stop_.load()) break;
            if (paced) {
                const double offset_ns =
                    opt_.pps > 0
                        ? static_cast<double>(i) * 1e9 / opt_.pps
                        : static_cast<double>(lap * lap_ns_ + (fr.ts_ns - ts0)) / opt_.speed;
                const auto due = t0 + std::chrono::nanoseconds(static_cast<int64_t>(offset_ns));
                if (due - clock::now() > slack) {
                    rings.flush_all();
                    std::this_thread::sleep_until(due - std::chrono::microseconds(20));
                    while (clock::now() < due) {
                    }
                }
            }
            uint8_t* s = rings.slot(fr.queue);
            if (!s) break;
            std::memcpy(s + kSlotHeader, pcap_.data.data() + fr.offset, fr.len);
            if (lap && fr.seq_off >= 0) {
                uint8_t* seq = s + kSlotHeader + fr.seq_off;
                store_be64(seq, load_be64(seq) + lap * lap_seq_span_);
            }
            rings.commit(fr.queue, fr.len);
            if (rings.pending(fr.queue) >= kFlushFrames) rings.flush(fr.queue);
            ++i;
            sent_.store(i, std::memory_order_relaxed);
        }
        if (!opt_.loop) break;
    }
    rings.flush_all();
    finished_.store(true);
}

std::string PcapSource::status() const {
    std::string s = " | sent " + si(static_cast<double>(sent_.load()));
    if (opt_.loop) s += " lap " + std::to_string(lap_.load());
    if (rings_ && rings_->full_waits())
        s += " ring-full " + si(static_cast<double>(rings_->full_waits()));
    return s;
}

std::string PcapSource::describe() const {
    char pace[64];
    if (opt_.pps < 0) std::snprintf(pace, sizeof pace, "realtime x%.3g", opt_.speed);
    else if (opt_.pps == 0) std::snprintf(pace, sizeof pace, "unpaced");
    else std::snprintf(pace, sizeof pace, "%.0f frames/s", opt_.pps);
    char buf[512];
    std::snprintf(buf, sizeof buf, "pcap %s: %zu frames (%llu skipped), %s%s; sent %llu, "
                  "%llu ring-full waits",
                  opt_.path.c_str(), frames_.size(), static_cast<unsigned long long>(skipped_),
                  pace, opt_.loop ? ", looped" : "",
                  static_cast<unsigned long long>(sent_.load()),
                  static_cast<unsigned long long>(rings_ ? rings_->full_waits() : 0));
    return buf;
}

}  // namespace sim
}  // namespace vf

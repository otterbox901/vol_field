#pragma once
//
// src/sim/sim.hpp - the simulated NIC.
//
// No ConnectX here, so frames reach VRAM the long way round: a host thread
// writes each frame into a 2 KiB slot of a pinned staging ring, a batch of
// slots is copied H2D into the same slots of a VRAM ring, and a produced
// counter is copied after it on the same stream - so by the time the kernel
// sees the counter move, the frames it covers have landed. The persistent
// kernel (sim_kernel.cu) has the DOCA receive kernel's shape: one block of
// 512 threads per queue, batches of up to 1024 frames, and the very same
// vf::dev::process_batch().
//
// The kernel publishes how far it has consumed into mapped host memory; the
// producer reads it for back-pressure before reusing a slot.
//

#include <atomic>
#include <cstdint>

#include "vf/field.hpp"

namespace vf {
namespace sim {

constexpr uint32_t kSlotBytes = 2048;                         ///< one frame per slot
constexpr uint32_t kSlotHeader = 8;                           ///< u32 frame length, u32 reserved
constexpr uint32_t kMaxFrameBytes = kSlotBytes - kSlotHeader; ///< longer frames are skipped
constexpr uint32_t kRingSlots = 16384;                        ///< per queue: 32 MiB VRAM + pinned
constexpr uint32_t kFlushFrames = 256;                        ///< unpaced: copy this many at once
static_assert((kRingSlots & (kRingSlots - 1)) == 0, "ring size must be a power of two");

/// One queue as the kernel sees it.
struct QueueDev {
    const uint8_t*            ring;      ///< VRAM: kRingSlots x kSlotBytes
    const unsigned long long* produced;  ///< VRAM: frames copied in so far
    unsigned long long*       consumed;  ///< mapped host memory: frames processed so far
};

/// Launch the persistent kernel, one block per queue, on `stream`. It runs
/// until *exit_flag is non-zero and every produced frame is consumed, or
/// max_run_ns passes. `queues` is a device array of n_queues entries.
bool launch_sim_kernel(CUstream_st* stream, const FieldDev& f, const QueueDev* queues,
                       uint32_t n_queues, const unsigned int* exit_flag, uint64_t max_run_ns,
                       uint32_t backoff_ns);

/// How many receive blocks can be resident at once. A persistent block that
/// is not resident never starts, so the launch must stay at or below this.
uint32_t max_resident_queues();

/// Queue a frame belongs to: a hash of its destination, like NIC RSS, so one
/// MoldUDP64 stream always lands on one queue (one block).
inline uint32_t queue_of(uint32_t dst_ip, uint16_t dst_port, uint32_t n_queues) {
    uint32_t h = dst_ip ^ (static_cast<uint32_t>(dst_port) * 0x9E3779B1u);
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return n_queues ? h % n_queues : 0;
}

/// The rings of every queue: staging (pinned), VRAM, counters, copy streams.
/// The producer side is single-threaded.
class Rings {
public:
    ~Rings() { destroy(); }

    bool create(uint32_t n_queues, std::atomic<bool>* abort);
    void destroy();

    /// Staging slot for the next frame of queue q (data goes at +kSlotHeader).
    /// Waits for the kernel to free one, flushing first so it can. nullptr if
    /// `abort` was raised while waiting.
    uint8_t* slot(uint32_t q);
    /// The frame written into slot(q) is `len` bytes long; it is now produced.
    void commit(uint32_t q, uint32_t len);
    /// Queue the H2D copy of every committed frame, then the produced counter.
    void flush(uint32_t q);
    void flush_all();
    uint32_t pending(uint32_t q) const;

    /// Wait for every queued copy to land.
    void sync();
    /// True when every queue's kernel block has consumed all it was given.
    bool drained() const;
    /// Tell the kernel to retire once drained.
    void request_exit();

    uint32_t queues() const { return n_; }
    const QueueDev* queues_dev() const { return queues_dev_; }
    const unsigned int* exit_flag_dev() const { return exit_dev_; }
    uint64_t produced(uint32_t q) const;
    uint64_t consumed(uint32_t q) const;
    uint64_t full_waits() const { return full_waits_; }

private:
    struct Queue;
    Queue*             q_ = nullptr;
    uint32_t           n_ = 0;
    QueueDev*          queues_dev_ = nullptr;
    unsigned long long* consumed_host_ = nullptr;  ///< mapped, n_ entries
    unsigned int*      exit_host_ = nullptr;       ///< mapped
    unsigned int*      exit_dev_ = nullptr;
    std::atomic<bool>* abort_ = nullptr;
    uint64_t           full_waits_ = 0;
};

}  // namespace sim
}  // namespace vf

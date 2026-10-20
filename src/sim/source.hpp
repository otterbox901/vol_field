#pragma once
//
// src/sim/source.hpp - where the simulator's frames come from.
//
//   PcapSource  replays a pcap, paced by its timestamps (--realtime, the
//               default; --speed scales it), at a fixed --pps, or unpaced
//               (--pps 0). --loop replays it forever, shifting every
//               MoldUDP64 sequence number by one lap's span per lap so the
//               field keeps seeing newer messages.
//   UdpSource   receives MoldUDP64 datagrams through the normal kernel stack
//               (recvmmsg). Not a bypass, but it needs no root, so two
//               WATcloud nodes can run sender and receiver. Each datagram
//               gets a synthetic 42-byte Eth/IPv4/UDP header so its slot
//               looks exactly like a frame off the wire.
//
// Either runs on its own thread and writes into sim::Rings.
//

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "vf/cli.hpp"

#include "pcap_reader.hpp"
#include "sim.hpp"

namespace vf {
namespace sim {

using vf::Endpoint;

class Source {
public:
    virtual ~Source() = default;
    virtual void start(Rings& rings) = 0;
    /// Ask the thread to stop, and wait for it. Pending frames are flushed.
    virtual void stop() = 0;
    /// True once every frame has been handed to the rings (never for --loop
    /// or a socket).
    virtual bool finished() const = 0;
    /// Short addition to the run loop's status line.
    virtual std::string status() const = 0;
    /// One line for the report.
    virtual std::string describe() const = 0;
};

struct PcapOptions {
    std::string           path;
    double                pps = -1.0;  ///< < 0: realtime; 0: unpaced; > 0: frames per second
    double                speed = 1.0; ///< realtime multiplier
    bool                  loop = false;
    std::vector<Endpoint> feeds;       ///< keep only frames to these; empty = every UDP frame
};

class PcapSource final : public Source {
public:
    bool load(const PcapOptions& opt, uint32_t n_queues, std::string& err);
    void start(Rings& rings) override;
    void stop() override;
    bool finished() const override { return finished_.load(); }
    std::string status() const override;
    std::string describe() const override;

private:
    struct Frame {
        uint64_t ts_ns;
        uint64_t offset;
        uint32_t len;
        uint32_t queue;
        int32_t  seq_off;  ///< offset of the MoldUDP64 sequence field, -1 if none
    };
    void run();

    PcapOptions           opt_;
    Pcap                  pcap_;
    std::vector<Frame>    frames_;
    uint64_t              lap_seq_span_ = 0;
    uint64_t              lap_ns_ = 0;
    uint64_t              skipped_ = 0;    ///< not UDP, other feeds, or oversize
    Rings*                rings_ = nullptr;
    std::thread           thread_;
    std::atomic<bool>     stop_{false};
    std::atomic<bool>     finished_{false};
    std::atomic<uint64_t> sent_{0};
    std::atomic<uint64_t> lap_{0};
};

class UdpSource final : public Source {
public:
    ~UdpSource() override;
    bool open(const std::vector<Endpoint>& feeds, uint32_t iface_ip, uint32_t n_queues,
              std::string& err);
    void start(Rings& rings) override;
    void stop() override;
    bool finished() const override { return false; }
    std::string status() const override;
    std::string describe() const override;

private:
    void run();

    std::vector<Endpoint> feeds_;
    std::vector<int>      fds_;
    uint32_t              n_queues_ = 1;
    Rings*                rings_ = nullptr;
    std::thread           thread_;
    std::atomic<bool>     stop_{false};
    std::atomic<uint64_t> received_{0};
    std::atomic<uint64_t> oversize_{0};
};

}  // namespace sim
}  // namespace vf

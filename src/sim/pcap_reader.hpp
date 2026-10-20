#pragma once
//
// src/sim/pcap_reader.hpp - classic libpcap files, read whole into memory.
//
// Just enough of the format to replay captures: microsecond and nanosecond
// variants in either byte order, Ethernet link type only. pcapng is not
// supported; convert with `editcap -F pcap in.pcapng out.pcap`.
//

#include <cstdint>
#include <string>
#include <vector>

namespace vf {
namespace sim {

struct PcapFrame {
    uint64_t ts_ns;   ///< capture timestamp
    uint64_t offset;  ///< into Pcap::data
    uint32_t len;     ///< captured bytes
};

struct Pcap {
    std::vector<uint8_t>   data;  ///< the whole file
    std::vector<PcapFrame> frames;
    uint64_t               truncated = 0;  ///< frames captured shorter than they were
};

/// Load `path`. On failure returns false with `err` saying why.
bool pcap_load(const char* path, Pcap& out, std::string& err);

}  // namespace sim
}  // namespace vf

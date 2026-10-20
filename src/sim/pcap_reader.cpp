//
// src/sim/pcap_reader.cpp - classic libpcap files, without libpcap.
//

#include "pcap_reader.hpp"

#include <cstdio>
#include <cstring>

namespace vf {
namespace sim {
namespace {

constexpr uint32_t kMagicUsec = 0xA1B2C3D4u;
constexpr uint32_t kMagicNsec = 0xA1B23C4Du;
constexpr uint32_t kMagicPcapng = 0x0A0D0D0Au;
constexpr uint32_t kLinkEthernet = 1;

uint32_t bswap32(uint32_t v) {
    return (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
}

}  // namespace

bool pcap_load(const char* path, Pcap& out, std::string& err) {
    out = Pcap{};
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        err = std::string("cannot open ") + path;
        return false;
    }
    std::fseek(fp, 0, SEEK_END);
    const long size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    out.data.resize(size > 0 ? static_cast<size_t>(size) : 0);
    const size_t got = out.data.empty() ? 0 : std::fread(out.data.data(), 1, out.data.size(), fp);
    std::fclose(fp);
    if (got != out.data.size() || out.data.size() < 24) {
        err = std::string(path) + ": too short to be a pcap file";
        return false;
    }

    const uint8_t* p = out.data.data();
    uint32_t magic;
    std::memcpy(&magic, p, 4);
    bool swap = false;
    bool nsec = false;
    if (magic == kMagicUsec || magic == kMagicNsec) {
        nsec = magic == kMagicNsec;
    } else if (bswap32(magic) == kMagicUsec || bswap32(magic) == kMagicNsec) {
        swap = true;
        nsec = bswap32(magic) == kMagicNsec;
    } else if (magic == kMagicPcapng) {
        err = std::string(path) +
              ": pcapng is not supported; convert with: editcap -F pcap in.pcapng out.pcap";
        return false;
    } else {
        err = std::string(path) + ": not a pcap file";
        return false;
    }
    auto u32 = [&](size_t off) {
        uint32_t v;
        std::memcpy(&v, p + off, 4);
        return swap ? bswap32(v) : v;
    };
    if (const uint32_t link = u32(20); link != kLinkEthernet) {
        err = std::string(path) + ": link type " + std::to_string(link) +
              " is not Ethernet (1); only Ethernet captures can be replayed";
        return false;
    }

    size_t off = 24;
    const size_t end = out.data.size();
    while (off + 16 <= end) {
        const uint64_t sec = u32(off);
        const uint64_t frac = u32(off + 4);
        const uint32_t incl = u32(off + 8);
        const uint32_t orig = u32(off + 12);
        off += 16;
        if (off + incl > end) break;  // cut-off final record
        out.frames.push_back(PcapFrame{sec * 1000000000ull + (nsec ? frac : frac * 1000ull), off,
                                       incl});
        if (incl < orig) ++out.truncated;
        off += incl;
    }
    return true;
}

}  // namespace sim
}  // namespace vf

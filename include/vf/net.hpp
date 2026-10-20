#pragma once
//
// vf/net.hpp - big-endian loads and the Ethernet / IPv4 / UDP layers.
//
// Every receive path hands the field a whole Ethernet frame: the NIC writes
// one into VRAM on the GPUNetIO path, the pcap holds one in the simulator,
// and the UDP socket source builds a synthetic header in front of each
// datagram so its slots look the same. parse_udp() is the one place that
// peels those layers off, on the host and on the device.
//
// All loads are byte-wise: frame offsets are not aligned (14-byte Ethernet
// header, variable-length IPv4 header, 2-byte MoldUDP64 length prefixes).
//

#include "vf/common.hpp"

namespace vf {

VF_HD VF_FORCEINLINE uint16_t load_be16(const uint8_t* p) {
    return static_cast<uint16_t>((p[0] << 8) | p[1]);
}

VF_HD VF_FORCEINLINE uint32_t load_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

VF_HD VF_FORCEINLINE uint64_t load_be64(const uint8_t* p) {
    return (static_cast<uint64_t>(load_be32(p)) << 32) | load_be32(p + 4);
}

VF_HD VF_FORCEINLINE void store_be16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v >> 8);
    p[1] = static_cast<uint8_t>(v);
}

VF_HD VF_FORCEINLINE void store_be32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v >> 24);
    p[1] = static_cast<uint8_t>(v >> 16);
    p[2] = static_cast<uint8_t>(v >> 8);
    p[3] = static_cast<uint8_t>(v);
}

VF_HD VF_FORCEINLINE void store_be64(uint8_t* p, uint64_t v) {
    store_be32(p, static_cast<uint32_t>(v >> 32));
    store_be32(p + 4, static_cast<uint32_t>(v));
}

constexpr uint32_t kEthHeaderBytes = 14;
constexpr uint32_t kIpv4HeaderBytes = 20;  ///< without options
constexpr uint32_t kUdpHeaderBytes = 8;
/// Eth + IPv4 (no options) + UDP: what the UDP source prepends to a datagram.
constexpr uint32_t kSyntheticHeaderBytes = kEthHeaderBytes + kIpv4HeaderBytes + kUdpHeaderBytes;

/// One UDP datagram located inside a frame.
struct UdpView {
    const uint8_t* payload;
    uint32_t       len;       ///< payload bytes
    uint32_t       src_ip;    ///< host byte order
    uint32_t       dst_ip;    ///< host byte order
    uint16_t       src_port;
    uint16_t       dst_port;
};

/// Locate the UDP payload of an Ethernet frame. Accepts one 802.1Q/802.1ad tag.
/// Returns false for anything that is not a complete, unfragmented IPv4/UDP
/// datagram. Trailing Ethernet padding past the IP total length is ignored.
VF_HD inline bool parse_udp(const uint8_t* f, uint32_t len, UdpView& out) {
    if (len < kEthHeaderBytes) return false;
    uint32_t off = 12;
    uint16_t ethertype = load_be16(f + off);
    if (ethertype == 0x8100 || ethertype == 0x88A8) {
        off += 4;
        if (len < off + 2) return false;
        ethertype = load_be16(f + off);
    }
    off += 2;
    if (ethertype != 0x0800 || len < off + kIpv4HeaderBytes) return false;

    const uint8_t* ip = f + off;
    if ((ip[0] >> 4) != 4) return false;
    const uint32_t ihl = (ip[0] & 0x0Fu) * 4u;
    if (ihl < kIpv4HeaderBytes || ip[9] != 17) return false;    // 17 = UDP
    if (load_be16(ip + 6) & 0x3FFFu) return false;              // MF set or non-zero offset
    const uint32_t ip_total = load_be16(ip + 2);
    if (ip_total < ihl + kUdpHeaderBytes || off + ip_total > len) return false;

    const uint8_t* udp = ip + ihl;
    const uint32_t udp_len = load_be16(udp + 4);
    if (udp_len < kUdpHeaderBytes || udp_len > ip_total - ihl) return false;

    out.payload = udp + kUdpHeaderBytes;
    out.len = udp_len - kUdpHeaderBytes;
    out.src_ip = load_be32(ip + 12);
    out.dst_ip = load_be32(ip + 16);
    out.src_port = load_be16(udp);
    out.dst_port = load_be16(udp + 2);
    return true;
}

/// Write a 42-byte Eth/IPv4/UDP header for a datagram that arrived through the
/// kernel stack, so it can sit in a slot exactly like a frame off the wire.
/// Checksums are left zero: nothing downstream verifies them. Multicast
/// destinations get the matching 01:00:5e group MAC.
VF_HD inline void write_synthetic_header(uint8_t* f, uint32_t src_ip, uint16_t src_port,
                                         uint32_t dst_ip, uint16_t dst_port,
                                         uint32_t payload_len) {
    const bool mcast = (dst_ip >> 28) == 0xE;
    const uint8_t dst_mac[6] = {
        static_cast<uint8_t>(mcast ? 0x01 : 0x02), 0x00, static_cast<uint8_t>(mcast ? 0x5E : 0x00),
        static_cast<uint8_t>(mcast ? (dst_ip >> 16) & 0x7F : 0x00),
        static_cast<uint8_t>(mcast ? dst_ip >> 8 : 0x00),
        static_cast<uint8_t>(mcast ? dst_ip : 0x01)};
    for (int i = 0; i < 6; ++i) f[i] = dst_mac[i];
    for (int i = 6; i < 12; ++i) f[i] = 0;
    f[6] = 0x02;  // locally administered source
    store_be16(f + 12, 0x0800);

    uint8_t* ip = f + kEthHeaderBytes;
    ip[0] = 0x45;
    ip[1] = 0;
    store_be16(ip + 2, static_cast<uint16_t>(kIpv4HeaderBytes + kUdpHeaderBytes + payload_len));
    store_be16(ip + 4, 0);
    store_be16(ip + 6, 0x4000);  // DF
    ip[8] = 64;
    ip[9] = 17;
    store_be16(ip + 10, 0);
    store_be32(ip + 12, src_ip);
    store_be32(ip + 16, dst_ip);

    uint8_t* udp = ip + kIpv4HeaderBytes;
    store_be16(udp, src_port);
    store_be16(udp + 2, dst_port);
    store_be16(udp + 4, static_cast<uint16_t>(kUdpHeaderBytes + payload_len));
    store_be16(udp + 6, 0);
}

}  // namespace vf

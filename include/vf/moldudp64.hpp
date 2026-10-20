#pragma once
//
// vf/moldudp64.hpp - MoldUDP64 downstream packets.
//
//   offset  size  field
//   0       10    session (alphanumeric)
//   10      8     sequence number of the first message in the packet
//   18      2     message count; 0 = heartbeat, 0xFFFF = end of session
//   20      ...   message blocks: [u16 length][length bytes of message]
//
// Message i of a packet has sequence number `seq + i`. That number is what the
// field orders quotes by: it is unique within a session, and the A and B feeds
// carry identical numbers, so a duplicate from the other feed is simply "not
// newer" and drops out.
//

#include "vf/common.hpp"
#include "vf/net.hpp"

namespace vf {
namespace mold {

constexpr uint32_t kHeaderBytes = 20;
constexpr uint32_t kSessionBytes = 10;
constexpr uint16_t kHeartbeat = 0;
constexpr uint16_t kEndOfSession = 0xFFFF;

struct Header {
    uint64_t seq;    ///< sequence number of the first message
    uint16_t count;  ///< message count, or kHeartbeat / kEndOfSession
};

VF_HD inline bool parse_header(const uint8_t* p, uint32_t len, Header& h) {
    if (len < kHeaderBytes) return false;
    h.seq = load_be64(p + kSessionBytes);
    h.count = load_be16(p + kSessionBytes + 8);
    return true;
}

/// Walks the message blocks of one packet.
///
///   mold::Cursor c(payload, len, hdr);
///   while (c.next(msg, msg_len, msg_seq)) { ... }
///   if (c.truncated()) ...   // a block ran past the end of the datagram
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    uint64_t       seq;
    uint32_t       left;  ///< blocks the header promised and we have not read

    VF_HD Cursor(const uint8_t* payload, uint32_t len, const Header& h)
        : p(payload + kHeaderBytes),
          end(payload + len),
          seq(h.seq),
          left(h.count == kEndOfSession ? 0u : h.count) {}

    VF_HD bool next(const uint8_t*& msg, uint32_t& msg_len, uint64_t& msg_seq) {
        if (left == 0 || end - p < 2) return false;
        const uint32_t n = load_be16(p);
        if (n == 0 || static_cast<uint32_t>(end - p) < 2 + n) return false;
        msg = p + 2;
        msg_len = n;
        msg_seq = seq++;
        p += 2 + n;
        --left;
        return true;
    }

    /// True when next() stopped before reading every block the header promised.
    VF_HD bool truncated() const { return left != 0; }
};

}  // namespace mold
}  // namespace vf

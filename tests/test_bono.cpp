//
// tests/test_bono.cpp - BONO v3.3 decoding against the spec's Appendix A, and
// MoldUDP64 / Eth-IPv4-UDP framing.
//
// The hex strings are copied from Appendix A. Where the example *prose*
// disagrees with the tables, the tables win (the bytes always match them):
//   * Example 3 is 39 bytes: it predates the MPV field at offset 39, so the
//     test appends one. Its prose also says expiry day 21; the byte is 0x16 = 22.
//   * Example 6's prose gives 456789123 ns; the bytes say ...124.
//   * Example 8 is "A" (ask side), not bid side, and its byte ranges are for a
//     short form; the 4-byte fields sit at 10-13 and 14-17.
//   * Example 10 lists the wrong byte ranges; the table has Original Cross ID
//     at 9, Original Price at 13, Original Volume at 17.
//

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vf/bono.hpp"
#include "vf/moldudp64.hpp"
#include "vf/net.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                        \
    } while (0)

std::vector<uint8_t> hex(const char* s) {
    std::vector<uint8_t> out;
    unsigned v = 0;
    int nibbles = 0;
    for (; *s; ++s) {
        const char c = *s;
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else continue;
        v = (v << 4) | static_cast<unsigned>(d);
        if (++nibbles == 2) {
            out.push_back(static_cast<uint8_t>(v));
            v = 0;
            nibbles = 0;
        }
    }
    return out;
}

bool same(const char* field, size_t n, const char* want) {
    std::string padded(want);
    padded.resize(n, ' ');
    return std::memcmp(field, padded.data(), n) == 0;
}

void test_appendix_a() {
    using namespace vf::bono;

    // Example 1 - Timestamp: 34200 s = 9:30:00.
    {
        const auto m = hex("54 00 00 85 98");
        CHECK(m.size() == message_size(kTimestamp));
        Timestamp t{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), t));
        CHECK(t.seconds == 34200);
    }
    // Example 2 - System Event: Start of Opening Process, interface 3.0.
    {
        const auto m = hex("53 07 5B CD 15 51 03 00");
        CHECK(m.size() == message_size(kSystemEvent));
        SystemEvent e{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), e));
        CHECK(e.nanos == 123456789);
        CHECK(e.code == 'Q');
        CHECK(e.version == 3 && e.subversion == 0);
    }
    // Example 3 - Options Directory. 39 bytes in the spec; MPV appended.
    {
        auto m = hex("44 0D FB 38 D3 00 01 4D 91 4F 49 48 31 20 20 0B 01 16 00 04 70 B8 43 02 4F"
                     "49 48 20 20 20 20 20 20 20 20 20 20 4E 59");
        CHECK(m.size() == 39);
        Directory d{};
        CHECK(!decode(m.data(), static_cast<uint32_t>(m.size()), d));  // short of the table size
        m.push_back('P');
        CHECK(m.size() == message_size(kDirectory));
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), d));
        CHECK(d.nanos == 234567891);
        CHECK(d.option_id == 85393);
        CHECK(same(d.symbol, sizeof d.symbol, "OIH1"));
        CHECK(d.exp_year == 11 && d.exp_month == 1 && d.exp_day == 22);
        CHECK(d.strike == 291000);  // $29.1000
        CHECK(d.option_type == 'C');
        CHECK(d.source == 2);
        CHECK(same(d.underlying, sizeof d.underlying, "OIH"));
        CHECK(d.closing_type == 'N');
        CHECK(d.tradable == 'Y');
        CHECK(d.mpv == 'P');
    }
    // Example 4 - Security Open.
    {
        const auto m = hex("4F 14 9A A4 40 00 01 4D 91 59");
        SecurityOpen o{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), o));
        CHECK(o.nanos == 345678912 && o.option_id == 85393 && o.state == 'Y');
    }
    // Example 5 - Best Bid AND Ask, short form: $2.50(200) x $2.60(300).
    {
        const auto m = hex("71 1B 3A 0C 83 00 01 4D 91 20 00 FA 00 C8 01 04 01 2C");
        CHECK(m.size() == message_size(kQuoteShort));
        Quote q{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), q));
        CHECK(q.nanos == 456789123 && q.option_id == 85393 && q.condition == ' ');
        CHECK(q.sides == (kSideBid | kSideAsk));
        CHECK(q.bid_price == 25000 && q.bid_size == 200);
        CHECK(q.ask_price == 26000 && q.ask_size == 300);
    }
    // Example 6 - Best Bid AND Ask, long form: $2.50(200) x $2.60(70000).
    {
        const auto m = hex("51 1B 3A 0C 84 00 01 4D 91 20 00 00 61 A8 00 00 00 C8"
                           " 00 00 65 90 00 01 11 70");
        CHECK(m.size() == message_size(kQuoteLong));
        Quote q{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), q));
        CHECK(q.nanos == 456789124);
        CHECK(q.sides == (kSideBid | kSideAsk));
        CHECK(q.bid_price == 25000 && q.bid_size == 200);
        CHECK(q.ask_price == 26000 && q.ask_size == 70000);
    }
    // Example 7 - Best Bid OR Ask, short form, bid: $2.55(300).
    {
        const auto m = hex("62 21 D9 55 22 00 01 4D 91 20 00 FF 01 2C");
        CHECK(m.size() == message_size(kBidShort));
        Quote q{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), q));
        CHECK(q.nanos == 567891234 && q.option_id == 85393);
        CHECK(q.sides == kSideBid);
        CHECK(q.bid_price == 25500 && q.bid_size == 300);
        CHECK(q.ask_price == 0 && q.ask_size == 0);
    }
    // Example 8 - Best Bid OR Ask, long form, *ask*: $2.60(69000).
    {
        const auto m = hex("41 21 D9 55 23 00 01 4D 91 20 00 00 65 90 00 01 0D 88");
        CHECK(m.size() == message_size(kAskLong));
        Quote q{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), q));
        CHECK(q.nanos == 567891235);
        CHECK(q.sides == kSideAsk);
        CHECK(q.ask_price == 26000 && q.ask_size == 69000);
        CHECK(q.bid_price == 0 && q.bid_size == 0);
    }
    // Example 9 - Trade Report: 10 @ $2.55, cross id 12345678, condition I.
    {
        const auto m = hex("52 28 77 61 59 00 01 4D 91 00 BC 61 4E 49 00 00 63 9C 00 00 00 0A");
        CHECK(m.size() == message_size(kTrade));
        Trade t{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), t));
        CHECK(t.nanos == 678912345 && t.option_id == 85393);
        CHECK(t.cross_id == 12345678 && t.condition == 'I');
        CHECK(t.price == 25500 && t.volume == 10);
    }
    // Example 10 - Broken Trade, decoded at the table's offsets.
    {
        const auto m = hex("58 2F 09 11 80 00 01 4D 91 00 BC 61 4E 00 00 63 9C 00 00 00 0A");
        CHECK(m.size() == message_size(kBrokenTrade));
        BrokenTrade b{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), b));
        CHECK(b.nanos == 789123456 && b.option_id == 85393);
        CHECK(b.cross_id == 12345678 && b.price == 25500 && b.volume == 10);
    }
    // Example 11 - Trading Action: halted.
    {
        const auto m = hex("48 35 1F 29 07 00 01 4D 91 48");
        TradingAction h{};
        CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), h));
        CHECK(h.nanos == 891234567 && h.option_id == 85393 && h.state == 'H');
    }
}

void test_message_rules() {
    using namespace vf::bono;
    // Sizes from the tables.
    CHECK(message_size('T') == 5 && message_size('S') == 8 && message_size('D') == 40);
    CHECK(message_size('H') == 10 && message_size('O') == 10);
    CHECK(message_size('q') == 18 && message_size('Q') == 26);
    CHECK(message_size('b') == 14 && message_size('a') == 14);
    CHECK(message_size('B') == 18 && message_size('A') == 18);
    CHECK(message_size('R') == 22 && message_size('X') == 21);
    CHECK(message_size('Z') == 0);

    // Too short fails, wrong type fails, longer than the table is accepted.
    auto m = hex("62 21 D9 55 22 00 01 4D 91 20 00 FF 01 2C");
    Quote q{};
    CHECK(!decode(m.data(), 13, q));
    Trade t{};
    CHECK(!decode(m.data(), static_cast<uint32_t>(m.size()), t));
    m.push_back(0xEE);
    CHECK(decode(m.data(), static_cast<uint32_t>(m.size()), q) && q.bid_price == 25500);

    // A 4-byte price with the sign bit set is not a price.
    auto neg = hex("42 00 00 00 01 00 00 00 07 20 FF FF FF FF 00 00 00 05");
    CHECK(decode(neg.data(), static_cast<uint32_t>(neg.size()), q));
    CHECK(q.sides == kSideBid && q.bid_price == 0 && q.bid_size == 5);

    // Short form: the largest 2-byte price is $655.35.
    auto big = hex("61 00 00 00 01 00 00 00 07 20 FF FF FF FF");
    CHECK(decode(big.data(), static_cast<uint32_t>(big.size()), q));
    CHECK(q.sides == kSideAsk && q.ask_price == 6553500 && q.ask_size == 65535);
}

/// Build a MoldUDP64 packet holding `msgs`, starting at `seq`. A negative
/// `count` writes the number of messages; otherwise `count` goes in verbatim.
std::vector<uint8_t> mold_packet(uint64_t seq, const std::vector<std::vector<uint8_t>>& msgs,
                                 int count = -1) {
    std::vector<uint8_t> p(vf::mold::kHeaderBytes);
    std::memcpy(p.data(), "SESSION001", 10);
    vf::store_be64(p.data() + 10, seq);
    vf::store_be16(p.data() + 18, static_cast<uint16_t>(count < 0 ? msgs.size() : count));
    for (const auto& m : msgs) {
        uint8_t len[2];
        vf::store_be16(len, static_cast<uint16_t>(m.size()));
        p.insert(p.end(), len, len + 2);
        p.insert(p.end(), m.begin(), m.end());
    }
    return p;
}

void test_moldudp64() {
    using namespace vf;
    const auto t = hex("54 00 00 85 98");
    const auto b = hex("62 21 D9 55 22 00 01 4D 91 20 00 FF 01 2C");
    const auto a = hex("41 21 D9 55 23 00 01 4D 91 20 00 00 65 90 00 01 0D 88");

    // Multi-message packet: sequence numbers are seq, seq+1, ...
    {
        const auto pkt = mold_packet(1000, {t, b, a});
        mold::Header h{};
        CHECK(mold::parse_header(pkt.data(), static_cast<uint32_t>(pkt.size()), h));
        CHECK(h.seq == 1000 && h.count == 3);
        mold::Cursor c(pkt.data(), static_cast<uint32_t>(pkt.size()), h);
        const uint8_t* msg = nullptr;
        uint32_t len = 0;
        uint64_t seq = 0;
        std::vector<std::pair<uint8_t, uint64_t>> seen;
        while (c.next(msg, len, seq)) {
            CHECK(len == bono::message_size(msg[0]));
            seen.emplace_back(msg[0], seq);
        }
        CHECK(!c.truncated());
        CHECK(seen.size() == 3);
        CHECK(seen.size() == 3 && seen[0] == std::make_pair(uint8_t('T'), uint64_t(1000)) &&
              seen[1] == std::make_pair(uint8_t('b'), uint64_t(1001)) &&
              seen[2] == std::make_pair(uint8_t('A'), uint64_t(1002)));
    }
    // Heartbeat: count 0, no blocks.
    {
        const auto pkt = mold_packet(77, {}, mold::kHeartbeat);
        mold::Header h{};
        CHECK(mold::parse_header(pkt.data(), static_cast<uint32_t>(pkt.size()), h));
        CHECK(h.count == mold::kHeartbeat && h.seq == 77);
        mold::Cursor c(pkt.data(), static_cast<uint32_t>(pkt.size()), h);
        const uint8_t* msg = nullptr;
        uint32_t len = 0;
        uint64_t seq = 0;
        CHECK(!c.next(msg, len, seq) && !c.truncated());
    }
    // End of session: count 0xFFFF, no blocks, not "truncated".
    {
        const auto pkt = mold_packet(78, {}, mold::kEndOfSession);
        mold::Header h{};
        CHECK(mold::parse_header(pkt.data(), static_cast<uint32_t>(pkt.size()), h));
        CHECK(h.count == mold::kEndOfSession);
        mold::Cursor c(pkt.data(), static_cast<uint32_t>(pkt.size()), h);
        const uint8_t* msg = nullptr;
        uint32_t len = 0;
        uint64_t seq = 0;
        CHECK(!c.next(msg, len, seq) && !c.truncated());
    }
    // A block that runs past the datagram stops the walk and is reported.
    {
        auto pkt = mold_packet(5, {t, b});
        pkt.resize(pkt.size() - 3);
        mold::Header h{};
        CHECK(mold::parse_header(pkt.data(), static_cast<uint32_t>(pkt.size()), h));
        mold::Cursor c(pkt.data(), static_cast<uint32_t>(pkt.size()), h);
        const uint8_t* msg = nullptr;
        uint32_t len = 0;
        uint64_t seq = 0;
        int n = 0;
        while (c.next(msg, len, seq)) ++n;
        CHECK(n == 1 && c.truncated());
    }
    // Header shorter than 20 bytes.
    {
        const auto pkt = mold_packet(1, {t});
        mold::Header h{};
        CHECK(!mold::parse_header(pkt.data(), 19, h));
    }
}

void test_udp_framing() {
    using namespace vf;
    const auto payload = mold_packet(1, {hex("54 00 00 85 98")});

    // Synthetic header round trip: what the UDP source writes, parse_udp reads.
    std::vector<uint8_t> f(kSyntheticHeaderBytes + payload.size());
    const uint32_t src = 0x0A000002, group = 0xE9360C6F;  // 10.0.0.2 -> 233.54.12.111
    write_synthetic_header(f.data(), src, 40000, group, 18001,
                           static_cast<uint32_t>(payload.size()));
    std::memcpy(f.data() + kSyntheticHeaderBytes, payload.data(), payload.size());
    CHECK(f[0] == 0x01 && f[1] == 0x00 && f[2] == 0x5E && f[3] == 0x36 && f[4] == 0x0C &&
          f[5] == 0x6F);

    UdpView u{};
    CHECK(parse_udp(f.data(), static_cast<uint32_t>(f.size()), u));
    CHECK(u.len == payload.size() && u.payload == f.data() + kSyntheticHeaderBytes);
    CHECK(u.src_ip == src && u.dst_ip == group && u.src_port == 40000 && u.dst_port == 18001);

    // Ethernet padding past the IP total length is ignored.
    auto padded = f;
    padded.resize(f.size() + 16, 0);
    CHECK(parse_udp(padded.data(), static_cast<uint32_t>(padded.size()), u) &&
          u.len == payload.size());

    // One VLAN tag is accepted.
    std::vector<uint8_t> tagged(f.begin(), f.begin() + 12);
    const uint8_t tag[4] = {0x81, 0x00, 0x00, 0x64};
    tagged.insert(tagged.end(), tag, tag + 4);
    tagged.insert(tagged.end(), f.begin() + 12, f.end());
    CHECK(parse_udp(tagged.data(), static_cast<uint32_t>(tagged.size()), u) &&
          u.dst_port == 18001);

    // Not UDP, fragmented, or cut short: rejected.
    auto tcp = f;
    tcp[kEthHeaderBytes + 9] = 6;
    CHECK(!parse_udp(tcp.data(), static_cast<uint32_t>(tcp.size()), u));
    auto frag = f;
    frag[kEthHeaderBytes + 6] = 0x20;  // MF
    CHECK(!parse_udp(frag.data(), static_cast<uint32_t>(frag.size()), u));
    CHECK(!parse_udp(f.data(), static_cast<uint32_t>(f.size()) - 1, u));
    auto arp = f;
    arp[12] = 0x08;
    arp[13] = 0x06;
    CHECK(!parse_udp(arp.data(), static_cast<uint32_t>(arp.size()), u));
}

}  // namespace

int main() {
    test_appendix_a();
    test_message_rules();
    test_moldudp64();
    test_udp_framing();
    std::printf("test_bono: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

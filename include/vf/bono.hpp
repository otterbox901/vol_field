#pragma once
//
// vf/bono.hpp - Nasdaq Best of Nasdaq Options (BONO) v3.3 messages.
//
// The tables in the spec are authoritative; Appendix A's example bytes match
// them, while some of its prose does not (see tests/test_bono.cpp).
//
//   * Integers are unsigned big-endian unless noted.
//   * Prices: 4-byte fields have 4 implied decimals. 2-byte (short form)
//     fields are pennies. Both decode to the same unit here: 1e-4 dollars.
//   * Strike is 4 bytes with 4 implied decimals.
//   * Alpha fields are left-justified and space-padded.
//   * Time: a T message (5 B) carries seconds past midnight; every other
//     message carries only the nanoseconds at offset 1.
//
// Message sizes: T5 S8 D40 H10 O10 q18 Q26 b/a14 B/A18 R22 X21. A message
// shorter than its table size is malformed; a longer one is accepted, so a
// future field appended at the end does not break decoding.
//
// Quotes (q/Q/b/a/B/A) travel only in the "Q" channel group; T/S/D/H/O go to
// both groups; trades (R/X) only to "T". The field consumes the Q group.
//

#include "vf/common.hpp"
#include "vf/net.hpp"

namespace vf {
namespace bono {

constexpr uint8_t kTimestamp     = 'T';
constexpr uint8_t kSystemEvent   = 'S';
constexpr uint8_t kDirectory     = 'D';
constexpr uint8_t kTradingAction = 'H';
constexpr uint8_t kSecurityOpen  = 'O';
constexpr uint8_t kQuoteShort    = 'q';  ///< both sides, 2-byte prices and sizes
constexpr uint8_t kQuoteLong     = 'Q';  ///< both sides, 4-byte prices and sizes
constexpr uint8_t kBidShort      = 'b';
constexpr uint8_t kAskShort      = 'a';
constexpr uint8_t kBidLong       = 'B';
constexpr uint8_t kAskLong       = 'A';
constexpr uint8_t kTrade         = 'R';
constexpr uint8_t kBrokenTrade   = 'X';

constexpr uint32_t kUnderlyingBytes = 13;
constexpr uint32_t kSymbolBytes = 6;

/// Table size of a message type, or 0 for a type this spec does not define.
VF_HD constexpr uint32_t message_size(uint8_t type) {
    switch (type) {
        case kTimestamp:     return 5;
        case kSystemEvent:   return 8;
        case kDirectory:     return 40;
        case kTradingAction: return 10;
        case kSecurityOpen:  return 10;
        case kQuoteShort:    return 18;
        case kQuoteLong:     return 26;
        case kBidShort:
        case kAskShort:      return 14;
        case kBidLong:
        case kAskLong:       return 18;
        case kTrade:         return 22;
        case kBrokenTrade:   return 21;
        default:             return 0;
    }
}

VF_HD constexpr bool is_quote(uint8_t type) {
    return type == kQuoteShort || type == kQuoteLong || type == kBidShort ||
           type == kAskShort || type == kBidLong || type == kAskLong;
}

/// A 4-byte price is "signed positive"; anything with the sign bit set is not
/// a price, and decodes as 0 (no price) rather than as a huge one.
VF_HD VF_FORCEINLINE uint32_t long_price(const uint8_t* p) {
    const uint32_t v = load_be32(p);
    return (v & 0x80000000u) ? 0u : v;
}

/// 2-byte prices are pennies: x100 gives the common 1e-4 unit.
VF_HD VF_FORCEINLINE uint32_t short_price(const uint8_t* p) { return load_be16(p) * 100u; }

struct Timestamp {
    uint32_t seconds;  ///< since midnight
};

struct SystemEvent {
    uint32_t nanos;
    char     code;  ///< O S Q N L E C
    uint8_t  version;
    uint8_t  subversion;
};

struct Directory {
    uint32_t nanos;
    uint32_t option_id;
    char     symbol[kSymbolBytes];          ///< option root, space-padded
    uint8_t  exp_year;                      ///< last two digits
    uint8_t  exp_month;
    uint8_t  exp_day;
    uint32_t strike;                        ///< 1e-4 dollars
    char     option_type;                   ///< 'C' or 'P'
    uint8_t  source;
    char     underlying[kUnderlyingBytes];  ///< space-padded
    char     closing_type;                  ///< 'N' normal, 'L' late
    char     tradable;                      ///< 'Y' / 'N'
    char     mpv;                           ///< 'E' / 'S' / 'P'
};

struct TradingAction {
    uint32_t nanos;
    uint32_t option_id;
    char     state;  ///< 'H' halted, 'T' trading
};

struct SecurityOpen {
    uint32_t nanos;
    uint32_t option_id;
    char     state;  ///< 'Y' open for auto-execution, 'N' closed
};

constexpr uint8_t kSideBid = 1;
constexpr uint8_t kSideAsk = 2;

/// Any of the six quote messages, normalised: which sides it carries, and
/// prices in 1e-4 dollars whatever the wire form. A one-sided update leaves
/// the other side's fields zero and its bit clear in `sides`.
struct Quote {
    uint32_t nanos;
    uint32_t option_id;
    char     condition;  ///< ' ' regular, 'F' non-firm, 'R' rotational, 'X'/'Y' one side not firm
    uint8_t  sides;      ///< kSideBid | kSideAsk
    uint32_t bid_price;
    uint32_t bid_size;
    uint32_t ask_price;
    uint32_t ask_size;
};

struct Trade {
    uint32_t nanos;
    uint32_t option_id;
    uint32_t cross_id;
    char     condition;
    uint32_t price;   ///< 1e-4 dollars
    uint32_t volume;
};

struct BrokenTrade {
    uint32_t nanos;
    uint32_t option_id;
    uint32_t cross_id;
    uint32_t price;   ///< 1e-4 dollars
    uint32_t volume;
};

/// Each decode_* returns false when `m` is not that message type or is
/// shorter than the table size.
VF_HD inline bool length_ok(const uint8_t* m, uint32_t len) {
    return len >= 1 && message_size(m[0]) != 0 && len >= message_size(m[0]);
}

VF_HD inline bool decode(const uint8_t* m, uint32_t len, Timestamp& out) {
    if (!length_ok(m, len) || m[0] != kTimestamp) return false;
    out.seconds = load_be32(m + 1);
    return true;
}

VF_HD inline bool decode(const uint8_t* m, uint32_t len, SystemEvent& out) {
    if (!length_ok(m, len) || m[0] != kSystemEvent) return false;
    out.nanos = load_be32(m + 1);
    out.code = static_cast<char>(m[5]);
    out.version = m[6];
    out.subversion = m[7];
    return true;
}

VF_HD inline bool decode(const uint8_t* m, uint32_t len, Directory& out) {
    if (!length_ok(m, len) || m[0] != kDirectory) return false;
    out.nanos = load_be32(m + 1);
    out.option_id = load_be32(m + 5);
    for (uint32_t i = 0; i < kSymbolBytes; ++i) out.symbol[i] = static_cast<char>(m[9 + i]);
    out.exp_year = m[15];
    out.exp_month = m[16];
    out.exp_day = m[17];
    out.strike = load_be32(m + 18);
    out.option_type = static_cast<char>(m[22]);
    out.source = m[23];
    for (uint32_t i = 0; i < kUnderlyingBytes; ++i)
        out.underlying[i] = static_cast<char>(m[24 + i]);
    out.closing_type = static_cast<char>(m[37]);
    out.tradable = static_cast<char>(m[38]);
    out.mpv = static_cast<char>(m[39]);
    return true;
}

VF_HD inline bool decode(const uint8_t* m, uint32_t len, TradingAction& out) {
    if (!length_ok(m, len) || m[0] != kTradingAction) return false;
    out.nanos = load_be32(m + 1);
    out.option_id = load_be32(m + 5);
    out.state = static_cast<char>(m[9]);
    return true;
}

VF_HD inline bool decode(const uint8_t* m, uint32_t len, SecurityOpen& out) {
    if (!length_ok(m, len) || m[0] != kSecurityOpen) return false;
    out.nanos = load_be32(m + 1);
    out.option_id = load_be32(m + 5);
    out.state = static_cast<char>(m[9]);
    return true;
}

VF_HD inline bool decode(const uint8_t* m, uint32_t len, Quote& out) {
    if (!length_ok(m, len) || !is_quote(m[0])) return false;
    out.nanos = load_be32(m + 1);
    out.option_id = load_be32(m + 5);
    out.condition = static_cast<char>(m[9]);
    out.bid_price = out.bid_size = out.ask_price = out.ask_size = 0;
    switch (m[0]) {
        case kQuoteShort:
            out.sides = kSideBid | kSideAsk;
            out.bid_price = short_price(m + 10);
            out.bid_size = load_be16(m + 12);
            out.ask_price = short_price(m + 14);
            out.ask_size = load_be16(m + 16);
            break;
        case kQuoteLong:
            out.sides = kSideBid | kSideAsk;
            out.bid_price = long_price(m + 10);
            out.bid_size = load_be32(m + 14);
            out.ask_price = long_price(m + 18);
            out.ask_size = load_be32(m + 22);
            break;
        case kBidShort:
            out.sides = kSideBid;
            out.bid_price = short_price(m + 10);
            out.bid_size = load_be16(m + 12);
            break;
        case kAskShort:
            out.sides = kSideAsk;
            out.ask_price = short_price(m + 10);
            out.ask_size = load_be16(m + 12);
            break;
        case kBidLong:
            out.sides = kSideBid;
            out.bid_price = long_price(m + 10);
            out.bid_size = load_be32(m + 14);
            break;
        default:  // kAskLong
            out.sides = kSideAsk;
            out.ask_price = long_price(m + 10);
            out.ask_size = load_be32(m + 14);
            break;
    }
    return true;
}

VF_HD inline bool decode(const uint8_t* m, uint32_t len, Trade& out) {
    if (!length_ok(m, len) || m[0] != kTrade) return false;
    out.nanos = load_be32(m + 1);
    out.option_id = load_be32(m + 5);
    out.cross_id = load_be32(m + 9);
    out.condition = static_cast<char>(m[13]);
    out.price = long_price(m + 14);
    out.volume = load_be32(m + 18);
    return true;
}

VF_HD inline bool decode(const uint8_t* m, uint32_t len, BrokenTrade& out) {
    if (!length_ok(m, len) || m[0] != kBrokenTrade) return false;
    out.nanos = load_be32(m + 1);
    out.option_id = load_be32(m + 5);
    out.cross_id = load_be32(m + 9);
    out.price = long_price(m + 13);
    out.volume = load_be32(m + 17);
    return true;
}

}  // namespace bono
}  // namespace vf

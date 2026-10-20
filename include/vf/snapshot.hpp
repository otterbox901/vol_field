#pragma once
//
// vf/snapshot.hpp - the surface snapshot the host shares with Python.
//
// It lives in POSIX shared memory (/dev/shm/vol_field by default), mmapped by
// the host and registered with cudaHostRegister(Mapped), so the surface kernel
// writes it straight over PCIe: the host CPU never copies or touches the data.
// tools/vfsnap.py mirrors these structs as numpy dtypes; the static_asserts
// below pin every offset it relies on. Change either side together, and bump
// kSnapLayoutVersion.
//
//   [SnapHeader, 512 B][ExpirySnap x kMaxExpiries][PairSnap x kMaxPairs]
//
// Only the first n_expiries / n_pairs records are current. Writer protocol
// (src/field/field.cu): seqlock -> odd, write records, __threadfence_system(),
// write the header, seqlock -> even. A reader copies everything between two
// equal, even seqlock reads, and retries otherwise.
//

#include <cstddef>

#include "vf/field.hpp"

namespace vf {

constexpr char kSnapMagic[8] = {'V', 'O', 'L', 'F', 'I', 'E', 'L', 'D'};
constexpr uint32_t kSnapLayoutVersion = 1;
constexpr uint32_t kSnapRunning = 0;
constexpr uint32_t kSnapFinal = 1;  ///< the run is over; this is its last surface

struct SnapHeader {
    char     magic[8];                                         // 0
    uint32_t layout_version;                                   // 8
    uint32_t header_bytes;                                     // 12
    uint64_t seqlock;                                          // 16: odd while being written
    uint64_t snapshot_id;                                      // 24
    uint64_t host_ns;                                          // 32: host steady clock at launch
    uint32_t state;                                            // 40: kSnapRunning / kSnapFinal
    uint32_t feed_sec;                                         // 44: seconds past midnight
    uint32_t n_expiries;                                       // 48
    uint32_t n_pairs;                                          // 52
    uint32_t n_options;                                        // 56
    uint32_t n_underlyings;                                    // 60
    uint32_t max_expiries;                                     // 64
    uint32_t max_pairs;                                        // 68
    uint32_t expiry_rec_bytes;                                 // 72
    uint32_t pair_rec_bytes;                                   // 76
    uint64_t expiry_offset;                                    // 80
    uint64_t pair_offset;                                      // 88
    double   rate;                                             // 96
    int32_t  trade_days;                                       // 104: days since 1970-01-01
    uint32_t overflow;                                         // 108
    char     underlyings[kMaxUnderlyings][kUnderlyingField];   // 112
    FeedCounters counters;                                     // 240: summed over blocks
    uint8_t  reserved[144];                                    // 368
};

struct ExpirySnap {
    uint32_t und;         // 0
    uint32_t yymmdd;      // 4
    int32_t  days;        // 8: expiry date - trade date
    uint32_t n_pairs;     // 12
    double   T;           // 16: years, at feed_sec
    double   forward;     // 24: NaN until parity finds one
    uint32_t n_forward;   // 32: pairs that voted
    uint32_t pad;         // 36
    double   reserved;    // 40
};

constexpr uint32_t kPairHasCall = 1u << 0;
constexpr uint32_t kPairHasPut = 1u << 1;
constexpr uint32_t kPairOtmCall = 1u << 2;   ///< iv_otm comes from the call (K >= F)
constexpr uint32_t kPairCallLive = 1u << 3;  ///< tradable and not halted
constexpr uint32_t kPairPutLive = 1u << 4;

struct PairSnap {
    uint32_t und;           // 0
    uint32_t expiry;        // 4: index into the ExpirySnap array
    double   strike;        // 8: dollars
    double   forward;       // 16
    double   T;             // 24
    float    iv_call;       // 32: NaN = none
    float    iv_put;        // 36
    float    iv_otm;        // 40: the out-of-the-money side, what the surface plots
    float    call_mid;      // 44: NaN unless two-sided
    float    put_mid;       // 48
    uint32_t flags;         // 52: kPair*
    uint32_t call_version;  // 56: quote version the call IV was solved from
    uint32_t put_version;   // 60
};

constexpr uint64_t kSnapExpiryOffset = sizeof(SnapHeader);
constexpr uint64_t kSnapPairOffset = kSnapExpiryOffset + uint64_t{kMaxExpiries} * 48;
constexpr uint64_t kSnapBytes = kSnapPairOffset + uint64_t{kMaxPairs} * 64;

static_assert(sizeof(SnapHeader) == 512, "snapshot header size");
static_assert(offsetof(SnapHeader, seqlock) == 16, "");
static_assert(offsetof(SnapHeader, state) == 40, "");
static_assert(offsetof(SnapHeader, expiry_offset) == 80, "");
static_assert(offsetof(SnapHeader, rate) == 96, "");
static_assert(offsetof(SnapHeader, underlyings) == 112, "");
static_assert(offsetof(SnapHeader, counters) == 240, "");
static_assert(sizeof(ExpirySnap) == 48, "expiry record size");
static_assert(offsetof(ExpirySnap, forward) == 24, "");
static_assert(sizeof(PairSnap) == 64, "pair record size");
static_assert(offsetof(PairSnap, iv_call) == 32, "");
static_assert(offsetof(PairSnap, flags) == 52, "");

/// The shared-memory mapping, from the host's side.
struct Snapshot {
    char        path[256];     ///< the file backing it
    void*       host = nullptr;
    SnapHeader* dev = nullptr; ///< the same memory, as the device sees it
    size_t      bytes = 0;
};

/// Create (or reuse) `name` - a /dev/shm name, or a path to any tmpfs file -
/// size it, map it, write the static header, and register it with CUDA.
bool snapshot_open(const char* name, const FieldParams& p, Snapshot& out);

/// Unregister and unmap. The file stays, so readers can still see the final
/// surface after the run ends.
void snapshot_close(Snapshot& s);

/// Seqlock-consistent copy of the header, for status lines. False if the
/// writer kept it busy.
bool snapshot_header(const Snapshot& s, SnapHeader& out);

}  // namespace vf

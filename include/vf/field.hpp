#pragma once
//
// vf/field.hpp - the implied-vol field: device tables in VRAM.
//
// The field is written concurrently by every receive thread as quotes arrive,
// and read concurrently by the surface kernel. A batch of frames is processed
// in no particular order (one thread per frame), so every update has to give
// the same result whatever order it lands in:
//
//   * Quote state sits behind a per-option seqlock. Bid and ask each remember
//     the sequence number of the message that set them, and a writer only
//     applies a side whose message is newer. Latest wins per side, one-sided
//     b/a updates merge, and an A/B-feed duplicate is just "not newer".
//   * Status flags (tradable, halted, open) are 64-bit words (seq << 1) | bit
//     updated with atomicMax: again, the newest message wins.
//   * The IV is one 64-bit word (quote version << 32) | float bits, replaced by
//     CAS only by an IV for the same or a newer quote version.
//   * Directory messages insert into lock-free open-addressing tables, in a
//     first pass over the batch; quotes, halts (H) and opens (O) are applied in
//     a second pass after a __syncthreads(), so a directory entry and anything
//     addressed to the same option in one batch always map.
//
// Capacities are compile-time constants so the snapshot layout is fixed.
// This header is plain C++: the device code is in vf/field_device.cuh.
//

#include "vf/common.hpp"

struct CUstream_st;  // cudaStream_t without the CUDA headers

namespace vf {

constexpr uint32_t kMaxUnderlyings = 8;
constexpr uint32_t kUnderlyingField = 16;          ///< BONO's 13 chars, space-padded, + padding
constexpr uint32_t kMaxOptions = 1u << 16;
constexpr uint32_t kOptionSlots = 2 * kMaxOptions; ///< hash slots: load factor <= 0.5
constexpr uint32_t kMaxExpiries = 256;
constexpr uint32_t kExpirySlots = 2 * kMaxExpiries;
constexpr uint32_t kMaxPairs = 1u << 15;
constexpr uint32_t kPairSlots = 2 * kMaxPairs;
constexpr uint32_t kMaxPairsPerExpiry = 1024;      ///< strikes listed per expiry

constexpr uint32_t kEmptyKey32 = 0xFFFFFFFFu;
constexpr unsigned long long kEmptyKey64 = ~0ull;
constexpr int32_t kPending = -1;  ///< hash value while the claimant initialises the entry
constexpr int32_t kFull = -2;     ///< key claimed, but its dense table was already full

/// Per-option quote state behind a seqlock. `lock` is odd while a writer holds
/// it; it advances by 2 per write, so lock / 2 is the quote version.
struct QuoteState {
    unsigned int lock;
    uint32_t bid_price;           ///< 1e-4 dollars; 0 = no bid
    uint32_t bid_size;
    uint32_t ask_price;
    uint32_t ask_size;
    uint32_t pad_;
    unsigned long long bid_seq;   ///< sequence number of the message the bid came from
    unsigned long long ask_seq;
};

/// One option. Aligned to a 128-byte line so no two options share one.
struct alignas(128) OptionRec {
    QuoteState q;
    unsigned long long iv;        ///< (quote version << 32) | float bits; NaN = no valid IV
    unsigned long long tradable;  ///< (seq << 1) | tradable
    unsigned long long halted;    ///< (seq << 1) | halted; starts halted, per spec 4.4
    unsigned long long open;      ///< (seq << 1) | open for auto-execution
    // Static fields: written once by the inserting thread, before `ready`.
    uint32_t option_id;
    uint32_t strike;              ///< 1e-4 dollars
    int32_t  expiry;              ///< dense expiry index
    int32_t  pair;                ///< dense pair index
    uint32_t und;                 ///< index into FieldParams::underlyings
    uint32_t call;                ///< 1 = call, 0 = put
    unsigned int ready;
};

/// One (underlying, expiry date).
struct ExpiryRec {
    uint32_t yymmdd;
    uint32_t und;
    int32_t  days;                ///< expiry date minus trade date, in days
    unsigned int n_pairs;         ///< strikes appended to the list (can pass the capacity)
    unsigned long long forward;   ///< double bits of the parity forward; NaN until found
    unsigned int n_forward;       ///< pairs that voted on it
    unsigned int ready;
};

/// One (expiry, strike): the call and put that parity pairs up.
struct PairRec {
    uint32_t expiry;
    uint32_t strike;              ///< 1e-4 dollars
    int32_t  call;                ///< dense option index, -1 until its directory arrives
    int32_t  put;
    unsigned int ready;
};

/// Counters one receive block keeps; only that block writes its entry. Also
/// the snapshot's counter block (the sum over blocks), so plain uint64_t.
struct FeedCounters {
    uint64_t frames;          ///< frames handed to the block
    uint64_t bytes;
    uint64_t not_feed;        ///< not IPv4/UDP, or too short for a MoldUDP64 header
    uint64_t heartbeats;
    uint64_t end_of_session;
    uint64_t messages;        ///< BONO messages walked
    uint64_t malformed;       ///< truncated MoldUDP64 blocks, short or unknown messages
    uint64_t directory;       ///< D messages for a configured underlying
    uint64_t admin;           ///< T, S, H, O, and D for other underlyings
    uint64_t quotes;          ///< quote messages for an option the field holds
    uint64_t stale;           ///< ... that applied no side: duplicate, or a newer one came first
    uint64_t unmapped;        ///< quotes for an option the field does not hold
    uint64_t iv_solved;       ///< on tick: OTM side, two-sided, forward known
    uint64_t iv_failed;       ///< ... and the solver returned NaN
    uint64_t trades;          ///< R and X, counted only
    uint64_t max_seq;         ///< highest message sequence number seen (max, not sum)
};
constexpr uint32_t kNumCounters = sizeof(FeedCounters) / sizeof(uint64_t);
constexpr uint32_t kMaxSeqCounter = 15;  ///< index of max_seq: reduced with max, not +
static_assert(kNumCounters == 16, "FeedCounters is mirrored by tools/vfsnap.py");

/// Host-chosen parameters, passed by value in every launch.
struct FieldParams {
    char     underlyings[kMaxUnderlyings][kUnderlyingField];  ///< space-padded to 13
    uint32_t n_underlyings;
    int32_t  trade_days;   ///< --date, days since 1970-01-01
    double   rate;         ///< --rate, continuously compounded
};

/// The device tables. Plain pointers, passed by value to every kernel.
struct FieldDev {
    FieldParams p;

    unsigned int*       opt_keys;   ///< option ID, kEmptyKey32 = free
    int32_t*            opt_vals;   ///< dense option index
    OptionRec*          opts;
    unsigned int*       n_opts;

    unsigned int*       exp_keys;   ///< (und << 24) | yymmdd
    int32_t*            exp_vals;
    ExpiryRec*          exps;
    unsigned int*       n_exps;
    int32_t*            exp_pairs;  ///< kMaxPairsPerExpiry pair indices per expiry, -1 = none

    unsigned long long* pair_keys;  ///< (expiry << 32) | strike
    int32_t*            pair_vals;
    PairRec*            pairs;
    unsigned int*       n_pairs;

    unsigned int*       feed_sec;   ///< latest T message (atomicMax)
    unsigned int*       overflow;   ///< inserts refused because a table was full
    FeedCounters*       counters;   ///< one per receive block
    uint32_t            n_blocks;
};

struct SnapHeader;  // vf/snapshot.hpp

/// Owns the device allocations and the surface kernel's state.
struct Field;

/// Allocate and initialise the tables in VRAM for `n_blocks` receive blocks.
Field* field_create(const FieldParams& p, uint32_t n_blocks);
void field_destroy(Field* f);
const FieldDev& field_dev(const Field* f);

/// Fill FieldParams from the run's underlyings, date and rate. Returns false
/// (after printing why) for too many underlyings or one longer than 13 chars.
bool field_params(const char* const* underlyings, uint32_t n, int32_t trade_days, double rate,
                  FieldParams& out);

/// Queue one surface pass on `stream`: re-derive each expiry's forward from
/// consistent quotes, re-solve every option's IV against it (never replacing
/// an IV for a newer quote), then write the snapshot at `snap` (host memory
/// mapped into the device) under its seqlock. Returns at once; the pass runs
/// concurrently with the receive kernel.
void field_launch_surface(Field* f, CUstream_st* stream, SnapHeader* snap, uint64_t host_ns,
                          uint32_t state);

/// Table sizes, read back from the device.
struct FieldSizes {
    uint32_t options;
    uint32_t expiries;
    uint32_t pairs;
    uint32_t overflow;
    uint32_t feed_sec;
};
FieldSizes field_sizes(const Field* f);

/// Copy the per-block counters back (n_blocks entries).
void field_counters(const Field* f, FeedCounters* per_block);

/// Sum per-block counters (max_seq is a max).
FeedCounters counters_sum(const FeedCounters* per_block, uint32_t n_blocks);

}  // namespace vf

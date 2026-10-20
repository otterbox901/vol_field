#pragma once
//
// vf/field_device.cuh - the per-frame code every receive kernel runs.
//
// The GPUNetIO receive kernel, the simulator's kernel and tests/test_field.cu
// all call process_batch() below, so the Ethernet -> MoldUDP64 -> BONO -> field
// logic is literally the same code on every path. The kernels differ only in
// where a batch of frames comes from.
//
// Memory model, briefly: the tables are shared by every block of the receive
// kernel and read by the surface kernel running concurrently on another
// stream. Everything that can change after publication is read with volatile
// loads (they go to L2, past the non-coherent L1) and written with volatile
// stores or atomics, with __threadfence() where an order must be visible.
//

#include <cuda_runtime.h>

#include "vf/black.hpp"
#include "vf/bono.hpp"
#include "vf/field.hpp"
#include "vf/moldudp64.hpp"
#include "vf/net.hpp"

namespace vf {
namespace dev {

/// Threads per receive block, and the most frames one batch can hold. The
/// DOCA kernel's shape: 512 threads splitting batches of up to 1024 frames.
constexpr uint32_t kBlockThreads = 512;
constexpr uint32_t kMaxBatch = 2 * kBlockThreads;

template <class T>
__device__ __forceinline__ T ld(const T* p) {
    return *const_cast<const volatile T*>(p);
}

template <class T>
__device__ __forceinline__ void st(T* p, T v) {
    *const_cast<volatile T*>(p) = v;
}

__device__ __forceinline__ uint32_t mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ uint32_t mix64(unsigned long long k) {
    k ^= k >> 33;
    k *= 0xFF51AFD7ED558CCDull;
    k ^= k >> 33;
    k *= 0xC4CEB9FE1A85EC53ull;
    k ^= k >> 33;
    return static_cast<uint32_t>(k);
}

__device__ __forceinline__ double as_double(unsigned long long bits) {
    return __longlong_as_double(static_cast<long long>(bits));
}

__device__ __forceinline__ unsigned long long double_bits(double v) {
    return static_cast<unsigned long long>(__double_as_longlong(v));
}

// ---------------------------------------------------------------------------
// Lock-free tables
// ---------------------------------------------------------------------------

/// Find or claim `key` in an open-addressing table (linear probing).
///
/// The thread that claims a new key gets `fresh = true`, a dense index from
/// `count`, and `slot`: it must initialise the dense record and then call
/// publish(). Any other thread that meets the key waits until it is published
/// - the claimant is never blocked, so this cannot deadlock under independent
/// thread scheduling. Returns -1 when the dense table is full.
template <class Key>
__device__ int32_t claim(Key* keys, int32_t* vals, uint32_t n_slots, Key key, Key empty,
                         uint32_t hash, unsigned int* count, uint32_t max, bool& fresh,
                         uint32_t& slot) {
    fresh = false;
    const uint32_t mask = n_slots - 1;
    uint32_t h = hash & mask;
    for (uint32_t probe = 0; probe < n_slots; ++probe, h = (h + 1) & mask) {
        Key k = ld(&keys[h]);
        if (k == empty) {
            k = atomicCAS(&keys[h], empty, key);
            if (k == empty) {
                const unsigned int idx = atomicAdd(count, 1u);
                if (idx >= max) {
                    st(&vals[h], kFull);
                    return -1;
                }
                fresh = true;
                slot = h;
                return static_cast<int32_t>(idx);
            }
        }
        if (k == key) {
            int32_t v;
            while ((v = ld(&vals[h])) == kPending) {
            }
            return v < 0 ? -1 : v;
        }
    }
    return -1;
}

/// Make a claimed entry visible: its record first, then the hash value that
/// lets other threads find it.
__device__ __forceinline__ void publish(unsigned int* ready, int32_t* val_slot, int32_t idx) {
    __threadfence();
    st(ready, 1u);
    st(val_slot, idx);
    __threadfence();
}

/// Dense index of an option, or a negative value if the field does not hold it
/// (or its directory entry is still being inserted).
__device__ __forceinline__ int32_t option_find(const FieldDev& f, uint32_t option_id) {
    constexpr uint32_t mask = kOptionSlots - 1;
    uint32_t h = mix32(option_id) & mask;
    for (uint32_t probe = 0; probe < kOptionSlots; ++probe, h = (h + 1) & mask) {
        const unsigned int k = ld(&f.opt_keys[h]);
        if (k == option_id) return ld(&f.opt_vals[h]);
        if (k == kEmptyKey32) return -1;
    }
    return -1;
}

/// The static fields of a published option.
struct OptionStatic {
    uint32_t strike;
    int32_t  expiry;
    uint32_t call;
};

__device__ __forceinline__ OptionStatic option_static(const OptionRec& o) {
    return OptionStatic{ld(&o.strike), ld(&o.expiry), ld(&o.call)};
}

__device__ __forceinline__ bool option_live(const OptionRec& o) {
    return (ld(&o.tradable) & 1ull) && !(ld(&o.halted) & 1ull);
}

__device__ __forceinline__ double expiry_forward(const ExpiryRec& x) {
    return as_double(ld(&x.forward));
}

__device__ inline int underlying_index(const FieldParams& p, const char* und) {
    for (uint32_t u = 0; u < p.n_underlyings; ++u) {
        bool eq = true;
        for (uint32_t i = 0; i < bono::kUnderlyingBytes && eq; ++i)
            eq = p.underlyings[u][i] == und[i];
        if (eq) return static_cast<int>(u);
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Quotes: seqlock writer and reader
// ---------------------------------------------------------------------------

/// A consistent copy of one option's quote.
struct QuoteCopy {
    uint32_t bid_price;
    uint32_t bid_size;
    uint32_t ask_price;
    uint32_t ask_size;
    uint32_t version;  ///< lock / 2 at the time of the copy
};

/// Apply the sides of `q` that are newer than what the option shows.
///
/// Takes the seqlock with a CAS spin (odd = held), so readers retry rather
/// than see half an update, and two writers to one option serialise. `now`
/// receives the state as left by this write, copied while still holding the
/// lock, and `moved` whether either price changed. Returns false when no side
/// was newer (an older message, or an A/B duplicate); the lock still advances.
__device__ inline bool apply_quote_locked(QuoteState& s, const bono::Quote& q,
                                          unsigned long long seq, QuoteCopy& now, bool& moved) {
    unsigned int v;
    for (;;) {
        v = ld(&s.lock);
        if (!(v & 1u) && atomicCAS(&s.lock, v, v + 1u) == v) break;
    }
    __threadfence();

    volatile QuoteState& w = s;
    const uint32_t old_bid = w.bid_price;
    const uint32_t old_ask = w.ask_price;
    bool applied = false;
    if ((q.sides & bono::kSideBid) && seq > w.bid_seq) {
        w.bid_seq = seq;
        w.bid_price = q.bid_price;
        w.bid_size = q.bid_size;
        applied = true;
    }
    if ((q.sides & bono::kSideAsk) && seq > w.ask_seq) {
        w.ask_seq = seq;
        w.ask_price = q.ask_price;
        w.ask_size = q.ask_size;
        applied = true;
    }
    now.bid_price = w.bid_price;
    now.bid_size = w.bid_size;
    now.ask_price = w.ask_price;
    now.ask_size = w.ask_size;
    now.version = (v + 2u) >> 1;
    moved = now.bid_price != old_bid || now.ask_price != old_ask;

    __threadfence();
    atomicAdd(&s.lock, 1u);  // even again: released
    return applied;
}

/// Seqlock reader: retries until it copies a state no writer touched meanwhile.
__device__ inline void read_quote(const QuoteState& s, QuoteCopy& out) {
    const volatile QuoteState& r = s;
    for (;;) {
        const unsigned int v1 = r.lock;
        if (v1 & 1u) continue;
        __threadfence();
        out.bid_price = r.bid_price;
        out.bid_size = r.bid_size;
        out.ask_price = r.ask_price;
        out.ask_size = r.ask_size;
        __threadfence();
        if (r.lock == v1) {
            out.version = v1 >> 1;
            return;
        }
    }
}

/// Mid in dollars, or NaN unless both sides are present and not crossed.
__device__ __forceinline__ double quote_mid(const QuoteCopy& q) {
    if (q.bid_price == 0 || q.ask_price == 0 || q.bid_size == 0 || q.ask_size == 0 ||
        q.ask_price < q.bid_price)
        return quiet_nan();
    return 0.5e-4 * (static_cast<double>(q.bid_price) + static_cast<double>(q.ask_price));
}

/// Replace the IV word unless it already holds an IV for a newer quote
/// version. Equal versions replace: that is how the surface pass re-solves an
/// unchanged quote against a fresh forward.
__device__ inline void publish_iv(unsigned long long* word, uint32_t version, float iv) {
    const unsigned long long w =
        (static_cast<unsigned long long>(version) << 32) | __float_as_uint(iv);
    unsigned long long cur = ld(word);
    while (static_cast<uint32_t>(cur >> 32) <= version) {
        const unsigned long long prev = atomicCAS(word, cur, w);
        if (prev == cur) return;
        cur = prev;
    }
}

/// The expiry's current forward for option `s`: NaN until the surface pass
/// has found one.
__device__ __forceinline__ double option_forward(const FieldDev& f, const OptionStatic& s) {
    return expiry_forward(f.exps[s.expiry]);
}

/// True when the option is out of the money against forward F: the side the
/// surface uses (calls at K >= F, puts below).
__device__ __forceinline__ bool is_otm(const OptionStatic& s, double F) {
    const double K = s.strike * 1e-4;
    return s.call ? K >= F : K < F;
}

/// IV of `mid` for option `s` against forward F (both valid).
__device__ inline float solve_iv(const FieldDev& f, const OptionStatic& s, double F, double mid,
                                 uint32_t feed_sec) {
    const double T = time_to_expiry(ld(&f.exps[s.expiry].days), feed_sec);
    const double df = exp(-f.p.rate * T);
    return static_cast<float>(implied_vol(mid, F, s.strike * 1e-4, T, df, s.call != 0));
}

// ---------------------------------------------------------------------------
// Message handlers
// ---------------------------------------------------------------------------

__device__ inline void apply_directory(const FieldDev& f, const bono::Directory& d,
                                unsigned long long seq, FeedCounters& c) {
    const int und = underlying_index(f.p, d.underlying);
    if (und < 0) {
        ++c.admin;
        return;
    }
    if ((d.option_type != 'C' && d.option_type != 'P') || d.exp_month < 1 || d.exp_month > 12 ||
        d.exp_day < 1 || d.exp_day > 31) {
        ++c.malformed;
        return;
    }
    ++c.directory;

    bool fresh = false;
    uint32_t slot = 0;

    // The expiry...
    const uint32_t yymmdd = d.exp_year * 10000u + d.exp_month * 100u + d.exp_day;
    const unsigned int ekey = (static_cast<unsigned int>(und) << 24) | yymmdd;
    const int32_t e = claim(f.exp_keys, f.exp_vals, kExpirySlots, ekey, kEmptyKey32, mix32(ekey),
                            f.n_exps, kMaxExpiries, fresh, slot);
    if (e < 0) {
        atomicAdd(f.overflow, 1u);
        return;
    }
    if (fresh) {
        ExpiryRec& x = f.exps[e];
        x.yymmdd = yymmdd;
        x.und = static_cast<uint32_t>(und);
        x.days = days_from_civil(2000 + d.exp_year, d.exp_month, d.exp_day) - f.p.trade_days;
        publish(&x.ready, &f.exp_vals[slot], e);
    }

    // ... the strike within it ...
    const unsigned long long pkey = (static_cast<unsigned long long>(e) << 32) | d.strike;
    const int32_t pr = claim(f.pair_keys, f.pair_vals, kPairSlots, pkey, kEmptyKey64, mix64(pkey),
                             f.n_pairs, kMaxPairs, fresh, slot);
    if (pr < 0) {
        atomicAdd(f.overflow, 1u);
        return;
    }
    if (fresh) {
        PairRec& r = f.pairs[pr];
        r.expiry = static_cast<uint32_t>(e);
        r.strike = d.strike;
        publish(&r.ready, &f.pair_vals[slot], pr);
        const unsigned int k = atomicAdd(&f.exps[e].n_pairs, 1u);
        if (k < kMaxPairsPerExpiry) st(&f.exp_pairs[e * kMaxPairsPerExpiry + k], pr);
        else atomicAdd(f.overflow, 1u);
    }

    // ... and the option itself.
    const int32_t o = claim(f.opt_keys, f.opt_vals, kOptionSlots, d.option_id, kEmptyKey32,
                            mix32(d.option_id), f.n_opts, kMaxOptions, fresh, slot);
    if (o < 0) {
        atomicAdd(f.overflow, 1u);
        return;
    }
    OptionRec& r = f.opts[o];
    if (fresh) {
        const bool call = d.option_type == 'C';
        r.option_id = d.option_id;
        r.strike = d.strike;
        r.expiry = e;
        r.pair = pr;
        r.und = static_cast<uint32_t>(und);
        r.call = call ? 1u : 0u;
        publish(&r.ready, &f.opt_vals[slot], o);
        st(call ? &f.pairs[pr].call : &f.pairs[pr].put, o);
    }
    // Canonical fields never change intra-day; tradability can (Tradable=N
    // removes an option). Newest message wins.
    atomicMax(&r.tradable, (seq << 1) | (d.tradable == 'Y' ? 1ull : 0ull));
}

__device__ inline void apply_quote(const FieldDev& f, const bono::Quote& q, unsigned long long seq,
                            uint32_t feed_sec, FeedCounters& c) {
    const int32_t i = option_find(f, q.option_id);
    if (i < 0) {
        ++c.unmapped;
        return;
    }
    ++c.quotes;
    OptionRec& o = f.opts[i];

    QuoteCopy now;
    bool moved = false;
    if (!apply_quote_locked(o.q, q, seq, now, moved)) {
        ++c.stale;
        return;
    }
    // A size-only change leaves the mid, and so the IV, where it was.
    if (!moved || !option_live(o)) return;

    // On tick, only the out-of-the-money side is solved: it is what the
    // surface shows, and a deep in-the-money mid is often within rounding of
    // intrinsic, where no volatility fits. The surface pass re-solves every
    // option, ITM included, against each fresh forward. No forward yet: the
    // surface pass solves this quote once there is one.
    const OptionStatic s = option_static(o);
    const double F = option_forward(f, s);
    if (!(F > 0.0) || !is_otm(s, F)) return;

    // Solved outside the lock, from the copy taken under it: the lock is held
    // for a few stores, not for a Newton solve. A quote that went one-sided
    // publishes NaN, so a stale IV does not linger.
    const double mid = quote_mid(now);
    float iv = static_cast<float>(quiet_nan());
    if (mid == mid) {
        iv = solve_iv(f, s, F, mid, feed_sec);
        if (iv == iv) ++c.iv_solved;
        else ++c.iv_failed;
    }
    publish_iv(&o.iv, now.version, iv);
}

// ---------------------------------------------------------------------------
// Frames and batches
// ---------------------------------------------------------------------------

/// One frame of a batch: a whole Ethernet frame and its length.
struct FrameRef {
    const uint8_t* data;
    uint32_t       len;
};

/// Locate the MoldUDP64 packet in a frame. False for anything else.
__device__ __forceinline__ bool frame_packet(const FrameRef& fr, UdpView& u, mold::Header& h) {
    return parse_udp(fr.data, fr.len, u) && mold::parse_header(u.payload, u.len, h);
}

/// Pass 1 over one frame: counters, the feed clock (T) and the directory (D).
/// Everything that refers to an option by ID waits for pass 2.
__device__ inline void admin_pass(const FieldDev& f, const FrameRef& fr, FeedCounters& c,
                                  uint32_t& feed_sec) {
    ++c.frames;
    c.bytes += fr.len;
    UdpView u;
    mold::Header h;
    if (!frame_packet(fr, u, h)) {
        ++c.not_feed;
        return;
    }
    if (h.count == mold::kHeartbeat) {
        ++c.heartbeats;
        return;
    }
    if (h.count == mold::kEndOfSession) {
        ++c.end_of_session;
        return;
    }

    mold::Cursor cur(u.payload, u.len, h);
    const uint8_t* m;
    uint32_t len;
    uint64_t seq;
    while (cur.next(m, len, seq)) {
        ++c.messages;
        if (seq > c.max_seq) c.max_seq = seq;
        if (!bono::length_ok(m, len)) {
            ++c.malformed;
            continue;
        }
        switch (m[0]) {
            case bono::kTimestamp: {
                bono::Timestamp t;
                bono::decode(m, len, t);
                if (t.seconds > feed_sec) feed_sec = t.seconds;
                ++c.admin;
                break;
            }
            case bono::kDirectory: {
                bono::Directory d;
                bono::decode(m, len, d);
                apply_directory(f, d, seq, c);
                break;
            }
            case bono::kTradingAction:
            case bono::kSecurityOpen:
            case bono::kSystemEvent:
                ++c.admin;  // H and O are applied in pass 2
                break;
            case bono::kTrade:
            case bono::kBrokenTrade:
                ++c.trades;
                break;
            default:  // quotes: pass 2
                break;
        }
    }
    if (cur.truncated()) ++c.malformed;
}

/// Pass 2 over one frame: everything addressed to an option by ID - quotes,
/// Trading Action (H) and Security Open (O) - so it finds a directory entry
/// that arrived anywhere in the same batch.
__device__ inline void option_pass(const FieldDev& f, const FrameRef& fr, FeedCounters& c,
                                   uint32_t feed_sec) {
    UdpView u;
    mold::Header h;
    if (!frame_packet(fr, u, h) || h.count == mold::kHeartbeat || h.count == mold::kEndOfSession)
        return;
    mold::Cursor cur(u.payload, u.len, h);
    const uint8_t* m;
    uint32_t len;
    uint64_t seq;
    while (cur.next(m, len, seq)) {
        if (bono::is_quote(m[0])) {
            bono::Quote q;
            if (bono::decode(m, len, q)) apply_quote(f, q, seq, feed_sec, c);
        } else if (m[0] == bono::kTradingAction) {
            bono::TradingAction a;
            if (!bono::decode(m, len, a)) continue;
            const int32_t i = option_find(f, a.option_id);
            if (i >= 0) atomicMax(&f.opts[i].halted, (seq << 1) | (a.state == 'H' ? 1ull : 0ull));
        } else if (m[0] == bono::kSecurityOpen) {
            bono::SecurityOpen a;
            if (!bono::decode(m, len, a)) continue;
            const int32_t i = option_find(f, a.option_id);
            if (i >= 0) atomicMax(&f.opts[i].open, (seq << 1) | (a.state == 'Y' ? 1ull : 0ull));
        }
    }
}

/// Process one batch of `n` frames with the whole block. `frame_at(i)` returns
/// frame i as a FrameRef. Every thread of the block must call this: it holds
/// barriers. Thread t takes frames t, t + blockDim.x, ...
///
///   pass 1: T S D          (the feed clock, directory inserts)
///   __syncthreads()
///   pass 2: quotes, H, O   (so anything addressed to an option always finds
///                           a directory entry that arrived anywhere in the
///                           same batch)
///
/// then the block's counters are reduced in shared memory and added to
/// f.counters[blockIdx.x] by thread 0.
template <class FrameAt>
__device__ inline void process_batch(const FieldDev& f, uint32_t n, const FrameAt& frame_at) {
    __shared__ unsigned long long s_sum[kNumCounters];
    if (threadIdx.x < kNumCounters) s_sum[threadIdx.x] = 0;

    FeedCounters c{};
    uint32_t feed_sec = 0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) admin_pass(f, frame_at(i), c, feed_sec);
    if (feed_sec) atomicMax(f.feed_sec, feed_sec);
    __syncthreads();

    const uint32_t now_sec = ld(f.feed_sec);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) option_pass(f, frame_at(i), c, now_sec);

    const uint64_t* mine = reinterpret_cast<const uint64_t*>(&c);
    for (uint32_t k = 0; k < kNumCounters; ++k) {
        if (!mine[k]) continue;
        if (k == kMaxSeqCounter) atomicMax(&s_sum[k], static_cast<unsigned long long>(mine[k]));
        else atomicAdd(&s_sum[k], static_cast<unsigned long long>(mine[k]));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t* g = reinterpret_cast<uint64_t*>(&f.counters[blockIdx.x]);
        for (uint32_t k = 0; k < kNumCounters; ++k) {
            const uint64_t prev = ld(&g[k]);
            const uint64_t add = s_sum[k];
            st(&g[k], k == kMaxSeqCounter ? (add > prev ? add : prev) : prev + add);
        }
    }
    __syncthreads();  // s_sum is zeroed again by the next batch
}

}  // namespace dev
}  // namespace vf

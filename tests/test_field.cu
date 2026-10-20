//
// tests/test_field.cu - the field's concurrency rules, on the GPU.
//
// Every test builds real frames (Eth/IPv4/UDP + MoldUDP64 + BONO) on the host
// and runs them through vf::dev::process_batch(), the same function the sim
// and GPUNetIO kernels call. Within a batch the frame order is shuffled, as a
// NIC batch would be; the results must not depend on it.
//
//   hammer       thousands of threads update one option with shuffled
//                sequence numbers; each side ends at its max-seq update and no
//                lock increment is lost
//   merge        b/a one-sided updates merge with q, in any order
//   same batch   a quote and its option's directory entry in one batch map
//   duplicates   A+B feed duplicates are stale and leave the same state
//   reference    random streams vs a sequential CPU reference
//   iv rule      the IV word is never replaced by an older quote version
//   surface      forward, IVs and the snapshot from a priced chain
//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "vf/black.hpp"
#include "vf/field.hpp"
#include "vf/field_device.cuh"
#include "vf/snapshot.hpp"

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

using Bytes = std::vector<uint8_t>;
using Frames = std::vector<Bytes>;

const int32_t kTradeDays = vf::days_from_civil(2026, 10, 20);

// ---------------------------------------------------------------------------
// Message and frame builders
// ---------------------------------------------------------------------------

Bytes head(char type, uint32_t id) {
    Bytes m(9, 0);
    m[0] = static_cast<uint8_t>(type);
    vf::store_be32(&m[5], id);
    return m;
}

void put16(Bytes& m, uint32_t v) {
    m.push_back(static_cast<uint8_t>(v >> 8));
    m.push_back(static_cast<uint8_t>(v));
}

void put32(Bytes& m, uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) m.push_back(static_cast<uint8_t>(v >> s));
}

Bytes msg_T(uint32_t sec) {
    Bytes m{'T'};
    put32(m, sec);
    return m;
}

Bytes msg_D(uint32_t id, const char* und, int yy, int mm, int dd, uint32_t strike, char cp,
            char tradable = 'Y') {
    Bytes m = head('D', id);
    for (int i = 0; i < 6; ++i) m.push_back(i < 4 ? static_cast<uint8_t>("TSTX"[i]) : ' ');
    m.push_back(static_cast<uint8_t>(yy));
    m.push_back(static_cast<uint8_t>(mm));
    m.push_back(static_cast<uint8_t>(dd));
    put32(m, strike);
    m.push_back(static_cast<uint8_t>(cp));
    m.push_back(1);
    char u[13];
    std::memset(u, ' ', sizeof u);
    std::memcpy(u, und, std::strlen(und));
    m.insert(m.end(), u, u + 13);
    m.push_back('N');
    m.push_back(static_cast<uint8_t>(tradable));
    m.push_back('P');
    return m;
}

Bytes msg_H(uint32_t id, char state) {
    Bytes m = head('H', id);
    m.push_back(static_cast<uint8_t>(state));
    return m;
}

/// q (short form, cents) or Q (long form, 1e-4).
Bytes msg_quote(bool long_form, uint32_t id, uint32_t bid, uint32_t bsz, uint32_t ask,
                uint32_t asz) {
    Bytes m = head(long_form ? 'Q' : 'q', id);
    m.push_back(' ');
    auto put = long_form ? put32 : put16;
    put(m, bid);
    put(m, bsz);
    put(m, ask);
    put(m, asz);
    return m;
}

/// b / a (short, cents) or B / A (long, 1e-4).
Bytes msg_side(char type, uint32_t id, uint32_t price, uint32_t size) {
    Bytes m = head(type, id);
    m.push_back(' ');
    auto put = (type == 'B' || type == 'A') ? put32 : put16;
    put(m, price);
    put(m, size);
    return m;
}

Bytes frame(uint64_t seq, const std::vector<Bytes>& msgs) {
    Bytes payload(vf::mold::kHeaderBytes, 0);
    std::memcpy(payload.data(), "TESTSESS01", 10);
    vf::store_be64(&payload[10], seq);
    vf::store_be16(&payload[18], static_cast<uint16_t>(msgs.size()));
    for (const Bytes& m : msgs) {
        put16(payload, static_cast<uint32_t>(m.size()));
        payload.insert(payload.end(), m.begin(), m.end());
    }
    Bytes f(vf::kSyntheticHeaderBytes + payload.size());
    vf::write_synthetic_header(f.data(), 0x0A000002, 40000, 0xE9360C6F, 18001,
                               static_cast<uint32_t>(payload.size()));
    std::memcpy(f.data() + vf::kSyntheticHeaderBytes, payload.data(), payload.size());
    return f;
}

// ---------------------------------------------------------------------------
// Running batches
// ---------------------------------------------------------------------------

struct DeviceFrames {
    const uint8_t*  buf;
    const uint32_t* off;
    const uint32_t* len;
    uint32_t        first;  ///< this block's first frame
    __device__ vf::dev::FrameRef operator()(uint32_t i) const {
        return vf::dev::FrameRef{buf + off[first + i], len[first + i]};
    }
};

/// Block b processes frames [b * per_block, ...) as one batch.
__global__ void k_batches(vf::FieldDev f, DeviceFrames frames, uint32_t per_block,
                          uint32_t total) {
    DeviceFrames mine = frames;
    mine.first = blockIdx.x * per_block;
    const uint32_t n = mine.first < total ? min(per_block, total - mine.first) : 0u;
    vf::dev::process_batch(f, n, mine);
}

/// Process `frames` as consecutive batches of at most `per_block` frames each,
/// all batches concurrently, one block per batch. Frames are used as given:
/// shuffle them first to model an unordered NIC batch.
void run_concurrent(vf::Field* field, const Frames& frames, uint32_t per_block) {
    std::vector<uint32_t> off, len;
    Bytes buf;
    for (const Bytes& f : frames) {
        off.push_back(static_cast<uint32_t>(buf.size()));
        len.push_back(static_cast<uint32_t>(f.size()));
        buf.insert(buf.end(), f.begin(), f.end());
    }
    uint8_t* d_buf = nullptr;
    uint32_t *d_off = nullptr, *d_len = nullptr;
    VF_CUDA_CHECK(cudaMalloc(&d_buf, buf.size() + 1));
    VF_CUDA_CHECK(cudaMalloc(&d_off, off.size() * sizeof(uint32_t) + 4));
    VF_CUDA_CHECK(cudaMalloc(&d_len, len.size() * sizeof(uint32_t) + 4));
    VF_CUDA_CHECK(cudaMemcpy(d_buf, buf.data(), buf.size(), cudaMemcpyHostToDevice));
    VF_CUDA_CHECK(cudaMemcpy(d_off, off.data(), off.size() * 4, cudaMemcpyHostToDevice));
    VF_CUDA_CHECK(cudaMemcpy(d_len, len.data(), len.size() * 4, cudaMemcpyHostToDevice));

    const uint32_t total = static_cast<uint32_t>(frames.size());
    const uint32_t blocks = (total + per_block - 1) / per_block;
    k_batches<<<blocks ? blocks : 1, vf::dev::kBlockThreads>>>(
        vf::field_dev(field), DeviceFrames{d_buf, d_off, d_len, 0}, per_block, total);
    VF_CUDA_CHECK(cudaGetLastError());
    VF_CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_buf);
    cudaFree(d_off);
    cudaFree(d_len);
}

/// Process `frames` in order as a sequence of batches (sizes from `sizes`),
/// each batch shuffled - the way a stream arrives batch after batch.
void run_stream(vf::Field* field, const Frames& frames, std::mt19937_64& rng,
                uint32_t max_batch = vf::dev::kMaxBatch) {
    size_t i = 0;
    std::uniform_int_distribution<uint32_t> size(1, max_batch);
    while (i < frames.size()) {
        const size_t n = std::min<size_t>(size(rng), frames.size() - i);
        Frames batch(frames.begin() + static_cast<long>(i),
                     frames.begin() + static_cast<long>(i + n));
        std::shuffle(batch.begin(), batch.end(), rng);
        run_concurrent(field, batch, vf::dev::kMaxBatch);
        i += n;
    }
}

vf::Field* make_field(uint32_t blocks) {
    const char* und[] = {"TEST"};
    vf::FieldParams p{};
    vf::field_params(und, 1, kTradeDays, 0.04, p);
    return vf::field_create(p, blocks);
}

/// Host copy of every option, keyed by option ID.
std::map<uint32_t, vf::OptionRec> options(const vf::Field* field) {
    const vf::FieldSizes sz = vf::field_sizes(field);
    std::vector<vf::OptionRec> v(sz.options);
    if (sz.options)
        VF_CUDA_CHECK(cudaMemcpy(v.data(), vf::field_dev(field).opts,
                                 sz.options * sizeof(vf::OptionRec), cudaMemcpyDeviceToHost));
    std::map<uint32_t, vf::OptionRec> out;
    for (const vf::OptionRec& o : v)
        if (o.ready) out[o.option_id] = o;
    return out;
}

vf::FeedCounters counters(const vf::Field* field, uint32_t blocks) {
    std::vector<vf::FeedCounters> per(blocks);
    vf::field_counters(field, per.data());
    return vf::counters_sum(per.data(), blocks);
}

/// The directory entry for option `id`, as its own frame.
Bytes dir_frame(uint64_t seq, uint32_t id, uint32_t strike = 1000000, char cp = 'C') {
    return frame(seq, {msg_D(id, "TEST", 26, 11, 20, strike, cp)});
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_hammer() {
    constexpr uint32_t kBlocks = 8;
    constexpr uint32_t kUpdates = kBlocks * vf::dev::kMaxBatch;  // 8192 frames, 4096 threads
    constexpr uint32_t kId = 4242;
    std::mt19937_64 rng(1);

    vf::Field* field = make_field(kBlocks);
    run_concurrent(field, {dir_frame(1, kId)}, vf::dev::kMaxBatch);

    // Sequence numbers 10..10+N-1, shuffled. Each update is a random kind;
    // prices encode the sequence number so the winner is identifiable.
    std::vector<uint64_t> seqs(kUpdates);
    std::iota(seqs.begin(), seqs.end(), 10);
    std::shuffle(seqs.begin(), seqs.end(), rng);
    uint64_t best_bid = 0, best_ask = 0;
    Frames frames;
    for (uint64_t s : seqs) {
        const uint32_t v = static_cast<uint32_t>(s);
        const int kind = static_cast<int>(rng() % 3);
        Bytes m;
        if (kind == 0) m = msg_quote(true, kId, v * 10, v, v * 10 + 5, v + 1);
        else if (kind == 1) m = msg_side('B', kId, v * 10, v);
        else m = msg_side('A', kId, v * 10 + 5, v + 1);
        if (kind != 2) best_bid = std::max(best_bid, s);
        if (kind != 1) best_ask = std::max(best_ask, s);
        frames.push_back(frame(s, {m}));
    }
    run_concurrent(field, frames, vf::dev::kMaxBatch);

    const auto opts = options(field);
    CHECK(opts.count(kId) == 1);
    const vf::OptionRec& o = opts.at(kId);
    CHECK(o.q.bid_seq == best_bid && o.q.bid_price == best_bid * 10 && o.q.bid_size == best_bid);
    CHECK(o.q.ask_seq == best_ask && o.q.ask_price == best_ask * 10 + 5 &&
          o.q.ask_size == best_ask + 1);
    CHECK(o.q.lock == 2 * kUpdates);  // every writer took and released the lock once

    const vf::FeedCounters c = counters(field, kBlocks);
    CHECK(c.quotes == kUpdates && c.unmapped == 0);
    CHECK(c.stale < kUpdates);
    std::printf("  hammer: %u updates from %u threads, %llu stale, lock %u\n", kUpdates,
                kBlocks * vf::dev::kBlockThreads, static_cast<unsigned long long>(c.stale),
                o.q.lock);
    vf::field_destroy(field);
}

void test_merge() {
    std::mt19937_64 rng(2);
    vf::Field* field = make_field(1);
    constexpr int kTrials = 24;
    Frames dirs;
    for (int t = 0; t < kTrials; ++t) dirs.push_back(dir_frame(1 + t, 100 + t));
    run_concurrent(field, dirs, vf::dev::kMaxBatch);

    for (int t = 0; t < kTrials; ++t) {
        const uint32_t id = 100 + t;
        Frames fr = {
            frame(1009, {msg_side('b', id, 50, 1)}),             // older than the q: loses
            frame(1010, {msg_quote(false, id, 100, 10, 120, 10)}),
            frame(1011, {msg_side('b', id, 105, 5)}),
            frame(1012, {msg_side('a', id, 115, 7)}),
        };
        std::shuffle(fr.begin(), fr.end(), rng);
        run_concurrent(field, fr, vf::dev::kMaxBatch);
    }
    const auto opts = options(field);
    bool all = true;
    for (int t = 0; t < kTrials; ++t) {
        const vf::OptionRec& o = opts.at(100 + t);
        all = all && o.q.bid_price == 10500 && o.q.bid_size == 5 && o.q.ask_price == 11500 &&
              o.q.ask_size == 7 && o.q.bid_seq == 1011 && o.q.ask_seq == 1012;
    }
    CHECK(all);
    vf::field_destroy(field);
}

void test_same_batch() {
    std::mt19937_64 rng(3);
    vf::Field* field = make_field(1);
    // Quote first in the batch, its directory entry 600 frames later (another
    // thread); plus one packet carrying the quote *before* the directory.
    Frames fr;
    fr.push_back(frame(101, {msg_quote(false, 7, 200, 3, 210, 4)}));
    for (uint32_t i = 0; i < 599; ++i) fr.push_back(frame(1000 + i, {msg_T(34200)}));
    fr.push_back(dir_frame(100, 7));
    fr.push_back(frame(200, {msg_quote(false, 8, 300, 1, 310, 2),
                             msg_D(8, "TEST", 26, 11, 20, 1000000, 'P')}));
    run_concurrent(field, fr, vf::dev::kMaxBatch);

    const vf::FeedCounters c = counters(field, 1);
    CHECK(c.quotes == 2 && c.unmapped == 0 && c.directory == 2);
    const auto opts = options(field);
    CHECK(opts.count(7) && opts.at(7).q.bid_price == 20000 && opts.at(7).q.ask_size == 4);
    CHECK(opts.count(8) && opts.at(8).q.bid_price == 30000 && opts.at(8).call == 0);
    vf::field_destroy(field);
}

/// A random feed: a directory spin for `n_opts` options, then quotes, halts
/// and resumes. Returns packets (frames) in sequence order.
struct Stream {
    Frames frames;
    std::vector<std::pair<uint64_t, Bytes>> msgs;  ///< (seq, message), seq order
};

Stream random_stream(std::mt19937_64& rng, uint32_t n_opts, uint32_t n_msgs) {
    Stream s;
    uint64_t seq = 1;
    std::vector<Bytes> pending;
    uint64_t first = seq;
    auto flush = [&] {
        if (pending.empty()) return;
        s.frames.push_back(frame(first, pending));
        pending.clear();
    };
    auto add = [&](Bytes m) {
        if (pending.empty()) first = seq;
        s.msgs.emplace_back(seq++, m);
        pending.push_back(std::move(m));
        if (pending.size() >= 1 + rng() % 8) flush();
    };
    add(msg_T(25200));
    for (uint32_t k = 0; k < n_opts; ++k)
        add(msg_D(5000 + k, "TEST", 26, 11 + k % 2, 20, 900000 + (k / 2) * 10000,
                  k % 2 ? 'P' : 'C'));
    for (uint32_t k = 0; k < n_opts; ++k) add(msg_H(5000 + k, 'T'));
    for (uint32_t i = 0; i < n_msgs; ++i) {
        const uint32_t id = 5000 + static_cast<uint32_t>(rng() % n_opts);
        const uint32_t px = 100 + static_cast<uint32_t>(rng() % 5000);
        const uint32_t sz = 1 + static_cast<uint32_t>(rng() % 500);
        switch (rng() % 7) {
            case 0: add(msg_quote(false, id, px, sz, px + 5, sz + 1)); break;
            case 1: add(msg_quote(true, id, px * 100, sz, px * 100 + 500, sz + 1)); break;
            case 2: add(msg_side('b', id, px, sz)); break;
            case 3: add(msg_side('a', id, px, sz)); break;
            case 4: add(msg_side('B', id, px * 100, sz)); break;
            case 5: add(msg_side('A', id, px * 100, sz)); break;
            default: add(msg_H(id, rng() % 2 ? 'H' : 'T')); break;
        }
    }
    flush();
    return s;
}

struct RefOption {
    uint32_t bid = 0, bsz = 0, ask = 0, asz = 0;
    bool halted = true;
};

/// Sequential reference: apply every message in sequence order.
std::map<uint32_t, RefOption> reference(const Stream& s) {
    std::map<uint32_t, RefOption> ref;
    for (const auto& [seq, m] : s.msgs) {
        (void)seq;
        const uint32_t id = vf::load_be32(&m[5]);
        if (m[0] == 'D') ref[id];
        else if (m[0] == 'H') ref[id].halted = m[9] == 'H';
        else if (vf::bono::is_quote(m[0])) {
            vf::bono::Quote q{};
            vf::bono::decode(m.data(), static_cast<uint32_t>(m.size()), q);
            RefOption& r = ref[id];
            if (q.sides & vf::bono::kSideBid) {
                r.bid = q.bid_price;
                r.bsz = q.bid_size;
            }
            if (q.sides & vf::bono::kSideAsk) {
                r.ask = q.ask_price;
                r.asz = q.ask_size;
            }
        }
    }
    return ref;
}

bool matches(const std::map<uint32_t, vf::OptionRec>& gpu,
             const std::map<uint32_t, RefOption>& ref) {
    if (gpu.size() != ref.size()) return false;
    for (const auto& [id, r] : ref) {
        auto it = gpu.find(id);
        if (it == gpu.end()) return false;
        const vf::QuoteState& q = it->second.q;
        if (q.bid_price != r.bid || q.bid_size != r.bsz || q.ask_price != r.ask ||
            q.ask_size != r.asz || ((it->second.halted & 1ull) != 0) != r.halted)
            return false;
    }
    return true;
}

void test_duplicates_and_reference() {
    for (int round = 0; round < 6; ++round) {
        std::mt19937_64 rng(100 + round);
        const Stream s = random_stream(rng, 40, 6000);
        const auto ref = reference(s);

        // Single feed, batch after batch.
        vf::Field* a = make_field(1);
        run_stream(a, s.frames, rng);
        const auto ga = options(a);
        const vf::FeedCounters ca = counters(a, 1);
        CHECK(matches(ga, ref));
        CHECK(ca.unmapped == 0);

        // A and B feeds: every packet twice, the twins wherever they land.
        Frames dup;
        for (const Bytes& f : s.frames) {
            dup.push_back(f);
            dup.push_back(f);
        }
        vf::Field* b = make_field(1);
        run_stream(b, dup, rng);
        const auto gb = options(b);
        const vf::FeedCounters cb = counters(b, 1);
        CHECK(matches(gb, ref));
        CHECK(cb.quotes == 2 * ca.quotes);
        CHECK(cb.stale >= ca.quotes);  // at least one twin of every quote is stale
        if (round == 0)
            std::printf("  reference: %zu messages; single feed %llu stale, A+B %llu stale of "
                        "%llu\n",
                        s.msgs.size(), static_cast<unsigned long long>(ca.stale),
                        static_cast<unsigned long long>(cb.stale),
                        static_cast<unsigned long long>(cb.quotes));
        vf::field_destroy(a);
        vf::field_destroy(b);
    }
}

__global__ void k_publish(unsigned long long* word, uint32_t version, float iv) {
    vf::dev::publish_iv(word, version, iv);
}

void test_iv_rule() {
    unsigned long long* w = nullptr;
    VF_CUDA_CHECK(cudaMalloc(&w, sizeof *w));
    const unsigned long long init = (5ull << 32) | 0x3E4CCCCDu;  // version 5, 0.2f
    VF_CUDA_CHECK(cudaMemcpy(w, &init, sizeof init, cudaMemcpyHostToDevice));
    auto read = [&] {
        unsigned long long v = 0;
        VF_CUDA_CHECK(cudaMemcpy(&v, w, sizeof v, cudaMemcpyDeviceToHost));
        float f;
        const uint32_t bits = static_cast<uint32_t>(v);
        std::memcpy(&f, &bits, 4);
        return std::make_pair(static_cast<uint32_t>(v >> 32), f);
    };
    k_publish<<<1, 1>>>(w, 4, 0.9f);  // older: ignored
    CHECK(read().first == 5 && read().second == 0.2f);
    k_publish<<<1, 1>>>(w, 5, 0.3f);  // same version: replaces (fresh forward)
    CHECK(read().first == 5 && read().second == 0.3f);
    k_publish<<<1, 1>>>(w, 9, 0.4f);
    CHECK(read().first == 9 && read().second == 0.4f);
    // Racing publishers: the highest version wins.
    k_publish<<<64, 256>>>(w, 100, 0.5f);
    k_publish<<<1, 1>>>(w, 99, 0.6f);
    CHECK(read().first == 100 && read().second == 0.5f);
    cudaFree(w);
}

void test_surface() {
    const double S = 100.0, r = 0.04;
    const uint32_t feed_sec = 34200;  // 09:30
    const int32_t days = vf::days_from_civil(2026, 11, 20) - kTradeDays;
    const double T = vf::time_to_expiry(days, feed_sec);
    const double F = S * std::exp(r * T), df = std::exp(-r * T);
    auto smile = [&](double K) {
        const double x = std::log(K / F);
        return 0.25 - 0.3 * x + 0.5 * x * x;
    };

    vf::Field* field = make_field(1);
    std::vector<Bytes> msgs = {msg_T(feed_sec)};
    uint32_t id = 1;
    std::vector<double> strikes;
    for (double K = 80.0; K <= 120.0 + 1e-9; K += 2.5) {
        strikes.push_back(K);
        for (char cp : {'C', 'P'}) {
            msgs.push_back(msg_D(id, "TEST", 26, 11, 20, static_cast<uint32_t>(K * 1e4), cp));
            msgs.push_back(msg_H(id, 'T'));
            const double theo = vf::black76_price(F, K, T, smile(K), df, cp == 'C');
            const uint32_t c = static_cast<uint32_t>(std::lround(theo * 100));
            msgs.push_back(msg_quote(false, id, c > 1 ? c - 1 : 0, 10, c + 1, 10));
            ++id;
        }
    }
    // One halted option: its IV must not show.
    msgs.push_back(msg_H(1, 'H'));
    Frames fr;
    for (size_t i = 0; i < msgs.size(); i += 6)
        fr.push_back(frame(1 + i, std::vector<Bytes>(msgs.begin() + i,
                                                     msgs.begin() + std::min(i + 6, msgs.size()))));
    run_concurrent(field, fr, vf::dev::kMaxBatch);

    vf::FeedCounters c = counters(field, 1);
    CHECK(c.iv_solved == 0);  // no forward yet: nothing solved on tick

    void* host = nullptr;
    VF_CUDA_CHECK(cudaHostAlloc(&host, vf::kSnapBytes, cudaHostAllocMapped));
    std::memset(host, 0, sizeof(vf::SnapHeader));
    void* dev = nullptr;
    VF_CUDA_CHECK(cudaHostGetDevicePointer(&dev, host, 0));
    auto* snap_dev = static_cast<vf::SnapHeader*>(dev);
    vf::field_launch_surface(field, nullptr, snap_dev, 123, vf::kSnapRunning);
    VF_CUDA_CHECK(cudaDeviceSynchronize());

    const auto* h = static_cast<const vf::SnapHeader*>(host);
    const auto* base = static_cast<const uint8_t*>(host);
    const auto* exps = reinterpret_cast<const vf::ExpirySnap*>(base + vf::kSnapExpiryOffset);
    const auto* pairs = reinterpret_cast<const vf::PairSnap*>(base + vf::kSnapPairOffset);
    CHECK(h->seqlock == 2 && h->snapshot_id == 1 && h->host_ns == 123);
    CHECK(h->n_expiries == 1 && h->n_pairs == strikes.size() && h->n_options == 2 * strikes.size());
    CHECK(h->feed_sec == feed_sec && h->counters.quotes == 2 * strikes.size());
    CHECK(exps[0].yymmdd == 261120 && exps[0].days == days);
    CHECK(exps[0].n_forward == vf::kForwardPairs);
    CHECK(std::fabs(exps[0].forward - F) < 0.01);

    double worst = 0.0;
    int halted_nan = 0, checked = 0;
    for (uint32_t p = 0; p < h->n_pairs; ++p) {
        const vf::PairSnap& s = pairs[p];
        if (!(s.flags & vf::kPairCallLive)) {  // option 1 is the 80 call
            halted_nan += std::isnan(s.iv_call) ? 1 : 0;
            continue;
        }
        const double vega = vf::black76_vega(F, s.strike, T, smile(s.strike), df);
        if (vega < 1.0) continue;
        const double err = std::fabs(s.iv_otm - smile(s.strike));
        // Mid is within half a cent of theo; allow that plus the forward error.
        CHECK(err < (0.006 + std::fabs(exps[0].forward - F)) / vega + 1e-4);
        worst = std::max(worst, err);
        ++checked;
        CHECK(((s.flags & vf::kPairOtmCall) != 0) == (s.strike >= exps[0].forward));
    }
    CHECK(halted_nan == 1 && checked > 10);
    std::printf("  surface: F %.5f (true %.5f), %d OTM IVs, worst |dsigma| %.2e\n",
                exps[0].forward, F, checked, worst);

    // Now there is a forward: a new out-of-the-money tick is solved on the
    // receive path, and its IV carries the newer quote version. An
    // in-the-money tick is left to the surface pass.
    auto tick = [&](uint32_t id, double K, bool call, uint64_t seq) {
        const double theo = vf::black76_price(F, K, T, smile(K), df, call);
        const uint32_t cents = static_cast<uint32_t>(std::lround(theo * 100));
        run_concurrent(field, {frame(seq, {msg_quote(false, id, cents - 2, 5, cents + 2, 5)})},
                       vf::dev::kMaxBatch);
    };
    const uint32_t otm_call = 25;  // ids run 1, 2 = 80 C/P; 82.5 C/P; ... 110 C = 25
    const uint32_t itm_call = 9;   // 90 C
    tick(itm_call, 90.0, true, 100000);
    c = counters(field, 1);
    CHECK(c.iv_solved == 0 && c.iv_failed == 0);
    tick(otm_call, 110.0, true, 100001);
    c = counters(field, 1);
    CHECK(c.iv_solved == 1 && c.iv_failed == 0);
    const auto opts = options(field);
    CHECK((opts.at(otm_call).iv >> 32) == opts.at(otm_call).q.lock / 2);
    CHECK((opts.at(itm_call).iv >> 32) < opts.at(itm_call).q.lock / 2);

    cudaFreeHost(host);
    vf::field_destroy(field);
}

}  // namespace

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::fprintf(stderr, "test_field: no CUDA device\n");
        return 1;
    }
    std::printf("test_field on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    test_hammer();
    test_merge();
    test_same_batch();
    test_duplicates_and_reference();
    test_iv_rule();
    test_surface();

    std::printf("test_field: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

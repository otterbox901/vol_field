//
// src/field/field.cu - field allocation and the surface pass.
//
// The receive kernels write the field (vf/field_device.cuh). The surface pass
// below is launched by the host every --snapshot-ms on its own non-blocking
// stream, so it runs concurrently with the persistent receive kernel:
//
//   1. forward   one block per expiry: the parity forward from the quotes as
//                they stand, each read under its seqlock
//   2. resolve   every option's IV re-solved against that forward, published
//                with the version rule so it never replaces a newer tick
//   3. snapshot  begin (seqlock odd) / write records / end (header, seqlock
//                even), straight into the mapped shared memory
//
// Kernel boundaries on one stream order the three snapshot steps, so no grid
// barrier is needed.
//

#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "vf/field.hpp"
#include "vf/field_device.cuh"
#include "vf/snapshot.hpp"

namespace vf {

/// Surface-pass state that lives on the device between launches.
struct SurfaceCtl {
    unsigned long long seqlock;      ///< mirror of the snapshot's seqlock
    unsigned long long snapshot_id;
    unsigned int n_exps;             ///< sizes frozen at the start of a snapshot
    unsigned int n_pairs;
    unsigned int n_opts;
};

struct Field {
    FieldDev    dev{};
    SurfaceCtl* ctl = nullptr;
    int         sms = 1;
};

namespace {

constexpr uint32_t kSurfaceThreads = 256;

__global__ void k_init(FieldDev f) {
    const unsigned long long nan_iv = __float_as_uint(static_cast<float>(quiet_nan()));
    const unsigned long long nan_fwd = dev::double_bits(quiet_nan());
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kMaxOptions;
         i += gridDim.x * blockDim.x) {
        f.opts[i].iv = nan_iv;   // version 0, no IV
        f.opts[i].halted = 1ull; // absent from the Trading Action spin = halted (spec 4.4)
        if (i < kMaxPairs) {
            f.pairs[i].call = -1;
            f.pairs[i].put = -1;
        }
        if (i < kMaxExpiries) f.exps[i].forward = nan_fwd;
    }
}

/// 1. One block per expiry: the parity forward from consistent mids.
__global__ void k_forward(FieldDev f) {
    __shared__ double s_strike[kMaxPairsPerExpiry];
    __shared__ double s_cmp[kMaxPairsPerExpiry];
    __shared__ unsigned int s_n;

    const uint32_t n_exps = min(dev::ld(f.n_exps), kMaxExpiries);
    const uint32_t e = blockIdx.x;
    if (e >= n_exps) return;
    ExpiryRec& x = f.exps[e];
    if (!dev::ld(&x.ready)) return;

    if (threadIdx.x == 0) s_n = 0;
    __syncthreads();

    const uint32_t n = min(dev::ld(&x.n_pairs), kMaxPairsPerExpiry);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int32_t p = dev::ld(&f.exp_pairs[e * kMaxPairsPerExpiry + i]);
        if (p < 0) continue;
        const PairRec& r = f.pairs[p];
        const int32_t c = dev::ld(&r.call);
        const int32_t q = dev::ld(&r.put);
        if (c < 0 || q < 0 || !dev::option_live(f.opts[c]) || !dev::option_live(f.opts[q]))
            continue;
        dev::QuoteCopy qc, qp;
        dev::read_quote(f.opts[c].q, qc);
        dev::read_quote(f.opts[q].q, qp);
        const double mc = dev::quote_mid(qc);
        const double mp = dev::quote_mid(qp);
        if (!(mc == mc) || !(mp == mp)) continue;
        const unsigned int k = atomicAdd(&s_n, 1u);
        s_strike[k] = dev::ld(&r.strike) * 1e-4;
        s_cmp[k] = mc - mp;
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_n > 0) {
        const double T = time_to_expiry(dev::ld(&x.days), dev::ld(f.feed_sec));
        uint32_t used = 0;
        const double F = parity_forward(s_strike, s_cmp, s_n, exp(f.p.rate * T), &used);
        if (F > 0.0) {
            dev::st(&x.forward, dev::double_bits(F));
            dev::st(&x.n_forward, used);
        }
    }
}

/// 2. Every option's IV against its expiry's fresh forward. Equal-version
/// replace: a newer tick that lands meanwhile is never overwritten.
__global__ void k_resolve(FieldDev f) {
    const uint32_t n = min(dev::ld(f.n_opts), kMaxOptions);
    const uint32_t feed_sec = dev::ld(f.feed_sec);
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        OptionRec& o = f.opts[i];
        if (!dev::ld(&o.ready) || !dev::option_live(o)) continue;
        const dev::OptionStatic s = dev::option_static(o);
        const double F = dev::option_forward(f, s);
        if (!(F > 0.0)) continue;
        dev::QuoteCopy q;
        dev::read_quote(o.q, q);
        const double mid = dev::quote_mid(q);
        const float iv = mid == mid ? dev::solve_iv(f, s, F, mid, feed_sec)
                                    : static_cast<float>(quiet_nan());
        dev::publish_iv(&o.iv, q.version, iv);
    }
}

/// 3a. Freeze the sizes and mark the snapshot as being written.
__global__ void k_snap_begin(FieldDev f, SurfaceCtl* ctl, SnapHeader* hdr) {
    ctl->n_exps = min(dev::ld(f.n_exps), kMaxExpiries);
    ctl->n_pairs = min(dev::ld(f.n_pairs), kMaxPairs);
    ctl->n_opts = min(dev::ld(f.n_opts), kMaxOptions);
    ctl->seqlock += 1;  // odd
    dev::st(&hdr->seqlock, static_cast<uint64_t>(ctl->seqlock));
    __threadfence_system();
}

__device__ void option_snap(const FieldDev& f, int32_t idx, float& iv, float& mid,
                            uint32_t& version, bool& live) {
    iv = mid = static_cast<float>(quiet_nan());
    version = 0;
    live = false;
    if (idx < 0) return;
    const OptionRec& o = f.opts[idx];
    live = dev::option_live(o);
    const unsigned long long w = dev::ld(&o.iv);
    version = static_cast<uint32_t>(w >> 32);
    dev::QuoteCopy q;
    dev::read_quote(o.q, q);
    mid = static_cast<float>(dev::quote_mid(q));
    if (live) iv = __uint_as_float(static_cast<uint32_t>(w));
}

/// 3b. Expiry and pair records, straight into host memory over PCIe.
__global__ void k_snap_write(FieldDev f, const SurfaceCtl* ctl, ExpirySnap* exps_out,
                             PairSnap* pairs_out) {
    const uint32_t n_exps = ctl->n_exps;
    const uint32_t n_pairs = ctl->n_pairs;
    const uint32_t feed_sec = dev::ld(f.feed_sec);
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t e = tid; e < n_exps; e += stride) {
        const ExpiryRec& x = f.exps[e];
        ExpirySnap s{};
        if (dev::ld(&x.ready)) {
            s.und = x.und;
            s.yymmdd = x.yymmdd;
            s.days = x.days;
            s.n_pairs = min(dev::ld(&x.n_pairs), kMaxPairsPerExpiry);
            s.T = time_to_expiry(x.days, feed_sec);
            s.forward = dev::expiry_forward(x);
            s.n_forward = dev::ld(&x.n_forward);
        } else {
            s.und = ~0u;
            s.forward = quiet_nan();
        }
        exps_out[e] = s;
    }

    for (uint32_t p = tid; p < n_pairs; p += stride) {
        const PairRec& r = f.pairs[p];
        PairSnap s{};
        s.iv_call = s.iv_put = s.iv_otm = s.call_mid = s.put_mid =
            static_cast<float>(quiet_nan());
        if (!dev::ld(&r.ready)) {
            s.und = ~0u;
            pairs_out[p] = s;
            continue;
        }
        const ExpiryRec& x = f.exps[r.expiry];
        const int32_t c = dev::ld(&r.call);
        const int32_t q = dev::ld(&r.put);
        s.und = dev::ld(&x.und);
        s.expiry = r.expiry;
        s.strike = r.strike * 1e-4;
        s.forward = dev::expiry_forward(x);
        s.T = time_to_expiry(dev::ld(&x.days), feed_sec);

        bool call_live = false, put_live = false;
        option_snap(f, c, s.iv_call, s.call_mid, s.call_version, call_live);
        option_snap(f, q, s.iv_put, s.put_mid, s.put_version, put_live);
        s.flags = (c >= 0 ? kPairHasCall : 0u) | (q >= 0 ? kPairHasPut : 0u) |
                  (call_live ? kPairCallLive : 0u) | (put_live ? kPairPutLive : 0u);
        if (s.forward > 0.0) {
            const bool otm_call = s.strike >= s.forward;
            s.iv_otm = otm_call ? s.iv_call : s.iv_put;
            if (otm_call) s.flags |= kPairOtmCall;
        }
        pairs_out[p] = s;
    }
    __threadfence_system();
}

/// 3c. Header and counters, then the seqlock even again.
__global__ void k_snap_end(FieldDev f, SurfaceCtl* ctl, SnapHeader* hdr, uint64_t host_ns,
                           uint32_t state) {
    FeedCounters sum{};
    uint64_t* s = reinterpret_cast<uint64_t*>(&sum);
    for (uint32_t b = 0; b < f.n_blocks; ++b) {
        const uint64_t* g = reinterpret_cast<const uint64_t*>(&f.counters[b]);
        for (uint32_t k = 0; k < kNumCounters; ++k) {
            const uint64_t v = dev::ld(&g[k]);
            s[k] = k == kMaxSeqCounter ? (v > s[k] ? v : s[k]) : s[k] + v;
        }
    }
    hdr->counters = sum;
    hdr->snapshot_id = ++ctl->snapshot_id;
    hdr->host_ns = host_ns;
    hdr->state = state;
    hdr->feed_sec = dev::ld(f.feed_sec);
    hdr->n_expiries = ctl->n_exps;
    hdr->n_pairs = ctl->n_pairs;
    hdr->n_options = ctl->n_opts;
    hdr->overflow = dev::ld(f.overflow);
    __threadfence_system();
    ctl->seqlock += 1;  // even
    dev::st(&hdr->seqlock, static_cast<uint64_t>(ctl->seqlock));
    __threadfence_system();
}

template <class T>
T* dev_alloc(size_t n, int fill) {
    T* p = nullptr;
    VF_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    VF_CUDA_CHECK(cudaMemset(p, fill, n * sizeof(T)));
    return p;
}

}  // namespace

bool field_params(const char* const* underlyings, uint32_t n, int32_t trade_days, double rate,
                  FieldParams& out) {
    std::memset(&out, 0, sizeof out);
    if (n == 0 || n > kMaxUnderlyings) {
        std::fprintf(stderr, "[vf] need 1 to %u underlyings, got %u\n", kMaxUnderlyings, n);
        return false;
    }
    for (uint32_t u = 0; u < n; ++u) {
        const size_t len = std::strlen(underlyings[u]);
        if (len == 0 || len > bono::kUnderlyingBytes) {
            std::fprintf(stderr, "[vf] underlying '%s' must be 1 to %u characters\n",
                         underlyings[u], bono::kUnderlyingBytes);
            return false;
        }
        std::memset(out.underlyings[u], ' ', bono::kUnderlyingBytes);
        std::memcpy(out.underlyings[u], underlyings[u], len);
    }
    out.n_underlyings = n;
    out.trade_days = trade_days;
    out.rate = rate;
    return true;
}

Field* field_create(const FieldParams& p, uint32_t n_blocks) {
    auto* f = new Field();
    FieldDev& d = f->dev;
    d.p = p;
    d.n_blocks = n_blocks;

    d.opt_keys = dev_alloc<unsigned int>(kOptionSlots, 0xFF);
    d.opt_vals = dev_alloc<int32_t>(kOptionSlots, 0xFF);  // kPending
    d.opts = dev_alloc<OptionRec>(kMaxOptions, 0);
    d.n_opts = dev_alloc<unsigned int>(1, 0);

    d.exp_keys = dev_alloc<unsigned int>(kExpirySlots, 0xFF);
    d.exp_vals = dev_alloc<int32_t>(kExpirySlots, 0xFF);
    d.exps = dev_alloc<ExpiryRec>(kMaxExpiries, 0);
    d.n_exps = dev_alloc<unsigned int>(1, 0);
    d.exp_pairs = dev_alloc<int32_t>(size_t{kMaxExpiries} * kMaxPairsPerExpiry, 0xFF);

    d.pair_keys = dev_alloc<unsigned long long>(kPairSlots, 0xFF);
    d.pair_vals = dev_alloc<int32_t>(kPairSlots, 0xFF);
    d.pairs = dev_alloc<PairRec>(kMaxPairs, 0);
    d.n_pairs = dev_alloc<unsigned int>(1, 0);

    d.feed_sec = dev_alloc<unsigned int>(1, 0);
    d.overflow = dev_alloc<unsigned int>(1, 0);
    d.counters = dev_alloc<FeedCounters>(n_blocks, 0);
    f->ctl = dev_alloc<SurfaceCtl>(1, 0);

    int device = 0;
    VF_CUDA_CHECK(cudaGetDevice(&device));
    VF_CUDA_CHECK(cudaDeviceGetAttribute(&f->sms, cudaDevAttrMultiProcessorCount, device));

    k_init<<<(kMaxOptions + 255) / 256, 256>>>(d);
    VF_CUDA_CHECK(cudaGetLastError());

    // Load the surface kernels now, before any persistent receive kernel is
    // resident. With CUDA's lazy module loading (the default since 12.2) the
    // first launch of a kernel loads it, and that waits for the device to go
    // idle - which a persistent kernel never lets happen. The surface pass and
    // the producer's copies would then stall until the receive kernel exits.
    cudaFuncAttributes attr{};
    VF_CUDA_CHECK(cudaFuncGetAttributes(&attr, k_forward));
    VF_CUDA_CHECK(cudaFuncGetAttributes(&attr, k_resolve));
    VF_CUDA_CHECK(cudaFuncGetAttributes(&attr, k_snap_begin));
    VF_CUDA_CHECK(cudaFuncGetAttributes(&attr, k_snap_write));
    VF_CUDA_CHECK(cudaFuncGetAttributes(&attr, k_snap_end));
    VF_CUDA_CHECK(cudaDeviceSynchronize());
    return f;
}

void field_destroy(Field* f) {
    if (!f) return;
    FieldDev& d = f->dev;
    void* ptrs[] = {d.opt_keys,  d.opt_vals,  d.opts,  d.n_opts,    d.exp_keys,
                    d.exp_vals,  d.exps,      d.n_exps, d.exp_pairs, d.pair_keys,
                    d.pair_vals, d.pairs,     d.n_pairs, d.feed_sec, d.overflow,
                    d.counters,  f->ctl};
    for (void* p : ptrs) cudaFree(p);
    delete f;
}

const FieldDev& field_dev(const Field* f) { return f->dev; }

void field_launch_surface(Field* f, CUstream_st* stream, SnapHeader* snap, uint64_t host_ns,
                          uint32_t state) {
    const FieldDev& d = f->dev;
    const uint32_t blocks = static_cast<uint32_t>(2 * f->sms);
    auto* base = reinterpret_cast<uint8_t*>(snap);
    auto* exps_out = reinterpret_cast<ExpirySnap*>(base + kSnapExpiryOffset);
    auto* pairs_out = reinterpret_cast<PairSnap*>(base + kSnapPairOffset);

    k_forward<<<kMaxExpiries, kSurfaceThreads, 0, stream>>>(d);
    k_resolve<<<blocks, kSurfaceThreads, 0, stream>>>(d);
    k_snap_begin<<<1, 1, 0, stream>>>(d, f->ctl, snap);
    k_snap_write<<<blocks, kSurfaceThreads, 0, stream>>>(d, f->ctl, exps_out, pairs_out);
    k_snap_end<<<1, 1, 0, stream>>>(d, f->ctl, snap, host_ns, state);
    VF_CUDA_CHECK(cudaGetLastError());
}

FieldSizes field_sizes(const Field* f) {
    const FieldDev& d = f->dev;
    unsigned int v[5] = {};
    unsigned int* src[5] = {d.n_opts, d.n_exps, d.n_pairs, d.overflow, d.feed_sec};
    for (int i = 0; i < 5; ++i)
        VF_CUDA_CHECK(cudaMemcpy(&v[i], src[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
    FieldSizes s{};
    s.options = v[0] < kMaxOptions ? v[0] : kMaxOptions;
    s.expiries = v[1] < kMaxExpiries ? v[1] : kMaxExpiries;
    s.pairs = v[2] < kMaxPairs ? v[2] : kMaxPairs;
    s.overflow = v[3];
    s.feed_sec = v[4];
    return s;
}

void field_counters(const Field* f, FeedCounters* per_block) {
    VF_CUDA_CHECK(cudaMemcpy(per_block, f->dev.counters, sizeof(FeedCounters) * f->dev.n_blocks,
                             cudaMemcpyDeviceToHost));
}

FeedCounters counters_sum(const FeedCounters* per_block, uint32_t n_blocks) {
    FeedCounters sum{};
    uint64_t* s = reinterpret_cast<uint64_t*>(&sum);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint64_t* g = reinterpret_cast<const uint64_t*>(&per_block[b]);
        for (uint32_t k = 0; k < kNumCounters; ++k)
            s[k] = k == kMaxSeqCounter ? (g[k] > s[k] ? g[k] : s[k]) : s[k] + g[k];
    }
    return sum;
}

}  // namespace vf

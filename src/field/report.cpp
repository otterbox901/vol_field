//
// src/field/report.cpp - the end-of-run summary.
//
// Replaces kernel_bypass's PollStats report: what matters here is not how
// fast frames were observed but what the field made of them - messages by
// kind, stale drops, unmapped quotes, IV failures, and how big the tables got.
//

#include <cstdio>
#include <vector>

#include "vf/run.hpp"

namespace vf {
namespace {

double pct(uint64_t part, uint64_t whole) {
    return whole ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
}

unsigned long long u(uint64_t v) { return static_cast<unsigned long long>(v); }

void counter_block(const FeedCounters& c, double secs, const char* indent) {
    std::printf("%sframes        %llu (%.2f MB), not a MoldUDP64 packet: %llu\n", indent,
                u(c.frames), static_cast<double>(c.bytes) / 1e6, u(c.not_feed));
    std::printf("%smessages      %llu (%s msg/s), malformed %llu\n", indent, u(c.messages),
                si(secs > 0 ? static_cast<double>(c.messages) / secs : 0).c_str(),
                u(c.malformed));
    std::printf("%s  directory   %llu for our underlyings; admin (T S H O, other D) %llu; "
                "trades (R X) %llu\n",
                indent, u(c.directory), u(c.admin), u(c.trades));
    std::printf("%s  packets     heartbeats %llu, end of session %llu\n", indent,
                u(c.heartbeats), u(c.end_of_session));
    std::printf("%squotes        %llu mapped, %llu unmapped\n", indent, u(c.quotes),
                u(c.unmapped));
    std::printf("%s  stale       %llu (%.2f%% of mapped: duplicates, or overtaken by newer)\n",
                indent, u(c.stale), pct(c.stale, c.quotes));
    std::printf("%sIV on tick    %llu solved, %llu failed (%.3f%%)\n", indent, u(c.iv_solved),
                u(c.iv_failed), pct(c.iv_failed, c.iv_solved + c.iv_failed));
    std::printf("%smax seq       %llu\n", indent, u(c.max_seq));
}

}  // namespace

void report(const RunConfig& cfg, const Field* field, const SurfaceStats& surface,
            uint64_t elapsed_ns, const std::string& source) {
    const uint32_t n = cfg.queues;
    std::vector<FeedCounters> per(n);
    field_counters(field, per.data());
    const FeedCounters sum = counters_sum(per.data(), n);
    const FieldSizes sz = field_sizes(field);
    const double secs = static_cast<double>(elapsed_ns) / 1e9;

    std::printf("\n=== vol_field run ===\n");
    std::printf("source        %s\n", source.c_str());
    std::printf("elapsed       %.2f s, %u receive block(s)\n", secs, n);
    counter_block(sum, secs, "");
    std::printf("field         %u options, %u expiries, %u strikes (pairs)%s\n", sz.options,
                sz.expiries, sz.pairs, sz.overflow ? "" : ", no table overflow");
    if (sz.overflow)
        std::printf("  OVERFLOW    %u inserts refused: a table is full (see vf/field.hpp)\n",
                    sz.overflow);
    std::printf("feed clock    %s\n", clock_string(sz.feed_sec).c_str());
    std::printf("surface       %llu passes every %u ms, GPU %.3f ms avg / %.3f ms max, "
                "%llu ticks skipped\n",
                u(surface.passes), cfg.snapshot_ms,
                surface.passes ? surface.gpu_ms_sum / static_cast<double>(surface.passes) : 0.0,
                surface.gpu_ms_max, u(surface.skipped));
    std::printf("snapshot      %s%s\n",
                cfg.shm.find('/') == std::string::npos ? "/dev/shm/" : "", cfg.shm.c_str());

    if (n > 1) {
        for (uint32_t q = 0; q < n; ++q) {
            std::printf("--- queue %u\n", q);
            counter_block(per[q], secs, "  ");
        }
    }
    if (sum.unmapped && sum.directory == 0)
        std::printf("note: no directory messages for %s were seen; a capture must include the "
                    "start-of-day directory spin (Glimpse late join is not supported)\n",
                    cfg.underlyings.front().c_str());
}

}  // namespace vf

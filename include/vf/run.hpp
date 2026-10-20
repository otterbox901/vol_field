#pragma once
//
// vf/run.hpp - the host side of a run, shared by vf_sim and vf_gpunetio: the
// periodic surface pass, the status line, Ctrl-C, and the final report.
//
// The host never touches packets or the surface data. During a run it only
// launches a surface pass every --snapshot-ms and reads the snapshot header
// (which the GPU wrote) to print one status line a second.
//

#include <functional>
#include <string>

#include "vf/config.hpp"
#include "vf/field.hpp"
#include "vf/snapshot.hpp"

struct CUevent_st;

namespace vf {

/// SIGINT / SIGTERM ask the run to finish cleanly (drain, final snapshot,
/// report) instead of killing it with a persistent kernel still resident.
void install_stop_signals();
bool stop_requested();

struct SurfaceStats {
    uint64_t passes = 0;      ///< surface passes that completed
    uint64_t skipped = 0;     ///< ticks skipped: the previous pass was still running
    double   gpu_ms_sum = 0;  ///< GPU time of completed passes
    double   gpu_ms_max = 0;
};

/// Launches surface passes on its own non-blocking stream, at most one in
/// flight, and times them with events.
class SurfaceRunner {
public:
    SurfaceRunner(Field* field, const Snapshot& snap);
    ~SurfaceRunner();
    SurfaceRunner(const SurfaceRunner&) = delete;
    SurfaceRunner& operator=(const SurfaceRunner&) = delete;

    /// Launch a pass unless the previous one is still running.
    void tick();
    /// Launch the last pass with state kSnapFinal and wait for it.
    void finish();

    const SurfaceStats& stats() const { return stats_; }

private:
    void collect(bool wait);

    Field*       field_;
    SnapHeader*  snap_dev_;
    CUstream_st* stream_ = nullptr;
    CUevent_st*  start_ = nullptr;
    CUevent_st*  stop_ = nullptr;
    bool         in_flight_ = false;
    SurfaceStats stats_;
};

/// Drive the run: a surface pass every cfg.snapshot_ms and a status line
/// every second, until cfg.duration_ms passes, stop_requested(), or
/// `source_done()` says the frame source has nothing more to deliver.
/// `source_status()` adds the source's own part of the status line.
/// Returns the elapsed host time in ns.
uint64_t run_loop(const RunConfig& cfg, SurfaceRunner& surface, const Snapshot& snap,
                  const std::function<bool()>& source_done,
                  const std::function<std::string()>& source_status);

/// End-of-run summary: field counters and table sizes, and the surface pass.
/// `source` is one line describing where the frames came from.
void report(const RunConfig& cfg, const Field* field, const SurfaceStats& surface,
            uint64_t elapsed_ns, const std::string& source);

/// "09:30:05" from seconds past midnight.
std::string clock_string(uint32_t feed_sec);

/// 1234567 -> "1.23M", for status lines.
std::string si(double v);

/// Check the current device is sm_75 or newer, print what it is, and return
/// its SM count (0 on failure, after printing why).
int check_device(bool verbose);

}  // namespace vf

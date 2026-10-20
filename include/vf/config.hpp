#pragma once
//
// vf/config.hpp - the run configuration both drivers share. Host only.
//

#include <string>
#include <vector>

#include "vf/common.hpp"

namespace vf {

/// Everything vf_sim and vf_gpunetio both take from the command line. Each
/// driver adds its own frame-source flags on top.
struct RunConfig {
    std::vector<std::string> underlyings;  ///< --underlying SYM[,SYM...] (required)
    std::string date;                      ///< --date YYYY-MM-DD, the trade date (default: today)
    int32_t     trade_days = 0;            ///< the same date, days since 1970-01-01
    double      rate = 0.0;                ///< --rate, continuously compounded
    uint32_t    queues = 1;                ///< --queues: receive blocks, one per queue
    uint32_t    duration_ms = 10000;       ///< --duration; 0 = until Ctrl-C
    uint32_t    snapshot_ms = 100;         ///< --snapshot-ms: surface pass period
    std::string shm = "vol_field";         ///< --shm: /dev/shm name, or a path on any tmpfs
    bool        verbose = false;
};

}  // namespace vf

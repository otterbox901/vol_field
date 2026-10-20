//
// src/gpunetio/main_gpunetio.cpp - vf_gpunetio: the live surface, NIC -> VRAM.
//
// Same run as vf_sim - field in VRAM, persistent receive kernel, a surface
// pass every --snapshot-ms into shared memory - but nothing on the data path:
// the NIC writes BONO frames into GPU memory and the kernel parses them there.
// This thread only launches surface passes and prints the status line.
//

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vf/cli.hpp"
#include "vf/config.hpp"
#include "vf/field.hpp"
#include "vf/gpunetio.hpp"
#include "vf/run.hpp"
#include "vf/snapshot.hpp"

namespace {

void usage(const char* argv0) {
    std::printf("usage: %s --nic PCI --gpu PCI --feed G:P[,..] --underlying SYM [options]\n\n",
                argv0);
    const auto line = [](const char* flag, const char* help) {
        std::printf("  %-*s%s\n", vf::kUsageWidth, flag, help);
    };
    std::printf("receive path:\n");
    line("--nic PCI", "NIC PCIe address, e.g. 0000:c1:00.0 (required)");
    line("--gpu PCI", "GPU PCIe address, e.g. 0000:01:00.0 (required)");
    line("--feed G:P[,..]", "BONO feeds to steer into VRAM: group (or :PORT for any) and port");
    std::printf("run:\n");
    vf::print_common_usage();
    std::printf("\nscripts/hw-preflight.sh checks the machine and prints a command line.\n");
}

bool parse_args(int argc, char** argv, vf::RunConfig& cfg, vf::GpunetioConfig& gcfg,
                bool& want_help) {
    for (int i = 1; i < argc; ++i) {
        switch (vf::parse_common_flag(argc, argv, i, cfg)) {
            case vf::FlagResult::kOk: continue;
            case vf::FlagResult::kBad: return false;
            case vf::FlagResult::kHelp: want_help = true; return true;
            case vf::FlagResult::kNotMine: break;
        }
        const char* a = argv[i];
        const char* s = nullptr;
        bool ok = true;
        if (!std::strcmp(a, "--nic")) {
            ok = vf::take_str(argc, argv, i, gcfg.nic_pci);
        } else if (!std::strcmp(a, "--gpu")) {
            ok = vf::take_str(argc, argv, i, gcfg.gpu_pci);
        } else if (!std::strcmp(a, "--feed")) {
            ok = vf::take_str(argc, argv, i, s) && vf::parse_endpoints(s, gcfg.feeds);
        } else {
            std::fprintf(stderr, "[vf] unknown option: %s\n", a);
            return false;
        }
        if (!ok) {
            std::fprintf(stderr, "[vf] bad or missing value for %s\n", a);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    vf::RunConfig cfg;
    vf::GpunetioConfig gcfg;
    bool want_help = false;
    if (!parse_args(argc, argv, cfg, gcfg, want_help)) return 2;
    if (want_help) {
        usage(argv[0]);
        return 0;
    }
    if (!vf::finish_config(cfg)) return 2;
    if (!gcfg.nic_pci || !gcfg.gpu_pci || gcfg.feeds.empty()) {
        std::fprintf(stderr, "[vf] --nic, --gpu and --feed are required\n");
        return 2;
    }

    std::vector<const char*> names;
    for (const std::string& u : cfg.underlyings) names.push_back(u.c_str());
    vf::FieldParams params{};
    if (!vf::field_params(names.data(), static_cast<uint32_t>(names.size()), cfg.trade_days,
                          cfg.rate, params))
        return 2;

    // NIC first: on a machine without one this is where the run ends.
    vf::GpunetioSession* s = vf::gpunetio_open(cfg, gcfg);
    if (!s) return 1;

    vf::install_stop_signals();
    vf::Field* field = vf::field_create(params, cfg.queues);
    vf::Snapshot snap;
    if (!vf::snapshot_open(cfg.shm.c_str(), params, snap)) {
        vf::field_destroy(field);
        vf::gpunetio_close(s);
        return 1;
    }
    if (!vf::gpunetio_launch(s, vf::field_dev(field))) {
        vf::snapshot_close(snap);
        vf::field_destroy(field);
        vf::gpunetio_close(s);
        return 1;
    }

    std::printf("[vf] %s\n", vf::gpunetio_describe(s).c_str());
    std::printf("[vf] %u receive block(s) x 512 threads; surface every %u ms into %s\n", cfg.queues,
                cfg.snapshot_ms, snap.path);
    std::printf("[vf] underlying %s, trade date %s, rate %.4g, %s\n", cfg.underlyings[0].c_str(),
                cfg.date.c_str(), cfg.rate,
                cfg.duration_ms ? ("for " + std::to_string(cfg.duration_ms) + " ms").c_str()
                                : "until Ctrl-C");

    // The surface passes run on their own stream next to the resident receive
    // kernel; field_create already loaded their module (lazy loading would
    // otherwise wait for the persistent kernel to exit).
    vf::SurfaceRunner surface(field, snap);
    const uint64_t elapsed = vf::run_loop(
        cfg, surface, snap, [&] { return vf::gpunetio_failed(s); }, [] { return std::string(); });

    const bool ok = vf::gpunetio_stop(s);
    surface.finish();
    vf::report(cfg, field, surface.stats(), elapsed, vf::gpunetio_describe(s));

    vf::snapshot_close(snap);
    vf::field_destroy(field);
    vf::gpunetio_close(s);
    return ok ? 0 : 1;
}

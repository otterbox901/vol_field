//
// src/sim/main_sim.cpp - vf_sim: the live surface from a pcap or a UDP socket.
//
// Same run as vf_gpunetio - field in VRAM, persistent receive kernel, surface
// pass every --snapshot-ms into shared memory - with the NIC replaced by a
// host thread that copies frames into a VRAM ring. Runs on any sm_75+ GPU.
//

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "vf/cli.hpp"
#include "vf/config.hpp"
#include "vf/field.hpp"
#include "vf/run.hpp"
#include "vf/snapshot.hpp"

#include "sim.hpp"
#include "source.hpp"

namespace {

struct SimConfig {
    const char*                    pcap = nullptr;
    std::vector<vf::Endpoint>      udp;
    uint32_t                       udp_iface = 0;
    vf::sim::PcapOptions           pcap_opt;
    uint32_t                       backoff_ns = 0;
};

void usage(const char* argv0) {
    std::printf("usage: %s (--pcap FILE | --udp G:P[,..]) --underlying SYM [options]\n\n", argv0);
    const auto line = [](const char* flag, const char* help) {
        std::printf("  %-*s%s\n", vf::kUsageWidth, flag, help);
    };
    std::printf("frame source (one of):\n");
    line("--pcap FILE", "replay an Ethernet pcap of MoldUDP64 / BONO frames");
    line("--udp G:P[,..]", "receive on UDP sockets; multicast groups are joined");
    line("--udp-iface IP", "interface address for the multicast joins (default any)");
    std::printf("pcap replay:\n");
    line("--feed G:P[,..]", "only frames to these destinations (default: every UDP frame)");
    line("--realtime", "pace by pcap timestamps (the default)");
    line("--speed X", "realtime, X times faster");
    line("--pps N", "N frames/s; 0 = unpaced, as fast as the GPU takes them");
    line("--loop", "replay forever, shifting sequence numbers each lap");
    std::printf("kernel:\n");
    line("--backoff NS", "__nanosleep when the ring is empty, 0 = spin (default 0)");
    std::printf("run:\n");
    vf::print_common_usage();
}

bool parse_args(int argc, char** argv, vf::RunConfig& cfg, SimConfig& sim, bool& want_help) {
    for (int i = 1; i < argc; ++i) {
        switch (vf::parse_common_flag(argc, argv, i, cfg)) {
            case vf::FlagResult::kOk: continue;
            case vf::FlagResult::kBad: return false;
            case vf::FlagResult::kHelp: want_help = true; return true;
            case vf::FlagResult::kNotMine: break;
        }
        const char* a = argv[i];
        const char* s = nullptr;
        uint64_t v = 0;
        bool ok = true;
        if (!std::strcmp(a, "--pcap")) {
            ok = vf::take_str(argc, argv, i, sim.pcap);
        } else if (!std::strcmp(a, "--udp")) {
            ok = vf::take_str(argc, argv, i, s) && vf::parse_endpoints(s, sim.udp);
        } else if (!std::strcmp(a, "--udp-iface")) {
            uint16_t unused = 0;
            ok = vf::take_str(argc, argv, i, s) &&
                 vf::parse_endpoint((std::string(s) + ":1").c_str(), sim.udp_iface, unused);
        } else if (!std::strcmp(a, "--feed")) {
            ok = vf::take_str(argc, argv, i, s) && vf::parse_endpoints(s, sim.pcap_opt.feeds);
        } else if (!std::strcmp(a, "--realtime")) {
            sim.pcap_opt.pps = -1.0;
        } else if (!std::strcmp(a, "--speed")) {
            ok = vf::take_double(argc, argv, i, sim.pcap_opt.speed) && sim.pcap_opt.speed > 0;
            sim.pcap_opt.pps = -1.0;
        } else if (!std::strcmp(a, "--pps")) {
            ok = vf::take_double(argc, argv, i, sim.pcap_opt.pps) && sim.pcap_opt.pps >= 0;
        } else if (!std::strcmp(a, "--loop")) {
            sim.pcap_opt.loop = true;
        } else if (!std::strcmp(a, "--backoff")) {
            ok = vf::take_u64(argc, argv, i, v) && v <= 1000000;
            sim.backoff_ns = static_cast<uint32_t>(v);
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
    SimConfig sim;
    bool want_help = false;
    if (!parse_args(argc, argv, cfg, sim, want_help)) return 2;
    if (want_help) {
        usage(argv[0]);
        return 0;
    }
    if (!vf::finish_config(cfg)) return 2;
    if ((sim.pcap != nullptr) == !sim.udp.empty()) {
        std::fprintf(stderr, "[vf] give exactly one of --pcap and --udp\n");
        return 2;
    }

    if (vf::check_device(cfg.verbose) == 0) return 1;
    const uint32_t max_queues = vf::sim::max_resident_queues();
    if (cfg.queues > max_queues) {
        std::fprintf(stderr,
                     "[vf] --queues %u: at most %u receive blocks fit on this GPU at once, and a "
                     "persistent block that is not resident never starts\n",
                     cfg.queues, max_queues);
        return 2;
    }

    // Frame source first: a bad pcap should fail before anything is allocated.
    std::unique_ptr<vf::sim::Source> source;
    std::string err;
    if (sim.pcap) {
        sim.pcap_opt.path = sim.pcap;
        auto p = std::make_unique<vf::sim::PcapSource>();
        if (!p->load(sim.pcap_opt, cfg.queues, err)) {
            std::fprintf(stderr, "[vf] %s\n", err.c_str());
            return 1;
        }
        source = std::move(p);
    } else {
        auto u = std::make_unique<vf::sim::UdpSource>();
        if (!u->open(sim.udp, sim.udp_iface, cfg.queues, err)) {
            std::fprintf(stderr, "[vf] %s\n", err.c_str());
            return 1;
        }
        source = std::move(u);
    }

    std::vector<const char*> names;
    for (const std::string& u : cfg.underlyings) names.push_back(u.c_str());
    vf::FieldParams params{};
    if (!vf::field_params(names.data(), static_cast<uint32_t>(names.size()), cfg.trade_days,
                          cfg.rate, params))
        return 2;

    vf::install_stop_signals();
    vf::Field* field = vf::field_create(params, cfg.queues);
    vf::Snapshot snap;
    if (!vf::snapshot_open(cfg.shm.c_str(), params, snap)) {
        vf::field_destroy(field);
        return 1;
    }

    std::atomic<bool> abort{false};
    vf::sim::Rings rings;
    rings.create(cfg.queues, &abort);

    cudaStream_t kstream = nullptr;
    VF_CUDA_CHECK(cudaStreamCreateWithFlags(&kstream, cudaStreamNonBlocking));
    // Hard ceiling, so a wedged host cannot leave the kernel resident forever.
    const uint64_t max_run_ms =
        cfg.duration_ms ? uint64_t{cfg.duration_ms} + 10000 : 7ull * 24 * 3600 * 1000;
    const uint64_t max_run_ns = max_run_ms * 1000000ull;
    if (!vf::sim::launch_sim_kernel(kstream, vf::field_dev(field), rings.queues_dev(), cfg.queues,
                                    rings.exit_flag_dev(), max_run_ns, sim.backoff_ns)) {
        std::fprintf(stderr, "[vf] receive kernel launch failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    std::printf("[vf] %s -> %u receive block(s) x %u threads; surface every %u ms into %s\n",
                sim.pcap ? sim.pcap : "UDP", cfg.queues, 512u, cfg.snapshot_ms, snap.path);
    std::printf("[vf] underlying %s, trade date %s, rate %.4g, %s\n", cfg.underlyings[0].c_str(),
                cfg.date.c_str(), cfg.rate,
                cfg.duration_ms ? ("for " + std::to_string(cfg.duration_ms) + " ms").c_str()
                                : "until Ctrl-C");

    vf::SurfaceRunner surface(field, snap);
    source->start(rings);
    const uint64_t elapsed = vf::run_loop(
        cfg, surface, snap, [&] { return source->finished() && rings.drained(); },
        [&] { return source->status(); });

    // Stop the producer, let the last copies land, then let the kernel drain
    // and retire. Only then is the final surface pass meaningful.
    abort.store(true);
    source->stop();
    rings.sync();
    rings.request_exit();
    const cudaError_t kerr = cudaStreamSynchronize(kstream);
    if (kerr != cudaSuccess)
        std::fprintf(stderr, "[vf] receive kernel failed: %s\n", cudaGetErrorString(kerr));
    surface.finish();

    vf::report(cfg, field, surface.stats(), elapsed, source->describe());

    source.reset();
    rings.destroy();
    cudaStreamDestroy(kstream);
    vf::snapshot_close(snap);
    vf::field_destroy(field);
    return kerr == cudaSuccess ? 0 : 1;
}

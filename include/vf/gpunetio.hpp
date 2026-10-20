#pragma once
//
// vf/gpunetio.hpp - the GPU-direct receive path of vf_gpunetio.
//
// The NIC writes whole Ethernet frames straight into GPU memory, and the
// persistent receive kernel parses them there into the vol field. No CPU
// thread touches a packet:
//
//   * DOCA Flow programs the NIC's hardware steering: a root control pipe with
//     one entry per --feed (destination group and UDP port) forwards to an RSS
//     pipe over the receive queues. Everything else - IGMP, ssh - falls
//     through to the kernel stack, so the host can still join the groups;
//   * each queue's packet buffer is doca_gpu_mem_alloc'd VRAM, mapped for the
//     NIC through dmabuf (or nvidia-peermem on older stacks);
//   * block q of the kernel calls doca_gpu_dev_eth_rxq_recv on queue q and
//     hands the batch to vf::dev::process_batch - the same code vf_sim runs.
//
// The host joins the multicast groups on plain sockets (on the NIC's netdev),
// so the switch forwards the feed; those sockets never see a datagram,
// because the NIC steers the frames into VRAM first.
//
// Needs a ConnectX-6 Dx / -7 or BlueField-2/3 NIC and a GPU with enough BAR1
// space; pre-Hopper GPUs (all of sm_75..sm_89) also need an MCST QP, which
// setup enables. Built only when the DOCA SDK was found: scripts/doca-build.sh.
//

#include <string>
#include <vector>

#include "vf/cli.hpp"
#include "vf/config.hpp"
#include "vf/field.hpp"

namespace vf {

struct GpunetioConfig {
    const char*           nic_pci = nullptr;  ///< NIC PCIe address, e.g. "0000:c1:00.0"
    const char*           gpu_pci = nullptr;  ///< GPU PCIe address, e.g. "0000:01:00.0"
    std::vector<Endpoint> feeds;              ///< steered into VRAM; multicast ones are joined
};

struct GpunetioSession;  ///< opaque: DOCA devices, queues, VRAM buffers, flow pipes, joins

/// Open the NIC (first, so a machine without one fails before anything else),
/// select and check the GPU, start cfg.queues receive queues with their
/// buffers in VRAM, install the steering for every feed, and join the
/// multicast groups. Returns nullptr after printing why, undoing what was set
/// up. Needs the same privileges any NIC-programming tool does.
GpunetioSession* gpunetio_open(const RunConfig& cfg, const GpunetioConfig& gcfg);

/// Launch the persistent receive kernel on its own stream: block q drains
/// queue q into the field `f` (made with n_blocks == cfg.queues).
bool gpunetio_launch(GpunetioSession* s, const FieldDev& f);

/// True once the kernel stopped by itself (a queue reported a receive error).
bool gpunetio_failed(const GpunetioSession* s);

/// Ask the kernel to retire and wait for it. False if it or the launch failed.
bool gpunetio_stop(GpunetioSession* s);

/// Leave the groups and tear down the flow, queues, buffers and devices.
void gpunetio_close(GpunetioSession* s);

/// One line for the report: NIC, netdev, GPU, feeds.
std::string gpunetio_describe(const GpunetioSession* s);

}  // namespace vf

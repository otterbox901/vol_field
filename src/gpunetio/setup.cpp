//
// src/gpunetio/setup.cpp - bring up the NIC -> VRAM receive path.
//
// Order, and every step matters:
//   1. open the NIC by PCI address; select the CUDA device with the GPU's PCI
//      address and check it (sm_75+) before DOCA attaches to it.
//   2. per queue: doca_eth_rxq_create (CYCLIC), its packet buffer in VRAM via
//      doca_gpu_mem_alloc, mapped for the NIC by dmabuf (or nvidia-peermem),
//      an MCST QP on pre-Hopper GPUs, and the data path on the GPU (GDAKI).
//   3. DOCA Flow: an RSS pipe over the queues, and a root control pipe with one
//      entry per --feed matching its destination address and UDP port. The
//      match runs in NIC hardware; unmatched traffic goes to the kernel stack.
//   4. join the multicast groups on plain sockets bound to the NIC's netdev,
//      so the switch forwards them (the sockets themselves stay empty).
//   5. GPU-visible control: exit flag and the queue handle array.
//

#include "vf/gpunetio.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_eth_rxq.h>
#include <doca_eth_rxq_gpu_data_path.h>
#include <doca_flow.h>
#include <doca_gpunetio.h>
#include <doca_mmap.h>

#include "vf/run.hpp"

#include "kernel.h"

namespace vf {
namespace {

// Big enough that the NIC does not wrap a queue's cyclic buffer while a batch
// is being processed; frames up to 2 KiB (BONO packets stay under the MTU).
constexpr uint32_t kMaxPktNum = 16384;
constexpr uint32_t kMaxPktSize = 2048;
constexpr uint32_t kFlowCounters = 64;

/// Log a failed DOCA call and return false, so callers read as a flat chain.
bool fail(const char* what, doca_error_t err) {
    std::fprintf(stderr, "[vf] %s: %s\n", what, doca_error_get_descr(err));
    return false;
}

size_t page_size() { return static_cast<size_t>(sysconf(_SC_PAGESIZE)); }

size_t align_up(size_t v, size_t a) { return ((v + a - 1) / a) * a; }

bool is_multicast(uint32_t ip) { return (ip >> 28) == 0xE; }

/// Open the NIC whose PCI address matches. DOCA has no open-by-address call;
/// its samples carry a helper like this one.
bool open_nic_by_pci(const char* pci_addr, doca_dev** out) {
    doca_devinfo** list = nullptr;
    uint32_t count = 0;

    doca_error_t r = doca_devinfo_create_list(&list, &count);
    if (r != DOCA_SUCCESS && r != DOCA_ERROR_NOT_FOUND) return fail("doca_devinfo_create_list", r);

    for (uint32_t i = 0; i < count; ++i) {
        uint8_t same = 0;
        if (doca_devinfo_is_equal_pci_addr(list[i], pci_addr, &same) != DOCA_SUCCESS || !same)
            continue;
        r = doca_dev_open(list[i], out);
        doca_devinfo_destroy_list(list);
        return r == DOCA_SUCCESS ? true : fail("doca_dev_open", r);
    }

    if (list) doca_devinfo_destroy_list(list);
    std::fprintf(stderr,
                 "[vf] no DOCA device at PCI %s. The GPU-direct path needs a ConnectX-6 Dx or "
                 "newer, or a BlueField-2/3; list candidates with `doca_caps --list-devs` or "
                 "`ibv_devinfo`, or run scripts/hw-preflight.sh.\n",
                 pci_addr);
    return false;
}

/// One receive queue: a DOCA queue whose packet buffer lives in GPU memory.
struct Queue {
    doca_eth_rxq*     rxq_cpu = nullptr;
    doca_gpu_eth_rxq* rxq_gpu = nullptr;
    doca_ctx*         ctx = nullptr;
    doca_mmap*        mmap = nullptr;
    void*             pkt_addr = nullptr;  ///< the VRAM the NIC writes into
    int               dmabuf_fd = -1;
};

}  // namespace

struct GpunetioSession {
    std::string            nic_pci;
    std::string            gpu_pci;
    std::string            iface;  ///< the NIC's netdev, for the joins
    std::vector<Endpoint>  feeds;

    doca_dev*       ddev = nullptr;
    doca_gpu*       gpu = nullptr;
    doca_flow_port* port = nullptr;
    doca_flow_pipe* rxq_pipe = nullptr;
    doca_flow_pipe* root_pipe = nullptr;
    cudaStream_t    stream = nullptr;
    bool            flow_inited = false;
    bool            launched = false;

    std::vector<Queue> queues;
    std::vector<int>   join_fds;

    // GPU-visible control, written by the host, read by the kernel.
    uint32_t*          exit_flag_gpu = nullptr;
    uint32_t*          exit_flag_cpu = nullptr;
    doca_gpu_eth_rxq** rxqs_gpu = nullptr;
};

namespace {

/// Make the GPU at `pci` the current CUDA device, so the field, the snapshot
/// mapping and DOCA all land on the same one, and check it.
bool select_gpu(const char* pci) {
    int id = -1;
    if (cudaDeviceGetByPCIBusId(&id, pci) != cudaSuccess) {
        std::fprintf(stderr, "[vf] no CUDA device at PCI %s (nvidia-smi --query-gpu=pci.bus_id "
                             "--format=csv lists them)\n", pci);
        return false;
    }
    if (cudaSetDevice(id) != cudaSuccess) {
        std::fprintf(stderr, "[vf] cudaSetDevice(%d) failed\n", id);
        return false;
    }
    return check_device(false) != 0;
}

/// Allocate one queue's VRAM packet buffer and start it with the data path on
/// the GPU. After this the NIC can DMA frames straight into GPU memory.
bool queue_create(GpunetioSession* s, Queue& q) {
    doca_error_t r;
    uint32_t buf_size = 0;

    if ((r = doca_eth_rxq_create(s->ddev, kMaxPktNum, kMaxPktSize, &q.rxq_cpu)) != DOCA_SUCCESS)
        return fail("doca_eth_rxq_create", r);
    if ((r = doca_eth_rxq_set_type(q.rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC)) != DOCA_SUCCESS)
        return fail("doca_eth_rxq_set_type", r);
    if ((r = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, kMaxPktSize,
                                                   kMaxPktNum, 0, 0, 0, &buf_size)) != DOCA_SUCCESS)
        return fail("doca_eth_rxq_estimate_packet_buf_size", r);

    if ((r = doca_mmap_create(&q.mmap)) != DOCA_SUCCESS) return fail("doca_mmap_create", r);
    if ((r = doca_mmap_add_dev(q.mmap, s->ddev)) != DOCA_SUCCESS)
        return fail("doca_mmap_add_dev", r);

    const size_t bytes = align_up(buf_size, page_size());
    if ((r = doca_gpu_mem_alloc(s->gpu, bytes, page_size(), DOCA_GPU_MEM_TYPE_GPU, &q.pkt_addr,
                                nullptr)) != DOCA_SUCCESS || !q.pkt_addr)
        return fail("doca_gpu_mem_alloc (packet buffer in VRAM)", r);

    // dmabuf is the supported way to let the NIC reach GPU memory; the
    // memrange path needs the legacy nvidia-peermem module instead.
    if (doca_gpu_dmabuf_fd(s->gpu, q.pkt_addr, bytes, &q.dmabuf_fd) != DOCA_SUCCESS) {
        std::printf("[vf] dmabuf unavailable, mapping %zu B of VRAM via nvidia-peermem\n", bytes);
        if ((r = doca_mmap_set_memrange(q.mmap, q.pkt_addr, bytes)) != DOCA_SUCCESS)
            return fail("doca_mmap_set_memrange", r);
    } else {
        if ((r = doca_mmap_set_dmabuf_memrange(q.mmap, q.dmabuf_fd, q.pkt_addr, 0, bytes)) !=
            DOCA_SUCCESS)
            return fail("doca_mmap_set_dmabuf_memrange", r);
    }

    if ((r = doca_mmap_set_permissions(q.mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE)) != DOCA_SUCCESS)
        return fail("doca_mmap_set_permissions", r);
    if ((r = doca_mmap_start(q.mmap)) != DOCA_SUCCESS) return fail("doca_mmap_start", r);
    if ((r = doca_eth_rxq_set_pkt_buf(q.rxq_cpu, q.mmap, 0, bytes)) != DOCA_SUCCESS)
        return fail("doca_eth_rxq_set_pkt_buf", r);

    // Pre-Hopper parts cannot drive the queue doorbell from the SM directly and
    // need the multicast-QP helper: every GPU this project supports (sm_75 to
    // sm_89) lands here.
    int device = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 9) {
        if ((r = doca_eth_rxq_gpu_enable_mcst_qp(q.rxq_cpu)) != DOCA_SUCCESS)
            return fail("doca_eth_rxq_gpu_enable_mcst_qp", r);
    }

    q.ctx = doca_eth_rxq_as_doca_ctx(q.rxq_cpu);
    if (!q.ctx) return fail("doca_eth_rxq_as_doca_ctx", DOCA_ERROR_UNEXPECTED);
    if ((r = doca_ctx_set_datapath_on_gpu(q.ctx, s->gpu)) != DOCA_SUCCESS)
        return fail("doca_ctx_set_datapath_on_gpu", r);
    if ((r = doca_ctx_start(q.ctx)) != DOCA_SUCCESS) return fail("doca_ctx_start", r);
    if ((r = doca_eth_rxq_get_gpu_handle(q.rxq_cpu, &q.rxq_gpu)) != DOCA_SUCCESS)
        return fail("doca_eth_rxq_get_gpu_handle", r);

    return true;
}

/// RSS pipe spreading matched traffic over every queue. A feed is one
/// (source, group, port) 4-tuple, so all of it hashes to one queue and the
/// feeds spread across queues.
bool create_rxq_pipe(GpunetioSession* s) {
    doca_flow_match match{};
    match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
    match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_UDP;

    std::vector<uint16_t> rss_queues(s->queues.size());
    for (size_t i = 0; i < s->queues.size(); ++i) {
        doca_eth_rxq_apply_queue_id(s->queues[i].rxq_cpu, static_cast<uint16_t>(i));
        rss_queues[i] = static_cast<uint16_t>(i);
    }

    doca_flow_fwd fwd{};
    fwd.type = DOCA_FLOW_FWD_RSS;
    fwd.rss_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
    fwd.rss.queues_array = rss_queues.data();
    fwd.rss.outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
    fwd.rss.nr_queues = static_cast<uint32_t>(rss_queues.size());

    // Anything that reaches this pipe already matched a feed, so a miss here
    // is traffic we asked for and cannot place: drop it rather than leak it to
    // the kernel stack.
    doca_flow_fwd miss{};
    miss.type = DOCA_FLOW_FWD_DROP;

    doca_flow_monitor monitor{};
    monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

    doca_flow_pipe_cfg* cfg = nullptr;
    doca_error_t r = doca_flow_pipe_cfg_create(&cfg, s->port);
    if (r != DOCA_SUCCESS) return fail("doca_flow_pipe_cfg_create", r);

    bool ok = (doca_flow_pipe_cfg_set_name(cfg, "VF_RXQ_PIPE") == DOCA_SUCCESS) &&
              (doca_flow_pipe_cfg_set_type(cfg, DOCA_FLOW_PIPE_BASIC) == DOCA_SUCCESS) &&
              (doca_flow_pipe_cfg_set_is_root(cfg, false) == DOCA_SUCCESS) &&
              (doca_flow_pipe_cfg_set_match(cfg, &match, nullptr) == DOCA_SUCCESS) &&
              (doca_flow_pipe_cfg_set_monitor(cfg, &monitor) == DOCA_SUCCESS);
    if (ok) {
        r = doca_flow_pipe_create(cfg, &fwd, &miss, &s->rxq_pipe);
        ok = (r == DOCA_SUCCESS);
    }
    doca_flow_pipe_cfg_destroy(cfg);
    if (!ok) return fail("rxq pipe creation", r);

    doca_flow_pipe_entry* entry = nullptr;
    if ((r = doca_flow_pipe_basic_add_entry(0, s->rxq_pipe, &match, 0, nullptr, nullptr, nullptr,
                                            DOCA_FLOW_ENTRY_FLAGS_NO_WAIT, nullptr, &entry)) !=
        DOCA_SUCCESS)
        return fail("rxq pipe entry", r);
    if ((r = doca_flow_entries_process(s->port, 0, 0, 0)) != DOCA_SUCCESS)
        return fail("doca_flow_entries_process", r);
    return true;
}

/// Root pipe: one entry per feed, matching IPv4/UDP to its destination
/// address (unless the feed gave none) and port, forwarding to the RSS pipe.
/// A zero field in a control entry's match is "don't care".
bool create_root_pipe(GpunetioSession* s) {
    doca_flow_monitor monitor{};
    monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

    doca_flow_pipe_cfg* cfg = nullptr;
    doca_error_t r = doca_flow_pipe_cfg_create(&cfg, s->port);
    if (r != DOCA_SUCCESS) return fail("doca_flow_pipe_cfg_create (root)", r);

    bool ok = (doca_flow_pipe_cfg_set_name(cfg, "VF_ROOT_PIPE") == DOCA_SUCCESS) &&
              (doca_flow_pipe_cfg_set_type(cfg, DOCA_FLOW_PIPE_CONTROL) == DOCA_SUCCESS) &&
              (doca_flow_pipe_cfg_set_is_root(cfg, true) == DOCA_SUCCESS) &&
              (doca_flow_pipe_cfg_set_monitor(cfg, &monitor) == DOCA_SUCCESS);
    if (ok) {
        r = doca_flow_pipe_create(cfg, nullptr, nullptr, &s->root_pipe);
        ok = (r == DOCA_SUCCESS);
    }
    doca_flow_pipe_cfg_destroy(cfg);
    if (!ok) return fail("root pipe creation", r);

    doca_flow_fwd fwd{};
    fwd.type = DOCA_FLOW_FWD_PIPE;
    fwd.next_pipe = s->rxq_pipe;

    for (const Endpoint& e : s->feeds) {
        doca_flow_match match{};
        match.outer.eth.type = htons(DOCA_FLOW_ETHER_TYPE_IPV4);
        match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
        match.outer.ip4.next_proto = IPPROTO_UDP;
        match.outer.ip4.dst_ip = htonl(e.ip);
        match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
        match.outer.udp.l4_port.dst_port = htons(e.port);

        doca_flow_pipe_entry* entry = nullptr;
        if ((r = doca_flow_pipe_control_add_entry(0, s->root_pipe, &match, nullptr, nullptr,
                                                  nullptr, nullptr, nullptr, nullptr, 0, &fwd,
                                                  nullptr, &entry)) != DOCA_SUCCESS) {
            std::fprintf(stderr, "[vf] root pipe entry for %s: %s\n", endpoint_string(e).c_str(),
                         doca_error_get_descr(r));
            return false;
        }
    }
    if ((r = doca_flow_entries_process(s->port, 0, 0, 0)) != DOCA_SUCCESS)
        return fail("doca_flow_entries_process (root)", r);
    return true;
}

bool flow_start(GpunetioSession* s) {
    doca_flow_cfg* fcfg = nullptr;
    doca_error_t r = doca_flow_cfg_create(&fcfg);
    if (r != DOCA_SUCCESS) return fail("doca_flow_cfg_create", r);
    bool ok = (doca_flow_cfg_set_pipe_queues(fcfg, 1) == DOCA_SUCCESS) &&
              (doca_flow_cfg_set_mode_args(fcfg, "vnf") == DOCA_SUCCESS) &&
              (doca_flow_cfg_set_nr_counters(fcfg, kFlowCounters) == DOCA_SUCCESS);
    if (ok) {
        r = doca_flow_init(fcfg);
        ok = (r == DOCA_SUCCESS);
    }
    doca_flow_cfg_destroy(fcfg);
    if (!ok) return fail("doca_flow_init", r);
    s->flow_inited = true;

    doca_flow_port_cfg* pcfg = nullptr;
    if ((r = doca_flow_port_cfg_create(&pcfg)) != DOCA_SUCCESS)
        return fail("doca_flow_port_cfg_create", r);
    ok = (doca_flow_port_cfg_set_port_id(pcfg, 0) == DOCA_SUCCESS) &&
         (doca_flow_port_cfg_set_dev(pcfg, s->ddev) == DOCA_SUCCESS);
    if (ok) {
        r = doca_flow_port_start(pcfg, &s->port);
        ok = (r == DOCA_SUCCESS);
    }
    doca_flow_port_cfg_destroy(pcfg);
    if (!ok) return fail("doca_flow_port_start", r);
    return true;
}

/// IGMP-join every multicast feed on the NIC's netdev. The NIC steers the
/// datagrams into VRAM before the kernel stack sees them, so these sockets
/// only exist to keep the membership (the kernel answers the switch's IGMP
/// queries, which the root pipe does not match).
bool join_groups(GpunetioSession* s) {
    const unsigned ifindex = s->iface.empty() ? 0 : if_nametoindex(s->iface.c_str());
    if (!ifindex)
        std::printf("[vf] note: no netdev found for the NIC; joining on the default route's "
                    "interface\n");
    for (const Endpoint& e : s->feeds) {
        if (!is_multicast(e.ip)) continue;
        const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd < 0) {
            std::perror("[vf] socket");
            return false;
        }
        s->join_fds.push_back(fd);
        ip_mreqn m{};
        m.imr_multiaddr.s_addr = htonl(e.ip);
        m.imr_ifindex = static_cast<int>(ifindex);
        if (setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &m, sizeof m) != 0) {
            std::fprintf(stderr, "[vf] join %s on %s: %s\n", endpoint_string(e).c_str(),
                         s->iface.empty() ? "default" : s->iface.c_str(), std::strerror(errno));
            return false;
        }
    }
    return true;
}

void destroy(GpunetioSession* s) {
    if (!s) return;
    for (int fd : s->join_fds) ::close(fd);  // leaves the groups
    if (s->root_pipe) doca_flow_pipe_destroy(s->root_pipe);
    if (s->rxq_pipe) doca_flow_pipe_destroy(s->rxq_pipe);
    if (s->port) doca_flow_port_stop(s->port);
    if (s->flow_inited) doca_flow_destroy();

    for (Queue& q : s->queues) {
        if (q.ctx) doca_ctx_stop(q.ctx);
        if (q.rxq_cpu) doca_eth_rxq_destroy(q.rxq_cpu);
        if (q.mmap) doca_mmap_destroy(q.mmap);
        if (q.dmabuf_fd >= 0) ::close(q.dmabuf_fd);
        if (q.pkt_addr) doca_gpu_mem_free(s->gpu, q.pkt_addr);
    }
    if (s->exit_flag_gpu) doca_gpu_mem_free(s->gpu, s->exit_flag_gpu);
    if (s->rxqs_gpu) doca_gpu_mem_free(s->gpu, s->rxqs_gpu);
    if (s->stream) cudaStreamDestroy(s->stream);
    if (s->gpu) doca_gpu_destroy(s->gpu);
    if (s->ddev) doca_dev_close(s->ddev);
    delete s;
}

/// Shared helper for the two small CPU+GPU allocations.
template <class T>
bool alloc_shared(GpunetioSession* s, size_t bytes, T*& gpu, T*& cpu, const char* what) {
    void* g = nullptr;
    void* c = nullptr;
    const doca_error_t r =
        doca_gpu_mem_alloc(s->gpu, bytes, 4096, DOCA_GPU_MEM_TYPE_CPU_GPU, &g, &c);
    if (r != DOCA_SUCCESS) return fail(what, r);
    gpu = static_cast<T*>(g);
    cpu = static_cast<T*>(c);
    std::memset(cpu, 0, bytes);
    return true;
}

GpunetioSession* open_impl(const RunConfig& cfg, const GpunetioConfig& gcfg) {
    auto* s = new GpunetioSession();
    s->nic_pci = gcfg.nic_pci;
    s->gpu_pci = gcfg.gpu_pci;
    s->feeds = gcfg.feeds;
    doca_error_t r;

    if (!open_nic_by_pci(gcfg.nic_pci, &s->ddev)) return destroy(s), nullptr;
    char iface[DOCA_DEVINFO_IFACE_NAME_SIZE] = {};
    if (doca_devinfo_get_iface_name(doca_dev_as_devinfo(s->ddev), iface, sizeof iface) ==
        DOCA_SUCCESS)
        s->iface = iface;

    if (!select_gpu(gcfg.gpu_pci)) return destroy(s), nullptr;
    const uint32_t max_queues = gpunetio::max_resident_queues();
    if (cfg.queues > max_queues) {
        std::fprintf(stderr, "[vf] --queues %u: at most %u receive blocks fit on this GPU\n",
                     cfg.queues, max_queues);
        return destroy(s), nullptr;
    }
    if ((r = doca_gpu_create(gcfg.gpu_pci, &s->gpu)) != DOCA_SUCCESS)
        return fail("doca_gpu_create", r), destroy(s), nullptr;
    if (cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "[vf] cudaStreamCreateWithFlags failed\n");
        return destroy(s), nullptr;
    }

    if (!flow_start(s)) return destroy(s), nullptr;
    s->queues.resize(cfg.queues);
    for (Queue& q : s->queues)
        if (!queue_create(s, q)) return destroy(s), nullptr;
    if (!create_rxq_pipe(s) || !create_root_pipe(s) || !join_groups(s))
        return destroy(s), nullptr;

    doca_gpu_eth_rxq** handles_cpu = nullptr;
    if (!alloc_shared(s, sizeof(uint32_t), s->exit_flag_gpu, s->exit_flag_cpu, "exit flag") ||
        !alloc_shared(s, sizeof(doca_gpu_eth_rxq*) * s->queues.size(), s->rxqs_gpu, handles_cpu,
                      "queue handles"))
        return destroy(s), nullptr;
    for (size_t i = 0; i < s->queues.size(); ++i) handles_cpu[i] = s->queues[i].rxq_gpu;
    return s;
}

}  // namespace

GpunetioSession* gpunetio_open(const RunConfig& cfg, const GpunetioConfig& gcfg) {
    if (!gcfg.nic_pci || !gcfg.gpu_pci || gcfg.feeds.empty()) {
        std::fprintf(stderr, "[vf] --nic, --gpu and --feed are required\n");
        return nullptr;
    }
    for (const Endpoint& e : gcfg.feeds) {
        if (e.ip && !is_multicast(e.ip))
            std::printf("[vf] note: feed %s is unicast; steered, but not joined\n",
                        endpoint_string(e).c_str());
    }
    return open_impl(cfg, gcfg);
}

bool gpunetio_launch(GpunetioSession* s, const FieldDev& f) {
    s->launched = gpunetio::launch_receive_kernel(s->stream, f, s->rxqs_gpu,
                                                  static_cast<uint32_t>(s->queues.size()),
                                                  s->exit_flag_gpu);
    if (!s->launched)
        std::fprintf(stderr, "[vf] receive kernel launch failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
    return s->launched;
}

bool gpunetio_failed(const GpunetioSession* s) {
    return reinterpret_cast<const volatile uint32_t*>(s->exit_flag_cpu)[0] == 2;
}

bool gpunetio_stop(GpunetioSession* s) {
    if (!s->launched) return false;
    const bool failed = gpunetio_failed(s);
    reinterpret_cast<volatile uint32_t*>(s->exit_flag_cpu)[0] = 1;
    const cudaError_t err = cudaStreamSynchronize(s->stream);
    if (err != cudaSuccess)
        std::fprintf(stderr, "[vf] receive kernel failed: %s\n", cudaGetErrorString(err));
    if (failed) std::fprintf(stderr, "[vf] a receive queue reported an error; the run stopped\n");
    return err == cudaSuccess && !failed;
}

void gpunetio_close(GpunetioSession* s) { destroy(s); }

std::string gpunetio_describe(const GpunetioSession* s) {
    std::string feeds;
    for (const Endpoint& e : s->feeds) feeds += (feeds.empty() ? "" : ", ") + endpoint_string(e);
    return "GPUNetIO: NIC " + s->nic_pci + (s->iface.empty() ? "" : " (" + s->iface + ")") +
           " -> VRAM of GPU " + s->gpu_pci + ", " + std::to_string(s->queues.size()) +
           " queue(s); feeds " + feeds;
}

}  // namespace vf

//
// src/field/shm.cpp - the snapshot's shared memory, mapped into the device.
//
// The file is created on tmpfs (shm_open for a plain name, open() for a path),
// mmapped MAP_SHARED, and the mapping is pinned and mapped into the device
// with cudaHostRegister(cudaHostRegisterMapped). The surface kernel then
// writes the snapshot over PCIe straight into the pages Python has mapped:
// no staging buffer, no host-side copy.
//

#include "vf/snapshot.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

namespace vf {

bool snapshot_open(const char* name, const FieldParams& p, Snapshot& out) {
    out = Snapshot{};
    const bool is_path = std::strchr(name, '/') != nullptr;
    if (is_path) std::snprintf(out.path, sizeof out.path, "%s", name);
    else std::snprintf(out.path, sizeof out.path, "/dev/shm/%s", name);

    int fd = -1;
    if (is_path) {
        fd = ::open(name, O_RDWR | O_CREAT, 0644);
    } else {
        char shm_name[256];
        std::snprintf(shm_name, sizeof shm_name, "/%s", name);
        fd = ::shm_open(shm_name, O_RDWR | O_CREAT, 0644);
    }
    if (fd < 0) {
        std::fprintf(stderr, "[vf] cannot create the snapshot at %s: %s\n", out.path,
                     std::strerror(errno));
        return false;
    }
    if (::ftruncate(fd, static_cast<off_t>(kSnapBytes)) != 0) {
        std::fprintf(stderr, "[vf] cannot size %s to %llu bytes: %s\n", out.path,
                     static_cast<unsigned long long>(kSnapBytes), std::strerror(errno));
        ::close(fd);
        return false;
    }
    void* mem = ::mmap(nullptr, kSnapBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mem == MAP_FAILED) {
        std::fprintf(stderr, "[vf] mmap %s: %s\n", out.path, std::strerror(errno));
        return false;
    }
    out.host = mem;
    out.bytes = kSnapBytes;

    // The static part of the header, written once by the host. Everything the
    // GPU writes later starts zeroed, with the seqlock even.
    auto* h = static_cast<SnapHeader*>(mem);
    std::memset(h, 0, sizeof *h);
    std::memcpy(h->magic, kSnapMagic, sizeof h->magic);
    h->layout_version = kSnapLayoutVersion;
    h->header_bytes = sizeof(SnapHeader);
    h->max_expiries = kMaxExpiries;
    h->max_pairs = kMaxPairs;
    h->expiry_rec_bytes = sizeof(ExpirySnap);
    h->pair_rec_bytes = sizeof(PairSnap);
    h->expiry_offset = kSnapExpiryOffset;
    h->pair_offset = kSnapPairOffset;
    h->rate = p.rate;
    h->trade_days = p.trade_days;
    h->n_underlyings = p.n_underlyings;
    std::memcpy(h->underlyings, p.underlyings, sizeof h->underlyings);
    h->state = kSnapRunning;

    cudaError_t e = cudaHostRegister(mem, kSnapBytes, cudaHostRegisterMapped);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[vf] cudaHostRegister(%s): %s\n", out.path, cudaGetErrorString(e));
        ::munmap(mem, kSnapBytes);
        out.host = nullptr;
        return false;
    }
    void* dev = nullptr;
    e = cudaHostGetDevicePointer(&dev, mem, 0);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[vf] cudaHostGetDevicePointer: %s\n", cudaGetErrorString(e));
        cudaHostUnregister(mem);
        ::munmap(mem, kSnapBytes);
        out.host = nullptr;
        return false;
    }
    out.dev = static_cast<SnapHeader*>(dev);
    return true;
}

void snapshot_close(Snapshot& s) {
    if (!s.host) return;
    cudaHostUnregister(s.host);
    ::munmap(s.host, s.bytes);
    s.host = nullptr;
    s.dev = nullptr;
}

bool snapshot_header(const Snapshot& s, SnapHeader& out) {
    const auto* h = static_cast<const SnapHeader*>(s.host);
    const volatile uint64_t* lock = &h->seqlock;
    for (int attempt = 0; attempt < 1000; ++attempt) {
        const uint64_t v1 = *lock;
        if (v1 & 1u) continue;
        std::atomic_thread_fence(std::memory_order_acquire);
        std::memcpy(&out, h, sizeof out);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (*lock == v1) return true;
    }
    return false;
}

}  // namespace vf

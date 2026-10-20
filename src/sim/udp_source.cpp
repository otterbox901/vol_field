//
// src/sim/udp_source.cpp - MoldUDP64 datagrams from UDP sockets.
//
// The kernel stack receives, recvmmsg() hands over up to 64 datagrams per
// call, and each one goes into a ring slot behind a synthetic Eth/IPv4/UDP
// header, so from there on it is indistinguishable from a captured frame.
//

#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

#include "vf/net.hpp"
#include "vf/run.hpp"

#include "source.hpp"

namespace vf {
namespace sim {
namespace {

constexpr uint32_t kBatch = 64;
constexpr uint32_t kMaxPayload = kMaxFrameBytes - kSyntheticHeaderBytes;
constexpr int kRcvBuf = 16 << 20;

}  // namespace

UdpSource::~UdpSource() {
    stop();
    for (int fd : fds_) ::close(fd);
}

bool UdpSource::open(const std::vector<Endpoint>& feeds, uint32_t iface_ip, uint32_t n_queues,
                     std::string& err) {
    feeds_ = feeds;
    n_queues_ = n_queues;
    for (const Endpoint& e : feeds) {
        const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd < 0) {
            err = std::string("socket: ") + std::strerror(errno);
            return false;
        }
        fds_.push_back(fd);
        const int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &kRcvBuf, sizeof kRcvBuf);

        const bool mcast = (e.ip >> 28) == 0xE;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(e.port);
        // A multicast socket binds the group, so it only sees that group's
        // datagrams even when several groups share a port.
        a.sin_addr.s_addr = htonl(mcast ? e.ip : INADDR_ANY);
        if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
            err = "bind " + endpoint_string(e) + ": " + std::strerror(errno);
            return false;
        }
        if (mcast) {
            ip_mreq m{};
            m.imr_multiaddr.s_addr = htonl(e.ip);
            m.imr_interface.s_addr = htonl(iface_ip);
            if (setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &m, sizeof m) != 0) {
                err = "join " + endpoint_string(e) + ": " + std::strerror(errno);
                return false;
            }
        }
    }
    return true;
}

void UdpSource::start(Rings& rings) {
    rings_ = &rings;
    thread_ = std::thread([this] { run(); });
}

void UdpSource::stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

void UdpSource::run() {
    Rings& rings = *rings_;
    std::vector<pollfd> pfds;
    for (int fd : fds_) pfds.push_back(pollfd{fd, POLLIN, 0});

    std::vector<uint8_t> bufs(size_t{kBatch} * kMaxPayload);
    mmsghdr msgs[kBatch];
    iovec iov[kBatch];
    sockaddr_in from[kBatch];

    while (!stop_.load()) {
        if (::poll(pfds.data(), pfds.size(), 10) <= 0) continue;
        for (size_t s = 0; s < pfds.size(); ++s) {
            if (!(pfds[s].revents & POLLIN)) continue;
            for (uint32_t k = 0; k < kBatch; ++k) {
                iov[k] = iovec{bufs.data() + size_t{k} * kMaxPayload, kMaxPayload};
                msgs[k] = mmsghdr{};
                msgs[k].msg_hdr.msg_iov = &iov[k];
                msgs[k].msg_hdr.msg_iovlen = 1;
                msgs[k].msg_hdr.msg_name = &from[k];
                msgs[k].msg_hdr.msg_namelen = sizeof from[k];
            }
            const int n = ::recvmmsg(pfds[s].fd, msgs, kBatch, MSG_DONTWAIT, nullptr);
            if (n <= 0) continue;

            const Endpoint& feed = feeds_[s];
            const uint32_t q = queue_of(feed.ip, feed.port, n_queues_);
            for (int k = 0; k < n; ++k) {
                if (msgs[k].msg_hdr.msg_flags & MSG_TRUNC) {
                    oversize_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                const uint32_t len = msgs[k].msg_len;
                uint8_t* slot = rings.slot(q);
                if (!slot) return;
                uint8_t* f = slot + kSlotHeader;
                write_synthetic_header(f, ntohl(from[k].sin_addr.s_addr), ntohs(from[k].sin_port),
                                       feed.ip, feed.port, len);
                std::memcpy(f + kSyntheticHeaderBytes, iov[k].iov_base, len);
                rings.commit(q, kSyntheticHeaderBytes + len);
            }
            received_.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
            rings.flush(q);
        }
    }
    rings.flush_all();
}

std::string UdpSource::status() const {
    return " | received " + si(static_cast<double>(received_.load()));
}

std::string UdpSource::describe() const {
    std::string s = "UDP sockets (kernel stack, recvmmsg):";
    for (const Endpoint& e : feeds_) s += " " + endpoint_string(e);
    s += "; " + std::to_string(received_.load()) + " datagrams";
    if (oversize_.load()) s += ", " + std::to_string(oversize_.load()) + " too large for a slot";
    return s;
}

}  // namespace sim
}  // namespace vf

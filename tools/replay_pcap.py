#!/usr/bin/env python3
"""Replay a pcap's UDP packets: as plain datagrams, or as raw frames on an interface.

send-udp (no scapy, no root) sends each UDP payload from an ordinary socket,
to the frame's own destination or to --dst. That is the feed for
`vf_sim --udp` on another node (or the same one), and multicast works too:

  # node A: listen on the group bono_gen.py writes into the pcap
  vf_sim --udp --feed 233.54.12.111:18001 --underlying TEST --date 2026-10-20
  # node B: replay at 20 kpps, three laps (sequence numbers keep rising)
  tools/replay_pcap.py send-udp --pcap chain.pcap --pps 20000 --loop 3
  # unicast to one host instead of the group
  tools/replay_pcap.py send-udp --pcap chain.pcap --dst nodeA:18001 --realtime

send (scapy, root) writes whole frames onto an interface at layer 2, so they
reach a NIC's hardware steering exactly like exchange traffic - the way to
feed vf_gpunetio from a second host:

  sudo tools/replay_pcap.py send --pcap chain.pcap --iface ens1f0 --pps 10000

make (scapy) writes a generic UDP capture; for BONO use tools/bono_gen.py.

With --loop, each MoldUDP64 sequence number is moved up by one lap's span per
lap, so a receiver never sees a replayed (stale) message as a duplicate.
"""

import argparse
import socket
import struct
import sys
import time

MOLD_HEADER = 20
MOLD_SESSION = 10


def need_scapy():
    try:
        import scapy.all as scapy
    except ImportError:
        sys.exit("this subcommand needs scapy: pip install scapy (or dnf install python3-scapy); "
                 "send-udp does not")
    return scapy


# --- pure-python pcap ---------------------------------------------------------

def read_pcap(path):
    """(ts_ns, frame bytes) for every record of a classic pcap file."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 24:
        sys.exit(f"{path}: too short for a pcap")
    magic = data[:4]
    order, nano = {
        b"\xd4\xc3\xb2\xa1": ("<", False), b"\xa1\xb2\xc3\xd4": (">", False),
        b"\x4d\x3c\xb2\xa1": ("<", True), b"\xa1\xb2\x3c\x4d": (">", True),
    }.get(magic, (None, None))
    if order is None:
        hint = " (pcapng: convert with editcap -F pcap)" if magic == b"\x0a\x0d\x0d\x0a" else ""
        sys.exit(f"{path}: not a classic pcap{hint}")
    linktype = struct.unpack_from(order + "I", data, 20)[0]
    if linktype != 1:
        sys.exit(f"{path}: link type {linktype}, only Ethernet (1) is supported")
    rec = struct.Struct(order + "IIII")
    off, frames = 24, []
    while off + rec.size <= len(data):
        sec, frac, incl, _orig = rec.unpack_from(data, off)
        off += rec.size
        frames.append((sec * 1_000_000_000 + (frac if nano else frac * 1000), data[off:off + incl]))
        off += incl
    return frames


def udp_of(frame):
    """(dst_ip, dst_port, payload) of an Ethernet/[VLAN]/IPv4/UDP frame, else None."""
    if len(frame) < 14:
        return None
    eth_type, off = struct.unpack_from(">H", frame, 12)[0], 14
    if eth_type == 0x8100 and len(frame) >= 18:
        eth_type, off = struct.unpack_from(">H", frame, 16)[0], 18
    if eth_type != 0x0800 or len(frame) < off + 20:
        return None
    vihl, _tos, total, _id, frag, _ttl, proto = struct.unpack_from(">BBHHHBB", frame, off)
    ihl = (vihl & 0x0F) * 4
    if vihl >> 4 != 4 or proto != 17 or frag & 0x3FFF or ihl < 20:
        return None
    udp = off + ihl
    if len(frame) < udp + 8:
        return None
    dport, ulen = struct.unpack_from(">HH", frame, udp + 2)
    end = min(udp + ulen, off + total, len(frame))
    if ulen < 8 or end < udp + 8:
        return None
    dst_ip = socket.inet_ntoa(frame[off + 16:off + 20])
    return dst_ip, dport, frame[udp + 8:end]


def mold_span(payloads):
    """Sequence numbers one lap covers: max(seq + count) - min(seq) over the
    MoldUDP64 packets (heartbeats and end-of-session carry no messages)."""
    lo, hi = None, 0
    for p in payloads:
        if len(p) < MOLD_HEADER:
            continue
        seq, count = struct.unpack_from(">QH", p, MOLD_SESSION)
        n = 0 if count == 0xFFFF else count
        lo = seq if lo is None else min(lo, seq)
        hi = max(hi, seq + n)
    return hi - lo if lo is not None and hi > lo else 0


def with_seq(payload, add):
    if not add or len(payload) < MOLD_HEADER:
        return payload
    seq = struct.unpack_from(">Q", payload, MOLD_SESSION)[0]
    return payload[:MOLD_SESSION] + struct.pack(">Q", seq + add) + payload[MOLD_SESSION + 8:]


def parse_dst(s):
    host, _, port = s.rpartition(":")
    if not port.isdigit():
        sys.exit(f"--dst {s}: expected HOST:PORT")
    return socket.gethostbyname(host or "127.0.0.1"), int(port)


def pace(args, start, i, ts_rel_ns):
    """Sleep until packet i is due. ts_rel_ns: its capture time since the first packet."""
    if args.realtime:
        due = start + ts_rel_ns / 1e9 / args.speed
    elif args.pps:
        due = start + i / args.pps
    else:
        return
    delay = due - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


def cmd_send_udp(args):
    pkts = []
    for ts, frame in read_pcap(args.pcap):
        u = udp_of(frame)
        if u is not None:
            pkts.append((ts, u))
        if args.count and len(pkts) >= args.count:
            break
    if not pkts:
        sys.exit(f"no IPv4/UDP packets in {args.pcap}")
    span = mold_span(p for _, (_, _, p) in pkts)
    dst = parse_dst(args.dst) if args.dst else None
    t0 = pkts[0][0]
    lap_ns = pkts[-1][0] - t0 + 1_000_000  # + 1 ms between laps, as vf_sim --loop does

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, args.ttl)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    if args.iface:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(args.iface))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)

    groups = sorted({f"{ip}:{port}" for _, (ip, port, _) in pkts})
    where = f"{dst[0]}:{dst[1]}" if dst else ", ".join(groups)
    print(f"[send-udp] {len(pkts)} datagrams per lap to {where}; "
          f"{'mold span %d per lap' % span if span else 'no MoldUDP64 sequence found'}", flush=True)

    sent = nbytes = 0
    start = time.perf_counter()
    try:
        lap = 0
        while args.loop == 0 or lap < args.loop:
            for i, (ts, (ip, port, payload)) in enumerate(pkts):
                pace(args, start, lap * len(pkts) + i, lap * lap_ns + ts - t0)
                body = with_seq(payload, lap * span)
                while True:
                    try:
                        sock.sendto(body, dst or (ip, port))
                        break
                    except BlockingIOError:
                        time.sleep(0.0001)
                    except OSError as e:
                        if e.errno != 105:  # ENOBUFS: the qdisc is full, wait for it
                            raise
                        time.sleep(0.0001)
                sent += 1
                nbytes += len(body)
            lap += 1
    except KeyboardInterrupt:
        pass
    elapsed = time.perf_counter() - start
    print(f"[send-udp] sent {sent} datagrams ({nbytes} payload bytes, {lap} lap(s)) in "
          f"{elapsed:.2f} s = {sent / max(elapsed, 1e-9):,.0f} pps")


def cmd_make(args):
    scapy = need_scapy()
    pkts = []
    for i in range(args.count):
        payload = i.to_bytes(4, "little") + bytes(max(0, args.size - 4))
        pkts.append(
            scapy.Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")
            / scapy.IP(src="10.99.0.2", dst="10.99.0.1")
            / scapy.UDP(sport=40000, dport=args.port)
            / scapy.Raw(payload)
        )
    scapy.wrpcap(args.out, pkts)
    print(f"wrote {len(pkts)} UDP packets ({args.size} B payload, dport {args.port}) to {args.out}")


def cmd_send(args):
    scapy = need_scapy()
    src_mac = scapy.get_if_hwaddr(args.iface)
    frames = []
    for pkt in scapy.rdpcap(args.pcap):
        if scapy.IP not in pkt or scapy.UDP not in pkt:
            continue
        ip = pkt[scapy.IP].copy()
        if args.dst_ip:
            ip.dst = args.dst_ip
        if args.dst_port:
            ip[scapy.UDP].dport = args.dst_port
        del ip.chksum, ip.len
        del ip[scapy.UDP].chksum, ip[scapy.UDP].len
        dst_mac = args.dst_mac or (pkt[scapy.Ether].dst if scapy.Ether in pkt else "ff:ff:ff:ff:ff:ff")
        frames.append(scapy.Ether(src=src_mac, dst=dst_mac) / ip)
        if args.count and len(frames) >= args.count:
            break

    if not frames:
        sys.exit(f"no IPv4/UDP packets in {args.pcap}")

    payload_bytes = sum(len(f[scapy.UDP].payload) for f in frames)
    raw = [bytes(f) for f in frames]
    interval = 1.0 / args.pps if args.pps else 0.0

    # One socket for the whole replay: sendp() opens and binds a new one per
    # packet, which caps it at a few dozen packets per second.
    sock = scapy.conf.L2socket(iface=args.iface)
    try:
        start = time.perf_counter()
        for i, frame in enumerate(raw):
            if interval:
                delay = start + i * interval - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            sock.send(frame)
        elapsed = time.perf_counter() - start
    finally:
        sock.close()
    print(f"sent {len(frames)} UDP packets ({payload_bytes} payload bytes) on {args.iface} "
          f"in {elapsed:.2f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)

    su = sub.add_parser("send-udp", help="send each UDP payload from a plain socket (no root)")
    su.add_argument("--pcap", required=True)
    su.add_argument("--dst", help="HOST:PORT for every datagram (default: each frame's own)")
    su.add_argument("--pps", type=float, default=0, help="pace to R datagrams/s, 0 = unpaced")
    su.add_argument("--realtime", action="store_true", help="pace by capture timestamps")
    su.add_argument("--speed", type=float, default=1.0, help="with --realtime: time scale (default 1)")
    su.add_argument("--loop", type=int, default=1, help="laps, 0 = until Ctrl-C (default 1)")
    su.add_argument("--count", type=int, default=0, help="first N datagrams only, 0 = all")
    su.add_argument("--ttl", type=int, default=4, help="multicast TTL (default 4)")
    su.add_argument("--iface", metavar="IP", help="local address multicast leaves from")
    su.set_defaults(func=cmd_send_udp)

    sd = sub.add_parser("send", help="replay a pcap's UDP frames onto an interface (scapy, root)")
    sd.add_argument("--pcap", required=True)
    sd.add_argument("--iface", required=True, help="interface to transmit on")
    sd.add_argument("--dst-mac", help="rewrite destination MAC (the receiving interface's)")
    sd.add_argument("--dst-ip", help="rewrite destination IPv4 address")
    sd.add_argument("--dst-port", type=int, help="rewrite UDP destination port")
    sd.add_argument("--count", type=int, default=0, help="stop after N packets, 0 = all")
    sd.add_argument("--pps", type=float, default=0, help="pace to R packets/s, 0 = as fast as possible")
    sd.set_defaults(func=cmd_send)

    mk = sub.add_parser("make", help="write a generic UDP pcap (scapy)")
    mk.add_argument("--out", default="udp.pcap")
    mk.add_argument("--count", type=int, default=10000)
    mk.add_argument("--size", type=int, default=512, help="UDP payload bytes (default 512)")
    mk.add_argument("--port", type=int, default=9000, help="UDP destination port (default 9000)")
    mk.set_defaults(func=cmd_make)

    args = ap.parse_args()
    if getattr(args, "realtime", False) and args.pps:
        sys.exit("--realtime and --pps are exclusive")
    args.func(args)


if __name__ == "__main__":
    main()

"""Read vol_field surface snapshots - the numpy mirror of include/vf/snapshot.hpp.

The GPU writes the snapshot into shared memory under a seqlock; a reader
copies it between two equal, even seqlock reads. Sources:

  ShmSource(name)        /dev/shm/NAME, or any path (same machine as the run)
  TcpSource(host, port)  compact snapshots from tools/snapshot_relay.py

Both return Snapshot objects (or None while nothing stable is available).
numpy only.
"""

import mmap
import os
import socket
import struct
import time

import numpy as np

MAGIC = b"VOLFIELD"
LAYOUT_VERSION = 1
STATE_FINAL = 1

COUNTERS = ["frames", "bytes", "not_feed", "heartbeats", "end_of_session", "messages",
            "malformed", "directory", "admin", "quotes", "stale", "unmapped", "iv_solved",
            "iv_failed", "trades", "max_seq"]

HEADER = np.dtype([
    ("magic", "S8"), ("layout_version", "<u4"), ("header_bytes", "<u4"),
    ("seqlock", "<u8"), ("snapshot_id", "<u8"), ("host_ns", "<u8"),
    ("state", "<u4"), ("feed_sec", "<u4"), ("n_expiries", "<u4"), ("n_pairs", "<u4"),
    ("n_options", "<u4"), ("n_underlyings", "<u4"), ("max_expiries", "<u4"), ("max_pairs", "<u4"),
    ("expiry_rec_bytes", "<u4"), ("pair_rec_bytes", "<u4"),
    ("expiry_offset", "<u8"), ("pair_offset", "<u8"),
    ("rate", "<f8"), ("trade_days", "<i4"), ("overflow", "<u4"),
    ("underlyings", "S16", (8,)), ("counters", "<u8", (16,)), ("reserved", "u1", (144,)),
])

EXPIRY = np.dtype([
    ("und", "<u4"), ("yymmdd", "<u4"), ("days", "<i4"), ("n_pairs", "<u4"),
    ("T", "<f8"), ("forward", "<f8"), ("n_forward", "<u4"), ("pad", "<u4"), ("reserved", "<f8"),
])

PAIR = np.dtype([
    ("und", "<u4"), ("expiry", "<u4"), ("strike", "<f8"), ("forward", "<f8"), ("T", "<f8"),
    ("iv_call", "<f4"), ("iv_put", "<f4"), ("iv_otm", "<f4"), ("call_mid", "<f4"),
    ("put_mid", "<f4"), ("flags", "<u4"), ("call_version", "<u4"), ("put_version", "<u4"),
])

assert HEADER.itemsize == 512 and EXPIRY.itemsize == 48 and PAIR.itemsize == 64
assert HEADER.fields["seqlock"][1] == 16 and HEADER.fields["counters"][1] == 240

HAS_CALL, HAS_PUT, OTM_CALL, CALL_LIVE, PUT_LIVE = 1, 2, 4, 8, 16
SEQLOCK_OFFSET = 16


class Snapshot:
    """One consistent surface: header fields, expiry records, pair records."""

    def __init__(self, header, expiries, pairs):
        self.header = header
        self.expiries = expiries
        self.pairs = pairs

    @property
    def id(self):
        return int(self.header["snapshot_id"])

    @property
    def feed_sec(self):
        return int(self.header["feed_sec"])

    @property
    def host_ns(self):
        return int(self.header["host_ns"])

    @property
    def final(self):
        return int(self.header["state"]) == STATE_FINAL

    @property
    def underlyings(self):
        n = int(self.header["n_underlyings"])
        return [u.decode("ascii").strip() for u in self.header["underlyings"][:n]]

    @property
    def counters(self):
        return dict(zip(COUNTERS, (int(v) for v in self.header["counters"])))

    def feed_clock(self):
        s = self.feed_sec
        return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"

    def surface(self, underlying=None):
        """OTM points: pair records with a forward and a finite iv_otm, plus
        each point's expiry record. Filtered to one underlying if given."""
        p = self.pairs
        keep = np.isfinite(p["iv_otm"]) & (p["forward"] > 0) & (p["und"] < len(self.underlyings))
        if underlying is not None:
            names = self.underlyings
            if underlying not in names:
                return p[:0], self.expiries[:0]
            keep &= p["und"] == names.index(underlying)
        p = p[keep]
        return p, self.expiries[p["expiry"]]

    def compact(self):
        """Header + current records only: what the relay sends."""
        return self.header.tobytes() + self.expiries.tobytes() + self.pairs.tobytes()


def parse_compact(buf):
    header = np.frombuffer(buf, HEADER, count=1)[0].copy()
    _check(header)
    ne, npairs = int(header["n_expiries"]), int(header["n_pairs"])
    off = HEADER.itemsize
    expiries = np.frombuffer(buf, EXPIRY, count=ne, offset=off).copy()
    pairs = np.frombuffer(buf, PAIR, count=npairs, offset=off + ne * EXPIRY.itemsize).copy()
    return Snapshot(header, expiries, pairs)


def _check(header):
    if bytes(header["magic"]) != MAGIC:
        raise ValueError("not a vol_field snapshot (bad magic)")
    if int(header["layout_version"]) != LAYOUT_VERSION:
        raise ValueError(f"snapshot layout {int(header['layout_version'])}, this reader knows "
                         f"{LAYOUT_VERSION}: rebuild or update tools/")


def shm_path(name):
    return name if "/" in name else os.path.join("/dev/shm", name)


class ShmSource:
    """Seqlock-stable copies out of the shared memory the GPU writes."""

    def __init__(self, name="vol_field", wait=True):
        self.path = shm_path(name)
        while not os.path.exists(self.path) or os.path.getsize(self.path) < HEADER.itemsize:
            if not wait:
                raise FileNotFoundError(self.path)
            time.sleep(0.2)
        f = open(self.path, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()

    def _seq(self):
        return struct.unpack_from("<Q", self.mm, SEQLOCK_OFFSET)[0]

    def read(self, tries=200):
        """A Snapshot, or None if the writer kept it busy or nothing is written yet."""
        for _ in range(tries):
            s1 = self._seq()
            if s1 & 1:
                time.sleep(0.0005)
                continue
            header = np.frombuffer(self.mm, HEADER, count=1)[0].copy()
            ne = min(int(header["n_expiries"]), int(header["max_expiries"]))
            npairs = min(int(header["n_pairs"]), int(header["max_pairs"]))
            eo, po = int(header["expiry_offset"]), int(header["pair_offset"])
            expiries = np.frombuffer(self.mm, EXPIRY, count=ne, offset=eo).copy()
            pairs = np.frombuffer(self.mm, PAIR, count=npairs, offset=po).copy()
            if self._seq() == s1:
                if bytes(header["magic"]) != MAGIC or int(header["snapshot_id"]) == 0:
                    return None
                _check(header)
                return Snapshot(header, expiries, pairs)
        return None


class TcpSource:
    """Latest snapshot from a relay: frames of [u32 length][compact snapshot]."""

    def __init__(self, host, port):
        self.addr = (host, port)
        self.sock = None
        self.buf = b""
        self.latest = None

    def _connect(self):
        try:
            self.sock = socket.create_connection(self.addr, timeout=2.0)
            self.sock.setblocking(False)
            self.buf = b""
        except OSError:
            self.sock = None

    def read(self):
        if self.sock is None:
            self._connect()
            if self.sock is None:
                return self.latest
        try:
            while True:
                chunk = self.sock.recv(1 << 20)
                if not chunk:
                    self.sock.close()
                    self.sock = None
                    break
                self.buf += chunk
        except BlockingIOError:
            pass
        except OSError:
            self.sock = None
        # Keep only the newest complete frame.
        while len(self.buf) >= 4:
            n = struct.unpack_from(">I", self.buf)[0]
            if len(self.buf) < 4 + n:
                break
            self.latest = parse_compact(self.buf[4:4 + n])
            self.buf = self.buf[4 + n:]
        return self.latest


def open_source(shm=None, tcp=None):
    if tcp:
        host, _, port = tcp.rpartition(":")
        return TcpSource(host or "localhost", int(port))
    return ShmSource(shm or "vol_field")

#!/usr/bin/env python3
"""Write a synthetic BONO feed as a pcap, plus the truth behind it.

One underlying's option chain, start of day to a few seconds of quoting,
framed exactly as Nasdaq sends it: Ethernet / IPv4 / UDP (multicast) /
MoldUDP64 / BONO v3.3 messages. Pure struct packing - no scapy.

  pre-open   T 07:00:00, S 'O', the Options Directory spin (D for every
             option), the Trading Action spin (H 'T'), S 'S'
  open       T 09:30:00, S 'Q', Security Open (O 'Y') for every option
  quoting    every step (--steps per second): spot takes a random-walk step,
             the volatility level drifts, every option is re-priced with
             Black-76 and quoted at theo +- a spread, rounded out to the MPV
             (penny pilot: $0.01 under $3, $0.05 above). An option whose
             BBO changed gets q/Q when both sides moved, b/a/B/A when one did.
             T every feed second, a MoldUDP64 heartbeat each second, and an
             end-of-session packet at the end.

The truth file has one row per option per step: the volatility and forward
the quotes were priced from, and the BBO that resulted. tools/check_surface.py
compares a vol_field snapshot against it.

  tools/bono_gen.py --out chain.pcap            # TEST, 8 expiries x 41 strikes, 15 s
  tools/bono_gen.py --out big.pcap --expiries 11 --strikes 101 --seconds 60 --dup
"""

import argparse
import datetime as dt
import math
import struct
import sys

import numpy as np

SECONDS_PER_YEAR = 365.0 * 86400.0
EXPIRY_CLOSE = 16 * 3600
EXPIRY_DAYS = [2, 9, 16, 30, 58, 93, 184, 366, 548, 731, 1096]
NICE_STEPS = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
MTU_PAYLOAD = 1400

_erfc = np.vectorize(math.erfc, otypes=[float])


def norm_cdf(x):
    return 0.5 * _erfc(-x / math.sqrt(2.0))


def black76(F, K, T, sigma, df, call):
    sd = sigma * np.sqrt(T)
    d1 = np.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    c = F * norm_cdf(d1) - K * norm_cdf(d2)
    p = K * norm_cdf(-d2) - F * norm_cdf(-d1)
    return df * np.where(call, c, p)


# --------------------------------------------------------------------------
# BONO messages (spec v3.3; big-endian, prices 1e-4 or pennies, space-padded)
# --------------------------------------------------------------------------

def pad(s, n):
    return s.encode("ascii")[:n].ljust(n, b" ")


def msg_T(sec):
    return struct.pack(">cI", b"T", sec)


def msg_S(ns, code):
    return struct.pack(">cIcBB", b"S", ns, code.encode(), 3, 0)


def msg_D(ns, oid, root, yy, mm, dd, strike_e4, cp, underlying, tradable=b"Y"):
    return struct.pack(">cII6sBBBIcB13sccc", b"D", ns, oid, pad(root, 6), yy, mm, dd, strike_e4,
                       cp, 1, pad(underlying, 13), b"N", tradable, b"P")


def msg_H(ns, oid, state):
    return struct.pack(">cIIc", b"H", ns, oid, state)


def msg_O(ns, oid, state):
    return struct.pack(">cIIc", b"O", ns, oid, state)


def msg_quote(ns, oid, bid, bsz, ask, asz, long_form):
    """Both sides; prices in 1e-4 dollars (always whole cents here)."""
    if long_form:
        return struct.pack(">cIIcIIII", b"Q", ns, oid, b" ", bid, bsz, ask, asz)
    return struct.pack(">cIIcHHHH", b"q", ns, oid, b" ", bid // 100, bsz, ask // 100, asz)


def msg_side(ns, oid, side, price, size, long_form):
    if long_form:
        return struct.pack(">cIIcII", (b"B" if side == "b" else b"A"), ns, oid, b" ", price, size)
    return struct.pack(">cIIcHH", side.encode(), ns, oid, b" ", price // 100, size)


def short_ok(*vals):
    prices, sizes = vals[0::2], vals[1::2]
    return all(p // 100 <= 0xFFFF for p in prices) and all(s <= 0xFFFF for s in sizes)


# --------------------------------------------------------------------------
# Framing: MoldUDP64 in UDP in IPv4 in Ethernet, into a pcap
# --------------------------------------------------------------------------

def ip_checksum(hdr):
    s = sum(struct.unpack(">10H", hdr))
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return (~s) & 0xFFFF


def ip_to_int(a):
    return struct.unpack(">I", bytes(int(x) for x in a.split(".")))[0]


class PcapWriter:
    def __init__(self, path, src, group, port):
        self.f = open(path, "wb")
        self.f.write(struct.pack("<IHHiIII", 0xA1B23C4D, 2, 4, 0, 0, 65535, 1))  # ns resolution
        self.src = ip_to_int(src)
        self.port = port
        self.frames = 0

    def frame(self, ts_ns, payload, group):
        dst = ip_to_int(group)
        mac = bytes([0x01, 0x00, 0x5E, (dst >> 16) & 0x7F, (dst >> 8) & 0xFF, dst & 0xFF])
        eth = mac + bytes([0x02, 0, 0, 0, 0, 1]) + b"\x08\x00"
        ip = bytearray(struct.pack(">BBHHHBBHII", 0x45, 0, 28 + len(payload), self.frames & 0xFFFF,
                                   0x4000, 16, 17, 0, self.src, dst))
        ip[10:12] = struct.pack(">H", ip_checksum(bytes(ip)))
        udp = struct.pack(">HHHH", 40000, self.port, 8 + len(payload), 0)
        data = eth + bytes(ip) + udp + payload
        self.f.write(struct.pack("<IIII", ts_ns // 1_000_000_000, ts_ns % 1_000_000_000, len(data),
                                 len(data)))
        self.f.write(data)
        self.frames += 1

    def close(self):
        self.f.close()


class MoldStream:
    """Packs messages into MoldUDP64 packets and numbers them."""

    def __init__(self, writer, session, groups, mtu):
        self.w = writer
        self.session = pad(session, 10)
        self.groups = groups  # A feed, then B feed if duplicating
        self.mtu = mtu
        self.seq = 1          # next message sequence number
        self.msgs = []
        self.size = 20
        self.ts = 0
        self.packets = 0
        self.messages = 0
        self.counts = {}

    def add(self, ts_ns, msg):
        if self.msgs and self.size + 2 + len(msg) > self.mtu:
            self.flush()
        if not self.msgs:
            self.ts = ts_ns
        self.msgs.append(msg)
        self.size += 2 + len(msg)
        key = chr(msg[0])
        self.counts[key] = self.counts.get(key, 0) + 1

    def flush(self):
        if not self.msgs:
            return
        body = b"".join(struct.pack(">H", len(m)) + m for m in self.msgs)
        self._send(self.ts, struct.pack(">10sQH", self.session, self.seq, len(self.msgs)) + body)
        self.seq += len(self.msgs)
        self.messages += len(self.msgs)
        self.msgs, self.size = [], 20

    def control(self, ts_ns, count):
        """Heartbeat (count 0) or end of session (0xFFFF): no messages."""
        self.flush()
        self._send(ts_ns, struct.pack(">10sQH", self.session, self.seq, count))

    def _send(self, ts_ns, payload):
        for g in self.groups:
            self.w.frame(ts_ns, payload, g)
        self.packets += 1

    @property
    def last_seq(self):
        return self.seq - 1


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------

def nice_step(raw):
    return min(NICE_STEPS, key=lambda s: abs(math.log(s / raw)))


def to_ticks(price, up):
    """Round a price out to the penny-pilot MPV, in 1e-4 dollars."""
    tick = np.where(price < 3.0, 100, 500)
    x = price * 1e4 / tick
    return (np.ceil(x - 1e-9) if up else np.floor(x + 1e-9)).astype(np.int64) * tick


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="pcap to write")
    ap.add_argument("--truth", help="truth CSV (default: OUT with .truth.csv)")
    ap.add_argument("--underlying", default="TEST")
    ap.add_argument("--date", default=None, help="trade date YYYY-MM-DD (default today)")
    ap.add_argument("--rate", type=float, default=0.04)
    ap.add_argument("--spot", type=float, default=100.0)
    ap.add_argument("--expiries", type=int, default=8, help=f"1..{len(EXPIRY_DAYS)} (default 8)")
    ap.add_argument("--strikes", type=int, default=41, help="strikes per expiry, odd (default 41)")
    ap.add_argument("--seconds", type=float, default=15, help="feed seconds of quoting (default 15)")
    ap.add_argument("--steps", type=int, default=10, help="re-quotes per feed second (default 10)")
    ap.add_argument("--spot-vol", type=float, default=0.002, help="spot random walk, per sqrt(second)")
    ap.add_argument("--vol-vol", type=float, default=0.004, help="vol level random walk, per sqrt(second)")
    ap.add_argument("--long-frac", type=float, default=0.2, help="long-form share when short fits")
    ap.add_argument("--dup", action="store_true", help="A+B feeds: every packet twice (group, group+1)")
    ap.add_argument("--halt", action="store_true", help="halt one option for the middle third")
    ap.add_argument("--group", default="233.54.12.111")
    ap.add_argument("--port", type=int, default=18001)
    ap.add_argument("--src", default="10.0.0.1")
    ap.add_argument("--mtu", type=int, default=MTU_PAYLOAD, help="max MoldUDP64 payload bytes")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not 1 <= args.expiries <= len(EXPIRY_DAYS):
        sys.exit(f"--expiries must be 1..{len(EXPIRY_DAYS)}")
    trade = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    truth_path = args.truth or (args.out[:-5] if args.out.endswith(".pcap") else args.out) + ".truth.csv"
    rng = np.random.default_rng(args.seed)

    # Options: per expiry, a strike grid sized to its own +-2.5 sd.
    half = args.strikes // 2
    S, lvl, r = args.spot, 0.0, args.rate
    rows = []  # (oid, expiry idx, K, call)
    expiries = []
    for e, days in enumerate(EXPIRY_DAYS[: args.expiries]):
        date = trade + dt.timedelta(days=days)
        T0 = (days * 86400 + EXPIRY_CLOSE - 34200) / SECONDS_PER_YEAR
        atm = 0.20 + 0.06 * math.exp(-T0 / 0.15)
        step = nice_step(2.5 * atm * math.sqrt(T0) * S / max(half, 1))
        centre = round(S / step) * step
        expiries.append((date, days, atm))
        for j in range(-half, half + 1):
            K = centre + j * step
            if K <= 0:
                continue
            for call in (True, False):
                rows.append((1000 + len(rows), e, K, call))
    n = len(rows)
    oid = np.array([x[0] for x in rows], dtype=np.int64)
    exp_idx = np.array([x[1] for x in rows])
    K = np.array([x[2] for x in rows])
    call = np.array([x[3] for x in rows])
    days = np.array([expiries[e][1] for e in exp_idx], dtype=float)
    atm0 = np.array([expiries[e][2] for e in exp_idx])
    strike_e4 = np.round(K * 1e4).astype(np.int64)

    def state(t):
        T = np.maximum(days * 86400 + EXPIRY_CLOSE - math.floor(t), 60.0) / SECONDS_PER_YEAR
        F = S * np.exp(r * T)
        a = np.clip(atm0 + lvl, 0.03, None)
        z = np.clip(np.log(K / F) / (a * np.sqrt(T)), -4, 4)
        sigma = np.clip(a * (1.0 - 0.10 * z + 0.035 * z * z), 0.03, 3.0)
        df = np.exp(-r * T)
        theo = black76(F, K, T, sigma, df, call)
        return T, F, sigma, theo

    w = PcapWriter(args.out, args.src, args.group, args.port)
    groups = [args.group]
    if args.dup:
        g = [int(x) for x in args.group.split(".")]
        groups.append(".".join(map(str, g[:3] + [g[3] + 1])))
    mold = MoldStream(w, args.underlying[:4] + "000001", groups, args.mtu)

    epoch = int(dt.datetime(trade.year, trade.month, trade.day, tzinfo=dt.timezone.utc).timestamp())
    start = 34200  # 09:30:00

    def ts(feed_sec):
        # pcap time: the pre-open is squeezed into the second before the open,
        # so a --realtime replay does not sit through two and a half hours.
        if feed_sec < start:
            return (epoch + start - 1) * 1_000_000_000 + int((feed_sec - 25200) * 1e3) % 1_000_000_000
        return int((epoch + feed_sec) * 1_000_000_000)

    # Pre-open: directory and trading-action spins.
    t = 25200.0
    mold.add(ts(t), msg_T(25200))
    mold.add(ts(t), msg_S(0, "O"))
    for i in range(n):
        d = expiries[exp_idx[i]][0]
        mold.add(ts(t), msg_D(1000 + i, int(oid[i]), args.underlying, d.year % 100, d.month, d.day,
                              int(strike_e4[i]), b"C" if call[i] else b"P", args.underlying))
    for i in range(n):
        mold.add(ts(t), msg_H(2000, int(oid[i]), b"T"))
    mold.add(ts(t), msg_S(3000, "S"))
    mold.flush()

    # Open.
    mold.add(ts(start), msg_T(start))
    mold.add(ts(start), msg_S(0, "Q"))
    for i in range(n):
        mold.add(ts(start), msg_O(1000, int(oid[i]), b"Y"))

    last = np.zeros((n, 4), dtype=np.int64)  # bid, bsz, ask, asz as last sent
    sizes = rng.integers(1, 300, size=(n, 2))
    halted = rng.integers(n) if args.halt else -1
    n_steps = int(round(args.seconds * args.steps)) + 1
    dt_s = 1.0 / args.steps
    cp = np.where(call, "C", "P")
    yymmdd = [expiries[e][0].strftime("%y%m%d") for e in exp_idx]
    truth = open(truth_path, "w")
    truth.write("step,time,seq_end,option_id,expiry,strike,cp,iv,F,T,bid,ask\n")

    for k in range(n_steps):
        t = start + k * dt_s
        sec = int(math.floor(t + 1e-9))
        ns = int(round((t - sec) * 1e9)) % 1_000_000_000
        if k > 0 and sec != int(math.floor(t - dt_s + 1e-9)):
            mold.control(ts(t), 0)  # heartbeat once a second
            mold.add(ts(t), msg_T(sec))
        if k > 0:
            S *= math.exp(args.spot_vol * math.sqrt(dt_s) * rng.standard_normal())
            lvl += -0.5 * lvl * dt_s + args.vol_vol * math.sqrt(dt_s) * rng.standard_normal()
        if args.halt and k == n_steps // 3:
            mold.add(ts(t), msg_H(ns, int(oid[halted]), b"H"))
        if args.halt and k == 2 * n_steps // 3:
            mold.add(ts(t), msg_H(ns, int(oid[halted]), b"T"))

        T, F, sigma, theo = state(t)
        hs = np.maximum(0.01, 0.02 * theo)
        bid = to_ticks(theo - hs, up=False)
        ask = to_ticks(theo + hs, up=True)
        bid = np.where(bid <= 0, 0, bid)
        resize = rng.random((n, 2)) < 0.1
        sizes = np.where(resize, rng.integers(1, 300, size=(n, 2)), sizes)
        bsz = np.where(bid > 0, sizes[:, 0], 0)
        new = np.stack([bid, bsz, ask, sizes[:, 1]], axis=1)
        moved_b = (new[:, 0:2] != last[:, 0:2]).any(axis=1)
        moved_a = (new[:, 2:4] != last[:, 2:4]).any(axis=1)
        long_draw = rng.random(n) < args.long_frac

        for i in np.nonzero(moved_b | moved_a)[0]:
            if i == halted and n_steps // 3 <= k < 2 * n_steps // 3:
                continue  # no quotes while halted
            b, bs, a, az = (int(v) for v in new[i])
            o = int(oid[i])
            if moved_b[i] and moved_a[i]:
                lf = long_draw[i] or not short_ok(b, bs, a, az)
                mold.add(ts(t), msg_quote(ns, o, b, bs, a, az, lf))
            elif moved_b[i]:
                mold.add(ts(t), msg_side(ns, o, "b", b, bs, long_draw[i] or not short_ok(b, bs)))
            else:
                mold.add(ts(t), msg_side(ns, o, "a", a, az, long_draw[i] or not short_ok(a, az)))
            last[i] = new[i]
        mold.flush()

        seq_end = mold.last_seq
        lines = [
            f"{k},{t:.3f},{seq_end},{oid[i]},{yymmdd[i]},{K[i]:.4f},{cp[i]},{sigma[i]:.8f},"
            f"{F[i]:.8f},{T[i]:.10f},{last[i, 0] / 1e4:.2f},{last[i, 2] / 1e4:.2f}\n"
            for i in range(n)
        ]
        truth.writelines(lines)

    mold.control(ts(start + args.seconds + dt_s), 0xFFFF)
    truth.close()
    w.close()

    kinds = " ".join(f"{k}={v}" for k, v in sorted(mold.counts.items()))
    print(f"wrote {args.out}: {w.frames} frames, {mold.messages} messages ({kinds}), "
          f"seq 1..{mold.last_seq}{', A+B duplicated' if args.dup else ''}")
    print(f"      {n} options ({args.expiries} expiries x {args.strikes} strikes x C/P) on "
          f"{args.underlying}, trade date {trade}, {args.seconds:g} s of quotes at {args.steps}/s")
    print(f"wrote {truth_path}: {n_steps} steps x {n} options")


if __name__ == "__main__":
    main()

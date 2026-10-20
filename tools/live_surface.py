#!/usr/bin/env python3
"""Draw the live implied-vol surface at the display's refresh rate.

A Qt Quick window with one QtGraphs Surface3D, refilled once per vsync. Each
row is an expiry and each column one of its OTM points (ln K/F within
--max-moneyness); the wireframe shows where the points are. Between snapshots
the surface is interpolated a frame or two behind the newest one, so it moves
on every frame instead of jumping when a snapshot lands. Drag to rotate, wheel
to zoom.

  python3 tools/live_surface.py                       # /dev/shm/vol_field, same machine
  python3 tools/live_surface.py --tcp localhost:7700  # through snapshot_relay.py --hz 120 (+ ssh -L)

Motion is only as fine as the snapshots: run vf_sim with --snapshot-ms 8 on a
feed that re-quotes often (bono_gen.py --steps 60). Needs numpy and PySide6
with QtGraphs (Qt 6.8+).
"""

import argparse
import collections
import os
import signal
import sys
import time

import numpy as np

import vfsnap


def rows(snap, und, max_m):
    """(days, x, iv, points): the OTM points of each expiry with at least two,
    sorted by ln K/F, as lists; days ascending."""
    pts, exps = snap.surface(und)
    x = np.log(pts["strike"] / pts["forward"])
    keep = np.abs(x) <= max_m
    pts, exps, x = pts[keep], exps[keep], x[keep]
    days = exps["days"]
    iv = pts["iv_otm"].astype(float) * 100.0
    out_d, out_x, out_v = [], [], []
    for d in np.unique(days):
        m = days == d
        if m.sum() < 2:
            continue
        order = np.argsort(x[m])
        out_d.append(max(float(d), 0.5))  # a log axis: same-day expiries sit at half a day
        out_x.append(x[m][order])
        out_v.append(iv[m][order])
    return out_d, out_x, out_v, int(keep.sum())


def resample(xs, vs, cols):
    """A row as cols points along its own polyline, by point index: a row with
    exactly cols points comes back unchanged, shorter ones are spread along
    their segments. The surface needs every row the same length."""
    u = np.linspace(0.0, len(xs) - 1, cols)
    i = np.arange(len(xs))
    return np.interp(u, i, xs), np.interp(u, i, vs)


class Timeline:
    """Recent surfaces on the local monotonic clock, and the surface between
    them at any time. Snapshot times (host_ns) are mapped with the smallest
    arrival - host_ns offset seen lately, so it works whatever clock the
    writer used and across the relay."""

    def __init__(self):
        self.frames = collections.deque(maxlen=16)  # (t_local_ns, (days, x, iv))
        self.offsets = collections.deque(maxlen=240)
        self.periods = collections.deque(maxlen=60)
        self.last_host = None

    def reset(self):
        self.frames.clear()
        self.offsets.clear()
        self.periods.clear()
        self.last_host = None

    def add(self, host_ns, days, x, iv):
        now = time.monotonic_ns()
        self.offsets.append(now - host_ns)
        if self.last_host is not None and host_ns > self.last_host:
            self.periods.append(host_ns - self.last_host)
        self.last_host = host_ns
        self.frames.append((host_ns + min(self.offsets), (days, x, iv)))

    def period_ns(self):
        return float(np.median(self.periods)) if self.periods else 100e6

    def at(self, t):
        """(days, x, iv) at local time t: the two surfaces around t blended,
        or the nearest one when t is outside them or their shapes differ."""
        if not self.frames:
            return None
        f = list(self.frames)
        if t <= f[0][0]:
            return f[0][1]
        for (ta, a), (tb, b) in zip(f[:-1], f[1:]):
            if ta <= t < tb:
                if a[0] != b[0] or a[1].shape != b[1].shape:
                    return a
                w = (t - ta) / (tb - ta)
                return a[0], a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2])
        return f[-1][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shm", default="vol_field", help="/dev/shm name or path (default vol_field)")
    ap.add_argument("--tcp", help="HOST:PORT of a snapshot_relay.py instead of --shm")
    ap.add_argument("--underlying", help="default: the first one in the snapshot")
    ap.add_argument("--max-moneyness", type=float, default=0.6, help="|ln K/F| shown (default 0.6)")
    ap.add_argument("--no-interp", action="store_true", help="show each snapshot as it lands")
    args = ap.parse_args()

    from PySide6.QtCore import QObject, QUrl, Slot
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtGraphs import QSurfaceDataItem
    from PySide6.QtQuick import QQuickView

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl-C closes the window
    app = QGuiApplication(sys.argv)

    class Backend(QObject):
        def __init__(self):
            super().__init__()
            self.src = None
            self.root = self.proxy = None
            self.timeline = Timeline()
            self.cols = 2
            self.prev = None       # last snapshot taken in
            self.shown = None      # last surface handed to the graph
            self.rate = 0.0
            self.ivlim = None
            self.dlim = None
            self.xlim = None
            self.ticks = collections.deque(maxlen=121)

        def open(self):
            if args.tcp:
                host, _, port = args.tcp.rpartition(":")
                self.src = vfsnap.TcpSource(host or "localhost", int(port))
                return
            try:
                self.src = vfsnap.ShmSource(args.shm, wait=False)
            except (FileNotFoundError, ValueError):
                self.src = None

        def poll(self):
            if self.src is None:
                self.open()
                if self.src is None:
                    return
            snap = self.src.read()
            prev = self.prev
            if snap is None or (prev is not None and snap.id == prev.id):
                return
            if prev is not None and snap.id < prev.id:
                self.timeline.reset()  # a new run
                prev = None
            c = snap.counters
            if prev is not None and snap.host_ns > prev.host_ns:
                dm = c["messages"] - prev.counters["messages"]
                if dm == 0 and not snap.final:
                    return  # no new quotes: keep prev, so the blend spans the real change
                rate = dm * 1e9 / (snap.host_ns - prev.host_ns)
                self.rate += 0.1 * (rate - self.rate)
            self.prev = snap

            und = args.underlying or (snap.underlyings[0] if snap.underlyings else None)
            days, xs, vs, n = rows(snap, und, args.max_moneyness)
            if len(days) < 2:
                return
            self.cols = max(self.cols, max(len(x) for x in xs))
            grid = [resample(x, v, self.cols) for x, v in zip(xs, vs)]
            x = np.array([g[0] for g in grid])
            iv = np.array([g[1] for g in grid])
            self.timeline.add(snap.host_ns, days, x, iv)

            # Axes only ever widen: a jumping axis reads as a jumping surface.
            lo, hi = np.percentile(iv, 1), np.percentile(iv, 99)
            z = self.ivlim
            self.ivlim = (min(lo, z[0]), max(hi, z[1])) if z else (lo, hi)
            d = self.dlim
            self.dlim = (min(days[0], d[0]), max(days[-1], d[1])) if d else (days[0], days[-1])
            lo, hi = x[:, 0].min(), x[:, -1].max()
            m = self.xlim
            self.xlim = (min(lo, m[0]), max(hi, m[1])) if m else (lo, hi)
            r = self.root
            r.setProperty("xMin", float(np.floor(self.xlim[0] * 20) / 20))
            r.setProperty("xMax", float(np.ceil(self.xlim[1] * 20) / 20))
            r.setProperty("ivMin", float(np.floor(self.ivlim[0] - 1)))
            r.setProperty("ivMax", float(np.ceil(self.ivlim[1] + 1)))
            r.setProperty("dMin", float(self.dlim[0]))
            r.setProperty("dMax", float(self.dlim[1]))
            tag = "final" if snap.final else f"{self.rate / 1e3:.1f}k msg/s"
            r.setProperty("title", f"{und}  ·  feed {snap.feed_clock()}  ·  {tag}  ·  snapshot #{snap.id}  ·  "
                                   f"{n} OTM IVs, {len(days)} expiries  ·  {c['messages']:,} msgs, "
                                   f"{c['unmapped']} unmapped")

        @Slot()
        def frame(self):
            self.poll()
            now = time.monotonic_ns()
            self.ticks.append(now)
            tl = self.timeline
            if args.no_interp:
                delay = 0.0
            else:
                # One snapshot period plus a frame and a half behind the newest,
                # so there is almost always a newer surface to blend toward.
                frame_ns = np.median(np.diff(self.ticks)) if len(self.ticks) > 2 else 16.7e6
                delay = tl.period_ns() + 1.5 * frame_ns
            s = tl.at(now - delay)
            if s is None:
                return
            if len(self.ticks) > 2:
                fps = (len(self.ticks) - 1) * 1e9 / (self.ticks[-1] - self.ticks[0])
                self.root.setProperty("status", f"{fps:5.1f} fps  ·  {delay / 1e6:4.0f} ms behind")
            if s is self.shown:
                return  # held surface (run over, or a gap): no need to rebuild it
            self.shown = s
            days, x, iv = s
            self.proxy.resetArray([
                [QSurfaceDataItem(xv, yv, d) for xv, yv in zip(xr, vr)]
                for d, xr, vr in zip(days, x.tolist(), iv.tolist())
            ])

    backend = Backend()
    view = QQuickView()
    view.setTitle("vol_field live surface")
    view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
    view.rootContext().setContextProperty("backend", backend)
    view.setSource(QUrl.fromLocalFile(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "live_surface.qml")))
    if view.status() == QQuickView.Status.Error:
        for e in view.errors():
            print(e.toString(), file=sys.stderr)
        return 1
    root = view.rootObject()
    backend.root = root
    backend.proxy = root.findChild(QObject, "series").dataProxy()
    view.show()
    rc = app.exec()
    del view
    return rc


if __name__ == "__main__":
    sys.exit(main())

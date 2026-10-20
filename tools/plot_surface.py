#!/usr/bin/env python3
"""Draw the live implied-vol surface from a vol_field snapshot.

Left: the out-of-the-money IV surface over (ln K/F, days to expiry), as a
plot_trisurf. Right: one smile per expiry. The title carries the underlying,
the feed clock and the message rate (from consecutive snapshots' counters).

  python3 tools/plot_surface.py                       # /dev/shm/vol_field, same machine
  python3 tools/plot_surface.py --tcp localhost:7700  # through snapshot_relay.py (+ ssh -L)
  python3 tools/plot_surface.py --save frames --frames 40   # headless: PNGs, e.g. in a SLURM job

numpy and matplotlib only.
"""

import argparse
import os
import sys
import time

import numpy as np

import vfsnap


def strips(x, days):
    """Triangles that only join neighbouring expiries. A Delaunay triangulation
    of raw (ln K/F, days) makes long slivers across the sparse long end and
    spikes where one expiry's wing reaches past the next one's."""
    from matplotlib.tri import Triangulation
    uniq = np.unique(days)
    rank = np.searchsorted(uniq, days).astype(float)
    span = np.ptp(x) or 1.0
    tri = Triangulation(x / span * len(uniq), rank).triangles
    r = rank[tri]
    keep = (r.max(axis=1) - r.min(axis=1)) == 1
    return tri[keep] if keep.any() else tri


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shm", default="vol_field", help="/dev/shm name or path (default vol_field)")
    ap.add_argument("--tcp", help="HOST:PORT of a snapshot_relay.py instead of --shm")
    ap.add_argument("--underlying", help="default: the first one in the snapshot")
    ap.add_argument("--fps", type=float, default=8.0, help="redraws per second (default 8)")
    ap.add_argument("--save", metavar="DIR", help="render PNG frames into DIR instead of a window")
    ap.add_argument("--frames", type=int, default=40, help="with --save: how many (default 40)")
    ap.add_argument("--max-moneyness", type=float, default=0.6, help="|ln K/F| shown (default 0.6)")
    args = ap.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    src = vfsnap.open_source(shm=args.shm, tcp=args.tcp)
    fig = plt.figure(figsize=(14, 6.2))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    state = {"prev": None, "rate": 0.0, "zlim": None, "last": None}

    def draw(_frame=None):
        snap = src.read()
        if snap is None:
            snap = state["last"]
            if snap is None:
                return
        state["last"] = snap
        prev = state["prev"]
        if prev is not None and snap.host_ns > prev.host_ns and snap.id != prev.id:
            dm = snap.counters["messages"] - prev.counters["messages"]
            state["rate"] = dm * 1e9 / (snap.host_ns - prev.host_ns)
        if prev is None or snap.id != prev.id:
            state["prev"] = snap

        und = args.underlying or (snap.underlyings[0] if snap.underlyings else "?")
        pts, exps = snap.surface(und)
        x = np.log(pts["strike"] / pts["forward"])
        keep = np.abs(x) <= args.max_moneyness
        pts, exps, x = pts[keep], exps[keep], x[keep]
        days = exps["days"].astype(float)
        iv = pts["iv_otm"].astype(float) * 100.0

        ax3.clear()
        ax2.clear()
        if len(iv) >= 3 and len(np.unique(days)) >= 2:
            lo, hi = np.percentile(iv, 1), np.percentile(iv, 99)
            z = state["zlim"]
            # Only ever widen the z range: a jumping axis reads as a jumping surface.
            z = (min(lo, z[0]), max(hi, z[1])) if z else (lo, hi)
            state["zlim"] = z
            try:
                ax3.plot_trisurf(x, days, np.clip(iv, z[0], z[1]), triangles=strips(x, days),
                                 cmap=cm.viridis, linewidth=0.1, antialiased=True,
                                 vmin=z[0], vmax=z[1])
            except (RuntimeError, ValueError):
                ax3.scatter(x, days, iv, c=iv, cmap=cm.viridis, s=6)
            ax3.set_zlim(z[0] - 1, z[1] + 1)
        ax3.set_xlabel("ln K/F")
        ax3.set_ylabel("days to expiry")
        ax3.set_zlabel("IV %")
        ax3.view_init(elev=24, azim=-58)

        uniq = np.unique(days)
        colors = cm.viridis(np.linspace(0, 0.9, max(len(uniq), 1)))
        for col, d in zip(colors, uniq):
            m = days == d
            order = np.argsort(x[m])
            ax2.plot(x[m][order], iv[m][order], "-o", ms=2.5, lw=1.2, color=col, label=f"{int(d)}d")
        ax2.set_xlabel("ln K/F")
        ax2.set_ylabel("OTM implied vol %")
        ax2.grid(alpha=0.3)
        if len(uniq):
            ax2.legend(fontsize=8, ncol=2, title="expiry")
        if state["zlim"]:
            ax2.set_ylim(state["zlim"][0] - 1, state["zlim"][1] + 1)

        c = snap.counters
        tag = "final" if snap.final else f"{state['rate'] / 1e3:.1f}k msg/s"
        fig.suptitle(f"{und}  ·  feed {snap.feed_clock()}  ·  {tag}  ·  snapshot #{snap.id}  ·  "
                     f"{len(iv)} OTM IVs  ·  {c['messages']:,} msgs, {c['unmapped']} unmapped",
                     fontsize=11)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        period = 1.0 / args.fps
        for i in range(args.frames):
            t0 = time.monotonic()
            draw()
            fig.savefig(os.path.join(args.save, f"frame_{i:04d}.png"), dpi=90)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        print(f"wrote {args.frames} frames to {args.save}/")
        return 0

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, draw, interval=1000.0 / args.fps, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    del anim
    return 0


if __name__ == "__main__":
    sys.exit(main())

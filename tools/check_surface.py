#!/usr/bin/env python3
"""Check a vol_field surface against the truth bono_gen.py priced it from.

Reads a snapshot (normally the final one, left in shared memory after the run)
and the generator's truth file, then:

  * picks the truth step the field had reached: the last step whose final
    message sequence number is <= the snapshot's max seq;
  * compares every out-of-the-money IV on the surface with the true
    volatility. The bound is spread-derived: half the quoted spread over vega,
    i.e. the range of volatilities the quote itself admits, widened by what
    the forward's own error moves the IV;
  * compares each expiry's parity forward with the true forward;
  * prints the run counters and checks unmapped = 0, IV failures ~ 0, no
    table overflow, and that every two-sided OTM quote has an IV.

Exit status 0 on pass, 1 on fail.

  tools/check_surface.py --truth out/chain.truth.csv            # /dev/shm/vol_field
  tools/check_surface.py --truth chain.truth.csv --tcp localhost:7700
"""

import argparse
import csv
import math
import sys
import time

import numpy as np

import vfsnap

FWD_TOL = 1e-3       # max relative forward error
WITHIN_FRAC = 0.95   # share of points that must sit inside their own bound


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def load_truth(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: empty")
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    t = {
        "step": cols["step"].astype(int), "time": cols["time"].astype(float),
        "seq_end": cols["seq_end"].astype(np.int64), "expiry": cols["expiry"].astype(int),
        "strike": cols["strike"].astype(float), "cp": cols["cp"], "iv": cols["iv"].astype(float),
        "F": cols["F"].astype(float), "T": cols["T"].astype(float),
        "bid": cols["bid"].astype(float), "ask": cols["ask"].astype(float),
    }
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--shm", default="vol_field")
    ap.add_argument("--tcp", help="HOST:PORT of a snapshot_relay.py")
    ap.add_argument("--underlying", default=None, help="default: the first one in the snapshot")
    ap.add_argument("--rate", type=float, default=None, help="default: the run's --rate")
    args = ap.parse_args()

    src = vfsnap.open_source(shm=args.shm, tcp=args.tcp)
    snap = None
    for _ in range(50):
        snap = src.read()
        if snap is not None:
            break
        time.sleep(0.1)
    if snap is None:
        sys.exit("no stable snapshot to check")
    und = args.underlying or snap.underlyings[0]
    rate = args.rate if args.rate is not None else float(snap.header["rate"])
    c = snap.counters
    print(f"snapshot #{snap.id} ({'final' if snap.final else 'live'}), feed {snap.feed_clock()}, "
          f"{int(snap.header['n_options'])} options, {len(snap.expiries)} expiries, "
          f"{len(snap.pairs)} strikes")

    truth = load_truth(args.truth)
    steps = np.unique(truth["step"])
    seq_end = {s: truth["seq_end"][truth["step"] == s][0] for s in steps}
    reached = [s for s in steps if seq_end[s] <= c["max_seq"]]
    if not reached:
        sys.exit(f"the snapshot (max seq {c['max_seq']}) is from before the first truth step")
    step = reached[-1]
    partial = step != steps[-1] and c["max_seq"] > seq_end[step]
    sel = truth["step"] == step
    t = {k: v[sel] for k, v in truth.items()}
    print(f"truth step {step} (feed {t['time'][0]:.1f} s, seq <= {seq_end[step]})"
          f"{'  [the replay stopped mid-step: expect a little extra error]' if partial else ''}")

    key = {(int(e), round(k, 4), cp): i for i, (e, k, cp) in
           enumerate(zip(t["expiry"], t["strike"], t["cp"]))}

    # Forward error per expiry.
    fwd_ok = True
    fwd_err = {}
    for i, x in enumerate(snap.expiries):
        if x["und"] >= len(snap.underlyings) or snap.underlyings[x["und"]] != und:
            continue
        m = t["expiry"] == int(x["yymmdd"])
        if not m.any():
            continue
        f_true = t["F"][m][0]
        rel = abs(x["forward"] - f_true) / f_true if np.isfinite(x["forward"]) else math.inf
        fwd_err[i] = x["forward"] - f_true
        fwd_ok &= rel <= FWD_TOL
        print(f"  expiry {int(x['yymmdd'])}  {int(x['days']):4d}d  F {x['forward']:10.4f}  true "
              f"{f_true:10.4f}  rel err {rel:.1e}  ({int(x['n_forward'])} pairs voted)")

    # OTM IVs against truth.
    pts, exps = snap.surface(und)
    errs, bounds, within = [], [], 0
    for p, x in zip(pts, exps):
        cp = "C" if p["flags"] & vfsnap.OTM_CALL else "P"
        i = key.get((int(x["yymmdd"]), round(float(p["strike"]), 4), cp))
        if i is None:
            continue
        F, K, T, sig = t["F"][i], t["strike"][i], t["T"][i], t["iv"][i]
        df = math.exp(-rate * T)
        d1 = math.log(F / K) / (sig * math.sqrt(T)) + 0.5 * sig * math.sqrt(T)
        vega = df * F * norm_pdf(d1) * math.sqrt(T)
        delta = df * (0.5 * math.erfc(-d1 / math.sqrt(2)) - (0 if cp == "C" else 1))
        if vega < 1e-6:
            continue
        err = abs(float(p["iv_otm"]) - sig)
        spread_bound = 0.5 * (t["ask"][i] - t["bid"][i]) / vega
        fwd_bound = abs(delta * fwd_err.get(int(p["expiry"]), 0.0)) / vega
        errs.append(err)
        bounds.append(spread_bound)
        within += err <= spread_bound + fwd_bound + 1e-6
    errs, bounds = np.array(errs), np.array(bounds)

    # Two-sided OTM quotes in the truth that the surface has no IV for.
    otm_truth = ((t["cp"] == "C") & (t["strike"] >= t["F"])) | ((t["cp"] == "P") & (t["strike"] < t["F"]))
    two_sided = otm_truth & (t["bid"] > 0)
    expected = int(two_sided.sum())

    ok = True
    print(f"OTM IVs: {len(errs)} on the surface, {expected} two-sided OTM quotes in the truth")
    if len(errs) == 0:
        print("  FAIL no OTM points matched the truth")
        ok = False
    else:
        med, medb = float(np.median(errs)), float(np.median(bounds))
        frac = within / len(errs)
        print(f"  median |IV error| {med:.2e}  vs median spread bound {medb:.2e}   "
              f"(max {errs.max():.2e}; 90th pct {np.percentile(errs, 90):.2e})")
        print(f"  {frac:.1%} of points inside their own spread + forward bound")
        if med > medb:
            print("  FAIL median IV error above the median spread bound")
            ok = False
        if frac < WITHIN_FRAC:
            print(f"  FAIL fewer than {WITHIN_FRAC:.0%} of points inside their bound")
            ok = False
        if len(errs) < 0.98 * expected:
            print(f"  FAIL {expected - len(errs)} two-sided OTM quotes have no IV on the surface")
            ok = False
    if not fwd_ok:
        print(f"  FAIL a forward is off by more than {FWD_TOL:.0e} relative")
        ok = False

    solved, failed = c["iv_solved"], c["iv_failed"]
    print(f"counters: messages {c['messages']}, quotes {c['quotes']}, stale {c['stale']}, "
          f"unmapped {c['unmapped']}, IV solved {solved} / failed {failed}, malformed "
          f"{c['malformed']}, overflow {int(snap.header['overflow'])}")
    if c["unmapped"]:
        print("  FAIL quotes for options the field does not hold")
        ok = False
    if failed > 0.001 * max(solved, 1):
        print("  FAIL more than 0.1% of on-tick IV solves failed")
        ok = False
    if int(snap.header["overflow"]):
        print("  FAIL a field table overflowed")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

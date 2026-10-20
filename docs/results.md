# Results

Laptop: GTX 1650 (sm_75, 16 SMs, FP64 at 1/32), CUDA 13.4, driver 615.78.

## End to end

`bono_gen` defaults: 656 options, 8 expiries 2 d - 1 y, 15 s, 76,162 messages
in 1,234 packets; realtime, 1 queue, 100 ms snapshots.

| | |
|---|---|
| messages / quotes | 76,162 / 74,174, 0 malformed, 0 unmapped, 0 stale |
| IV on tick | 28,053 solved, 0 failed |
| surface pass | 0.41 ms GPU average (forward + re-solve 656 options + snapshot) |
| forward error | <= 1.3e-4 relative on every expiry |
| OTM IV error | median 3.2e-4 (vol points / 100) vs a median spread bound of 3.5e-3; 301 / 301 points inside their bound |
| `check_surface.py` | PASS |

The same check passes for the A+B duplicated feed over 4 queues (exactly half
the quotes stale, the two feeds racing on the same options from two blocks),
for two laps over UDP sockets via `send-udp`, and for an unpaced single pass
(the whole pcap in two batches: the final state is identical).

## Throughput

`job.sbatch bench`: unpaced, looped, 20 s per run; msg/s counts every BONO
message, quotes being ~97% of them.

| GPU | feed | queues | msg/s | quotes stale | IV on tick/s | surface pass avg / max |
|---|---|---|---|---|---|---|
| GTX 1650 | 656 options | 1 | 3.14M | 87.3% | 135k | 0.41 / 0.93 ms |
| GTX 1650 | 2200 options | 1 | 2.16M | 70.5% | 194k | 0.54 / 0.70 ms |
| GTX 1650 | 2200 options, A+B | 2 | 7.59M | 85.2% | 353k | 0.57 / 1.28 ms |

On WATcloud: **not run yet** - the login nodes were unreachable from the
network this was built on (port 22 timed out). The job script was exercised
end to end on the laptop with the `dev` preset in place of CVMFS; the first
`bench` jobs on the 2080 Ti, 3090 and 4090 fill in this table.

## Reading the numbers

* One receive block runs on one SM. Two queues on the A+B feed carry 3.5x the
  single-queue message rate: two SMs, and half the messages are duplicates
  that stop at the seqlock without an IV solve. Real BONO is spread over many
  multicast groups, and RSS spreads those over queues.
* "Stale" is high when unpaced: a 1024-frame batch holds most of a lap of the
  synthetic chain, so each option gets dozens of updates in one batch, and
  only those that beat the newest already applied count as applied. That is
  coalescing, not loss - the unpaced check above ends in exactly the truth.
* The on-tick IV solve (`double`, a few Newton steps around `erfc`) is the
  most expensive step per quote, and consumer GPUs run FP64 at 1/32 (Turing)
  or 1/64 (Ampere, Ada) of FP32. Throughput drops once the first surface pass
  provides forwards and on-tick solving starts.
* Contention is the synthetic worst case: 656 - 2200 options with a dozen
  updates per option per second each; a real chain spreads the same message
  rate over far more options.

# Building and running

## Build

Requirements: CUDA 12.2+, CMake 3.25+, a C++17 host compiler, and an sm_75+
GPU to run anything. Tested with CUDA 13.4 / gcc 16 on the laptop, 13.0 in the
DOCA container, and 12.2 / gcc 11.3 (`nvidia/cuda:12.2.2-devel-rockylinux9`,
the version WATcloud's CVMFS stack provides): the `watcloud` preset builds
without warnings there, and its tests and an end-to-end check pass on the
laptop GPU. Configuring for anything older than sm_75 is a configure error:
the per-option seqlock spins on a CAS and needs independent thread scheduling.
The Python tools need numpy and matplotlib (`bono_gen.py` and
`replay_pcap.py send-udp` need nothing beyond the standard library);
`live_surface.py` needs PySide6 with QtGraphs, Qt 6.8+ (Fedora:
`python3-pyside6`) instead of matplotlib.

```sh
cmake --preset dev              # vf_sim + tests, for this machine's GPU
cmake --build --preset dev -j
ctest --preset dev              # bono, black, field
```

| preset | builds | for |
|---|---|---|
| `dev` | vf_sim, tests | the GPU in this machine (`native`) |
| `debug` | same, `-O0 -g` | |
| `watcloud` | vf_sim, tests | sm_75;86;89 - every WATcloud GPU; configures on a login node with no GPU |
| `gpunetio` | vf_gpunetio only | inside the DOCA container (`scripts/doca-build.sh`) |

**vf_gpunetio** needs the DOCA SDK, which has no Fedora build:

```sh
scripts/doca-build.sh           # pulls nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host once
```

builds `build/gpunetio/vf_gpunetio` against DOCA 3.5 / CUDA 13.0 in the
container, then smoke-tests it there: with no NIC it must stop cleanly at
`no DOCA device at PCI ...`.

## On the laptop

```sh
python3 tools/bono_gen.py --out out/chain.pcap --date 2026-10-20     # + out/chain.truth.csv
build/dev/vf_sim --pcap out/chain.pcap --underlying TEST --date 2026-10-20 --rate 0.04 \
    --snapshot-ms 100 --duration 20000 &
python3 tools/plot_surface.py                                          # live window
python3 tools/check_surface.py --truth out/chain.truth.csv            # after the run: PASS / FAIL
```

`bono_gen.py` writes one synthetic underlying's chain (default 8 expiries x 41
strikes x C/P) exactly as Nasdaq frames it: a pre-open with the directory
spin, Trading Action and Security Open, then 15 s of quoting at 10 steps/s -
random-walk spot, a Black-76 smile whose level, front end, skew and curvature
drift as smooth mean-reverting factors (`--motion` scales them, 0 holds the
smile still), theo +- a spread rounded to the minimum price variation, several
messages per packet, a heartbeat each second, end-of-session. Up to 28
expiries (past 11: weeklies, monthlies and LEAPS) and any odd number of
strikes. `--dup` sends every packet on an A and a B feed; `--halt`
halts one option mid-session. The truth file has the volatility and forward
every quote was priced from, and `check_surface.py` compares a snapshot
against it: forwards within 1e-3 relative, OTM IVs within the spread the quote
itself admits, coverage, and the counters (unmapped = 0, IV failures under
0.1%, no table overflow).

The GTX 1650 in the laptop also drives the display, so it has a kernel
run-time watchdog: keep `--duration` to a few tens of seconds there.

### Frame sources and pacing

| flag | |
|---|---|
| `--pcap FILE` | replay a classic Ethernet pcap (pcapng: `editcap -F pcap`) |
| `--realtime`, `--speed X` | pace by capture timestamps (the default), X times faster |
| `--pps N` | N frames/s; `--pps 0` is unpaced: as fast as the GPU takes them |
| `--loop` | replay forever, moving MoldUDP64 sequence numbers up one lap's span per lap |
| `--feed G:P[,..]` | only frames to these destinations |
| `--udp G:P[,..]` | receive datagrams instead (multicast groups are joined; `--udp-iface IP`) |
| `--queues N` | receive blocks; frames hash to a queue by destination, like RSS |
| `--backoff NS` | `__nanosleep` while the ring is empty, instead of spinning |

and the common run flags: `--underlying SYM[,..]` (up to 8), `--date`,
`--rate`, `--duration MS` (0 = until Ctrl-C), `--snapshot-ms`, `--shm NAME`
(or a path on any tmpfs), `--verbose`.

### UDP

E.g. from another host (no root, no scapy):

```sh
build/dev/vf_sim --udp 233.54.12.111:18001 --underlying TEST --date 2026-10-20 --rate 0.04
tools/replay_pcap.py send-udp --pcap out/chain.pcap --pps 20000 --loop 2      # sender
```

### 60 fps

`tools/live_surface.py` draws the same snapshot in a Qt Quick 3D window
refilled every vsync - one row per expiry, one column per OTM point, the
wireframe showing where the points are - and interpolates between snapshots, a
frame or two behind the newest, so the surface moves on every frame. Feed it a
dense chain re-quoted 60 times a second and snapshot often:

```sh
python3 tools/bono_gen.py --out out/live.pcap --date 2026-10-20 --expiries 22 --strikes 81 \
    --seconds 20 --steps 60 --no-truth                                 # 3558 options, ~2M messages
build/dev/vf_sim --pcap out/live.pcap --underlying TEST --date 2026-10-20 --rate 0.04 \
    --snapshot-ms 8 --duration 25000 &
python3 tools/live_surface.py                                          # drag to rotate, wheel to zoom
```

On the GTX 1650 that holds 60 fps about 40 ms behind the feed, at ~110k
messages/s with every 8 ms surface pass on time.

### Remote plot

`tools/snapshot_relay.py --port 7700` next to the run serves each new snapshot
over TCP; `python3 tools/plot_surface.py --tcp localhost:7700` (or
`live_surface.py`, with the relay at `--hz 120`) through an `ssh -L` tunnel
draws it. `plot_surface.py --save DIR --frames N` renders PNGs instead of a
window (headless jobs).

## On WATcloud

Nodes are headless SLURM nodes without a CUDA toolkit or root;
`scripts/watcloud/env.sh` loads gcc, CUDA 12.2, cmake and Python
(scipy-stack) from the Alliance CVMFS stack and points the loader at the
node's driver.

```sh
ssh wato-login
cd vol_field && source scripts/watcloud/env.sh
cmake --preset watcloud && cmake --build --preset watcloud -j 8     # login node is fine
sbatch scripts/watcloud/job.sbatch                                  # functional (shard of any GPU)
sbatch --gres gpu:rtx_3090:1 scripts/watcloud/job.sbatch bench      # whole GPU, named
sbatch --gres gpu:rtx_2080_ti:1 scripts/watcloud/job.sbatch bench
sbatch --gres gpu:rtx_4090:1 scripts/watcloud/job.sbatch bench
sbatch --nodes 2 scripts/watcloud/job.sbatch udp                    # vf_sim --udp <- send-udp
sbatch scripts/watcloud/job.sbatch live                             # then, on the laptop:
#   ssh -N -L 7700:<node>:7700 wato-login     (the job log prints the line)
#   python3 tools/live_surface.py --tcp localhost:7700
```

| mode | does |
|---|---|
| `functional` | ctest; realtime run with PNG frames and `check_surface`; A+B feed over 4 queues, checked |
| `bench` | unpaced `--pps 0 --loop` runs: 656- and 2200-option chains on one queue, the 2200 A+B chain on two; table in `bench.md` |
| `live` | `--loop` replay of a 22-expiry chain at 60 steps/s for 25 min, snapshot every 8 ms, `snapshot_relay.py --hz 120` on port 7700 |
| `udp` | `vf_sim --udp` on node 1, `send-udp` from node 2, checked |

Everything lands in `out/watcloud/<jobid>/`. The GTX 1080 node is sm_61:
`vf_sim` refuses it, so name a GPU type when it matters.

## On real hardware (vf_gpunetio)

```sh
scripts/hw-preflight.sh --feed 233.54.12.111:18001   # read-only checks + the command line
sudo build/gpunetio/vf_gpunetio --nic 0000:c1:00.0 --gpu 0000:01:00.0 \
    --feed 233.54.12.111:18001,233.54.12.112:18002 --underlying SPY,QQQ --date 2026-10-20 --rate 0.04
python3 tools/plot_surface.py
```

`hw-preflight.sh` checks the kernel (>= 6.2 for dmabuf), the NVIDIA open
kernel modules or `nvidia_peermem`, sm_75+ and BAR1, a ConnectX-6 Dx / -7 or
BlueField-2/3 with Ethernet link type, DOCA's view of it, the NIC/GPU PCIe
topology and NUMA nodes, and picks the best pair. `vf_gpunetio` installs one
root-pipe entry per `--feed` (destination group and UDP port; `:PORT` for any
address), RSS over `--queues` receive queues, and joins the multicast groups
on the NIC's netdev from plain sockets so the switch forwards them; anything
not matched (IGMP, ssh) still reaches the kernel stack. To feed it from a
second host: `sudo tools/replay_pcap.py send --pcap chain.pcap --iface <tx>`.

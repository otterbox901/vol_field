#!/usr/bin/env bash
# vol_field on a WATcloud GPU node: build, test, generate a feed, run, check.
#
#   sbatch scripts/watcloud/job.sbatch [MODE]          # from the repo root
#
# MODE
#   functional (default)  ctest; bono_gen chain -> vf_sim (realtime) with PNG
#                         frames from plot_surface --save; check_surface; then
#                         an A+B duplicated feed over 4 queues, checked again
#   bench                 unpaced (--pps 0 --loop) runs: small and large chain on
#                         one queue, large A+B chain on two; results in bench.md
#   live                  vf_sim replays a dense chain re-quoted 60 times a second
#                         in a loop, snapshotting every 8 ms, while
#                         snapshot_relay.py serves it on port VF_RELAY_PORT
#                         (7700); the job log prints the ssh -L line for the laptop
#   udp                   vf_sim --udp on the first node, replay_pcap.py send-udp
#                         from the second (sbatch --nodes 2), then check
#
# GPU: a shard of any GPU by default. Benchmarks want a whole, named one:
#   sbatch --gres gpu:rtx_3090:1 scripts/watcloud/job.sbatch bench
#   sbatch --gres gpu:rtx_2080_ti:1 ... / --gres gpu:rtx_4090:1 ...
# (the GTX 1080 is sm_61, below the sm_75 floor: vf_sim refuses it.)
#
# Output: out/watcloud/<jobid>/ (pcaps, truth, logs, frames, final snapshot).
# Environment: VF_DATE (default today), VF_RATE (0.04), VF_RELAY_PORT (7700),
# VF_LIVE_MS (live run length, default 25 min), VF_BENCH_MS (per run, 20000).
#
#SBATCH --job-name=vol_field
#SBATCH --gres=shard:4096
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=vol_field-%j.out

set -euo pipefail

mode=${1:-functional}
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f CMakePresets.json && -d tools ]] || { echo "run from the vol_field repo root" >&2; exit 2; }

set +u   # the CVMFS profile and Lmod read unset variables
# shellcheck disable=SC1091
source scripts/watcloud/env.sh
set -u

job=${SLURM_JOB_ID:-local$$}
out=out/watcloud/$job
mkdir -p "$out"
date_=${VF_DATE:-$(date +%F)}
rate=${VF_RATE:-0.04}
shm=vf_$job                  # /dev/shm is shared by every job on the node
bin=build/watcloud/vf_sim

gpu=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader -i 0)
echo "== $(hostname -s): $gpu; mode $mode; output $out"

cleanup() {
    jobs -p | xargs -r kill 2>/dev/null || true
    [[ -e /dev/shm/$shm ]] && cp "/dev/shm/$shm" "$out/final.snap" && rm -f "/dev/shm/$shm"
    rm -f "/dev/shm/${shm}_dup"
}
trap cleanup EXIT

echo "== build"
cmake --preset watcloud > "$out/configure.log"
cmake --build --preset watcloud -j "${SLURM_CPUS_PER_TASK:-4}" > "$out/build.log"

common=(--underlying TEST --date "$date_" --rate "$rate")
fail=0

check() {   # check NAME TRUTH SHM
    echo "== check $1"
    if python3 tools/check_surface.py --truth "$2" --shm "$3" | tee "$out/check_$1.txt"; then
        echo "ok    $1"
    else
        echo "FAIL  $1"
        fail=1
    fi
}

case $mode in
functional)
    echo "== ctest"
    ctest --preset watcloud
    python3 tools/bono_gen.py --out "$out/chain.pcap" --date "$date_" --rate "$rate"
    echo "== vf_sim, realtime, 1 queue"
    "$bin" --pcap "$out/chain.pcap" "${common[@]}" --duration 30000 --shm "$shm" \
        > "$out/sim.log" 2>&1 &
    sim=$!
    python3 tools/plot_surface.py --shm "$shm" --save "$out/frames" --frames 60 --fps 4 &
    wait "$sim" || fail=1
    wait || true
    sed -n '/=== vol_field run/,/^snapshot/p' "$out/sim.log"
    check realtime "$out/chain.truth.csv" "$shm"

    python3 tools/bono_gen.py --out "$out/dup.pcap" --date "$date_" --rate "$rate" --dup --halt
    echo "== vf_sim, A+B duplicated feed, 4 queues, x4 speed"
    "$bin" --pcap "$out/dup.pcap" "${common[@]}" --queues 4 --speed 4 --duration 20000 \
        --shm "${shm}_dup" > "$out/sim_dup.log" 2>&1 || fail=1
    sed -n '/=== vol_field run/,/^snapshot/p' "$out/sim_dup.log"
    check dup "$out/dup.truth.csv" "${shm}_dup"
    ;;

bench)
    ms=${VF_BENCH_MS:-20000}
    python3 tools/bono_gen.py --out "$out/chain.pcap" --date "$date_" --rate "$rate"
    python3 tools/bono_gen.py --out "$out/big.pcap" --date "$date_" --rate "$rate" \
        --expiries 11 --strikes 101 --seconds 20
    python3 tools/bono_gen.py --out "$out/bigdup.pcap" --date "$date_" --rate "$rate" \
        --expiries 11 --strikes 101 --seconds 20 --dup
    {
        echo "| GPU | feed | queues | msg/s | quotes stale | IV on tick/s | surface pass avg / max |"
        echo "|---|---|---|---|---|---|---|"
    } > "$out/bench.md"
    run_bench() {   # run_bench NAME PCAP QUEUES
        local log=$out/bench_${1// /_}.log
        echo "== bench $1: --pps 0 --loop, $3 queue(s), $ms ms"
        "$bin" --pcap "$2" "${common[@]}" --pps 0 --loop --queues "$3" --duration "$ms" \
            --shm "$shm" > "$log" 2>&1 || fail=1
        sed -n '/=== vol_field run/,/^snapshot/p' "$log"
        local msgs stale ivs secs surf
        msgs=$(awk '/^messages/ {gsub(/[(]/, "", $3); print $3}' "$log")
        stale=$(awk '/^  stale/ {gsub(/[(]/, "", $3); print $3}' "$log")
        ivs=$(awk '/^IV on tick/ {print $4}' "$log")
        secs=$(awk '/^elapsed/ {print $2}' "$log")
        surf=$(awk '/^surface/ {print $8, "/", $12, "ms"}' "$log")
        printf '| %s | %s | %s | %s | %s | %.0f | %s |\n' "${gpu%%,*}" "$1" "$3" "$msgs" "$stale" \
            "$(awk -v n="$ivs" -v s="$secs" 'BEGIN {print (s > 0 ? n / s : 0)}')" "$surf" \
            >> "$out/bench.md"
    }
    run_bench "chain 656 opts" "$out/chain.pcap" 1
    run_bench "big 2200 opts" "$out/big.pcap" 1
    run_bench "big A+B" "$out/bigdup.pcap" 2
    echo "== results"
    cat "$out/bench.md"
    ;;

live)
    port=${VF_RELAY_PORT:-7700}
    python3 tools/bono_gen.py --out "$out/live.pcap" --date "$date_" --rate "$rate" --seconds 60 \
        --expiries 22 --strikes 81 --steps 60 --no-truth
    "$bin" --pcap "$out/live.pcap" "${common[@]}" --loop --duration "${VF_LIVE_MS:-1500000}" \
        --snapshot-ms 8 --shm "$shm" > "$out/sim.log" 2>&1 &
    sim=$!
    python3 tools/snapshot_relay.py --shm "$shm" --port "$port" --hz 120 &
    sleep 2
    echo "== live on $(hostname -s):$port until the run ends (scancel $job to stop). On the laptop:"
    echo "   ssh -N -L $port:$(hostname -s):$port wato-login"
    echo "   python3 tools/live_surface.py --tcp localhost:$port     # or plot_surface.py"
    wait "$sim" || fail=1
    tail -20 "$out/sim.log"
    ;;

udp)
    python3 tools/bono_gen.py --out "$out/chain.pcap" --date "$date_" --rate "$rate"
    port=18001
    nodes=($(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname -s)}"))
    rx=${nodes[0]}
    tx=${nodes[1]:-${nodes[0]}}
    [[ $rx == "$tx" ]] && echo "== one node only: sending over loopback (sbatch --nodes 2 for two)"
    echo "== vf_sim --udp on $rx, send-udp from $tx"
    srun --nodes 1 --ntasks 1 -w "$rx" --overlap \
        "$bin" --udp ":$port" "${common[@]}" --duration 25000 --shm "$shm" \
        > "$out/sim_udp.log" 2>&1 &
    sim=$!
    sleep 4
    srun --nodes 1 --ntasks 1 -w "$tx" --overlap --gres none \
        python3 tools/replay_pcap.py send-udp --pcap "$out/chain.pcap" --dst "$rx:$port" \
        --pps 20000 --loop 2 | tee "$out/send_udp.log"
    wait "$sim" || fail=1
    sed -n '/=== vol_field run/,/^snapshot/p' "$out/sim_udp.log"
    check udp "$out/chain.truth.csv" "$shm"
    ;;

*)
    echo "unknown mode '$mode' (functional | bench | live | udp)" >&2
    exit 2
    ;;
esac

echo "== done: $([[ $fail == 0 ]] && echo PASS || echo FAIL); output in $out"
exit $fail

#!/usr/bin/env bash
# Read-only checks before the first vf_gpunetio run on real hardware.
#
#   scripts/hw-preflight.sh [--feed G:P[,..]] [--underlying SYM] [--nic PCI] [--gpu PCI]
#
# Looks at what GPU-direct receive needs, changes nothing, and ends with the
# vf_gpunetio command line for the best NIC/GPU pair it found. Some queries
# (mlxconfig) need root; without it they are skipped with a note.
#
#   kernel     >= 6.2 (dmabuf export of GPU memory to the NIC)
#   driver     NVIDIA open kernel modules (dmabuf), else nvidia_peermem loaded
#   GPU        sm_75+, BAR1 big enough for the receive buffers
#   NIC        ConnectX-6 Dx / ConnectX-7 / BlueField-2/3, Ethernet link type,
#              netdev up; DOCA sees it
#   topology   NIC and GPU on the same NUMA node, ideally behind one PCIe switch
#   memory     hugepages (informational)

set -uo pipefail

feed="233.54.12.111:18001"
und="TEST"
want_nic=""
want_gpu=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --feed) feed=$2; shift 2 ;;
        --underlying) und=$2; shift 2 ;;
        --nic) want_nic=$2; shift 2 ;;
        --gpu) want_gpu=$2; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

ok=0 warn=0 bad=0
pass() { printf '  ok    %s\n' "$*"; ok=$((ok + 1)); }
note() { printf '  note  %s\n' "$*"; warn=$((warn + 1)); }
fail() { printf '  FAIL  %s\n' "$*"; bad=$((bad + 1)); }
have() { command -v "$1" > /dev/null 2>&1; }
section() { printf '\n%s\n' "$*"; }

# Receive buffer per queue: 16384 packets x 2 KiB (src/gpunetio/setup.cpp).
buf_mib=32

section "kernel"
kver=$(uname -r)
if printf '6.2\n%s\n' "${kver%%-*}" | sort -V -C; then
    pass "Linux $kver (>= 6.2)"
else
    fail "Linux $kver: dmabuf GPU memory for the NIC wants >= 6.2 (or nvidia_peermem)"
fi

section "NVIDIA driver"
if [[ -r /proc/driver/nvidia/version ]]; then
    drv=$(head -1 /proc/driver/nvidia/version)
    if grep -q "Open Kernel Module" /proc/driver/nvidia/version; then
        pass "open kernel modules, $(grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' <<< "$drv" | head -1)"
    else
        note "proprietary kernel modules: dmabuf needs the open ones; nvidia_peermem is the fallback"
    fi
else
    fail "no NVIDIA driver loaded (/proc/driver/nvidia/version)"
fi
if lsmod 2> /dev/null | grep -q '^nvidia_peermem'; then
    pass "nvidia_peermem loaded (fallback path available)"
else
    note "nvidia_peermem not loaded (only needed when dmabuf is unavailable)"
fi

section "GPUs"
declare -a gpus=()
if have nvidia-smi; then
    while IFS=, read -r bus name cc bar1; do
        bus=$(tr '[:upper:]' '[:lower:]' <<< "${bus// /}")
        bus=${bus#0000}; bus="0000${bus:4}"
        bdf=${bus}
        numa=$(cat "/sys/bus/pci/devices/$bdf/numa_node" 2> /dev/null || echo "?")
        gpus+=("$bdf|$numa")
        line="$bdf ${name# } sm_${cc//[ .]/} BAR1 ${bar1# } MiB, NUMA $numa"
        if [[ ${cc//[ .]/} -lt 75 ]]; then
            fail "$line: below sm_75"
        elif [[ ${bar1// /} =~ ^[0-9]+$ && ${bar1// /} -lt $((buf_mib * 2)) ]]; then
            note "$line: small BAR1 - enable Resizable BAR, or keep --queues low"
        else
            pass "$line"
        fi
    done < <(nvidia-smi --query-gpu=pci.bus_id,name,compute_cap,memory.total \
                 --format=csv,noheader 2> /dev/null |
             while IFS=, read -r b n c _; do
                 bar=$(nvidia-smi -q -i "${b// /}" -d MEMORY 2> /dev/null |
                       awk '/BAR1 Memory Usage/ {f=1} f && /Total/ {print $3; exit}')
                 echo "$b,$n,$c,${bar:-?}"
             done)
else
    fail "nvidia-smi not found"
fi

section "NICs (Mellanox / NVIDIA networking)"
declare -a nics=()
if have lspci; then
    while read -r bdf _ rest; do
        dev=$(lspci -n -s "$bdf" | awk '{print $3}')
        id=${dev#*:}
        case $id in
            101d) model="ConnectX-6 Dx" ;;
            101f) model="ConnectX-6 Lx" ;;
            1021) model="ConnectX-7" ;;
            1023) model="ConnectX-8" ;;
            a2d6) model="BlueField-2" ;;
            a2dc) model="BlueField-3" ;;
            101b|101c) model="ConnectX-6" ;;
            *) model="" ;;
        esac
        [[ $bdf == *.* && ${#bdf} -lt 12 ]] && bdf="0000:$bdf"
        numa=$(cat "/sys/bus/pci/devices/$bdf/numa_node" 2> /dev/null || echo "?")
        netdev=$(ls "/sys/bus/pci/devices/$bdf/net" 2> /dev/null | head -1)
        state=""
        if [[ -n $netdev ]]; then
            state="$netdev $(cat "/sys/class/net/$netdev/operstate" 2> /dev/null)"
            state+=", MTU $(cat "/sys/class/net/$netdev/mtu" 2> /dev/null)"
        fi
        line="$bdf ${model:-$rest} (15b3:$id), NUMA $numa${state:+, $state}"
        case $model in
            "ConnectX-6 Dx"|ConnectX-7|ConnectX-8|BlueField-2|BlueField-3)
                pass "$line"; nics+=("$bdf|$numa|$netdev") ;;
            "")
                note "$line: not a GPUNetIO receive device" ;;
            *)
                fail "$line: GPUNetIO needs ConnectX-6 Dx or newer" ;;
        esac
        if [[ -n $model ]] && have mlxconfig; then
            if [[ $EUID -eq 0 ]]; then
                lt=$(mlxconfig -d "$bdf" q 2> /dev/null | awk '/LINK_TYPE_P1/ {print $2}')
                if [[ $lt == ETH* ]]; then
                    pass "  $bdf LINK_TYPE_P1 $lt"
                elif [[ -n $lt ]]; then
                    fail "  $bdf LINK_TYPE_P1 $lt (needs ETH(2))"
                fi
            else
                note "  $bdf: mlxconfig -d $bdf q | grep LINK_TYPE needs root; skipped"
            fi
        fi
    done < <(lspci -d 15b3: 2> /dev/null | awk '{$2=$2; print $1, "x", substr($0, index($0, $2))}')
    [[ ${#nics[@]} -eq 0 ]] && fail "no ConnectX-6 Dx / -7 or BlueField-2/3 found (lspci -d 15b3:)"
else
    note "lspci not found; cannot list NICs"
fi
if have doca_caps; then
    if doca_caps --list-devs > /dev/null 2>&1; then
        pass "doca_caps lists: $(doca_caps --list-devs 2> /dev/null | awk '/PCI/ {printf "%s ", $NF}')"
    else
        note "doca_caps --list-devs failed"
    fi
elif have ibv_devinfo; then
    pass "ibv_devinfo: $(ibv_devinfo -l 2> /dev/null | tail -n +2 | xargs)"
else
    note "neither doca_caps nor ibv_devinfo installed (DOCA host packages missing?)"
fi

section "topology"
if have nvidia-smi && [[ ${#nics[@]} -gt 0 ]]; then
    nvidia-smi topo -m 2> /dev/null | sed 's/^/  | /'
    echo "  (PIX/PXB: same PCIe switch - best; NODE: same NUMA node; SYS: across sockets - avoid)"
fi

section "memory"
hp=$(awk '/HugePages_Total/ {print $2}' /proc/meminfo)
hs=$(awk '/Hugepagesize/ {print $2}' /proc/meminfo)
pass "hugepages: ${hp:-0} x ${hs:-?} kB (informational; DOCA Flow may want some)"

# Pick the pair: the requested ones, else prefer a NIC and GPU on one NUMA node.
nic="" gpu="" netdev="" same=0
for n in "${nics[@]}"; do
    IFS='|' read -r nb nn nd <<< "$n"
    [[ -n $want_nic && $nb != "$want_nic" ]] && continue
    for g in "${gpus[@]}"; do
        IFS='|' read -r gb gn <<< "$g"
        [[ -n $want_gpu && $gb != "$want_gpu" ]] && continue
        s=0
        [[ $nn == "$gn" ]] && s=1
        if [[ -z $nic || $s -gt $same ]]; then
            nic=$nb gpu=$gb netdev=$nd same=$s
        fi
    done
done
[[ -z $nic && -n $want_nic && -n $want_gpu ]] && nic=$want_nic gpu=$want_gpu

section "summary: $ok ok, $warn notes, $bad failed"
if [[ -n $nic && -n $gpu ]]; then
    cat << EOF

Run (root, or CAP_NET_ADMIN + CAP_NET_RAW + access to /dev/infiniband):
  sudo build/gpunetio/vf_gpunetio --nic $nic --gpu $gpu \\
      --feed $feed --underlying $und --date $(date +%F) --rate 0.04
then, as your user on the same machine:
  python3 tools/plot_surface.py
Feed it from a second host (frames must reach ${netdev:-the NIC}'s port):
  sudo tools/replay_pcap.py send --pcap chain.pcap --iface <tx-iface> --pps 20000
EOF
else
    echo
    echo "No usable NIC/GPU pair: vf_gpunetio cannot run here. vf_sim runs anywhere (sm_75+)."
fi
[[ $bad -eq 0 ]]

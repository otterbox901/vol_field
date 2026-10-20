#!/usr/bin/env bash
# Build vf_gpunetio inside NVIDIA's DOCA container, then smoke-test it.
#
# DOCA has no Fedora build, and neither this laptop nor WATcloud has a
# ConnectX NIC, so the container exists to compile and link against genuine
# DOCA headers and libraries - not to receive anything. It ships DOCA 3.5 and
# CUDA 13, so no Dockerfile of our own is needed.
#
#   scripts/doca-build.sh            configure + build + smoke test
#   scripts/doca-build.sh --shell    drop into the container instead
#
# The smoke test runs the binary with a PCI address that cannot exist and
# expects it to stop cleanly at "no DOCA device at PCI ...": the binary loads,
# its arguments parse, and the NIC lookup is the first thing that runs.
#
# The build tree lands in build/gpunetio/, owned by your user via --userns.

set -euo pipefail

IMAGE="${VF_DOCA_IMAGE:-nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! podman image exists "$IMAGE" 2>/dev/null; then
    echo "[vf] pulling $IMAGE (~15 GB, once)" >&2
    podman pull "$IMAGE"
fi

# Only ask for a TTY when we have one, so piping the output stays quiet.
TTY_FLAGS=()
[[ -t 0 ]] && TTY_FLAGS=(-it)

run() {
    podman run --rm "${TTY_FLAGS[@]}" \
        --userns=keep-id \
        -v "$ROOT":/src:Z \
        -w /src \
        "$IMAGE" "$@"
}

if [[ "${1:-}" == "--shell" ]]; then
    exec run bash
fi

# nvcc cannot probe a GPU here (the container has no driver), so the preset
# pins CMAKE_CUDA_ARCHITECTURES instead of using `native`. The container has
# no libcuda.so.1 either; the toolkit's stub stands in for the smoke test,
# which never gets as far as a CUDA call.
run bash -lc '
set -euo pipefail
cmake --preset gpunetio '"$*"' && cmake --build --preset gpunetio -j"$(nproc)"
stub=$(mktemp -d)
ln -s /usr/local/cuda/lib64/stubs/libcuda.so "$stub/libcuda.so.1"
set +e
out=$(LD_LIBRARY_PATH="$stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" build/gpunetio/vf_gpunetio \
      --nic 0000:ff:1f.7 --gpu 0000:ff:1f.6 --feed 233.54.12.111:18001 \
      --underlying TEST --date 2026-10-20 2>&1)
code=$?
set -e
echo "$out" | sed "s/^/  | /"
if [[ $code == 1 ]] && grep -q "no DOCA device at PCI 0000:ff:1f.7" <<<"$out"; then
    echo "[vf] smoke test ok: vf_gpunetio built, loads, and stops cleanly without a NIC"
else
    echo "[vf] smoke test FAILED (exit $code)" >&2
    exit 1
fi
'

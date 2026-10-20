# Source me on WATcloud (login node or inside a job): the CVMFS toolchain for
# building and running vol_field.
#
#   source scripts/watcloud/env.sh
#   cmake --preset watcloud && cmake --build --preset watcloud -j 8   # login node is fine
#
# The nodes have no CUDA toolkit of their own; this loads gcc, CUDA 12.2, cmake
# and Python (numpy + matplotlib via scipy-stack) from the Alliance (Compute
# Canada) software stack on CVMFS. Override any module with VF_MODULES.
#
# The stack's loader does not search the host's driver directory, so on a GPU
# node the driver libraries (libcuda and the JIT helpers, nothing else) are
# linked into a node-local directory that goes on LD_LIBRARY_PATH.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "source this file: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

_vf_cvmfs=/cvmfs/soft.computecanada.ca/config/profile/bash.sh
if [[ ! -r $_vf_cvmfs ]]; then
    echo "vf env: $_vf_cvmfs not found - is CVMFS mounted on $(hostname)?" >&2
    return 1
fi
# shellcheck disable=SC1090
source "$_vf_cvmfs"

VF_MODULES=${VF_MODULES:-"StdEnv/2023 gcc cuda/12.2 cmake python/3.11 scipy-stack"}
# shellcheck disable=SC2086
if ! module load $VF_MODULES; then
    echo "vf env: 'module load $VF_MODULES' failed; see 'module spider <name>' and set VF_MODULES" >&2
    return 1
fi

# Driver libraries for the CVMFS loader (GPU nodes only).
_vf_libcuda=$(/sbin/ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 .*x86-64/ {print $NF; exit}')
if [[ -z $_vf_libcuda ]]; then
    for _d in /usr/lib/x86_64-linux-gnu /usr/lib64/nvidia /usr/lib64; do
        [[ -e $_d/libcuda.so.1 ]] && _vf_libcuda=$_d/libcuda.so.1 && break
    done
fi
if [[ -n $_vf_libcuda ]]; then
    _vf_drv="${SLURM_TMPDIR:-/tmp}/vf-driver-$USER-$(hostname -s)"
    mkdir -p "$_vf_drv"
    _vf_dir=$(dirname "$(readlink -f "$_vf_libcuda")")
    for _f in "$_vf_dir"/libcuda.so* "$_vf_dir"/libnvidia-ptxjitcompiler.so* \
              "$_vf_dir"/libnvidia-nvvm.so*; do
        [[ -e $_f ]] && ln -sf "$_f" "$_vf_drv/"
    done
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$_vf_drv:"*) ;;
        *) export LD_LIBRARY_PATH="$_vf_drv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

_vf_cmake=$(cmake --version 2>/dev/null | awk 'NR==1 {print $3}')
if [[ -n $_vf_cmake ]] && [[ $(printf '%s\n3.25\n' "$_vf_cmake" | sort -V | head -1) != 3.25 ]]; then
    echo "vf env: cmake $_vf_cmake is older than the presets need (3.25);" \
         "try VF_MODULES with cmake/3.27.7" >&2
fi
_vf_gcc=$(gcc -dumpfullversion 2> /dev/null)
_vf_cuda=$(nvcc --version 2> /dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo "vf env: gcc ${_vf_gcc:-?}, CUDA ${_vf_cuda:-?}, cmake ${_vf_cmake:-?}," \
     "$(python3 --version 2>&1)${_vf_libcuda:+, driver from $_vf_dir}"
unset _vf_cvmfs _vf_libcuda _vf_drv _vf_dir _f _d _vf_cmake _vf_gcc _vf_cuda

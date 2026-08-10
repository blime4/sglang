#!/bin/bash
# DL: Route B — build sglang's OWN _vllm_fa2_C.so from the DLIN flash-attention
# source (no copy from a vLLM venv). This produces the exact extension sglang
# imports via torch.ops._vllm_fa2_C.varlen_fwd (the clean, graph-safe
# varlen+block_table decode path), compiled with dlcc against the sglang venv's
# torch, and installs it into the sglang venv's vllm_flash_attn/ package.
#
# Why this works without vLLM: the DLIN _vllm_fa2_C target is pure C++
# (flash_api.cpp + flash_attn_dlgpu.cpp; no .cu, no gencode). The flash-attention
# repo's own top-level CMakeLists.txt is a self-contained project that defines
# and installs _vllm_fa2_C. We drive that CMakeLists as the top-level project
# (dlcc as CXX compiler, sglang venv's torch/python), build only the _vllm_fa2_C
# target, then drop the .so + the repo's python wrappers into the sglang venv.
#
# Usage: bash scripts/dl/build_dlin_vllm_flash_attn.sh
#   SDK_DIR=... DLIN_FA_SRC=... SG_VENV=...  (all optional, have sensible defaults)
set -euo pipefail

# --- config (env-overridable) ---
HERE="$(cd "$(dirname "$0")" && pwd)"
SGLANG_ROOT="$(cd "$HERE/../.." && pwd)"
SDK_DIR="${SDK_DIR:-/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sdk-0401}"
SG_VENV="${SG_VENV:-$SGLANG_ROOT/.venv}"
DLIN_FA_SRC="${DLIN_FA_SRC:-$SGLANG_ROOT/../flash-attention}"
BUILD_DIR="${BUILD_DIR:-$SGLANG_ROOT/build/dlin_vllm_fa2}"
TOOLS_CMAKE="${TOOLS_CMAKE:-/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/tools/cmake-3.31.6-linux-x86_64/bin/cmake}"

# --- sanity checks ---
[ -f "$SDK_DIR/env.sh" ]     || { echo "[dl-fa] ERROR: SDK env.sh not found: $SDK_DIR"; exit 1; }
[ -d "$SG_VENV" ]            || { echo "[dl-fa] ERROR: sglang venv not found: $SG_VENV"; exit 1; }
[ -f "$DLIN_FA_SRC/CMakeLists.txt" ] || { echo "[dl-fa] ERROR: DLIN FA source not found: $DLIN_FA_SRC"; exit 1; }
[ -f "$DLIN_FA_SRC/csrc/flash_attn/flash_api.cpp" ] || { echo "[dl-fa] ERROR: flash_api.cpp missing in $DLIN_FA_SRC"; exit 1; }
CMAKE="$TOOLS_CMAKE"
[ -x "$CMAKE" ] || CMAKE="$(command -v cmake)"
[ -n "$CMAKE" ] || { echo "[dl-fa] ERROR: cmake not found"; exit 1; }

# --- environment: SDK (dlcc + DLIN runtime) then sglang venv (torch/python) ---
# Temporarily relax nounset: the SDK env.sh references $1 (unbound under set -u).
# shellcheck disable=SC1091
set +u; source "$SDK_DIR/env.sh"; set -u
# shellcheck disable=SC1091
set +u; source "$SG_VENV/bin/activate"; set -u

DLCC="$(command -v dlcc)"
[ -n "$DLCC" ] || { echo "[dl-fa] ERROR: dlcc not on PATH after sourcing SDK env"; exit 1; }
PY_EXE="$(command -v python)"
TORCH_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
DEST="$SITE_PKGS/vllm_flash_attn"

echo "[dl-fa] SDK_DIR        = $SDK_DIR"
echo "[dl-fa] DLIN_FA_SRC    = $DLIN_FA_SRC"
echo "[dl-fa] SG_VENV        = $SG_VENV"
echo "[dl-fa] BUILD_DIR      = $BUILD_DIR"
echo "[dl-fa] cmake          = $CMAKE ($($CMAKE --version | head -1))"
echo "[dl-fa] dlcc (CXX)     = $DLCC"
echo "[dl-fa] python         = $PY_EXE ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "[dl-fa] torch          = $(python -c 'import torch; print(torch.__version__)')"
echo "[dl-fa] torch prefix   = $TORCH_PREFIX"
echo "[dl-fa] install dest   = $DEST"

# --- configure: drive the FA repo's own CMakeLists.txt as the top-level project ---
echo "[dl-fa] ---- cmake configure ----"
"$CMAKE" -S "$DLIN_FA_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$DLCC" \
  -DCMAKE_CXX_STANDARD=17 \
  -DPython_EXECUTABLE="$PY_EXE" \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$SITE_PKGS"

# --- build only the _vllm_fa2_C target (pure C++, 2 files; minutes) ---
echo "[dl-fa] ---- cmake build _vllm_fa2_C ----"
"$CMAKE" --build "$BUILD_DIR" --target _vllm_fa2_C -j

# --- locate the built .so ---
SO="$(find "$BUILD_DIR" -type f -name '_vllm_fa2_C*.so' | head -1)"
[ -n "$SO" ] || { echo "[dl-fa] ERROR: build did not produce _vllm_fa2_C*.so under $BUILD_DIR"; find "$BUILD_DIR" -name '*.so' | head; exit 1; }
echo "[dl-fa] built .so = $SO ($($(command -v stat||echo stat) -c%s "$SO" 2>/dev/null || stat -f%z "$SO" 2>/dev/null) bytes)"

# --- install the .so + the repo's python wrappers into the sglang venv ---
echo "[dl-fa] ---- install into $DEST ----"
mkdir -p "$DEST"
# back up any prior .so for traceability (previous from-source build, or a
# legacy copied-from-vLLM one)
[ -f "$DEST/_vllm_fa2_C"*.so ] && mv "$DEST"/_vllm_fa2_C*.so "$DEST/_vllm_fa2_C.so.prev.bak" 2>/dev/null || true
cp -f "$SO" "$DEST/"
# python wrappers (canonical: the FA repo's own — what vLLM's install copies too)
cp -rf "$DLIN_FA_SRC/vllm_flash_attn/"*.py "$DEST/" 2>/dev/null || true
cp -rf "$DLIN_FA_SRC/vllm_flash_attn/layers" "$DEST/" 2>/dev/null || true
cp -rf "$DLIN_FA_SRC/vllm_flash_attn/ops"    "$DEST/" 2>/dev/null || true
echo "[dl-fa] installed:"
ls -la "$DEST"/ | sed 's/^/     /'

# --- verify ---
echo "[dl-fa] ---- verify ----"
python - <<'PY'
import torch, vllm_flash_attn
assert hasattr(torch.ops, "_vllm_fa2_C"), "torch.ops._vllm_fa2_C namespace not registered"
_ = torch.ops._vllm_fa2_C.varlen_fwd   # raises if the compiled _C extension is absent
print("[dl-fa] OK: import vllm_flash_attn + torch.ops._vllm_fa2_C.varlen_fwd present")
PY
echo "[dl-fa] DONE. _vllm_fa2_C.so built from source and installed into the sglang venv."

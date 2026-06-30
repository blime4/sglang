#!/bin/bash
# Copy the DLIN-patched vllm_flash_attn (with _vllm_fa2_C.so) from an existing
# vLLM DL venv into the sglang venv. The dl19 build is ABI-compatible with dl24
# torch (both torch 2.9.1, same SDK runtime). The standalone test crashes on
# cold JIT ("to bc failed") but model warmup preheats the dleol JIT cache.
#
# Usage: bash scripts/dl/setup_vllm_flash_attn.sh [source_venv]
# Default source: ../venv-vllm021 (vLLM 0.21.0 + dl19 torch)
set -e
SG_VENV="${SG_VENV:-$(pwd)/.venv}"
SRC="${1:-$(pwd)/../venv-vllm021}"
DEST="$SG_VENV/lib/python3.12/site-packages/vllm_flash_attn"

if [ -f "$DEST/_vllm_fa2_C.cpython-312-x86_64-linux-gnu.so" ]; then
  echo "[vllm_flash_attn] already present, skipping."
  exit 0
fi

# Find _vllm_fa2_C.so in the source venv
SRC_SO=$(find "$SRC/lib" -name "_vllm_fa2_C*.so" 2>/dev/null | head -1)
if [ -z "$SRC_SO" ]; then
  echo "[vllm_flash_attn] ERROR: no _vllm_fa2_C.so found in $SRC"
  echo "Install vLLM DL first: dl-env skill or venv-vllm021"
  exit 1
fi
SRC_DIR=$(dirname "$SRC_SO")

# Backup vanilla vllm_flash_attn if present
if [ -d "$DEST" ] && [ ! -f "$DEST/_vllm_fa2_C*.so" ]; then
  mv "$DEST" "${DEST}_vanilla_bak"
  echo "[vllm_flash_attn] backed up vanilla stub."
fi

# Copy the DLIN-patched vllm_flash_attn
mkdir -p "$DEST"
cp "$SRC_DIR"/* "$DEST/" 2>/dev/null || true
# Also copy the python wrapper from venv-vllm021's vllm/vllm_flash_attn
SRC_PY=$(find "$SRC/lib" -path "*/vllm_flash_attn/__init__.py" 2>/dev/null | head -1)
if [ -n "$SRC_PY" ]; then
  SRC_PKG=$(dirname "$SRC_PY")
  cp "$SRC_PKG"/*.py "$DEST/" 2>/dev/null || true
fi
echo "[vllm_flash_attn] copied from $SRC_DIR"
echo "[vllm_flash_attn] sglang will auto-detect and use it (see _dlin_vllm_flash_attn_ok)."

#!/bin/bash
# DL: DEPRECATED shim — now BUILDS _vllm_fa2_C.so from the DLIN flash-attention
# source (Route B) instead of copying vLLM's prebuilt .so. Kept under the old
# name so existing docs/invocations still work; it simply delegates to the
# from-source build. Call the build script directly for clarity:
#   scripts/dl/build_dlin_vllm_flash_attn.sh
set -euo pipefail
echo "[setup_vllm_flash_attn] DEPRECATED: no longer copies from a vLLM venv."
echo "[setup_vllm_flash_attn] building _vllm_fa2_C.so from source (Route B) ->"
exec bash "$(dirname "$0")/build_dlin_vllm_flash_attn.sh"

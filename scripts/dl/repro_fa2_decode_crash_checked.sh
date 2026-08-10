#!/usr/bin/env bash
# Self-checking wrapper for repro_fa2_decode_crash.py (Bug 18483).
#
# `to bc failed` is a JIT/bitcode-load crash INSIDE libdleol.so. It only appears
# on the EXACT pinned combo below; any divergence -> no crash, or a DIFFERENT
# crash. This script prints the full env fingerprint, forces the clean env, runs
# the repro, and tells you which branch you landed in.
#
# Usage:
#   SDK_DIR=/path/to/sdk PYTHON=/path/to/venv/bin/python ./repro_fa2_decode_crash_checked.sh
#   # or: ./repro_fa2_decode_crash_checked.sh /path/to/sdk
#   # optional: CUDA_VISIBLE_DEVICES=<free GPU>
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$SCRIPT_DIR/repro_fa2_decode_crash.py"
LOG="$SCRIPT_DIR/repro_fa2_decode_crash.last.log"

# --- libdleol.so this bug was filed against (reporter's box, dated 2026-06-01) ---
REF_MD5="1660c77b447b81ae60c04d4158a5b99b"
REF_SDK_TAG="MRrc-4.2.0-202606161052"

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'

# ---- resolve SDK_DIR ----
SDK_DIR="${SDK_DIR:-${1:-}}"
if [[ -z "$SDK_DIR" || ! -d "$SDK_DIR/lib" ]]; then
  echo "${RED}ERROR:${RST} set SDK_DIR (env or \$1) pointing at the SDK root with lib/libdleol.so"
  echo "  e.g. SDK_DIR=/LocalRun/.../sdk $0"
  exit 2
fi
SDK_DIR="$(cd "$SDK_DIR" && pwd)"
PY="${PYTHON:-python3}"
INHERITED_LDLP="${LD_LIBRARY_PATH:-<empty>}"
INHERITED_JIT="${DLEOL_DISABLE_JIT:-<unset>}"

echo "${CYN}================ ENV FINGERPRINT ================${RST}"
echo "SDK_DIR             : $SDK_DIR"
echo "PYTHON              : $PY"

LIB="$SDK_DIR/lib/libdleol.so"
if [[ -f "$LIB" ]]; then
  size=$(stat -c '%s' "$LIB" 2>/dev/null)
  mtime=$(stat -c '%y' "$LIB" 2>/dev/null | cut -d. -f1)
  md5=$(md5sum "$LIB" | awk '{print $1}')
  echo "libdleol.so         : $LIB"
  echo "  size              : $size bytes"
  echo "  mtime             : $mtime"
  echo "  md5               : $md5"
  echo "  ${CYN}bug reference${RST}      : md5 $REF_MD5 (SDK $REF_SDK_TAG, dated 2026-06-01)"
  if [[ "$md5" == "$REF_MD5" ]]; then
    echo "  >>> ${GRN}MATCH${RST} — same libdleol.so the bug was filed against."
  else
    echo "  >>> ${YLW}MISMATCH${RST} — different libdleol.so build. This is the #1 reason"
    echo "      'to bc failed' does not reproduce: a newer build may have the bitcode"
    echo "      load already fixed; an older/different build hits a different path."
  fi
else
  echo "${RED}libdleol.so NOT FOUND at $LIB${RST}"
fi
echo "inherited LD_LIBRARY_PATH : $INHERITED_LDLP"
echo "inherited DLEOL_DISABLE_JIT: $INHERITED_JIT"
echo "${CYN}=================================================${RST}"
echo

# ---- pitfall checks on the INHERITED env (what the colleague had) ----
n_dirs=$(printf '%s\n' "$INHERITED_LDLP" | tr ':' '\n' | grep -c . || true)
if [[ "$n_dirs" -gt 1 ]]; then
  echo "${YLW}[WARN]${RST} inherited LD_LIBRARY_PATH had $n_dirs dirs."
  echo "   A 2nd SDK dir -> duplicate libhcrt/libLLVM -> LLVM PassBuilder crash,"
  echo "   which is a DIFFERENT failure than 'to bc failed'."
fi
if [[ "$INHERITED_JIT" == "1" ]]; then
  echo "${YLW}[WARN]${RST} DLEOL_DISABLE_JIT=1 was set in the inherited env."
  echo "   That turns the failure into std::invalid_argument: stoi (SIGABRT),"
  echo "   NOT 'to bc failed'."
fi

# ---- source SDK env, then FORCE the clean single-dir LD_LIBRARY_PATH ----
if [[ -f "$SDK_DIR/env.sh" ]]; then
  # env.sh may reference positional args / unbound vars; relax -u while sourcing.
  set +u
  # shellcheck disable=SC1091
  source "$SDK_DIR/env.sh"
  set -u
fi
export CUDA_HOME="$SDK_DIR"
export DLI_V2=ON
export LD_LIBRARY_PATH="$SDK_DIR/lib"      # ONLY this dir (bug requirement)
unset DLEOL_DISABLE_JIT                     # ensure JIT path is the one tested

echo "${CYN}================ RUNTIME ENV (forced) ================${RST}"
echo "CUDA_HOME           : $CUDA_HOME"
echo "DLI_V2              : $DLI_V2"
echo "LD_LIBRARY_PATH     : $LD_LIBRARY_PATH  (must be exactly one SDK/lib dir)"
echo "DLEOL_DISABLE_JIT   : ${DLEOL_DISABLE_JIT:-<unset, JIT enabled — correct>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<UNSET — set to a free GPU via dlsmi>}"

echo "torch / flash_attn  :"
DLIN_ROUTE=""
FA_INFO=$("$PY" - <<'PY' 2>/dev/null
import torch, flash_attn, os, glob
print("torch=%s" % torch.__version__)
print("flash_attn=%s" % getattr(flash_attn, "__version__", "?"))
print("fa_path=%s" % os.path.dirname(flash_attn.__file__))
so = glob.glob(os.path.join(os.path.dirname(flash_attn.__file__), "flash_attn_2_cuda*.so"))
print("C_EXT=" + (so[0] if so else ""))
PY
)
if [[ -z "$FA_INFO" ]]; then
  echo "   ${RED}(python/torch import failed — venv or SDK mismatch)${RST}"
else
  echo "$FA_INFO" | sed 's/^/   /'
  CEXT=$(echo "$FA_INFO" | sed -n 's/^C_EXT=//p')
  if [[ -n "$CEXT" && -f "$CEXT" ]]; then
    if nm -D "$CEXT" 2>/dev/null | grep -qi "dldnn_mha_fwd_kvcache" \
       || strings "$CEXT" 2>/dev/null | grep -qi "dldnn_mha_fwd_kvcache"; then
      DLIN_ROUTE=1
      echo "   DLIN kvcache routing: ${GRN}PRESENT${RST} — flash_attn_with_kvcache reaches"
      echo "      cudnnMHAForwardKVCacheWithSinks (the crashing dleol op). Correct build."
    else
      DLIN_ROUTE=0
      echo "   DLIN kvcache routing: ${YLW}ABSENT${RST} — this is a standard/upstream flash_attn build."
      echo "      flash_attn_with_kvcache does NOT route into the dleol kvcache op, so"
      echo "      'to bc failed' CANNOT occur here regardless of libdleol.so."
      echo "      Need DLIN FA2 build V2_SOFTWARE_master_202606031721 + torch 2.9.1+dl24.sdk202606031721."
    fi
  fi
fi
echo "${CYN}====================================================${RST}"
echo

if [[ ! -f "$REPRO" ]]; then
  echo "${RED}ERROR:${RST} repro not found: $REPRO"; exit 2
fi

echo "${CYN}================ RUN repro_fa2_decode_crash.py ================${RST}"
"$PY" "$REPRO" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "----------------------------------------------------------------"
echo "exit code: $rc   (stderr log saved to: $LOG)"
echo
if grep -q "to bc failed" "$LOG"; then
  echo "${RED}>>> REPRODUCED: 'to bc failed' (SIGSEGV in libdleol.so).${RST}"
  echo "    Bug 18483 is live on this libdleol.so + DLIN flash_attn build."
elif [[ $rc -eq 0 ]]; then
  if [[ "$DLIN_ROUTE" == "1" ]]; then
    echo "${GRN}>>> NOT reproduced — kernel returned OK with DLIN routing PRESENT.${RST}"
    echo "    Same routing as the bug, but no crash -> this libdleol.so build has the"
    echo "    bitcode load FIXED. Report its md5/date (above) to confirm the fix shipped."
  else
    echo "${YLW}>>> NOT reproduced — but DLIN kvcache routing is ABSENT.${RST}"
    echo "    flash_attn_with_kvcache never reached the dleol op (ran the standard kernel"
    echo "    instead). Install the DLIN FA2 build V2_SOFTWARE_master_202606031721 with"
    echo "    torch 2.9.1+dl24.sdk202606031721, then re-run. libdleol.so is irrelevant here."
  fi
else
  echo "${YLW}>>> DIFFERENT failure (not 'to bc failed').${RST}"
  echo "    Check the pitfalls above (LD_LIBRARY_PATH double-dir, JIT flag),"
  echo "    and the libdleol.so / torch / flash_attn fingerprint."
fi

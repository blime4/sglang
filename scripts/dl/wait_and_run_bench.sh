#!/bin/bash
# DL: wait until a target QUAD of GPUs frees up, then auto-run the feature benchmark.
# Handles the shared-DLIN-machine contention (e.g. a colleague's TP32 job taking all GPUs).
# Confirms free for 2 consecutive checks (30s apart) before launching, to avoid grabbing
# during a transient.
# NOTE: no `set -u` — sdk env.sh references unset vars.
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh 2>/dev/null
TARGET_GPUS="28 29 30 31"
TARGET_CSV="28,29,30,31"
MAX_WAIT=${MAX_WAIT:-5400}   # 90 min cap
POLL=${POLL:-45}
LOG=/tmp/bench_features.log

mem_used() {  # $1 = gpu idx -> MiB (integer) or 99999 on error
  dlsmi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null \
    | awk -F, -v i="$1" '{ gsub(/[^0-9]/,"",$1); if ($1+0==i) { u=$2; gsub(/[^0-9]/,"",u); print u+0; exit } }'
}

quad_free() {
  for i in $TARGET_GPUS; do
    u=$(mem_used "$i")
    [ -z "$u" ] && return 1
    [ "$u" -gt 1000 ] && return 1
  done
  return 0
}

START=$(date +%s)
echo "[$(date '+%H:%M:%S')] wait_and_run: polling GPUs $TARGET_GPUS every ${POLL}s (cap ${MAX_WAIT}s)"
while true; do
  now=$(date +%s); elapsed=$((now-START))
  if [ "$elapsed" -gt "$MAX_WAIT" ]; then
    echo "[$(date '+%H:%M:%S')] TIMEOUT after ${elapsed}s — GPUs never freed. Giving up." | tee -a "$LOG"
    exit 2
  fi
  if quad_free; then
    echo "[$(date '+%H:%M:%S')] GPUs free (check 1/2); confirming in 30s..."
    sleep 30
    if quad_free; then
      echo "[$(date '+%H:%M:%S')] GPUs free (check 2/2) — launching benchmark" | tee -a "$LOG"
      MEM_FRAC=0.55 CUDA_VISIBLE_DEVICES=$TARGET_CSV \
        .venv/bin/python scripts/dl/bench_features_sglang_vllm.py 2>&1 | tee -a "$LOG"
      exit ${PIPESTATUS[0]}
    else
      echo "[$(date '+%H:%M:%S')] transient — GPUs re-occupied, keep waiting"
    fi
  fi
  # status line every poll
  st=""
  for i in $TARGET_GPUS; do st="$st $i:$(mem_used $i)M"; done
  echo "[$(date '+%H:%M:%S')] elapsed=${elapsed}s |$st"
  sleep "$POLL"
done

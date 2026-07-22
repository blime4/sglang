#!/bin/bash
# DL: wait for GPUs 28-31, then run the full feature comparison (3 configs) sequentially.
# Phase 1: full script (sglang FP8+cuda-graph + vLLM Int4+eager)
# Phase 2: sglang FP8+eager (mode-matched to vLLM)
# No `set -u` (sdk env.sh references unset vars).
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh 2>/dev/null
TARGET_GPUS="28 29 30 31"
TARGET_CSV="28,29,30,31"
MAX_WAIT=${MAX_WAIT:-5400}
POLL=${POLL:-45}

mem_used() {
  dlsmi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null \
    | awk -F, -v i="$1" '{ gsub(/[^0-9]/,"",$1); if ($1+0==i) { u=$2; gsub(/[^0-9]/,"",u); print u+0; exit } }'
}
quad_free() {
  for i in $TARGET_GPUS; do u=$(mem_used "$i"); [ -z "$u" ] && return 1; [ "$u" -gt 1000 ] && return 1; done
  return 0
}

echo "[$(date '+%H:%M:%S')] wait_run_full: polling GPUs $TARGET_GPUS every ${POLL}s (cap ${MAX_WAIT}s)"
START=$(date +%s)
while true; do
  now=$(date +%s); elapsed=$((now-START))
  if [ "$elapsed" -gt "$MAX_WAIT" ]; then echo "[$(date '+%H:%M:%S')] TIMEOUT waiting for GPUs"; exit 2; fi
  if quad_free; then
    echo "[$(date '+%H:%M:%S')] free (check 1/2); confirm in 30s"; sleep 30
    if quad_free; then break; else echo "[$(date '+%H:%M:%S')] transient; keep waiting"; fi
  fi
  sleep "$POLL"
done

echo "[$(date '+%H:%M:%S')] ===== PHASE 1: full (sglang-CG + vLLM-eager) =====" | tee /tmp/run_full.log
MEM_FRAC=0.55 CUDA_VISIBLE_DEVICES=$TARGET_CSV \
  .venv/bin/python scripts/dl/bench_features_sglang_vllm.py 2>&1 | tee -a /tmp/run_full.log
echo "[$(date '+%H:%M:%S')] PHASE 1 done (rc=${PIPESTATUS[0]})" | tee -a /tmp/run_full.log

echo "[$(date '+%H:%M:%S')] ===== PHASE 2: sglang-eager (mode-matched) =====" | tee /tmp/run_sglang_eager.log
SKIP_VLLM=1 SGLANG_EAGER=1 MEM_FRAC=0.55 CUDA_VISIBLE_DEVICES=$TARGET_CSV \
  .venv/bin/python scripts/dl/bench_features_sglang_vllm.py 2>&1 | tee -a /tmp/run_sglang_eager.log
echo "[$(date '+%H:%M:%S')] PHASE 2 done (rc=${PIPESTATUS[0]})" | tee -a /tmp/run_sglang_eager.log
echo "[$(date '+%H:%M:%S')] ===== BOTH PHASES COMPLETE ====="

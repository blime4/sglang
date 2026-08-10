#!/usr/bin/env bash
# DL begin — vLLM TP serve + HTTP TPOT benchmark for sglang-vs-vLLM comparison on DLIN.
# Why serving mode: vLLM offline LLM() hangs/crashes on DLIN for spec decoding
# (decode CG capture device page fault). Always use `vllm serve` + HTTP. See
# .claude/skills/dl-compare-sglang-vllm/SKILL.md §4/§6.
#
# Usage:
#   vllm_serve_bench.sh <gpu_csv> <mrv:1|2> <spec:none|mtp> [tp] [model] [max_new_tokens]
# Examples:
#   vllm_serve_bench.sh 20,21,22,23 2 mtp 4 /mars/aebox/LLM/model/Qwen3.5-35B-A3B-GPTQ-Int4 160
#   vllm_serve_bench.sh 20 2 none 1 /mars/aebox/LLM/model/Qwen3.5-35B-A3B-GPTQ-Int4 160
#
# Env: SUDO_PW (for dlsmi reset), optional VLLM_PORT (default auto per GPU).
# Prereq: source sdk-dlop-07-13-20-30/env.sh; PYTHONPATH=vllm-new-overlay; venv-vllm021.
set +e
cd "$(dirname "$0")/../.." || exit 1

GPUS="${1:?usage: gpu_csv mrv spec [tp] [model] [max_new]}"
MRV="${2:?mrv: 1|2}"; SPEC="${3:?spec: none|mtp}"
TP="${4:-1}"; MODEL="${5:-/mars/aebox/LLM/model/Qwen3.5-35B-A3B-GPTQ-Int4}"
MAXNEW="${6:-160}"
FIRST_GPU="${GPUS%%,*}"
PORT="${VLLM_PORT:-$((8200 + FIRST_GPU))}"

export CUDA_VISIBLE_DEVICES="$GPUS"
[ "$MRV" = "2" ] && export VLLM_USE_V2_MODEL_RUNNER=1 || unset VLLM_USE_V2_MODEL_RUNNER
export DLEOL_USE_CU_MQA_TILEKV=1 VLLM_MAX_MOE_CU_TOKENS=128 DLEOL_FLA_ENABLE_PINGPONG=1 \
       DLEOL_FLA_UNROLL_COUNT=8 DLEOL_CACHE_SIZE=1024 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPEC_ARG=""
[ "$SPEC" = "mtp" ] && SPEC_ARG='--speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":3}'
TAG="MRV${MRV}-${SPEC}-TP${TP}"; export TAG

echo "### vllm serve $TAG gpus=$GPUS port=$PORT ###"
PYTHONPATH=vllm-new-overlay ../venv-vllm021/bin/vllm serve "$MODEL" \
  --port "$PORT" --dtype half --max-model-len 8192 --tensor-parallel-size "$TP" \
  --max-num-seqs 64 \
  --compilation-config '{"cudagraph_capture_sizes":[16],"max_cudagraph_capture_size":16}' \
  --trust-remote-code --served-model-name bench $SPEC_ARG \
  > "/tmp/vsrv_${TAG}.log" 2>&1 &
SRV=$!

# wait for health (up to 25 min — DLIN FULL-CG capture is slow)
ready=0
for _ in $(seq 1 150); do
  sleep 10
  if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$SRV" 2>/dev/null || { echo "### $TAG DIED ###"; grep -aiE "Error|ValueError|page fault|ConstraintViolation" "/tmp/vsrv_${TAG}.log" | tail -3; exit 2; }
done
[ "$ready" = 1 ] || { echo "### $TAG NOT ready (capture too slow?) ###"; tail -6 "/tmp/vsrv_${TAG}.log"; kill -9 "$SRV" 2>/dev/null; exit 3; }

# HTTP TPOT benchmark (warmup 3, best-of-3)
P="$PORT" MX="$MAXNEW" .venv/bin/python - <<'PYEOF'
import time, urllib.request, json, os
P=os.environ["P"]; MX=int(os.environ["MX"])
URL=f"http://localhost:{P}/v1/completions"
PROMPT=("Write a Python function that takes a list of integers and returns the "
        "longest increasing subsequence. Use dynamic programming. Include comments.")
def gen(mx):
    body=json.dumps({"model":"bench","prompt":PROMPT,"temperature":0,"max_tokens":mx}).encode()
    r=json.loads(urllib.request.urlopen(urllib.request.Request(URL,data=body,
        headers={"Content-Type":"application/json"}),timeout=180).read())
    return r["usage"]["completion_tokens"]
for _ in range(3): gen(16)
best=9e9; bt=0; n=MX
for _ in range(3):
    t0=time.time(); n=gen(MX); dt=time.time()-t0; tpot=dt/n*1000
    if tpot<best: best=tpot; bt=n/dt
print(f"[RESULT {os.environ.get('CUDA_VISIBLE_DEVICES','?')} {os.environ.get('TAG','')}] "
      f"TPOT={best:.2f}ms tps={bt:.1f}", flush=True)
PYEOF

kill -9 "$SRV" 2>/dev/null
sleep 3
echo "### $TAG done ###"
# DL end

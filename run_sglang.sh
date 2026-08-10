#!/bin/bash
#===============================================================================
# run_sglang.sh
#-------------------------------------------------------------------------------
# One-stop driver for building/running SGLang on DLIN (DLIN) GPUs.
#
# Phases (run individually or all at once):
#   setup        source DLIN SDK, create a uv venv (py3.12), install DLIN torch
#                (+dl24 build, exposes torch.version.dl) + build deps.
#   build-kernel best-effort build of sgl-kernel from source with dlcc.
#                (Full DLIN kernel coverage is the subject of the operator-
#                 integration plan; this phase may be partial by design.)
#   install      editable-install the sglang python package using the DLIN
#                pyproject (python/pyproject_dl.toml). Non-destructive.
#   test|smoke   import smoke (torch.version.dl + import sglang) + a GPU UT.
#   gen          QUICK TEST: end-to-end Qwen3 generation on DLIN (in-process
#                sglang.Engine). The handy one for verifying a model runs.
#   serve        launch the OpenAI-compatible HTTP server (launch_server).
#   bench        concurrent serving benchmark (wraps sglang.bench_serving;
#                auto-starts a server) or --offline per-batch latency
#                (sglang.bench_one_batch, no server).
#   benchrun     vLLM-benchrun-format serving bench (wraps scripts/dl/
#                benchrun_sglang.py). Same config schema + output as the
#                team's `vllm bench run` -> sglang vs vLLM diff directly.
#   compare      sglang vs vLLM showcase (wraps scripts/dl/
#                showcase_prefix_sharing.py) — SC1 prefix-sharing / SC2
#                multi-turn / SC3 concurrent-batch. Runs BOTH engines on the
#                same GPUs (fresh process each), caches metrics, prints a
#                side-by-side gap table. The one-click tracker: re-run after
#                any sglang change to see if the gap moved.
#   sop          DLIN UPGRADE verification standard (wraps scripts/dl/
#                sop_verify.py). Correctness gates (absolute) + perf gates
#                (within a tolerance band of a recorded baseline) -> PASS/FAIL.
#                The judging bar for porting DLIN changes onto a new sglang tag.
#   all          setup -> build-kernel -> install -> test  (default)
#
# Usage:
#   ./run_sglang.sh                       # run all phases
#   ./run_sglang.sh setup                 # just create/refresh the env
#   ./run_sglang.sh test                  # import + GPU smoke against an existing env
#   ./run_sglang.sh gen                   # quick e2e generation (default Qwen3-1.7B)
#   ./run_sglang.sh gen -m /opt/dataset/Qwen3-1.7B -p "Hello" -n 32
#   ./run_sglang.sh serve --port 30000
#   ./run_sglang.sh bench -c 32 --num-prompts 200   # concurrent bench (autostarts server)
#   ./run_sglang.sh bench --offline                  # per-batch latency, no server
#   ./run_sglang.sh benchrun -M qwen35-35b --num-prompts 8   # vLLM-format bench (autostarts server)
#   ./run_sglang.sh benchrun --template                      # write a config_serving.json to edit
#   ./run_sglang.sh compare                                  # sglang vs vLLM showcase (SC1-3, gap table)
#   ./run_sglang.sh compare --scenarios SC2,SC3              # skip the ~90s SC1 cold prefill
#   ./run_sglang.sh compare --only sglang                    # re-measure sglang, diff vs cached vLLM
#
# gen / serve options (override the env vars below):
#   -m, --model PATH         model path           (default $MODEL_PATH)
#   -p, --prompt TEXT        single prompt       (default builtin demos)
#   -n, --max-new-tokens N   decode length       (default 16)
#   -b, --backend NAME       attention backend   (default fa3)
#   -g, --cuda-graph         enable cuda graph   (default off; safer on DLIN)
#   --port N / --host H      serve only          (default 30000 / 127.0.0.1)
# bench options: -c/--concurrency, --num-prompts, --input-len/--output-len,
#   --batch-size '1 4 8 16', --base-url URL, --offline
#
# Env overrides (all optional):
#   SDK_DIR          DLIN SDK root (env.sh). Default: see below.
#   ARTIFACTORY_DIR  repo with piplike/uv.toml/pip.conf. Default: see below.
#   VENV_DIR         venv location (default: .venv).
#   PYTHON_VERSION   default 3.12.
#   TORCH_SPEC       torch pin (default: 2.9.1+dl24.sdk202606031721).
#   SKIP_KERNEL=1    skip the build-kernel phase.
#   DL_DOWNLOAD_METHOD  how to fetch Artifactory files: jfrog (default, uses the
#                       configured jfrog CLI, auto-falls back to wget) | wget
#                       (basic auth, prompts --user/--password, caches ~/.netrc).
#   JFROG_SERVER_ID     jfrog server-id to use (else first configured is picked).
#   Quick-test (gen/serve): MODEL_PATH, ATTN_BACKEND, USE_CUDA_GRAPH,
#   MAX_NEW_TOKENS, PROMPT, SERVE_PORT, SERVE_HOST.
#===============================================================================
set -eo pipefail

#-------------------------------------------------------------------------------
# Config
#-------------------------------------------------------------------------------
SGLANG_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# SDK_DIR resolution. The .venv is built against a specific DLIN SDK build; a
# stale SDK (older libhcrt/libLLVM) segfaults the sglang scheduler at model
# load (exit -11) or hangs in JIT. So prefer the SDK co-located with the .venv
# over a globally-exported SDK_DIR (e.g. a stale sdk-0401 in ~/.bashrc):
#   1. SGSDK_DIR (run_sglang-specific override)            -> use verbatim
#   2. newest SGLANG_DIR/sdk-dlop-*/ (next to the .venv)   -> authoritative
#   3. inherited SDK_DIR (e.g. from profile)               -> fallback
#   4. hardcoded ../debug/sdk                              -> last resort
if [ -n "${SGSDK_DIR:-}" ]; then
  SDK_DIR="$SGSDK_DIR"
else
  __latest=""
  for __cand in "$SGLANG_DIR"/sdk-dlop-*; do
    [ -d "$__cand" ] && [ -f "$__cand/env.sh" ] && __latest="$__cand"
  done
  if [ -n "$__latest" ]; then
    SDK_DIR="$__latest"
  else
    SDK_DIR="${SDK_DIR:-/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sdk}"
  fi
  unset __cand __latest
fi
ARTIFACTORY_DIR="${ARTIFACTORY_DIR:-/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/flash-attention/artifactory}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-$SGLANG_DIR/.venv}"

# DLIN-patched torch that exposes torch.version.dl (NOT vanilla upstream 2.9.1).
TORCH_SPEC="${TORCH_SPEC:-torch==2.9.1+dl24.sdk202606031721}"

# DLIN triton 3.3.0 with dlgpu backend — compiles @triton.jit kernels to
# dlgput64 format that the DLIN driver can load. Without this, Triton kernels
# produce NVIDIA ELF -> "Unsupported elf format" at CG capture.
TRITON_URL="${TRITON_URL:-http://ext-artifactory.denglin.com:8082/artifactory/sw-triton/V2_SOFTWARE_master_202607201723/cp312-cp312-manylinux_2_28_x86_64/triton-3.3.0%2Bgit0f83b16e-cp312-cp312-manylinux_2_28_x86_64.whl}"

# DLIN Artifactory indexes (http -> needs trusted/allow-insecure host).
DL_PYPI_INDEX="http://ext-artifactory.denglin.com:8082/artifactory/api/pypi/dl-pypi-remote/simple"
DL_VIRTUAL_INDEX="http://ext-artifactory.denglin.com:8082/artifactory/api/pypi/dl-virtual/simple"
DL_HOST="ext-artifactory.denglin.com"
DL_ARTIFACTORY_ROOT="${DL_ARTIFACTORY_ROOT:-http://ext-artifactory.denglin.com:8082/artifactory}"

# How to fetch files hosted on the DLIN Artifactory (ext-artifactory.denglin.com).
#   jfrog (default) -> `jfrog rt download` using the configured jfrog CLI (server
#                      + access token from `jfrog config`). Needs no password on
#                      the command line. If the CLI is absent / unconfigured,
#                      dl_download auto-falls back to wget. Set this if you ran
#                      `jfrog config add`.
#   wget            -> wget with basic auth. External-net access needs
#                      --user/--password, prompted once and cached in ~/.netrc.
#                      Use this if you have NOT configured the jfrog CLI.
# Override the jfrog server-id with JFROG_SERVER_ID (else the first configured
# server is auto-selected).
DL_DOWNLOAD_METHOD="${DL_DOWNLOAD_METHOD:-jfrog}"

#-------------------------------------------------------------------------------
# Quick-test (gen/serve) config. Overridable by env vars AND by the gen/serve
# CLI flags (-m/-p/-n/-b/-g, --port/--host). See header for the full list.
#-------------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/opt/dataset/Qwen3-1.7B}"
ATTN_BACKEND="${ATTN_BACKEND:-fa3}"          # fa3 -> FlashAttention -> DLIN FA2
USE_CUDA_GRAPH="${USE_CUDA_GRAPH:-0}"         # 0 = disable_cuda_graph (safer on DLIN)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
PROMPT="${PROMPT:-}"                          # empty -> builtin demo prompts
SERVE_PORT="${SERVE_PORT:-30000}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"

# NGRAM speculative decoding — 2.8-3.2x vLLM on qwen35-35b (docs §7.12).
# Enabled with -S / --spec-ngram. Needs the fused-MoE path + extra mem headroom
# for the num_draft=8 draft tree; apply_ngram_overrides() sets those when on.
USE_NGRAM="${USE_NGRAM:-0}"
DL_WARMUP="${DL_WARMUP:-0}"    # -W/--dl-warmup: pre-compile dlcc kernels (MoE + GDN dl_chunk) at serve start.
# Measured effect (2026-07-28, with SGLANG_DL_GDN_DLIN_EXTEND=1): only ~293ms (5.7%) per
# new shape — the dl_chunk first-use JIT (2K prefill reps 5426->5133ms). The BIG prefill
# win was the GDN flag itself (41s->5.1s, sglang beats vLLM 6/9), NOT this warmup. Use -W
# only to shave first-request latency / serving suffix-JIT spikes; it is NOT the prefill lever.
# GOTCHA: a crashed run can corrupt dl_chunk's triton cache -> restore ~/.triton/cache.good_backup.
NGRAM_NUM_DRAFT="${NGRAM_NUM_DRAFT:-8}"
NGRAM_MIN_BFS="${NGRAM_MIN_BFS:-1}"
NGRAM_MAX_BFS="${NGRAM_MAX_BFS:-1}"

# bench options (wraps sglang's official bench_serving / bench_one_batch).
# NOTE: concurrency defaults to 1 -- the DLIN server crashes under batched
# prefill (extend) once >~2 requests overlap, because the DLIN FA2 wrapper's
# q.reshape (jit_kernel/flash_attention.py, flash_attn_with_kvcache DLIN branch)
# assumes a uniform seqlen per batch row. Raising -c probes the ceiling; real
# concurrency needs that wrapper routed to flash_attn_varlen_func (ragged).
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-1}"      # --max-concurrency (online)
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-64}"     # total requests (online)
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"       # random prompt len (online)
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-128}"      # random output len (online)
BATCH_SIZE="${BATCH_SIZE:-1 4 8 16}"             # --batch-size sweep (offline)
BENCH_BASE_URL="${BENCH_BASE_URL:-}"             # nonempty -> bench external server
BENCH_OFFLINE="${BENCH_OFFLINE:-0}"              # 1 -> bench_one_batch (no server)

# benchrun options (wraps scripts/dl/benchrun_sglang.py — vLLM `bench run` format).
# --config PATH runs that config verbatim; --template just writes config_serving.json;
# otherwise a config is built from the flags + (-M) preset, then run.
BENCHRUN_CONFIG="${BENCHRUN_CONFIG:-}"           # --config PATH (passthrough to the harness)
BENCHRUN_TEMPLATE="${BENCHRUN_TEMPLATE:-0}"      # 1 -> write the config template, don't run
BENCHRUN_TP="${DLIN_TP_SIZE:-1}"                 # tensor-parallel size (set by -M qwen35-35b)

# compare options (wraps scripts/dl/showcase_prefix_sharing.py — sglang vs vLLM
# RadixAttention/APC showcase; the one-click gap tracker).
#   --scenarios SC1,SC2,SC3[,SC4]   which showcases to run (SC1 cold is ~90s).
#   --only sglang|vllm              re-run just one side, diff vs cached other.
# Both engines run on the SAME GPUs sequentially (fresh process each), metrics
# cached per engine so `--only` re-measures one side cheaply against the last run
# of the other. FAIR baseline: vLLM MRV1 + CG + APC ON (prefix cache ON) — the only
# vLLM config that enables prefix caching on this hybrid-Mamba model (MRV2 rejects
# mamba_cache_mode='align'). sglang uses the GDN dl_chunk flag (default-on) +
# RadixAttention. With this flag sglang WINS most scenarios; without it sglang loses
# (old r009). See docs/dl/sglang-beats-vllm-dlin-technical-report.zh.md.
COMPARE_SCENARIOS="${COMPARE_SCENARIOS:-SC1,SC2,SC3}"
COMPARE_ONLY="${COMPARE_ONLY:-}"                # --only sglang|vllm-mrv2|vllm-mrv1
COMPARE_SHOW="${COMPARE_SHOW:-0}"               # --show: re-render table from cache, no GPU run
COMPARE_HISTORY="${COMPARE_HISTORY:-0}"         # --history: print the results log, no GPU run
COMPARE_LIST="${COMPARE_LIST:-0}"               # --list: print all scenarios + ASCII diagrams, no GPU run
COMPARE_LIST_ZH="${COMPARE_LIST_ZH:-0}"         # --zh: Chinese version of --list
COMPARE_RECORD="${COMPARE_RECORD:-1}"           # --no-record: don't append to the JSON store
COMPARE_RUN_MRV1="${COMPARE_RUN_MRV1:-1}"       # DL: MRV1+CG+APC = the FAIR baseline (default on). +~10min (528 capture)
COMPARE_BASELINE="${COMPARE_BASELINE:-}"        # --baseline <id|commit>: diff vs this (else previous run)
COMPARE_VERBOSE="${COMPARE_VERBOSE:-0}"         # --verbose: print engine commands
COMPARE_MEM_FRAC="${COMPARE_MEM_FRAC:-0.55}"
COMPARE_METRICS_DIR="${COMPARE_METRICS_DIR:-/tmp/sglang_compare}"
COMPARE_STORE="${COMPARE_STORE:-$SGLANG_DIR/docs/dl/compare_results.json}"

# compare --correctness: sglang vs vLLM GREEDY (temperature=0) output-equality
# test (wraps scripts/dl/compare_correctness.py). NOT a perf bench — runs the
# same prompt(s) on BOTH engines and reports whether the greedy text/token-ids
# are bit-identical, with the first diverging token. Model-agnostic; built for
# DeepSeek-V4-Flash (`-M dsv4-flash`). Each engine runs in a FRESH process.
COMPARE_CORRECTNESS="${COMPARE_CORRECTNESS:-0}"       # --correctness
COMPARE_VLLM_PYTHON="${COMPARE_VLLM_PYTHON:-}"        # --vllm-python PATH: vLLM venv if .venv lacks vllm
COMPARE_MAX_MODEL_LEN="${COMPARE_MAX_MODEL_LEN:-2048}"  # --max-model-len

# chat options (wraps scripts/dl/chat.py — interactive OpenAI-compatible client).
# Default: connect to $SERVE_HOST:$SERVE_PORT, auto-detect model.
CHAT_QUICK="${CHAT_QUICK:-}"                        # -q: single message
CHAT_URL="${CHAT_URL:-}"                            # --url: server API base URL
CHAT_MODEL="${CHAT_MODEL:-}"                        # --chat-model: explicit model name
CHAT_SYSTEM_PROMPT="${CHAT_SYSTEM_PROMPT:-}"        # --system-prompt
CHAT_NO_STREAM="${CHAT_NO_STREAM:-0}"                # --no-stream

# sop options (wraps scripts/dl/sop_verify.py — the DLIN UPGRADE verification
# standard). The judging bar for porting DLIN changes onto a new sglang tag
# (dl-dev-v0.5.15 / dl-dev-v0.5.16). Runs correctness gates (absolute: DLIN stack
# smoke, canonical probes, greedy determinism, no-gibberish, JSON) + perf gates
# (decode/prefill tok/s within a tolerance band of a recorded baseline) and emits
# a single PASS/FAIL verdict + JSON report. Default model = Qwen3-1.7B (fast, ~1-
# 2 min incl JIT); -M qwen35-35b = full 35B TP4 gate (~10+ min). Use `record` on
# the known-good dl-main to capture the baseline golden+perf, then `verify` on the
# ported branch diffs against it. See scripts/dl/sop_verify.py for the gate list.
SOP_MODE="${SOP_MODE:-verify}"                     # record|verify|show
SOP_QUICK="${SOP_QUICK:-0}"                         # --quick: correctness only (skip perf)
SOP_BASELINE="${SOP_BASELINE:-}"                    # --sop-baseline PATH (else docs/dl/sop_baseline_<model>.json)
SOP_MAX_NEW="${SOP_MAX_NEW:-64}"                    # decode length for the perf gate

log()  { echo -e "\033[1;34m[run_sglang]\033[0m $*"; }
warn() { echo -e "\033[1;33m[run_sglang WARN]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[run_sglang OK]\033[0m $*"; }
die()  { echo -e "\033[1;31m[run_sglang ERROR]\033[0m $*"; exit 1; }

#-------------------------------------------------------------------------------
# pick_model — resolve a -M preset name to model path + optimized env.
#-------------------------------------------------------------------------------
pick_model() {
  case "$1" in
    qwen3-1.7b)
      MODEL_PATH="/opt/dataset/Qwen3-1.7B"
      ;;
    qwen35-35b)
      # Local copy of the Qwen3.5/3.6-35B-A3B hybrid-Mamba MoE FP8 model
      # (qwen3_5_moe, 256 experts, top-8). Local storage loads far faster than
      # the /mars network copy. Override with -m /mars/.../Qwen3.5-35B-A3B-FP8/
      # if you need the original.
      MODEL_PATH="/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
      DLIN_TP_SIZE=4; USE_CUDA_GRAPH=1; DLIN_CG_MAX_BS=2
      # mem_fraction=0.60 (not 0.85): the stable bf16-bmm prefill path (M>16) dequants
      # expert weights and needs several GB headroom, else OOM (docs §7.14). 0.85 OOMs.
      DLIN_MEM_FRACTION=0.60; DLIN_CONTEXT_LEN=4096; DLIN_PAGE_SIZE=16
      # DLIN MoE routing (see fp8.py + docs §7.13/§7.14):
      #   FUSED_MAX_M=2048 — fused (invoke_fused_moe_opt) for decode + ALL prefill
      #                     (serve/gen default chunked_prefill_size=2048 on 32GB KS38;
      #                     compare uses 512). Verified 2026-07-23: the prior "CRASHES
      #                     (dleol tu_program.cc:625) for M>=~100" note was STALE — the
      #                     dleol issue was fixed since; invoke_fused_moe_opt runs clean
      #                     at M=2048 with output bit-identical to bf16-bmm (math 17x23=391,
      #                     sc3-style "6144 TFLOPS" both paths). Raising 16->2048 makes
      #                     prefill use the fast fused path instead of slow bf16-bmm:
      #                     compare SC3 6.0->20.2 tok/s, SC1/SC2 prefill 2.5-5x faster.
      #   MAX_BF16_M=2048 — bf16-bmm fallback only for M>2048 (shouldn't occur w/ chunk<=2048).
      export SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=2048 SGLANG_DL_MOE_MAX_BF16_M=2048
      # DL: route prefill MoE (M>1) through invoke_fused_moe_opt_v3 (vLLM's fast kernel).
      # sglang's sgl_kernel port only ships the non-v3 invoke_fused_moe_opt, which is
      # 3.7x slower per-token. With chunked_prefill_size=2048 (1 chunk for 2K prefill),
      # v3 at BM=128 matches vLLM's 9.16ms/GEMM: 2K cold-prefill 396->1565 tok/s, now
      # BEATING vLLM (1457). BM auto-selects by M (fp8.py). Uses vLLM's moe_align_block_size
      # (sglang's port OOB-reads at M>=~100). 2026-07-29.
      export SGLANG_DL_MOE_V3=1
      # Matches the tuned TP4 config in scripts/dl/compare_tp4.py (~27ms TPOT = ~vLLM parity):
      # FP8 Q2 GEMM, DLIN GDN op, multi-step decode, FLA pingpong/unroll.
      export SGLANG_DL_FP8_Q2=1 SGLANG_DL_GDN_DLIN=1 SGLANG_DL_MULTI_STEP=1
      # DL: route GDN prefill(extend) to the DLIN dl_chunk kernel (8x faster prefill).
      # Default triton extend = 89% of prefill time (self_attention/GDN; MoE only 11%).
      # SGLANG_DL_GDN_DLIN_EXTEND=1 -> 2K prefill 41s->5.1s, correct, and sglang BEATS
      # vLLM on prefill-heavy scenarios (SC1-warm 1.03s<1.2s, SC3 61.5>41.9, SC7/8/10
      # 1.13-1.26x). gdn_backend.py:76-93. 2026-07-28.
      # GOTCHA (dl_chunk triton-cache corruption): if a sglang run crashes mid-way
      # (NCCL collective-timeout desync / SIGSEGV), it can leave a CORRUPT dl_chunk
      # cache entry in ~/.triton/cache -> every subsequent run crashes. FIX: restore the
      # known-good cache:  rm -rf ~/.triton/cache && cp -a ~/.triton/cache.good_backup ~/.triton/cache
      export SGLANG_DL_GDN_DLIN_EXTEND=1
      export DLEOL_CU_ADDRESS_CHECK=0 DLEOL_FLA_ENABLE_PINGPONG=1 DLEOL_FLA_UNROLL_COUNT=8
      # TP=4 needs 4 GPUs; ensure CUDA_VISIBLE_DEVICES lists >=4 devices.
      local _ndev
      _ndev=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
      if [ "$_ndev" -lt 4 ]; then
        CUDA_VISIBLE_DEVICES="0,1,2,3"
        log "TP=4: set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (override with CUDA_VISIBLE_DEVICES=...)"
      fi
      ;;
    dsv4-flash)
      # DeepSeek-V4-Flash 149B (deepseek_v4: MHC+MLA hybrid attn, 256-expert MoE
      # top-6, FP8 weights, 1 NextN/MTP layer). Needs TP8 on 32GB KS38.
      # Source: docs/dl-sglang-usage-guide.zh.md §4.2/§8.2.
      MODEL_PATH="/LocalRun/hao.dong/DeepSeek-V4-Flash"
      DLIN_TP_SIZE=8; DLIN_MEM_FRACTION=0.90; DLIN_CONTEXT_LEN=2048; DLIN_PAGE_SIZE=16
      USE_CUDA_GRAPH=1; DLIN_CG_MAX_BS=1
      # V4-Flash MANDATORY env block (usage guide §8.1): the 7 topk/MHC flags
      # default the WRONG way in code (EnvBool True/False) -> must be exported
      # verbatim before sgl.Engine is built, else crash or silent all-zero output.
      export SGLANG_DL_MOE_FUSED=1 SGLANG_DL_FP8_Q2=1 SGLANG_DL_GDN_DLIN=1 SGLANG_DL_MOE_FUSED_MAX_M=2048
      export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_TOPK_TRANSFORM_512_TORCH=1
      export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_FUSED_HASH_TOPK=0 SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0
      export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
      export SGLANG_DL_IDX_TRITON=1 TORCHDYNAMO_DISABLE=1
      export DLEOL_FLA_ENABLE_PINGPONG=1 DLEOL_FLA_UNROLL_COUNT=8
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      # TP=8 needs 8 GPUs; ensure CUDA_VISIBLE_DEVICES lists >=8 devices.
      local _ndev
      _ndev=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
      if [ "$_ndev" -lt 8 ]; then
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        log "TP=8: set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (override with CUDA_VISIBLE_DEVICES=...)"
      fi
      ;;
    *) die "unknown preset '$1'. Available: qwen3-1.7b qwen35-35b dsv4-flash" ;;
  esac
  log "preset '$1': model=$MODEL_PATH tp=${DLIN_TP_SIZE:-1} cg=$USE_CUDA_GRAPH"
}

#-------------------------------------------------------------------------------
# apply_ngram_overrides — when -S/--spec-ngram is set, force the config that
# makes NGRAM num_draft=8 hit 2.8-3.2x vLLM (docs §7.12): fused MoE covering the
# verify batch (M≈num_draft+1), CG on, and lower mem_fraction for the draft tree.
# Run AFTER pick_model so it overrides the preset's mem/cg.
#-------------------------------------------------------------------------------
apply_ngram_overrides() {
  [ "$USE_NGRAM" = "1" ] || return 0
  USE_CUDA_GRAPH=1                                  # NGRAM needs CG for fast decode
  export SGLANG_DL_MOE_FUSED=1
  export SGLANG_DL_MOE_FUSED_MAX_M="${SGLANG_DL_MOE_FUSED_MAX_M:-16}"   # covers verify M=9
  export SGLANG_DL_MOE_MAX_BF16_M="${SGLANG_DL_MOE_MAX_BF16_M:-2048}"
  DLIN_MEM_FRACTION=0.60                            # was 0.85; draft tree needs headroom
  DLIN_CG_MAX_BS="${DLIN_CG_MAX_BS:-8}"
  log "NGRAM spec-decode ON: num_draft=$NGRAM_NUM_DRAFT bfs=$NGRAM_MIN_BFS-$NGRAM_MAX_BFS | " \
      "fused_max_m=$SGLANG_DL_MOE_FUSED_MAX_M mem=$DLIN_MEM_FRACTION cg_max_bs=$DLIN_CG_MAX_BS"
  log "  (on 2x32GB, CG capture of bs=7-8 may log OOM — sglang auto-recovers; decode/verify" \
      "still hit 35-40 tok/s. Use TP=4 or CG_MAX_BS=4 to silence.)"
}

#-------------------------------------------------------------------------------
# Artifactory download — fetch files from the DLIN Artifactory host.
#
# Method ($DL_DOWNLOAD_METHOD, default "jfrog"):
#   jfrog -> `jfrog rt download` using the configured jfrog CLI (server + access
#            token from `jfrog config`). No password on the command line. If the
#            CLI is absent / has no configured server, dl_download auto-falls
#            back to the wget path. This is the default.
#   wget  -> wget with basic auth. External-net access needs --user/--password,
#            prompted once and cached in ~/.netrc (chmod 600) so later runs are
#            silent. Use this if you have NOT configured the jfrog CLI.
#
# Override the jfrog server-id with JFROG_SERVER_ID (else the first configured
# server is auto-selected). pip/uv always read ~/.netrc natively regardless.
#-------------------------------------------------------------------------------
# Which configured jfrog server-id to use (JFROG_SERVER_ID override else first).
_dl_jfrog_server_id() {
  if [ -n "${JFROG_SERVER_ID:-}" ]; then echo "$JFROG_SERVER_ID"; return 0; fi
  local sid
  # `|| true`: tolerate `jfrog config show` failing (CLI present but unconfigured)
  # so pipefail never aborts the script — an empty result just means "no server".
  sid="$(jfrog config show 2>/dev/null | awk '/^[[:space:]]*Server ID:/ {print $3; exit}')" || true
  [ -n "$sid" ] && echo "$sid"
}

# 0 if the jfrog CLI is present AND has at least one configured server.
_dl_jfrog_configured() {
  command -v jfrog >/dev/null 2>&1 || return 1
  [ -n "$(_dl_jfrog_server_id)" ] || return 1
}

# ensure_jfrog_credentials — make sure ~/.netrc has an entry for $DL_HOST.
# Entry already present: silent (caller uses --netrc).
# Missing: prompt for username/password (external-net basic auth), store to
# ~/.netrc (chmod 600), and export DL_WGET_USER/DL_WGET_PASS so the caller can
# pass --user/--password on this first run.
ensure_jfrog_credentials() {
  local host="$DL_HOST"
  local netrc="$HOME/.netrc"
  DL_WGET_USER=""; DL_WGET_PASS=""

  if grep -q "machine ${host}" "$netrc" 2>/dev/null; then
    return 0   # cached: caller should use --netrc
  fi

  log "JFrog credentials needed for $host (stored in ~/.netrc, chmod 600)"
  printf "  JFrog username: "; read -r DL_WGET_USER
  printf "  JFrog password: "; read -rs DL_WGET_PASS; echo

  if [ -z "$DL_WGET_USER" ] || [ -z "$DL_WGET_PASS" ]; then
    die "Username and password are required"
  fi

  # Append entry (create file if missing)
  printf "\nmachine %s\n  login %s\n  password %s\n" \
    "$host" "$DL_WGET_USER" "$DL_WGET_PASS" >> "$netrc"
  chmod 600 "$netrc"
  export DL_WGET_USER DL_WGET_PASS
  log "Credentials saved to ~/.netrc (chmod 600). Will be reused automatically."
}

# dl_download <dest> <url> — fetch one Artifactory file per $DL_DOWNLOAD_METHOD.
dl_download() {
  local dest="$1" url="$2"
  local method="${DL_DOWNLOAD_METHOD:-jfrog}"

  if [ "$method" = "jfrog" ]; then
    if _dl_jfrog_configured; then
      if _dl_download_jfrog "$dest" "$url"; then return 0; fi
      warn "jfrog rt download failed (log: /tmp/jfrog_dl_error.$$.log); falling back to wget."
    else
      log "jfrog CLI not configured -> using wget (--user/--password)"
    fi
  fi
  _dl_download_wget "$dest" "$url"
}

# _dl_download_jfrog <dest> <url> — jfrog rt download via the configured server.
_dl_download_jfrog() {  # returns 0 on success, 1 on any failure (caller falls back)
  local dest="$1" url="$2" sid repo base tmpd errlog
  sid="$(_dl_jfrog_server_id)" || return 1
  [ -n "$sid" ] || return 1
  # Full URL -> Artifactory repo path (strip the artifactory root).
  case "$url" in
    "$DL_ARTIFACTORY_ROOT"/*) repo="${url#"$DL_ARTIFACTORY_ROOT"/}" ;;
    */artifactory/*)          repo="${url#*artifactory/}" ;;
    *)                        repo="$url" ;;
  esac
  base="$(basename "$repo")"
  tmpd="$(mktemp -d)"; errlog="$tmpd/jfrog.log"
  log "jfrog rt download --server-id=$sid : $repo"
  # Capture output: on auth failure jfrog retries ~4x (noisy) then returns 0
  # anyway, so the [ -f ] check is authoritative. Keep the log for diagnosis.
  if jfrog rt download --flat --server-id="$sid" "$repo" "$tmpd/" >"$errlog" 2>&1 \
     && [ -f "$tmpd/$base" ]; then
    mkdir -p "$(dirname "$dest")"
    mv "$tmpd/$base" "$dest"
    rm -rf "$tmpd"
    return 0
  fi
  mv "$errlog" "/tmp/jfrog_dl_error.$$.log" 2>/dev/null || true
  rm -rf "$tmpd"
  return 1
}

# _dl_download_wget <dest> <url> — wget basic auth (prompt once, cache ~/.netrc).
_dl_download_wget() {
  local dest="$1" url="$2"
  ensure_jfrog_credentials
  mkdir -p "$(dirname "$dest")"
  local auth_flag=(--netrc)
  # First-run (just prompted): pass --user/--password explicitly; repeat runs
  # read ~/.netrc (silent + no password in the process list).
  if [ -n "${DL_WGET_USER:-}" ] && [ -n "${DL_WGET_PASS:-}" ]; then
    auth_flag=(--user="$DL_WGET_USER" --password="$DL_WGET_PASS")
  fi
  wget "${auth_flag[@]}" -q --show-progress -O "$dest" "$url" \
    || die "Failed to download $url via wget"
}

#-------------------------------------------------------------------------------
activate_uv_indexes() {
  # Primary index = dl-pypi-remote (PyPI proxy, used by artifactory/uv.toml).
  # Extra index  = dl-virtual (DLIN-patched wheels: torch +dl24, ...).
  export UV_INDEX_URL="$DL_PYPI_INDEX"
  export UV_EXTRA_INDEX_URL="$DL_VIRTUAL_INDEX"
  export UV_ALLOW_INSECURE_HOST="$DL_HOST"
  export PIP_INDEX_URL="$DL_PYPI_INDEX"
  export PIP_EXTRA_INDEX_URL="$DL_VIRTUAL_INDEX"
  export PIP_TRUSTED_HOST="$DL_HOST"
}

#-------------------------------------------------------------------------------
# dl_patch_tvm_ffi — keep tvm_ffi's C++-extension builder DLIN-aware.
#
# WHY THIS EXISTS: sglang's jit_kernel runtime JIT (load_jit ->
# tvm_ffi.cpp.load_inline -> _build_impl -> _generate_ninja_build) compiles CUDA
# via tvm_ffi's OWN ninja builder. That builder is INDEPENDENT of torch — it has
# zero references to torch.utils.cpp_extension — so DLIN torch's auto-dlcc routing
# (which covers sgl-kernel's AOT build via setup_dl.py) does NOT reach it. The
# official apache-tvm-ffi cpp/extension.py hardcodes nvcc / -lcudart / lib64 /
# nvcc-dependency flags, none of which exist on DLIN (dlcc / -lcurt / lib /
# -MMD). The patch (scripts/dl/tvm_ffi_cpp_extension_dl.patch) swaps those 4
# spots so jit_kernel kernels JIT-compile with dlcc for the dlgput64 arch.
#
# Idempotent and cheap: it only re-applies when the installed file is the
# unpatched official one (e.g. after `uv pip install` reinstalled apache-tvm-ffi,
# which is unpinned in python/pyproject_dl.toml — a reinstall silently clobbers
# the patch without this). Version-guarded to 0.1.12 (the version the patch was
# cut against); any other version is left untouched with a warning so the patch
# can be regenerated rather than blindly corrupting a new release.
#-------------------------------------------------------------------------------
dl_patch_tvm_ffi() {
  local patch_file="$SGLANG_DIR/scripts/dl/tvm_ffi_cpp_extension_dl.patch"
  [ -f "$patch_file" ] || { warn "tvm_ffi DL patch asset missing: $patch_file"; return 0; }
  command -v python >/dev/null 2>&1 || return 0

  # Locate the installed file, its site-packages dir, version, and whether it's
  # already patched �� in one python call. NB: locate via find_spec WITHOUT
  # importing tvm_ffi — tvm_ffi/__init__.py imports torch, which needs the DLIN
  # SDK on LD_LIBRARY_PATH (not always set yet, e.g. during phase_install).
  # find_spec + metadata.version avoid that import entirely. Silent no-op if
  # tvm_ffi isn't installed yet.
  local probe ext_py sp ver patched
  probe="$(python - <<'PY' 2>/dev/null
import importlib.util, os
from importlib.metadata import version, PackageNotFoundError
spec = importlib.util.find_spec("tvm_ffi")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit
pkg_dir = spec.submodule_search_locations[0]
ext_py = os.path.join(pkg_dir, "cpp", "extension.py")
if not os.path.isfile(ext_py):
    raise SystemExit
sp = os.path.dirname(pkg_dir)
try:
    ver = version("apache-tvm-ffi")
except PackageNotFoundError:
    ver = "0.0.0"
patched = "--cuda-gpu-arch=dlgput64" in open(ext_py).read()
print(ext_py)
print(sp)
print(ver)
print(1 if patched else 0)
PY
)" || return 0
  { read -r ext_py; read -r sp; read -r ver; read -r patched; } <<< "$probe"
  [ -n "$ext_py" ] && [ -f "$ext_py" ] || return 0

  # Already patched -> silent (the common, idempotent case after first setup).
  [ "$patched" = "1" ] && return 0

  # Version guard: the patch was cut against 0.1.12.
  if [ "$ver" != "0.1.12" ]; then
    warn "apache-tvm-ffi=$ver: DL dlcc patch targets 0.1.12 — skipped. jit_kernel JIT may fail on DLIN until the patch is regenerated."
    return 0
  fi

  log "Patching tvm_ffi ($ver) for DLIN dlcc/libcurt/dlgput64 ..."
  if ( cd "$sp" && patch -p1 --no-backup-if-mismatch --forward < "$patch_file" ) >/dev/null 2>&1 \
     && grep -q -- "--cuda-gpu-arch=dlgput64" "$ext_py"; then
    ok "tvm_ffi patched for DLIN dlcc (jit_kernel JIT now uses libcurt/dlgput64)."
  else
    warn "tvm_ffi DL patch did not apply cleanly; jit_kernel JIT may fail. Inspect: $ext_py"
  fi
}

#-------------------------------------------------------------------------------
# DLIN runtime environment for ANY process that imports torch / runs GPU code.
#
# CRITICAL: LD_LIBRARY_PATH must contain ONLY the DLIN SDK lib dir. If it also
# holds another SDK's lib (e.g. a stale sdk-0401 from a prior session), two
# copies of libhcrt/libLLVM/libcurt get loaded and the DLIN JIT crashes inside
# an LLVM PassBuilder static initializer (_GLOBAL__sub_I_PassBuilder.cpp) the
# first time a kernel is compiled. So we OVERWRITE LD_LIBRARY_PATH to a single
# deterministic value rather than prepending.
#-------------------------------------------------------------------------------
dlin_runtime_env() {
  export CUDA_HOME="$SDK_DIR"
  export CPATH="$SDK_DIR/include:${CPATH:-}"
  # DLI_V2=ON is the DLIN SDK's own activation flag (env.sh exports it too);
  # set it here so the runtime env is deterministic regardless of how the
  # script was launched.
  export DLI_V2=ON
  # OVERWRITE LD_LIBRARY_PATH: only the active SDK's runtime libs may appear,
  # else a stale second SDK's libhcrt/libLLVM can load and the DLIN JIT crashes
  # inside an LLVM PassBuilder static initializer on first kernel compile.
  export LD_LIBRARY_PATH="$SDK_DIR/lib"
  # DL: DLEOL JIT cache — default is too small, causing kernel eviction + 140×
  # recompilation overhead (15.5ms/call → 0.11ms/call). ALSO fixes garbage output
  # (recompilation artifacts corrupted intermediate tensors). VERIFIED: 67.9ms TPOT
  # CG TP2, "Paris." correct — beats vLLM 70.3ms.
  export DLEOL_CACHE_SIZE="${DLEOL_CACHE_SIZE:-1024}"
  export DLEOL_CACHE_GRAPH_SIZE="${DLEOL_CACHE_GRAPH_SIZE:-1024}"
  # DL: Disable torch.compile/dynamo/inductor — torchinductor's own codegen
  # may still emit incompatible code even with DLIN triton on the path.
  export TORCHDYNAMO_DISABLE=1
  # Prepend (do not overwrite) so the venv bin and SDK bin stay on PATH.
  export PATH="$VENV_DIR/bin:$SDK_DIR/bin:$SDK_DIR/tools:/usr/bin:/bin:${HOME:-}/.local/bin"
  : "${CUDA_VISIBLE_DEVICES:=0}"
  export CUDA_VISIBLE_DEVICES
  # Re-apply the DLIN dlcc patch to tvm_ffi if a reinstall clobbered it. Cheap +
  # idempotent (one python probe; silent when already patched). See dl_patch_tvm_ffi.
  dl_patch_tvm_ffi
}

#-------------------------------------------------------------------------------
# Phase: setup
#-------------------------------------------------------------------------------
phase_setup() {
  log "Phase [setup]: DLIN SDK + uv venv + DLIN torch"

  [ -f "$SDK_DIR/env.sh" ] || die "SDK env.sh not found at $SDK_DIR"
  # shellcheck disable=SC1091
  source "$SDK_DIR/env.sh"
  export CUDA_HOME="${CUDA_HOME:-$SDK_DIR}"
  export CPATH="${SDK_DIR}/include:${CPATH:-}"
  export LD_LIBRARY_PATH="${SDK_DIR}/lib:${LD_LIBRARY_PATH:-}"
  command -v dlcc >/dev/null 2>&1 || die "dlcc not on PATH after sourcing SDK env.sh"
  log "SDK activated: SDK_DIR=$SDK_DIR  CUDA_HOME=$CUDA_HOME  dlcc=$(command -v dlcc)"

  command -v uv >/dev/null 2>&1 || die "uv not found (install: pip install uv or see dl-env skill)"
  if [ ! -d "$VENV_DIR" ]; then
    log "Creating uv venv ($PYTHON_VERSION) at $VENV_DIR ..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --seed
  else
    ok "venv already exists: $VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  uv pip install --upgrade pip wheel setuptools

  activate_uv_indexes
  log "Installing DLIN torch ($TORCH_SPEC) ..."
  uv pip install "$TORCH_SPEC"

  # DL begin
  # Install DLIN triton 3.3.0 (dlgpu backend) BEFORE the editable install
  # so uv doesn't resolve a vanilla build from dl-pypi-remote and so torch's
  # triton dependency is already satisfied by the correct version.
  local triton_whl="$SGLANG_DIR/whl-for-compare/$(basename "${TRITON_URL//%2B/+}")"
  if [ ! -f "$triton_whl" ]; then
    log "Downloading DLIN triton 3.3.0 (dlgpu backend) [method=${DL_DOWNLOAD_METHOD:-jfrog}] ..."
    dl_download "$triton_whl" "${TRITON_URL//%2B/+}"
  fi
  log "Installing DLIN triton 3.3.0 from $triton_whl ..."
  uv pip install --no-deps "$triton_whl"
  # DL end

  # Sanity: confirm this is the DLIN-patched torch (torch.version.dl set),
  # using a CLEAN LD_LIBRARY_PATH (see dlin_runtime_env comment).
  dlin_runtime_env
  python - <<'PY' || die "torch.version.dl is not set — wrong torch wheel (got upstream vanilla?)"
import torch
assert getattr(torch.version, "dl", None), "torch.version.dl is None"
print("torch:", torch.__version__, "| torch.version.dl:", torch.version.dl)
PY
  ok "DLIN torch installed."

  log "Installing build/runtime deps ..."
  uv pip install ninja packaging cmake "setuptools-scm>=8" wheel jinja2 \
      "scikit-build-core>=0.10" einops numpy scipy pyyaml tqdm
  ok "Phase [setup] done."
}

#-------------------------------------------------------------------------------
# Phase: build-kernel (dlcc, via setup_dl.py)
#   Builds sgl-kernel's common_ops extension with dlcc using the per-backend
#   setup_dl.py (torch CUDAExtension — torch's dl-aware cpp_extension drives
#   dlcc + --cuda-gpu-arch=dlgput64 automatically). The default CMake/scikit-build
#   path is NOT used: CMake's enable_language(CUDA) can't see dlcc, and the
#   CMakeLists hardcode NVIDIA gencode.
#   Source set is a flashinfer/CUTLASS/libcudacxx-free subset; coverage grows
#   as those headers are fetched in later phases (see plan §5).
#-------------------------------------------------------------------------------
phase_build_kernel() {
  log "Phase [build-kernel]: sgl-kernel common_ops via dlcc (setup_dl.py)"
  if [ "${SKIP_KERNEL:-0}" = "1" ]; then warn "SKIP_KERNEL=1 -> skipping"; return 0; fi
  [ -n "${VIRTUAL_ENV:-}" ] || die "run 'setup' first (venv not active)"
  dlin_runtime_env   # CUDA_HOME=$SDK, PATH includes $SDK/bin (so nvcc->dlcc wrapper works)

  pushd "$SGLANG_DIR/sgl-kernel" >/dev/null
  # DL begin
  # Swap in the DLIN pyproject (setuptools backend, package discovery) so the
  # editable install does NOT trigger the default scikit-build/CMake path.
  local py py_dl bak
  py=pyproject.toml; py_dl=pyproject_dl.toml; bak=pyproject.toml.cuda-bak
  [ -f "$py_dl" ] || die "missing $py_dl"
  # Refresh the CUDA backup every run (see phase_install) — never let it go
  # stale, never snapshot the DL variant as the "CUDA" backup.
  if [ ! -f "$bak" ] || ! diff -q "$py" "$py_dl" >/dev/null 2>&1; then cp "$py" "$bak"; fi
  cp "$py_dl" "$py"
  # Install the python wrapper package (editable), then build the dlcc .so in place.
  uv pip install -e . --no-build-isolation
  cp "$bak" "$py"   # restore original CUDA pyproject
  if python setup_dl.py build_ext --inplace; then
    # Editable install points at python/sgl_kernel/, where build_ext --inplace
    # dropped the .so, so `import sgl_kernel` already sees it.
    ok "sgl-kernel common_ops built (dlcc) and installed (editable)."
  else
    warn "sgl-kernel build did not complete under dlcc. See plan §5 for the header-dep roadmap."
  fi
  # DL end
  popd >/dev/null
}

#-------------------------------------------------------------------------------
# Phase: install sglang (editable, DLIN pyproject) — non-destructive swap.
#-------------------------------------------------------------------------------
phase_install() {
  log "Phase [install]: editable sglang with python/pyproject_dl.toml"
  [ -n "${VIRTUAL_ENV:-}" ] || die "run 'setup' first (venv not active)"
  activate_uv_indexes
  local py="$SGLANG_DIR/python/pyproject.toml"
  local dl="$SGLANG_DIR/python/pyproject_dl.toml"
  local bak="$SGLANG_DIR/python/pyproject.toml.cuda-bak"
  [ -f "$dl" ] || die "missing $dl"
  # vcs-versioning registers a setuptools_scm plugin incompatible with
  # setuptools-scm>=8.0 (config.scm attribute missing). Remove it if present.
  uv pip uninstall vcs-versioning --yes >/dev/null 2>&1 || true
  # Back up the CUDA pyproject, then swap in the DLIN variant. Refresh the
  # backup EVERY run (the old "back up once" logic froze it at first run and it
  # went stale as HEAD advanced — install would restore a months-old pyproject,
  # leaving the working tree perpetually 'M'). Skip the refresh only when the
  # working tree currently holds the DL variant (left over from a crashed prior
  # swap), so the DL file is never snapshotted as the "CUDA" backup.
  if [ ! -f "$bak" ] || ! diff -q "$py" "$dl" >/dev/null 2>&1; then cp "$py" "$bak"; fi
  cp "$dl" "$py"
  trap 'cp "$bak" "$py"; warn "restored original pyproject.toml after error"' ERR

  pushd "$SGLANG_DIR/python" >/dev/null
  log "uv pip install -e . (DLIN pyproject; extras: ${SGLANG_EXTRAS:-srt_dl})"
  uv pip install -e ".[${SGLANG_EXTRAS:-srt_dl}]" --no-build-isolation
  popd >/dev/null

  # Restore the original CUDA pyproject (editable install already baked metadata).
  cp "$bak" "$py"
  trap - ERR
  ok "Phase [install] done (original pyproject.toml restored)."
  # The editable install pulls apache-tvm-ffi (a sglang dep). Re-apply the DLIN
  # dlcc patch now so jit_kernel JIT works (idempotent — no-op if already patched).
  dl_patch_tvm_ffi
}

#-------------------------------------------------------------------------------
# Phase: test — DLIN stack basic UT.
#
# Gating checks (must pass):
#  1. DLIN torch detected + a real DLIN GPU compute (bf16 matmul) succeeds.
#  2. `import sglang` works and current_platform resolves to DlinSRTPlatform.
# Both require dlin_runtime_env: a CLEAN LD_LIBRARY_PATH (single SDK/lib) AND
# DLIN triton 3.3.0 (dlgpu backend, compiles @triton.jit to dlgput64 format).
#
# sgl-kernel is best-effort (build-kernel phase); absent → torch fallbacks.
#-------------------------------------------------------------------------------
phase_test() {
  log "Phase [test]: basic UT (DLIN stack smoke)"
  [ -n "${VIRTUAL_ENV:-}" ] || { source "$VENV_DIR/bin/activate" || die "run 'setup' first"; }
  dlin_runtime_env

  log "1) gating: DLIN torch + torch.version.dl + DLIN GPU bf16 matmul"
  python - <<'PY'
import torch
dl = getattr(torch.version, "dl", None)
assert dl, f"torch.version.dl not set (torch={torch.__version__}); wrong wheel"
assert torch.cuda.is_available(), "torch.cuda not available"
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
print(f"  torch={torch.__version__} dl={dl} | gpu={torch.cuda.get_device_name(0)} "
      f"| bf16 matmul OK sum={c.float().sum().item():.1f}")
PY
  ok "DLIN stack smoke PASSED (torch+dl + GPU compute)."

  log "2) sglang installed (version via metadata, avoids native-import segfaults):"
  python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
try:
    print("  sglang =", version("sglang"))
except PackageNotFoundError:
    print("  sglang: NOT installed")
try:
    print("  sgl_kernel =", version("sglang-kernel"))
except PackageNotFoundError:
    print("  sgl_kernel: NOT installed (build-kernel phase)")
PY

  log "3) gating: full 'import sglang' + DLIN platform resolves"
  python - <<'PY' || die "import sglang / platform resolution failed"
import sglang
from sglang.srt.platforms import current_platform
from sglang.srt.utils.common import is_dlin
assert is_dlin(), "is_dlin() is False"
assert current_platform.is_dlin(), "current_platform.is_dlin() is False"
assert type(current_platform).__name__ == "DlinSRTPlatform"
print(f"  import sglang OK: {sglang.__version__}")
print(f"  current_platform = {type(current_platform).__name__} "
      f"(is_dlin={current_platform.is_dlin()}, is_cuda={current_platform.is_cuda()}, "
      f"cc={current_platform.get_device_capability(0)})")
PY
  ok "import sglang + DlinSRTPlatform resolved."
  ok "Phase [test] done."
}

#-------------------------------------------------------------------------------
# Usage + arg parsing for the quick-test phases (gen/serve).
#-------------------------------------------------------------------------------
usage() {
  cat <<'EOF'
run_sglang.sh — build/run SGLang on DLIN (DLIN) GPUs.

Usage:
  ./run_sglang.sh                       # all phases: setup -> build-kernel -> install -> test
  ./run_sglang.sh setup                 # create/refresh the DLIN uv venv + torch
  ./run_sglang.sh build-kernel          # build sgl-kernel with dlcc (best-effort)
  ./run_sglang.sh install               # editable-install sglang (DLIN pyproject)
  ./run_sglang.sh test | smoke          # import + GPU smoke (torch.version.dl, matmul, platform)

One-click run Qwen3.5-35B-A3B-FP8 (TP4) — the DEFAULT model for gen/serve:
  ./run_sglang.sh serve                 # OpenAI-compatible HTTP server on :30000
  ./run_sglang.sh gen                   # one-shot generation (prints text + tok/s)
  ./run_sglang.sh gen -m <model> -p "prompt" -n 32 -b fa3   # override model/prompt
  ./run_sglang.sh serve -M qwen3-1.7b   # any other model via -M/-m
  ./run_sglang.sh bench -c 16           # concurrent bench (auto-starts a server)
  ./run_sglang.sh bench --offline       # per-batch latency, no server (bench_one_batch)
  ./run_sglang.sh benchrun -M qwen35-35b --num-prompts 8   # vLLM-format bench (auto-starts a server)
  ./run_sglang.sh benchrun --template                      # write a config_serving.json to hand-edit
  ./run_sglang.sh benchrun /path/config_serving.json       # run an existing vLLM-format config

sglang vs vLLM showcase (one-click gap tracker; Qwen3.5-35B-A3B-FP8 TP4):
  ./run_sglang.sh compare --list                # list all scenarios + ASCII diagrams (no GPU run)
  ./run_sglang.sh compare                       # SC1+SC2+SC3, both engines, prints gap table
  ./run_sglang.sh compare --scenarios SC2,SC3   # skip the ~90s SC1 cold prefill
  ./run_sglang.sh compare --only sglang         # re-measure sglang only, diff vs cached vLLM
  ./run_sglang.sh compare --show                 # re-print last gap table from cache (no GPU run)
  ./run_sglang.sh compare --history              # print the commit-keyed results log (no GPU run)
  ./run_sglang.sh compare --no-record            # run but don't append to the JSON store
  ./run_sglang.sh compare --baseline r001        # diff the new run vs run r001 (else vs previous)
  ./run_sglang.sh compare --verbose              # print the exact engine commands being executed
  ./run_sglang.sh compare --list --verbose       # list scenarios + show engine commands
  # FULL showcase (all 8 scenarios, ~25-30 min; MRV1+APC adds ~10min for the 528 capture):
  ./run_sglang.sh compare --scenarios SC1,SC2,SC3,SC5,SC7,SC8,SC9,SC10
  ./run_sglang.sh chat                           # interactive chat (connect to existing server on :30000)
  ./run_sglang.sh chat -q "hello"                # quick single message
  ./run_sglang.sh chat --url http://10.0.0.1:30000/v1   # custom endpoint
  ./run_sglang.sh chat --chat-model Qwen3-1.7B --system-prompt "You are helpful"
  Scenarios: SC1 prefix-share, SC2 multi-turn, SC3 batch, SC4 JSON,
    SC5 multi-user fork (radix tree), SC7 long-RAG throughput, SC8 parallel
    sampling (best-of-N), SC9 pure long decode, SC10 shared system-prompt.
    With the GDN dl_chunk flag ON: sglang WINS most prefix-reuse scenarios
    (SC1w/3/5/7/8/10, 1.01-1.45x) AND pure decode (SC9 +5.7%); vLLM wins only
    raw/unique prefill (SC6) and one-time cold-prefill. See
    docs/dl/sglang-beats-vllm-dlin-technical-report.zh.md.
  Runs both engines on the same GPUs (fresh process each), caches metrics to
  /tmp/sglang_compare. vLLM runs MRV1+CG+APC ON (the FAIR baseline — MRV2 can't
  enable APC on this hybrid-Mamba model). sglang uses the GDN dl_chunk flag
  (default-on) + RadixAttention. Run dl_safe_reset.sh before each launch to avoid
  the dl_chunk triton-cache-corruption crash. See
  docs/dl/sglang-beats-vllm-dlin-technical-report.zh.md.

Optimized model presets (-M flag):
  -M qwen3-1.7b    Qwen3-1.7B (default, bf16)
  -M qwen35-35b    Qwen3.5-35B-A3B-FP8 (TP=4, fused MoE + multi-step, CG; ~27ms TPOT = ~vLLM) [gen/serve default]
                   add -S for NGRAM spec-decode -> 35-40 tok/s = 2.8-3.2x vLLM (docs 7.12)

  Example: ./run_sglang.sh gen -M qwen35-35b -S -n 128      # 2x+ vLLM generation
           ./run_sglang.sh serve -M qwen35-35b -S --port 30000

gen/serve options:
  -m, --model PATH          model path        (default /opt/dataset/Qwen3-1.7B)
  -p, --prompt TEXT         single prompt     (default: builtin demos)
  -n, --max-new-tokens N    decode length     (default 16)
  -b, --backend NAME        attention backend (default fa3; do not use 'triton' on DLIN)
  -g, --cuda-graph          enable cuda graph (default off; safer on DLIN)
  -S, --spec-ngram          NGRAM spec-decode (qwen35-35b: num_draft=8; needs -M qwen35-35b)
  -W, --dl-warmup           serve only: pre-compile fused-MoE dlcc kernels at startup by
                            iterating the engine's capture-size lists (vLLM-style).
                            ~20-85s/shape one-time, cached. Subset via
                            $SGLANG_DL_WARMUP_SHAPES (csv)
  --port N / --host H       serve only        (default 30000 / 127.0.0.1)
# chat options:
#   -q, --quick TEXT          single message (non-interactive)
#   --url URL                 server API base URL (default http://$SERVE_HOST:$SERVE_PORT/v1)
#   --chat-model NAME         model name (default: auto-detect from /v1/models)
#   --system-prompt TEXT      system prompt
#   --no-stream               disable streaming output

bench options (wraps sglang's official bench_serving / bench_one_batch):
  -c, --concurrency N       max concurrent requests   (default 1; see note)
  --num-prompts N           total requests            (default 64)
  --input-len/--output-len  random prompt/output lens (default 1024 / 128)
  --batch-size '1 4 8 16'   offline batch-size sweep
  --base-url URL            bench an external server (skip autostart)
  --offline                 bench_one_batch (no server; per-batch latency)
  Online bench auto-starts a server on --port if none is up, runs bench_serving
  (throughput, TTFT, ITL, p50/p99), then stops it. Result JSON: /tmp/sglang_bench_*.json
  NOTE: concurrency defaults to 1 -- on DLIN the server crashes once >~2 requests
  overlap a prefill batch (DLIN FA2 q.reshape in jit_kernel/flash_attention.py).
  Raise -c to probe the ceiling; fix = route batched extend to flash_attn_varlen_func.

benchrun options (vLLM `bench run` format, wraps scripts/dl/benchrun_sglang.py):
  -M <preset> / -m <path>    model (preset or path; default Qwen3-1.7B)
  --num-prompts N            requests            (default 64)
  --input-len/--output-len   random in/out lens  (default 1024 / 128)
  -c, --concurrency N        max concurrent      (default 1; DLIN FA2 ceiling)
  --port N                   server port         (default 30000)
  --config PATH              run that config verbatim (passthrough)
  --template                 just write config_serving.json, don't run
  Builds a vLLM-schema config (server_params/client_params/fixed_params), auto-starts
  the sglang server per batch, runs sglang.bench_serving, and emits the SAME artifacts
  as `vllm bench run`: benchrun_result.json (dict keyed by full_key_name), per-case
  client/server log dirs, and a GitHub-markdown summary. Diff directly with vLLM output.
  DLIN defaults baked in: fa3, page_size=16, random-ids, HF offline, max_concurrency=1.
  For Qwen3.5-35B-A3B-FP8 the blockwise-FP8 path needs DLIN dlblas (gptq_dlblas_gemmex);
  if the env lacks it the run falls back to the slow bf16-bmm MoE path. Fused MoE
  (SGLANG_DL_MOE_FUSED=1) is opt-in — the -M qwen35-35b preset exports it.

sop options (DLIN upgrade verification standard, wraps scripts/dl/sop_verify.py):
  ./run_sglang.sh sop                       # verify vs baseline (default model Qwen3-1.7B)
  ./run_sglang.sh sop record                # CAPTURE current tree as the baseline golden+perf
                                            #   (run this on known-good dl-main FIRST)
  ./run_sglang.sh sop show                  # re-print the latest report (no GPU run)
  ./run_sglang.sh sop --quick               # correctness gates only (skip perf) — fast iteration
  ./run_sglang.sh sop -M qwen35-35b         # full gate on 35B TP4 (~10+ min)
  ./run_sglang.sh sop --sop-baseline PATH   # compare vs a specific baseline json
  Gates (absolute correctness): G1 DLIN stack smoke, G2 canonical probes
    (capital of France->Paris, 1+1=->2, ...), G3 greedy determinism, G4 no-gibberish,
    G5 JSON. Regression: R1 exact greedy match vs baseline golden. Perf (tolerance band):
    P1 decode tok/s, P2 prefill tok/s. Verdict PASS only if all gates pass; exit 0/2.
  Baseline workflow: `sop record` on dl-main -> `sop` (verify) on dl-dev-v0.5.15/.16.

Env overrides: SDK_DIR, VENV_DIR, TORCH_SPEC, SKIP_KERNEL,
               MODEL_PATH, ATTN_BACKEND, USE_CUDA_GRAPH, MAX_NEW_TOKENS, PROMPT, ...
EOF
}

parse_test_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      -m|--model)          MODEL_PATH="$2"; MODEL_EXPLICIT=1; shift 2;;
      -M|--model-preset)   pick_model "$2"; MODEL_EXPLICIT=1; shift 2;;
      -p|--prompt)         PROMPT="$2"; shift 2;;
      -n|--max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
      -b|--backend)        ATTN_BACKEND="$2"; shift 2;;
      -g|--cuda-graph)     USE_CUDA_GRAPH=1; shift;;
      -S|--spec-ngram)     USE_NGRAM=1; shift;;
      -W|--dl-warmup)      DL_WARMUP=1; shift;;
      --port)              SERVE_PORT="$2"; shift 2;;
      --host)              SERVE_HOST="$2"; shift 2;;
      # bench
      -c|--concurrency)    BENCH_CONCURRENCY="$2"; shift 2;;
      --num-prompts)       BENCH_NUM_PROMPTS="$2"; shift 2;;
      --input-len)         BENCH_INPUT_LEN="$2"; shift 2;;
      --output-len)        BENCH_OUTPUT_LEN="$2"; shift 2;;
      --batch-size)        BATCH_SIZE="$2"; shift 2;;
      --base-url)          BENCH_BASE_URL="$2"; shift 2;;
      --offline)           BENCH_OFFLINE=1; shift;;
      # benchrun
      --config)            BENCHRUN_CONFIG="$2"; shift 2;;
      --template)          BENCHRUN_TEMPLATE=1; shift;;
      # compare
      --scenarios)         COMPARE_SCENARIOS="$2"; shift 2;;
      --only)              COMPARE_ONLY="$2"; shift 2;;
      --show)              COMPARE_SHOW=1; shift;;
      --history)           COMPARE_HISTORY=1; shift;;
      --list)              COMPARE_LIST=1; shift;;
      --zh)                COMPARE_LIST_ZH=1; shift;;
      --no-record)         COMPARE_RECORD=0; shift;;
       --baseline)          COMPARE_BASELINE="$2"; shift 2;;
       --verbose)           COMPARE_VERBOSE=1; shift;;
      # compare --correctness (greedy output-equality vs vLLM)
      --correctness)       COMPARE_CORRECTNESS=1; shift;;
      --vllm-python)       COMPARE_VLLM_PYTHON="$2"; shift 2;;
      --max-model-len)     COMPARE_MAX_MODEL_LEN="$2"; shift 2;;
      # chat
      -q|--quick)           CHAT_QUICK="$2"; shift 2;;
      --url)                CHAT_URL="$2"; shift 2;;
      --chat-model)         CHAT_MODEL="$2"; shift 2;;
      --system-prompt)      CHAT_SYSTEM_PROMPT="$2"; shift 2;;
      --no-stream)          CHAT_NO_STREAM=1; shift;;
      # sop (DLIN upgrade verification standard)
      record|verify|show)   SOP_MODE="$1"; shift;;
      --quick)              SOP_QUICK=1; shift;;
      --sop-baseline)       SOP_BASELINE="$2"; shift 2;;
      -h|--help)           usage; exit 0;;
      *) die "unknown option '$1' for '$PHASE' (see -h)";;
    esac
  done
}

#-------------------------------------------------------------------------------
# Phase: gen -- quick end-to-end generation on DLIN (in-process sglang.Engine).
#   Reuses scripts/dl/run_qwen3_1_7b.py, driven by env vars.
#   attention_backend=fa3 + disable_cuda_graph are the DLIN defaults (see header);
#   they can be overridden with -b / -g, but the defaults are what actually runs.
#-------------------------------------------------------------------------------
phase_gen() {
  log "Phase [gen]: end-to-end generation on DLIN (in-process Engine)"
  [ -n "${VIRTUAL_ENV:-}" ] || { source "$VENV_DIR/bin/activate" || die "run 'setup' first"; }
  dlin_runtime_env

  # If a DLIN preset was selected (-M), use the optimized gen script that
  # respects TP/mem_fraction/context_len/cuda_graph_max_bs env vars.
  local tp="${DLIN_TP_SIZE:-1}"
  if [ "$tp" -gt 1 ] 2>/dev/null || [ -n "${DLIN_MEM_FRACTION:-}" ]; then
    local gen_py="$SGLANG_DIR/scripts/dl/qwen35_sg_tps.py"
    [ -f "$gen_py" ] || die "missing $gen_py (needed for TP>1 / preset models)"
    export MODEL_PATH ATTN_BACKEND USE_CUDA_GRAPH MAX_NEW_TOKENS PROMPT
    export TP_SIZE="$tp"
    export MEM_FRAC="${DLIN_MEM_FRACTION:-0.80}"
    export CONTEXT_LEN="${DLIN_CONTEXT_LEN:-4096}"
    export PAGE_SIZE="${DLIN_PAGE_SIZE:-16}"
    export CG_MAX_BS="${DLIN_CG_MAX_BS:-0}"
    export USE_NGRAM NGRAM_NUM_DRAFT NGRAM_MIN_BFS NGRAM_MAX_BFS
    log "model=$MODEL_PATH | tp=$TP_SIZE | backend=$ATTN_BACKEND | cg=$USE_CUDA_GRAPH | mem=$MEM_FRAC | ctx=$CONTEXT_LEN | ngram=$USE_NGRAM"
    [ -n "$PROMPT" ] && log "prompt: $PROMPT" || log "prompt: <builtin demos>"
    log "(first run JIT-compiles, ~5 min; cached after)"
    python "$gen_py"
  else
    local gen_py="$SGLANG_DIR/scripts/dl/run_qwen3_1_7b.py"
    [ -f "$gen_py" ] || die "missing $gen_py"
    export MODEL_PATH ATTN_BACKEND USE_CUDA_GRAPH MAX_NEW_TOKENS PROMPT
    log "model=$MODEL_PATH | backend=$ATTN_BACKEND | cuda_graph=$USE_CUDA_GRAPH | max_new_tokens=$MAX_NEW_TOKENS"
    [ -n "$PROMPT" ] && log "prompt: $PROMPT" || log "prompt: <builtin demos>"
    log "(first run JIT-compiles ~42 buckets, ~5 min; cached after -> <90s)"
    python "$gen_py"
  fi
}

#-------------------------------------------------------------------------------
# Phase: serve -- OpenAI-compatible HTTP server (launch_server) on DLIN.
#   Verified: launch_server imports cleanly (no flashinfer ImportError) and the
#   TP4 server boots for Qwen3.5-35B-A3B-FP8 — same launch_server path the bench
#   harness (start_server_bg) uses. First launch JIT-compiles (~5 min), then cached.
#-------------------------------------------------------------------------------
phase_serve() {
  log "Phase [serve]: HTTP server on DLIN (launch_server)"
  [ -n "${VIRTUAL_ENV:-}" ] || { source "$VENV_DIR/bin/activate" || die "run 'setup' first"; }
  dlin_runtime_env
  local tp="${DLIN_TP_SIZE:-1}"
  local page_size="${DLIN_PAGE_SIZE:-16}"
  local mem_frac="${DLIN_MEM_FRACTION:-}"
  local ctx_len="${DLIN_CONTEXT_LEN:-}"
  local cg_max_bs="${DLIN_CG_MAX_BS:-}"
  local cg_flag=""
  if [ "$USE_CUDA_GRAPH" != "1" ]; then
    cg_flag="--disable-cuda-graph"
  fi
  local extra_flags=""
  [ -n "$mem_frac" ] && extra_flags="$extra_flags --mem-fraction-static $mem_frac"
  [ -n "$ctx_len" ] && extra_flags="$extra_flags --context-length $ctx_len"
  [ -n "$cg_max_bs" ] && [ "$USE_CUDA_GRAPH" = "1" ] && extra_flags="$extra_flags --cuda-graph-max-bs-decode $cg_max_bs"
  # DL: TP>1 on DLIN must use NCCL — the custom allreduce kernel hits HC_CUK Error=28.
  [ "$tp" -gt 1 ] && extra_flags="$extra_flags --disable-custom-all-reduce"
  # DL: opt-in segmented prefill CG via --cuda-graph-backend-prefill. "breakable"
  # bypasses the multimodal auto-disable (server_args.py:1639 is tc_piecewise-only;
  # breakable's only rule is MLA @ _disable_breakable_cudagraph_if_incompatible).
  # Targets sglang raw-prefill (still ~3.7x slower than vLLM even with dl_chunk:
  # sglang ~399 vs vLLM ~1457 tok/s steady; the big 8x gap was the GDN triton path,
  # now fixed by SGLANG_DL_GDN_DLIN_EXTEND=1). Breakable prefill CG is an ADDITIONAL
  # opt-in that captures prefill in CG instead of eager. See opt-plan P1-1.
  [ -n "$DL_PREFILL_BACKEND" ] && extra_flags="$extra_flags --cuda-graph-backend-prefill $DL_PREFILL_BACKEND"
  local ngram_flags=""
  [ "$USE_NGRAM" = "1" ] && ngram_flags="--speculative-algorithm NGRAM --speculative-num-draft-tokens $NGRAM_NUM_DRAFT --speculative-ngram-min-bfs-breadth $NGRAM_MIN_BFS --speculative-ngram-max-bfs-breadth $NGRAM_MAX_BFS"
  # DL: -W/--dl-warmup pre-compiles fused-MoE dlcc kernels at startup by iterating
  # the engine's capture-size lists (server_args.cuda_graph_config.{prefill,decode}.bs)
  # — same approach as DLIN vLLM (warmup_sizes = compile + cg_capture sizes). One-time
  # per shape (~20-85s), cached at ~/.triton/cache. Subset via $SGLANG_DL_WARMUP_SHAPES
  # (csv). See python/sglang/srt/entrypoints/warmup.py:dlin_capture_sizes.
  local warmup_flags=""
  [ "$DL_WARMUP" = "1" ] && warmup_flags="--warmups dlin_capture_sizes"
  log "model=$MODEL_PATH | tp=$tp | backend=$ATTN_BACKEND | page=$page_size | host=$SERVE_HOST:$SERVE_PORT | ngram=$USE_NGRAM | dl_warmup=${DL_WARMUP:-0}"
  exec python -m sglang.launch_server \
    --model-path "$MODEL_PATH" --page-size "$page_size" --dtype bfloat16 \
    --tp-size "$tp" \
    --attention-backend "$ATTN_BACKEND" $cg_flag $extra_flags $ngram_flags $warmup_flags \
    --skip-server-warmup \
    --port "$SERVE_PORT" --host "$SERVE_HOST"
}

#-------------------------------------------------------------------------------
# Server lifecycle helpers (shared by serve-foreground and bench-autostart).
# launch_server spawns scheduler subprocesses, so we kill the whole process group.
#-------------------------------------------------------------------------------
health_up() {  # $1=url -> 0 if /health responds
  curl -s -o /dev/null --max-time 3 "$1/health" 2>/dev/null
}
wait_for_server() {  # $1=url  $2=timeout_s
  local url="$1" t="${2:-240}" elapsed=0
  while [ "$elapsed" -lt "$t" ]; do
    health_up "$url" && return 0
    sleep 5; elapsed=$((elapsed + 5))
  done
  return 1
}
start_server_bg() {  # launches launch_server detached; sets SERVER_PID
  local tp="${DLIN_TP_SIZE:-1}"
  local page_size="${DLIN_PAGE_SIZE:-16}"
  local mem_frac="${DLIN_MEM_FRACTION:-}"
  local ctx_len="${DLIN_CONTEXT_LEN:-}"
  local cg_max_bs="${DLIN_CG_MAX_BS:-}"
  local cg_flag=""
  if [ "$USE_CUDA_GRAPH" != "1" ]; then
    # DL: breakable CG backend — same rationale as phase_serve above.
    cg_flag="--disable-cuda-graph"
  fi
  local extra_flags=""
  [ -n "$mem_frac" ] && extra_flags="$extra_flags --mem-fraction-static $mem_frac"
  [ -n "$ctx_len" ] && extra_flags="$extra_flags --context-length $ctx_len"
  [ -n "$cg_max_bs" ] && [ "$USE_CUDA_GRAPH" = "1" ] && extra_flags="$extra_flags --cuda-graph-max-bs-decode $cg_max_bs"
  # DL: TP>1 on DLIN must use NCCL — the custom allreduce kernel hits HC_CUK Error=28.
  [ "$tp" -gt 1 ] && extra_flags="$extra_flags --disable-custom-all-reduce"
  # DL: opt-in segmented prefill CG via --cuda-graph-backend-prefill. "breakable"
  # bypasses the multimodal auto-disable (server_args.py:1639 is tc_piecewise-only;
  # breakable's only rule is MLA @ _disable_breakable_cudagraph_if_incompatible).
  # Targets sglang raw-prefill (still ~3.7x slower than vLLM even with dl_chunk:
  # sglang ~399 vs vLLM ~1457 tok/s steady; the big 8x gap was the GDN triton path,
  # now fixed by SGLANG_DL_GDN_DLIN_EXTEND=1). Breakable prefill CG is an ADDITIONAL
  # opt-in that captures prefill in CG instead of eager. See opt-plan P1-1.
  [ -n "$DL_PREFILL_BACKEND" ] && extra_flags="$extra_flags --cuda-graph-backend-prefill $DL_PREFILL_BACKEND"
  local ngram_flags=""
  [ "$USE_NGRAM" = "1" ] && ngram_flags="--speculative-algorithm NGRAM --speculative-num-draft-tokens $NGRAM_NUM_DRAFT --speculative-ngram-min-bfs-breadth $NGRAM_MIN_BFS --speculative-ngram-max-bfs-breadth $NGRAM_MAX_BFS"
  BENCH_LOG="${BENCH_LOG:-/tmp/sglang_bench_server.log}"
  # DL: cache self-healing — restore known-good dl_chunk triton cache + write-protect
  # before launching sglang. Prevents the contagious crash-corrupts-cache failure.
  if [ -d ~/.triton/cache.good_backup ]; then
    rm -rf ~/.triton/cache && cp -a ~/.triton/cache.good_backup ~/.triton/cache
    chmod -R a-w ~/.triton/cache 2>/dev/null
  fi
  nohup python -m sglang.launch_server \
    --model-path "$MODEL_PATH" --page-size "$page_size" --dtype bfloat16 \
    --tp-size "$tp" \
    --attention-backend "$ATTN_BACKEND" $cg_flag $extra_flags $ngram_flags \
    --skip-server-warmup \
    --port "$SERVE_PORT" --host "$SERVE_HOST" \
    > "$BENCH_LOG" 2>&1 &
  SERVER_PID=$!
}
stop_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  local pgid
  pgid=$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')
  if [ -n "$pgid" ]; then
    kill -TERM -"$pgid" 2>/dev/null; sleep 3
    kill -KILL -"$pgid" 2>/dev/null || true
  fi
  SERVER_PID=""
}

#-------------------------------------------------------------------------------
# Phase: bench -- concurrent serving benchmark. Wraps sglang's official tools:
#   * online (default): bench_serving vs a server. Auto-starts one on
#     $SERVE_HOST:$SERVE_PORT if none is up (or use --base-url for external).
#   * --offline: bench_one_batch spins its own engine (no server), sweeps
#     --batch-size for per-batch prefill/decode latency.
#-------------------------------------------------------------------------------
phase_bench() {
  log "Phase [bench]: concurrent serving benchmark"
  [ -n "${VIRTUAL_ENV:-}" ] || { source "$VENV_DIR/bin/activate" || die "run 'setup' first"; }
  dlin_runtime_env
  command -v curl >/dev/null || die "curl is required for bench"

  # Force local-only HF/transformers: the bench client loads a tokenizer, and on
  # an offline host huggingface_hub's httpx client throws "client has been
  # closed"; offline mode makes it read the model dir directly. Also picks the
  # ShareGPT-free 'random-ids' dataset (the 'random' one downloads ShareGPT).
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  if [ "$BENCH_CONCURRENCY" -gt 1 ] 2>/dev/null; then
    warn "concurrency=$BENCH_CONCURRENCY: the DLIN server crashes once >~2 requests"
    warn "overlap a prefill batch (DLIN FA2 q.reshape, jit_kernel/flash_attention.py)."
    warn "If it dies, retry with -c 1 (stable single-stream)."
  fi

  if [ "$BENCH_OFFLINE" = "1" ]; then
    log "[offline] bench_one_batch | model=$MODEL_PATH backend=$ATTN_BACKEND batch-size='$BATCH_SIZE'"
    python -m sglang.bench_one_batch \
      --model-path "$MODEL_PATH" --page-size 16 --dtype bfloat16 \
      --attention-backend "$ATTN_BACKEND" \
      $( [ "$USE_CUDA_GRAPH" = "1" ] || echo --disable-cuda-graph ) \
      --batch-size $BATCH_SIZE --input-len "$BENCH_INPUT_LEN" --output-len "$BENCH_OUTPUT_LEN"
    return
  fi

  local url autostarted=0
  if [ -n "$BENCH_BASE_URL" ]; then
    url="$BENCH_BASE_URL"
  else
    url="http://$SERVE_HOST:$SERVE_PORT"
    if ! health_up "$url"; then
      log "no server at $url -> autostarting (log: /tmp/sglang_bench_server.log)"
      start_server_bg
      autostarted=1
      wait_for_server "$url" 300 || { warn "server log tail:"; tail -n 15 /tmp/sglang_bench_server.log; stop_server; die "server did not become ready"; }
      ok "server ready at $url"
    else
      log "using existing server at $url"
    fi
  fi
  [ "$autostarted" = "1" ] && trap stop_server EXIT INT TERM

  local outfile="/tmp/sglang_bench_$(basename "$MODEL_PATH").json"
  log "[online] bench_serving | url=$url concurrency=$BENCH_CONCURRENCY num-prompts=$BENCH_NUM_PROMPTS in/out=$BENCH_INPUT_LEN/$BENCH_OUTPUT_LEN"
  python -m sglang.bench_serving \
    --backend sglang-oai --base-url "$url" --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --dataset-name random-ids --num-prompts "$BENCH_NUM_PROMPTS" \
    --random-input-len "$BENCH_INPUT_LEN" --random-output-len "$BENCH_OUTPUT_LEN" \
    --max-concurrency "$BENCH_CONCURRENCY" \
    --output-file "$outfile"
  local rc=$?
  [ "$autostarted" = "1" ] && { log "stopping autostarted server"; stop_server; }
  [ $rc -eq 0 ] && ok "bench done. results: $outfile"
  return $rc
}

#-------------------------------------------------------------------------------
# Phase: benchrun -- vLLM-benchrun-format serving benchmark. Wraps scripts/dl/
#   benchrun_sglang.py (a port of vLLM's DLIN benchrun_serving). Emits the SAME
#   config schema + artifacts as `vllm bench run` so sglang vs vLLM results diff
#   directly. The harness auto-starts/stops the sglang server per batch.
#
#   * --config PATH : run that vLLM-schema config verbatim (passthrough).
#   * --template    : write a config_serving.json template (hand-edit path).
#   * otherwise     : build a config from the flags + (-M) preset, then run.
#
#   DLIN defaults are baked into the harness (fa3, page_size=16, random-ids, HF
#   offline, max_concurrency=1). -M qwen35-35b exports SGLANG_DL_MOE_FUSED=1
#   (needs the dlblas fused-MoE op; without it the run falls back to slow bf16-bmm).
#-------------------------------------------------------------------------------
phase_benchrun() {
  log "Phase [benchrun]: vLLM-benchrun-format serving benchmark (sglang)"
  [ -n "${VIRTUAL_ENV:-}" ] || { source "$VENV_DIR/bin/activate" || die "run 'setup' first"; }
  dlin_runtime_env
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  local script="$SGLANG_DIR/scripts/dl/benchrun_sglang.py"
  [ -f "$script" ] || die "harness not found: $script"

  # --config PATH -> passthrough to the harness verbatim.
  if [ -n "$BENCHRUN_CONFIG" ]; then
    log "[benchrun] config: $BENCHRUN_CONFIG"
    python "$script" "$BENCHRUN_CONFIG"
    return $?
  fi

  # --template -> let the harness write config_serving.json in CWD and exit.
  if [ "$BENCHRUN_TEMPLATE" = "1" ]; then
    log "[benchrun] writing config_serving.json template in $(pwd)"
    python "$script"
    return $?
  fi

  # Otherwise build a vLLM-schema config from flags + preset, then run it.
  local tp="${DLIN_TP_SIZE:-1}"
  local page="${DLIN_PAGE_SIZE:-16}"
  local mem="${DLIN_MEM_FRACTION:-0.85}"
  local timeout="${BENCHRUN_TIMEOUT:-900}"
  # cuda graph: off by default on DLIN unless -g / -M qwen35-35b turned it on.
  local cg_field="true"   # disable_cuda_graph
  [ "$USE_CUDA_GRAPH" = "1" ] && cg_field="false"

  local work cfg
  work="$(mktemp -d -t sglang_benchrun_XXXXXX)"
  cfg="$work/config_serving.json"
  cat > "$cfg" <<EOF
{
  "server_params": {"tensor_parallel_size":$tp,"attention_backend":"$ATTN_BACKEND","page_size":$page,"dtype":"bfloat16","mem_fraction_static":$mem,"disable_cuda_graph":$cg_field,"trust_remote_code":""},
  "client_params": {"dataset_name":"random-ids","random_input_len":[$BENCH_INPUT_LEN],"random_output_len":[$BENCH_OUTPUT_LEN],"num_prompts":[$BENCH_NUM_PROMPTS],"temperature":0,"max_concurrency":[$BENCH_CONCURRENCY],"profile":[false]},
  "fixed_params": {"mode":"serving","model":"$MODEL_PATH","port":$SERVE_PORT,"time_out":$timeout,"output_file":"benchrun_result.json","output_dir":"$work","ci_log_format":false,"output_json_format":false}
}
EOF
  local cg_state="off"; [ "$USE_CUDA_GRAPH" = "1" ] && cg_state="on"
  log "[benchrun] model=$MODEL_PATH tp=$tp backend=$ATTN_BACKEND mem=$mem cg=$cg_state | prompts=$BENCH_NUM_PROMPTS in/out=$BENCH_INPUT_LEN/$BENCH_OUTPUT_LEN conc=$BENCH_CONCURRENCY"
  log "[benchrun] work dir (config + artifacts): $work"
  python "$script" "$cfg"
  local rc=$?
  # Tidy the transient script-dir copies the harness writes next to itself.
  rm -f "$SGLANG_DIR/scripts/dl/benchrun_result.json" "$SGLANG_DIR/scripts/dl/profiler_serving_data.json" 2>/dev/null || true
  local model_tail; model_tail="$(basename "${MODEL_PATH%/}")"
  log "[benchrun] result: $work/benchrun_${model_tail}/benchrun_result.json"
  return $rc
}

#-------------------------------------------------------------------------------
# Phase: chat -- interactive OpenAI-compatible chat client.
#   Connects to an already-running SGLang (or any OpenAI-compatible) server.
#   Defaults: url=http://$SERVE_HOST:$SERVE_PORT/v1, model=auto-detect.
#   Supports interactive prompt loop with history, or -q for quick single-shot.
#-------------------------------------------------------------------------------
phase_chat() {
  local script="$SGLANG_DIR/scripts/dl/chat.py"
  [ -f "$script" ] || die "chat script not found: $script"
  [ -n "${VIRTUAL_ENV:-}" ] || { source "$VENV_DIR/bin/activate" || die "run 'setup' first"; }

  local url="${CHAT_URL:-http://$SERVE_HOST:$SERVE_PORT/v1}"
  local stream_flag=""
  [ "$CHAT_NO_STREAM" = "1" ] && stream_flag="--no-stream"

  log "Phase [chat]: interactive OpenAI-compatible chat client"
  log "  url=$url  model=${CHAT_MODEL:-<auto>}  quick=${CHAT_QUICK:-<interactive>}"

  python "$script" \
    --url "$url" \
    ${CHAT_MODEL:+--model "$CHAT_MODEL"} \
    ${CHAT_SYSTEM_PROMPT:+--system-prompt "$CHAT_SYSTEM_PROMPT"} \
    ${CHAT_QUICK:+--quick "$CHAT_QUICK"} \
    $stream_flag
}

#-------------------------------------------------------------------------------
# Phase: sop -- DLIN upgrade VERIFICATION STANDARD.
#   Wraps scripts/dl/sop_verify.py. Runs correctness gates (absolute) + perf
#   gates (within tolerance of a recorded baseline) and emits PASS/FAIL + a JSON
#   report to /tmp/sglang_sop/. This is the judging bar for porting DLIN changes
#   onto a new sglang tag: record the baseline on known-good dl-main, then run
#   `sop` (verify) on the ported branch and require PASS.
#
#   Default model = Qwen3-1.7B (fast gate). -M qwen35-35b = full 35B TP4 gate.
#   Modes:  verify (default) | record | show.
#   --quick : correctness gates only (skip perf) — fastest, for rapid iteration.
#   --sop-baseline PATH : compare vs that baseline (else docs/dl/sop_baseline_<model>.json).
#-------------------------------------------------------------------------------
phase_sop() {
  log "Phase [sop]: DLIN upgrade verification standard (mode=${SOP_MODE:-verify})"
  [ -n "${VIRTUAL_ENV:-}" ] || { source "$VENV_DIR/bin/activate" || die "run 'setup' first"; }
  dlin_runtime_env
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  # Map the model preset (pick_model via -M, else the 1.7B default) -> SOP env.
  # sop is NOT in the one-click 35B default list, so MODEL_PATH stays 1.7B unless
  # the user passes -M qwen35-35b (the full-gate path).
  local model_tag; model_tag="$(basename "${MODEL_PATH:-/opt/dataset/Qwen3-1.7B}")"
  export SOP_MODEL="${MODEL_PATH:-/opt/dataset/Qwen3-1.7B}"
  export SOP_TP="${DLIN_TP_SIZE:-1}"
  export SOP_BACKEND="${ATTN_BACKEND:-fa3}"
  export SOP_PAGE_SIZE="${DLIN_PAGE_SIZE:-16}"
  export SOP_MEM_FRACTION="${DLIN_MEM_FRACTION:-0.80}"
  export SOP_CONTEXT_LEN="${DLIN_CONTEXT_LEN:-4096}"
  export SOP_CG="${USE_CUDA_GRAPH:-0}"
  export SOP_CG_MAX_BS="${DLIN_CG_MAX_BS:-0}"
  export SOP_MAX_NEW SOP_MODE SOP_QUICK
  [ -n "${SOP_BASELINE:-}" ] && export SOP_BASELINE
  local script="$SGLANG_DIR/scripts/dl/sop_verify.py"
  [ -f "$script" ] || die "SOP script not found: $script"
  local btag="${SOP_BASELINE:-docs/dl/sop_baseline_${model_tag}.json}"
  log "[sop] model=$model_tag tp=$SOP_TP cg=$SOP_CG quick=$SOP_QUICK mode=$SOP_MODE baseline=$btag"
  # NB: pass --quick only when SOP_QUICK=="1" — ${VAR:+x} treats "0" as non-empty.
  local quick_arg=""; [ "${SOP_QUICK:-0}" = "1" ] && quick_arg="--quick"
  python "$script" "$SOP_MODE" $quick_arg ${SOP_BASELINE:+--baseline "$SOP_BASELINE"}
  local rc=$?
  if [ "$rc" -eq 0 ]; then ok "SOP VERDICT: PASS"; else die "SOP VERDICT: FAIL (exit $rc)"; fi
  return $rc
}

#-------------------------------------------------------------------------------
# Phase: compare -- sglang vs vLLM showcase (RadixAttention vs APC).
#   Wraps scripts/dl/showcase_prefix_sharing.py. Runs BOTH engines on the same
#   GPUs sequentially (fresh process each — required for clean cache state on
#   DLIN), caches the per-engine METRICS block to $COMPARE_METRICS_DIR, and
#   prints a side-by-side gap table. The "one-click" tracker: re-run after any
#   sglang change to see if the gap moved.
#
#   --only sglang|vllm  re-measure just that side and diff vs the cached metrics
#                       of the other (skip its ~90s load). Useful when iterating
#                       on one engine.
#   --list              print every scenario with an ASCII workload diagram + the
#                       win/loss verdict (no GPU run) — then pick with --scenarios.
#   --scenarios SC1,SC2,SC3[,SC4,SC5,SC7,SC8,SC9,SC10]
#                       default SC1,SC2,SC3. SC4=JSON. DL: SC5=multi-user fork
#                       (radix tree), SC7=long-RAG throughput, SC8=parallel
#                       sampling (best-of-N), SC9=pure long decode,
#                       SC10=shared system-prompt throughput. The full showcase
#                       (all 8) ~25-30 min (MRV1+APC adds ~10min for the 528 capture).
#
#   NOTE: the showcase script is Qwen3.5-35B-A3B-FP8 / TP4 specific (hardcoded
#   prompts + the FP8 fused-MoE path). -M is accepted but only to set TP/mem env;
#   the model path it passes must be the FP8 Qwen3.5-35B.
#-------------------------------------------------------------------------------
_metric_val() {  # $1=KEY $2=file -> echoes value (empty if absent)
  # `|| true`: grep returns 1 on no-match; under `set -eo pipefail` an unguarded
  # failure here would kill the whole compare run when a key is absent (e.g.
  # SC1_* keys in an SC3-only run). Always return 0.
  grep "^METRIC $1=" "$2" 2>/dev/null | tail -1 | cut -d= -f2- || true
}
_compare_row() {  # $1=label  $2=KEY  $3=dir(lower|higher) — 3 engines: sglang | MRV2 | MRV1
  local label="$1" key="$2" dir="$3"
  local s m2 m1
  s=$(_metric_val  "$key" "$COMPARE_METRICS_DIR/metrics_sglang.txt")
  m2=$(_metric_val "$key" "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt")
  m1=$(_metric_val "$key" "$COMPARE_METRICS_DIR/metrics_vllm_mrv1.txt")
  [ -z "$s" ] && [ -z "$m2" ] && [ -z "$m1" ] && return 0
  local sd=$([ -n "$s" ] && echo "$s" || echo "NA")
  local m2d=$([ -n "$m2" ] && echo "$m2" || echo "FAIL")
  local m1d=$([ -n "$m1" ] && echo "$m1" || echo "FAIL")
  local verdict="(need sglang+MRV2)"
  if [ -n "$s" ] && [ -n "$m2" ]; then
    verdict=$(awk -v s="$s" -v v="$m2" -v dir="$dir" 'BEGIN{
      r=(dir=="higher")?(s/v):(v/s);
      if (r+0>=1.0) printf "sglang %.2fx vs MRV2", r; else printf "MRV2 %.2fx", 1/r;
    }')
  fi
  printf "  %-24s | %-10s | %-10s | %-10s | %s\n" "$label" "$sd" "$m2d" "$m1d" "$verdict"
}

# Chinese version: sglang vs vLLM 场景中文描述（compare --list --zh）
_compare_list_zh() {
  cat <<'COMPARE_EOF'

================ sglang vs vLLM — 场景对比列表（--list --zh）================
每个场景代表一种不同的负载模式。选子集运行：
    ./run_sglang.sh compare --scenarios SC2,SC5,SC7            # 逗号分隔
    ./run_sglang.sh compare --scenarios SC5,SC7,SC8,SC9,SC10   # 完整新版展示
两引擎：同模型（Qwen3.5/3.6-35B-A3B-FP8）、TP4、同 GPU、独立进程、
temperature 0（SC8/SC8b=0.7，真实 best-of-N）、预热后 best-of-2。
公平基线：vLLM MRV1 + CG + APC ON（前缀缓存开）。sglang 使用 GDN
dl_chunk 开关（SGLANG_DL_GDN_DLIN_EXTEND=1，默认开）+ RadixAttention。
  => 开 GDN 开关后，sglang 在大多数前缀复用 + serving 场景获胜
     （RadixAttention 复用完整 hybrid-Mamba 状态，含 Mamba 循环状态，
     vLLM APC 只能部分缓存）。vLLM 仅在纯 prefill（SC6）和首次冷 prefill 获胜。
注意：不开 GDN 开关时 sglang 多数场景输（旧 r009 基线）——该开关是关键杠杆。
详见技术报告 docs/dl/sglang-beats-vllm-dlin-technical-report.zh.md。
图例：[sglang win ~Nx] / [vLLM win ~Nx]   [probe] = 诊断探针

---------------------------------------------------------------------------
SC1  前缀共享（扁平前缀）                    [sglang win ~1.14x warm]
     [共享前缀] -- req1（独立后缀）
                -- req2
                -- req3
     冷：prefill 前缀；热：缓存命中 → 完全跳过 prefill。

SC2  多轮对话（单一线性问题）                 [vLLM win ~1.17x]
     sys+history -> turn1 -> turn2 -> turn3 -> turn4 -> turn5（单个用户，增长中）
     每一轮的 suffix（生成文本 + 新问题）更大 → sglang 略输
     （其绝对 prefill 仍较慢；vLLM APC 复用增长中的前缀）。

SC3  并发批处理（一个前缀，批量 decode）       [sglang win ~1.45x]
     [共享 prompt] -- 分支到 N 个 decode 同时（单次调用，多个输出）
     聚合批处理 prefill+decode 吞吐。受 prefill 瓶颈（融合 FP8 MoE）。

SC4  结构化 JSON（短约束解码）               [~tie / vLLM +~9%]
     prompt -> { "name": "...", "age": ... }（greedy JSON）
     受 decode 瓶颈，短输出，无前缀复用 → 纯 decode + JSON 路径决定胜负。

SC5  多用户分支（基数树，共享根）              [sglang win ~1.01x（平）]
              [共享 system-prompt 根]   <- prefill 一次，KV 共享
                /              \
         用户 A 分支        用户 B 分支
        turn1->2->3->4      turn1->2->3->4（交错，分支树）
     多租户 / 多个用户共享一个 system prompt。

SC6  纯 prefill 对标（唯一 prompt，无缓存）    [probe -- vLLM ~3-4x 更快]
     唯一 prompt1   唯一 prompt2   ...（不同前缀 → 缓存无法命中）
     隔离纯 prefill 速率。sglang 稳态 prefill ~399 vs vLLM ~1457 tok/s
     （sglang dl_chunk 仍慢于 vLLM 的绝对 prefill kernel）。

SC7  长 RAG 吞吐（~2K 文档 × 8 查询）         [sglang win ~1.12x]
     [~2K-token 文档]   <- 共享；sglang 缓存完整状态，vLLM APC 部分缓存
          |- Q1 |
          |- Q2 |    8 个不同查询基于同一文档；聚合 decode tok/s。
          |- ...|    RAG 长共享文档场景。

SC8  重复 best-of-N（RLHF 循环，同 prompt）    [sglang win ~1.25x]
     [prompt]  <- 跨轮次复用；sglang 缓存它
       /  |  |  \     n=4 采样补全（temp=0.7）
      c1 c2 c3 c4
     RLHF 拒绝采样循环。缓存主导；单次调用 best-of-N 见 SC8b。

SC8b 冷启动单次 best-of-N（唯一 prompt）       [probe -- sglang ~2x]
     每次调用唯一 prompt -> n=4 采样（无跨调用缓存）。
     从 SC8 的缓存因子中隔离单次 best-of-N。

SC9  纯长 decode（短 prompt + 128 tok）        [sglang win +5.7%]
     [短 ~14-tok prompt] -> decode 128 token（单流，无共享结构）
     受 decode 瓶颈；sglang overlap scheduler + vLLM APC/mamba-align 开销 →
     sglang 略胜（5 次重复分布不重叠）。需充分预热 + 干净卡，否则偏低 ~8 tok/s。

SC10 共享 system-prompt（多租户）              [sglang win ~1.13x]
     [~0.9K system prompt]   <- 12 个租户共享
          |- tenant1 |
          |- tenant2 |   24 个短请求，同一角色；RadixAttention 生产经典场景。
          |- ...     |
---------------------------------------------------------------------------
Serving（非场景，单独）：并发 HTTP，共享前缀 → sglang 1.2-1.6x
  并发 4-32（exp_b_serving_client.py）。详见报告 §3.5。
完整文档：docs/dl/sglang-beats-vllm-dlin-technical-report.zh.md（决定性报告），
          docs/dl/sglang-vs-vllm-showcase-dlin.md（SC1-4），
          docs/dl/sglang-vs-vllm-new-scenarios.md（SC5/7/8/9/10）。
COMPARE_EOF
}

# Print every compare scenario with an ASCII diagram of its workload shape, the
# win/loss verdict, and the typical ratio — so you can pick which to run via
# --scenarios SCx,SCy. No GPU/model/env needed. (compare --list)
_compare_list() {
  cat <<'COMPARE_EOF'

================ sglang vs vLLM — compare scenarios (--list) ================
Each scenario is a distinct workload shape. Pick any subset:
    ./run_sglang.sh compare --scenarios SC2,SC5,SC7            # comma-list
    ./run_sglang.sh compare --scenarios SC5,SC7,SC8,SC9,SC10   # full new showcase
Both engines: same model (Qwen3.5/3.6-35B-A3B-FP8), TP4, same GPUs, fresh
process each, temp 0 (SC8/SC8b=0.7 for real best-of-N), best-of-2 after warmup.
FAIR baseline: vLLM MRV1 + CG + APC ON (prefix cache ON). sglang uses the GDN
dl_chunk flag (SGLANG_DL_GDN_DLIN_EXTEND=1, default-on) + RadixAttention.
  => With the GDN flag, sglang WINS most prefix-reuse + serving scenarios
     (RadixAttention reuses the full hybrid-Mamba state, incl. Mamba recurrent,
     which vLLM APC can only partially cache). vLLM still wins raw/unique prefill
     (SC6) and one-time cold-prefill. NOTE: without the GDN flag sglang LOSES most
     (the old r009 baseline) — the flag is the lever. See technical report
     docs/dl/sglang-beats-vllm-dlin-technical-report.zh.md.
legend: [sglang win ~Nx] / [vLLM win ~Nx]   [probe] = rigor diagnostic

---------------------------------------------------------------------------
SC1  prefix-sharing  (flat prefix)                  [sglang win ~1.14x warm]
     [shared prefix] -- req1 (independent suffix)
                    -- req2
                    -- req3
     cold: prefill the prefix; warm: cache hit -> skip prefill entirely.

SC2  multi-turn  (single linear conversation)        [vLLM win ~1.17x]
     sys+history -> turn1 -> turn2 -> turn3 -> turn4 -> turn5   (one user, growing)
     each turn's suffix (generated text + new Q) is larger -> sglang loses slightly
     (its absolute prefill is still slower; vLLM APC reuses the growing prefix).

SC3  concurrent batch  (one prefix, batched decodes) [sglang win ~1.45x]
     [shared prompt] -- fork to a BATCH of N decodes at once (one call, many outputs)
     aggregate batch prefill+decode throughput. Prefill-bound (fused FP8 MoE).

SC4  structured JSON  (short constrained decode)     [~tie / vLLM +~9%]
     prompt -> { "name": "...", "age": ... }   (greedy JSON)
     decode-bound, short output, NO prefix reuse -> raw decode + JSON path decides it.

SC5  multi-user fork  (radix TREE, shared root)      [sglang win ~1.01x (tie)]
                [shared system-prompt root]   <- prefill ONCE, KV shared
                   /              \
            user-A branch        user-B branch
          turn1->2->3->4         turn1->2->3->4    (interleaved, branching tree)
     multi-tenant / many users behind one system prompt.

SC6  raw-prefill parity  (UNIQUE prompts, NO cache)  [probe -- vLLM ~3-4x FASTER]
     unique prompt1   unique prompt2   ...   (distinct prefixes -> cache cannot hit)
     isolates RAW prefill rate. sglang steady prefill ~399 vs vLLM ~1457 tok/s
     (sglang dl_chunk is still slower than vLLM's absolute prefill kernel).

SC7  long-RAG throughput  (~2K doc x 8 queries)      [sglang win ~1.12x]
     [~2K-token doc]   <- shared; sglang caches full state, vLLM APC partially
          |- Q1 |
          |- Q2 |    8 diverse queries over the SAME doc; aggregate decode tok/s.
          |- ...|    RAG over a long shared document.

SC8  repeated best-of-N  (RLHF loop, same prompt)    [sglang win ~1.25x]
     [prompt]  <- reused across reps; sglang caches it
       /  |  |  \     n=4 sampled completions (temp=0.7)
      c1 c2 c3 c4
     RLHF rejection-sampling loop. Cache-dominated; single-call best-of-N is SC8b.

SC8b cold single-call best-of-N  (UNIQUE prompt)     [probe -- sglang ~2x]
     unique prompt per call -> n=4 samples (no cross-call cache).
     isolates single-call best-of-N edge from SC8's caching factor.

SC9  pure long decode  (short prompt + 128 tok)      [sglang win ~+5.7%]
     [short ~14-tok prompt] -> decode 128 tokens (single stream, no shared structure)
     decode-bound; sglang overlap scheduler + vLLM's APC/mamba-align overhead ->
     sglang edges ahead (5-rep distributions non-overlapping). Measure with full
     warmup + clean cards or it reads ~8 tok/s low (see report §6).

SC10 shared system-prompt  (many tenants)            [sglang win ~1.13x]
     [~0.9K system prompt]   <- shared by 12 tenants
          |- tenant1 |
          |- tenant2 |   24 short reqs, same persona; canonical RadixAttention-in-prod.
          |- ...     |
---------------------------------------------------------------------------
Serving (not a scenario, separate): concurrent HTTP, shared prefix -> sglang 1.2-1.6x
  at concurrency 4-32 (exp_b_serving_client.py). See report §3.5.
Full docs: docs/dl/sglang-beats-vllm-dlin-technical-report.zh.md (definitive report),
           docs/dl/sglang-vs-vllm-showcase-dlin.md (SC1-4),
           docs/dl/sglang-vs-vllm-new-scenarios.md (SC5/7/8/9/10).
COMPARE_EOF
}

_print_compare_commands() {
  local script="$SGLANG_DIR/scripts/dl/showcase_prefix_sharing.py"
  local py="${VENV_DIR:-.venv}/bin/python"
  echo ""
  echo "========================================================================="
  echo "  FULL engine launch commands (inside showcase_prefix_sharing.py)"
  echo "========================================================================="
  echo ""
  echo "# 1) Env vars set by the script (os.environ.setdefault):"
  echo "  SGLANG_DL_FP8_Q2=1"
  echo "  SGLANG_DL_MOE_FUSED=1"
  echo "  SGLANG_DL_MOE_FUSED_MAX_M=2048"
  echo "  SGLANG_DL_MOE_V3=1            # prefill MoE via vLLM invoke_fused_moe_opt_v3 (3.7x faster; sglang beats vLLM on prefill)"
  echo "  SGLANG_DL_GDN_DLIN=1"
  echo "  SGLANG_DL_GDN_DLIN_EXTEND=1   # GDN prefill on DLIN dl_chunk (8x faster; the win lever)"
  echo "  SGLANG_DL_MULTI_STEP=1"
  echo "  DLEOL_CACHE_SIZE=1024"
  echo "  DLEOL_FLA_ENABLE_PINGPONG=1"
  echo "  DLEOL_FLA_UNROLL_COUNT=8"
  echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "  HF_HUB_OFFLINE=1"
  echo "  TRANSFORMERS_OFFLINE=1"
  echo "  DLEOL_USE_CU_MQA_TILEKV=1"
  echo "  VLLM_MAX_MOE_CU_TOKENS=128"
  echo ""
  echo "# 2) Wrapper command (what run_sglang.sh executes):"
  echo "  $py $script \\"
  echo "      --engine <sglang|vllm> \\"
  echo "      --vllm-runner <mrv2|mrv1> \\"
  echo "      --mem-frac $COMPARE_MEM_FRAC \\"
  echo "      --scenarios $COMPARE_SCENARIOS"
  echo ""
  echo "# 3) Internal engine launch (Python API calls inside the script):"
  echo ""
  echo "  --- sglang (sgl.Engine) ---"
  echo "  engine = sgl.Engine("
  echo "      model_path=${MODEL_PATH:-<MODEL_PATH>},"
  echo "      tp_size=${DLIN_TP_SIZE:-4},"
  echo "      dtype=bfloat16,"
  echo "      context_length=4096,"
  echo "      mem_fraction_static=$COMPARE_MEM_FRAC,"
  echo "      max_running_requests=4,"
  echo "      disable_cuda_graph=False,"
  echo "      cuda_graph_max_bs_decode=4,"
  echo "      attention_backend=${ATTN_BACKEND:-fa3},"
  echo "      page_size=16,"
  echo "      disable_custom_all_reduce=True,"
  echo "      trust_remote_code=True,"
  echo "      chunked_prefill_size=512,"
  echo "  )"
  echo ""
  echo "  --- vLLM MRV2 (LLM + CG, APC unsupported -> will FAIL on this model) ---"
  echo "  llm = LLM("
  echo "      model=${MODEL_PATH:-<MODEL_PATH>},"
  echo "      tensor_parallel_size=${DLIN_TP_SIZE:-4},"
  echo "      dtype=bfloat16,"
  echo "      max_model_len=4096,"
  echo "      gpu_memory_utilization=$COMPARE_MEM_FRAC,"
  echo "      trust_remote_code=True,"
  echo "      max_num_seqs=4,"
  echo "      disable_log_stats=True,"
  echo "      compilation_config={cudagraph_capture_sizes: [1,2,4],"
  echo "                           max_cudagraph_capture_size: 4},"
  echo "  )"
  echo "  env: VLLM_USE_V2_MODEL_RUNNER=1"
  echo "  NOTE: MRV2 does NOT support APC (mamba_cache_mode='align' assertion) -> recorded as fail."
  echo ""
  echo "  --- vLLM MRV1 (LLM + CG + APC) = FAIR BASELINE (prefix cache ON) ---"
  echo "  llm = LLM("
  echo "      model=${MODEL_PATH:-<MODEL_PATH>},"
  echo "      tensor_parallel_size=${DLIN_TP_SIZE:-4},"
  echo "      dtype=bfloat16,"
  echo "      max_model_len=4096,"
  echo "      gpu_memory_utilization=$COMPARE_MEM_FRAC,"
  echo "      trust_remote_code=True,"
  echo "      max_num_seqs=4,"
  echo "      disable_log_stats=True,"
  echo "      enable_prefix_caching=True,"
  echo "      compilation_config={mode: NONE,"
  echo "                           cudagraph_capture_sizes: [1,2,4,528],"
  echo "                           max_cudagraph_capture_size: 528},"
  echo "  )"
  echo "  env: VLLM_USE_V2_MODEL_RUNNER=0"
  echo "========================================================================="
  echo ""
}

#-------------------------------------------------------------------------------
# compare --correctness: sglang vs vLLM GREEDY (temperature=0) output-equality.
#   Wraps scripts/dl/compare_correctness.py. Runs each engine in a FRESH process
#   (clean GPU/cache state on DLIN), dumps JSON {text, token_ids} per prompt,
#   then diffs them: exact-match PASS/FAIL + first diverging token.
#
#   Default model = DeepSeek-V4-Flash (`-M dsv4-flash`, TP8). Model-agnostic, so
#   `-m <any model>` works too. Reuses -p (single prompt), -n (max tokens),
#   --only sglang|vllm (re-run one side vs the cached JSON of the other).
#-------------------------------------------------------------------------------
_phase_compare_correctness() {
  log "Phase [compare]: CORRECTNESS — sglang vs vLLM, temperature=0, exact-match"
  [ -n "${VIRTUAL_ENV:-}" ] || { source "$VENV_DIR/bin/activate" || die "run 'setup' first"; }
  [ -f "$SDK_DIR/env.sh" ] && { source "$SDK_DIR/env.sh" 2>/dev/null || warn "SDK env.sh sourced with warnings"; }
  dlin_runtime_env
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  local script="$SGLANG_DIR/scripts/dl/compare_correctness.py"
  [ -f "$script" ] || die "correctness harness not found: $script"
  mkdir -p "$COMPARE_METRICS_DIR"
  local sgl_json="$COMPARE_METRICS_DIR/correctness_sglang.json"
  local vllm_json="$COMPARE_METRICS_DIR/correctness_vllm.json"
  local tp="${DLIN_TP_SIZE:-8}"
  local max_new="${MAX_NEW_TOKENS:-64}"
  local sgl_py="$VENV_DIR/bin/python"
  local vllm_py="${COMPARE_VLLM_PYTHON:-$sgl_py}"
  [ -x "$vllm_py" ] || vllm_py="python"

  local prompt_args=()
  [ -n "$PROMPT" ] && prompt_args=(--prompt "$PROMPT")

  _cc_run() {  # $1=engine  -> 0 ok, 1 fail
    local eng="$1" py out
    if [ "$eng" = "sglang" ]; then py="$sgl_py"; out="$sgl_json"; else py="$vllm_py"; out="$vllm_json"; fi
    log "[correctness] running $eng (fresh process, python=$py)"
    "$py" "$script" run --engine "$eng" --model "$MODEL_PATH" --tp "$tp" \
        --max-tokens "$max_new" --max-model-len "$COMPARE_MAX_MODEL_LEN" \
        "${prompt_args[@]}" --out "$out" || { warn "$eng correctness run FAILED (see above)"; return 1; }
  }

  case "$COMPARE_ONLY" in
    sglang) _cc_run sglang || die "sglang run failed" ;;
    vllm)   _cc_run vllm   || die "vllm run failed" ;;
    "")     _cc_run sglang || die "sglang run failed (cannot compare without it)"
            _cc_run vllm   || warn "vllm run failed (will diff vs cached if present)" ;;
    *)      die "with --correctness, --only must be sglang|vllm (got '$COMPARE_ONLY')" ;;
  esac

  if [ ! -s "$sgl_json" ] || [ ! -s "$vllm_json" ]; then
    local s="MISSING" v="MISSING"
    [ -s "$sgl_json" ] && s="OK"
    [ -s "$vllm_json" ] && v="OK"
    warn "result file sglang=$s vllm=$v; cannot diff. Re-run the missing side with --only <engine>."
    log "  sglang: $sgl_json (.fail alongside if it crashed)"
    log "  vllm  : $vllm_json"
    return 1
  fi
  "$sgl_py" "$script" diff --sglang "$sgl_json" --vllm "$vllm_json" ${COMPARE_VERBOSE:+--verbose}
}

phase_compare() {
  # --correctness: greedy output-equality (sglang vs vLLM). Separate flow.
  [ "$COMPARE_CORRECTNESS" = "1" ] && { _phase_compare_correctness; return $?; }
  [ "$COMPARE_LIST_ZH" = "1" ] && { _compare_list_zh; [ "$COMPARE_VERBOSE" = "1" ] && _print_compare_commands; return 0; }
  [ "$COMPARE_LIST" = "1" ] && { _compare_list; [ "$COMPARE_VERBOSE" = "1" ] && _print_compare_commands; return 0; }
  log "Phase [compare]: sglang vs vLLM (MRV2 + MRV1) showcase ($COMPARE_SCENARIOS)"
  [ -n "${VIRTUAL_ENV:-}" ] || { source "$VENV_DIR/bin/activate" || die "run 'setup' first"; }
  # Source the SDK env.sh so TP-worker subprocesses inherit PYTHONPATH (SDK
  # python/nne/tvm/pycuda), DLICC_PATH, etc. Without these the scheduler
  # segfaults (-11) at model load. Then lock a clean LD_LIBRARY_PATH via
  # dlin_runtime_env (its overwrite runs after, wins).
  [ -f "$SDK_DIR/env.sh" ] && { source "$SDK_DIR/env.sh" 2>/dev/null || warn "SDK env.sh sourced with warnings"; }
  dlin_runtime_env
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export MODEL_PATH
  export TP_SIZE="${DLIN_TP_SIZE:-4}"
  local script="$SGLANG_DIR/scripts/dl/showcase_prefix_sharing.py"
  local cr="$SGLANG_DIR/scripts/dl/compare_results.py"
  [ -f "$script" ] || die "showcase script not found: $script"
  [ -f "$cr" ] || die "compare_results.py not found: $cr"
  mkdir -p "$COMPARE_METRICS_DIR"

  # --history: just print the results log, no GPU run.
  if [ "$COMPARE_HISTORY" = "1" ]; then
    python "$cr" history --store "$COMPARE_STORE" || warn "no results store yet ($COMPARE_STORE)"
    return 0
  fi

  # Validate --only.
  case "$COMPARE_ONLY" in sglang|vllm-mrv2|vllm-mrv1|"") ;; *) die "COMPARE_ONLY='$COMPARE_ONLY' (want: sglang|vllm-mrv2|vllm-mrv1)";; esac

  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  COMPARE_FAIL_TAGS=""
  # _compare_run_one ENGINE [RUNNER]. vLLM needs a runner (mrv1|mrv2). Caches
  # metrics_{engine}[_{runner}].txt and {tag}_last.log. On failure: removes the
  # stale metrics file (so status=fail is recorded) and returns 1 (non-fatal for vLLM).
  _compare_run_one() {
    local eng="$1" runner="${2:-}" tag mfile vllm_args="" logf cmd
    if [ "$eng" = "vllm" ]; then
      tag="vllm-$runner"; mfile="metrics_vllm_${runner}.txt"; vllm_args="--vllm-runner $runner"
    else
      tag="$eng"; mfile="metrics_${eng}.txt"
    fi
    logf="$COMPARE_METRICS_DIR/${tag}_${stamp}.log"
    # DL: log the FULL command + config (reproducibility + fairness audit) to console + log head.
    cmd="python $script --engine $eng $vllm_args --mem-frac $COMPARE_MEM_FRAC --scenarios $COMPARE_SCENARIOS"
    log "[compare] running $tag (scenarios=$COMPARE_SCENARIOS, log: $logf)"
    log "[compare] COMMAND: $cmd"
    if [ "$COMPARE_VERBOSE" = "1" ]; then
      echo ""
      echo "============================================================"
      echo "  Running: $tag"
      echo "============================================================"
      echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
      echo "  MODEL_PATH=${MODEL_PATH:-}"
      echo "  TP_SIZE=${DLIN_TP_SIZE:-4}"
      echo "  ATTN_BACKEND=${ATTN_BACKEND:-fa3}"
      echo "  MEM_FRAC=$COMPARE_MEM_FRAC"
      echo "  SCENARIOS=$COMPARE_SCENARIOS"
      echo ""
      echo "  \$ $cmd"
      echo ""
      if [ "$eng" = "sglang" ]; then
        echo "  Internal engine launch:"
        echo "  engine = sgl.Engine("
        echo "      model_path=${MODEL_PATH:-<MODEL_PATH>},"
        echo "      tp_size=${DLIN_TP_SIZE:-4},"
        echo "      dtype=bfloat16,"
        echo "      context_length=4096,"
        echo "      mem_fraction_static=$COMPARE_MEM_FRAC,"
        echo "      max_running_requests=4,"
        echo "      disable_cuda_graph=False,"
        echo "      cuda_graph_max_bs_decode=4,"
        echo "      attention_backend=${ATTN_BACKEND:-fa3},"
        echo "      page_size=16,"
        echo "      disable_custom_all_reduce=True,"
        echo "      trust_remote_code=True,"
        echo "      chunked_prefill_size=512,"
        echo "  )"
      else
        local _mrv2="1"; [ "$runner" = "mrv1" ] && _mrv2="0"
        echo "  env: VLLM_USE_V2_MODEL_RUNNER=$_mrv2"
        echo "  Internal engine launch:"
        echo "  llm = LLM("
        echo "      model=${MODEL_PATH:-<MODEL_PATH>},"
        echo "      tensor_parallel_size=${DLIN_TP_SIZE:-4},"
        echo "      dtype=bfloat16,"
        echo "      max_model_len=4096,"
        echo "      gpu_memory_utilization=$COMPARE_MEM_FRAC,"
        echo "      trust_remote_code=True,"
        echo "      max_num_seqs=4,"
        echo "      disable_log_stats=True,"
        if [ "$runner" = "mrv1" ]; then
          echo "      enable_prefix_caching=True,   # MRV1 only (APC)"
          echo "      compilation_config={mode: NONE,"
          echo "                           cudagraph_capture_sizes: [1,2,4,528],"
          echo "                           max_cudagraph_capture_size: 528,"
        else
          echo "      # enable_prefix_caching not set (MRV2: APC unsupported)"
          echo "      compilation_config={"
        fi
        echo "                           cudagraph_capture_sizes: [1,2,4],"
        echo "                           max_cudagraph_capture_size: 4},"
        echo "  )"
      fi
      echo "============================================================"
      echo ""
    fi
    if ! { echo "[showcase] === COMMAND: $cmd ===";
           echo "[showcase] === CONFIG: model=$(basename ${MODEL_PATH%/}) tp=${DLIN_TP_SIZE:-4} mem-frac=$COMPARE_MEM_FRAC backend=${ATTN_BACKEND:-fa3} temp=0(SC8/SC8b=0.7) scenarios=$COMPARE_SCENARIOS ===";
           python "$script" --engine "$eng" $vllm_args --mem-frac "$COMPARE_MEM_FRAC" \
                  --scenarios "$COMPARE_SCENARIOS"; } > "$logf" 2>&1; then
      warn "$tag run FAILED. Tail of $logf:"; tail -n 18 "$logf" 2>/dev/null
      rm -f "$COMPARE_METRICS_DIR/$mfile"
      cp -f "$logf" "$COMPARE_METRICS_DIR/${tag}_last.log" 2>/dev/null || true
      COMPARE_FAIL_TAGS="$COMPARE_FAIL_TAGS $tag"
      return 1
    fi
    sed -n '/^=== METRICS /,/^=== END METRICS ===/p' "$logf" | grep '^METRIC ' \
        > "$COMPARE_METRICS_DIR/$mfile" || true
    cp -f "$logf" "$COMPARE_METRICS_DIR/${tag}_last.log"
    ok "$tag done ($(wc -l < "$COMPARE_METRICS_DIR/$mfile") metrics cached)."
  }

  # Decide what to run.
  if [ "$COMPARE_SHOW" = "1" ]; then
    log "[compare] --show: re-rendering from cached metrics ($COMPARE_METRICS_DIR), no GPU run."
  else
    case "$COMPARE_ONLY" in
      sglang)      _compare_run_one sglang     || die "sglang run failed (cannot compare without it)" ;;
      vllm-mrv2)   _compare_run_one vllm mrv2  || warn "vllm-mrv2 failed (will be recorded as fail)" ;;
      vllm-mrv1)   _compare_run_one vllm mrv1  || warn "vllm-mrv1 failed (recorded as fail)" ;;
      "")
        _compare_run_one sglang    || die "sglang run failed (cannot compare without it)"
        _compare_run_one vllm mrv2 || warn "vllm-mrv2 failed (recorded as fail)"
        # DL: MRV1 opt-in (APC+CG-off, works but adds ~10min to runtime).
        if [ "$COMPARE_RUN_MRV1" = "1" ]; then
          _compare_run_one vllm mrv1 || warn "vllm-mrv1 failed (recorded as fail)"
        else
          log "[compare] skipping vllm-mrv1 (default; adds ~10min). Set COMPARE_RUN_MRV1=1 to include."
        fi
        ;;
    esac
  fi

  # ---- Record to JSON store (skip for --show/--no-record) ----
  if [ "$COMPARE_SHOW" != "1" ] && [ "$COMPARE_RECORD" = "1" ]; then
    local _commit _branch _dirty _sdk_base
    _commit=$(git -C "$SGLANG_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
    _branch=$(git -C "$SGLANG_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if git -C "$SGLANG_DIR" diff --quiet 2>/dev/null; then _dirty=""; else _dirty="--dirty"; fi
    _sdk_base=$(basename "${SDK_DIR:-}")
    local cr_args=(append --store "$COMPARE_STORE" --commit "$_commit" --branch "$_branch"
      ${_dirty} --model "$(basename "${MODEL_PATH%/}")" --tp "${DLIN_TP_SIZE:-4}"
      --sdk "$_sdk_base" --mem-frac "$COMPARE_MEM_FRAC" --scenarios "$COMPARE_SCENARIOS")
    # Per engine: --metrics if file non-empty, else --fail (tail of its last log).
    if [ -s "$COMPARE_METRICS_DIR/metrics_sglang.txt" ]; then
      cr_args+=(--metrics "sglang=$COMPARE_METRICS_DIR/metrics_sglang.txt")
    else cr_args+=(--fail "sglang=$COMPARE_METRICS_DIR/sglang_last.log"); fi
    if [ -s "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" ]; then
      cr_args+=(--metrics "vllm_mrv2=$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt")
    else cr_args+=(--fail "vllm_mrv2=$COMPARE_METRICS_DIR/vllm-mrv2_last.log"); fi
    if [ -s "$COMPARE_METRICS_DIR/metrics_vllm_mrv1.txt" ]; then
      cr_args+=(--metrics "vllm_mrv1=$COMPARE_METRICS_DIR/metrics_vllm_mrv1.txt")
    else cr_args+=(--fail "vllm_mrv1=$COMPARE_METRICS_DIR/vllm-mrv1_last.log"); fi
    python "$cr" "${cr_args[@]}" || warn "failed to append to results store"
  fi

  # ---- 3-column side-by-side gap table (sglang | MRV2 | MRV1) ----
  echo
  log "============= sglang vs vLLM (MRV2 + MRV1) — showcase gap ============="
  local model_tag
  model_tag=$(grep -m1 -h '^=== METRICS ' "$COMPARE_METRICS_DIR/sglang_last.log" 2>/dev/null \
              | sed -n 's/.*model=\([^ ]*\).*/\1/p' || true)
  [ -z "$model_tag" ] && model_tag=$(basename "${MODEL_PATH%/}")
  echo  "  model=$model_tag  tp=${DLIN_TP_SIZE:-4}  scenarios=$COMPARE_SCENARIOS"
  echo  "  (same GPUs, FP8, fresh process each; vLLM MRV1+CG+APC ON = FAIR baseline"
  echo  "   (MRV2 can't APC on hybrid-Mamba, recorded as fail); sglang uses the GDN"
  echo  "   dl_chunk flag SGLANG_DL_GDN_DLIN_EXTEND=1 (default-on) + RadixAttention.)"
  echo  "  fairness audit: same model/TP/temp/warmup; see each {tag}.log COMMAND+CONFIG header."
  echo  "  With the GDN flag: sglang WINS most prefix-reuse + decode (RadixAttention reuses"
  echo  "   the full hybrid-Mamba state; vLLM APC only caches attention KV). vLLM still wins"
  echo  "   raw/unique prefill (SC6) + one-time cold-prefill. dl_safe_reset.sh before each launch."
  echo  "  $(date '+%Y-%m-%d %H:%M:%S')"
  echo  "  -----------------------------------------------------------------------"
  echo  "  metric                   | sglang    | vLLM-MRV2 | vLLM-MRV1 | sglang vs MRV2"
  echo  "  -------------------------+-----------+-----------+-----------+----------------"
  _compare_row "SC1 warm latency (ms)"   SC1_warm_ms        lower
  _compare_row "SC1 cold->warm speedup"  SC1_speedup_x      higher
  _compare_row "SC2 avg turn (ms)"       SC2_avg_ms         lower
  _compare_row "SC2 turn-5 (ms)"         SC2_turn5_ms       lower
  _compare_row "SC3 throughput (tok/s)"  SC3_throughput_tps higher
  _compare_row "SC3 per-req (ms)"        SC3_per_req_ms     lower
  if grep -q '^METRIC SC4_' "$COMPARE_METRICS_DIR/metrics_sglang.txt" 2>/dev/null \
     || grep -q '^METRIC SC4_' "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" 2>/dev/null; then
    _compare_row "SC4 JSON (tok/s)"      SC4_tps            higher
  fi
  # DL: SC5 multi-user fork (radix tree), SC7 long-RAG throughput, SC8 parallel
  # sampling (decode-bound control). Conditional rows — render only if the
  # scenario was run (--scenarios SC5,SC7,SC8). See showcase_prefix_sharing.py.
  if grep -q '^METRIC SC5_' "$COMPARE_METRICS_DIR/metrics_sglang.txt" 2>/dev/null \
     || grep -q '^METRIC SC5_' "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" 2>/dev/null; then
    _compare_row "SC5 fork total (ms)"   SC5_total_ms       lower
    _compare_row "SC5 fork avg-turn(ms)" SC5_avg_turn_ms    lower
  fi
  if grep -q '^METRIC SC7_' "$COMPARE_METRICS_DIR/metrics_sglang.txt" 2>/dev/null \
     || grep -q '^METRIC SC7_' "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" 2>/dev/null; then
    _compare_row "SC7 long-RAG (tok/s)"  SC7_throughput_tps higher
  fi
  if grep -q '^METRIC SC8_' "$COMPARE_METRICS_DIR/metrics_sglang.txt" 2>/dev/null \
     || grep -q '^METRIC SC8_' "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" 2>/dev/null; then
    _compare_row "SC8 parallel-samp(t/s)" SC8_tps           higher
  fi
  if grep -q '^METRIC SC9_' "$COMPARE_METRICS_DIR/metrics_sglang.txt" 2>/dev/null \
     || grep -q '^METRIC SC9_' "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" 2>/dev/null; then
    _compare_row "SC9 pure decode (t/s)"  SC9_tps           higher
  fi
  if grep -q '^METRIC SC10_' "$COMPARE_METRICS_DIR/metrics_sglang.txt" 2>/dev/null \
     || grep -q '^METRIC SC10_' "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" 2>/dev/null; then
    _compare_row "SC10 sys-prompt(t/s)"  SC10_throughput_tps higher
  fi
  if grep -q '^METRIC SC11_' "$COMPARE_METRICS_DIR/metrics_sglang.txt" 2>/dev/null \
     || grep -q '^METRIC SC11_' "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" 2>/dev/null; then
    _compare_row "SC11 online-conc(t/s)" SC11_throughput_tps higher
  fi
  # DL: rigor diagnostic probes (run via --scenarios SC6 / SC8B). See
  # docs/dl/sglang-vs-vllm-rigor-analysis.md. SC6 = raw-prefill parity (unique
  # prompts, no cache); SC8B = cold single-call best-of-N (no cross-call cache).
  if grep -q '^METRIC SC6_' "$COMPARE_METRICS_DIR/metrics_sglang.txt" 2>/dev/null \
     || grep -q '^METRIC SC6_' "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" 2>/dev/null; then
    _compare_row "SC6 raw-prefill(t/s)"  SC6_prefill_tps     higher
  fi
  if grep -q '^METRIC SC8b_' "$COMPARE_METRICS_DIR/metrics_sglang.txt" 2>/dev/null \
     || grep -q '^METRIC SC8b_' "$COMPARE_METRICS_DIR/metrics_vllm_mrv2.txt" 2>/dev/null; then
    _compare_row "SC8b cold best-of-N(t/s)" SC8b_cold_tps    higher
  fi
  echo  "  ======================================================================="
  log "metrics cached: $COMPARE_METRICS_DIR/metrics_{sglang,vllm_mrv1,vllm_mrv2}.txt"
  log "results store:  $COMPARE_STORE   (./run_sglang.sh compare --history to view)"

  # ---- Diff vs previous run (skip for --show) ----
  if [ "$COMPARE_SHOW" != "1" ] && [ "$COMPARE_RECORD" = "1" ]; then
    echo
    local diff_args=(diff --store "$COMPARE_STORE")
    [ -n "$COMPARE_BASELINE" ] && diff_args+=(--baseline "$COMPARE_BASELINE")
    python "$cr" "${diff_args[@]}" || true
  fi
}

#-------------------------------------------------------------------------------
# Dispatch. gen/serve parse their trailing flags first; everything else ignores
# extra args (backward compatible with the original positional phases).
#-------------------------------------------------------------------------------
PHASE="${1:-all}"; shift || true
case "$PHASE" in
  gen|serve|bench|benchrun|chat) parse_test_args "$@" ;;
  compare)                        parse_test_args "$@" ;;
  sop)                            parse_test_args "$@" ;;
esac
# One-click default: gen/serve with no -m/-M -> Qwen3.5-35B-A3B-FP8 (TP4 preset).
if [ -z "${MODEL_EXPLICIT:-}" ]; then
  case "$PHASE" in gen|serve|compare) pick_model qwen35-35b ;; esac
  # chat: no model preset needed — connects to an existing server
fi
apply_ngram_overrides   # no-op unless -S/--spec-ngram; must run AFTER pick_model

case "$PHASE" in
  setup)        phase_setup ;;
  build-kernel) phase_setup 2>/dev/null || true; source "$VENV_DIR/bin/activate" 2>/dev/null || true; phase_build_kernel ;;
  install)      source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_install ;;
  test|smoke)   source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_test ;;
  gen)          source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_gen ;;
  serve)        source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_serve ;;
  bench)        source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_bench ;;
  benchrun)     source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_benchrun ;;
  chat)         source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_chat ;;
  sop)          source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_sop ;;
  compare)      source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_compare ;;
  all)
    phase_setup
    phase_build_kernel || warn "build-kernel phase best-effort; continuing"
    phase_install
    phase_test
    ;;
  -h|--help|help) usage; exit 0 ;;
  *) die "unknown phase '$PHASE' (use: setup|build-kernel|install|test|smoke|gen|serve|bench|benchrun|chat|sop|compare|all)" ;;
esac
ok "Done ($PHASE)."

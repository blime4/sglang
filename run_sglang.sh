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
DL_WARMUP="${DL_WARMUP:-0}"    # -W/--dl-warmup: pre-compile fused-MoE prefill-M dlcc shapes at serve start
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
# of the other. vLLM runs APC-OFF — its prefix cache can't be enabled on this
# hybrid Mamba model (MRV2 rejects mamba_cache_mode='align'); see docs/dl/
# sglang-vs-vllm-showcase-dlin.md.
COMPARE_SCENARIOS="${COMPARE_SCENARIOS:-SC1,SC2,SC3}"
COMPARE_ONLY="${COMPARE_ONLY:-}"                # --only sglang|vllm-mrv2|vllm-mrv1
COMPARE_SHOW="${COMPARE_SHOW:-0}"               # --show: re-render table from cache, no GPU run
COMPARE_HISTORY="${COMPARE_HISTORY:-0}"         # --history: print the results log, no GPU run
COMPARE_LIST="${COMPARE_LIST:-0}"               # --list: print all scenarios + ASCII diagrams, no GPU run
COMPARE_RECORD="${COMPARE_RECORD:-1}"           # --no-record: don't append to the JSON store
COMPARE_RUN_MRV1="${COMPARE_RUN_MRV1:-0}"       # DL: MRV1 opt-in (APC+CG-off, runs but +10min)
COMPARE_BASELINE="${COMPARE_BASELINE:-}"        # --baseline <id|commit>: diff vs this (else previous run)
COMPARE_VERBOSE="${COMPARE_VERBOSE:-0}"         # --verbose: print engine commands
COMPARE_MEM_FRAC="${COMPARE_MEM_FRAC:-0.55}"
COMPARE_METRICS_DIR="${COMPARE_METRICS_DIR:-/tmp/sglang_compare}"
COMPARE_STORE="${COMPARE_STORE:-$SGLANG_DIR/docs/dl/compare_results.json}"

# chat options (wraps scripts/dl/chat.py — interactive OpenAI-compatible client).
# Default: connect to $SERVE_HOST:$SERVE_PORT, auto-detect model.
CHAT_QUICK="${CHAT_QUICK:-}"                        # -q: single message
CHAT_URL="${CHAT_URL:-}"                            # --url: server API base URL
CHAT_MODEL="${CHAT_MODEL:-}"                        # --chat-model: explicit model name
CHAT_SYSTEM_PROMPT="${CHAT_SYSTEM_PROMPT:-}"        # --system-prompt
CHAT_NO_STREAM="${CHAT_NO_STREAM:-0}"                # --no-stream

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
      # Matches the tuned TP4 config in scripts/dl/compare_tp4.py (~27ms TPOT = ~vLLM parity):
      # FP8 Q2 GEMM, DLIN GDN op, multi-step decode, FLA pingpong/unroll.
      export SGLANG_DL_FP8_Q2=1 SGLANG_DL_GDN_DLIN=1 SGLANG_DL_MULTI_STEP=1
      # DL: GDN prefill(extend) root cause + identified fix (NOT yet default-on).
      # The default triton extend path is ~8x slower than vLLM and is 89% of prefill
      # (self_attention/GDN; MoE only 11%). SGLANG_DL_GDN_DLIN_EXTEND=1 routes extend
      # to the DLIN dl_chunk kernel -> 2K prefill 41s->5.1s (8x), correct ("capital of
      # France"->" Paris"), sglang then beats vLLM on SC1-warm (0.96 vs 1.2s) & SC3
      # (66.6 vs 41.9 tok/s). BUT dl_chunk is UNSTABLE on TP4: later/broader runs hit
      # NCCL collective-timeout desyncs (hung [sglang::schedul]). Re-verify in a clean
      # env (fresh cards/cache) and stabilize dl_chunk before enabling broadly.
      # To try: export SGLANG_DL_GDN_DLIN_EXTEND=1   (gdn_backend.py:76-93). 2026-07-28.
      # export SGLANG_DL_GDN_DLIN_EXTEND=1   # <-- OPT-IN (crashes full compare today)
      export DLEOL_CU_ADDRESS_CHECK=0 DLEOL_FLA_ENABLE_PINGPONG=1 DLEOL_FLA_UNROLL_COUNT=8
      # TP=4 needs 4 GPUs; ensure CUDA_VISIBLE_DEVICES lists >=4 devices.
      local _ndev
      _ndev=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
      if [ "$_ndev" -lt 4 ]; then
        CUDA_VISIBLE_DEVICES="0,1,2,3"
        log "TP=4: set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (override with CUDA_VISIBLE_DEVICES=...)"
      fi
      ;;
    *) die "unknown preset '$1'. Available: qwen3-1.7b qwen35-35b" ;;
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
# JFrog credential helper — stores credentials in ~/.netrc (chmod 600).
# First run: prompts for username/password, saves.
# Subsequent runs: silent (wget/curl/pip/uv all read ~/.netrc natively).
#-------------------------------------------------------------------------------
ensure_jfrog_credentials() {
  local host="$DL_HOST"
  local netrc="$HOME/.netrc"

  if grep -q "machine ${host}" "$netrc" 2>/dev/null; then
    return 0
  fi

  log "JFrog credentials needed for $host (stored in ~/.netrc, chmod 600)"
  printf "  JFrog username: "; read -r jfrog_user
  printf "  JFrog password: "; read -rs jfrog_pass; echo

  if [ -z "$jfrog_user" ] || [ -z "$jfrog_pass" ]; then
    die "Username and password are required"
  fi

  # Append entry (create file if missing)
  printf "\nmachine %s\n  login %s\n  password %s\n" \
    "$host" "$jfrog_user" "$jfrog_pass" >> "$netrc"
  chmod 600 "$netrc"
  log "Credentials saved to ~/.netrc (chmod 600). Will be reused automatically."
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
    log "Downloading DLIN triton 3.3.0 (dlgpu backend) ..."
    ensure_jfrog_credentials
    wget --netrc -q --show-progress -O "$triton_whl" \
      "${TRITON_URL//%2B/+}" || die "Failed to download DLIN triton"
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
  [ -f "$bak" ] || cp "$py" "$bak"
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
  # Back up the CUDA pyproject once, then swap in the DLIN variant.
  if [ ! -f "$bak" ]; then cp "$py" "$bak"; fi
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
  # FULL showcase (all 8 scenarios, ~25-30 min; vLLM re-prefills so it is slow):
  ./run_sglang.sh compare --scenarios SC1,SC2,SC3,SC5,SC7,SC8,SC9,SC10
  ./run_sglang.sh chat                           # interactive chat (connect to existing server on :30000)
  ./run_sglang.sh chat -q "hello"                # quick single message
  ./run_sglang.sh chat --url http://10.0.0.1:30000/v1   # custom endpoint
  ./run_sglang.sh chat --chat-model Qwen3-1.7B --system-prompt "You are helpful"
  Scenarios: SC1 prefix-share, SC2 multi-turn, SC3 batch, SC4 JSON,
    SC5 multi-user fork (radix tree), SC7 long-RAG throughput, SC8 parallel
    sampling (best-of-N), SC9 pure long decode (decode-bound control),
    SC10 shared system-prompt throughput. SC5/SC7/SC8/SC10 = sglang wins
    (RadixAttention); SC9 = vLLM-favored control. See
    docs/dl/sglang-vs-vllm-new-scenarios.md.
  Runs both engines on the same GPUs (fresh process each), caches metrics to
  /tmp/sglang_compare. vLLM runs APC-OFF — its prefix cache can't be enabled on
  this hybrid Mamba model (MRV2 rejects mamba_cache_mode='align'). See
  docs/dl/sglang-vs-vllm-showcase-dlin.md.

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
      --no-record)         COMPARE_RECORD=0; shift;;
       --baseline)          COMPARE_BASELINE="$2"; shift 2;;
       --verbose)           COMPARE_VERBOSE=1; shift;;
      # chat
      -q|--quick)           CHAT_QUICK="$2"; shift 2;;
      --url)                CHAT_URL="$2"; shift 2;;
      --chat-model)         CHAT_MODEL="$2"; shift 2;;
      --system-prompt)      CHAT_SYSTEM_PROMPT="$2"; shift 2;;
      --no-stream)          CHAT_NO_STREAM=1; shift;;
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
  # Targets sglang raw-prefill 1.55x slower than vLLM (SC6: 49 vs 76 tok/s) by
  # capturing prefill instead of eager. See opt-plan P1-1.
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
  # Targets sglang raw-prefill 1.55x slower than vLLM (SC6: 49 vs 76 tok/s) by
  # capturing prefill instead of eager. See opt-plan P1-1.
  [ -n "$DL_PREFILL_BACKEND" ] && extra_flags="$extra_flags --cuda-graph-backend-prefill $DL_PREFILL_BACKEND"
  local ngram_flags=""
  [ "$USE_NGRAM" = "1" ] && ngram_flags="--speculative-algorithm NGRAM --speculative-num-draft-tokens $NGRAM_NUM_DRAFT --speculative-ngram-min-bfs-breadth $NGRAM_MIN_BFS --speculative-ngram-max-bfs-breadth $NGRAM_MAX_BFS"
  BENCH_LOG="${BENCH_LOG:-/tmp/sglang_bench_server.log}"
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
#                       sampling (best-of-N), SC9=pure long decode (control),
#                       SC10=shared system-prompt throughput. The full showcase
#                       (all 8) is slow on vLLM (it re-prefills); ~25-30 min.
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
vLLM runs APC-OFF (its prefix cache is structurally unsupported on hybrid-Mamba),
so it RE-PREFILLS shared prefixes; sglang RadixAttention keeps the shared KV.
  => sglang wins every KV-reuse workload; vLLM wins raw decode.
legend: [sglang win ~Nx] / [vLLM win ~Nx]   [probe] = rigor diagnostic

---------------------------------------------------------------------------
SC1  prefix-sharing  (flat prefix)                  [sglang win ~2.3x warm / 16.4x cold->warm]
     [shared prefix] -- req1 (independent suffix)
                    -- req2
                    -- req3
     cold: prefill the prefix; warm: cache hit -> skip prefill entirely.

SC2  multi-turn  (single linear conversation)        [sglang win ~1.6x]
     sys+history -> turn1 -> turn2 -> turn3 -> turn4 -> turn5   (one user, growing)
     each turn extends the linear prefix; vLLM re-prefills the whole history each turn.

SC3  concurrent batch  (one prefix, batched decodes) [sglang win ~2.3x  (prefill 3.4x)]
     [shared prompt] -- fork to a BATCH of N decodes at once (one call, many outputs)
     aggregate batch prefill+decode throughput. Prefill-bound (fused FP8 MoE).

SC4  structured JSON  (short constrained decode)     [vLLM win ~+9%]
     prompt -> { "name": "...", "age": ... }   (greedy JSON)
     decode-bound, short output, NO prefix reuse -> raw decode + JSON path decides it.

SC5  multi-user fork  (radix TREE, shared root)      [sglang win ~8.2x]
                [shared system-prompt root]   <- prefill ONCE, KV shared
                   /              \
            user-A branch        user-B branch
          turn1->2->3->4         turn1->2->3->4    (interleaved, branching tree)
     multi-tenant / many users behind one system prompt.

SC6  raw-prefill parity  (UNIQUE prompts, NO cache)  [probe -- vLLM actually ~1.55x FASTER]
     unique prompt1   unique prompt2   ...   (distinct prefixes -> cache cannot hit)
     isolates RAW prefill rate. Proves the SC5/7/8/10 wins are 100% caching,
     NOT faster raw prefill (sglang 49 < vLLM 76 tok/s here).

SC7  long-RAG throughput  (~2K doc x 8 queries)      [sglang win ~16.3x]
     [~2K-token doc]   <- prefill once (sglang) / re-prefill EACH query (vLLM)
          |- Q1 |
          |- Q2 |    8 diverse queries over the SAME doc; aggregate decode tok/s.
          |- ...|    RAG over a long shared document.

SC8  repeated best-of-N  (RLHF loop, same prompt)    [sglang win ~5.8x  (= cache 3.9x x single 2.0x)]
     [prompt]  <- reused across reps; sglang caches it
       /  |  |  \     n=4 sampled completions (temp=0.7)
      c1 c2 c3 c4
     RLHF rejection-sampling loop. Cache-dominated; single-call best-of-N is SC8b (~2x).

SC8b cold single-call best-of-N  (UNIQUE prompt)     [probe -- sglang ~2x]
     unique prompt per call -> n=4 samples (no cross-call cache).
     isolates single-call best-of-N edge from SC8's caching factor.

SC9  pure long decode  (short prompt + 128 tok)      [vLLM win ~1.1-1.3x  (control)]
     [short ~14-tok prompt] -> decode 128 tokens (single stream, no shared structure)
     decode-bound, nothing to cache -> vLLM's raw decode-IPC edge wins. The honest loss.

SC10 shared system-prompt  (many tenants)            [sglang win ~7.1x]
     [~0.9K system prompt]   <- shared by 12 tenants
          |- tenant1 |
          |- tenant2 |   24 short reqs, same persona; canonical RadixAttention-in-prod.
          |- ...     |
---------------------------------------------------------------------------
Full docs: docs/dl/sglang-vs-vllm-showcase-dlin.md (SC1-4),
           docs/dl/sglang-vs-vllm-new-scenarios.md (SC5/7/8/9/10),
           docs/dl/sglang-vs-vllm-rigor-analysis.md (SC6/SC8b probes).
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
  echo "  SGLANG_DL_MOE_FUSED_MAX_M=32"
  echo "  SGLANG_DL_GDN_DLIN=1"
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
  echo "  --- vLLM MRV2 (LLM + CG, APC-OFF) ---"
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
  echo "  NOTE: MRV2 does NOT support APC (mamba_cache_mode='align' assertion)."
  echo ""
  echo "  --- vLLM MRV1 (LLM + CG + APC) ---"
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
  echo "                           cudagraph_capture_sizes: [1,2,4],"
  echo "                           max_cudagraph_capture_size: 4},"
  echo "  )"
  echo "  env: VLLM_USE_V2_MODEL_RUNNER=0"
  echo "========================================================================="
  echo ""
}

phase_compare() {
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
  echo  "  (same GPUs, FP8, fresh process each; vLLM APC-OFF (structurally unsupported on"
  echo  "   hybrid-Mamba -> re-prefills shared prefixes); MRV1 APC+CG-on opt-in)"
  echo  "  fairness audit: same model/TP/temp/warmup; see each {tag}.log COMMAND+CONFIG header."
  echo  "  win source: sglang KV-reuse > MRV2 (default) via RadixAttention, NOT raw speed;"
  echo  "   MRV1 APC+CG-on has faster DLIN native path and can outrun sglang."
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
  compare)      source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_compare ;;
  all)
    phase_setup
    phase_build_kernel || warn "build-kernel phase best-effort; continuing"
    phase_install
    phase_test
    ;;
  -h|--help|help) usage; exit 0 ;;
  *) die "unknown phase '$PHASE' (use: setup|build-kernel|install|test|smoke|gen|serve|bench|benchrun|chat|compare|all)" ;;
esac
ok "Done ($PHASE)."

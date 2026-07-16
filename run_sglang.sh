#!/bin/bash
#===============================================================================
# run_sglang.sh
#-------------------------------------------------------------------------------
# One-stop driver for building/running SGLang on DLIN (登临) GPUs.
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
SDK_DIR="${SDK_DIR:-/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sdk}"
ARTIFACTORY_DIR="${ARTIFACTORY_DIR:-/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/flash-attention/artifactory}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-$SGLANG_DIR/.venv}"

# DLIN-patched torch that exposes torch.version.dl (NOT vanilla upstream 2.9.1).
TORCH_SPEC="${TORCH_SPEC:-torch==2.9.1+dl24.sdk202606031721}"

# DLIN-pinned triton. MUST be 3.1.0: its bundled LLVM matches the SDK's
# libLLVM-15.so (LLVM 15) so they coexist; triton>=3.2 bundles a newer LLVM that
# collides via symbol interposition -> PassBuilder static-init segfault (plan §6).
# Installed explicitly from dl-virtual so uv doesn't grab the vanilla build.
TRITON_SPEC="${TRITON_SPEC:-triton==3.1.0}"

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
      MODEL_PATH="/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
      DLIN_TP_SIZE=4; USE_CUDA_GRAPH=1; DLIN_CG_MAX_BS=2
      # mem_fraction=0.60 (not 0.85): the stable bf16-bmm prefill path (M>16) dequants
      # expert weights and needs several GB headroom, else OOM (docs §7.14). 0.85 OOMs.
      DLIN_MEM_FRACTION=0.60; DLIN_CONTEXT_LEN=4096; DLIN_PAGE_SIZE=16
      # DLIN MoE routing (see fp8.py + docs §7.13/§7.14):
      #   FUSED_MAX_M=16  — fused (invoke_fused_moe_opt) for decode M=1 + NGRAM verify
      #                     M≈9 + short prefill. CRASHES (dleol tu_program.cc:625) for
      #                     M>=~100, so kept small for serving stability.
      #   MAX_BF16_M=2048 — bf16-bmm (torch-native, stable) for larger prefill (17-2048).
      #                     Slow but the only non-crashing path on dl24.
      export SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=16 SGLANG_DL_MOE_MAX_BF16_M=2048
      # Matches the tuned TP4 config in scripts/dl/compare_tp4.py (~27ms TPOT = ~vLLM parity):
      # FP8 Q2 GEMM, DLIN GDN op, multi-step decode, FLA pingpong/unroll.
      export SGLANG_DL_FP8_Q2=1 SGLANG_DL_GDN_DLIN=1 SGLANG_DL_MULTI_STEP=1
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
# uv config helpers — mirror the artifactory method (uv.toml / pip.conf) but add
# dl-virtual as an extra index so the +dl24 DLIN torch resolves.
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
  # Install DLIN-pinned triton 3.1.0 from dl-virtual BEFORE the editable install
  # so uv doesn't resolve a vanilla build from dl-pypi-remote and so torch's
  # triton dependency is already satisfied by the correct version.
  log "Installing DLIN triton ($TRITON_SPEC from dl-virtual) ..."
  uv pip install --index-url "$DL_VIRTUAL_INDEX" --trusted-host "$DL_HOST" \
      --no-deps "$TRITON_SPEC"
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
# the DLIN-pinned triton 3.1.0 (whose bundled LLVM matches the SDK's libLLVM-15
# so they coexist — see DLIN_INTEGRATION_PLAN.md §6).
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
run_sglang.sh — build/run SGLang on DLIN (登临) GPUs.

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
  --port N / --host H       serve only        (default 30000 / 127.0.0.1)

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
  [ "$USE_CUDA_GRAPH" = "1" ] || cg_flag="--disable-cuda-graph"
  local extra_flags=""
  [ -n "$mem_frac" ] && extra_flags="$extra_flags --mem-fraction-static $mem_frac"
  [ -n "$ctx_len" ] && extra_flags="$extra_flags --context-length $ctx_len"
  [ -n "$cg_max_bs" ] && [ "$USE_CUDA_GRAPH" = "1" ] && extra_flags="$extra_flags --cuda-graph-max-bs-decode $cg_max_bs"
  # DL: TP>1 on DLIN must use NCCL — the custom allreduce kernel hits HC_CUK Error=28.
  [ "$tp" -gt 1 ] && extra_flags="$extra_flags --disable-custom-all-reduce"
  local ngram_flags=""
  [ "$USE_NGRAM" = "1" ] && ngram_flags="--speculative-algorithm NGRAM --speculative-num-draft-tokens $NGRAM_NUM_DRAFT --speculative-ngram-min-bfs-breadth $NGRAM_MIN_BFS --speculative-ngram-max-bfs-breadth $NGRAM_MAX_BFS"
  log "model=$MODEL_PATH | tp=$tp | backend=$ATTN_BACKEND | page=$page_size | host=$SERVE_HOST:$SERVE_PORT | ngram=$USE_NGRAM"
  exec python -m sglang.launch_server \
    --model-path "$MODEL_PATH" --page-size "$page_size" --dtype bfloat16 \
    --tp-size "$tp" \
    --attention-backend "$ATTN_BACKEND" $cg_flag $extra_flags $ngram_flags \
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
  [ "$USE_CUDA_GRAPH" = "1" ] || cg_flag="--disable-cuda-graph"
  local extra_flags=""
  [ -n "$mem_frac" ] && extra_flags="$extra_flags --mem-fraction-static $mem_frac"
  [ -n "$ctx_len" ] && extra_flags="$extra_flags --context-length $ctx_len"
  [ -n "$cg_max_bs" ] && [ "$USE_CUDA_GRAPH" = "1" ] && extra_flags="$extra_flags --cuda-graph-max-bs-decode $cg_max_bs"
  # DL: TP>1 on DLIN must use NCCL — the custom allreduce kernel hits HC_CUK Error=28.
  [ "$tp" -gt 1 ] && extra_flags="$extra_flags --disable-custom-all-reduce"
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
# Dispatch. gen/serve parse their trailing flags first; everything else ignores
# extra args (backward compatible with the original positional phases).
#-------------------------------------------------------------------------------
PHASE="${1:-all}"; shift || true
case "$PHASE" in
  gen|serve|bench|benchrun) parse_test_args "$@" ;;
esac
# One-click default: gen/serve with no -m/-M -> Qwen3.5-35B-A3B-FP8 (TP4 preset).
if [ -z "${MODEL_EXPLICIT:-}" ]; then
  case "$PHASE" in gen|serve) pick_model qwen35-35b ;; esac
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
  all)
    phase_setup
    phase_build_kernel || warn "build-kernel phase best-effort; continuing"
    phase_install
    phase_test
    ;;
  -h|--help|help) usage; exit 0 ;;
  *) die "unknown phase '$PHASE' (use: setup|build-kernel|install|test|smoke|gen|serve|bench|benchrun|all)" ;;
esac
ok "Done ($PHASE)."

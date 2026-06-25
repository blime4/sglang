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
#   test         import smoke (torch.version.dl + import sglang) + a CPU UT.
#   all          setup -> build-kernel -> install -> test  (default)
#
# Usage:
#   ./run_sglang.sh             # run all phases
#   ./run_sglang.sh setup       # just create/refresh the env
#   ./run_sglang.sh test        # run the basic UT against an existing env
#
# Env overrides (all optional):
#   SDK_DIR          DLIN SDK root (env.sh). Default: see below.
#   ARTIFACTORY_DIR  repo with piplike/uv.toml/pip.conf. Default: see below.
#   VENV_DIR         venv location (default: .venv).
#   PYTHON_VERSION   default 3.12.
#   TORCH_SPEC       torch pin (default: 2.9.1+dl24.sdk202606031721).
#   SKIP_KERNEL=1    skip the build-kernel phase.
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

PHASE="${1:-all}"

log()  { echo -e "\033[1;34m[run_sglang]\033[0m $*"; }
warn() { echo -e "\033[1;33m[run_sglang WARN]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[run_sglang OK]\033[0m $*"; }
die()  { echo -e "\033[1;31m[run_sglang ERROR]\033[0m $*"; exit 1; }

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
case "$PHASE" in
  setup)        phase_setup ;;
  build-kernel) phase_setup 2>/dev/null || true; source "$VENV_DIR/bin/activate" 2>/dev/null || true; phase_build_kernel ;;
  install)      source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_install ;;
  test)         source "$VENV_DIR/bin/activate" 2>/dev/null || die "run 'setup' first"; phase_test ;;
  all)
    phase_setup
    phase_build_kernel || warn "build-kernel phase best-effort; continuing"
    phase_install
    phase_test
    ;;
  *) die "unknown phase '$PHASE' (use: setup|build-kernel|install|test|all)" ;;
esac
ok "Done ($PHASE)."

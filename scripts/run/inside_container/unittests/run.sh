source ${repo_path}/scripts/run/inside_container/utils.sh

# sglang DLIN smoke suite.
#
# Gating (must pass — sets FAILED / PASSED that check_log_result parses):
#   1. DLIN torch detected + a real DLIN GPU compute (bf16 matmul) succeeds.
#   2. `import sglang` works and current_platform resolves to DlinSRTPlatform
#      (loads the full sgl_kernel module — catches op-registration aborts).
#   3. pure-logic frontend DSL: test_choices + test_separate_reasoning.
#
# Best-effort (logged, NON-gating):
#   4. pytest tests/dl/  (sglang DLIN unit tests; need sgl_kernel)
#   5. scripts/dl/v4_smoke.py  (DSV4 smoke — may need a model/multi-GPU)
smoke_unittests() {
  START=$(date +%s.%N)
  local failed=0

  echo "[smoke] 1) gating: DLIN torch + torch.version.dl + DLIN GPU bf16 matmul"
  python3 - <<'PY' || { echo "FAILED"; failed=1; }
import torch
dl = getattr(torch.version, "dl", None)
assert dl, f"torch.version.dl not set (torch={torch.__version__}); wrong wheel"
assert torch.cuda.is_available(), "torch.cuda not available"
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
print(f"PASSED torch={torch.__version__} dl={dl} gpu={torch.cuda.get_device_name(0)} bf16 matmul OK sum={c.float().sum().item():.1f}")
PY

  echo "[smoke] 2) gating: import sglang + DLIN platform resolves"
  python3 - <<'PY' || { echo "FAILED"; failed=1; }
import sglang
from sglang.srt.platforms import current_platform
from sglang.srt.utils.common import is_dlin
assert is_dlin(), "is_dlin() is False"
assert current_platform.is_dlin(), "current_platform.is_dlin() is False"
assert type(current_platform).__name__ == "DlinSRTPlatform", type(current_platform).__name__
print(f"PASSED import sglang OK: {sglang.__version__}; platform={type(current_platform).__name__}")
PY

  # ── gating: pure-logic frontend DSL (no GPU/model/server) ───────────────
  # test_choices + test_separate_reasoning exercise sglang's choices/regex and
  # reasoning-parser logic — pure Python, verified to pass reliably in CI. The
  # server-dependent lang_frontend tests (openai_backend, bind_cache) stay out
  # of the gate.
  if [ -f "${repo_path}/test/manual/lang_frontend/test_choices.py" ]; then
    echo "[smoke] 3) gating: pure-logic frontend (test_choices, test_separate_reasoning)"
    pytest -q "${repo_path}/test/manual/lang_frontend/test_choices.py" \
              "${repo_path}/test/manual/lang_frontend/test_separate_reasoning.py" \
      || { echo "FAILED"; failed=1; }
  else
    echo "[smoke] 3) skip: lang_frontend pure-logic tests not present"
  fi

  # ── best-effort, non-gating (need sgl_kernel, which the build produced) ──
  if python3 -c "import sgl_kernel" 2>/dev/null; then
    HAS_SGL_KERNEL=1
  else
    HAS_SGL_KERNEL=0
    echo "[smoke] sgl_kernel not installed — skipping sgl_kernel-dependent steps"
  fi

  if [ "${HAS_SGL_KERNEL}" = "1" ] && [ -d "${repo_path}/tests/dl" ]; then
    echo "[smoke] 4) best-effort: pytest tests/dl/"
    CUDA_VISIBLE_DEVICES=0 pytest -v -s "${repo_path}/tests/dl" \
      || echo "[warn] tests/dl had failures (best-effort, non-gating)"
  else
    echo "[smoke] 4) skip: tests/dl (needs sgl_kernel or dir absent)"
  fi

  if [ -f "${repo_path}/scripts/dl/v4_smoke.py" ]; then
    echo "[smoke] 5) best-effort: DSV4 smoke (scripts/dl/v4_smoke.py)"
    CUDA_VISIBLE_DEVICES=0 python3 "${repo_path}/scripts/dl/v4_smoke.py" \
      || echo "[warn] v4_smoke failed (best-effort, non-gating)"
  else
    echo "[smoke] 5) skip: scripts/dl/v4_smoke.py not present"
  fi

  if [ ${failed} -eq 0 ]; then
    echo "smoke_unittests PASSED"
  else
    echo "smoke_unittests FAILED"
  fi

  END=$(date +%s.%N)
  DIFF=$(echo "$END - $START" | bc)
  echo "Execute ${case_name}, time elapsed: $DIFF s"
}

# sglang's CPU UT suite (GPU-free, pure-Python): base-a/b/c-test-cpu via sglang's
# own runner (test/run_suite.py), which executes each registered test file with
# `python3 <file> -f`. These are sglang's base unit tests (logic, parsing,
# reference impls) — they don't need a GPU/model, so they broaden DLIN CI
# coverage from the tiny smoke gate to hundreds of UTs. Run best-effort
# (allow_failure on the job) until the DLIN pass rate is confirmed.
cpu_unittests() {
  cd ${repo_path}
  local failed=0
  for suite in base-a-test-cpu base-b-test-cpu base-c-test-cpu; do
    echo "[ut_cpu] ===== suite: ${suite} ====="
    python3 test/run_suite.py --hw cpu --suite "${suite}" --continue-on-error \
      || { echo "FAILED suite=${suite}"; failed=1; }
  done
  if [ ${failed} -eq 0 ]; then
    echo "cpu_unittests PASSED"
  else
    echo "cpu_unittests FAILED"
  fi
}

pre_running() {
  echo "CASE_TYPE: ${case_type}"
  echo "CASE_NAME: ${case_name};CASE_CMD:${case_name}"

  echo "pre_running"

  copy_repo_sdk_into_docker
  set_pip_source
  install_wheels
  set_tests_path
  source_env
  disable_pytest_shard
  cd $tests_path
}

running() {
  echo "running"
  eval "$case_name"
}

post_running() {
  echo "post_runing"
}

run() {
  pre_running
  running
  post_running
}

source ${repo_path}/scripts/run/inside_container/utils.sh

remove_redundant_directory() {
  echo "remove ${repo_path}/.deps"
  rm -rf ${repo_path}/.deps
  echo "remove ${repo_path}/build"
  rm -rf ${repo_path}/build
}

configure_ccache() {
  ccache --set-config compiler_check=content
  ccache --set-config cache_dir=${CCACHE_DIR}
  ccache --set-config max_size=20G
  ccache --zero-stats
}

pre_running() {
  echo "CASE_TYPE: ${case_type}"
  echo "CASE_NAME: ${case_name};CASE_CMD:${case_name}"

  echo "pre_runing"
  set_pip_source
  copy_repo_sdk_into_docker
  install_wheels
  remove_redundant_directory
  source_env
  configure_ccache
}

# sglang's python package is pure-Python (py3-none-any wheel), so no
# linux_${arch} → ${AUDITWHEEL_PLAT} rename is needed. sgl-kernel (dlcc) is
# DLIN-specific and not manylinux-repairable, so it is shipped as-built.
running() {
  cd ${repo_path}

  echo "[build] sglang wheel (python/pyproject_dl.toml)"
  cp python/pyproject.toml python/pyproject.toml.cuda-bak
  cp python/pyproject_dl.toml python/pyproject.toml
  ( cd python && python -m build --wheel --no-isolation )
  local rc_main=$?
  cp python/pyproject.toml.cuda-bak python/pyproject.toml

  mkdir -p ${repo_path}/dist
  mv python/dist/sglang*.whl ${repo_path}/dist/ 2>/dev/null || true

  # Best-effort sgl-kernel build with dlcc (non-fatal if the dlcc subset does
  # not compile; sglang falls back to torch ops). Mirrors run_sglang.sh.
  if [ "${SKIP_KERNEL:-0}" != "1" ]; then
    echo "[build] sgl-kernel wheel (setup_dl.py, dlcc) — best-effort"
    # setup_dl.py does `import torch`, but the build venv only has build tools.
    # Install the DLIN torch (version from config_env.json) + its deps so the
    # kernel extension can compile against torch headers via dlcc.
    TORCH_SPEC=$(jq -r --arg arch "$(uname -m)" \
      '.wheel_required.runtime[$arch][] | select(startswith("torch-"))' \
      "${repo_path}/scripts/run/config_env.json" 2>/dev/null | head -1)
    if [ -n "${TORCH_SPEC}" ]; then
      TORCH_VER=$(echo "${TORCH_SPEC}" | sed -E 's/^torch-([^-]+(\+[^-]+)*)-cp3.*/\1/')
      echo "[build] installing torch==${TORCH_VER} (+deps) for sgl-kernel build"
      uv pip install "torch==${TORCH_VER}" \
        || echo "[warn] torch install failed; sgl-kernel build will be skipped"
    else
      echo "[warn] torch spec not found in config_env.json; sgl-kernel build skipped"
    fi
    cp sgl-kernel/pyproject.toml sgl-kernel/pyproject.toml.cuda-bak 2>/dev/null || true
    cp sgl-kernel/pyproject_dl.toml sgl-kernel/pyproject.toml 2>/dev/null || true
    ( cd sgl-kernel && python setup_dl.py bdist_wheel ) \
      && mv sgl-kernel/dist/sglang_kernel*.whl ${repo_path}/dist/ 2>/dev/null \
      || echo "[warn] sgl-kernel wheel not produced (best-effort, non-fatal)"
    cp sgl-kernel/pyproject.toml.cuda-bak sgl-kernel/pyproject.toml 2>/dev/null || true
  fi

  if [ ${rc_main} -eq 0 ] && ls ${repo_path}/dist/sglang*.whl >/dev/null 2>&1; then
    echo "BUILD_SUCCESS"
  else
    echo "BUILD_FAILED"
  fi
}

post_running() {
  echo "post_running"
  ccache --show-stats
  if [ -d ${export_sglang_whl_path} ]; then
    set -x
    # Remove stale sglang/sglang_kernel wheels first so the shared SDK path
    # holds ONLY this build's wheels — otherwise the smoke's `ls -t | head -1`
    # can pick a stale wheel from a previous pipeline (saw aarch64 install an
    # old 0.0.0.dev sglang this way).
    rm -f ${export_sglang_whl_path}/sglang-*.whl ${export_sglang_whl_path}/sglang_kernel-*.whl
    cp /sglang_workspace/sglang/dist/* ${export_sglang_whl_path}
    set +x
  fi
}

run() {
  pre_running
  running
  post_running
}

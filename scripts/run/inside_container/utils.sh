source "$(dirname "${BASH_SOURCE[0]}")/../triton_utils.sh"


get_sglang_dev() {
  declare -n sglang_whl_path_ref="$1"
  # Use setuptools_scm to get consistent version across x86 and aarch64
  pushd ${repo_path} > /dev/null
  local version_head=$(python3 -c "from setuptools_scm import get_version; print(get_version())")
  popd > /dev/null
  # sglang python package is pure-Python → py3-none-any wheel (no +cu117 local ver).
  # Tail = python/platform tag onward, e.g. -py3-none-any.whl
  local version_tail=$(echo "$sglang_whl_path_ref" | grep -oE '\-(py3|cp3[0-9]+)[^ ]*\.whl')
  sglang_whl_path_ref="sglang-${version_head}${version_tail}"
  echo "[debug] Generated sglang wheel name: ${sglang_whl_path_ref}"
}

install_container_missing_wheels() {
  echo "[info] install missing wheels"
  # install dl pynvml
  uv pip install ${sdk_path}/python/pynvml-1.0.0-py3-none-any.whl
}

# DL begin
# loongarch64 has no prebuilt dl-virtual wheels for several native sglang
# runtime deps (outlines-core, tiktoken, xgrammar, pillow, ...), so uv builds
# them from source. av/PyAV is platform-gated off loong (no ffmpeg-devel in the
# OpenCloudOS base repos). Install the system build deps the rest need:
# cargo/rustc (outlines-core, tiktoken), pkgconfig + cmake (xgrammar, general),
# gfortran + openblas (scipy), image codec devel libs (pillow). No-op on
# x86_64/aarch64 (prebuilt wheels).
ensure_loong_native_builddeps() {
  if [[ "$(uname -m)" != "loongarch64" ]]; then
    return 0
  fi
  # Curated set; installed individually so one missing pkg doesn't abort others.
  local -a deps=(rust cargo pkgconfig cmake gcc-gfortran openblas-devel \
    zlib-devel libjpeg-turbo-devel freetype-devel libtiff-devel libwebp-devel)
  local mgr
  if command -v dnf >/dev/null 2>&1; then mgr=dnf
  elif command -v yum >/dev/null 2>&1; then mgr=yum
  else echo "[warn] no dnf/yum — cannot install loong native build deps"; return 0; fi
  echo "[info] loongarch64: installing native build deps via $mgr (best-effort per pkg)"
  local p
  for p in "${deps[@]}"; do
    case "$p" in
      pkgconfig) $mgr install -y pkgconfig pkgconf-pkg-config 2>/dev/null || echo "[warn] $mgr install 'pkgconfig' failed" ;;
      *) $mgr install -y "$p" 2>/dev/null || echo "[warn] $mgr install '$p' failed" ;;
    esac
  done
  # rustup fallback if the system package lacked rust+cargo.
  if ! command -v cargo >/dev/null 2>&1; then
    echo "[info] trying rustup fallback (sh.rustup.rs)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs 2>/dev/null \
      | sh -s -- -y --profile minimal --default-toolchain stable \
      || echo "[warn] rustup failed — outlines-core/tiktoken source builds will fail"
    # shellcheck disable=SC1091
    source "${HOME}/.cargo/env" 2>/dev/null || true
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi
  command -v cargo >/dev/null 2>&1 && echo "[info] rust ready: $(rustc --version 2>/dev/null)" \
    || echo "[warn] rust still missing"
  # scipy's meson resolves OpenBLAS via pkg-config. OpenCloudOS's openblas-devel
  # ships libopenblas.so but NO openblas.pc, so scipy fails ('OpenBLAS not
  # found'). If pkg-config can't see openblas, generate a minimal openblas.pc
  # pointing at the installed lib. (Durable fix: bake into image — see
  # docs/dl/ci-loongarch64-native-deps.md.)
  if pkg-config --exists openblas 2>/dev/null; then
    echo "[info] openblas visible to pkg-config ($(pkg-config --modversion openblas 2>/dev/null))"
  else
    local libso libdir pcdir incdir oblaver
    libso="$(find /usr/lib* -name 'libopenblas.so' 2>/dev/null | head -1)"
    pcdir="$(find /usr/lib* -type d -name pkgconfig 2>/dev/null | head -1)"
    incdir="$(dirname "$(find /usr/include -name 'cblas.h' 2>/dev/null | head -1)" 2>/dev/null)"
    libdir="$(dirname "${libso:-/usr/lib64/libopenblas.so}")"
    oblaver="$(rpm -q --qf '%{VERSION}' openblas-devel 2>/dev/null | head -1)"; oblaver="${oblaver:-0.3.26}"
    if [ -n "$libso" ] && [ -n "$pcdir" ]; then
      cat > "${pcdir}/openblas.pc" <<EOF
prefix=/usr
exec_prefix=\${prefix}
libdir=${libdir}
includedir=${incdir:-/usr/include}

Name: OpenBLAS
Description: OpenBLAS shim (OpenCloudOS openblas-devel ships no .pc)
Version: ${oblaver}
Libs: -L\${libdir} -lopenblas
Cflags: -I\${includedir}
EOF
      pkg-config --exists openblas 2>/dev/null \
        && echo "[info] generated ${pcdir}/openblas.pc -> openblas visible ($(pkg-config --libs openblas 2>/dev/null))" \
        || echo "[warn] generated openblas.pc but pkg-config --exists openblas still fails"
    else
      echo "[warn] libopenblas.so ($libso) or pkgconfig dir ($pcdir) not found — scipy build will fail"
    fi
  fi
}
# DL end

install_build_dependencies() {
  echo "[info] install build wheels"
  : "${VIRTUAL_ENV:?VIRTUAL_ENV must be set for build dependency installation}"
  uv pip install --python "${VIRTUAL_ENV}/bin/python" \
    -r "${repo_path}/requirements/build/dl.txt"
}

get_test_requirements_file() {
  case "$ARCH" in
    aarch64)
      echo "${repo_path}/requirements/test/dl.txt"
      ;;
    loongarch64)
      echo "${repo_path}/requirements/test/dl.txt"
      ;;
    *)
      echo "${repo_path}/requirements/test/dl.txt"
      ;;
  esac
}

install_test_dependencies() {
  echo "[info] install test wheels"
  local test_requirements
  test_requirements="$(get_test_requirements_file)"
  uv pip install -r "${test_requirements}"
}

install_sglang_wheel() {
  local test_requirements
  test_requirements="$(get_test_requirements_file)"
  # Install the built wheel WITH the srt_dl extra: sglang's heavy runtime deps
  # (orjson/aiohttp/pydantic/...) live in extras (runtime_common), not the
  # wheel's core requires, so a bare wheel install omits them.
  local -a constraint_args=(-c "${test_requirements}")
  if [[ "${ARCH}" == "loongarch64" ]]; then
    # loong-only pins: some packages' latest version has no loong wheel and
    # fails to source-build; pin to a version that ships a loong wheel.
    constraint_args+=(-c "${repo_path}/requirements/test/dl_loongarch64.txt")
  fi
  uv pip install "${constraint_args[@]}" "${1}[srt_dl]"
}

install_config_env_specified_dependencies() {
  echo "[info] install config_env.json specified wheels"
  if [[ "$case_type" == "build" ]]; then
    select_run_type="build"
  else
    select_run_type="runtime"
  fi
  # install wheels
  jq -rc ".wheel_required.${select_run_type}.${ARCH}[]" "${repo_path}/scripts/run/config_env.json" | while read -r params; do
    if [[ "${case_name}" == "smoke_unittests" || "${case_name}" == "cpu_unittests" ]];then
      if [[ ${params} == "sglang-"* ]];then
        # Install the sglang wheel the build produced. The SDK path is shared
        # across pipelines and accumulates wheels, so pick the NEWEST (the build
        # just copied the current one) — `ls | head -1` would pick a stale
        # alphabetically-first wheel instead.
        wheel_path="$(ls -t ${sdk_path}/sglang-*.whl 2>/dev/null | head -1)"
        if [ -n "${wheel_path}" ]; then
          install_sglang_wheel "${wheel_path}"
        else
          echo "[warn] no sglang*.whl found in ${sdk_path} — smoke import will fail"
        fi
      elif [[ ${params} == "triton-"* ]];then
        ensure_triton_from_sdk_tag install "${sdk_path}"
      else
        version=$(echo "${params}" | cut -d '-' -f2 | cut -d '-' -f1)
        wheel_name="${params%%-*}"
        uv pip install --force-reinstall --no-deps "${wheel_name}==${version}"
      fi
    else
      # Non-smoke case (e.g. cpu_unittests): install the framework wheels by
      # fuzzy match, and sglang by NEWEST wheel (ls -t) to avoid a stale one.
      if [[ ${params} == "sglang-"* ]];then
        wheel_path="$(ls -t ${sdk_path}/sglang-*.whl 2>/dev/null | head -1)"
        if [[ -n "${wheel_path}" ]]; then
          echo "[info] install sglang wheel: $(basename "${wheel_path}")"
          install_sglang_wheel "${wheel_path}"
        else
          echo "[warn] no sglang*.whl in ${sdk_path}"
        fi
      else
        wheel_name="${params%%-*}-*"
        echo "[info] install fuzzy matching wheel: ${wheel_name}"
        uv pip install --force-reinstall --no-deps ${sdk_path}/${wheel_name}
      fi
    fi
  done
}

install_wheels() {
  ensure_loong_native_builddeps
  if [[ "$case_type" == "build" ]]; then
    install_build_dependencies
  else
    install_test_dependencies
    install_container_missing_wheels
  fi
  install_config_env_specified_dependencies
  # In the runtime/smoke container, install the sgl_kernel wheel the build job
  # produced. The moe_align_block_size schema drift (sgl-kernel/csrc) that used
  # to abort `import sgl_kernel` is fixed, so this now enables the
  # sgl_kernel-dependent smoke steps (tests/dl, lang_frontend, v4_smoke).
  if [[ "$case_type" != "build" ]]; then
    local kernel_whl
    kernel_whl="$(ls -t ${sdk_path}/sglang_kernel*.whl 2>/dev/null | head -1)"
    if [[ -n "${kernel_whl}" ]]; then
      echo "[info] install sgl_kernel wheel: $(basename "${kernel_whl}")"
      uv pip install --no-deps "${kernel_whl}" \
        || echo "[warn] sgl_kernel wheel install failed (non-fatal)"
    else
      echo "[info] no sgl_kernel wheel in ${sdk_path} (build did not produce one)"
    fi
  fi
}

copy_repo_sdk_into_docker() {
  mkdir -p /sglang_workspace
  echo "copy ${repo_path} into /sglang_workspace"
  cp -a ${repo_path} /sglang_workspace
  sudo chown -R $(whoami):$(whoami) /sglang_workspace/sglang
  declare -g repo_path=/sglang_workspace/sglang

  # check repo status
  echo "check repo status"
  cd ${repo_path}
  git status
  git describe --tags
  git remote -v

  echo "copy ${sdk_path} into /sglang_workspace"
  cp --dereference -r ${sdk_path} /sglang_workspace
  declare -g sdk_path=/sglang_workspace/sdk
}

set_tests_path() {
  mkdir -p /sglang_workspace/sglang_tests
  cp -r $repo_path/tests /sglang_workspace/sglang_tests
  declare -g tests_path=/sglang_workspace/sglang_tests/tests
}

source_env() {
  source ${sdk_path}/env.sh
  export XFORMERS_FORCE_DISABLE_TRITON=1
  export DISABLE_ADDMM_CUDA_LT=1 # skip error of cublasLtMatmul

  # remove empty env SGLANG_TORCH_PROFILER_DIR, or torch.profile would be enabled.
  if [ -z "$SGLANG_TORCH_PROFILER_DIR" ]; then
    unset SGLANG_TORCH_PROFILER_DIR
  fi
}

json2args() {
  # transforms the JSON string to command line args, and '_' is replaced to '-'
  # example:
  # input: { "model": "meta-llama/Llama-2-7b-chat-hf", "tensor_parallel_size": 1 }
  # output: --model meta-llama/Llama-2-7b-chat-hf --tensor-parallel-size 1
  local json_string=$1
  local args=$(
    echo "$json_string" | jq -r '
      to_entries |
      map("--" + (.key | gsub("_"; "-")) + " " + (.value | tostring)) |
      join(" ")
    '
  )
  echo "$args"
}

check_gpu_memory() {
  # wait until all GPU memory usage smaller than 1GB or stop this script
  local timecount=0
  while [ $(dlsmi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -n 1) -ge 1000 ]; do
    sleep 1
    ((timecount++))
    if [ $timecount -eq 5 ]; then
        echo "Please make sure all selected gpus are unoccupied. Use env CUDA_VISIBLE_DEVICES to select gpus."
        exit 1
    fi
  done
}

check_gpus() {
  # check the number of GPUs and GPU type.
  declare -g gpu_count=$(dlsmi --list-gpus | wc -l)
  if [[ $gpu_count -gt 0 ]]; then
    echo "${gpu_count} GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  declare -g gpu_type=$(echo $(dlsmi --query-gpu=name --format=csv,noheader | awk 'NR==1 {print}'))
  echo "GPU type is $gpu_type"
  check_gpu_memory
}

replace_model_and_dataset_path() {
  if [[ -n "${SGLANG_USE_LOCAL_MODEL_PATH}" && -e "${SGLANG_USE_LOCAL_MODEL_PATH}" ]]; then
    echo "SGLANG_USE_LOCAL_MODEL_PATH: ${SGLANG_USE_LOCAL_MODEL_PATH} exists, replace path in json"
    declare -n server_or_clinet_params=$1
    for var in "model" "dataset_path"; do
      if [ $(echo "$server_or_clinet_params" | jq --arg field "$var" 'has($field)') == "true" ]; then
        old_path=$(echo "$server_or_clinet_params" | jq -r --arg var "$var" '.[$var]')
        old_path_basename=$(basename "$old_path")
        new_path="${SGLANG_USE_LOCAL_MODEL_PATH%/}/$old_path_basename"
        server_or_clinet_params=$(echo "$server_or_clinet_params" | jq --arg var "$var" --arg new_path "$new_path" '.[$var] = $new_path')
      fi
    done
  fi
}

merge_ks58_server_parameters() {
  declare -n ref_server_params=$1
  declare -n ref_params=$2
  if ! echo "$ref_params" | jq -e 'has("ks58_server_parameters")' > /dev/null; then
    return
  fi
  gpu_info=$(dlsmi -L | head -n 1)
  if ! echo "$gpu_info" | grep -q "KS58"; then
    return
  fi
  ks58_server_parameters=$(echo "$ref_params" | jq -r '.ks58_server_parameters')
  ref_server_params=$(echo "$ref_server_params" | jq --argjson ks58_server_parameters "$ks58_server_parameters" '. *= $ks58_server_parameters')
}

merge_ks38_server_parameters() {
  declare -n ref_server_params=$1
  declare -n ref_params=$2
  if ! echo "$ref_params" | jq -e 'has("ks38_server_parameters")' > /dev/null; then
    return
  fi
  gpu_info=$(dlsmi -L | head -n 1)
  if ! echo "$gpu_info" | grep -q "KS38"; then
    return
  fi
  ks38_server_parameters=$(echo "$ref_params" | jq -r '.ks38_server_parameters')
  ref_server_params=$(echo "$ref_server_params" | jq --argjson ks38_server_parameters "$ks38_server_parameters" '. *= $ks38_server_parameters')
}

merge_device_server_parameters() {
  merge_ks38_server_parameters "$@"
  merge_ks58_server_parameters "$@"
}

add_default_max_model_len() {
  declare -n ref_server_params=$1
  if ! echo "$ref_server_params" | jq -e 'has("max_model_len")' > /dev/null; then
    echo "max_model_len not found in server_parameters, setting default value 4096"
    ref_server_params=$(echo "$ref_server_params" | jq '.max_model_len = 4096')
  else
    echo "max_model_len already exists in server_parameters: $(echo "$ref_server_params" | jq -r '.max_model_len')"
  fi
}

disable_pytest_shard() {
  # DL: disable pytest_shard output, or CI would break
  python3 -m pip  uninstall pytest_shard -y
}

copy_source_to_dist_packages() {
  echo "copy ${repo_path}/python/sglang into /opt/venv/lib/python3.12/site-packages"
  cp -r $repo_path/python/sglang /opt/venv/lib/python3.12/site-packages
}

set_pip_source() {
  local arch
  arch=$(uname -m)
  local repo_suffix=""
  if [[ "${arch}" == "loongarch64" ]]; then
    repo_suffix="-loongarch64"
  fi

  if [[ "$IS_DL_INTERNAL_NET" == "TRUE" ]]; then
    set -x
    pip3 config set global.index-url \
    http://artifactory.denglin.com:8082/artifactory/api/pypi/dl-virtual${repo_suffix}/simple
    pip3 config set install.trusted-host artifactory.denglin.com
    export UV_DEFAULT_INDEX="http://artifactory.denglin.com:8082/artifactory/api/pypi/dl-virtual${repo_suffix}/simple"
    set +x
  else
    set -x
    pip3 config set global.index-url \
    http://ext-artifactory.denglin.com:8082/artifactory/api/pypi/dl-virtual${repo_suffix}/simple
    pip3 config set install.trusted-host ext-artifactory.denglin.com
    export UV_DEFAULT_INDEX="http://ext-artifactory.denglin.com:8082/artifactory/api/pypi/dl-virtual${repo_suffix}/simple"
    set +x
  fi
}

set_profile_env() {
  if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
    echo "SGLANG_TORCH_PROFILER_DIR:${SGLANG_TORCH_PROFILER_DIR} is not empty"
    export DLPTI_AUTO_LOAD=1
  fi
}

add_profile_client_param() {
  declare -n ref_client_args="$1"
  if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
    ref_client_args="${ref_client_args} --profile"
    echo "ref_client_args: ${ref_client_args}"
  fi
}

wait_profile_result() {
  if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
    timeout=60
    echo "Wait ${timeout}s: adjust it when no profile result produced"
    echo "Maybe set the env variable SGLANG_RPC_TIMEOUT to a big number like export SGLANG_RPC_TIMEOUT=1800000"
    sleep ${timeout}
  fi
}

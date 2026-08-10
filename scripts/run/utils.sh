source "$(dirname "${BASH_SOURCE[0]}")/triton_utils.sh"



check_REPO_PATH() {
  if [[ ! -d "${REPO_PATH}" ]]; then
    echo "REPO_PATH ${REPO_PATH} is not exist!"
    exit 1
  fi
}

check_SDK_PATH() {
  if [[ ! -d "${SDK_PATH}" ]]; then
    echo "SDK_PATH ${SDK_PATH} is not exist!"
    exit 1
  fi
}

check_export_sglang_whl_path() {
  if [[ "$case_type" == "build" ]]; then
    if [[ ! -d "${export_sglang_whl_path}" ]]; then
      echo "export_sglang_whl_path ${export_sglang_whl_path} is not exist!"
      exit 1
    fi
  fi
}

check_case_type() {
  :
}

read_checkpoint_path() {
  checkpoint_path=$(jq -r ".checkpoint_path" ${REPO_PATH}/scripts/run/config_env.json)
  if [[ "$SGLANG_USE_LOCAL_MODEL_PATH" == "" ]]; then
    SGLANG_USE_LOCAL_MODEL_PATH=${checkpoint_path}
  fi

  if [[ ! -d "${SGLANG_USE_LOCAL_MODEL_PATH}" ]]; then
    echo "SGLANG_USE_LOCAL_MODEL_PATH ${SGLANG_USE_LOCAL_MODEL_PATH} is not exist!"
  fi
}

read_driver_version() {
  DRVIER_VERSION=$(dlsmi |grep 'Driver version'  | awk '{print $(NF-1)}')
  # check if dlsmi is ready
  if [ $? -ne 0 ]; then
      echo "dlsmi failed, please check it!"
      exit 1
  fi
}

read_gpu_num() {
  GPU_NUM=$(dlsmi -L | wc -l)
}

read_cpu_arch() {
  declare -g ARCH=$(uname -m)
}

is_dl_internal_net() {
  curl --max-time 1 http://ext-artifactory.denglin.com:8082/ > /dev/null 2>&1
  curl_result=$?
  declare -g IS_DL_INTERNAL_NET=''
  if [ $curl_result -eq 0 ]; then
    IS_DL_INTERNAL_NET="FALSE"
  else
    IS_DL_INTERNAL_NET="TRUE"
  fi
}

get_docker_image() {
  select_run_type=''
  if [[ "$case_type" == "build" ]]; then
    select_run_type="build"
    SGLANG_DL_DOCKER_OS=""
    jq_filter=".docker_images.${select_run_type}.${ARCH}"
  else
    select_run_type="runtime"

    # select default docker os
    if [[ "$SGLANG_DL_DOCKER_OS" == "" ]]; then
      if [[ "$ARCH" == "x86_64" ]]; then
        SGLANG_DL_DOCKER_OS="ubuntu22_04"
      elif [[ "$ARCH" == "aarch64" ]]; then
        SGLANG_DL_DOCKER_OS="ubuntu22_04"
      elif [[ "$ARCH" == "loongarch64" ]]; then
        SGLANG_DL_DOCKER_OS="openeuler"
      fi
    fi
    jq_filter=".docker_images.${select_run_type}.${ARCH}.${SGLANG_DL_DOCKER_OS}"
  fi

  image_name=$(jq -r ${jq_filter} \
    ${REPO_PATH}/scripts/run/config_env.json)

  if [[ "$IS_DL_INTERNAL_NET" == "FALSE" ]]; then
    image_name="ext-${image_name}"
  fi
  declare -g DOCKE_IMAGE=$image_name
}

read_docker_image_info() {
  declare -g TESTLOG_OS=$(docker run --rm ${DOCKE_IMAGE}  cat \
    /etc/os-release | grep 'PRETTY_NAME' | cut -d'=' -f2 | tr -d '"')
  declare -g TESTLOG_PYTHON=$(docker run --rm ${DOCKE_IMAGE} \
    python3 --version)
  declare -g TESTLOG_DOCKER_IMAGE=${DOCKE_IMAGE}
}

check_log_result() {
  log_file=$1
  echo "log_file: ${log_file}"
  # check result from log, then exit to trigger ci precheck result
  if [ "$case_type" == "build" ]; then
    if grep -q "BUILD_FAILED" ${log_file} || \
       ! grep -q "BUILD_SUCCESS" ${log_file}; then
        echo "sglang build FAILED!"
        exit 1
    fi
  elif [ "$case_name" == "smoke_unittests" ]; then
    # Gate on the EXPLICIT smoke markers emitted by unittests/run.sh, not on a
    # generic FAILED/PASSED substring: best-effort steps (tests/dl, v4_smoke)
    # legitimately print lines like "ENGINE LOAD FAILED" / pytest failures that
    # would otherwise trip a naive grep.
    if grep -q "smoke_unittests FAILED" ${log_file} || \
       ! grep -q "smoke_unittests PASSED" ${log_file}; then
        echo "smoke test FAILED!"
        exit 1
    fi
  fi
}

read_git_tag() {
  sglang_git_tag=$(git describe --tags)
}

run_utils() {
  check_REPO_PATH
  check_SDK_PATH
  check_export_sglang_whl_path
  read_checkpoint_path
  read_cpu_arch
  is_dl_internal_net
  get_docker_image
  read_docker_image_info
  read_driver_version
  read_gpu_num
  read_git_tag
}

run_utils

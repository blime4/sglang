#!/bin/bash
date
case_name=$1
case_type=$2

source utils.sh

echo " "
echo "******************BASIC INFO*******************"
echo "REPO_PATH: ${REPO_PATH}"
echo "SDK_PATH: ${SDK_PATH}"
echo "CCACHE_DIR: ${CCACHE_DIR}"
echo "UV_CACHE_DIR: ${UV_CACHE_DIR}"
echo "CPU ARCH: ${ARCH}"
echo "DRVIER_VERSION: ${DRVIER_VERSION}"
echo "GPU_NUM: ${GPU_NUM}"
echo "IS_DL_INTERNAL_NET: ${IS_DL_INTERNAL_NET}"
echo "DOCKE_IMAGE: ${DOCKE_IMAGE}"
echo "SGLANG_USE_LOCAL_MODEL_PATH: ${SGLANG_USE_LOCAL_MODEL_PATH}"
echo "export_sglang_whl_path: ${export_sglang_whl_path}"
echo "CI_ENV_PATH: ${CI_ENV_PATH}"
echo "git tag: ${sglang_git_tag}"
echo "TESTLOG_OS: ${TESTLOG_OS}"
echo "TESTLOG_PYTHON: ${TESTLOG_PYTHON}"
echo "TESTLOG_DOCKER_IMAGE: ${TESTLOG_DOCKER_IMAGE}"
echo "***********************************************"
echo " "

# select run.sh script
run() {
    source ${repo_path}/scripts/run/inside_container/${case_type}/run.sh
    run
}

uv_cache_args=()
if [[ -n "${UV_CACHE_DIR}" ]]; then
    uv_cache_args+=(--env "UV_CACHE_DIR=${UV_CACHE_DIR}")
    uv_cache_args+=(--volume "${UV_CACHE_DIR}:${UV_CACHE_DIR}")
fi

# read CI ENV from ${CI_ENV_PATH}
ci_env_args=()
if [[ -n "${CI_ENV_PATH}" ]]; then
    ci_env_args+=(--env-file "${CI_ENV_PATH}")
fi

docker run --shm-size=32gb \
           --rm \
           --runtime=dlrt \
           "${ci_env_args[@]}" \
           --env DENGLIN_DEVICES=${CUDA_VISIBLE_DEVICES:-all} \
           --env sdk_path=${SDK_PATH} \
           --env repo_path=${REPO_PATH} \
           --env CCACHE_DIR=${CCACHE_DIR} \
           --env UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-500} \
           --env SGLANG_USE_LOCAL_MODEL_PATH=${SGLANG_USE_LOCAL_MODEL_PATH} \
           --env export_sglang_whl_path=${export_sglang_whl_path} \
           --env SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR} \
           --env case_type=${case_type} \
           --env case_name=${case_name} \
           --env ARCH=${ARCH} \
           --env IS_DL_INTERNAL_NET=${IS_DL_INTERNAL_NET} \
           --volume ${SDK_PATH}:${SDK_PATH} \
           --volume ${REPO_PATH}:${REPO_PATH} \
           --volume ${CCACHE_DIR}:${CCACHE_DIR} \
           "${uv_cache_args[@]}" \
           --volume ${SGLANG_USE_LOCAL_MODEL_PATH}:${SGLANG_USE_LOCAL_MODEL_PATH} \
           ${DOCKE_IMAGE} \
           bash -c "$(declare -f run); run" \
           | tee docker_sglang_${case_type}_${case_name}.log

check_log_result docker_sglang_${case_type}_${case_name}.log

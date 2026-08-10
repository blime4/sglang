#!/bin/bash
# =============================================================================
# bundle-run.sh — CI entry point for tag-triggered .run bundle packaging
# =============================================================================
# Downloads runtime wheels declared in scripts/run/config_env.json directly
# from JFrog Artifactory (dl-pypi), then combines them with the pipeline-built
# sglang wheel (available as a GitLab artifact from the compile job) and packages
# everything into a self-extracting .run installer via devops-scripts.
#
# Required environment variables (set by GitLab CI):
#   CI_PROJECT_DIR   - repository root
#   CI_COMMIT_TAG    - tag that triggered this pipeline
#   SDK_TAG          - SDK version string (e.g. V2_SOFTWARE_master_202607101443)
#
# Optional:
#   PUBLISH_DIR      - JFrog target directory (default: sglang-bundles-precheck)
#
# Assumptions:
#   - compile job artifact: processed_wheels/${CI_PIPELINE_ID}/sglang*.whl
#   - runtime wheels listed in scripts/run/config_env.json under
#     wheel_required.runtime.<arch> (sglang entry is skipped; sourced from artifact)
#   - non-sglang wheels other than triton are available at dl-pypi/<pkg_name>/<filename>
#   - triton is fetched from sw-triton/${SDK_TAG} to match compile/test flow
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGING_DIR="${SCRIPT_DIR}/devops-scripts/packaging"
source "${PACKAGING_DIR}/common/log.sh"
source "${CI_PROJECT_DIR}/scripts/run/triton_utils.sh"

# ── Parse arguments ───────────────────────────────────────────────────────────
ARCH=""
SGLANG_WHL_FROM_JFROG=""   # When set, download this sglang wheel filename from JFrog instead of using artifact
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --sglang-whl-from-jfrog) SGLANG_WHL_FROM_JFROG="$2"; shift 2 ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "${ARCH}" ]] && { log_error "--arch is required"; exit 1; }

# ── Derived paths ─────────────────────────────────────────────────────────────
CONFIG_JSON="${CI_PROJECT_DIR}/scripts/run/config_env.json"
SGLANG_WHL_DIR="${CI_PROJECT_DIR}/processed_wheels/${CI_PIPELINE_ID}"
WHL_DIR="${CI_PROJECT_DIR}/output/whls/${ARCH}"
OUT_DIR="${CI_PROJECT_DIR}/output/${ARCH}"

mkdir -p "${WHL_DIR}" "${OUT_DIR}"

log_info "=== Bundle packaging: arch=${ARCH} tag=${CI_COMMIT_TAG} sdk=${SDK_TAG} ==="

# ── Step 1/4: Download runtime wheels from Artifactory ───────────────────────
log_info "Step 1/4: Downloading runtime wheels from config_env.json"

if [[ ! -f "${CONFIG_JSON}" ]]; then
    log_error "config_env.json not found: ${CONFIG_JSON}"
    exit 1
fi

# Extract runtime wheel list for this arch, skip sglang (sourced from artifact)
WHEEL_LIST="$(python3 -c "
import json, sys
with open('${CONFIG_JSON}') as f:
    cfg = json.load(f)
for whl in cfg['wheel_required']['runtime']['${ARCH}']:
    if not whl.startswith('sglang-'):
        print(whl)
")"

if [[ -z "${WHEEL_LIST}" ]]; then
    log_error "No runtime wheels found in config_env.json for arch=${ARCH}"
    exit 1
fi

log_info "Runtime wheels to download:"
echo "${WHEEL_LIST}"

DL_FAIL=0
while IFS= read -r WHL_FILENAME; do
    [[ -z "${WHL_FILENAME}" ]] && continue

    if [[ "${WHL_FILENAME}" == triton-* ]]; then
        log_info "  Downloading from sw-triton: ${WHL_FILENAME}"
        if ! ensure_triton_from_sdk_tag download "${WHL_DIR}" "${ARCH}" "${SDK_TAG}"; then
            log_error "  FAILED: ${WHL_FILENAME}"
            DL_FAIL=$((DL_FAIL + 1))
        fi
        continue
    fi

    PKG_NAME="${WHL_FILENAME%%-*}"
    DEST="${WHL_DIR}/${WHL_FILENAME}"
    if [[ -f "${DEST}" ]]; then
        log_info "  Already present: ${WHL_FILENAME}"
        continue
    fi
    log_info "  Downloading: ${WHL_FILENAME}"
    if ! jf rt dl "dl-pypi/${PKG_NAME}/${WHL_FILENAME}" "${WHL_DIR}/" --flat=true; then
        log_error "  FAILED: ${WHL_FILENAME}"
        DL_FAIL=$((DL_FAIL + 1))
    fi
done <<< "${WHEEL_LIST}"

if [[ ${DL_FAIL} -gt 0 ]]; then
    log_error "${DL_FAIL} wheel(s) failed to download"
    exit 1
fi

# ── Step 2/4: Add pynvml wheel from SDK ──────────────────────────────────────
log_info "Step 2/4: Adding pynvml wheel from SDK"

PYNVML_WHL="$(ls -1 "${SDK_PATH}/python/pynvml"*.whl 2>/dev/null | head -1 || true)"
if [[ -z "${PYNVML_WHL}" || ! -f "${PYNVML_WHL}" ]]; then
    log_error "No pynvml wheel found in ${SDK_PATH}/python/"
    exit 1
fi
log_info "Using pynvml wheel from SDK: $(basename "${PYNVML_WHL}")"
cp "${PYNVML_WHL}" "${WHL_DIR}/"

# ── Step 3/4: Add sglang wheel ────────────────────────────────────────────────
log_info "Step 3/4: Adding sglang wheel"

if [[ -n "${SGLANG_WHL_FROM_JFROG}" ]]; then
    log_info "Downloading sglang wheel from JFrog: ${SGLANG_WHL_FROM_JFROG}"
    if ! jf rt dl "dl-pypi/sglang/${SGLANG_WHL_FROM_JFROG}" "${WHL_DIR}/" --flat=true; then
        log_error "Failed to download sglang wheel from JFrog: ${SGLANG_WHL_FROM_JFROG}"
        exit 1
    fi
    SGLANG_WHL="${WHL_DIR}/${SGLANG_WHL_FROM_JFROG}"
else
    SGLANG_WHL="$(ls -1 "${SGLANG_WHL_DIR}"/sglang*.whl 2>/dev/null | head -1 || true)"
    if [[ -z "${SGLANG_WHL}" || ! -f "${SGLANG_WHL}" ]]; then
        log_error "No sglang wheel found in ${SGLANG_WHL_DIR}"
        log_error "Expected artifact from compile job: processed_wheels/${CI_PIPELINE_ID}/sglang*.whl"
        log_error "Tip: pass --sglang-whl-from-jfrog <filename> to fetch from JFrog instead"
        exit 1
    fi
    log_info "Using pipeline-built sglang wheel: $(basename "${SGLANG_WHL}")"
    cp "${SGLANG_WHL}" "${WHL_DIR}/"
fi

log_info "All wheels collected:"
ls -lh "${WHL_DIR}"/*.whl

# ── Step 4/4: Create and publish .run bundle ──────────────────────────────────
log_info "Step 4/4: Creating .run bundle"

BUNDLE_NAME="$(basename "${SGLANG_WHL}" .whl)-bundle"

bash "${PACKAGING_DIR}/create-run-bundle.sh" \
    --arch "${ARCH}" \
    --sdk-tag "${SDK_TAG}" \
    --whl-dir "${WHL_DIR}" \
    --output-dir "${OUT_DIR}" \
    --name "${BUNDLE_NAME}"

PUBLISH_DIR="${PUBLISH_DIR:-sglang-bundles-precheck}"
log_info "Publishing to JFrog (${PUBLISH_DIR})"
bash "${PACKAGING_DIR}/ci/publish.sh" --publish-dir "${PUBLISH_DIR}" "${OUT_DIR}"

log_info "=== Bundle packaging complete ==="

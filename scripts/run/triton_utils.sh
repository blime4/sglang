ensure_triton_from_sdk_tag() {
  local action="$1"
  local sdk_root="$2"
  local arch="${3:-$ARCH}"
  local sdk_tag="${4:-${SDK_TAG:-}}"

  local triton_sw_dir=""
  local triton_wheel
  local config_env_json="$(dirname "${BASH_SOURCE[0]}")/config_env.json"

  case "$arch" in
    x86_64)
      triton_sw_dir="cp312-cp312-manylinux_2_28_x86_64"
      ;;
    aarch64)
      triton_sw_dir="cp312-cp312-manylinux_2_28_aarch64"
      ;;
    loongarch64)
      triton_sw_dir="cp312-cp312-manylinux_2_38_loongarch64"
      ;;
    *)
      echo "Unsupported arch for triton wheel: ${arch}"
      exit 1
      ;;
  esac

  triton_wheel="$(jq -r --arg arch "$arch" \
    '[.wheel_required.runtime[$arch][]? | select(startswith("triton-"))] | first // empty' \
    "${config_env_json}")"
  if [[ -z "${triton_wheel}" ]]; then
    echo "triton wheel not configured for arch: ${arch}"
    exit 1
  fi

  local triton_dest="${sdk_root}/${triton_wheel}"

  case "$action" in
    download_then_install)
      ensure_triton_from_sdk_tag download "$sdk_root" "$arch" "$sdk_tag"
      ensure_triton_from_sdk_tag install "$sdk_root" "$arch" "$sdk_tag"
      ;;
    download)
      if [[ -f "${triton_dest}" ]]; then
        echo "[info] triton wheel already present: ${triton_dest}"
        return
      fi

      local triton_src="sw-triton/${sdk_tag}/${triton_sw_dir}/${triton_wheel}"
      echo "[info] download triton wheel from ${triton_src}"
      jf rt dl "${triton_src}" "${sdk_root}/" --flat=true
      ;;
    install)
      if [[ ! -f "${triton_dest}" ]]; then
        echo "triton wheel not found: ${triton_dest}"
        exit 1
      fi

      echo "[info] install triton from ${triton_dest}"
      python3 -m pip install --force-reinstall --no-deps "${triton_dest}"
      ;;
    *)
      echo "Unsupported triton action: ${action}"
      exit 1
      ;;
  esac
}

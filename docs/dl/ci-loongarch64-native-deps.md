# loongarch64 CI: native-dependency wheel gap

**Status:** `compile_test_sglang_loongarch64` build is GREEN; the **smoke** step fails
because `pip install sglang[srt_dl]` pulls the scientific Python stack, and
**loongarch64 has no prebuilt wheels** for it in `dl-virtual-loongarch64`. uv falls
back to source builds, each needing native build toolchain the manylinux loong image
lacks. Job is `allow_failure: true` (non-blocking) until resolved.

## Root cause

`runtime_common` (pulled by `srt_dl`) includes several **native** packages. On x86_64 /
aarch64 these come as prebuilt wheels; on loongarch64 there are no wheels in the mirror,
so uv builds from source:

| Package (resolved ver) | Native build needs | System pkg |
|---|---|---|
| outlines-core==0.1.26 (via outlines==0.1.11) | Rust (cargo/rustc) | `rust cargo` |
| tiktoken (latest) | Rust | `rust cargo` |
| xgrammar==0.2.1 | C++/cmake | `cmake` |
| torchao==0.9.0 | gcc + torch headers | (present) |
| pillow==12.3.0 | image codec devel | `zlib-devel libjpeg-turbo-devel freetype-devel libtiff-devel libwebp-devel` |
| scipy==1.18.0 | Fortran + OpenBLAS | `gcc-gfortran openblas-devel` — **+ scipy meson must find openblas via pkg-config (currently NOT found — `openblas.pc` location TBD on OpenCloudOS)** |
| av==18.0.0 (PyAV) | FFmpeg devel | `ffmpeg-devel` — **NOT in OpenCloudOS base repos (EPEL/RPM Fusion)**; currently **platform-gated off loong** in `python/pyproject_dl.toml` |
| numpy==2.3.x | — | has a wheel / builds fine |

`ensure_loong_native_builddeps()` in `scripts/run/inside_container/utils.sh` installs the
above on loongarch64 as a **transitional fallback**. It fixes outlines-core/tiktoken/
xgrammar/pillow, but **scipy is the wall** (openblas pkg-config detection) and av has no
ffmpeg-devel — so the durable fix is one of the two infra options below.

## Fix — Option A (recommended): bake the toolchain into the base image

Add to the `c-41:manylinux_2_38-loongarch64` image (`ext-artifactory.../ci-docker-images/c-41`):

```sh
dnf install -y \
  rust cargo gcc-gfortran \
  openblas-devel openblas-static \
  pkgconfig cmake \
  zlib-devel libjpeg-turbo-devel freetype-devel libtiff-devel libwebp-devel
# Then make openblas visible to scipy's meson (locate the .pc and export):
#   export PKG_CONFIG_PATH="$(dirname "$(find /usr -name 'openblas*.pc' | head -1)"):${PKG_CONFIG_PATH}"
```

If `av` (multimodal video) is wanted on loong, also enable EPEL/RPM Fusion and
`dnf install -y ffmpeg-devel`, then drop the `platform_machine != 'loongarch64'`
gate on `av` in `python/pyproject_dl.toml`.

Once the image has these, `ensure_loong_native_builddeps()` becomes a no-op (all deps
already present) and can be simplified/removed.

## Fix — Option B: provision loongarch64 wheels in the artifactory

Build (on a loongarch64 machine) or mirror (from Loongson's PyPI) these wheels into
`dl-pypi` / `dl-virtual-loongarch64`, matching the versions CI resolves:

- `scipy==1.18.0`
- `pillow==12.3.0`
- `outlines_core==0.1.26`
- `tiktoken` (latest compatible)
- `xgrammar==0.2.1`
- `torchao==0.9.0`
- `av==18.0.0` (only if dropping the loong gate)
- `numpy` (current 2.3.x — likely already mirrored; verify)

With wheels present, uv fetches binaries and no source build / toolchain is needed —
the cleanest, fastest CI. `ensure_loong_native_builddeps()` can then be removed entirely.

## Verification

After either fix, on the `dl-loongarch64-outlines-rust` branch (or dl-merge once merged):
the `compile_test_sglang_loongarch64` job should reach `smoke_unittests` with
`Installed sglang-0...` (no `Failed to build ...`) and `import sglang` succeeding.
Then flip `allow_failure: true` → `false` in `.gitlab-ci.yml` so loong gates the MR.

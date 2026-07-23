# SGLang × DLIN (DLIN) — High-Performance Operator Integration Plan

Status: **Phase 0 (environment) working**; this document plans the operator
integration (Phases 1–5). It is the blueprint referenced by `run_sglang.sh`.

Companion artifacts already delivered:
- `run_sglang.sh` — uv env + DLIN torch + editable sglang + DLIN GPU smoke UT.
- `python/pyproject_dl.toml` — DLIN dependency variant (mirrors `pyproject_{cpu,npu,xpu,other}.toml`).
- `python/sglang/srt/utils/common.py` — torchvision import made lazy (native ext
  segfaults on DLIN; see §6).
- **Phase 1 implemented**: `is_dlin()` + `torch.version.dl` shim (`common.py`),
  `is_dlin()` on `DeviceMixin`, `platforms/dlin.py::DlinSRTPlatform`, wired into
  `platforms/__init__.py` (DLIN checked before the CUDA fallback).

**Patch marker convention** (mirrors vLLM): every DLIN change is wrapped in
`# DL begin` … `# DL end` for multi-line blocks, or tagged `# DL:` for inline
edits, so the DL diff is greppable (`grep -rn "# DL" python/`). vLLM uses the
same convention across ~240 markers plus `if(USE_DLIN)` in CMake.

---

## 0. What already works (Phase 0, verified 2026-06-22)

Host: 12× DLIN **KS38 QUAD** (32 GB each), Driver 2.3.26, DL-SMI 12.0.
SDK: `MRrc-4.2.0-202606161052` (`SDK_DIR`). DLIN torch: `2.9.1+dl24.sdk202606031721`.

| Check | Result |
|---|---|
| `uv venv` (py3.12) + DLIN torch install | ✅ |
| `torch.version.dl == "11.7"` (DLIN build, not vanilla) | ✅ |
| `torch.cuda.is_available()` / device_count / get_device_name | ✅ |
| **bf16 GPU matmul on DLIN device** (real JIT compile + run) | ✅ |
| sglang editable install (`0.5.13.post2.dev560+…`) | ✅ |
| Full `import sglang` + `current_platform`→`DlinSRTPlatform` | ✅ (triton-LLVM preload, §6) |
| `sgl-kernel common_ops` **built with dlcc** (real ops run on DLIN GPU) | ✅ partial (§5: clean-source subset) |
| `sglang.Engine` **imports** (was blocked by `sgl_kernel`) | ✅ |
| `sglang.Engine(...)` actually loads a model + generates | ❌ next blocker: FlashInfer/CUTLASS ops (§5) |

**Critical environment rule** (the root cause of every "DLIN segfault" seen
during bring-up): `LD_LIBRARY_PATH` must contain **only the active SDK's
`lib`**. A second SDK lib dir on the path loads two copies of
`libhcrt`/`libLLVM`/`libcurt`, and the DLIN JIT crashes inside an LLVM
`PassBuilder` static initializer (`_GLOBAL__sub_I_PassBuilder.cpp`) on first
kernel compile. `run_sglang.sh::dlin_runtime_env` enforces this by
*overwriting* (not prepending to) `LD_LIBRARY_PATH`.

---

## 1. Design principle: DLIN is "CUDA-shaped"

DLIN exposes itself as CUDA: `device_type=="cuda"`, `torch.cuda.*` works,
`CUDA_VISIBLE_DEVICES`, `torch.version.cuda` is set (alongside
`torch.version.dl`). So **most of sglang runs unchanged**; the work is to
*route hot operators to DLIN-optimized implementations* and to *build native
extensions with `dlcc`*. This is the same posture vLLM's `DlPlatform` takes
(`_enum = PlatformEnum.CUDA`, `device_type = "cuda"`, `is_dl() -> True`).

Detection: `torch.version.dl is not None` (string, e.g. `"11.7"`). sglang has
**no** DLIN awareness today — all greenfield.

---

## 2. Where DLIN hooks plug into SGLang (the map)

SGLang already has the abstractions vLLM uses; we reuse them, we do not invent
new ones.

| Concern | SGLang location | DLIN action |
|---|---|---|
| Device detection | `python/sglang/srt/utils/common.py` (`is_cuda/is_hip/is_npu/…`) | add `is_dlin()` |
| Platform | `python/sglang/srt/platforms/{__init__.py,interface.py,cuda.py}` | add `platforms/dlin.py` `DlinSRTPlatform` (or OOT plugin) |
| Platform selection | `SGLANG_PLATFORM` env / entry-point `sglang.srt.platforms` | default-detect via `is_dlin()` |
| **Attention backend** | `layers/attention/attention_registry.py`; chosen in `server_args.py::_get_default_attn_backend` via `current_platform.get_default_attention_backend()` | return a DLIN attention backend |
| Kernel import | `import sgl_kernel` in `utils/common.py` (already try/except) | DLIN-built `sgl_kernel` or OOT DLIN ops |
| Op env-vars | `python/sglang/srt/environ.py` (+ skill `env-var-conventions`) | `SGLANG_USE_DL_*` toggles (mirror vLLM `dl_envs.py`) |
| Build | `sgl-kernel/{CMakeLists.txt,setup_rocm.py,pyproject*.toml}` | add `setup_dl.py` + `pyproject_dl.toml` (§5) |

The single highest-leverage hook is **`get_default_attention_backend()`** on the
platform — it is literally the one call OOT platforms use to redirect attention
(see `server_args.py`: `if current_platform.is_out_of_tree(): return
current_platform.get_default_attention_backend()`).

---

## 3. Phased roadmap

### Phase 1 — DLIN platform plugin (unblocks routing, no kernels yet) — ✅ DONE
1. ✅ `common.py`: `torch.version.dl` shim + `@lru_cache def is_dlin()`.
2. ✅ `device_mixin.py`: `is_dlin()` (default False).
3. ✅ `platforms/dlin.py`: `DlinSRTPlatform(CudaSRTPlatform)` — `is_dlin()->True`,
   `get_device_capability()->(12,0)`, `get_default_attention_backend()->"triton"`
   (Phase-2 TODO: return the DLIN flash-attn backend). Inherits CUDA so
   `is_cuda()` stays True (DLIN is CUDA-shaped, like vLLM's `DlPlatform`).
4. ✅ `platforms/__init__.py`: `_is_dlin_available()` checked **before** the CUDA
   fallback so a DLIN box resolves to `DlinSRTPlatform`.
5. Verified: files compile; `is_dlin()==True`, `_is_dlin_available()==True`,
   `torch.cuda.is_available()==True` on the KS38 host. (Runtime
   `current_platform.is_dlin()` blocked only by the host native-ext gap, §6.)

### Phase 2 — Attention: DLIN flash-attention backend
Register a `"dl_flash_attn"` backend in `attention_registry.py` that wraps
DLIN's flash-attn op (the `flash_attn` wheel on `dl-virtual`, or DLIN's MHA
kernel via `torch.ops`). `get_default_attention_backend()` returns it. Mirror
`vllm.plugins.dl_platform_plugin.dl_flash_attn.FlashAttentionBackend`.
- Gating env var: `SGLANG_USE_DL_FLASH_ATTN` (default on when `is_dlin()`).
- MLA models route to the DLIN MLA path.

### Phase 3 — GEMM / linear / norm / RoPE
Route through DLIN libs (cublas/dlblas, dldnn) via either:
- **torch op registration** (`torch.library.impl`) so sglang's `Linear`/`RMSNorm`
  calls dispatch to DLIN custom ops, or
- a thin `sgl_kernel`-style module of DLIN op wrappers.
Gate each with a `SGLANG_USE_DL_*` env var; default-on under `is_dlin()`.

### Phase 4 — MoE / quantization
DLIN has grouped-GEMM and quant kernels. Wire MoE (`fused_moe`) and quant
(AWQ/GPTQ/FP8) to DLIN paths, guarded by `is_dlin()`. This is the largest
surface; do it op-by-op with a numerical-correctness test per op (§7).

### Phase 5 — Full `sgl-kernel` DLIN build (AOT kernels)
Build the sgl-kernel C/CUDA sources natively with `dlcc` (§5). Until then,
sglang runs on DLIN via Triton + DLIN torch ops (Phases 1–4), with sgl-kernel
optional.

---

## 4. Operator packaging: three options, pick per-op

| Option | When | How |
|---|---|---|
| **A. torch op registration** | DLIN op exists as a `torch.ops.*` (from DLIN torch/FA wheel) | `torch.library.impl("aten::…", "dl")` or custom op; sglang calls unchanged |
| **B. JIT (Triton) kernel** | logic-shape op, DLIN torch has Triton | add to `python/sglang/srt/jit_kernel/` (see skill `add-jit-kernel`) |
| **C. AOT C/CUDA kernel** | perf-critical, no Triton equivalent | add to `sgl-kernel/csrc/` built with `dlcc` (skill `add-sgl-kernel`) |

Prefer **A** (zero build, uses DLIN's own wheels) wherever a DLIN op already
exists; fall back to **B**, then **C**.

---

## 5. Building sgl-kernel with `dlcc`

**Do not** use the default scikit-build/CMake path as-is: `project(... LANGUAGES
CUDA)` + `find_package(CUDAToolkit)` fails because CMake cannot recognize `dlcc`
as the CUDA compiler (`CMakeCUDAFindToolkit.cmake:148`, reproduced). Two viable
routes:

- **Route 1 (recommended, lowest risk) — mirror ROCm:** add
  `sgl-kernel/setup_dl.py` modeled on `setup_rocm.py`, using
  `torch.utils.cpp_extension.CUDAExtension` with `dlcc` as the `nvcc` compiler
  (`extra_compile_args={"nvcc": [...]}`). This bypasses CMake's CUDA language
  detection entirely (it's how ROCm/hipcc already works in this repo).
- **Route 2 — CMake toolchain file:** a `dlcc.toolchain.cmake` that sets
  `CMAKE_CUDA_COMPILER=$SDK/bin/dlcc` and satisfies `find_package(CUDAToolkit)`
  against the SDK layout.

Either way, three things must change vs. the NVIDIA build:
1. **GPU arch**: replace hardcoded `-gencode=arch=compute_90,code=sm_90`
   (CMakeLists lines ~127/196–237) with DLIN arch
   `--cuda-gpu-arch=dlgput64` (llama.cpp uses `dlgput64,dlgpu31`).
2. **FetchContent** (CUTLASS/FlashInfer/FlashAttention): gate or replace with
   DLIN-provided headers; not all upstream NVIDIA sources compile under `dlcc`.
3. **Subset first**: start with the elementwise/norm/pos-enc kernels that dlcc
   compiles cleanly; add GEMM/MoE/attention later. (See `setup_rocm.py`'s
   curated `sources=[...]` list for the pattern of an explicit subset.)

Add `sgl-kernel/pyproject_dl.toml` (build-backend = setuptools, like
`pyproject_rocm.toml`) and a `make build-dl` target.

---

## 6. The DLIN segfault — root cause & fix (RESOLVED)

During bring-up, `import sglang` (and `import torchvision` / `compressed_tensors`)
crashed with SIGSEGV. Systematic debugging (gdb + `LD_DEBUG` + order-swap tests)
showed **all** of these were a single root cause — not the per-wheel ABI gaps
suspected earlier.

### Root cause: two LLVM builds in one process (symbol interposition)
- `triton` ships `triton/_C/libtriton.so` (~440 MB) which **statically bundles
  its own LLVM**. **triton ≥ 3.2 bundles a newer LLVM (18/19)**; the DLIN SDK
  ships **LLVM 15** (`libLLVM-15.so`, loaded by DLIN torch).
- LLVM keeps extensive **non-isolated global state** (PassRegistry, DebugCounter,
  ManagedStatic). When torch loads the SDK's `libLLVM-15.so` first and then
  `libtriton.so` is dlopen'd, the two LLVMs interpose on each other's globals;
  `libtriton.so`'s `_GLOBAL__sub_I_PassBuilder.cpp` static init then frees a
  garbage pointer (`free(0x11)`) → SIGSEGV.
- I had been installing **vanilla `triton==3.7.1`** (from `dl-pypi-remote`) —
  that is what triggered it.

### Proof (order-dependent, reproducible)
| Test (DLIN torch 2.9.1+dl24) | Result |
|---|---|
| triton 3.7.x: `import torch` → `import triton._C` | **rc=139 (crash)** |
| triton 3.7.x: `import triton._C` → `import torch` | rc=0 (OK) |
| vanilla torch 2.12.1 (doesn't load `libLLVM-15.so`) | rc=0 any order |
| **triton 3.1.0** (any order) | **rc=0 (OK)** |
| gdb `import {torchvision,compressed_tensors}` | identical frame: `libtriton.so::_GLOBAL__sub_I_PassBuilder.cpp` |

### Fix: pin Denglin's triton 3.1.0 (proper, no workaround)
Denglin deliberately pins **`triton==3.1.0`** everywhere — `artifactory/piplike`
and `artifactory/download` both hardcode
`triton-3.1.0-cp312-cp312-manylinux_2_28_{arch}.whl`. Its bundled LLVM is **LLVM
15**, the same major version as the SDK's `libLLVM-15.so`, so the two copies are
ABI-compatible and coexist without corruption.

- `python/pyproject_dl.toml`: `srt_dl` pins `triton==3.1.0`.
- `run_sglang.sh`: installs `triton==3.1.0` explicitly from `dl-virtual`
  (`--no-deps`) before the editable install, so uv doesn't grab the vanilla build
  from `dl-pypi-remote`.
- (An earlier import-order preload in `sglang/__init__.py` was removed — it was a
  workaround for the wrong triton and is unnecessary once 3.1.0 is pinned.)

With this, `import sglang` + `current_platform`→`DlinSRTPlatform` + a DLIN GPU
matmul all pass — verified via `./run_sglang.sh test`.

### If a newer triton is ever required
A triton built to **link the SDK's shared `libLLVM-15.so`** (instead of bundling
its own) would remove the duplicate-LLVM dependency entirely and unlock newer
triton versions. There is no such `+dl` triton on `dl-virtual` today; it's a
wheel-build task.

### Remaining real version gaps (not segfaults, lower priority)
- `torchvision 0.24.1+sdk202606031721`: lazy-imported in `common.py` (GPU JPEG
  path only) pending a build matching the runtime SDK.
- `torchao`: sglang pins `==0.9.0` which forces `torch>=2.10` (vanilla). Needs a
  DLIN torchao 0.9.0 build, or relax the pin + guard quant paths.

**Runtime note (host vs Docker):** basic DLIN compute + `import sglang` now work
on this host; for full serving validation prefer the Docker path (`dev_ai.sh`)
as vLLM does, to sidestep host SDK/driver drift.

---

## 7. Test ladder (mirror vLLM `tests/dl/`)

Create `test/srt/dl/` (skill `write-sglang-test`):
1. `test_dlin_detect.py` — `is_dlin()`, platform resolves to `DlinSRTPlatform`, CC 12.0.
2. Per-op numerical tests: attention (ref vs DLIN), rmsnorm, rope, gemm, moe —
   each gated by `is_dlin()` so they skip on NVIDIA CI.
3. E2E: tiny model generate on 1× DLIN GPU, compare logits to CPU ref.
4. Benchmark hook: reuse `sgl-kernel/benchmark/` + skill `generate-profile`.

Gate the suite behind `SGLANG_PLATFORM=dlin` / `is_dlin()` so it never runs on
NVIDIA CI.

---

## 8. Sequenced next actions

1. **Phase 1**: `is_dlin()` + `platforms/dlin.py` + register; boot server with
   Triton backend on 1 GPU. (smallest credible "sglang runs on DLIN" milestone)
2. Publish/request DLIN torchvision matching runtime → drop the
   `common.py` lazy-import workaround.
3. **Phase 2**: DLIN flash-attn backend + numerical test.
4. `sgl-kernel/setup_dl.py` (Route 1) + `pyproject_dl.toml`; build elementwise
   subset with `dlcc`.
5. **Phase 3/4**: GEMM/norm/RoPE, then MoE/quant, op-by-op with tests.
6. DLIN compressed-tensors + torchao builds → full `import sglang` on host.

---

## Appendix — reference adaptation points (from sibling repos)

- **vLLM** (primary template): `vllm/plugins/dl_platform_plugin/dl_platform.py`
  (`DlPlatform`, `get_attn_backend_cls`, `get_device_capability`→12.0,
  `import_kernels`→`vllm._dl_C`), `vllm/dl_envs.py`, `vllm/torch_dl_version.py`
  (`torch.version.dl` shim), `setup.py::_is_dl()` + `get_dlcc_cuda_version()`
  (`$CUDA_HOME/bin/dlcc --version`), `tests/dl/`.
- **llama.cpp**: `run_llama.sh` (`source $SDK_DIR/env.sh`, cmake
  `-DGGML_DLCU=ON -DGGML_BACKEND_DL=ON -DSDK_DIR=…`), `ggml-dlcu/CMakeLists.txt`
  (compiler→`${SDK_DIR}/bin/dlcc`, `--cuda-gpu-arch=dlgput64`).
- **mooncake**: `run_mooncake.sh` (6-phase uv→bundle→compile→wheel→test),
  cmake `-DUSE_DLIN=ON -DDLIN_ROOT=$SDK`, exports
  `LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/stub`.

---

## 7. Known runtime blocker — triton JIT crashes the DLIN host runtime

**Status (2026-06-26): blocks end-to-end inference on the host. Deferred for
follow-up; does NOT block the sglang-side wiring (all done, see §0/§2/§5).**

Reproducible in-process with a clean env (`LD_LIBRARY_PATH=$SDK/lib` only,
venv-first PATH): non-trivial triton kernels segfault on the DLIN device.

| triton kernel | result |
|---|---|
| vector-add (1-D, elementwise) | sometimes rc=0 (flaky) |
| **matmul** (`tl.dot` / tensor cores) | **rc=139 SIGSEGV** |
| **softmax** (reductions, no `tl.dot`) | **rc=139 SIGSEGV** |

Crash frame (gdb): `dl::hc::ModuleImpl::GenSingleKernel` → `loadBinary` — the
DLIN runtime's kernel-binary loader crashes on triton-generated binaries.

Ruled out: NOT sglang code, NOT the FA backend (DLIN FA2 works in isolation:
varlen prefill returns correct output), NOT sgl-kernel (AOT ops run), NOT env
contamination (reproduced with `env -u LD_LIBRARY_PATH` + clean PATH, across
all local SDKs).

Suspected fix (needs follow-up):
- The installed triton is the **vanilla `manylinux_2_17`** from `dl-virtual`.
  Denglin blesses a **`manylinux_2_28`** triton shipped inside the SDK tarball
  (`artifactory/download` → `wheels/pytorch2.7.1/cp312/triton-3.1.0-cp312-cp312-manylinux_2_28_x86_64.whl`),
  which is presumably patched to emit DLIN-loadable binaries. SDK repos need
  auth (403 / `~/.netrc`) — not fetchable in this env.
- Or run under Docker (`dev_ai.sh`); the dl-env skill notes host JIT is unreliable.

Repro snippets: `/tmp/dl_triton_matmul.py`, `/tmp/dl_triton_softmax.py`,
`/tmp/dl_fa2_test.py` (FA2, works). See memory `dlin-sglang-adaptation`.

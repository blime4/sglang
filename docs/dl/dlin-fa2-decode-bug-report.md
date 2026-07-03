# DLIN: flash_attn_with_kvcache (paged-decode) crashes inside libdleol.so

sglang's decode step calls `flash_attn_with_kvcache` (the paged-KV-cache decode
API). On DLIN this reaches the dedicated dldnn op `cudnnMHAForwardKVCacheWithSinks`,
which crashes the process inside `libdleol.so`. A single standalone call
reproduces it (no model, no sglang). This report covers: what errors, why,
how to reproduce, how to fix.

## What goes wrong

Process dies (exit 255 / SIGSEGV). Last stderr line:

```
 to bc failed.
CUDA Driver API error = 0001 from file <.../dleol/dleol/src/op/flash_attn_kvcache_mha_op.cc>, line 867.
```

Call routing (confirmed from DLIN flash-attention source + the error filename;
the crash is a hard SIGSEGV inside closed libdleol.so, so no Python traceback --
only the dleol-emitted line above is printed):

```
flash_attn_with_kvcache   (Python)
  -> mha_fwd_kvcache              (flash_api.cpp:1487)
  -> dldnn_mha_fwd_kvcache        (flash_attn_dlgpu.cpp:958)
  -> cudnnMHAForwardKVCacheWithSinks  (flash_attn_dlgpu.cpp:1202)
  -> dleol flash_attn_kvcache_mha_op.cc:867   <-- "to bc failed" + CUDA err 1
  -> SIGSEGV
```

Routing is correct -- the call does reach the dedicated kvcache op
(`cudnnMHAForwardKVCacheWithSinks`), not the generic prefill op. The bug is
INSIDE that op's dleol JIT-bitcode handling, not in sglang dispatch.

## Why it errors (root cause)

dleol JITs the paged-decode kernel (`flash_attn_kvcache_mha_int32`) as:
generate C++/CUDA source -> compile to LLVM bitcode via dlcc -> load via the CUDA
driver (`cuModuleLoadData` / `getCUfunctionFromFatbin`). "to bc failed" means
that bitcode pipeline failed. Evidence pins it to the driver-side LOAD (not the
source, not dlcc):

+---+-----------------------------------------------------+----------------------+
| # | Finding (all verified on the env below)            | Implication          |
+---+-----------------------------------------------------+----------------------+
| 1 | The generated kernel SOURCE compiles FINE          | NOT a source / dlcc  |
|   | standalone with dlcc to BOTH .o AND .bc (94 KB     | bug. Bitcode itself  |
|   | .bc; only C++17-constexpr-if warnings under        | is valid.            |
|   | -std=c++14). Extracted from the dleol "compile     |                      |
|   | src:" dump and compiled manually.                  |                      |
+---+-----------------------------------------------------+----------------------+
| 2 | The failure is CUDA Driver error 0001              | The driver REJECTS   |
|   | (INVALID_VALUE) at flash_attn_kvcache_mha_op.cc:867| the bitcode when     |
|   | -- i.e. consuming the bitcode to build the kvcache | building the kvcache |
|   | kernel fails.                                      | kernel.              |
+---+-----------------------------------------------------+----------------------+
| 3 | Config-INDEPENDENT: page_size in {16,32,64,128}    | Not a shape/param    |
|   | all crash identically (head_dim 128, bf16).        | tuning issue.        |
+---+-----------------------------------------------------+----------------------+
| 4 | Reproduces on FRESH, never-used GPUs.              | Not device-state     |
|   |                                                     | corruption.          |
+---+-----------------------------------------------------+----------------------+
| 5 | Prefill (flash_attn_func) works on the SAME SDK.   | Isolated to this op. |
+---+-----------------------------------------------------+----------------------+
| 6 | libdleol_aot.so (1.2 GB) ALREADY contains          | Kernel exists AOT;   |
|   | precompiled __flash_attn_kvcache_mha_int32         | the runtime          |
|   | variants (tile 64/128/512, bf16).                  | JIT/AOT-load path is |
|   |                                                     | what is broken.      |
+---+-----------------------------------------------------+----------------------+
| 7 | DLEOL_DISABLE_JIT=1 changes the failure to         | The AOT lookup path  |
|   | std::invalid_argument: stoi (SIGABRT), NOT the JIT | has a SEPARATE       |
|   | segfault.                                          | stoi-parse bug.      |
+---+-----------------------------------------------------+----------------------+

dlecc invocation dleol uses for JIT (from libdleol.so strings):

```
dlcc -std=c++14 --offload-arch=dlgput64,dlgpux64   (then -emit-llvm for bitcode)
```

Build path prefix of the failing op:
`/LocalRun/jenkins/build/V2_SOFTWARE_master_manylinux_2_28-x86_64/ai_software/dlop/dleol/dleol/src/op/flash_attn_kvcache_mha_op.cc`

## How to reproduce

1. Set the clean DLIN runtime env (see header of `repro_fa2_decode_crash.py`):
   `SDK_DIR`, `CUDA_HOME=$SDK_DIR`, `DLI_V2=ON`,
   `LD_LIBRARY_PATH=$SDK_DIR/lib` (only), `CUDA_VISIBLE_DEVICES=<free GPU>`.
2. Activate the venv with `torch==2.9.1+dl24.sdk202606031721`.
3. Run the standalone repro (no model, no sglang; ~30 s):

```
python repro_fa2_decode_crash.py
```

Repro body (Qwen3-1.7B decode shape: GQA 16/8 heads, head_dim 128, page 16):

```
from flash_attn import flash_attn_with_kvcache
import torch
q  = torch.randn(2,1,16,128, dtype=torch.bfloat16, device="cuda")
kc = torch.randn(4,16,8,128, dtype=torch.bfloat16, device="cuda")
vc = torch.randn(4,16,8,128, dtype=torch.bfloat16, device="cuda")
bt = torch.zeros(2,4, dtype=torch.int32, device="cuda")
csl= torch.full((2,), 32, dtype=torch.int32, device="cuda")
flash_attn_with_kvcache(q, kc, vc, cache_seqlens=csl, block_table=bt, causal=True)
```

Expected: returns `[B,1,Hq,D]` + softmax_lse. Actual: SIGSEGV.

## Workaround (sglang-side, verified)

The crash is specific to the paged-KV-cache op `flash_attn_with_kvcache`
(`cudnnMHAForwardKVCacheWithSinks`). sglang does NOT need that op: its DLIN
decode path routes through `vllm_flash_attn.flash_attn_varlen_func(block_table=)`
instead, which hits a DIFFERENT dldnn op -- `cudnnMHAVarlenForward*` (the prefill
varlen op, which works and is cuda-graph-capturable). See
`python/sglang/jit_kernel/flash_attention.py` (`_dlin_vllm_flash_attn_ok()` branch).

What unblocks sglang serving in practice:

+---+--------------------------------------------+----------------------------------------+
| # | Step                                      | Where                                  |
+---+--------------------------------------------+----------------------------------------+
| 1 | Copy the DLIN-patched `_vllm_fa2_C.so`    | `scripts/dl/setup_vllm_flash_attn.sh`  |
|   | from a vLLM DL venv (dl19 build,          |                                        |
|   | ABI-compatible with dl24 torch) into the  |                                        |
|   | sglang venv.                              |                                        |
+---+--------------------------------------------+----------------------------------------+
| 2 | Let model warmup run first -- per the     | sglang engine warmup                   |
|   | setup script, this preheats the dleol JIT |                                        |
|   | cache so decode never hits the cold-JIT   |                                        |
|   | "to bc failed" path at serve time.        |                                        |
+---+--------------------------------------------+----------------------------------------+

**Update (2026-07-01, Route B):** the copy step above is now superseded — sglang
builds its own `_vllm_fa2_C.so` from the DLIN flash-attention source via
`scripts/dl/build_dlin_vllm_flash_attn.sh` (no vLLM venv needed). The built .so is
bit-identical to the copied one (max diff 0.0, vs SDPA max_err 5e-4). `setup_vllm_flash_attn.sh`
is now a thin shim that delegates to the from-source build.

Measured result (Qwen3-1.7B, batch=1, KS38): eager **19.52 tok/s** and a CLEAN
FULL cuda graph **16.64 tok/s** (output "Paris..." correct). See
`docs/dl/sglang-vs-vllm-perf-gap.md`.

Caveat: this ROUTES AROUND the bug, it does not fix `flash_attn_with_kvcache`
itself. Any framework that must call `flash_attn_with_kvcache` directly still
hits the crash and needs the DLIN-side fix below. The standalone repro
(`repro_fa2_decode_crash.py`) still crashes because it calls the kvcache op
cold, with no warmup.

## How to fix

The bug is in closed `libdleol.so` (the kvcache op's bitcode load). DLIN-side
fix, any one unblocks it:

1. Fix the AOT lookup `stoi` parse bug so the precompiled
   `flash_attn_kvcache_mha_int32` variant (already in libdleol_aot.so) is
   resolved. Ship the page-16 / head-128 instantiation if missing.
2. OR fix the JIT-bitcode driver load (`cuModuleLoadData` /
   `getCUfunctionFromFatbin`) at `flash_attn_kvcache_mha_op.cc:867` that
   returns CUDA_ERROR_INVALID_VALUE ("to bc failed").

Either makes `flash_attn_with_kvcache` work, which also unblocks cuda-graph
decode (this kernel is the graph-safe decode path).

## Impact

- Blocks autoregressive decode for any framework using `flash_attn_with_kvcache`
  for paged decode (sglang). Prefill / matmul unaffected.

## Environment

+--------------------------+------------------------------------------------------+
| Item                     | Value                                                |
+--------------------------+------------------------------------------------------+
| SDK                      | denglin MRrc-4.2.0-202606161052                      |
| torch                    | 2.9.1+dl24.sdk202606031721 (DLIN-patched)            |
| flash_attn               | DLIN FA2 build V2_SOFTWARE_master_202606031721       |
| dlcc                     | clang 15.0.6, --cuda-gpu-arch=dlgput64               |
| libdleol.so              | /sdk/lib/libdleol.so (dated 2026-06-01)              |
| libdleol_aot.so          | /sdk/lib/libdleol_aot.so, 1.2 GB (precompiled kerns) |
| Driver                   | 2.3.26 (DL-SMI 12.0)                                 |
| GPU                      | Denglin KS38 QUAD, 32 GB                             |
| OS                       | Linux x86_64                                         |
| LD_LIBRARY_PATH          | $SDK_DIR/lib ONLY (a second SDK dir -> a different   |
|                          | crash: duplicate libhcrt/libLLVM, LLVM PassBuilder)  |
+--------------------------+------------------------------------------------------+

## Artifacts

+-----------------------------------+----------------------------------------------+
| Artifact                          | Description                                  |
+-----------------------------------+----------------------------------------------+
| repro_fa2_decode_crash.py         | Standalone minimal repro (this report's dir) |
| dleol error line                  | "to bc failed" + CUDA err 1 at               |
|                                   | flash_attn_kvcache_mha_op.cc:867             |
| generated kernel source           | ~16k-line dleol "compile src:" dump;         |
|                                   | compiles fine standalone (dlcc -c and        |
|                                   | -emit-llvm)                                  |
| AOT kernel symbol                 | __flash_attn_kvcache_mha_int32T... variants  |
|                                   | in libdleol_aot.so (proof kernel exists AOT) |
+-----------------------------------+----------------------------------------------+

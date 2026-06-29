# Request: dl24 `vllm_flash_attn` wheel for sglang (DLIN)

## Ask

Please provide a **DLIN-patched `vllm_flash_attn` wheel** built against
**torch==2.9.1+dl24.sdk202606031721** (Python 3.12, linux x86_64), shipping the
compiled `_vllm_fa2_C` extension (the one that routes to `cudnnMHAVarlenForward*`).

## Why sglang needs it

On DLIN, sglang's decode currently can't use the clean, graph-safe path:

- The dedicated paged-decode op `cudnnMHAForwardKVCacheWithSinks` (reached via
  `flash_attn_with_kvcache`) crashes inside `libdleol.so` ("to bc failed" -- a
  separate bug, reported in `dlin-fa2-decode-bug-report.md`).
- The working paged-decode path is the **varlen op `cudnnMHAVarlenForward*`**,
  reached via `vllm_flash_attn.flash_attn_varlen_func(..., block_table=,
  seqused_k=...)`. **vLLM uses exactly this and decodes fine.**
- sglang's venv has only a **vanilla `vllm_flash_attn` (pure-python stub, no
  `_vllm_fa2_C`)**, so it cannot call that path.

With a dl24 DLIN `vllm_flash_attn`, sglang routes decode through the working
varlen op (fixed-shape tensors, graph-safe) -> decode works AND cuda-graph decode
is unblocked. sglang's integration is already written and waiting.

## Why the existing dl9 / dl19 builds do NOT work

The `_vllm_fa2_C.so` already on the box were compiled against a different DLIN
torch generation:

+-------------------------------+----------------------------------------+
| venv                          | torch                                  |
+-------------------------------+----------------------------------------+
| sglang (target)               | 2.9.1+dl24.sdk202606031721             |
| venv-vllm021                  | 2.9.1+dl19.sdk202603121743             |
| support-qwen3.6 / flash-attn  | 2.9.1+dl9.sdk202603121743              |
+-------------------------------+----------------------------------------+

`+dlNN` are different DLIN torch builds -> ABI mismatch; copying or installing a
dl9/dl19 `_vllm_fa2_C` into sglang's dl24 venv fails (undefined symbol / crash).
artifactory has no dl24 `vllm_flash_attn` wheel either.

## How to produce it (DLIN side)

`vllm_flash_attn` (with `_vllm_fa2_C`) is produced by **vLLM's CMake build**
(`cmake/external_projects/vllm_flash_attn.cmake`, FetchContent of the DLIN
flash-attention source, wrapped into the `vllm_flash_attn` package). The DLIN
`dev.sh` / `dev_ai.sh` trigger this via `-e VLLM_FLASH_ATTN_SRC_DIR=<flash-attn
repo>`. So: run that vLLM build **against torch==2.9.1+dl24.sdk202606031721** and
ship the resulting `vllm_flash_attn` package (it contains
`_vllm_fa2_C.cpython-312-x86_64-linux-gnu.so`).

(The flash-attention repo's standalone `setup.py` builds `flash_attn`, not
`vllm_flash_attn` -- so it must go through vLLM's build wrapping.)

## What sglang does once the wheel is installed (ready)

Switch decode to the vLLM-style call (graph-safe), drop the current gather
workaround:

```
vllm_flash_attn.flash_attn_varlen_func(
    q, k_cache, v_cache,
    max_seqlen_q=1, cu_seqlens_q=cu_seqlens_q,
    max_seqlen_k=max_seqlen_k, seqused_k=cache_seqlens,
    softmax_scale=scale, causal=causal, block_table=page_table,
)
```

Then verify: output vs torch SDPA, end-to-end Qwen3 decode, and cuda-graph
on/off tok/s.

## Target spec

- package: `vllm_flash_attn` (DLIN-patched), with compiled `_vllm_fa2_C`
- torch: `2.9.1+dl24.sdk202606031721`
- python: 3.12
- platform: manylinux x86_64
- must expose `torch.ops._vllm_fa2_C.varlen_fwd` (the block_table paged-decode path)

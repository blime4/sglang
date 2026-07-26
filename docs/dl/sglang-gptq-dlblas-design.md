---
title: "SGLang GPTQ-DLBLAS — DLIN-native GPTQ (2/3/4/8-bit) Support (Design Spec)"
date: 2026-07-20
status: design — awaiting review
model: Qwen3.5-35B-A3B-GPTQ-Int4
hardware: Denglin DLIN KS38 (×4, TP4)
baseline: vLLM 0.21.1 GPTQ-Int4 token-for-token output (same checkpoint)
references:
  - docs/dl/dflash-on-dlin-debug-blog.md             (§5: FP8 reuse of gptq_dlblas_gemmex)
  - vllm-new-overlay/vllm/plugins/dl_quantization_plugin/gptq_dlblas.py  (reference impl)
  - vllm-new-overlay/vllm/_custom_ops.py             (gptq_dlblas_gemmex op signature)
  - python/sglang/srt/layers/quantization/fp8.py     (FP8 reuse path — pattern to mirror)
  - python/sglang/srt/layers/quantization/fp8_utils.py (_ensure_dl_C, gemmex call sites)
---

# SGLang GPTQ-DLBLAS — DLIN-native GPTQ (2/3/4/8-bit) Support

> **One-line goal.** Give SGLang a DLIN-native GPTQ path (W4A16, plus 2/3/8-bit)
> by porting vLLM's already-shipped `GPTQDLBLAS*` scheme into SGLang's GPTQ
> module — **reusing the `gptq_dlblas_gemmex` kernel SGLang's FP8 path already
> loads**, no new kernel work. First milestone is correctness
> (token-for-token match vs vLLM GPTQ on the same checkpoint); performance
> (batched MoE GEMM, vLLM-overlay decoupling) is Phase 2.

## 0. Goal & success criteria

| | Criterion | Target |
|---|---|---|
| G1 | SGLang loads & serves Qwen3.5-35B-A3B-GPTQ-Int4 end-to-end on DLIN | greedy decode runs, no crash, no Marlin-PTX fallback |
| G2 | Dense linear + MoE both route through `gptq_dlblas_gemmex(quant_type=0, bit=w)` | confirmed by env trace (`SGLANG_DL_GPTQ_TRACE`), gemmex fires for q/k/v/o_proj and every MoE expert |
| G3 | Correctness: token-for-token match vs vLLM GPTQ-Int4 on same checkpoint | identical greedy output for the test prompts (≤ allowed numerical tolerance on logits, exact on argmax tokens) |
| G4 | Bit-width coverage: 2/3/4/8-bit all dispatch correctly | per-bit repack unit tests pass; int4 is the primary checkpoint |
| G5 | CG-capturable | the GPTQ decode path runs under the existing CUDA-graph capture (kernel already CG-proven via FP8) |

**Constraints (user-confirmed).**
- **Bit-width scope:** 2/3/4/8-bit full coverage (not int4-only) — the 3-bit
  unpack/repack path is in scope.
- **Repack helpers:** Phase 1 **imports** them from `vllm-new-overlay` (same
  coupling the FP8 path already has via `_dl_C`); Phase 2 migrates them into
  SGLang-native code. A single thin wrapper file isolates the dependency.
- **Correctness baseline:** the same Qwen3.5-35B-A3B-GPTQ-Int4 checkpoint run
  through vLLM — token-for-token, not vs bf16.
- **Execution:** **staged** — Phase 1 correctness only (no MoE batched-GEMM
  optimization); Phase 2 perf + decoupling.

## 1. Context — the key fact, audited against code

**The decisive fact:** the kernel SGLang's FP8 path calls is *itself a GPTQ
kernel*. Its vLLM op signature (`vllm-new-overlay/vllm/_custom_ops.py:742`):

```python
def gptq_dlblas_gemmex(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                       quant_type, bit: int = 4)   # bit defaults to 4
```

`quant_type` semantics (from vLLM `gptq_dlblas.py` `_apply_original` comment):

| quant_type | meaning |
|---|---|
| 0 | **gptq** (the native/main mode) |
| 1 | fp8 per-channel |
| 2 | fp8 blockwise ← what SGLang FP8 uses today |
| 3 | blockwise int8 |

SGLang's existing FP8 path reuses this kernel with `quant_type=2, bit=8`
(`fp8_utils.py:562`, `fp8.py:1983`). The kernel is already dlopened + registered
opaque by `_ensure_dl_C()` (`fp8_utils.py:503`), and already proven
CG-capturable (FP8 decodes under CUDA graph with it). **Therefore: no new kernel
work, no new `.so`, no new op registration is needed for GPTQ.** The GPTQ main
mode is `quant_type=0` with `bit = weight_bits`.

vLLM ships a complete reference implementation in
`vllm-new-overlay/vllm/plugins/dl_quantization_plugin/gptq_dlblas.py`:
- `GPTQDLBLASConfig` (config + dispatch, `:138`)
- `GPTQDLBLASLinearMethod` (dense linear, `:370`)
- `GPTQDLBLASMoEMethod` (MoE W4A16, `:958`) — `create_weights` with uint8
  packed qweight + fp16 scales/qzeros + group_size halving for TP; per-expert
  gemmex calls.
- `process_weights_after_loading` — int32→uint8 repack, qzeros→fp16, sym fill,
  desc_act shuffle.

SGLang's GPTQ module (`python/sglang/srt/layers/quantization/gptq/`) currently
has schemes for NVIDIA (`GPTQLinearScheme` Triton, `GPTQMarlin*` PTX), Ascend
NPU, and Intel AMX (`gptq/schemes/__init__.py`) — **no DLIN scheme**. On DLIN,
the default Triton `gptq_gemm` path is slow/untested, and Marlin is a hard
dead-end (DLIN dlcc cannot compile the NVIDIA-specific PTX — noted at
`fp8.py:352`). So a fully-quantized GPTQ-Int4 checkpoint cannot be served on
DLIN today.

## 2. Architecture — components & boundaries

Three new/changed units, each with one clear purpose:

### 2.1 `gptq/schemes/gptq_dlblas.py` (NEW) — the DLIN GPTQ scheme

Two classes mirroring vLLM's, but speaking SGLang's scheme interfaces:

- **`GPTQDLBLASLinearScheme`** (dense: q/k/v/o_proj, lm_head-ish linears)
  - `create_weights` — int32 `qweight`/`qzeros`, fp16 `scales`, `g_idx`; weight
    attrs aligned with vLLM's layout (SGLang's existing `GPTQLinearScheme`
    weight attrs are the scaffold).
  - `process_weights_after_loading` — delegate bit-width repack to
    `_dl_gptq_utils` (§2.2).
  - `apply` — `torch.ops._dl_C.gptq_dlblas_gemmex(x, qweight, qzeros, scales,
    quant_type=0, bit=weight_bits)`.

- **`GPTQDLBLASMoEScheme`** (MoE: the compute bulk on 35B-A3B)
  - `create_weights` — port vLLM `GPTQDLBLASMoEMethod.create_weights` (`:958`):
    uint8 `w13_qweight[E, 2*inter, K//pack]` / `w2_qweight[E, hidden, K//pack]`,
    fp16 `w13_scales`/`w2_scales`, `w13_qzeros`/`w2_qzeros`, with the
    `group_size` halving loop for TP. SGLang's `GPTQMoEAscendScheme`
    (`gptq_moe.py:20`, already group-quant with int32 qweight/qzeros/scales) is
    the closest in-repo scaffold — adapt its shapes to uint8 + the DLIN kernel.
  - `apply_weights` — **Phase 1: per-expert loop** mirroring `fp8.py:1974-2019`
    (`SGLANG_DL_MOE_GEMMEX=1`): for each routed expert k, call
    `gptq_dlblas_gemmex(x_k, w13_qweight[k], w13_qzeros[k], w13_scales[k],
    quant_type=0, bit)` then likewise for w2. Phase 2 upgrades to batched GEMM.

### 2.2 `gptq/_dl_gptq_utils.py` (NEW) — thin repack-helper wrapper

Single file that imports the vLLM-overlay helpers behind SGLang-named
functions, so Phase 2 migration is a one-file swap:

```python
# Phase 1: import from vLLM overlay (same coupling as _dl_C)
from vllm.plugins.dl_quantization_plugin.gptq_dlblas import (
    param_int32_to_2_4_8bit_uint8,   # 2/4/8-bit shared repack
    unpack_int32_to_3bit_uint8_row,  # 3-bit
    pack_3bit_to_uint8,              # 3-bit
    param_int32_to_fp16_weights,     # qzeros -> fp16
)
from vllm import ops as vllm_ops     # ops.gptq_shuffle (desc_act)
```

(Symbols to be confirmed present at import time during implementation; if a
helper lives elsewhere in the plugin, this file is where the indirection
happens. The point is the scheme code never imports vLLM directly.)

### 2.3 `gptq/gptq.py` config dispatch (CHANGED)

Add a DLIN branch in:
- `GPTQConfig.get_quant_method` (`:172` linear, `:206` MoE) and
- `GPTQMarlinConfig.get_quant_method` (`:410`)

…gated by `sglang.srt.utils.common.is_dlin()` (`utils/common.py:165`, checks
`torch.version.dl is not None`) — **the exact check the FP8 path uses**
(`fp8.py:357`, `fp8_utils.py:632`). When `is_dlin()` is true, return the new
`GPTQDLBLAS*` scheme. **Hard rule: on DLIN, never select Marlin**
(`GPTQMarlinConfig.get_quant_method` must short-circuit to the DLIN scheme, not
the PTX path).

## 3. Bit-width / special-branch handling

Driven by the checkpoint's `quantize_config.json` (`bits`, `group_size`, `sym`,
`desc_act`):

| bit | qweight repack | notes |
|---|---|---|
| 2 / 4 / 8 | `param_int32_to_2_4_8bit_uint8` shared branch | main path; int4 is the target checkpoint |
| 3 | `unpack_int32_to_3bit_uint8_row` → `pack_3bit_to_uint8` | requires hidden/inter `% 8 == 0` |
| `sym=True` | `qzeros.fill_(8 if bit==4 else 2 if bit==2 else 128)` | ignore checkpoint qzeros |
| `desc_act=True` | `ops.gptq_shuffle(qweight, argsort(g_idx), bits)` + g_idx argsort | activation reorder path |

`group_size` interacts with TP (halving loop in `create_weights`); `pack_factor
= 8 // bits` for 2/4/8.

## 4. Phasing

### Phase 1 — Correctness (the committed milestone)
1. `_dl_gptq_utils.py` wrapper importing vLLM-overlay helpers.
2. `GPTQDLBLASLinearScheme` (dense) — repack + gemmex apply.
3. `GPTQDLBLASMoEScheme` (MoE) — per-expert loop + repack.
4. Config dispatch in `gptq.py` (DLIN branch, Marlin guard).
5. CG smoke (decode runs under existing capture; kernel is already CG-proven).
6. **Acceptance G3:** token-for-token match vs vLLM GPTQ-Int4.

### Phase 2 — Performance + decoupling (deferred, separate spec/plan)
1. MoE batched GEMM (`GEMMEX=2/3` style: w1 batched over experts, w2 per-expert).
2. Migrate `_dl_gptq_utils` helpers to SGLang-native (remove vLLM-overlay
   coupling for the repack layer).
3. torch.compile + CG full validation; TPOT bench vs FP8.
4. Document whether INT4 beats FP8 on this workload (see §5 risk).

## 5. Risks & open questions

- **vLLM-overlay coupling (Phase 1).** Accepted — same dependency shape as the
  FP8 path's `_dl_C`. Removed in Phase 2.
- **3-bit path complexity.** Separate unpack/repack; needs a dedicated unit
  test comparing packed output to vLLM's for identical int32 input.
- **MoE per-expert loop performance (Phase 1).** Deliberately unoptimized; may
  be slow. Acceptable because Phase 1's gate is correctness, not throughput.
- **TPOT vs FP8 — no preset conclusion.** Per prior DLIN measurements on
  35B-A3B, the target verify is *compute-bound* (GDN/attention-extend
  dominated, not MoE-memory-bound). FP8's hardware-fused blockwise dequant may
  remain faster than INT4's W4A16 (dequant-to-fp16 + bf16 GEMM). INT4's real
  upside may be **memory footprint / larger batch**, not lower TPOT. The plan
  must measure, not assume.
- **Helper symbol availability.** Whether every repack helper is importable at
  the expected path is to be confirmed at implementation start; the wrapper
  file is the designed place to absorb any path drift.
- **`quantize_config.json` field variance.** Different GPTQ checkpoints set
  `sym`/`desc_act`/`group_size` differently; all four branches (sym×desc_act)
  should be exercised if multiple checkpoints are available, else document
  which the single target checkpoint exercises.

## 6. Out of scope (YAGNI)

- AWQ / W4AF8 / compressed-tensors / MXFP4 — separate quant formats, not
  touched here.
- GPTQ **training** or re-quantization — inference serving only.
- Per-expert mixed bit-width (vLLM has `parse_moe_dynamics_key` for this) —
  single bit-width per layer only, matching the target checkpoint.

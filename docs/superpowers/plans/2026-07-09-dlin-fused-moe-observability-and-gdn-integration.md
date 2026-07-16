# DLIN Fused MoE Observability and GDN Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current DLIN Fused MoE path explicitly observable and verifiable without regressing the Qwen3.5-35B-A3B-FP8 throughput baseline, then add DLIN-native GDN stage-timing measurement so decode/prefill/verify costs are visible before any kernel work.

**Architecture:** Keep the existing inline DLIN specialization inside `Fp8MoEMethod.apply()` — do not promote it to a formal `MoeRunner` backend. Add (a) a pure routing-selector function that is the single source of truth for the MoE path label, (b) structured, env-gated path tracing around the fused / bf16-bmm / generic / fallback-error branches, (c) a GPU-free unit test for the selector, (d) a real fused-vs-reference correctness matrix and microbench for `M={1,9,16}`, and (e) stage timing on the real `GDNKernelDispatcher` methods. No speculative kernel scaffolding is added — GDN-native integration is deferred until a native kernel exists.

**Tech Stack:** Python, PyTorch, sglang runtime, DLIN `_dl_C` custom ops, the `Envs` env-var registry (`python/sglang/srt/environ.py`), existing `scripts/dl/*` benchmarking harnesses.

## Global Constraints

- **Env vars go through `Envs`.** All new `SGLANG_*` vars are registered as `EnvField` descriptors in `python/sglang/srt/environ.py` and accessed via `envs.NAME.get()`. Never add a new `get_bool_env_var("SGLANG_…")` or `os.environ.get("SGLANG_…")` call site (see the `env-var-conventions` skill). Debug/trace knobs use the `DEBUG_` verb.
- Reuse the existing DLIN inline MoE path in `python/sglang/srt/layers/quantization/fp8.py`; do **not** promote it to a formal `MoeRunner` backend.
- Preserve current DLIN behavior for large `M`: fused path stays threshold-gated (`SGLANG_DL_MOE_FUSED_MAX_M`, code default 16) and larger `M` continues to use the stable bf16-bmm fallback (`SGLANG_DL_MOE_MAX_BF16_M`, code default 2048; team serving baseline uses 128 via `DLIN_SERVER_ENV`).
- Make DLIN path selection observable: fused hit, bf16-bmm hit, generic fall-through, and fallback-on-error must be distinguishable without reading code.
- Do not introduce silent fallback for DLIN MoE branches; on exception, emit `fallback_error` then fall through to the upstream generic runner (current behavior).
- Tracing is **capture-time only**: the trace calls are pure-Python side-effects, so under cuda graph they fire during capture and are skipped on replay. They must add zero overhead when the trace flag is off (early-return) and must not branch on tensor values.
- **Non-regression is throughput-only.** Output-quality (coherence) is **not** a pass criterion for NGRAM/MTP runs in this plan — the separate spec-verify prompt-regeneration bug makes speculative output quality unreliable today. Only the non-spec e2e run keeps a light coherence sanity check. Guardrail band: decode `18.32 tok/s`, short e2e `16.04 tok/s`, prefill `~50 tok/s`, NGRAM `35–40 tok/s`.
- Fused MoE correctness coverage must include representative `M={1,9,16}` cases (decode M=1, NGRAM verify M≈9, short prefill M=16).
- Reuse existing scripts where possible: `scripts/dl/run_qwen35_35b.py`, `scripts/dl/qwen35_sg_tps.py`, `scripts/dl/e2e_correctness_speed.py`, `scripts/dl/ngram_test.py`, `scripts/dl/benchrun_sglang.py`, `scripts/dl/ttft_tpot.py`, `scripts/dl/test_moe_dlblas.py`.
- Do not add new `@dataclass` types; if a structured container is needed, use `msgspec.Struct`.
- Speculative-decoding identifiers (if any are introduced) must follow the `speculative-naming` skill. This plan adds none — it only prints traces inside `ngram_test.py`.
- Follow project Python style and keep changes surgical; every changed line must map to this plan's goals. DL edits must be wrapped in `# DL begin` / `# DL end` markers (enforced by `scripts/dl/check_dl_markers.py`).

---

## File Structure

- `python/sglang/srt/environ.py`
  - The `Envs` registry. Gains a new `# DLIN (Denglin) debug instrumentation` section with two `EnvBool(False)` trace knobs.
- `python/sglang/srt/layers/quantization/fp8.py`
  - Existing DLIN Fused MoE (≈line 1905) and bf16-bmm (≈line 1957) inline branches, plus the `except` (≈line 2007).
  - Gains the pure `_dlin_moe_select_path()` selector, the trace helpers, and one trace call per branch/fallback. `envs` is already imported (line 19).
- `scripts/dl/test_dlin_moe_routing.py` *(new)*
  - GPU-free unit test for `_dlin_moe_select_path()` over the routing table.
- `scripts/dl/test_moe_dlblas.py`
  - Existing standalone DLIN fused-MoE op microbench/correctness script.
  - Turned into a parameterized correctness + microbench script over `M={1,9,16}` with real `fused_moe()` / `ref_moe()` / `run_case()` functions.
- `scripts/dl/e2e_correctness_speed.py`, `scripts/dl/qwen35_sg_tps.py`, `scripts/dl/ngram_test.py`, `scripts/dl/ttft_tpot.py`
  - Existing in-process engine harnesses (each builds `sglang.Engine`, prints a `[cfg]`/`[benchN]`/`[OUT-START]…[OUT-END]` convention, and calls `e.shutdown()`).
  - Each gains a reusable `print_dl_moe_trace_summary()` call right before `e.shutdown()` and trace env values in its config banner.
- `scripts/dl/benchrun_sglang.py`
  - Existing serving harness. `DLIN_SERVER_ENV` (≈line 42) gains one added key; never replace the dict (it holds the critical `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` offline keys).
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`
  - `GDNKernelDispatcher` (line 58) with methods `packed_decode` (152), `decode` (185), `extend` (214), `target_verify` (239). Already logs the selected kernel class at line 146.
  - Gains env-gated stage-timing helpers and a timing wrap around `decode`/`extend`/`target_verify`/`packed_decode`. Needs `import time` and `from sglang.srt.environ import envs` added.
- `docs/dl/dlin-vllm-sglang-gap-analysis.md`
  - Existing gap-analysis doc. Updated with MoE-observability status and GDN timing checkpoint.

## Task 1: Register DLIN trace env vars, add the MoE routing selector + path tracing

**Files:**
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/layers/quantization/fp8.py`
- Test: `scripts/dl/test_dlin_moe_routing.py`

**Interfaces:**
- Consumes:
  - existing envs `SGLANG_DL_MOE_FUSED`, `SGLANG_DL_MOE_FUSED_MAX_M`, `SGLANG_DL_MOE_DLBLAS`, `SGLANG_DL_MOE_MAX_BF16_M` (read inline in `fp8.py` today; left as-is — pre-existing tech debt, out of scope)
  - `is_dlin()` from `sglang.srt.utils.common`
- Produces:
  - `Envs.SGLANG_DEBUG_DL_MOE_TRACE = EnvBool(False)` and `Envs.SGLANG_DEBUG_DL_GDN_TRACE = EnvBool(False)`
  - `_dlin_moe_select_path(*, m, is_dlin, fused_enabled, fused_max_m, dlblas_enabled, bf16_max_m) -> str` returning one of `"fused" | "bf16_bmm" | "generic_runner"`
  - `_record_dlin_moe_path(path: str, m: int, topk: int, reason: str | None = None) -> None`
  - `_get_dlin_moe_trace_summary() -> str`

- [ ] **Step 1: Write the failing GPU-free routing test**

Create `scripts/dl/test_dlin_moe_routing.py`:

```python
#!/usr/bin/env python3
"""GPU-free unit test for the DLIN MoE routing selector (no engine, no GPU).

Validates that _dlin_moe_select_path matches the documented routing of the
inline branches in fp8.py:
  fused       : is_dlin and FUSED==1 and m <= FUSED_MAX_M
  bf16_bmm    : is_dlin and DLBLAS!=0 and m <= BF16_MAX_M   (only if fused did not apply)
  generic_runner : otherwise
"""
from sglang.srt.layers.quantization.fp8 import _dlin_moe_select_path as select

# (kwargs, expected_path)
CASES = [
    (dict(m=1,    is_dlin=True,  fused_enabled=True,  fused_max_m=16, dlblas_enabled=True,  bf16_max_m=2048), "fused"),
    (dict(m=9,    is_dlin=True,  fused_enabled=True,  fused_max_m=16, dlblas_enabled=True,  bf16_max_m=2048), "fused"),
    (dict(m=16,   is_dlin=True,  fused_enabled=True,  fused_max_m=16, dlblas_enabled=True,  bf16_max_m=2048), "fused"),
    (dict(m=17,   is_dlin=True,  fused_enabled=True,  fused_max_m=16, dlblas_enabled=True,  bf16_max_m=2048), "bf16_bmm"),
    (dict(m=16,   is_dlin=True,  fused_enabled=False, fused_max_m=16, dlblas_enabled=True,  bf16_max_m=2048), "bf16_bmm"),
    (dict(m=17,   is_dlin=True,  fused_enabled=True,  fused_max_m=16, dlblas_enabled=False, bf16_max_m=2048), "generic_runner"),
    (dict(m=2049, is_dlin=True,  fused_enabled=True,  fused_max_m=16, dlblas_enabled=True,  bf16_max_m=2048), "generic_runner"),
    (dict(m=1,    is_dlin=False, fused_enabled=True,  fused_max_m=16, dlblas_enabled=True,  bf16_max_m=2048), "generic_runner"),
]

for kwargs, want in CASES:
    got = select(**kwargs)
    assert got == want, f"{kwargs} -> got={got!r} want={want!r}"
print(f"ALL PASS ({len(CASES)} routing cases)")
```

- [ ] **Step 2: Run the test to confirm it fails (selector absent)**

Run: `source "$SDK_DIR/env.sh" && python scripts/dl/test_dlin_moe_routing.py`
Expected: FAIL with `ImportError: cannot import name '_dlin_moe_select_path'`.

- [ ] **Step 3: Register the two trace env vars in `environ.py`**

Insert this new section in the `Envs` class immediately **before** the `# Scheduler: memory leak test` section comment:

```python
    # DLIN (Denglin) debug instrumentation
    # Emit per-call path/timing traces for the DLIN MoE and GDN code paths so
    # routing decisions and stage costs are observable without reading code.
    # See scripts/dl/test_dlin_moe_routing.py and docs/dl/dlin-vllm-sglang-gap-analysis.md.
    SGLANG_DEBUG_DL_MOE_TRACE = EnvBool(False)
    SGLANG_DEBUG_DL_GDN_TRACE = EnvBool(False)
```

- [ ] **Step 4: Add the selector and trace helpers to `fp8.py`**

Add near the top-level (after the existing `from sglang.srt.environ import envs` at line 19 and the module logger):

```python
# DL begin — DLIN MoE observability (SGLANG_DEBUG_DL_MOE_TRACE=1)
_DLIN_MOE_TRACE = envs.SGLANG_DEBUG_DL_MOE_TRACE.get()
_DLIN_MOE_TRACE_BUFFER: list[str] = []


def _dlin_moe_select_path(
    *,
    m: int,
    is_dlin: bool,
    fused_enabled: bool,
    fused_max_m: int,
    dlblas_enabled: bool,
    bf16_max_m: int,
) -> str:
    """Pure model of the inline DLIN MoE routing in Fp8MoEMethod.apply.

    Single source of truth for the trace path label. Keep in sync with the
    guards at the fused / bf16-bmm branches; test_dlin_moe_routing.py pins it.
    """
    if is_dlin and fused_enabled and m <= fused_max_m:
        return "fused"
    if is_dlin and dlblas_enabled and m <= bf16_max_m:
        return "bf16_bmm"
    return "generic_runner"


def _record_dlin_moe_path(path: str, m: int, topk: int, reason: str | None = None) -> None:
    if not _DLIN_MOE_TRACE:
        return
    msg = f"DL_MOE_TRACE path={path} m={m} topk={topk}"
    if reason:
        msg += f" reason={reason}"
    _DLIN_MOE_TRACE_BUFFER.append(msg)
    print(msg, flush=True)


def _get_dlin_moe_trace_summary() -> str:
    if not _DLIN_MOE_TRACE_BUFFER:
        return "DL_MOE_TRACE_SUMMARY none"
    return "DL_MOE_TRACE_SUMMARY " + " | ".join(_DLIN_MOE_TRACE_BUFFER)


# DL end
```

- [ ] **Step 5: Run the routing test to confirm it now passes**

Run: `source "$SDK_DIR/env.sh" && python scripts/dl/test_dlin_moe_routing.py`
Expected: PASS printing `ALL PASS (8 routing cases)`.

- [ ] **Step 6: Wire tracing into the four MoE execution points in `fp8.py`**

All four insertions are additive and gate on `_DLIN_MOE_TRACE` (no-op when off).

(a) Inside the fused branch, after `M`/`topk` are defined (after current line ≈1935), before the grouped GEMM:

```python
                _record_dlin_moe_path("fused", M, topk, reason=f"fused_max_m={_DL_MOE_FUSED_MAX_M}")
```

(b) Inside the bf16-bmm branch, after `M`/`topk` are defined (after current line ≈1978), before the dequant:

```python
                _record_dlin_moe_path("bf16_bmm", M, topk, reason=f"bf16_max_m={_DL_MOE_MAX_BF16_M}")
```

(c) In the exception handler (current line ≈2007), making fallback-on-error observable:

```python
        except Exception as _dl_moe_err:
            # DL begin — non-silent fallback trace
            _record_dlin_moe_path(
                "fallback_error", x.shape[0],
                dispatch_output.topk_output[1].shape[1], reason=repr(_dl_moe_err),
            )
            print(f"DL_MOE_ERR: {_dl_moe_err}", flush=True)
            # DL end
```

(d) Immediately after the try/except block ends (after the `# DL end` at current line ≈2009, before `if use_intel_xpu_backend():`), tracing the generic fall-through. On the error path this prints `fallback_error` then `generic_runner`, which is the intended non-silent behavior:

```python
        # DL begin — trace fall-through to the upstream generic runner
        _record_dlin_moe_path("generic_runner", x.shape[0], dispatch_output.topk_output[1].shape[1])
        # DL end
```

- [ ] **Step 7: Run syntax verification on both edited modules**

Run: `source "$SDK_DIR/env.sh" && python -m py_compile python/sglang/srt/environ.py python/sglang/srt/layers/quantization/fp8.py && python scripts/dl/check_dl_markers.py`
Expected: PASS with no output (and `check_dl_markers.py` reports no unmarked DL blocks).

- [ ] **Step 8: Commit**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/layers/quantization/fp8.py scripts/dl/test_dlin_moe_routing.py
git commit -m "feat(dl): add DLIN MoE routing selector and env-gated path tracing"
```

## Task 2: Surface DLIN MoE trace summaries in the in-process benchmark scripts

**Files:**
- Modify: `scripts/dl/e2e_correctness_speed.py` (has `log()` and `e.shutdown()` at line 68)
- Modify: `scripts/dl/qwen35_sg_tps.py` (uses `print`; `e.shutdown()` at line 53)
- Modify: `scripts/dl/ngram_test.py` (has `log()` and `e.shutdown()` at line 76)
- Modify: `scripts/dl/ttft_tpot.py` (has `log()` and `e.shutdown()` at line 70)

**Interfaces:**
- Consumes:
  - `SGLANG_DEBUG_DL_MOE_TRACE` (registered in Task 1)
  - `_get_dlin_moe_trace_summary()` from `fp8.py`
- Produces:
  - a `print_dl_moe_trace_summary()` helper in each script, called once right before `e.shutdown()`
  - trace env values added to each script's config banner

- [ ] **Step 1: Add the helper to each of the four scripts**

Near the other helpers (e.g. next to `log`/after imports) in each of the four files:

```python
def print_dl_moe_trace_summary() -> None:
    try:
        from sglang.srt.layers.quantization.fp8 import _get_dlin_moe_trace_summary
        print(_get_dlin_moe_trace_summary(), flush=True)
    except Exception as exc:
        print(f"DL_MOE_TRACE_SUMMARY unavailable error={exc}", flush=True)
```

- [ ] **Step 2: Call the helper right before `e.shutdown()` in each script**

- `e2e_correctness_speed.py`: insert `print_dl_moe_trace_summary()` immediately before the existing `e.shutdown()` (line 68).
- `qwen35_sg_tps.py`: insert immediately before `e.shutdown()` (line 53).
- `ngram_test.py`: insert immediately before `e.shutdown()` (line 76).
- `ttft_tpot.py`: insert immediately before `e.shutdown()` (line 70).

- [ ] **Step 3: Extend each script's config banner with trace env values**

- `e2e_correctness_speed.py` — extend the existing `log(f"[cfg] ...")` (line 31) to append:
```python
        f" moe_trace={os.environ.get('SGLANG_DEBUG_DL_MOE_TRACE', '0')}"
```
- `ngram_test.py` — extend the existing `log(f"[cfg] ...")` (line 34) with the same appended field.
- `qwen35_sg_tps.py` and `ttft_tpot.py` — add a new banner line near the top of `main()`:
```python
    print(f"[cfg] moe_trace={os.environ.get('SGLANG_DEBUG_DL_MOE_TRACE', '0')} "
          f"fused={os.environ.get('SGLANG_DL_MOE_FUSED', '0')} "
          f"fused_max_m={os.environ.get('SGLANG_DL_MOE_FUSED_MAX_M', '16')} "
          f"max_bf16_m={os.environ.get('SGLANG_DL_MOE_MAX_BF16_M', '2048')}", flush=True)
```

- [ ] **Step 4: Verify all four scripts compile and actually call the helper**

Run: `source "$SDK_DIR/env.sh" && python -m py_compile scripts/dl/e2e_correctness_speed.py scripts/dl/qwen35_sg_tps.py scripts/dl/ngram_test.py scripts/dl/ttft_tpot.py`
Expected: PASS with no output.

Run: `grep -n "print_dl_moe_trace_summary()" scripts/dl/e2e_correctness_speed.py scripts/dl/qwen35_sg_tps.py scripts/dl/ngram_test.py scripts/dl/ttft_tpot.py`
Expected: each file shows the call exactly once.

- [ ] **Step 5: Commit**

```bash
git add scripts/dl/e2e_correctness_speed.py scripts/dl/qwen35_sg_tps.py scripts/dl/ngram_test.py scripts/dl/ttft_tpot.py
git commit -m "feat(dl): surface DLIN MoE trace summaries in benchmark scripts"
```

## Task 3: Turn `scripts/dl/test_moe_dlblas.py` into a reusable correctness matrix

**Files:**
- Modify: `scripts/dl/test_moe_dlblas.py`

**Interfaces:**
- Consumes:
  - `torch.ops._dl_C.invoke_fused_moe_opt` (loaded at top of the script)
  - `moe_align_block_size` from `sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size`
- Produces:
  - `ref_moe(x, w13, sc13, w2, sc2, topk_ids, topk_weights, M, inter, hidden, top_k) -> torch.Tensor`
  - `fused_moe(x, w13, sc13, w2, sc2, topk_ids, topk_weights, M, inter, hidden, top_k) -> torch.Tensor`
  - `run_case(case_m, ...) -> float` returning mean relative error
  - a CLI matrix over `M={1,9,16}`; non-zero exit if any case exceeds `REL_ERR_LIMIT`

- [ ] **Step 1: Write the failing matrix scaffold at the bottom of the script**

Append (this drives the refactor — it will fail because `run_case` does not exist yet):

```python
CASE_MS = [1, 9, 16]
results = [(m, run_case(m)) for m in CASE_MS]
assert all(rel is not None for _, rel in results), results
```

- [ ] **Step 2: Run the script to confirm the scaffold fails**

Run: `source "$SDK_DIR/env.sh" && python scripts/dl/test_moe_dlblas.py`
Expected: FAIL with `NameError: name 'run_case' is not defined`.

- [ ] **Step 3: Refactor the straight-line script into the three functions plus the matrix loop**

Keep the existing library load (top of file), `dev`, `hidden/inter/E/top_k`, `BN/BK`, and `make_fp8_blockwise`. Replace the global `M=3` straight-line body with parameterized functions. The FP8 blockwise weights are generated once (they depend only on `E/inter/hidden`, not on `M`) and reused across cases; only `x`, `topk_ids`, `topk_weights` depend on `case_m`.

```python
REL_ERR_LIMIT = float(os.environ.get("REL_ERR_LIMIT", "0.10"))
CASE_MS = [int(s) for s in os.environ.get("CASE_MS", "1,9,16").split(",") if s.strip()]

w13, sc13 = make_fp8_blockwise((E, 2 * inter, hidden))
w2, sc2 = make_fp8_blockwise((E, hidden, inter))


def ref_moe(x, topk_ids, topk_weights, M):
    """bf16 per-token/per-expert reference."""
    out = torch.zeros(M, hidden, dtype=torch.bfloat16, device=dev)
    for t in range(M):
        for k in range(top_k):
            eid = topk_ids[t, k].item()
            sc13f = sc13[eid].repeat_interleave(BN, 0).repeat_interleave(BK, 1)
            w13_bf = (w13[eid].float() * sc13f).to(torch.bfloat16)
            gu = x[t] @ w13_bf.t()
            g, u = gu[:inter], gu[inter:]
            he = torch.nn.functional.silu(g) * u
            sc2f = sc2[eid].repeat_interleave(BN, 0).repeat_interleave(BK, 1)
            w2_bf = (w2[eid].float() * sc2f).to(torch.bfloat16)
            de = he @ w2_bf.t()
            out[t] += de * topk_weights[t, k]
    return out


def fused_moe(x, topk_ids, topk_weights, M):
    """DLIN grouped FP8 GEMM path (invoke_fused_moe_opt)."""
    G = torch.ops._dl_C.invoke_fused_moe_opt
    srt, eid_m, npp = moe_align_block_size(topk_ids, 16, E)
    c13 = torch.empty(M, top_k, 2 * inter, dtype=torch.bfloat16, device=dev)
    G(x, w13, c13, None, sc13.contiguous(), None,
      topk_weights.contiguous(), topk_ids.contiguous(), srt, eid_m, npp,
      False, top_k, 16, 128, 128, True, False, False, False, [128, 128], M)
    gate, up = c13[:, :, :inter], c13[:, :, inter:]
    he = (torch.nn.functional.silu(gate) * up).contiguous()
    c2 = torch.empty(M, top_k, hidden, dtype=torch.bfloat16, device=dev)
    G(he, w2, c2, None, sc2.contiguous(), None,
      topk_weights.contiguous(), topk_ids.contiguous(), srt, eid_m, npp,
      True, top_k, 16, 128, 128, True, False, False, False, [128, 128], M)
    return c2.sum(dim=1)


def run_case(case_m):
    torch.manual_seed(42)
    x = torch.randn(case_m, hidden, dtype=torch.bfloat16, device=dev)
    topk_ids = torch.randint(0, E, (case_m, top_k), device=dev, dtype=torch.int32)
    topk_weights = torch.rand(case_m, top_k, device=dev, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    ref = ref_moe(x, topk_ids, topk_weights, case_m)
    out = fused_moe(x, topk_ids, topk_weights, case_m)
    rel = (out.float() - ref.float()).abs().mean().item() / (ref.abs().mean().item() + 1e-9)
    print(f"[case] M={case_m} rel_err={rel:.6f}{' OK' if rel < REL_ERR_LIMIT else ' <<< OVER LIMIT'}")
    return rel


print(f"[cfg] fused_max_m={os.environ.get('SGLANG_DL_MOE_FUSED_MAX_M', '16')} "
      f"max_bf16_m={os.environ.get('SGLANG_DL_MOE_MAX_BF16_M', '2048')} "
      f"rel_err_limit={REL_ERR_LIMIT} cases={CASE_MS}")
results = [(m, run_case(m)) for m in CASE_MS]
assert all(rel < REL_ERR_LIMIT for _, rel in results), f"rel_err over limit: {results}"
print("ALL PASS")
```

> **Note:** `REL_ERR_LIMIT=0.10` compares the FP8 grouped-GEMM path against a bf16 reference, so some error is expected. If a case exceeds the limit, the script exits non-zero — investigate the op, do **not** silently widen the limit. `M=9` and `M=16` are the regimes this matrix is meant to cover (decode, NGRAM verify, short prefill); confirm they pass before relying on them.

- [ ] **Step 4: Run the correctness matrix and verify all cases pass**

Run: `source "$SDK_DIR/env.sh" && python scripts/dl/test_moe_dlblas.py`
Expected: PASS printing `[case] M=1`, `M=9`, `M=16`, then `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add scripts/dl/test_moe_dlblas.py
git commit -m "test(dl): add DLIN fused MoE correctness matrix for M={1,9,16}"
```

## Task 4: Add a fused-only MoE microbenchmark mode

**Files:**
- Modify: `scripts/dl/test_moe_dlblas.py`
- Modify: `scripts/dl/benchrun_sglang.py`

**Interfaces:**
- Consumes:
  - the refactored `ref_moe` / `fused_moe` from Task 3
- Produces:
  - `bench_fused_ms=... bench_ref_ms=... speedup=...` lines from the standalone script
  - a comment pointer in `benchrun_sglang.py`

- [ ] **Step 1: Add the failing microbench block at the end of `test_moe_dlblas.py`**

Append (drives the implementation — fails because `time_it` is undefined):

```python
bench_fused_ms = time_it(lambda: fused_moe(x_b, topk_ids_b, topk_weights_b, BENCH_M))
bench_ref_ms = time_it(lambda: ref_moe(x_b, topk_ids_b, topk_weights_b, BENCH_M))
print(f"bench_fused_ms={bench_fused_ms:.3f} bench_ref_ms={bench_ref_ms:.3f} speedup={bench_ref_ms / max(bench_fused_ms, 1e-9):.3f}")
```

- [ ] **Step 2: Run the script to confirm the microbench block fails**

Run: `source "$SDK_DIR/env.sh" && python scripts/dl/test_moe_dlblas.py`
Expected: FAIL with `NameError: name 'time_it' is not defined` (after the correctness `ALL PASS`).

- [ ] **Step 3: Implement the timed loop**

Insert before the microbench block:

```python
BENCH_ITERS = int(os.environ.get("BENCH_ITERS", "20"))
BENCH_M = int(os.environ.get("BENCH_M", "9"))


def time_it(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(BENCH_ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / BENCH_ITERS


torch.manual_seed(7)
x_b = torch.randn(BENCH_M, hidden, dtype=torch.bfloat16, device=dev)
topk_ids_b = torch.randint(0, E, (BENCH_M, top_k), device=dev, dtype=torch.int32)
topk_weights_b = torch.rand(BENCH_M, top_k, device=dev, dtype=torch.float32)
topk_weights_b = topk_weights_b / topk_weights_b.sum(dim=-1, keepdim=True)
```

(`time` is already imported at the top of the script.)

- [ ] **Step 4: Add the pointer comment in `benchrun_sglang.py`**

Near the top-of-file docstring/`DLIN_SERVER_ENV` definition (≈line 42), add:

```python
# MoE-only throughput should be checked with scripts/dl/test_moe_dlblas.py;
# benchrun_sglang.py is an end-to-end serving harness, not a MoE microbench.
```

- [ ] **Step 5: Run the script and verify microbench metrics print**

Run: `source "$SDK_DIR/env.sh" && python scripts/dl/test_moe_dlblas.py`
Expected: PASS — correctness `ALL PASS`, then `bench_fused_ms=... bench_ref_ms=... speedup=...`.

- [ ] **Step 6: Commit**

```bash
git add scripts/dl/test_moe_dlblas.py scripts/dl/benchrun_sglang.py
git commit -m "perf(dl): add standalone fused MoE microbench metrics"
```

## Task 5: Wire path-trace visibility into the serving benchmark harness

**Files:**
- Modify: `scripts/dl/benchrun_sglang.py`

**Interfaces:**
- Consumes:
  - the existing `DLIN_SERVER_ENV` dict (≈line 42), which currently holds `SGLANG_DL_MOE_MAX_BF16_M`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`
- Produces:
  - one added key `SGLANG_DEBUG_DL_MOE_TRACE` propagated into the server environment
  - a config log line showing MoE env state

- [ ] **Step 1: Add a failing source-level check for the new key**

Run: `source "$SDK_DIR/env.sh" && grep -n '"SGLANG_DEBUG_DL_MOE_TRACE"' scripts/dl/benchrun_sglang.py`
Expected: no match (the key is not yet present).

- [ ] **Step 2: Add the trace key to `DLIN_SERVER_ENV` (do NOT replace the dict)**

The `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` keys are required for DLIN offline serving — keep them. Add the trace key alongside the existing entries:

```python
DLIN_SERVER_ENV = {
    # bf16-bmm MoE path (commit a2048d00fc; fp8.py:1962). Robust default — only needs
    # gptq_dlblas_gemmex (FP8 linear), not the fused-MoE op. Gives ~13-17 tok/s.
    "SGLANG_DL_MOE_MAX_BF16_M": "128",
    # Offline tokenizer/model load (DLIN hosts have no HF network; avoids httpx
    # "client has been closed" in the bench client's tokenizer load). See run_sglang.sh phase_bench.
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    # DLIN MoE path tracing (opt-in); forwarded to the server env below.
    "SGLANG_DEBUG_DL_MOE_TRACE": os.environ.get("SGLANG_DEBUG_DL_MOE_TRACE", "0"),
}
```

- [ ] **Step 3: Add a config log line where `DLIN_SERVER_ENV` is applied (≈line 466)**

Right after the loop that copies `DLIN_SERVER_ENV` into the server environment, add:

```python
    print(
        f"[DLIN] moe_trace={env.get('SGLANG_DEBUG_DL_MOE_TRACE', '0')} "
        f"fused={env.get('SGLANG_DL_MOE_FUSED', '0')} "
        f"fused_max_m={env.get('SGLANG_DL_MOE_FUSED_MAX_M', '16')} "
        f"max_bf16_m={env.get('SGLANG_DL_MOE_MAX_BF16_M', '128')}",
        flush=True,
    )
    # Per-case server/client logs may include DL_MOE_TRACE / DL_MOE_TRACE_SUMMARY
    # lines when SGLANG_DEBUG_DL_MOE_TRACE=1; use them to confirm fused vs fallback.
```

(Use whatever local variable name the surrounding code uses for the built env dict — it is `env` in the existing apply loop; adjust only if that name differs.)

- [ ] **Step 4: Verify compile and that the key is present without dropping offline keys**

Run: `source "$SDK_DIR/env.sh" && python -m py_compile scripts/dl/benchrun_sglang.py`
Expected: PASS with no output.

Run: `grep -nE '"(SGLANG_DEBUG_DL_MOE_TRACE|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE|SGLANG_DL_MOE_MAX_BF16_M)"' scripts/dl/benchrun_sglang.py`
Expected: all four keys present in `DLIN_SERVER_ENV`.

- [ ] **Step 5: Commit**

```bash
git add scripts/dl/benchrun_sglang.py
git commit -m "feat(dl): expose DLIN MoE tracing in the serving harness"
```

## Task 6: Add GDN backend stage timing on `GDNKernelDispatcher`

**Files:**
- Modify: `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- Modify: `scripts/dl/ttft_tpot.py`

**Interfaces:**
- Consumes:
  - `envs.SGLANG_DEBUG_DL_GDN_TRACE` (registered in Task 1)
  - the existing `GDNKernelDispatcher.decode/extend/target_verify/packed_decode` methods
- Produces:
  - `_record_dl_gdn_trace(stage: str, start_time: float) -> None`
  - `_get_dl_gdn_trace_summary() -> str`
  - `DL_GDN_TRACE stage=<name> ms=<value>` lines for `decode`, `extend`, `target_verify`, `packed_decode`

- [ ] **Step 1: Write the failing probe in `ttft_tpot.py`**

Temporarily add near the top of `main()`:

```python
    from sglang.srt.layers.attention.linear.gdn_backend import _get_dl_gdn_trace_summary
    log(_get_dl_gdn_trace_summary())
```

- [ ] **Step 2: Run an import check to confirm the helper is absent**

Run: `source "$SDK_DIR/env.sh" && python -c "from sglang.srt.layers.attention.linear import gdn_backend as g; assert hasattr(g, '_get_dl_gdn_trace_summary')"`
Expected: FAIL with `AssertionError`.

- [ ] **Step 3: Add the imports and timing helpers to `gdn_backend.py`**

Add `import time` and `from sglang.srt.environ import envs` to the import block at the top (currently neither is imported). Then, after the imports (after line ≈55), add:

```python
# DL begin — DLIN GDN stage timing (SGLANG_DEBUG_DL_GDN_TRACE=1)
_DL_GDN_TRACE = envs.SGLANG_DEBUG_DL_GDN_TRACE.get()
_DL_GDN_TRACE_BUFFER: list[str] = []


def _record_dl_gdn_trace(stage: str, start_time: float) -> None:
    if not _DL_GDN_TRACE:
        return
    ms = (time.time() - start_time) * 1000.0
    msg = f"DL_GDN_TRACE stage={stage} ms={ms:.3f}"
    _DL_GDN_TRACE_BUFFER.append(msg)
    print(msg, flush=True)


def _get_dl_gdn_trace_summary() -> str:
    if not _DL_GDN_TRACE_BUFFER:
        return "DL_GDN_TRACE_SUMMARY none"
    return "DL_GDN_TRACE_SUMMARY " + " | ".join(_DL_GDN_TRACE_BUFFER)


# DL end
```

- [ ] **Step 4: Wrap the four dispatcher methods with timing capture**

Each method currently does a bare `return self.<x>_kernel.<method>(…)`. Capture `t0` before the call, record after, and return the result. Apply to all four:

`decode` (line 185):
```python
    def decode(self, q, k, v, a, b, *, A_log, dt_bias, ssm_states, cache_indices, query_start_loc, **kwargs) -> torch.Tensor:
        _t0 = time.time()
        out = self.decode_kernel.decode(
            q, k, v, a, b,
            A_log=A_log, dt_bias=dt_bias, ssm_states=ssm_states,
            cache_indices=cache_indices, query_start_loc=query_start_loc, **kwargs,
        )
        _record_dl_gdn_trace("decode", _t0)
        return out
```

`extend` (line 214) — same shape, stage `"extend"`, wrapping `self.extend_kernel.extend(…)`:
```python
    def extend(self, q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, **kwargs) -> tuple:
        _t0 = time.time()
        out = self.extend_kernel.extend(
            q, k, v, g, beta,
            ssm_states=ssm_states, cache_indices=cache_indices, query_start_loc=query_start_loc, **kwargs,
        )
        _record_dl_gdn_trace("extend", _t0)
        return out
```

`target_verify` (line 239) — stage `"target_verify"`, wrapping `self.verify_kernel.target_verify(…)`:
```python
    def target_verify(self, A_log, dt_bias, q, k, v, a, b, *, ssm_states, cache_indices, query_start_loc, **kwargs) -> torch.Tensor:
        _t0 = time.time()
        out = self.verify_kernel.target_verify(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            ssm_states=ssm_states, cache_indices=cache_indices, query_start_loc=query_start_loc, **kwargs,
        )
        _record_dl_gdn_trace("target_verify", _t0)
        return out
```

`packed_decode` (line 152) — stage `"packed_decode"`; preserve the early `return None` for unsupported kernels, wrapping only the real call:
```python
    def packed_decode(self, mixed_qkv, a, b, *, A_log, dt_bias, scale, ssm_states, cache_indices, num_v_heads, head_v_dim, **kwargs) -> Optional[torch.Tensor]:
        if not self.supports_packed_decode:
            return None
        _t0 = time.time()
        out = self.decode_kernel.packed_decode(
            mixed_qkv, a, b,
            A_log=A_log, dt_bias=dt_bias, scale=scale, ssm_states=ssm_states,
            cache_indices=cache_indices, num_v_heads=num_v_heads, head_v_dim=head_v_dim, **kwargs,
        )
        _record_dl_gdn_trace("packed_decode", _t0)
        return out
```

- [ ] **Step 5: Print the GDN trace summary from `ttft_tpot.py`**

Replace the temporary probe from Step 1 with the real call placed right before `e.shutdown()` (line 70):

```python
    try:
        from sglang.srt.layers.attention.linear.gdn_backend import _get_dl_gdn_trace_summary
        log(_get_dl_gdn_trace_summary())
    except Exception as exc:
        log(f"DL_GDN_TRACE_SUMMARY unavailable error={exc}")
```

- [ ] **Step 6: Verify and commit**

Run: `source "$SDK_DIR/env.sh" && python -c "from sglang.srt.layers.attention.linear import gdn_backend as g; assert hasattr(g, '_get_dl_gdn_trace_summary')" && python -m py_compile python/sglang/srt/layers/attention/linear/gdn_backend.py scripts/dl/ttft_tpot.py && python scripts/dl/check_dl_markers.py`
Expected: PASS with no output.

```bash
git add python/sglang/srt/layers/attention/linear/gdn_backend.py scripts/dl/ttft_tpot.py
git commit -m "feat(dl): add GDN backend stage timing traces"
```

## Task 7: Update the DLIN gap-analysis document with implementation checkpoints

**Files:**
- Modify: `docs/dl/dlin-vllm-sglang-gap-analysis.md`

**Interfaces:**
- Consumes:
  - the current gap-analysis document
  - the MoE observability (Tasks 1–5) and GDN timing (Task 6) work
- Produces:
  - an "DLIN implementation checkpoints" section

- [ ] **Step 1: Append a checkpoint section at the end of the document**

```markdown
## DLIN implementation checkpoints

- Fused MoE path tracing (env `SGLANG_DEBUG_DL_MOE_TRACE=1`) distinguishes `fused`, `bf16_bmm`, `generic_runner`, and `fallback_error` runs. The label comes from a single pure selector, `_dlin_moe_select_path`, pinned by `scripts/dl/test_dlin_moe_routing.py`.
- `scripts/dl/test_moe_dlblas.py` covers `M={1,9,16}` correctness (FP8 fused vs bf16 reference) and prints standalone fused-vs-reference microbench metrics.
- Serving/decode/e2e/NGRAM/TTFT harnesses surface `DL_MOE_TRACE_SUMMARY` so routing is visible in benchmark artifacts.
- GDN stage timing (env `SGLANG_DEBUG_DL_GDN_TRACE=1`) emits per-call `decode`/`extend`/`target_verify`/`packed_decode` ms on `GDNKernelDispatcher`, complementing the existing kernel-class dispatch log.
- No DLIN-native GDN kernel exists yet; kernel replacement is the next stage once timing identifies the bottleneck.
```

- [ ] **Step 2: Verify the document has no leftover placeholders**

Run: `! grep -nE "TODO|TBD|PLACEHOLDER" docs/dl/dlin-vllm-sglang-gap-analysis.md`
Expected: PASS with no output.

- [ ] **Step 3: Commit**

```bash
git add docs/dl/dlin-vllm-sglang-gap-analysis.md
git commit -m "docs(dl): update DLIN fused MoE and GDN checkpoints"
```

## Task 8: Run the verification matrix and record non-regression evidence

**Files:**
- Test: `scripts/dl/test_dlin_moe_routing.py`
- Test: `scripts/dl/test_moe_dlblas.py`
- Test: `scripts/dl/e2e_correctness_speed.py`
- Test: `scripts/dl/qwen35_sg_tps.py`
- Test: `scripts/dl/ngram_test.py`
- Test: `scripts/dl/ttft_tpot.py`
- Test: `scripts/dl/benchrun_sglang.py`

**Interfaces:**
- Consumes:
  - all prior code changes and the throughput guardrail band
- Produces:
  - a verification log: routing unit test, MoE correctness + microbench, decode/e2e/NGRAM throughput non-regression, real-engine routing confirmation, and GDN timing visibility

> **Pass criteria are throughput-only.** Do not assert output coherence for NGRAM/MTP runs (spec-verify prompt-regen bug). The e2e (non-spec) run keeps a light coherence sanity check only.

- [ ] **Step 1: Run the GPU-free routing unit test**

Run: `source "$SDK_DIR/env.sh" && python scripts/dl/test_dlin_moe_routing.py`
Expected: `ALL PASS (8 routing cases)`.

- [ ] **Step 2: Run the fused-MoE correctness matrix + microbench**

Run: `source "$SDK_DIR/env.sh" && python scripts/dl/test_moe_dlblas.py`
Expected: `[case] M=1/9/16` all under `REL_ERR_LIMIT`, `ALL PASS`, then `bench_fused_ms=… speedup=…`.

- [ ] **Step 3: Confirm real-engine routing matches the selector (decode → fused)**

Run: `source "$SDK_DIR/env.sh" && SGLANG_DEBUG_DL_MOE_TRACE=1 SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=16 python scripts/dl/qwen35_sg_tps.py`
Expected: a `DL_MOE_TRACE path=fused …` line appears (decode M=1 ≤ 16, fused on), plus `DL_MOE_TRACE_SUMMARY …`. This confirms the real path label matches `_dlin_moe_select_path(m=1, fused_enabled=True, fused_max_m=16) → "fused"`.

- [ ] **Step 4: Steady-state decode throughput non-regression**

Run: `source "$SDK_DIR/env.sh" && SGLANG_DEBUG_DL_MOE_TRACE=1 python scripts/dl/e2e_correctness_speed.py`
Expected: `[benchN] … tok/s` in the baseline band (≈18 tok/s decode, ≈16 tok/s short e2e), coherent `[OUT-START]…[OUT-END]` text for this non-spec run, and a `DL_MOE_TRACE_SUMMARY …` line.

- [ ] **Step 5: NGRAM throughput non-regression (throughput-only, no coherence assertion)**

Run: `source "$SDK_DIR/env.sh" && SGLANG_DEBUG_DL_MOE_TRACE=1 python scripts/dl/ngram_test.py`
Expected: `[benchN] … tok/s` in the `35–40 tok/s` band and a `DL_MOE_TRACE_SUMMARY …` line. Do **not** fail the step on output text quality.

- [ ] **Step 6: GDN stage timing visibility**

Run: `source "$SDK_DIR/env.sh" && SGLANG_DEBUG_DL_MOE_TRACE=1 SGLANG_DEBUG_DL_GDN_TRACE=1 python scripts/dl/ttft_tpot.py`
Expected: TTFT/TPOT lines, a `DL_MOE_TRACE_SUMMARY …` line, and a `DL_GDN_TRACE_SUMMARY …` line (with `stage=decode`/`extend`/`target_verify` entries).

- [ ] **Step 7: Serving-harness trace wiring sanity**

Run: `source "$SDK_DIR/env.sh" && SGLANG_DEBUG_DL_MOE_TRACE=1 python scripts/dl/benchrun_sglang.py` (write/use a `config_serving.json` pointing at the Qwen3.5-35B-A3B-FP8 model)
Expected: the `[DLIN] moe_trace=1 …` config line prints and the run does not regress from a syntax/runtime error. A full serving pass is environment-dependent; a clean startup + trace config line is the minimum bar.

- [ ] **Step 8: Commit verification-only fixes if any were required**

```bash
git status --short
# Only if verification found and fixed a script/runtime issue:
git add <fixed-files>
git commit -m "fix(dl): address verification regressions"
# Otherwise, do not create an extra commit.
```

## Self-Review

- **Spec coverage:**
  - Fused MoE observability (path tracing + summary in harnesses): Tasks 1, 2, 5.
  - Deterministic routing validation: Task 1 (selector + GPU-free unit test), confirmed on the real engine in Task 8 Step 3.
  - Correctness matrix for `M={1,9,16}`: Task 3.
  - Fused-only microbenchmark: Task 4.
  - Non-silent fallback: Task 1 Step 6(c)–(d) (`fallback_error` then `generic_runner`).
  - GDN stage timing: Task 6 (no speculative native-kernel scaffolding — deferred by design).
  - Qwen throughput non-regression: Task 8 (throughput-only; quality caveat documented in Global Constraints).
  - Documentation: Task 7.
- **Placeholder scan:** No `TODO`, `TBD`, or "similar to task N" in executable steps. The one `> **Note:**` in Task 3 is guidance, not a deferred step.
- **Type consistency:** `_dlin_moe_select_path`, `_record_dlin_moe_path`, `_get_dlin_moe_trace_summary`, `_record_dl_gdn_trace`, `_get_dl_gdn_trace_summary`, `run_case`, `ref_moe`, `fused_moe`, `time_it`, `print_dl_moe_trace_summary` are used consistently across tasks. Env var names are `SGLANG_DEBUG_DL_MOE_TRACE` / `SGLANG_DEBUG_DL_GDN_TRACE` everywhere (registered in Task 1, consumed in Tasks 2/5/6/8).
- **Env-var convention:** both new vars are `EnvBool(False)` in `Envs`, accessed via `envs.NAME.get()`, named with the `DEBUG_` verb — compliant with the `env-var-conventions` skill.

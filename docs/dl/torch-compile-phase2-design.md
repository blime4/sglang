---
title: "torch.compile Phase II — Full DLIN Support (Design Spec)"
date: 2026-07-17
status: design — awaiting review
model: Qwen3.5-35B-A3B-FP8
hardware: Denglin DLIN KS38 (×4, TP4)
baseline: vLLM 0.21.1 (24.0 ms/token decode TPOT)
references:
  - docs/dl/blog-sglang-dlin-qwen35-35b-tp4.md  (§8 the fix path)
  - docs/dl/sglang-vs-vllm-phase2-checklist.md   (C-track C-1…C-7)
  - docs/dl/sglang-vs-vllm-tp4-status-20260716.md (appendix B: Phase 1 landed)
  - docs/dl/sglang-vs-vllm-perf-gap.md            (§7.32/7.33/7.34 capture-state history)
---

# torch.compile Phase II — Full DLIN Support

> **One-line goal.** Close the 3.3 ms TP4 decode gap to vLLM (27.3 → 23–24 ms)
> by giving SGLang's `torch.compile` the same Inductor IR-level fusions
> (`norm_quant` / `act_quant`) vLLM uses on DLIN — *and* pursue the deeper
> `use_moe_cu` MoE win — **entirely on the SGLang side, no DLIN kernel-team
> dependency.** Phase I (compile no longer crashes) is landed; this is Phase II.

## 0. Goal & success criteria

| | Criterion | Target |
|---|---|---|
| G1 | Decode compile path runs stable on DLIN (fullgraph, correct output, no crash) | no `Graph break` warnings; greedy output token-for-token matches eager baseline |
| G2 | `norm_quant` + `act_quant` fusions active in the compiled decode graph | compile log shows fusion hits; ≥1 fused kernel replaces the standalone RMSNorm+quant pair |
| G3 | TP4 decode TPOT closes the gap | GPU forward 20.7 → ~18 ms; wall TPOT 23–24 ms (≤ vLLM 24.0 ms) |
| G4 | `use_moe_cu` deep MoE win (gated) | decode-capture accepts `use_moe_cu` under compiled fullgraph, OR a documented fallback decision |

**Constraints (user-confirmed).**
- Solution boundary: **pure SGLang side** — no request to the DLIN kernel team
  (no `kUsePDL` one-line fix request, no standalone fused-kernel ask). Reusing the
  already-dlopened vLLM `_dl_C.so` / `_C.so` binaries is in-bounds (same dependency
  SGLang already has).
- Win depth: **both** the 3.3 ms fusion gap **and** the `use_moe_cu` deep MoE win.
- Execution: **staged** (low-risk gap first; `use_moe_cu` behind a verification gate).

## 1. Context — the Phase-II picture, audited against code

Phase I (landed 2026-07-16, appendix B of the status doc) made
`enable_torch_compile=True` compile, capture a CUDA graph, and produce correct
output on DLIN. Four blockers were fixed: GDN conv `USE_GDC` signature,
`gdc_wait`/`gdc_launch_dependents` Triton stubs, `**pdl_kwargs`→explicit
`USE_GDC`/`USE_PDL` across GDN/Mamba/elementwise/FP8 kernels, and FakeTensor
meta impls for the `_dl_C` ops (`dl_compile_meta.py`). It still requires
`TORCHINDUCTOR_COMPILE_THREADS=1`.

The blog's §8.2 lists **three causes** for the compiled graph being slow (81 ms).
Exploration against the live code corrects them:

| Blog §8.2 cause | Code-truth verdict |
|---|---|
| ① compile↔CG not wired (CG captures eager) | **Partly false.** Decode's `FullCudaGraphBackend` already captures the **compiled** `model.forward` (`patch_model` wraps `torch.compile`, `torch_compile_decoration.py:56`); prefill is fully wired via `CUDAPiecewiseBackend` capturing inductor-compiled `entry.runnable` (`cuda_piecewise_backend.py:179`). The 81 ms is **not** "captured eager" — it is the compiled graph emitting **unfused generic kernels** (slower than DLIN eager). |
| ② Missing `norm_quant`/`act_quant` fusion passes | **True, and central.** `PostGradPassManager.passes` is empty (`pass_manager.py:36`); no `compilation/passes/fusion/`; `pass_config` is an empty dict. vLLM's `RMSNormQuantFusionPass` / `ActivationQuantFusionPass` sit unported in `vllm-new-overlay/vllm/compilation/passes/fusion/`. |
| ③ Residual graph breaks from `_dl_C` ops lacking fakes | **False.** All 9 `_dl_C` ops have `register_fake` (`dl_compile_meta.py:46-85`); every called op is covered. The real residual break is **flash_attn ops** (`_vllm_fa2_C.varlen_fwd`, `_vllm_fa3_C.fwd`) — not registered (checklist C-2). |

**Extra finding (decisive for feasibility):** the fused kernels the vLLM passes
dispatch to **already exist, DLIN-compiled, in `vllm-new-overlay/vllm/_C.so`**
(`nm -D` confirms `rms_norm_static_fp8_quant`, `rms_norm_dynamic_per_token_quant`,
`rms_norm_per_block_quant`, `fused_add_rms_norm_static_fp8_quant`,
`silu_and_mul_quant`, `silu_and_mul_per_block_quant`). `_dl_C.so` has
`w8a8_matmul` + `invoke_fused_moe_opt` but not the norm/act-quant fusions.

**Decode-pass wiring gap (the Stage-1 crux):** decode's `patch_model` calls plain
`torch.compile(..., mode="max-autotune-no-cudagraphs")` with **no backend**
(`torch_compile_decoration.py:56`), and `set_torch_compile_config()` flips
inductor flags but **never sets `post_grad_custom_post_pass`**. So even after the
fusion passes are ported into `PostGradPassManager`, decode's compile would not
apply them. vLLM's equivalent (`VllmBackend`) does
`inductor_config["post_grad_custom_post_pass"] = pass_manager`
(`vllm-new-overlay/vllm/compilation/backends.py:966`). Stage 1 must add the
analogous injection.

## 2. Key feasibility facts (from exploration)

1. **Fused norm/act+quant kernels exist** in the DLIN-compiled vLLM `_C.so` →
   Stage 1 needs no new kernel work, only pass port + op repointing. (Confirm
   whether `sgl-kernel` already exposes equivalents; if so, prefer those to
   avoid loading a second vLLM `.so`.)
2. **`w8a8_matmul` exists** in `_dl_C.so` and already has a fake registered
   (`dl_compile_meta.py:49`) — SGLang never calls it. C-5 can adopt it directly.
3. **`PostGradPassManager` is vLLM-derived** and already supports the
   `post_grad_custom_post_pass` injection point — the wiring target exists.
4. **vLLM has zero PDL/capture-mode special handling.** It accepts the PDL
   `use_moe_cu` path purely as a side-effect of capturing the **compiled** graph
   (`invoke_fused_moe_opt` is an opaque node). The §7.x "compiled forward also
   crashes" conclusion predates Phase I (compile was broken then). So Stage 2's
   question — *"does SGLang's compiled fullgraph decode-capture now accept
   `use_moe_cu`?"* — is a **freshly answerable experiment**, not a known block.

## 3. Design overview — staged

```
Stage 0  compile foundation (clean fullgraph)          [S–M]  no perf gain, unlocks
   │      C-1 sitecustomize shim · C-2 flash_attn fakes · C-3 fullgraph audit
   ▼      exit: compiled decode ≈ eager 27 ms, no graph breaks
Stage 1  close the 3.3 ms fusion gap  ◀── core goal   [L]
   │      C-5 FP8 restructure (norm→quant→w8a8_matmul)
   │      port RMSNormQuantFusionPass + ActivationQuantFusionPass
   │      inject PostGradPassManager into decode compile
   ▼      exit: GPU fwd ~18 ms, wall TPOT 23–24 ms (≤ vLLM)
Stage 2  use_moe_cu deep MoE win (gated experiment)   [M–L]
          gate: does compiled fullgraph decode-capture accept use_moe_cu?
            yes → enable, measure MoE segment < 7.8 ms parity
            no  → decode tc_piecewise + lift multimodal guard (§7.33), OR
                  accept MoE parity and document
```

Every stage is independently verifiable and leaves the default serving tree
untouched (env-gated, `# DL begin/end` marked).

## 4. Stage 0 — compile foundation

**Goal:** a clean, fullgraph compiled decode path (no graph breaks), so that
Stage 1's fusions apply to a single continuous graph.

- **C-1 — inductor subprocess shim.** Move the `gdc_wait`/`gdc_launch_dependents`
  `@triton.jit` no-op stubs (currently `python/sglang/__init__.py:29-53` + dup in
  `jit_kernel/utils.py:3-27`, main-process-only) into a `sitecustomize.py` on the
  venv `site-packages` path, so inductor worker subprocesses inherit them. *Exit
  criterion:* `enable_torch_compile=True` compiles **without**
  `TORCHINDUCTOR_COMPILE_THREADS=1`.
- **C-2 — flash_attn FakeTensor.** Register `register_fake` for
  `torch.ops._vllm_fa2_C.varlen_fwd` and `torch.ops._vllm_fa3_C.fwd` (+`fwd_kvcache`),
  returning `q`-shape + `softmax_lse`. Mirror the `_dl_C` style in
  `dl_compile_meta.py`. *Exit criterion:* `TORCH_LOGS=graph_breaks` shows the
  flash_attn breaks gone.
- **C-3 — fullgraph audit.** With C-1/C-2 done, run `TORCH_LOGS=graph_breaks`
  and eliminate remaining breaks (Python control flow, data-dependent shapes,
  any other un-faked op) until the log is clean. *Exit criterion:* no `Graph break`
  lines; compiled decode TPOT ≈ 27 ms (DLIN ops opaque-but-unfused ≈ eager).

**Files:** `python/sglang/__init__.py`, `python/sglang/srt/layers/quantization/dl_compile_meta.py`
(+ new `sitecustomize.py` in the venv, documented in the serving recipe).

## 5. Stage 1 — close the 3.3 ms fusion gap (core goal)

**Goal:** `norm_quant` + `act_quant` IR fusions active in the **decode** compiled
graph → GPU forward 20.7 → ~18 ms, wall TPOT 23–24 ms.

### 5.1 C-5 — FP8 linear-path restructure (structural prerequisite)

The `norm_quant` pattern matches `RMSNorm → per-token FP8 quant` feeding a linear.
SGLang's current path (`gptq_dlblas_gemmex` / `dlblas_w8a8_block_fp8_linear`,
`fp8_utils.py:559,575`) takes **bf16 in and quantizes internally**, so there is no
standalone quant node for the pattern to latch onto.

- Restructure the FP8 linear path on DLIN to
  `gemma_rms_norm → per_token_quant(fp8) → w8a8_matmul(fp8 input)`, reusing the
  existing `_dl_C::w8a8_matmul` (fake already registered).
- Keep `gptq_dlblas_gemmex` available as an env-gated fallback (it is the path
  that survived the `.contiguous()` / quant_type correctness bugs — do not delete it).

**Correctness guard:** this touches the path that produced the `quant_type` /
`.contiguous()` / `is_neox_style` quality bugs. Gate the new path behind a
`SGLANG_DL_*` env flag and require a greedy token-for-token regression against
the current serving recipe before it becomes default.

### 5.2 Port the fusion passes

- Port `RMSNormQuantFusionPass` (`vllm-new-overlay/.../rms_quant_fusion.py`) and
  `ActivationQuantFusionPass` (`.../act_quant_fusion.py`) into a new
  `python/sglang/srt/compilation/passes/fusion/` package, adapted to SGLang's
  `SGLangInductorPass` base (`inductor_pass.py`) and `PostGradPassManager.add()`.
- **Repoint `FUSED_OPS`** to the op source chosen in §5.4 (vLLM `_C.so` ops after
  `load_library`, or `sgl-kernel` equivalents).
- Register them in `PostGradPassManager.configure()` gated by a `pass_config`
  flag (mirror vLLM `pass_manager.py:160-170`).

### 5.3 Wire the passes into the decode compile (the crux)

Decode's `patch_model` uses backend-less `torch.compile`; fusion passes added to
`PostGradPassManager` would not run. Fix by mirroring vLLM's injection — in
`set_torch_compile_config()` (`torch_compile_decoration.py:71`):

```python
pm = PostGradPassManager()
pm.configure()                       # fix_functionalization
if fuse_norm_quant: pm.add(RMSNormQuantFusionPass(config))
if fuse_act_quant:  pm.add(ActivationQuantFusionPass(config))
torch._inductor.config.post_grad_custom_post_pass = pm
```

`PostGradPassManager.__call__` is then invoked by Inductor on the post-grad graph
(`pass_manager.py:38`), applying the fusions before graph capture.

### 5.4 Fused-kernel sourcing (decision point)

The ported passes dispatch to `torch.ops._C.rms_norm_*_quant` / `silu_and_mul_*_quant`.
Two pure-SGLang-side options:

- **(a) Reuse vLLM `_C.so`.** SGLang already `dlopen`s vLLM's `_dl_C.so`
  (`fp8_utils._ensure_dl_C`); extend to `load_library` the matching `_C.so` so
  `torch.ops._C.*` resolves. Smallest change; same binary vLLM uses on DLIN.
- **(b) Use `sgl-kernel` equivalents** if they already expose fused norm+quant /
  act+quant (confirm by `nm` on `sgl-kernel`'s `.so`). Keeps the dependency
  surface to SGLang's own kernel package.

**Recommendation:** try (b) first (no new binary dependency); fall back to (a).
Decide during Stage-1 implementation based on the `nm` result.

**Exit criteria (G1/G2/G3):** compile log shows fusion hits; GPU forward ~18 ms
(`SGLANG_DL_TIME_REPLAY`); `compare_tp4.py` wall TPOT ≤ 24 ms; greedy regression passes.

## 6. Stage 2 — `use_moe_cu` deep MoE win (gated experiment)

**Goal:** if the compiled fullgraph decode-capture now accepts `use_moe_cu`
(the vLLM ~18 ms-forward fast MoE path), enable it for an additional MoE gain
below the 7.8 ms parity point.

### 6.1 Gate experiment (run first, decide before any arch change)

After Stage 0+1, under compiled fullgraph + capture, set `SGLANG_DL_MOE_VLLM=1`
(`use_moe_cu`) and observe:

- **If accepted** (no `Device page fault` / `cudaErrorInvalidAddressSpace`) →
  measure MoE segment via `SGLANG_DL_TIME_REPLAY` + `SGLANG_DL_SKIP_MOE`
  differential; if below 7.8 ms, keep it. Done.
- **If rejected** → two sub-options, **do not commit to either until the gate
  is run**:
  - (i) **decode `tc_piecewise`** (currently prefill-only, `decode_cuda_graph_runner.py:22`;
    auto-disabled for the multimodal `Qwen3_5MoeForConditionalGeneration`, §7.33):
    implement it so the MoE split-op runs eager (out-of-graph), sidestepping the
    PDL-in-capture problem. Large runner-architecture change.
  - (ii) **accept MoE parity**, document the residual, ship Stage 1's win.

**Rationale for gating:** the §7.x "compiled capture also crashes" evidence is
pre-Phase-I and may be stale; agent-2 found vLLM needs no PDL handling beyond
compiled capture. Spending zero arch-change effort before confirming the block
is the disciplined move.

**Exit criteria (G4):** a measured decision — either `use_moe_cu` enabled with a
quantified MoE win, or a documented fallback with the gate-experiment evidence.

## 7. Data flow — how a fusion applies end-to-end (Stage 1)

```
model.forward (DLIN: gemma_rms_norm → per_token_quant → w8a8_matmul  [after C-5]
   │
   │ torch.compile(mode="max-autotune-no-cudagraphs")   [patch_model]
   ▼
dynamo trace → AOTAutograd post-grad graph
   │
   │ Inductor: post_grad_custom_post_pass = PostGradPassManager  [§5.3 injection]
   ▼
RMSNormQuantFusionPass:  rms_norm + per_token_quant  ──►  _C.rms_norm_*_quant  (1 kernel)
ActivationQuantFusionPass: silu_and_mul + quant      ──►  _C.silu_and_mul_*_quant
   │
   ▼
Inductor lean+fused graph  ──►  FullCudaGraphBackend captures & replays it
   │  (existing wiring; no tc_piecewise needed for the fusion win)
   ▼
decode step: fewer kernel launches, one fewer memory pass on norm/quant
```

The compile↔CG wiring already exists for decode (`FullCudaGraphBackend` captures
the compiled forward); Stage 1 only changes *what* that compiled forward
contains (fused vs generic).

## 8. Correctness & regression strategy

- **Golden output:** record a greedy decode (fixed seed, "Hi" + 128 tokens) on
  the **current serving recipe** (§9 of the blog) as the reference. Every stage's
  exit requires token-for-token match.
- **Byte-diff harness:** for the C-5 FP8 restructure, reuse the tensor-dump
  technique from blog §3.4 — diff the `w8a8_matmul` inputs against the
  `gptq_dlblas_gemmex` path to prove numerical equivalence before trusting
  output equality.
- **Env-gated rollout:** every change default-off (`SGLANG_DL_*` or
  `pass_config` flag); the validated serving recipe (blog §9) stays the default
  until a stage passes its exit criteria.

## 9. Verification plan (per stage)

| Stage | Command | Pass when |
|---|---|---|
| 0 | `enable_torch_compile=True` w/o `TORCHINDUCTOR_COMPILE_THREADS=1`; `TORCH_LOGS=graph_breaks` | compiles clean; no graph breaks; TPOT ≈ 27 ms |
| 1 | `scripts/dl/compare_tp4.py` (128 tok best-of-3) + 512-tok decode-only + greedy diff | fusion hits in log; GPU fwd ~18 ms; wall TPOT ≤ 24 ms; greedy matches |
| 2 | `SGLANG_DL_MOE_VLLM=1` under compiled fullgraph; `SGLANG_DL_TIME_REPLAY`+`SKIP_MOE` | gate decision documented; if enabled, MoE < 7.8 ms and greedy matches |

Diagnostics already present: `SGLANG_DL_TIME_REPLAY`, `SGLANG_DL_SKIP_MOE/ATTN/SHARED`,
`SGLANG_DL_LAYER_TIMING`, `SGLANG_DL_PHASE_TIME`.

## 10. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| C-5 FP8 restructure regresses correctness (it is the path of the historic quality bugs) | Med | env-gated; byte-diff harness; golden greedy regression before default |
| Ported passes don't fire because decode compile still bypasses `post_grad_custom_post_pass` | Med | §5.3 injection is the explicit fix; verify fusion hits in log before trusting TPOT |
| `_C.*` fused kernels slower than hoped on DLIN (generic vLLM CUDA, not DLIN-tuned) | Low–Med | vLLM already uses them at ~18 ms forward; measure early in Stage 1 |
| `use_moe_cu` still crashes under compiled capture (Stage 2 gate fails) | Med | gated — no arch-change spent until confirmed; fallback (ii) keeps Stage 1's win |
| Slow iteration (10–30 min model load under `/mars` contention) | High | front-load instrumentation per load (graph_breaks + fusion logs + replay timing together) |

## 11. Out of scope

- Asking the DLIN kernel team for the `kUsePDL` one-line fix or standalone fused
  kernels (Route B / Route C) — explicitly excluded by the user's "pure SGLang
  side" constraint.
- Host-track (H-1…H-4) work — orthogonal; host overhead is already at parity
  with vLLM (~6 ms each). May be revisited to go from "match" to "beat."
- Speculative-decoding verify-path quality regression (NGRAM/MTP prompt
  regurgitation) — separate work item, tracked in the blog roadmap.
- Making `torch.compile` a model-agnostic first-class path across all models —
  this spec targets Qwen3.5-35B-A3B-FP8 (the perf-gap model); generalization is
  a follow-up.

## 12. Conventions adherence

- **sglang-modify:** every change marked `# DL begin` … `# DL end`, env-gated,
  default-off, so the DL diff stays greppable and survives upstream syncs
  (enforced by `scripts/dl/check_dl_markers.py`).
- **env-var-conventions:** any new `SGLANG_DL_*` flag (e.g. the C-5 path switch,
  fusion on/off) defined per `python/sglang/srt/environ.py` conventions; read the
  `env-var-conventions` skill before adding.
- **no-dataclasses:** new config containers use `msgspec.Struct`, not
  `@dataclass`.
- **large-class-init-style:** if Stage 1/2 touch `Scheduler`/`TokenizerManager`/
  `ModelRunner.__init__`, read that skill first (unlikely for the pass work, but
  the C-5 path may touch `ModelRunner`).

## 13. Milestones & effort

| M | Deliverable | Effort | Depends on |
|---|---|---|---|
| M0 | Stage 0 complete — clean fullgraph compile, no `TORCHINDUCTOR_COMPILE_THREADS=1` | S–M | — |
| M1 | C-5 FP8 restructure (env-gated, byte-diff-verified) | L | M0 |
| M2 | Fusion passes ported + injected into decode compile; **TPOT ≤ 24 ms** | L | M1 |
| M3 | Stage 2 gate experiment run + decision | M | M2 |
| M4 | (conditional) decode `tc_piecewise` for `use_moe_cu`, or documented acceptance | L | M3 |

---

*Summary. Phase II is bounded and mostly mechanical given the exploration
findings: the fused kernels already exist, `w8a8_matmul` is available, the
`PostGradPassManager` injection point exists, and the decode compile↔CG wiring
already captures a compiled graph. The load-bearing novel work is (a) the §5.3
injection so decode's compile actually runs the passes, (b) the C-5 FP8
restructure (correctness-risky, gated), and (c) the Stage-2 `use_moe_cu` gate
experiment. No DLIN kernel-team dependency is required.*

---
name: dl-sglang-update
description: Upgrade the Denglin (DLIN) sglang fork onto a new upstream sglang release tag (e.g. v0.5.14 -> v0.5.15 -> v0.5.16) while preserving every DL adaptation, verified by the SOP. Use when porting DLIN changes to a new sglang version, rebasing the fork onto a fresh tag, or auditing a prior port. Read before starting any sglang version upgrade.
---

# dl-sglang-update — port the DLIN sglang fork onto a new upstream tag

**Goal:** move the DLIN (登临) adaptations onto a new upstream sglang release with
zero DL regressions, gated by the SOP (`run_sglang.sh sop`). This is the
Denglin-side upgrade SOP — NOT an upstream PR.

The DL fork lives on `dl-main` (currently v0.5.14-based; history is squashed so
no v0.5.x tag is a git ancestor — do NOT trust `sglang.__version__`, it is a stale
setuptools-scm string from install time). DL changes are greppable: `grep -rn "# DL"`.

## When to use

- A new upstream sglang tag exists and DL must track it.
- Asked to "upgrade / port / rebase sglang to v0.5.X".
- Auditing whether a branch is correctly ported (run the SOP).

If you are editing DL source for a *feature* (not a version bump), use
[`sglang-modify`](../sglang-modify/SKILL.md) instead.

## The 7-step workflow

### 0. Pre-flight (no GPU needed)
```bash
git checkout dl-main && git pull --ff-only
# confirm current base = nearest tag (smallest 2-dot tree diff = base):
for t in v0.5.14 v0.5.15 v0.5.16; do echo "== $t =="; git diff --shortstat "$t" HEAD; done
# health: SDK + venv + GPU
./run_sglang.sh test                       # torch.version.dl + is_dlin + GPU matmul
dlsmi -L | head                            # KS38 cards present, none hung ([bracket] proc)
```
Confirm the target tag exists: `git tag -l v0.5.X`.

### 1. Record the SOP baseline on the KNOWN-GOOD branch (dl-main) FIRST
This is the regression target — the new tag must match it.
```bash
git checkout dl-main
./run_sglang.sh sop record                 # Qwen3-1.7B (fast); saves docs/dl/sop_baseline_Qwen3-1.7B.json
# optional full gate (the real workload, ~10+ min, TP4):
./run_sglang.sh sop record -M qwen35-35b
```
`record` REFUSES if any gate fails — only a passing tree is a valid baseline.

### 2. Branch + merge the upstream tag (NOT rebase/cherry-pick)
```bash
git checkout dl-main
git checkout -b dl-dev-v0.5.X
git merge --no-commit --no-ff v0.5.X       # 3-way merge via real merge-base (~5deca2d3)
```
**Why merge:** the fork is squashed + divergent, but `git merge-base dl-main v0.5.X`
is a real shared ancestor, so a 3-way merge correctly fuses DL changes with the
upstream diff and only conflicts where DL touched the same lines upstream changed.
Rebase would replay 158 squashed commits and explode; cherry-pick is for projects
that forbid rebase (llama.cpp) — sglang has no such constraint.

### 3. Resolve conflicts preserving DL (read [`sglang-modify`](../sglang-modify/SKILL.md))
Conflicts are typically few (v0.5.15 was 9 files). For each hunk:
- **DL intent is independent of the upstream change** → keep BOTH (e.g. a DL `elif`
  branch + an upstream `elif _is_xpu` branch; a DL `# DL` env override + an upstream
  refactor of the surrounding line).
- **Upstream renamed/refactored a symbol DL uses** → re-apply DL onto the new name
  (e.g. `server_args` -> `view`, `isinstance(...,torch.Tensor)` -> `has_sampled_token_ids`,
  `self.enable_dp_attention` -> `self._resolved().enable_dp_attention`).
- **DL + upstream changed the SAME logic** → keep DL when it's a validated correctness
  fix (e.g. NO_BREAK-CG `seq_lens=ctx_len`), else prefer upstream + re-apply only the
  DL piece that still matters. Document the call in the commit.
- **DL optimization that upstream obsoleted** → drop it (e.g. the low-leverage
  decode skip-copy was superseded by upstream's copy_stream overlap). Note in commit.
- Keep `# DL begin`/`# DL end` balanced; every resolution keeps the DL block intact.
- Before touching `speculative/` read [`speculative-naming`](../speculative-naming/SKILL.md);
  before `Scheduler`/`ModelRunner` `__init__` read [`large-class-init-style`](../large-class-init-style/SKILL.md).

Validate before committing:
```bash
python3 -m py_compile <each resolved file>                       # syntax
python scripts/dl/check_dl_markers.py <each resolved file>       # DL balance
grep -rln -E '^<<<<<<<|^>>>>>>>' python/sglang/ || echo CLEAN    # no markers left
```

### 4. Commit the merge
```bash
git add -u && git commit --no-verify -m "merge(dl): upstream sglang v0.5.X into dl-dev-v0.5.X
<per-file resolution notes>"
```
`--no-verify` is correct here: the `dl-markers` pre-commit hook misflags every
unwrapped upstream line in a merge (it gates single DL commits). Verify the 9-ish DL
resolutions separately with `check_dl_markers.py` (step 3) — that is the real check.

### 5. Build (dlcc)
```bash
./run_sglang.sh install          # editable sglang (DLIN pyproject) — picks up new tree
./run_sglang.sh build-kernel     # sgl-kernel common_ops via dlcc (best-effort)
```
If build fails → [`omc-reference`](../../..) build-error-resolver; common causes are
in §Gotchas. Re-run until `import sglang` is clean.

### 6. Verify — the SOP MUST pass (this is the gate)
```bash
./run_sglang.sh sop verify       # Qwen3-1.7B, fast; diffs vs the step-1 baseline
./run_sglang.sh sop verify -M qwen35-35b    # full 35B TP4 gate
```
**Pass criteria:** VERDICT PASS (exit 0). Correctness gates are absolute; perf gates
must stay within the tolerance band (decode ±20%, prefill ±30%) of the baseline;
R1 exact-greedy-match must hold. If a correctness gate fails → the port broke
behavior; bisect the conflict resolutions. If only perf regressed outside the band →
real regression; investigate the relevant DL path. Iterate steps 3-6 until green.

### 7. Commit verdict + update memory
```bash
./run_sglang.sh sop show > docs/dl/sop_v0.5.X_report.txt   # archive the verdict
```
Write a memory entry recording: tag, conflict count + resolution decisions, SOP
numbers, any DL opt dropped, gotchas hit. Branch is ready to become the new dl-main
once green.

## v0.5.16 is a STRUCTURAL-REFACTOR release (much harder than v0.5.15)

v0.5.15 ported clean (9 conflicts, all textual). **v0.5.16 relocated code tree-wide** —
expect ~23 conflicts AND a cross-file import migration. Do NOT treat it like v0.5.15.
The merge produces 23 conflicts; resolving them is necessary but NOT sufficient — the
build then surfaces import errors from DL files still using the old paths.

Relocation map (old -> new) that DL imports must follow:
- `sglang.srt.layers.attention.fla.*` -> `sglang.kernels.ops.attention.fla.*`
- `sglang.srt.layers.attention.dsv4.*` -> `sglang.kernels.ops.attention.dsv4.*`
- `sglang.srt.layers.quantization.fp8_kernel` -> `sglang.kernels.ops.quantization.fp8_kernel`
- `sglang.jit_kernel.utils` (one file) -> `sglang.jit_kernel.utils.{arch,common,compile,deps}` (a PACKAGE):
  - `_get_default_target_flags` -> `arch.py:get_default_target_flags`
  - `is_arch_support_pdl`, `get_jit_cuda_arch`, `override_jit_cuda_arch` -> `arch.py`
  - `register_dependency`, `get_*_include_paths`, `_find_package_root` -> `deps.py`
  - `__init__.py` re-exports the public names (so `from sglang.jit_kernel.utils import is_arch_support_pdl` still works once arch.py has it)
- model_runner methods `init_aux_hidden_state_capture`, `model_specific_adjustment`,
  `remote_instance_init_transfer_engine` were MOVED OUT of model_runner.py (find their
  new home and re-apply DL's `init_aux_hidden_state_capture` FROZEN_KV_MTP block there).
- prefill_cuda_graph_runner: the inline replay in `execute`/`load_batch` was EXTRACTED
  into `_uses_eager_prefill_tail()` + `_execute_body_capture(...)`. DL's NO_BREAK-CG
  `seq_lens=ctx_len` fix lives in `capture_prepare` (auto-merged, preserved); DL's
  replay-path timing moves into `_execute_body_capture`.
- qwen3_5 MoE call: wrapped in `with get_forward().scoped(fuse_mlp_allreduce=..., mlp_reduce_scatter=...):`
  and the call simplified to `self.mlp(hidden_states)`. DL's per-layer timing
  (`SGLANG_DL_LAYER_TIMING`/`SGLANG_DL_SKIP_MOE`, both opt-in/off) wrapped the OLD call —
  DROP it (re-add inside the scoped block later if needed); adopt the scoped call.
- MoE dispatch: `should_allreduce_fusion` -> `fuse_mlp_allreduce`.
- `get_global_server_args()` still EXISTS in v0.5.16 (NOT renamed) — DL blocks using it are fine.

**Critical JIT step (the v0.5.16 blocker):** because `utils.py` was split, DL's three
utils.py edits MUST move to `arch.py` (their new home) or prefill/FLA kernels miscompile:
1. the gdc_wait/gdc_launch_dependents triton shim (top of arch.py),
2. the DLIN dlcc branch in `get_default_target_flags`,
3. the `is_dlin() -> False` check in `is_arch_support_pdl`.
For compile.py's rename conflicts, take **theirs** (the functions moved out).

Resolve-order that worked (got to 20/23 + the arch.py JIT migration before hitting the
2 deepest): the clear textual conflicts first, the package-split rename via take-theirs
+ re-apply-to-arch.py, then qwen3_5 (take theirs, drop DL timing). The 2 that need careful
manual surgery (not blind take-theirs): `prefill_cuda_graph_runner` (DL timing inside the
extracted method) and `decode_cuda_graph_runner` (DL bs=1 fastpath vs upstream `is_ragged`).
After all conflicts resolve, grep the tree for old import paths and migrate them, then build.



Reusable patterns (DL = HEAD/ours, UP = upstream/theirs):

| File | Pattern | Resolution |
|---|---|---|
| `layernorm_gated.py` | DL pdl-config line + UP independent `device_ctx` xpu fix | keep BOTH |
| `speculative_hook.py` | DL `is_dlin()` clause + UP `server_args`->`view` rename | re-apply DL on new name |
| `batch_result_processor.py` | DL multi-step fast path + UP unified `extend()` | keep DL branch on UP path |
| `dflash_worker_v2.py` | DL timing + UP `_draft_sampler` fast path | keep BOTH (sampler is None on DLIN -> DL path) |
| `eagle_utils.py` | DL DLIN `build_tree` torch fallback + UP xpu branch | keep BOTH elifs |
| `frozen_kv_mtp_worker_v2.py` | DL draft-KV pool setup + UP `init_cuda_graphs()` method | keep DL block, adopt UP method, drop inline call |
| `prefill_cuda_graph_runner.py` | DL NO_BREAK-CG `seq_lens=ctx_len` (correctness) vs UP `lens_cpu` | KEEP DL (validated fix) |
| `server_args.py` | DL `is_dlin` relaxations + UP `_resolved()`/piecewise flag | keep DL relaxations (UP flag irrelevant to text MoE) |
| `scheduler.py` | DL multi-step + WAR-barrier env; UP `_relay_forward_payload`+copy_stream | keep DL multi-step/WAR-override, adopt UP relay (drop DL skip-copy) |

## Gotchas (will bite — from prior ports)

- **One SDK only.** `LD_LIBRARY_PATH` must hold ONLY the active SDK's `lib`. A stale
  second SDK's `libhcrt`/`libLLVM` crashes the JIT in a PassBuilder static init. The
  harness overwrites it in `dlin_runtime_env()` — don't prepend.
- **Triton cache corruption.** A crashed run can leave a corrupt `dl_chunk` cache entry
  in `~/.triton/cache` → every later run crashes. Fix: `rm -rf ~/.triton/cache && cp -a
  ~/.triton/cache.good_backup ~/.triton/cache`. Run `scripts/dl/dl_safe_reset.sh` before launches.
- **`SOP_QUICK="0"` is non-empty.** `${VAR:+--x}` passes `--x` for "0". Gate flags on
  `== "1"`, not `:+`.
- **torch.compile OFF.** `TORCHDYNAMO_DISABLE=1`. DLIN triton emits dlgput64; inductor
  codegen is incompatible. Don't enable compile unless validating the dual-compile path.
- **TP>1 needs NCCL.** Custom allreduce hits `HC_CUK Error=28` → `--disable-custom-all-reduce`.
- **MoE FP4 vs FP8.** Qwen3.5 experts are mxfp4/FP4; calling the fused-MoE op with
  `use_fp8_w8a8` on FP4 weights → 9000× blowup → gibberish. Use `use_mxfp4_w4a16`
  (`SGLANG_DL_MOE_FP4=1` / `SGLANG_DL_MOE_V3=1`).
- **GDN prefill lever.** `SGLANG_DL_GDN_DLIN_EXTEND=1` (dl_chunk) = 8× faster prefill;
  without it sglang loses most scenarios. Default-on in `pick_model qwen35-35b`.
- **DLEOL JIT cache.** `DLEOL_CACHE_SIZE=1024` (default too small → 140× recompile +
  garbage output from corrupted intermediates).
- **Greedy diverges across engines** on long prefill (FP8 drift) — that's why the SOP
  gates on regression-vs-baseline, NOT vLLM agreement.

## Critical DL env vars (must survive a port)

`SGLANG_DL_MOE_FUSED`, `SGLANG_DL_MOE_FUSED_MAX_M`, `SGLANG_DL_MOE_V3`,
`SGLANG_DL_MOE_FP4`, `SGLANG_DL_FP8_Q2`, `SGLANG_DL_GDN_DLIN`,
`SGLANG_DL_GDN_DLIN_EXTEND`, `SGLANG_DL_MULTI_STEP`, `DLEOL_CACHE_SIZE`,
`DLEOL_FLA_ENABLE_PINGPONG`, `DLEOL_FLA_UNROLL_COUNT`, `DLEOL_CU_ADDRESS_CHECK`.
These are set in `run_sglang.sh::pick_model qwen35-35b` and `showcase_prefix_sharing.py`.

## Rollback

```bash
git checkout dl-main          # the port is isolated on dl-dev-v0.5.X
git branch -D dl-dev-v0.5.X   # discard a failed port; the baseline json is preserved
```
The SOP baseline (`docs/dl/sop_baseline_*.json`) is committed on dl-main, so it
survives branch deletion.

## One-line summary

`record baseline on dl-main` -> `branch dl-dev-v<tag>` -> `git merge v<tag>` ->
`resolve 9-ish conflicts keeping # DL blocks` -> `build` -> `sop verify PASS` -> commit.

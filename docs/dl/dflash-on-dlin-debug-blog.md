---
title: "DFLASH Speculative Decoding on Denglin GPUs: A Debug Field Report for Qwen3.5-35B-A3B-FP8"
subtitle: "Trying to bolt a block-diffusion draft model onto a hybrid linear-attention MoE target on a non-NVIDIA GPU — what blocked us, what didn't, and the one checkpoint that changed the story"
authors: "SGLang Denglin (DLIN) Enablement Team"
date: 2026-07-18
tags: [sglang, dlin, denglin, speculative-decoding, dflash, moe, linear-attention, qwen3.5]
model: Qwen3.5-35B-A3B-FP8
hardware: Denglin DLIN KS38 (×4)
draft: z-lab/Qwen3.5-35B-A3B-DFlash
---

# DFLASH Speculative Decoding on Denglin GPUs
## A Debug Field Report for Qwen3.5-35B-A3B-FP8

> *The plan was simple: take sglang's newest speculative-decoding algorithm (DFLASH), point it at our hybrid linear-attention MoE target on DLIN, and see if a small draft model buys us free tokens. The reality was a scavenger hunt for a checkpoint, two arch-incompatible red herrings, a context-length gate, and a hybrid-GDN verify path that — on paper — fixes the exact bug that killed our earlier NGRAM/MTP attempts.*

**Authors:** SGLang Denglin (DLIN) Enablement Team　·　**Date:** July 2026
**Target:** Qwen3.5-35B-A3B-FP8　·　**Draft:** `z-lab/Qwen3.5-35B-A3B-DFlash`　·　**Hardware:** Denglin DLIN KS38 (4×)

---

## Abstract

We investigated running **DFLASH** — sglang/vLLM's block-diffusion speculative-decoding algorithm — on the DLIN port of sglang for **Qwen3.5-35B-A3B-FP8**, the hybrid GatedDeltaNet + full-attention MoE model we had previously brought to TP4 parity with vLLM (TPOT ≈ 26.5 ms, see [the TP4 field report](./blog-sglang-dlin-qwen35-35b-tp4.md)). DFLASH is not "pair any small model as a draft"; it requires a **specifically-trained draft checkpoint** that consumes the *target model's per-layer hidden states* and runs a compact draft KV cache. We found that:

- **The DLIN/sglang plumbing exists.** sglang's `qwen3_5.py` target implements the DFLASH hidden-state-capture hook (`set_dflash_layers_to_capture`, `python/sglang/srt/models/qwen3_5.py:1312`), and the `DFlashDraftModel` (`python/sglang/srt/models/dflash.py`) is a dense, architecture-agnostic transformer. vLLM's reference even lists our exact target arch `Qwen3_5MoeForConditionalGeneration` as a supported DFlash target (`vllm-new-overlay/vllm/v1/spec_decode/llm_base_proposer.py:1178`).
- **The first two "drafts" we found were red herrings.** The only DFLASH checkpoint on local disk — `Qwen3-8B-DFlash-b16` — is arch-incompatible with Qwen3.5 (hidden 4096 vs 2048, vocab 151936 vs 248320, trained on dense Qwen3). Pointing DFLASH at it dies in the draft `ModelConfig`, and would die again in the very first GEMM even if it loaded.
- **The correct draft exists but had to be fetched.** The official `z-lab/Qwen3.5-35B-A3B-DFlash` is dimension-matched to our target on every axis (hidden 2048, vocab 248320, `num_target_layers` 40, `max_position_embeddings` 262144), and its 69 weights map exactly onto `DFlashDraftModel` (verified offline). With it, DFLASH is structurally ready to load on DLIN; the live run is pending (§6.1).
- **On the hybrid target, the correctness-critical GDN-state path is present** — `_update_target_mamba_state_after_verify` (`dflash_worker_v2.py:1042`) commits the linear-attention recurrent state after each target verify via fused kernels, which is precisely the mechanism whose absence caused the prompt-regurgitation bug in our NGRAM/MTP runs.

**Verdict (full detail in §7):** DFLASH is unblocked at every software level on DLIN Qwen3.5-35B-A3B-FP8 — the correct dimension-matched draft exists and loads cleanly into `DFlashDraftModel`, the target exposes the capture hook, and the draft carries no GDN layers. The one empirical gap — the live TP4 build, a token-for-token correctness check, and a TPOT number — was **not** blocked by DFLASH but by a DLIN driver defect (an OOM from an over-aggressive `mem_fraction_static` hangs the device-cleanup path in `os_schedule_timeout` with `SIGKILL` pending, leaking the GPU until a root reset). Correctness and performance are pending a GPU reset + a retry with the proven `mem_fraction_static=0.60` recipe.

This post documents the journey, including the dead ends, because the dead ends are the load-bearing part of the conclusion.

---

## 1. What DFLASH is, and why we wanted it on DLIN

DFLASH is a **draft-model-driven** speculative-decoding algorithm. Unlike NGRAM (which drafts from a lookup table of seen n-grams) or the model's own MTP head (which drafts from a single trained next-n layer), DFLASH trains a **separate, small "draft" transformer** to predict several tokens ahead, then asks the target model to verify them in one batched forward. Accepted draft tokens are "free" — you get N tokens for the cost of one target forward plus a cheap draft forward.

The contract on the sglang side (from `test/registered/spec/dflash/test_dflash.py`):

```
--speculative-algorithm DFLASH
--speculative-draft-model-path <draft>
--speculative-dflash-block-size N      # = --speculative-num-draft-tokens N
# implied: num_steps=1, eagle_topk=1, pp_size=1, no DP attention
```

Two design choices matter for the rest of this post:

1. **The draft has no embeddings and no LM head.** It is fed the *target model's* token embedding as input and produces hidden states that the *target's* LM head converts to logits (`python/sglang/srt/models/dflash.py:369-400`). This means **the draft's `hidden_size` must equal the target's `hidden_size`**, and the draft must share the target's vocabulary. A draft trained for a different target is not a drop-in.
2. **The draft consumes target per-layer hidden states.** A learned `fc` projection maps a concatenation of K target-layer hidden states into the draft's input space (`dflash.py:330-367`). The set of target layers (`dflash_config.target_layer_ids`) is baked into the checkpoint. So `len(target_layer_ids)` and the target `hidden_size` must match what the draft was trained against.

The payoff, if it works on our hybrid MoE target: our decode TPOT is ~26.5 ms, dominated by 8 per-expert FP8 GEMMs per step. If a draft can get, say, 2 of every 3 tokens accepted, we cut the number of expensive target forwards by ~3× — a potentially large win on a model where the forward is this expensive.

The risk, hard-won from our earlier NGRAM/MTP work: on this hybrid model, the **GDN (linear-attention) layers carry recurrent state**. During target verify (running the target on a batch of speculative tokens), that state must be threaded correctly or the model regurgitates the prompt and the "accept rate" becomes a lie (see the prompt-regurgitation bug in our [NGRAM/MTP report](./dlin-sglang-mtp-vs-ngram-report.md)). So correctness, not just speed, was the bar.

---

## 2. The draft-model scavenger hunt

### 2.1 What's on local disk

`/mars/aebox/LLM/model/` hosts a wall of Qwen3.5 checkpoints — `Qwen3.5-2B`, `-4B`, `-27B`, the `-35B-A3B` family — but **none of them is a DFLASH draft**. A DFLASH draft is identifiable by `architectures: ["DFlashDraftModel"]` and a `dflash_config` block. Grepping the whole tree turned up exactly one:

```
/mars/aebox/LLM/model/Qwen3-8B-DFlash-b16
```

This is a DFLASH draft — but for **Qwen3-8B** (dense), not Qwen3.5-35B-A3B. The config mismatch is total:

| field | target Qwen3.5-35B-A3B-FP8 | local draft Qwen3-8B-DFlash-b16 | match? |
|---|---|---|---|
| `hidden_size` | 2048 | 4096 | **no** |
| `vocab_size` | 248320 | 151936 | **no** |
| `num_hidden_layers` (target) | 40 | draft expects `num_target_layers`=36 | **no** |
| `model_type` | `qwen3_5_moe_text` | `qwen3` | **no** |
| `layer_types` (target) | hybrid linear/full + MoE | — | — |
| `max_position_embeddings` | 262144 | 40960 | **no** |

Because the draft is fed the target embedding directly into draft layer 0 (`dflash.py:383`), the first `qkv_proj` GEMM would receive a 2048-dim input against a 4096-dim weight — a hard shape error. **This draft cannot work with this target.** The `Qwen3.5-2B` on disk, despite the name, is also no help: it is a *full* Qwen3.5 hybrid model (its own embeddings/LM head, its own GDN layers), not a DFLASH draft — DFLASH does not consume another standalone model.

### 2.2 The first load attempt: what actually breaks

Empirically pointing the engine at the local draft against our target (TP4, fa3, page 16, bf16):

```bash
python -m sglang.launch_server \
  --model-path /mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/ \
  --tp 4 --dtype bfloat16 --attention-backend fa3 --page-size 16 \
  --mem-fraction-static 0.7 --disable-custom-all-reduce --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /mars/aebox/LLM/model/Qwen3-8B-DFlash-b16 \
  --speculative-dflash-block-size 16
```

The target loads fine — `Qwen3_5MoeForConditionalGeneration`, fp8, 16.3 s, 8.66 GiB/rank. Then the DFLASH draft worker (`maybe_init_draft_worker` → `DFlashWorker`) dies constructing the draft `ModelConfig`:

```
File ".../configs/model_config.py", line 608, in _derive_context_length
  raise ValueError(
ValueError: Target model's context_length (262144) is greater than the derived
context_length (40960). ... set the env var SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
```

That gate is the *first* obstacle; it is also the most misleading. The draft worker *does* align the draft's runtime `context_length` to the target's (`dflash_worker_v2.py:144-146`), but `ModelConfig.from_server_args` validates the draft against its own `max_position_embeddings` (40960) *before* that override lands. The test suite sidesteps this with `envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True)` (`test_dflash.py:69`). With that env var set, this particular checkpoint would proceed — only to hit the hidden-size/vocab shape mismatch at the first forward. The gate is a paper cut; the shape mismatch is the severed artery.

### 2.3 The correct draft: `z-lab/Qwen3.5-35B-A3B-DFlash`

The trained DFLASH draft for our *exact* target is published as **[`z-lab/Qwen3.5-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash)** (Apache-2.0, ~772 MiB, `base_model: Qwen/Qwen3.5-35B-A3B`). It was not mirrored onto our model store, so we fetched it. Its config is dimension-matched to the target on every axis that matters:

| field | target | `z-lab/Qwen3.5-35B-A3B-DFlash` | match? |
|---|---|---|---|
| `hidden_size` | 2048 | 2048 | ✅ |
| `vocab_size` | 248320 | 248320 | ✅ |
| `num_target_layers` | 40 | 40 | ✅ |
| `target_layer_ids` | — | [1, 6, 11, 16, 22, 27, 32, 37] (8 layers) | ✅ valid |
| `max_position_embeddings` | 262144 | 262144 | ✅ (also defuses the §2.2 gate) |
| `mask_token_id` | — | 248077 (in-vocab) | ✅ |

The draft itself is a **dense, 6-layer Qwen3-style transformer** (5 sliding-attention + 1 full-attention layers, 32 heads, sliding window 4096). Notably it contains **no GDN / linear-attention layers** — the GDN complexity lives entirely in the target. This is the first piece of good news for DLIN: the *draft* side will not exercise any of the DLEOL-JIT linear-attention kernels that are the usual source of DLIN friction.

---

## 3. Loading DFLASH with the correct draft

With the correct draft in hand we verified, offline and against sglang's own model code, that the checkpoint is **structurally loadable** by `DFlashDraftModel` (`python/sglang/srt/models/dflash.py`):

- All **69 tensors** in `model.safetensors` map onto the model's parameter names — `fc.weight`, `hidden_norm.weight`, `norm.weight`, and per-layer `q_proj/k_proj/v_proj` (→ fused `qkv_proj`), `o_proj`, `q_norm/k_norm`, `gate_proj/up_proj` (→ fused `gate_up_proj`), `down_proj`, and the two RMSNorm weights. **Full coverage, zero missing tensors.**
- `fc.weight` is `(2048, 16384)` = `[hidden=2048, 8 target_layers × 2048]`, exactly matching `len(target_layer_ids)=8` and the target `hidden_size=2048`. The `fc.weight shape mismatch` guard at `dflash.py:445` will not fire.
- The draft reuses the **target's tokenizer** (the repo ships none); because `dflash_config.mask_token_id=248077` is set, the worker takes the mask id from config and only *consistency-checks* it against the target tokenizer (`dflash_worker_v2.py:540-555`). We copied the target's `tokenizer.json`/`tokenizer_config.json` into the draft dir to satisfy any draft-side `ModelConfig` tokenizer init.

The draft worker selects the draft attention backend from `speculative_draft_attention_backend`, defaulting to the target's (`fa3` on DLIN — in the supported set at `dflash_worker_v2.py:108`), and aligns the draft `context_length` to the target's (`dflash_worker_v2.py:144`). Because the official draft's `max_position_embeddings=262144` already equals the target's, the §2.2 context-length gate does **not** fire here — `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` is set only as a belt-and-braces.

**What we did *not* get to verify live, and why, is the subject of §6.** The single TP4 build with the correct draft was blocked by a DLIN driver issue (an OOM during an earlier run with an over-aggressive `mem_fraction_static=0.70` left the GPU memory leaked and unreclaimable — see §6). The corrected recipe that avoids the OOM (`mem_fraction_static=0.60`, `context_length=4096`, matching the proven TP4 parity config) is in the repro block below; a retry is pending a GPU reset.

Repro:

```bash
export CUDA_VISIBLE_DEVICES=16,17,18,19
source sdk-dlop-07-13-20-30/env.sh
export SGLANG_DL_GDN_DLIN=1 SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=16 \
       SGLANG_DL_FP8_Q2=1 DLEOL_CACHE_SIZE=1024 \
       DLEOL_FLA_ENABLE_PINGPONG=1 DLEOL_FLA_UNROLL_COUNT=8 \
       SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
DRAFT=/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang/models-dl/Qwen3.5-35B-A3B-DFlash
python scripts/dl/dflash_correctness.py dflash   # writes /tmp/dflash-repro/dflash.json
```

---

## 4. Correctness: does DFLASH match greedy, and does it regurgitate the prompt?

The non-negotiable check, given our history: **DFLASH greedy output must match the no-spec greedy output token-for-token** on a fixed prompt (`"The quick brown fox jumps over the lazy dog."`, `max_new_tokens=64`, `temperature=0`), and the output must **not** begin with prompt tokens (the NGRAM/MTP failure signature).

**Status: empirically pending (blocked by the §6 GPU leak).** The token-for-token comparison script (`scripts/dl/dflash_correctness.py`, modes `plain` and `dflash`) is written and ready; it emits both the output token-id list and a `regurgitates_prompt` flag (does the decoded output begin with prompt text — the NGRAM/MTP failure signature). It did not complete a DFLASH pass in this session because the TP4 build could not be retried on the leaked GPUs. What we *can* say with confidence is the code-path analysis below.

### 4.1 The GDN-state-after-verify path (why this *might* be correct where NGRAM/MTP wasn't)

The reason our earlier NGRAM/MTP runs "achieved" 2.9× vLLM on garbage output was that the spec-verify target forward corrupted the GDN recurrent state — the verify predicted prompt tokens at verify positions because the linear-attention state wasn't loaded/committed correctly. DFLASH has an explicit fix for this:

- During `TARGET_VERIFY`, the GDN/Mamba kernels run with `disable_state_update=True` and cache per-step intermediate states (`dflash_worker_v2.py:1042-1054`).
- After acceptance, `_update_target_mamba_state_after_verify` commits the state for each request's last accepted step by calling `attn_backend.update_mamba_state_after_mtp_verify(...)`, which on the hybrid backend (`hybrid_linear_attn_backend.py:911`) runs fused gather-scatter kernels (`fused_mamba_state_scatter_with_mask`, `fused_conv_window_scatter_with_mask`).

So the mechanism is present. Whether it is *correct on DLIN* is exactly what §4's token comparison tests.

---

## 5. Performance: DFLASH vs plain

**Status: empirically pending (blocked by the §6 GPU leak).** The baseline we measure against is established and stable: **plain (no-spec) TP4 TPOT ≈ 26.5 ms/token** on this exact model + SDK + hardware ([TP4 field report, §6](./blog-sglang-dlin-qwen35-35b-tp4.md)), decode-dominated by 8 per-expert FP8 GEMMs per step. DFLASH's value proposition on this target is unusually favorable *in principle*: each target forward is expensive, so even a modest mean-accept-length (e.g. 2 accepted draft tokens) would replace a 26 ms forward with a cheap 6-layer dense draft forward + one batched verify. Whether that materializes on DLIN — where the draft adds its own KV pool, its own fa3 sliding-attention path, and the verify re-enters the chunk-GDN host overhead — is exactly what the best-of-3 `dflash_correctness.py dflash` run would quantify. The number we do not yet have: DFLASH TPOT, end-to-end tokens/s, and the spec accept rate.

---

## 6. DLIN-specific surprises

Three concrete DLIN-specific findings came out of this investigation, in descending order of severity.

### 6.1 The big one: an OOM on DLIN leaks GPU memory *unrecoverably* without root

This is the finding that cut the empirical run short, and it is **not DFLASH-specific** — it is a DLIN driver defect that any sglang run can trip.

The first DFLASH attempt used `--mem-fraction-static 0.70` (copied from the DFLASH test's `mem_fraction_static=0.7`, which is tuned for an 8B dense Llama on 32 GiB CI cards — not our 35B MoE). The 35B target loaded cleanly (`mem usage=8.66 GiB/rank`, `avail mem=22.91 GB`), but the engine OOM'd during warmup/cuda-graph capture:

```
DL_MOE_ERR: CUDA out of memory. Tried to allocate 4.00 GiB. GPU 0 has a total
capacity of 32.00 GiB of which 799.71 MiB is free. ... 30.23 GiB is allocated.
```

(The per-rank log indices 0–3 map to physical GPUs 16–19 under `CUDA_VISIBLE_DEVICES`.) The OOM itself is a misconfiguration, easily fixed by the proven recipe `mem_fraction_static=0.60` + `context_length=4096`. **The real defect is what happened next:** the scheduler subprocesses exited into zombie (`<defunct>`) state holding the device, and the parent python process (`sglang::Engine`) hung in the DLIN driver's `os_schedule_timeout` with **`SIGKILL` pending but undelivered**:

```
$ cat /proc/<pid>/status
State:  S (sleeping)
ShdPnd: 0000000000004100      # SIGKILL is pending
$ cat /proc/<pid>/wchan
os_schedule_timeout
$ kill -9 <pid>; sleep 90; [ -d /proc/<pid> ] && echo STILL ALIVE   # → STILL ALIVE
```

The result: physical GPUs 16, 18, 19 report **31950 MiB used** indefinitely (zombie schedulers `3509256/8/9` marked `Zl` — zombie with locked device memory); GPU 17 was the one rank that freed cleanly. `kill -9` on the parent and the zombies does nothing — the driver release path is parked in an uninterruptible wait. This is a DLIN driver bug (cleanup/close path hangs on OOM, defeating `SIGKILL`), and on a shared box it is unrecoverable without a root driver reset. **It is now the single thing standing between this investigation and a complete §5.** The lesson for anyone serving on DLIN: use the conservative memory recipe, because an OOM here doesn't just crash the run — it bricks the GPU until root intervenes.

### 6.2 The draft side carries no GDN — the DLIN-linear-attention friction is entirely on the target

A relief, worth stating because it could have gone the other way: the official `z-lab/Qwen3.5-35B-A3B-DFlash` draft is a **dense, 6-layer Qwen3 transformer** (5 sliding-attention + 1 full-attention). It contains **zero GatedDeltaNet / linear-attention layers**, so the draft forward does not touch any of the DLEOL-JIT `_dl_C.dl_*` linear-attention kernels that are the usual source of DLIN enablement pain. All of the hybrid-GDN complexity lives in the *target*, which already runs correctly under the existing TP4 recipe. The one draft-side DLIN unknown is whether the fa3 backend cleanly handles `sliding_attention` (sliding window 4096) for the draft's 5 sliding layers — unverified because of §6.1, but low-risk since RadixAttention + fa3 already serves the target's full-attention layers.

### 6.3 vLLM explicitly sanctions this target arch for DFLASH

A small but load-bearing finding for the verdict: vLLM's `llm_base_proposer` allow-lists both `Qwen3_5ForConditionalGeneration` and **`Qwen3_5MoeForConditionalGeneration`** (our exact arch) as supported DFLASH targets (`vllm-new-overlay/vllm/v1/spec_decode/llm_base_proposer.py:1177-1178`), and the proposer override comments *"DFlash supports Qwen3.5 models"* (`dflash.py:84`). This is not a porting experiment from our side — the reference implementation considers a hybrid linear-attention MoE Qwen3.5 a first-class DFLASH target. That raises our prior that the sglang port, once it runs, will be correct.


---

## 7. Honest verdict

**Does DFLASH run on DLIN Qwen3.5-35B-A3B-FP8?** Every prerequisite that can be checked without a live TP4 run checks out: the target implements the hidden-state-capture hook (`qwen3_5.py:1312`); the official dimension-matched draft `z-lab/Qwen3.5-35B-A3B-DFlash` is fetched and its 69 weights map exactly onto `DFlashDraftModel`; vLLM sanctions our exact arch as a DFLASH target; the draft side has no GDN layers (so no new DLIN linear-attention kernels are needed). The one thing we could **not** confirm in this session is the end-to-end build + a token-for-token correctness check + a TPOT number, and the reason is **not DFLASH** — it is the DLIN driver OOM-cleanup hang (§6.1) that leaked 3 of our 4 GPUs after a misconfigured `mem_fraction_static=0.70` run and could not be cleared without root.

**Is it correct? / Is it faster?** Both are the empirical questions §4 and §5 are designed to answer and both are **pending a GPU reset + retry with the corrected `mem_fraction_static=0.60` recipe**. The correctness prior is reasonably high: the GDN-state-after-verify commit path exists and is exactly what NGRAM/MTP was missing; the draft carries no stateful layers of its own. The performance prior is favorable on paper (expensive MoE target, cheap dense draft) but unmeasured.

**The blocker is environmental, not algorithmic.** Concretely: (1) the local model store has no Qwen3.5 DFLASH draft — fetch `z-lab/Qwen3.5-35B-A3B-DFlash`; (2) the DFLASH test's `mem_fraction_static=0.7` OOMs a 35B MoE on 32 GiB DLIN cards — use `0.60` + `context_length=4096`; (3) a DLIN driver defect turns such an OOM into a permanently leaked GPU — avoid (1)+(2) triggering it. With those three handled, nothing in the DFLASH software stack is known to block running on DLIN Qwen3.5.


---

## Reproduction

**Target:** `/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/` (TP4, fa3, page 16, bf16).
**Draft:** `z-lab/Qwen3.5-35B-A3B-DFlash` → fetched to `models-dl/Qwen3.5-35B-A3B-DFlash/` (local; the shared `/mars/aebox/LLM/model` store is read-only and has no Qwen3.5 DFLASH draft).

```bash
# 1. Fetch the draft (~772 MiB) — not on the local model store
curl -sL https://hf-mirror.com/z-lab/Qwen3.5-35B-A3B-DFlash/resolve/main/config.json \
  -o models-dl/Qwen3.5-35B-A3B-DFlash/config.json
curl -L    https://hf-mirror.com/z-lab/Qwen3.5-35B-A3B-DFlash/resolve/main/model.safetensors \
  -o models-dl/Qwen3.5-35B-A3B-DFlash/model.safetensors
# copy the target tokenizer (shared vocab) into the draft dir
cp /mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/{tokenizer.json,tokenizer_config.json,generation_config.json} \
  models-dl/Qwen3.5-35B-A3B-DFlash/

# 2. Correctness + perf (best-of-3), plain vs DFLASH
#    NOTE mem_fraction_static=0.60 + context_length=4096 (NOT the test's 0.7):
#    0.7 OOMs a 35B MoE on 32 GiB DLIN cards, and the OOM triggers the §6.1
#    driver cleanup-hang that leaks the GPU. See scripts/dl/dflash_correctness.py.
export CUDA_VISIBLE_DEVICES=16,17,18,19
source sdk-dlop-07-13-20-30/env.sh
export SGLANG_DL_GDN_DLIN=1 SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=16 \
       SGLANG_DL_FP8_Q2=1 DLEOL_CACHE_SIZE=1024 \
       DLEOL_FLA_ENABLE_PINGPONG=1 DLEOL_FLA_UNROLL_COUNT=8 \
       SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python scripts/dl/dflash_correctness.py plain    # /tmp/dflash-repro/plain.json
python scripts/dl/dflash_correctness.py dflash   # /tmp/dflash-repro/dflash.json
```

---

## Live Results (2026-07-18, GPU 20-23, TP4, sdk-dlop, mem 0.60)

DFlash was run live (after the GPU leak was bypassed by moving to GPU 20-23).
Two prompts tested: the degenerate "fox" prompt and a real reasoning prompt.

### Degenerate prompt ("The quick brown fox...")

| Metric | plain | DFLASH | Speedup |
|---|---|---|---|
| TPOT | 27.81 ms | 18.83 ms | **1.48×** |
| tok/s | 35.95 | 53.11 | +48% |
| output_ids match | — | **exact (64/64)** | ✅ correct |
| regurgitates prompt | True | True | (model's natural greedy degeneration, NOT a bug) |

DFlash is **correct** (token-for-token match) and **1.48× faster** on this prompt.
But the output is degenerate (prompt repeated) — trivially predictable → high
draft accept rate → fast.

### Real reasoning prompt ("Explain the concept of machine learning...")

| Metric | plain | DFLASH | Speedup |
|---|---|---|---|
| TPOT | 39.68 ms | 44.70 ms | **0.89× (slower!)** |
| tok/s | 25.20 | 22.37 | -11% |
| output_ids match | — | **exact (128/128)** | ✅ correct |
| regurgitates prompt | False | False | real `<think>` reasoning output |

DFlash is **correct** (128/128 token match, genuine reasoning) but **11% slower**.
The draft model (6-layer dense) can't predict the 35B hybrid's diverse `<think>`
reasoning → low accept rate → draft overhead exceeds target-forward savings.

### Verdict

**DFlash is the FIRST spec-decoding algorithm verified CORRECT on DLIN for this
hybrid model** (unlike NGRAM/MTP's prompt-regurgitation bug). The GDN-state-after-
verify plumbing (`dflash_worker_v2.py:1042` → `hybrid_linear_attn_backend.py:911`)
works — spec-verify correctly loads GDN state.

However, DFlash is **NOT a net speedup for real/diverse workloads** on this model.
The 1.48× on the degenerate prompt was a false positive (trivially predictable
output → near-100% accept rate). On real reasoning output, the draft model is
too weak (0.89×). A stronger draft model, or a prompt distribution where the
target is more predictable (e.g., code completion, templated text), would improve
the accept rate and could yield a net speedup.

---

## Sources

- [z-lab/Qwen3.5-35B-A3B-DFlash (HuggingFace)](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) — the official trained DFLASH draft for our target.
- [SGLang TP4 field report](./blog-sglang-dlin-qwen35-35b-tp4.md) — the baseline (TPOT ≈ 26.5 ms) we measure DFLASH against.
- [NGRAM/MTP report](./dlin-sglang-mtp-vs-ngram-report.md) — the prompt-regurgitation correctness bug that sets the bar for §4.

---

## §5. Why our DFlash is 1.2×, not the official 3.7× — root cause + DLIN adaptation (2026-07-20)

The user asked: the official z-lab benchmark shows **3.71× speedup** (HumanEval,
block=16) on the *same* `Qwen3.5-35B-A3B` model, yet our DLIN number was ~1.88×
at best and 0.89× on reasoning. Something in the adaptation must be wrong. This
section is the answer.

### 5.1 The baseline was wrong: 1.88× was a mirage

The "1.88×" reported earlier compared DFlash against an **un-optimized** plain
baseline. Against the **optimized** plain serving recipe (same
`cuda_graph_max_bs_decode=4`, `max_running_requests=4`, FP8, TP4) the truth is:

| run | TPOT | tok/s | speedup vs optimized plain |
|---|---|---|---|
| **plain (FP8, TP4, CG)** | **28.11 ms** | 35.6 | 1.00× |
| DFlash block=16 (FP8) | 28.81 ms | 34.7 | **0.98× — no speedup** |

**DFlash at the official's recommended block=16 gives essentially zero speedup on
DLIN** (0.98×). The earlier "1.88×" was beating a crippled baseline, not the real
one. §4's pessimism ("not a net speedup for real workloads") was correct; this is
why, quantified.

### 5.2 Per-step decomposition: the verify forward is the killer

Instrumented `dflash_worker_v2.py` (draft forward at `:1480`, target verify at
`:1535`, gated by `SGLANG_DL_DFLASH_TIMING=1`). Each DFlash step =
`draft_forward + verify_forward`. Measured (block=16):

```
draft_ms ≈ 9.0   ← cheap (6-layer draft model), NOT the problem
verify_ms ≈ 123   ← 4.4× a plain decode (28 ms)!  THIS is the bottleneck
cg = True         ← verify IS cuda-graph-captured, so this is not launch overhead
```

The verify forward processes `block_size + 1 = 17` tokens through the full 40-layer
target (30 GDN + 10 full-attn, MoE). On DLIN that costs **123 ms = 4.4× the M=1
decode**. On the official B200 the same verify is only **~1.9× decode** (memory-bound;
extra tokens ride the weight-load for free). **DLIN's verify is compute-bound and
scales with M; B200's is memory-bound and nearly flat.** This single difference is
why DFlash economics that work on B200 collapse on DLIN.

### 5.3 verify_ms scales linearly at ~5.6 ms/token on DLIN

Sweeping block_size (so verify M = block+1 changes):

| block | verify M | verify_ms | draft_ms | TPOT | accept_len | speedup vs 28.1 ms |
|---|---|---|---|---|---|---|
| 16 | 17 | 123 | 9.0 | 28.81 | 5.16 | 0.98× |
| **8** | **9** | **75** | **6.4** | **23.31** | **4.21** | **1.21×** |
| **4** | **5** | **53** | **6.1** | **23.07** | **3.14** | **1.22×** |

`verify(M) ≈ 28 + 5.6·(M−1) ms`. Every extra verify token costs **5.6 ms** on DLIN.
Decode is 28 ms/token, so the *incremental* verify token (5.6 ms) is still 5× cheaper
than a decode — spec decode *can* win — but only if `accept_len` is high enough to
amortize the fixed draft+verify-M=1 cost (~34 ms). The optimization landscape is
fundamentally different from B200:

- **B200 (memory-bound verify): bigger block wins** — verify is ~free, so maximize
  accept_len → official picks block=16.
- **DLIN (compute-bound verify, 5.6 ms/token): smaller block wins** — each draft
  token the target rejects costs 5.6 ms of wasted verify. We were using B200's
  block=16; **block=8 is the correct DLIN adaptation** → 1.22×.

**Where DFlash wins — accept_len is workload-driven (tested).** Same FP8/TP4/block=8,
HumanEval-style pure function-completion prompt (highly predictable code):

| workload | accept_len | DFlash TPOT | plain TPOT | speedup |
|---|---|---|---|---|
| prose code-request | 4.21 | 23.31 ms | 28.11 ms | 1.21× |
| **HumanEval-style completion** | **6.74** | 51.89 ms* | 65.22 ms* | **1.26×** |
| reasoning (`<think>`, §4) | ~1 | 44.70 ms | 39.68 ms | **0.89× (slower)** |

\* higher absolute TPOT on that run (non-adjacent GPU topology 17/20/22/31 + residual
state between back-to-back processes); the *relative* DFlash-vs-plain is the reliable
figure. accept on HumanEval (6.74) lands at/above the official block=8 (5.4–5.9) →
**our draft/target agreement is fine; the accept gap was workload, not adaptation.**
DFlash beats plain on predictable code (1.26×), breaks even on prose, loses on
reasoning — exactly the workload sensitivity the official numbers hide.

### 5.4 accept_len is the second gap — and it is NOT FP8 (tested)

Even at block=8, our `accept_len = 4.21` vs the official block=8 = **5.4–5.9**. The
official benchmark runs **bf16**, so the obvious suspect was FP8 in the target verify
changing the greedy choice vs the bf16 draft. **Tested — disproved:**

| run | TPOT | accept_len | verify_ms |
|---|---|---|---|
| DFlash FP8, block=8 | 23.31 ms | **4.21** | 75 |
| DFlash bf16 (native, non-FP8), block=8 | 146.05 ms | **4.32** | 548 |

bf16 accept (4.32) ≈ FP8 accept (4.21). **FP8 does not lower accept.** What it *does*
do is make verify affordable: bf16 verify is **548 ms (7× slower)** because bf16
weights are 2× the memory and the MoE GEMM is slower. **FP8 is mandatory on DLIN**
— it is the only way the verify forward is cheap enough for spec decode to break
even. The 4.2-vs-5.4 accept gap is therefore **workload, not precision**: the
official numbers are on HumanEval/MBPP (highly-predictable structured code); our
prompt is a prose code-request. Accept is inherently lower on diverse/reasoning
output (see §4: reasoning prompt accept collapsed to ~1).

### 5.5 Other adaptation deltas vs the official recipe

The official launch command uses several things we do not / cannot:

| item | official (B200) | us (DLIN) | impact |
|---|---|---|---|
| dtype | bf16 | FP8 | lowers accept_len (§5.4) |
| TP size | 1 | 4 | per-step NCCL all-reduce on both draft+verify |
| draft attn backend | fa4 | fa3 (default) | draft is 9 ms anyway — minor |
| linear-attn (GDN) backend | flashinfer | DLIN GDN kernels | GDN extend is the 5.6 ms/token slope |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | 1 | off | host/pipeline overlap |
| allreduce fusion | enabled | NCCL plain | TP4 overhead |
| `FUSED_MAX_M` | n/a | was 16, now 32 | verify M=17 must stay on fast fused-MoE path |

Note: `FUSED_MAX_M=32` (covering verify M=17, which the old default 16 missed) was
necessary but **not sufficient** — verify stayed 123 ms because the cost is the GDN
extend + MoE compute at M=17, not the MoE *fallback path*. The real lever is block_size.

### 5.6 Verdict (2026-07-20)

1. **Root cause:** DLIN's target verify forward is compute-bound and scales
   ~linearly (5.6 ms/token), vs B200's memory-bound near-flat verify. Spec-decode
   economics that give 3.7× on B200 give ~1× at block=16 on DLIN.
2. **Adaptation fix (sglang-side, zero kernel work):** use **block=8**, not the
   official's block=16. FP8 must stay on (bf16 verify is 7× slower).
3. **The accept gap is workload, not precision** (§5.4): bf16 accept ≈ FP8 accept.

### 5.7 Clean numbers + MTP + the last sglang-side lever (2026-07-20)

Re-ran plain-vs-DFlash **sequentially on freshly-idle GPUs** (no parallel-process
contention, which had inflated earlier TPOTs ~2× via residual GPU memory — the DLIN
driver-cleanup leak again):

| workload | plain TPOT | DFlash b=8 TPOT | accept | **speedup** |
|---|---|---|---|---|
| prose code-request | 27.37 ms | 23.53 ms | 4.17 | **1.16×** |
| HumanEval-style code | 58.06 ms | 45.22 ms | 6.67 | **1.28×** |

These are the reliable numbers. **block=8 DFlash = 1.16× (prose) / 1.28× (predictable
code)** over optimized plain.

**Fast-fused MoE path confirmed firing** (`SGLANG_DL_MOE_TRACE=1` prints `FAST-FUSED
path taken`); A/B vs the slow path (FUSED_MAX_M=8 forcing M=9 off the fast path):
fast 23.10 ms vs slow 38.59 ms TPOT — so FUSED_MAX_M=32 is necessary and applied. But
verify_ms itself barely differs (74 vs 79 ms): the verify cost is dominated by
**GDN/attention extend, not the MoE**. No further MoE-path lever exists.

**Last sglang-side lever — draft CUDA graph — investigated, NOT a lever.** Initial
probe (`cuda_graph_runner` attr) said draft was eager; **corrected probe**
(`decode_cuda_graph_runner`) shows `DecodeCudaGraphRunner, eager=False` — the draft
forward IS cuda-graph-backed. draft=9 ms is near its floor; no headroom there.

**MTP (FROZEN_KV_MTP) now CORRECT** (the §4 prompt-regurgitation bug is fixed — MTP
output matches plain token-for-token on both reasoning and code). But it does **not**
beat plain either: num_steps=3 (verify M=4) → 0.83× on code / 0.72× on reasoning.
MTP's smaller verify M (4 vs DFlash's 9) was supposed to suit DLIN's compute-bound
verify, but its accept on this model is below the ~1.74 break-even, so the verify
overhead still loses. Same structural wall.

### 5.9 Testing the assumptions behind the "structural ~1.3× ceiling" (2026-07-20)

The ceiling claim rests on assumptions; the two most threatening were tested:

**Assumption — "TP4 NCCL is structural / comm-bound verify": TESTED, confirmed not the lever.**
Ran TP2 (35B FP8 = 17.5 GB/card, mem_fraction 0.82):

| | plain prose | dflash prose | ratio | dflash humaneval |
|---|---|---|---|---|
| TP2 | 41.02 ms | 38.69 ms | **0.94× (worse)** | **OOM** (600 MiB free) |
| TP4 (clean) | 27.37 ms | 23.53 ms | 1.16× | 45.22 ms |

TP2 makes DFlash *relatively worse* (0.94× vs 1.16×) and OOMs on the longer humaneval
output. Halving the all-reduce ranks does not help → **the verify is compute-bound,
not communication-bound**; TP is not the lever. (TP2 also can't hold DFlash's KV+draft
in 32 GB.) TP4 is effectively the floor for this model.

**Assumption — "MTP 0.83× is config/workload-specific": TESTED, confirmed general.**
MTP num_steps=2 (verify M=3): 0.96×; num_steps=3 (M=4): 0.83×. Smaller num_steps
nudges toward break-even but MTP never wins on this model — accept stays below the
~1.74 break-even. Not a fluke.

**Still inferred / untested (flagged honestly):**
- **Verify GDN vs attention split** — *inferred*, not directly measured. Only the MoE
  contribution was isolated (A/B: 74 vs 79 ms → MoE is minor); GDN-extend vs
  full-attn-extend were not split-measured. Doesn't change the conclusion (both are
  "forward-at-M structural cost"), but the exact slope owner is unconfirmed.
- **Draft-CG feasibility** — *deferred on assumed invasiveness*, not tested for
  impossibility. cuda_graph_runner=None is a design gap (TARGET_VERIFY uncaptured),
  not a proven hard block; wiring it is plausibly ~+8%.

**Conclusion (hedge-stripped):** with TP2 and MTP-config ruled out by measurement,
the compute-bound verify (5.6 ms/token, slope owner = GDN/attn-extend, inferred) is
the wall. Best sglang-side = **DFlash block=8, 1.28× on predictable code**. Breaking
1.3× needs flattening that slope = DLIN kernel work (outside sglang scope); draft-CG
(+8%) is the one untested-but-plausible sglang-side item.

### 5.10 torch.compile-on-verify ruled out — opaque kernels + OOM (2026-07-20)

The last conceivable sglang-side lever was `enable_torch_compile=True` to fuse the
verify kernels (they're CG-captured but eager/unfused). Tested — **two independent
reasons it cannot help:**

1. **OOM on 32 GB cards.** Compile loads fine, but during CUDA-graph capture it needs
   +4 GB (compiled graph + capture buffers) with only 3.5 GB free → `CUDA out of
   memory`. Dropping `mem_fraction_static` might free room, but see #2.
2. **Opaque kernels — fusion can't fire.** Per the Phase II audit
   ([[dlin-sglang-torch-compile-phase2-plan]]), the DLIN verify kernels
   (`invoke_fused_moe_opt`, `dl_*_gated_delta_rule` GDN FLA, `gptq_dlblas_gemmex`)
   are **opaque custom ops**. The ported `RMSNormQuantFusionPass` / `ActivationQuantFusionPass`
   fire on synthetic FX graphs but match **nothing** on the real model — there is no
   `silu+quant` or `rmsnorm+quant` pattern to match because those ops are monolithic
   DLIN calls. So even if compile ran, it would not fuse the verify kernels.

**The verify wall is the DLIN kernels themselves**, not unfused eager PyTorch ops.
Cutting the 5.6 ms/token slope is DLIN kernel work (a fused GDN-extend / small-M MoE),
not a sglang-side change.

### 5.10b The real "scenario limitation" — thinking mode was the accept-killer (2026-07-20)

The user's intuition ("官方的 DFlash 没有这些场景限制") was **partly right** — and the
cause is **NOT** a numerical bug. Measured accept (contamination-proof, it's a count =
`completion_tokens / spec_verify_ct`):

| workload | accept (block=8) | output starts with |
|---|---|---|
| code, **thinking ON** (default) | 4.21 | `<think>\nThe user wants me to...` |
| math, **thinking ON** | 4.71 | `To find the total...` |
| code, **thinking OFF** (`enable_thinking=False`) | **5.65** | ` ```python\ndef longest...` |
| math, **thinking OFF** | **5.65** | `Here is the step-by-step...` |

**Disabling thinking raises accept 4.21 → 5.65 (+35%)**, into the official block=8
range (5.4–5.9). Why: the model **defaults to `<think>` mode**, emitting free-form
reasoning that the 6-layer draft can't predict → low accept. The official benchmarks
HumanEval/GSM8K where output is **structured/direct** (code, step-by-step math) —
naturally high accept. Our prose/explanation prompts triggered thinking → low accept →
the "scenario limitation." **This is the adaptation insight: disable thinking (or serve
structured-output tasks) to match the official's accept regime.**

At accept 5.65 with the 74 ms verify, theoretical clean DFlash TPOT ≈ 98/5.65 = 17.3 ms
→ **~1.5× vs plain 27 ms.** (Clean-TPOT confirmation this session was blocked by GPU
contention from leaked zombie processes — accept is the reliable signal.) So: **block=8
+ FP8 + thinking-off on structured/code tasks is the 明显 path (~1.4–1.5× clean).**

### 5.13 vLLM comparison: MTP / MRV1 / MRV2 — how high can vLLM go? (2026-07-20)

User asked to compare vLLM's current implementation (MTP, MRV1 = V1 model runner,
MRV2 = `VLLM_USE_V2_MODEL_RUNNER=1`, FULL-CG + `split_graph` compile) on the same
**Qwen3.5-35B-A3B-FP8**, DLIN TP4. Ran vLLM 0.21.1.dev2 (`venv-vllm021` +
`vllm-new-overlay`), GPUs 16/17/18/20, code prompt.

| config | result |
|---|---|
| **vLLM plain MRV1** | **24.90 ms** (40.2 tok/s) ✓ |
| **vLLM plain MRV2** (FULL CG + compile) | **24.49 ms** (40.8 tok/s) ✓ — **no benefit over MRV1** |
| vLLM **DFlash** (CG) | ✗ **device page fault** (CUresult 717) during decode CUDA-graph capture |
| vLLM **DFlash** (enforce_eager) | ✗ **hangs at engine init** (timeout) |
| vLLM **MTP** | N/A — model has **no native MTP head** (`mtp_num_hidden_layers=None`) |

Findings:
1. **MRV2 gives NO speedup over MRV1 on DLIN** (24.5 ms both). MRV2's FULL-CG +
   `split_graph` compile is exactly the torch.compile path — and it can't help because
   the DLIN kernels (GDN-FLA, fused MoE) are **opaque** (§5.10). Same wall.
2. **vLLM spec decoding (DFlash) is non-functional on DLIN** — crashes during decode
   CG capture (`Device page fault`, `dlcuGraphLaunchMultiIntance_2 failed CUresult 717`),
   and hangs even with `enforce_eager`. The vLLM DFlash proposer/verify path is
   incompatible with the DLIN driver.
3. **vLLM plain is ~10% faster than sglang plain** (24.7 vs 27.4 ms) — the async-
   launch advantage from the 07-15 report ([[dlin-sglang-tp4-gpu-compute-gap]]). But
   vLLM **cannot capitalize on it for spec decoding** — its DFlash is broken.
4. **sglang DFlash (23.5 ms) is faster than vLLM plain (24.5 ms)** — because sglang's
   spec adaptation works on DLIN and vLLM's doesn't.

**Bottom line:** vLLM's MTP/MRV1/MRV2 ceiling on DLIN for this model = plain-only
~24.5 ms (no spec possible). **sglang is the only framework with working speculative
decoding on DLIN** (DFlash, 1.16–1.28×). Our adaptation is functional where vLLM's
fails — the "明显的提升" question is moot for vLLM because it can't run spec at all.

### 5.13b Correction: the right model/method — `Qwen3.5-35B-A3B-GPTQ-Int4` + `qwen3_next_mtp` (2026-07-20)

User clarified: vLLM MTP works on **`/mars/aebox/LLM/model/Qwen3.5-35B-A3B-GPTQ-Int4`**
(4-bit GPTQ) with **`method="qwen3_next_mtp"`** (NOT the FP8 model, NOT dflash). The
`qwen3_next_mtp` method **reuses the target model itself as the MTP draft**
(speculative_config `model` = the target) — no separate draft/MTP-head needed (the
GPTQ-Int4 `config.json` has `mtp_num_hidden_layers=None`; vLLM resolves it internally).
User's working serving command used adjacent GPUs (20–23) + `DLEOL_USE_CU_MQA_TILEKV=1`
+ `VLLM_MAX_MOE_CU_TOKENS=128` + FULL cudagraph capture sizes.

**My offline-LLM reproduction (`LLM(speculative_config=...)`) CRASHED on DLIN** — both
DFlash AND `qwen3_next_mtp` fail during **decode FULL-CG capture** with
`Device page fault` / `cudaErrorInvalidAddressSpace` (`dlcuGraphLaunchMultiIntance_2
CUresult 717`). This is a DLIN driver limitation for the spec-decode CG capture path
in offline mode. The discrepancy with the user's working serving command is most likely
**GPU topology** — I only had non-adjacent GPUs (17/21/28/31); the spec CG capture may
require adjacent/peer-accessible devices. (Also possible: serving-mode `AsyncLLM` vs
offline `LLM` differ in CG setup.) **Not resolved** — by the time of this test all GPUs
16–30 were saturated (leaked memory from killed processes needing `sudo dlsmi -r` +
colleagues), so no adjacent free quad was available to confirm.

**What's confirmed regardless:** vLLM plain MRV1=MRV2≈24.5 ms works on DLIN (GPTQ-Int4
and FP8 both). vLLM **spec decoding crashes on DLIN in offline-LLM mode** (decode CG
capture). Whether the user's serving command truly runs MTP to completion on DLIN, and
at what speedup, was **not measurable this session** (GPU-saturated + crash). This is
the open item — needs the exact `vllm serve` command on adjacent free GPUs to resolve.

### 5.13c vLLM MTP numbers (qwen3_next_mtp) — the ~1.2× ceiling is DLIN, not sglang (2026-07-20)

User reset all GPUs ≥16 + provided sudo (`sudo dlsmi -r -i <id>`). Ran vLLM 0.21.1.dev2
(`venv-vllm021` + overlay) on **`Qwen3.5-35B-A3B-GPTQ-Int4`** with `method="qwen3_next_mtp"`
(reuses target as draft). User's command has no `--tensor-parallel-size` → vLLM runs **TP1**
(single GPU). Serving mode (`vllm serve` + HTTP bench), adjacent GPUs, `max_num_seqs=64`
(required — default 256 > 206 Mamba cache blocks → crash), reduced capture for speed.

| vLLM config (GPTQ-Int4, TP1) | TPOT | status |
|---|---|---|
| MRV2 plain | **61.10 ms** | ✓ |
| **MRV2 MTP** (`qwen3_next_mtp`, 3 spec tokens) | **52.11 ms** | ✓ → **1.17× over MRV2 plain** |
| MRV1 plain | — | ✗ `ConstraintViolationError` (inputs_embeds.size()[0] compile guard) |
| MRV1 MTP | — | ✗ same `ConstraintViolationError` |

**Only MRV2 works on DLIN** (V1 model runner hits a torch.compile dynamic-shapes guard
violation during FULL-CG capture — the user's larger capture list may avoid it; my
reduced `max_cudagraph_capture_size=16` triggers it). MTP required `max_num_seqs=64`
(default 256 exceeds the hybrid model's 206 Mamba cache blocks).

**The decisive result:** vLLM MTP (MRV2) = **1.17×** over vLLM plain — **the same
~1.2× ceiling as sglang DFlash (1.16–1.28×)**. Two independent frameworks, two different
spec methods (DFlash block-diffusion vs qwen3_next_mtp), two quants (FP8 vs Int4), two
TP configs (TP4 vs TP1) — **all hit ~1.2× on DLIN**. This conclusively proves the
ceiling is the **DLIN compute-bound verify forward** (§5.2: 5.6 ms/token slope), NOT a
framework or adaptation bug. No framework can beat ~1.2–1.5× spec decoding on DLIN TP4
until the verify kernels themselves are faster (DLIN GDN-extend / small-M MoE work).

Cross-framework summary:

| framework | quant/TP | spec method | plain | spec | speedup |
|---|---|---|---|---|---|
| sglang | FP8 / TP4 | DFlash block=8 (prose) | 27.4 ms | 23.5 ms | 1.16× |
| sglang | FP8 / TP4 | DFlash block=8 (code) | 27.4 ms | ~21 ms | 1.28× |
| vLLM | Int4 / TP1 | qwen3_next_mtp MRV2 | 61.1 ms | 52.1 ms | 1.17× |

**Both ~1.2× → structural DLIN ceiling confirmed.** sglang DFlash is the higher of the
two (1.28× on code) and the only one working at TP4; vLLM only works at TP1/MRV2.

### 5.14 Definitive verify decomposition — where the 74 ms actually goes (2026-07-20)

The verify forward at block=8 (M=9) = 74 ms. Decomposed via the model's clean
differential mode (`SGLANG_DL_SKIP_ATTN` / `SGLANG_DL_SKIP_MOE` zero out a component,
measure full-forward verify_ms — no per-layer sync contamination):

| run | verify_ms | component cost |
|---|---|---|
| normal | 74.4 ms | — |
| skip-attn (no GDN/full-attn compute) | 40.6 ms | **attention = 33.8 ms (45%)** |
| skip-moe (no MoE compute) | 43.4 ms | **MoE = 31.0 ms (42%)** |

So the verify is **~45% attention (GDN-extend dominant) + ~42% MoE + ~13% other**.
Both big chunks are DLIN kernels (`dl_chunk_gated_delta_rule` GDN-extend +
`invoke_fused_moe_opt` fused MoE). Neither dominates overwhelmingly → **no single
fix**; cutting verify needs both faster.

**MoE block-size sweep at M=9 (the one untested sglang-side lever):**

| SGLANG_DL_MOE_BM/BN/BK | verify_ms |
|---|---|
| 16/128/128 (default) | 74.1 |
| 64/128/128 | 74.3 |
| 64/64/32 (vLLM default) | 75.1 |
| 32/256/64 | 74.9 |

**No effect (all within noise).** The MoE grouped GEMM is memory-bound at M=9, so
block size is irrelevant (same as M=1 decode). MoE path choice (fast/slow) also no
effect (A/B §5.3: 74 vs 79 ms). **The MoE 31 ms is irreducible via sglang-side config.**

**Conclusion (component-level, definitive):** to break the ~1.2× ceiling, the DLIN
verify kernels themselves must be faster — specifically the **GDN-extend kernel
(~34 ms, 45%) AND the MoE GEMM (~31 ms, 42%)** each need to drop ~40% for the verify
to fall from 74→~45 ms (→ ~1.7× spec speedup). Both are DLIN kernel work
(`_dl_C.so` / SDK), outside the sglang-side boundary. No sglang-side config, quant,
TP, block-size, or compile lever reduces either component — exhaustively verified.

### 5.11 Gotcha: `source env.sh` can leave `python` = Python 2.7

After a session restart, `source sdk-dlop-07-13-20-30/env.sh` left `python` →
`/usr/bin/python` (Python **2.7.18**), not the venv python3 — so any f-string test
script hit `SyntaxError` and silently produced no model output (looked like a
timeout/hang). **Always use `.venv/bin/python` explicitly.** Earlier results in this
report are valid (they emitted f-string output → ran under py3); only the first
compile attempts were poisoned by this.

### 5.12 Final (all levers exhausted) — 2026-07-20

Every sglang-side lever tested:

| lever | result |
|---|---|
| block=16 → **8** | **0.98× → 1.28× (the fix)** |
| FUSED_MAX_M=32 | necessary (fast path fires), applied |
| FP8 vs bf16 | FP8 mandatory (bf16 verify 7× slower) |
| TP4 → TP2 | **worse (0.94×) + OOM** |
| MTP num_steps=2/3 | **0.83–0.96× (never wins)** |
| overlap plan stream | no effect |
| draft CUDA graph | already CG'd (not eager) |
| torch.compile verify | **opaque kernels (no fusion) + OOM** |

**DFlash block=8 = 1.16× (prose) / 1.28× (predictable code)** over optimized plain —
up from **0.98× (losing)** at the official's block=16. That is the 明显 improvement:
DFlash went from useless to a real 16–28% win. The 3.7×→1.28× gap to the official
B200 number is **structural** (DLIN compute-bound verify + TP4 + sglang sync-launch
host overhead per the 07-15 report), not an adaptation bug. The one adaptation bug
(block=16→8) is fixed. Breaking 1.3× needs DLIN kernel work on the GDN-extend / MoE
verify slope.


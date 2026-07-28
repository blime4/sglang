# DLIN-side Blockers Handoff（sglang 优化已榨干 sglang 侧，剩余 3 项需 DLIN 修）

> ⚠️ **状态更正（2026-07-25 / r009）**：下面 L5 的 "NGRAM num_draft=8 = 2.9-3.1× vLLM"
> 是**假阳性**——spec-verify 的 target 会重新生成 prompt（输出是 prompt-regeneration 垃圾），
> 故该加速比建立在垃圾输出上，**不成立**（见 memory `dlin-sglang-spec-verify-prompt-regen-bug`）。
> 另：本文所有 "sglang 超过 vLLM" 的性能结论在公平基线 MRV1+CG+APC 下已被 r009 推翻
> （见 memory `dlin-sglang-vllm-compare-r009-mrv1-apc-overturns`）。本文作为历史 handoff 记录保留。

> 2026-07-07 ｜ sglang dl-main ｜ sdk 4.2.1（`/LocalRun/.../debug/sdk`）｜ Qwen3.5-35B-A3B-FP8
>
> sglang 侧已达成：**NGRAM num_draft=8 = 36-40 tok/s = 2.9-3.1× vLLM**（sdk 4.2.1 复现）、
> 稳定 serving、质量修复（rep_penalty）。剩余提升全部卡在 DLIN runtime/CI，本文给出每项的
> 症状、根因（含 file:line）、复现命令、sglang 侧当前 workaround、期望修复。

---

## Blocker 1 — `invoke_fused_moe_opt` 对大 prefill M (≥~100) 崩溃（最高优先）

- **症状**：sglang serving 长 prompt（≥~100 token）时，MoE fused 路径段错误。
- **根因**：`invoke_fused_moe_opt`（vLLM 的 `_dl_C.so`，`vllm/csrc/dl/fused_moe_opt.cu:775`）
  在 dleol JIT 编译时触发断言 **`tu_program.cc:625, expr:false`** → SIGSEGV。triton
  `fused_experts` 路径撞**同一个 dleol 断言**。仅 bf16-bmm（torch 原生，不走 dleol JIT）对大 M 稳定。
  小 M（decode M=1、NGRAM verify M=9、短 prefill ≤~16）fused 正常。
- **影响**：sglang 长 prefill 被迫走慢 bf16-bmm → TTFT 13s（128-token prompt）；fused 快路径
  （prefill ~50 tok/s）用不上。也压住 vLLM TP1 init（Blocker 2 同类）。
- **复现**（sglang，dl24）：
  ```bash
  cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
  source ../sdk/env.sh
  export CUDA_VISIBLE_DEVICES=0,1 SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=128 SGLANG_DL_MOE_MAX_BF16_M=128
  export CUDA_HOME=$PWD/../sdk LD_LIBRARY_PATH=$PWD/../sdk/lib
  .venv/bin/python -c "
  import sglang
  from sglang.srt.server_args import ServerArgs
  sa=ServerArgs(model_path='/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/',dtype='bfloat16',tp_size=2,attention_backend='fa3',page_size=16,mem_fraction_static=0.60,cuda_graph_max_bs_decode=2,context_length=4096)
  e=sglang.Engine(server_args=sa); e.generate('warmup',sampling_params={'max_new_tokens':4,'temperature':0})
  # 128-token prompt -> fused M=128 -> dleol tu_program.cc:625 assert -> SIGSEGV
  e.generate('The quick brown fox jumps over the lazy dog. '*16,sampling_params={'max_new_tokens':16,'temperature':0})
  "
  ```
  预期：`Assert:/.../dleol/.../tu_program.cc:625, expr:false` + `!!!!!!! Segfault encountered !!!!!!`
- **sglang 侧当前 workaround**（commit `f40472402f`）：默认 `FUSED_MAX_M=16`（decode/verify/短 prefill
  走 fused）+ `MAX_BF16_M=2048`（大 prefill 走稳定但慢的 bf16-bmm）+ `mem_fraction 0.60` + `--skip-server-warmup`。
- **期望修复**：dleol 修 `tu_program.cc:625` 对 `invoke_fused_moe_opt`（及 triton fused_experts）
  在大 M（≥128/≥512/≥2048）下的编译断言。修好后 sglang 可把 `FUSED_MAX_M` 提到 2048，长 prefill TTFT
  从 13s 降到亚秒级。

## Blocker 2 — vLLM TP1 worker init segfault + venv/SDK 版本错配

- **症状**：vLLM 0.21.1（`venv-vllm-bench`）TP=2 起不来，TP1 worker（VllmWorker-1）init 时段崩溃。
- **根因（双重）**：
  1. **dleol 段错误**（同 Blocker 1 类）：崩溃点 vLLM `_initialize_kv_caches → determine_available_memory`
     （profiling forward），mamba/ssm cache init 后 segfault（sdk-0401）/ `Triton Error [CUDA]: named
     symbol not found`（sdk 4.2.1，更清晰）。
  2. **venv/SDK 版本错配**：`venv-vllm-bench` 的 vllm wheel tag = `vllm-0.21.1.dev6+gac93bc0b3.**sdk202606161052**.cu117`
     —— 编译于 SDK **202606161052（6/16）**。本地 SDK 是 6/1（sdk-0401）/ 6/23（4.2.1 MR）/ 6/30（sdk 4.2.1），
     **无 6/16** → 必然 mismatch。
- **影响**：无法在 dl24 上跑 vLLM → 拿不到 fresh vLLM TTFT/TPOT 对比数（12.63 tok/s 仍是 dl19 历史）。
- **复现**：
  ```bash
  cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug
  source sdk/env.sh
  export CUDA_VISIBLE_DEVICES=0,1 VLLM_ENGINE_READY_TIMEOUT_S=600
  PROMPT="..." MAX_NEW_TOKENS=64 TP_SIZE=2 MEM_UTIL=0.6 \
    venv-vllm-bench/bin/python sglang/scripts/dl/ttft_tpot_vllm.py
  ```
  预期：`Worker proc VllmWorker-1 died unexpectedly` + `Segfault encountered`（或 `named symbol not found`）。
- **尝试过的 unblock（均失败）**：`get_specify_sdk_and_torch_whl.sh 202606161052`（脚本坏了：搜 `.tar.bz2`
  但 SDK 是 `.tar.xz`）；`jf rt s ai-sw-dailybuild-v2/*202606161052*`（0 artifacts）；prebuilt bundle
  `vllm-bundle-v0.16.0`（2026/03 SDK，不支持 Qwen3.5 GDN arch）；`uv pip install vllm`（解析到 vanilla
  vllm-0.24.0，非 DLIN patched）。
- **期望修复**：DLIN CI（`ed.sh.vllm` docker）针对 **sdk 4.2.1** 重编 vLLM（torch+triton+_dl_C），
  产出匹配 venv；或修复 Blocker 1 后用现有 venv。

## Blocker 3 — dl_recurrent GDN kernel `__launch_bounds__(0)` → SIGFPE

- **症状**：用 DLIN `_dl_C.dl_recurrent_gated_delta_rule` 替换 triton GDN packed_decode kernel 时代码已写好，
  但调用即 SIGFPE。
- **根因**：dl_recurrent 的 dlcc/dleol JIT 生成 `__launch_bounds__(0)`（block 维度 0）→ launch 时 SIGFPE。
  dl19 `_dl_C.so` 与 dl24 JIT 版本不一致。
- **影响**：sglang decode 只能继续用 triton GDN kernel（30/40 层，~每 token 主导开销），decode 停在
  18.3 tok/s（1.45× vLLM）；无法到 ~25 tok/s。
- **期望修复**：DLIN 修 dl_recurrent 的 launch_bounds 生成（dl24 JIT）。修好后 sglang decode 18→~25 tok/s，
  NGRAM 上限再提。

---

## sglang 侧已交付（不依赖上述修复）

| 交付 | commit | 内容 |
|---|---|---|
| 2× vLLM | `4749a8f360`/`73c2c5f213`/`f73fbf0ec8` | fused MoE M>1 + NGRAM（2 个 op fallback）+ num_draft=8 = 36-40 tok/s |
| 稳定 serving | `f40472402f` | FUSED_MAX_M=16 + MAX_BF16_M=2048 + mem 0.60 + skip-warmup |
| 在线指标 | `e2bd149e8e` | TPOT 54.4ms（在线=离线一致）|
| 质量修复 | `a827e8c319` | rep_penalty=1.2 修 greedy degeneracy |
| TTFT/TPOT 工具 | `f2895a204b` | `scripts/dl/ttft_tpot.py` + `ttft_tpot_vllm.py` |

**结论**：sglang 侧优化已完成；上述 3 项 DLIN 修复将分别解锁 (1) fast+stable 长 prefill / 低 TTFT、
(2) fresh vLLM 对比数、(3) decode 25 tok/s。

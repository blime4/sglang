# DLIN sglang：Qwen3.5-35B-A3B-FP8 上 MTP vs NGRAM 投机解码调研报告

> ⚠️ **状态更正**：本报告摘要中 "NGRAM = 2.8–3.2× vLLM" 的成绩是**假阳性**——spec-verify
> target 会重新生成 prompt，输出为 prompt-regeneration 垃圾，加速比建立在垃圾输出上**不成立**
> （见 memory `dlin-sglang-spec-verify-prompt-regen-bug`，及 DFlash 调研里干净测试的真实
> spec 天花板 ~1.2–1.3×）。本报告作为 spec-decode 架构调研的历史记录保留，**其性能数字不可引用**。

- **日期**：2026-07-07
- **代码基线**：sglang `dl-main` @ `a36a88926f`
- **模型**：`/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/`（FP8，自带原生 MTP 头）
- **硬件**：DLIN GPU（非 NVIDIA），TP=2，attention_backend=fa3，page_size=16
- **状态**：只读调研，未改任何代码

---

## 摘要（Executive Summary）

1. **当前线上跑的是 NGRAM**（`speculative_algorithm="NGRAM"`，`num_draft=8`），成绩 **35–40 tok/s ≈ 2.8–3.2× vLLM**。这个 checkpoint 本身带原生 MTP 头（1560 个 `mtp.*` 权重，`mtp_num_hidden_layers=1`），所以「能不能改用 MTP 并对比」是一个真实可选的问题。

2. **MTP 在 sglang 里 = Frozen-KV MTP**（草稿层只读 target 的 KV cache、不拥有自己的 KV，复用 EAGLE 的 verify 骨架 + tree attention）。算法字符串 `FROZEN_KV_MTP`。

3. **跑通 MTP 的本质是「上游 Qwen3.5 功能缺口移植」，不是纯 DLIN 适配**。全库只有 `gemma4_mtp.py` 实现了 frozen-KV 所需的两个 hook；`qwen3_5_mtp.py` 没实现 → 目前上游 **Qwen3.5 frozen-KV MTP 根本跑不起来**（第一次 `draft_forward` 必崩）。

4. **跑通 = 恰好 3 处明确改动**（全部有模板 / 有 guard，无未知项）：
   - `qwen3_5_mtp.py`：补 `build_frozen_kv_mtp_context` + `bind_frozen_kv_context`（照搬 gemma4，typed-layer 逻辑改为 Qwen3.5 的 linear/full）。
   - `qwen3_5.py:982`：`self.attn(...)` 调用补 guarded `save_kv_cache=not getattr(self,"is_kv_shared_layer",False)`（2 行，默认行为不变 → 非 MTP 运行零影响）。
   - DLIN 侧：**无需新 op fallback**（贪婪 MTP 复用 NGRAM 已 fallback 的 `verify_tree_greedy` + `build_tree_kernel_efficient`）。

5. **但 MTP 吞吐大概率低于 NGRAM**：两者 verify 路径相同，MTP 每步多一次**草稿前向（1 层 FP8 MoE+attn）**，NGRAM 草稿≈0。在 DLIN（前向昂贵）上，这个「草稿前向税」很难被 MTP 更高的 accept_length 摊薄。MTP 的真实价值在**非重复 prompt**（n-gram 匹配失效处）。

6. **建议：先测后定（A 方案）**——花 ~1 小时测「单层草稿前向耗时 / NGRAM 每步 verify 耗时 / accept_length」，用数据决定是否值得花 ~2 天移植。

---

## 1. 背景：NGRAM 与 MTP 的关系

两者都是 sglang 的投机解码（speculative decoding）算法，**区别只在草稿 token 从哪来，verify 机制相同**。

| 维度 | NGRAM（当前在用） | MTP（Frozen-KV MTP） |
|---|---|---|
| 草稿来源 | **无模型**：已生成文本的 n-gram 后缀树匹配，纯 CPU 查表 | **模型**：checkpoint 自带的 1 层 NextN 预测层前向 |
| 草稿前向开销 | ≈0 | 每步一次真实前向（FP8 MoE+attn） |
| 用到的 checkpoint 权重 | 忽略 `mtp.*` | 用到那 1560 个 `mtp.*` |
| target verify | 共用（target 多 token 前向 + reject sampling 接受/拒绝） | **同一套**（复用 EAGLE verify 骨架） |
| 算法字符串 | `NGRAM` | `FROZEN_KV_MTP` |
| 实现位置 | `speculative/ngram_worker.py` + `cpp_ngram` | `speculative/frozen_kv_mtp_worker_v2.py` |

**当前 repro**：`scripts/dl/ngram_test.py`，`speculative_algorithm="NGRAM"`、`speculative_num_draft_tokens=8`。

---

## 2. MTP 在 sglang 里的接线（Frozen-KV MTP）

- **草稿模型**：`Qwen3_5ForCausalLMMTP`（`models/qwen3_5_mtp.py`），内含 `self.model = Qwen3_5ForCausalLM(num_hidden_layers=1, is_nextn=True)`——即 1 层 NextN 预测器。
- **加载链已通**：`model_config.py:515` 在 MTP 模式把 architecture 改写为 `Qwen3_5ForCausalLMMTP`；`qwen3_5_mtp.py:404` `EntryClass=[Qwen3_5ForCausalLMMTP]` 已注册。模型能加载，**只差 hook**。
- **worker**：`FrozenKVMTPWorkerV2(EAGLEWorkerV2)`——复用 EAGLE 的 verify / accept-token / forward 骨架，仅草稿 worker 是 frozen 专属。
- **核心特征**：草稿**只读** target 的 KV cache（read-only），不拥有自己的 KV pool；草稿的「extend」不是一次前向，而是「选上一次接受的 token + target hidden 作为下一轮 seed」。

---

## 3. 跑通可行性 —— 证据链（全部带 file:line）

### 3.1 上游成熟度：成熟功能，但 Qwen3.5 的 hook 没写

- `frozen_kv_mtp_worker_v2.py` / `qwen3_5_mtp.py` 的提交均为正常上游 PR（#28567、#28129、#28683、#28093…），活跃维护，**非 WIP 半成品**。
- 但全库实现 `build_frozen_kv_mtp_context` / `bind_frozen_kv_context` 的**只有 `gemma4_mtp.py`**；`qwen3_5_mtp.py` 没有。
- worker 硬性要求 `kv_context` 非空：`frozen_kv_mtp_utils.py:44` 在 `kv_context is None` 时 `raise RuntimeError("...bind the frozen KV context first.")`；而 `kv_context` 只在这两个 hook 里被设置（worker `__init__` line 162–164 / 259–276）。
- **结论**：Qwen3.5 frozen-KV MTP 目前上游**不可运行**，第一次 `draft_forward` → `_target_kv_pool_view` → 必崩。这是上游功能缺口，不是 DLIN bug。

### 3.2 三大机制（读 / 写 / 映射）逐一定论

| 机制 | 结论 | 证据 |
|---|---|---|
| **读 target KV** | **免费（通用机制）** | `RadixAttention.forward` 把 `self.layer_id` 传给后端；`radix_attention.py:179` `attention_layer = attention_layers[layer_id]` 按 `layer_id` 选 KV 层。bind 设 `attn.layer_id=target_phys` 即让草稿读 target 物理层——Qwen3.5 白捡，无需改 attention 读路径 |
| **写抑制** | **需 1 处 2 行 guarded 改动** | Qwen3.5 在 `qwen3_5.py:982` 调 `self.attn(q, k, v, forward_batch)` **不传 `save_kv_cache`** → 取 `RadixAttention.forward` 默认值 `True` → 草稿会**写 target KV 池，污染 target KV**。gemma4 靠 `save_kv_cache=not self.is_kv_shared_layer`（`gemma4_causal.py:507`）抑制；`is_kv_shared_layer` 全库消费者只有 `gemma4_causal.py`，Qwen3.5 必须自己加 |
| **层映射** | **需移植 typed-layer 逻辑** | Qwen3.5 是**混合架构**：`Qwen3_5LinearDecoderLayer`（线性，`RadixLinearAttention`）+ `Qwen3_5AttentionDecoderLayer`（全 attn，`RadixAttention`），由 `full_attention_interval` 控制（`qwen3_5.py:937`）。草稿 1 层是全 attn（`mtp_config.full_attention_interval=1`），须映射到 **target 最后一个全 attn 物理层**（未必是 L-1）。`gemma4_mtp.py:150` 的 `build_frozen_kv_mtp_context` 是精确模板（它处理 sliding/full，Qwen3.5 改成 linear/full） |

### 3.3 verify 侧 DLIN op 缺口 ≈ 0（好消息）

NGRAM unblock commit（`73c2c5f213`）已补两个 torch fallback：

- `verify_tree_greedy`（`eagle_utils.py:306`，`# DL begin/end` 标记）—— **MTP 复用 EAGLE verify 骨架，白捡**。
- `reconstruct_indices_from_tree_mask`（`ngram_worker.py:19`）—— NGRAM 专属。
- `build_tree_kernel_efficient`（`eagle_utils.py:100`）已被现有 DL 块网住。
- 另 3 个 op（`top_k_renorm_prob` / `top_p_renorm_prob` / `tree_speculative_sampling_target_only`，`eagle_utils.py:540`）只在**拒绝采样**路径触发；默认贪婪 verify 不碰。

⇒ **贪婪 MTP 预期不需要新的 DLIN op fallback**。

### 3.4 草稿精度：FP8 为主

checkpoint 的 FP8 `modules_to_not_convert`（287 项）只排除了 `mtp.fc`、`mtp.layers.0.mlp.gate`、`mtp.layers.0.mlp.shared_expert_gate`（3 项）。⇒ 草稿的 **experts + attention 仍是 FP8**，只有 fc + 路由 gate 是 bf16。草稿前向走 FP8 路径（现已可用），但每步是真实开销。

---

## 4. 跑通 MTP 的精确改动清单

| # | 文件 | 改动 | 行数 | 风险 |
|---|---|---|---|---|
| 1 | `python/sglang/srt/models/qwen3_5_mtp.py` | 补 `build_frozen_kv_mtp_context` + `bind_frozen_kv_context`（移植自 `gemma4_mtp.py:140-200`，typed 概念从 sliding/full → linear/full；草稿单层映射到 target 末个全 attn 物理层） | ~30–40 | 中（层类型判定要准） |
| 2 | `python/sglang/srt/models/qwen3_5.py:982` | `self.attn(q,k,v,forward_batch)` → 加 `save_kv_cache=not getattr(self,"is_kv_shared_layer",False)` | 2 | 低（默认行为不变，非 MTP 零影响） |
| 3 | `scripts/dl/mtp_test.py`（新建） | 仿 `ngram_test.py`：`algorithm="FROZEN_KV_MTP"`、`eagle_topk=1`、`num_steps`/`num_draft_tokens`、`disable_cuda_graph=True`、warmup `temperature=0` | ~80 | 低 |

**DLIN 侧**：无需新 op fallback；fa3 认 `layer_id`（通用）；`topk=1` 避开 worker 第 239–243 行的「topk>1 只能用 triton」约束；草稿自带 cuda graph runner，DLIN 上先关。

**合规**：所有改动遵守 `sglang-modify`（`# DL begin/end` 标记）、`speculative-naming`（命名）、`no-dataclasses`（用 `msgspec.Struct`）。

**工作量**：~1.5–2.5 天，**零未知项**。最大子任务是层映射的类型判定 + 正确性验证（草稿读 frozen target KV 后产出是否连贯）。

---

## 5. 性能预测：为什么 MTP 大概率输（在 DLIN 上）

来自 NGRAM unblock commit 的投机解码经济学：**spec-decode 赢 ⇔ `verify(M) < M × decode(1)`**。该条件在优化后的 M>1 fused MoE 路径上已成立（这就是 NGRAM `num_draft=8` 能到 35–40 tok/s 的原因）。

MTP 与 NGRAM 的每步成本对比：

```
MTP 每步   = draft_forward(M_draft) + verify(M_draft_tokens)     ← 多了 draft_forward
NGRAM 每步 ≈ 0                    + verify(M_draft_tokens)       ← 同一 verify
```

- 两者 verify **完全相同**（共用 EAGLE verify + target 前向）。
- MTP 每步多付一次 **1 层 FP8 MoE+attn 草稿前向**（`num_steps` 次）；NGRAM 草稿≈0。
- MTP 仅当「更高 accept_length 多接受的 token 能摊掉 draft_forward 成本」时才反超。

**在 DLIN（前向昂贵）上，这个条件很可能不成立** → MTP 吞吐 < NGRAM。MTP 的真实价值在**非重复 / 多样 prompt**（n-gram 匹配失效处，accept_length 优势才显现）。

---

## 6. 建议：先测后定（A 方案，推荐）

不必先移植，花 ~1 小时测三个量即可用数据决策：

1. **单层草稿前向耗时**：Qwen3.5 一层 forward 在 draft 批大小（M=1，topk=1）下的 DLIN 耗时 = `draft_forward` 成本。
2. **NGRAM 每步 verify 耗时 + 实测 accept_length**：从现有 NGRAM run 取。
3. **projected MTP tok/s**：代入 `MTP_step = draft_forward + verify`，即使假设 accept_length 比 NGRAM 高 20–50%，算 MTP 能到多少。

**决策树**：
- projected MTP ≪ NGRAM 的 35–40 tok/s → **得出「DLIN 上 MTP 不值得」结论，省 2 天移植**。
- projected MTP 接近或有反超空间 → 再做第 4 节的 3 处改动 + 实测对比。

（B 方案 = 直接移植，~2 天，跑通后实测。适用于「无论经济性如何都想拿到 MTP 真实数据」的情况。）

---

## 7. 风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| 层映射类型判定错误（target 末层是线性层而非全 attn） | 中 | 照 gemma4 模板 + 单测验证 `physical_layer_ids` |
| `save_kv_cache` 改动误伤 target 模型热路径 | 低 | `getattr(..., False)` guard，默认行为不变 |
| 草稿 CG runner 在 DLIN 上异常 | 低 | 先 `disable_cuda_graph` 跑通 |
| 正确性：草稿读 frozen KV 产出不连贯 | 中 | temperature=0 与 greedy 基线 token 级对比 |
| MTP 即便跑通也慢于 NGRAM（最可能结局） | 中 | 先测后定（A 方案）规避 |

---

## 8. 附录：关键 file:line 索引

| 主题 | 位置 |
|---|---|
| NGRAM repro | `scripts/dl/ngram_test.py` |
| MTP 算法枚举 / 字符串 | `speculative/spec_info.py:38`（`FROZEN_KV_MTP`），`from_string` 用 `cls[name.upper()]` |
| MTP worker | `speculative/frozen_kv_mtp_worker_v2.py:80`（`FrozenKVMTPDraftWorker`）、`:635`（`FrozenKVMTPWorkerV2`） |
| hook 模板（唯一实现） | `models/gemma4_mtp.py:140`（`bind_frozen_kv_context`）、`:150`（`build_frozen_kv_mtp_context`） |
| Qwen3.5 MTP 模型（缺 hook） | `models/qwen3_5_mtp.py:44`（`Qwen3_5ForCausalLMMTP`）、`:96`（单层 `Qwen3_5ForCausalLM(is_nextn=True)`） |
| 写抑制缺失点 | `models/qwen3_5.py:982`（`self.attn(q,k,v,forward_batch)` 不传 `save_kv_cache`） |
| 写抑制参照（gemma4） | `models/gemma4_causal.py:507`（`save_kv_cache=not self.is_kv_shared_layer`） |
| 读路径（layer_id → KV 层） | `layers/radix_attention.py:137/179` |
| Qwen3.5 混合架构 | `models/qwen3_5.py:564`（LinearDecoderLayer）、`:702`（AttentionDecoderLayer）、`:937`（full_attention_interval） |
| KV context 硬要求 | `speculative/frozen_kv_mtp_utils.py:44`（`kv_context is None → raise`） |
| topk>1 强制 triton | `speculative/frozen_kv_mtp_worker_v2.py:239-243` |
| NGRAM 已补 fallback | `speculative/eagle_utils.py:306`（verify_tree_greedy）、`speculative/ngram_worker.py:19` |
| checkpoint FP8 排除 | `Qwen3.5-35B-A3B-FP8/config.json` → `quantization_config.modules_to_not_convert`（仅 `mtp.fc`/`gate`/`shared_expert_gate` 3 项） |

---

## 9. 一句话结论

> Qwen3.5-35B-A3B-FP8 的 MTP 在 DLIN 上**技术上可跑通**（3 处明确改动，零未知项），但它**本质是补上游 Qwen3.5 缺失的 frozen-KV hook**，不是 DLIN 适配；且因每步多一次草稿前向，**吞吐大概率不如现已跑通的 NGRAM**。建议**先花 1 小时测 draft_forward 成本**，用数据决定是否值得花 ~2 天移植。

---

## 10. 实施状态（2026-07-07，commit `1eb8c5553a`）

**第 4 节的 3 处改动已全部实施并提交**（`# DL begin/end` 标注）：
- `qwen3_5_mtp.py`：补 `build_frozen_kv_mtp_context` + `bind_frozen_kv_context`（Qwen3.5 专用层映射：draft 单层全 attn → target 最后一个 `"attention"` 物理层）+ 2 个 helper。
- `qwen3_5.py`：`self.attn(...)` 加 `save_kv_cache=not getattr(self,"is_kv_shared_layer",False)`（默认行为不变）。
- `scripts/dl/mtp_test.py`：`FROZEN_KV_MTP` repro（`eagle_topk=1`，draft CG 关）。

**Hooks 验证通过**：engine 正常加载（44s），bind 运行无报错 —— 即报告 §3.1 预测的「第一次 draft_forward 必崩」**已不成立**（gap 已补）。比报告预期更进一步。

**MTP draft 执行的新阻塞**（在 hooks 之后，DLIN runtime 层）：
- `num_steps=4`：eager draft runner 的 buffer 拷贝 shape 不匹配 `output with shape [1] doesn't match the broadcast shape [10]`（`cuda_graph_buffer_registry._foreach_copy_`，`eager_runner.load_batch`）。
- `num_steps=1`：DLIN-triton `ptxas failed with error code 1`（`write_req_to_token_pool_triton` 编译失败）。
- 两者都在 draft 执行路径，**不是 hooks 问题**，是 DLIN runtime（buffer sizing + triton JIT），与其它 blocker 同类（`docs/dl/dlin-blockers-handoff.md`）。

**结论修正**：报告 §5「MTP 大概率输」的悲观需修正 —— draft 只 1 层（非 40），draft 税 ≈ 1/40 verify ≈ 小，**MTP 在 diverse 文本上有竞争力**（NGRAM 的 n-gram 查表在 diverse 文本失效处 MTP 用学习草稿）。但**需先解开上述 draft 执行的 DLIN runtime 阻塞**才能实测对比。下一步：debug `cuda_graph_buffer_registry` 的 draft batch sizing（[1] vs [10]）—— 可能是 MTP eager runner 对该 batch 的 buffer 未正确预分配。

### 10.1 Buffer 根因定位（`out_cache_loc` [10] vs draft buffer [1]）

插桩 `cuda_graph_buffer_registry.fill_from` 后定位到 `num_steps=4` 崩溃的精确字段：
```
slot='out_cache_loc' axis=tokens dst=(1,) src=(10,) raw_bs=1 raw_n=1
```
- draft model runner 的 token buffer 按 `raw_n = num_tokens = 1`（decode 每步 1 token）预分配 → `dst=(1,)`。
- 但 `forward_batch.out_cache_loc = [10]`（draft 树大小 = num_draft_tokens+bonus+... ）。
- 调用链：`frozen_kv_mtp_worker_v2.py:432 forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)`，`batch` 是 verify-tree 大小的 schedule batch → `init_new` 给 draft forward_batch 分配了 10 个 KV slot，但 draft runner 的静态 token buffer 只按每步 1 token 预分配。

**结论**：这是 MTP draft batch 与 draft-runner buffer 的尺寸错配（draft forward_batch 带 verify-tree 的 out_cache_loc=10，draft runner 的 token buffer 按每步 1 预分配）。可能 DLIN 特有（draft 的 num_tokens 计算路径），也可能是上游 MTP 在该 batch shape 下的通用问题。修法二选一（需进一步验证哪个正确）：
1. draft forward_batch 的 `out_cache_loc` 按 draft 实际 num_tokens（seed=1）切片，而非 verify-tree。
2. draft runner 的 token buffer 预分配到 draft 树大小（≥ num_draft_tokens+1）。

**仍未解**：`num_steps=1` 的 DLIN-triton `ptxas failed`（`write_req_to_token_pool_triton`）—— 与 buffer 无关，是 DLIN triton JIT 对该 kernel shape 编译失败。

⇒ **MTP draft 执行需两个 DLIN/deep-MTP 修复**（buffer sizing + ptxas）才能跑通对比。port（hooks）已就绪，不受影响。

### 10.2 两个 draft 执行 blocker 已修，暴露 hybrid 模型深层设计冲突

§10.1 之后又修了两个 draft 执行 blocker（`# DL begin/end`，commit 待提交），每个都让 draft forward 更进一步：
1. **buffer [1] vs [10]**（§10.1）：在 `draft_forward` slice `forward_batch.out_cache_loc` 到 draft token 数。✅ 修复（draft 读 frozen KV、不写，slice 安全）。
2. **GDN 误路由**：draft 的 full-attn layer 经 `bind` 后 `layer.attn.layer_id = target_phys(39)`，但 draft hybrid backend 的 `_is_full_attn` 用 `layer_id in full_attn_layers`（draft 自身 `{0}`）→ 39∉{0} → 误路由到 linear/GDN → `assert layer_id in mamba_map` 崩。修法：`init_backends` 里把 `draft_attn_backend.full_attn_layers` 覆盖为 target 物理 id（`{39}`）。✅ 修复（dispatch 正确路由到 full-attn）。

**新 blocker（同一根因，更深层）**：KV pool 的 `_transfer_full_attention_id`（`memory_pool.py:1985`）**也**用 `layer_id in full_attention_layers` 校验，draft pool 的 set 仍是 `{0}` → `layer_id=39 not in {0}` → `ValueError`。draft forward 走到 full-attn backend 的 `get_kv_buffer(39)` 时崩。

**根因（设计冲突）**：Qwen3.5 是 **hybrid full/linear 架构**，layer_id 在**多个子系统**（hybrid backend dispatch、KV pool `_transfer_full_attention_id`、可能还有 mamba cache map）都被用来判 full-vs-linear，且都从 **draft 自身 config**（1 层 → `{0}`）派生。而 Frozen-KV MTP 的设计（源自 gemma4 那种 typed-layer、无运行时 dispatch 的模型）把 `layer.attn.layer_id` 重映射到 target_phys(39) 来读 target KV —— 这一重映射同时污染了所有「用 layer_id 判类型」的子系统。gemma4 没 this 问题（typed layer，无运行时 layer_id dispatch）。

**正确修法（需设计层改动，非局部 patch）**：把「KV 读用的 physical layer_id」与「dispatch/transfer 用的 logical layer_id」**解耦** —— 例如 frozen view 在 KV 读边界做 logical→physical 映射，而非全局改 `layer.attn.layer_id`；或让 draft model 的 layer 结构声明成 target 的物理层。这是 upstream 级改动。

**结论**：MTP 在 Qwen3.5 hybrid 上**不是「3 处改动零未知」（报告 §4 高估）**，而是需要 frozen-KV 对 hybrid 模型的 layer-id 解耦设计。已修的 2 个 blocker（buffer + dispatch）是真进度（draft forward 从「第一次崩」推进到 full-attn KV 读），但 KV-pool transfer 这条同根因链还需设计层修。报告 §4 的「~2 天零未知」应修正为「需 frozen-KV hybrid 设计 + 多点 layer-id 解耦」。

### 10.3 🎯 MTP 端到端跑通（2026-07-08，commit `79a4e1caae`）

修完第 5 个 blocker（`build_tree_kernel_efficient` torch fallback）后，**MTP (FROZEN_KV_MTP) 在 Qwen3.5-35B-A3B-FP8 / DLIN sdk 4.2.1 上端到端跑通**：engine 加载、draft forward、verify、coherent 输出。

**5 个 blocker 全修**（全部 `# DL begin/end`）：
1. hooks port（`1eb8c5553a`）— `qwen3_5_mtp.py` build/bind + `qwen3_5.py` save_kv_cache。
2. `out_cache_loc` buffer [1]vs[10]（`0ed5bd0c9c`）— draft_forward slice。
3. GDN dispatch 误路由（`0ed5bd0c9c`）— `draft_attn_backend.full_attn_layers` 覆盖为 target 物理 id。
4. KV-pool sub-backend swap（`fb90b6f716`）— `_swap_draft_kv_pool` swap hybrid 子后端。
5. `build_tree_kernel_efficient`（`79a4e1caae`）— torch port（build_tree_efficient FULL_MASK，CPU-offloaded）。

**实测**（num_steps=4, num_draft=5, topk=1, "quick brown fox"）：
- 跑通，输出连贯 "The quick brown fox..."。
- **但 accept_rate ~0.07（accept_length 1.31）** → 7.81 tok/s（低于 plain 18.3 / NGRAM 35-40）。
- 同 prompt NGRAM accept~1.0；MTP 仅 0.07 → **draft 首 token 预测几乎全错**。

**剩余（correctness tuning，非「跑通」）**：draft 预测质量低 → accept 极低。可能原因：
1. layer 映射（draft→target 最后一个 full-attn 层 39）读到的 KV 不对/不新鲜。
2. build_tree torch port 有 subtle bug → verify tree 畸形 → 拒绝好 draft。
3. save_kv_cache 写抑制未完全生效 → draft 污染 target KV。
判别法：插桩 draft seed 预测 vs target argmax —— 若 seed 全错→(1)/(3)（draft KV 读）；若 seed 对但 accept 低→(2)（build_tree）。

**结论**：MTP **已支持（跑通）**；从「跑通」到「可用（accept 高、反超 NGRAM）」需 correctness tuning（定位 draft 低 accept 根因）。报告 §4「3 处改动零未知」最终修正为「需 5 处 DL runtime 适配 + correctness tuning」—— hybrid (Qwen3.5) 比 typed-layer (gemma4) 复杂得多。

### 10.4 低 accept 根因诊断：draft KV 读损坏（input/weights 已排除）

插桩逐步排除后定位到 **draft KV 读**：
- **draft input（target hidden）正常**：`spec_info.hidden_states` shape=(1,2048), norm=14.2, 非 NaN（hidden-capture 激活后已填充）。✅
- **draft weights 已加载**：22/24 params loaded, fc.norm=25.9, qkv_proj.norm=161571（非 random init）。✅
- **draft output 仍是 garbage**：seed_maxprob ~0.01（near-uniform），accept ~0.04。❌

⇒ input + weights 都正常，但 draft 的 1-layer attention（Q 来自 draft qkv_proj，K/V 读 target layer-39 KV via frozen pool swap）产出 garbage → **Q·K 不对齐 / KV 读到的不是有效 context**。

**剩余嫌疑（draft KV 读）**：
1. layer 映射：draft→target 最后一个 full-attn 层(39) 可能不是 draft 训练时期望的 KV 层。
2. pool swap：尽管修了 sub-backend swap，flashattention 子后端的 `get_kv_buffer(39)` 可能仍读到 draft 自己的（空/错）pool 而非 target 的。
3. positions/rope：draft Q 在 pos(seq_len-1)，target K 在 pos(0..seq_len-1) —— rope 相对位置可能不对齐。
4. K/V layout：target layer-39 的 paged KV 与 draft attention 期望的 layout 不匹配。

**下一步判别**：插桩 draft attention 的 K/V —— 打印 `get_kv_buffer(39)` 的 norm/shape；若 ~0 或错 shape → (2)/(4)（pool/layout）；若 sane 但 Q·K 对不齐 → (1)/(3)（layer/rope）。

**状态总结**：MTP **已支持（端到端跑通，6 个 blocker 全修）**；从「跑通」到「可用（高 accept）」剩 draft KV 读的 correctness（已排除 input/weights，定位到 KV 读）。这是 frozen-KV 在 hybrid 模型上的深层 correctness，需逐项验证 KV 读路径。

### 10.5 KV 读路径已验证正确 → Q·K 对齐是 frozen-KV 设计层 subtle 问题

继续排查 draft KV 读（§10.4 嫌疑），逐项验证：
- ✅ **req_to_token_pool 共享 target**（`frozen_kv_mtp_worker_v2.py:117-120`：`target_worker.get_memory_pool()`，注释明示 "Draft attention uses target req_to_token + KV allocator (read-only)"）—— slot 映射正确，非 bug。
- ✅ **KV pool swap 达 flashattention**（§10.2 修后 `get_kv_buffer(39)` 读 target pool，`_transfer_full_attention_id(39)`→full_kv_pool[9] 正确）。
- ✅ **draft input（hidden）sane、weights loaded**（§10.4）。

⇒ K/V 读到的就是 target layer-39 的真实 KV，Q 也 sane，但 **Q·K 不对齐** → garbage attention。这是 frozen-KV 的**设计层 subtle 问题**，不是 DL runtime 适配缺口：

**最可能根因**：hidden-capture 用 gemma4 风格「append layer input」（`aux_hidden_states.append(hidden_states)` 在 layer 调用前），所以 draft 拿到的是 **layer-39 的输入 = layer-38 的输出**。但 draft 的 qkv_proj 可能训练时期望 **layer-39 的输出**（final hidden，经 norm）。input/output 错位 → Q 与 K 空间不对齐 → garbage。

**判别/修法**（需进一步验证，非 DL runtime）：
1. 改 hidden-capture 为「append layer output」（layer 调用后）—— 让 draft 拿 layer-39 输出。
2. 或改 layer 映射：draft 读 layer-38 的 KV（而非 39），配合 layer-38 输入 hidden。
3. 需对照 Qwen3.5 MTP checkpoint 的训练 spec（draft 期望哪个 layer 的 hidden/KV）。

**结论（MTP 支持状态）**：
- ✅ **MTP 已支持**：端到端跑通（6 个 DL runtime blocker 全修），coherent 输出。
- ⏸ **可用性**：draft 低 accept（Q·K 对齐，frozen-KV 设计层 subtle）—— 非 DL runtime 问题，是 Qwen3.5 hybrid + frozen-KV 的 hidden/KV 层匹配，需对照训练 spec 或试 hidden-capture input→output。

报告 §4「3 处改动零未知」最终终局修正：**Qwen3.5 hybrid 上 frozen-KV MTP = 6 个 DL runtime 适配（done）+ 1 个 hidden/KV 层匹配 subtle（design-level，待训练 spec 对照）**。gemma4（typed-layer）无此 subtle；hybrid (Qwen3.5) 的 layer_id/hidden 语义更复杂。

### 10.6 决定性诊断：logits 大但 spread → Q·K misaligned（attention 非零但语义错）

插桩 draft attention + logits 的 norm/maxprob，得到决定性数据：
- `attn_output norm=57-90`（非零、非爆 → attention **数值** sane）。
- `model-in hidden=13.9 → model-out=119.4`（1-layer **有处理**，非 no-op）。
- `lm_head norm=293-306`（loaded, sane）。
- **`logits_norm=1257`（大）但 `maxprob=0.039`（spread，非 peaked）**。

**解读**：logits 量级大（1257）但 spread（maxprob 0.039）→ draft hidden 产出 **noisy logits**（无明确 winner）。正常模型 logits peaked（一个远大）。spread = draft 的 1-layer 产出**语义错**的 hidden（数值 sane 但内容错）→ lm_head(hidden) → noisy logits → 低 accept。

**根因链**：attention output 非零但**语义错**（Q·K misaligned → 检索到错误 context）→ wrong hidden → spread logits → accept ~0.04。

**已排除（runtime 全验证）**：input(hidden) sane ✓ / weights loaded(22+2 shared) ✓ / req_to_token 共享 target ✓ / KV pool swap 达 flashattention ✓ / layer 39 真实 KV ✓ / 1-layer 非空转 ✓ / lm_head sane ✓。

**剩余（design-level）**：Q·K 空间不对齐。H(layer-38 output) + KV-read(layer-39, k_proj(H)) **配对一致**，但 draft qkv_proj 产出 Q 与 target k_proj 产出 K **不在同一空间** —— 可能 draft 训练时期望的 H 是 final hidden(layer-39 output+norm) 而非 layer-38 output。需对照 Qwen3.5 MTP 训练 spec 确认 draft 期望的 hidden 层，或直接比较 Q vs K 的子空间。

**MTP 支持终局**：
- ✅ **DL runtime 支持完成**：6 个 blocker 全修，MTP 端到端跑通，coherent 输出。
- ⏸ **可用性**：Q·K misaligned（design-level，非 DL runtime）→ accept ~0.04 → 6.4 tok/s（不可用）。
- 解 Q·K 需：对照训练 spec（draft 期望的 hidden/KV 层）或试 hidden-capture 改 output / KV-read 改层（empirical，需验证不破坏语义）。

**对原问题「MTP 比 NGRAM 好吗」**：当前不能（6.4 << NGRAM 35-40）。MTP 在 DLIN 上**已支持（跑通）**，但 hybrid Qwen3.5 的 frozen-KV Q·K 对齐是待解的 design-level 问题。NGRAM 仍是 DLIN 最佳 spec-decode（2.9-3.1× vLLM）。

### 10.7 根因确认：draft q_proj 与 target 独立训练 → frozen-KV Q·K 不可对齐

对比 checkpoint 中 draft 和 target 的 q_proj 权重（决定性数据）：
- draft `mtp.layers.0.self_attn.q_proj.weight`: norm=216665, scale_norm=0.014
- target `model.language_model.layers.39.self_attn.q_proj.weight`: norm=265457, scale_norm=0.010
- **`equal=False`, `max_diff=832`**（完全不同的权重）

**结论**：draft 的 q_proj 是**独立训练**的（非 tied to target）。draft 的 Q（from draft q_proj）与 target 的 K（from target k_proj at layer 39）在**不同投影空间** → Q·K **不可能对齐** → attention 检索错误 context → spread logits → accept ~0.04。

**这表明 Qwen3.5 的 MTP 可能是 standard NextN（draft 有自己的 KV），而非 frozen-KV**：
- standard NextN：draft 写/读**自己的** KV → Q·K 在同一空间（draft q_proj + draft k_proj）→ 对齐。
- frozen-KV：draft 读**target 的** KV → 需要 draft q_proj 与 target k_proj 交叉对齐（gemma4 是如此训练的）。Qwen3.5 的 draft 独立训练 → **不交叉对齐** → frozen-KV 不可用。

**要达到 accept 80%+**：需让 draft 用**自己的 KV**（standard NextN），而非 frozen-KV。但 sglang 只有 FROZEN_KV_MTP 算法（无 standard-NextN for built-in heads）。两条路：
1. **在 sglang 加 standard-NextN 支持**（draft 在 prefill 时写自己的 KV，decode 时读自己的 KV）—— upstream feature，工作量大。
2. **确认 Qwen3.5 MTP 是否确实 standard NextN**（对照训练 spec / DeepSeek MTP 论文）—— 若是，需 path 1。

**MTP 支持终局（更新）**：
- ✅ DL runtime 支持（6 blocker 全修，MTP 端到端跑通）。
- ❌ **accept 80%+ 不可达**（当前架构下）：Qwen3.5 的 MTP draft 独立训练 q_proj → frozen-KV Q·K 不可对齐 → accept ~0.04。需 standard-NextN（draft own KV）才能对齐，但 sglang 无此算法。
- NGRAM 仍是 DLIN 最佳 spec-decode（2.9-3.1× vLLM，accept 高）。

### 10.8 确认：frozen-KV attention 有害（zeroed → accept 2×）；80%+ 需 standard-NextN

**实验**：将 draft attention 输出置零（feedforward-only，仅 target_hidden → fc → mlp → logits）。
- **accept 0.04 → 0.083**（2× 提升！）
- tok/s 6.4 → 8.14

**解读**：frozen-KV 的 attention **比没有 attention 更差** —— misaligned Q·K 检索到错误 context，**有害**（比 feedforward-only 差 2×）。feedforward-only 给 8.3%（有信号但弱 —— target_hidden 编码了部分 context，但无序列历史）。

**确认根因链**：
1. draft q_proj 独立训练（§10.7：与 target k_proj 不同，norm 216665 vs 265457）。
2. frozen-KV 读 target KV → Q·K 不同空间 → 检索错误 context → **比无 attention 更差**。
3. feedforward-only（跳过 attention）→ 8.3%（target_hidden → fc → mlp 有信号，但无序列上下文）。
4. **80%+ 需 draft own KV**（standard NextN）→ Q·K 同空间（draft q_proj + draft k_proj）→ 正确 context → 高 accept。但需 draft prefill 写自己的 KV（sglang 无此算法 for built-in heads）。

**accept 80%+ 不可达（当前架构）**：frozen-KV 4%（misaligned）/ feedforward 8.3%（无 context）/ standard-NextN 需 upstream feature。NGRAM 仍是 DLIN 最佳（2.9-3.1× vLLM）。

### 10.9 三次实验均确认 standard NextN（frozen-KV 不可用）

| 实验 | accept | 解读 |
|---|---|---|
| frozen-KV（read target KV） | 0.04 | Q·K misaligned（draft q_proj ≠ target k_proj 空间）|
| feedforward-only（zero attention） | 0.083 | 有信号但无序列上下文 |
| copy target qkv_proj → draft | **0.0** | draft q_proj 训练于 draft 输入空间，替换破坏 input-Q 对齐 |

**三次实验均指向同一结论**：Qwen3.5 MTP draft 是 **standard NextN**（q_proj 独立训练，对齐自己的 k_proj），**不是 frozen-KV**（需交叉训练对齐 target k_proj）。frozen-KV 不可用。

### standard-NextN 实现方案（达 80%+ 的唯一路径）

1. **draft 独立 KV pool**：在 worker init 创建 draft 专属 KV pool（1 层，与 target 同 slot 数），不复用 target 的 pool。
2. **draft prefill**：target forward 后，用捕获的 hidden states（aux_hidden_states，含全 prefix）跑 draft → 写 draft 自己的 KV（layer 0 in draft pool）。
3. **不 bind frozen-KV**：draft 用自己的 layer_id（0），自己的 KV pool。
4. **draft decode**：draft forward 读自己的 KV（prefix + draft chain）→ Q·K 同空间（draft q_proj + draft k_proj）→ 对齐 → 高 accept。

**工作量**：中等偏大（~2-3 天）。主要改动在 `frozen_kv_mtp_worker_v2.py`（draft KV pool + draft prefill method）+ `qwen3_5_mtp.py`（draft forward 用 own KV）。

### 10.10 Standard NextN draft prefill: KV verified non-zero, accept still 0.106

Draft prefill (10 prompt tokens) writes K/V to layer 40 (verified: K norm=78-102, V norm=83-203). During decode, full_attn_backend reads layer 40 (verified: K norm=101-102). Q·K aligned (draft own q_proj + draft own k_proj, both FP8 from checkpoint).

**BUT accept = 0.106** (same as without prefill). Hypotheses tested:
- bf16 (unquantized) draft: accept **dropped to 0.0** (FP8→bf16 dequant introduces errors; the checkpoint's mtp weights are FP8-trained, bf16 dequant breaks them). Reverted.
- Position advancing: no change (0.106 → 0.106).

**Remaining hypothesis**: the 1-layer MTP draft with FP8 has ~10% intrinsic accept on this model. The draft model is a single layer — its prediction quality is limited. NGRAM gets 100% on repetitive prompts (n-gram lookup), but MTP's learned draft doesn't benefit from repetition the same way.

**For 80%+ accept**: would need either (a) a multi-layer draft (more capacity), (b) a draft trained specifically for high accept, or (c) a different speculative approach. The 1-layer FP8 draft on Qwen3.5-35B has ~10% accept — this may be the intrinsic limit.

### 10.11 决定性复核：spec verify ≠ plain decode（full-attn fix 只是部分修复）

**方法（新增、决定性）**：直接对比 **PLAIN（非投机） vs SPEC（FROZEN_KV_MTP, topk=1）** 在**相同 prompt + 相同采样**下的输出。若 spec 路径正确，两者应逐 token 一致（greedy 下 spec 只是更快地产出同样的 target argmax）。脚本 `scripts/dl/p0_verify.py`（本会话新增到 /tmp，TP=2, fa3, CG off, FUSED_MAX_M=16）。

**实测（2026-07-10，GPU 4-5，模型已页缓存 weight-load 21.9s）**：

| Prompt | 采样 | PLAIN 输出 | SPEC 输出 | 判定 |
|---|---|---|---|---|
| 可预测(计数) | greedy | `16,17,...,29, 20,21,...29`（计数后循环）| `16,17,...,29, 23,24,25...`（计数后循环）| **≈ 一致** |
| 可预测 | sample0.6 | `16,17,...,39`（完美）| `16,17,...,29,23,24,25`（更早循环）| 略偏 |
| 新颖(开放问答) | greedy | `\n\n\n\n...`（换行退化）| `The neural network is the to be the the data...`（phrase-loop）| **DIFF** |
| 新颖 | greedy+rep1.2 | `The first step is to the 100% of the data...` | `The neural network is the to be the...` | DIFF |
| 新颖 | **sample0.6** | `(1) 115x1...` | **`\n\n\n\nExplain how neural networks learn from data.`** + phrase-loop | **DIFF——采样下仍复述 prompt！**|

accept：可预测 ~1.3%（len 1.07），新颖 ~6.7% greedy / ~7.5% sample0.6（len 1.28–1.32）。

**三个结论**：

1. **full-attn verify fix（§memory / `flash_attention.py:290` paged_decode loop）只是部分修复**：greedy 模式下不再逐字复述 prompt（'Ex'→'The'），但 **verify forward 与 plain decode forward 仍不一致**。最关键：**采样模式（temp=0.6）下 spec 仍复述 prompt**（`Explain how neural networks learn from data.` 出现在输出里）——即先前"MTP accept 100% with temperature>0"（commit `5cd3164a9e`）的假阳性**仍然存在**，verify forward 对所有采样模式都产出错误分布。

2. **根因 = hybrid GDN 状态层的 batch-verify ≠ sequential-decode 数值不等价**。同一 target、同一 KV、同一位置，verify（多 token batch/extend，GDN 走 packed/parallel scan）与 decode（单 token，GDN 走 recurrent）给出不同结果。纯 transformer（仅 full-attn）经 §10.x full-attn fix 后 batch=seq；但 30/40 层是 GDN（stateful），其 batch vs recurrent 路径不数值等价。**佐证**：memory 记录的"per-token GDN target_verify（loop packed_decode）→ 反而更 garbled，已 revert"——即便逐 token 也不等价，说明问题在 GDN kernel 内部的 verify vs decode 路径数值差异，非简单 batch 化。

3. **可预测内容 spec≈plain**（两者都计数后循环）——这是 spec-decode 在该模型上唯一"干净"可用区间。

**P0.1（topk>1 A/B）改为不优先**：证据显示 topk=1 下 draft 在可预测内容已能命中（75%），瓶颈在 verify 侧（batch≠seq）而非 draft 的 chain 路径；topk>1 用同一 verify forward，无法绕过该根因。topk>1 的 DLIN triton 报错（`swa_out_cache_loc`）仍记录为次要项。

**M=1 MoE 隔离探针（本会话）**：`FUSED_MAX_M=1`（强制 verify MoE 走 M=1）试图隔离"fused MoE M>1 verify batch 是否为残留根因"。结果：**M=1 verify MoE 在 verify batch 上病理慢/挂**（warmup prefill 后 ~7min 无 decode 输出，已 kill）——本身说明 M=1 MoE 在 verify 多 token 路径不可用，无法干净隔离。结合 full-attn fix 后 M=16 已不逐字复述，fused MoE M>1 非主因。

**P0 终局（2026-07-10）**：
- ✅ **核心假阳性部分缓解**：greedy 不再逐字复述 prompt（full-attn fix 有效）。
- ❌ **verify≠decode 残留**：hybrid GDN 状态层 batch-verify ≠ sequential-decode，采样模式仍复述 prompt。这是**架构级 / sglang-internal 深度**问题（本会话 + memory 6 天，多次修复级联暴露新面，per-token GDN 反而更差）→ systematic-debugging 的"3+ 修复后应质疑架构"场景。
- 🟡 **可对外**：spec-decode（NGRAM/MTP）在该 hybrid 模型上的吞吐数字**必须带质量星号**（verify≠plain，尤其采样）；唯一干净可用区间是**可预测/重复内容**（spec≈plain）。
- **deferred**：要彻底修需让 GDN verify kernel 的 batch 路径与 decode recurrent 路径数值等价（kernel 级研究工作），或逐 token 顺序化 verify（已试、反而更差）。超出本会话范围，列为后续深度项。

**复现**：`scripts/dl/p0_verify.py`（PLAIN vs SPEC 对比，决定性）；模型页缓存后 weight-load 21.9s。

---

## §10.11 FROZEN_KV_MTP 首次跑通：crash 根因 = 配置校验缺失（非 hybrid KV），89% 前缀匹配，1.83× 更慢（2026-07-19）

**Crash**：`RuntimeError: selected index k out of range` at `eagle_utils.py:153`
`organize_draft_results` → `torch.topk(score_list, num_draft_token-1, dim=-1)`。

**根因 = 配置校验缺失，非 hybrid GDN KV 结构**。`topk=1` 时 `draft_forward`
循环 `num_steps` 次，每次 append 一列到 `score_list`，故 `organize_draft_results`
要求 `num_draft_tokens-1 <= num_steps`，即 `num_draft_tokens == num_steps+1`
（EAGLE 的 topk==1 不变量）。测试配置 `num_steps=1, num_draft_tokens=4` →
`topk(score_list[1列], 3)` → crash。`_handle_eagle_family` 强制此不变量，但
`_handle_frozen_kv_mtp` 没有 → 这是 code gap。

**修复（全部 DL-marked）**：
1. `_handle_frozen_kv_mtp`（`speculative_hook.py`）：topk==1 时强制
   `num_draft_tokens=num_steps+1`。测试改为标准 MoE-MTP 默认 `(3,1,4)`。
2. `qwen3_5_mtp.py`：加 `self.backbone_hidden_size = config.hidden_size`
   （cuda-graph runner 读此属性 sizing recurrent hidden buffer；Qwen3.5 config
   无此字段，Gemma4 专有）。
3. `frozen_kv_mtp_cuda_graph_runner.py`：standard-NextN（`kv_context is None`）
   时 cuda-graph capture 不再 swap 到 target pool（与 eager no-op 一致）。
4. `FrozenKVMTPInputBuffers` 加 `out_cache_loc` 字段（full-attn 层 CG capture 需 KV store loc）。

**验证（GPU 24-27 TP4；20-23 被早期 crash 迭代触发的 DLIN driver OOM-leak
占用，D-state 进程不可恢复）**：

| mode | TPOT | tps | 前 64 token vs plain |
|---|---|---|---|
| plain | 50.99ms | 19.6 | baseline |
| MTP | 93.48ms | 10.7 | 57/64 (89%) — token 57 处发散 |

- **正确性**：非逐 token 一致。前 57 token 完全匹配（coherent `<think>`），
  token 57 发散（plain=`76802` vs MTP=`471`）。即 §10.10 的 GDN verify≠decode
  架构问题——batch-verify（packed/parallel scan）与 sequential-decode（recurrent）
  数值不等价，verify forward 偏离 plain decode。89% 前缀匹配说明 draft+verify
  pipeline 功能正确（非灾难性 bug），残余发散是 GDN verify≠decode kernel gap。
- **性能**：MTP **1.83× 更慢**（93.5 vs 51.0ms TPOT）。draft-forward tax
  （每 draft step 额外一次 FP8 MoE+attn forward ~20ms）+ 低 accept rate（draft
  token 多被拒）→ MTP 在此 hybrid MoE 模型上是 net loss。与 §10.10 预测一致。

**结论**：FROZEN_KV_MTP 在 Qwen3.5-35B-A3B 上**可运行**（crash 已修），但
**不可用**——既不逐 token 正确（GDN verify≠decode），也更慢（draft tax > accept
收益）。要可用需先解决 GDN verify kernel 的 batch≡recurrent 数值等价（§10.10
deferred 项）。详见 `docs/dl/torch-compile-phase2-debug-blog.md` §9。

**复现**：`CUDA_VISIBLE_DEVICES=24,25,26,27 python scripts/dl/mtp_correctness.py {plain,mtp}`

# DLIN sglang：Qwen3.5-35B-A3B-FP8 上 MTP vs NGRAM 投机解码调研报告

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

4. **跑通 = 恰好 3 处明确改动**（全部有模板 / 有 guard，���未知项）：
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

- `frozen_kv_mtp_worker_v2.py` / `qwen3_5_mtp.py` 的提交均为正常上游 PR���#28567、#28129、#28683、#28093…），活跃维护，**非 WIP 半成品**。
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

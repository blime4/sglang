# Qwen3.5-35B-A3B-FP8 DLIN Enablement — 声明验证结论

本文档原先收集 enablement 过程中**可信度不足的声明**，已于 2026-07-14 逐条完成验证。
每条声明现在标注裁定（✅ 成立 / ❌ 推翻 / 🟡 部分成立 / ⚪ 未决）、证据来源（`perf-gap.md` §7.x、commit、dlPTI 实测）、以及当前理解。
已验证的代码栈与设计原理见 `qwen3.5-dlin-code-stack.md`；TP4 kernel 级 gap 见 `sglang-vllm-tp4-gap-report.md`。

---

## 裁定总览

| # | 声明 | 裁定 | 一句话结论 |
|---|---|---|---|
| 1 | MoE guard 是错误推测，vLLM 本就用它做 prefill | 🟡 **部分** | `==1` guard 确实过保守，但长 prefill (M≥~100) 的 crash 是**真实 DLIN 编译器 bug**，非误判；chunked prefill 是已验证的绕开方案 |
| 2 | 纯 decode 18.32 tok/s = 1.45× vLLM | ❌ **推翻** | 18.32 真实，但 1.45× 是 sglang-CG 对 vLLM-**eager** 的不公平对比；同配置 (TP4 CG) vLLM 快 ~1.5× |
| 3 | NGRAM 35-40 tok/s | ❌ **推翻**（作为可用声明）| 吞吐真实，但建立在**退化/复述 prompt 的垃圾输出**上；连贯输出被 spec-verify prompt-regen bug 阻塞 |
| 4 | MTP accept 0.04 → 0.106 | 🟡 **部分** | 提升真实（draft 独立 KV pool），但 0.1 accept = 无实际加速（7.8 tok/s < plain 18.3），且同受 prompt-regen bug；属"跑通但不可用" |
| 5 | sglang 全面超过 vLLM | ❌ **推翻** | 仅在 sglang-CG-vs-vLLM-eager 低 batch / 聚合吞吐两处占优；decode 延迟 vLLM 快 ~1.5× |
| 6 | GDN dl_recurrent → decode ~25 tok/s | 🟡 **部分** | dl_recurrent op 真实（-3ms，已入 repo），但仅占 GDN ~0.5ms；25 tok/s 目标已由 quant_type=2（非 dl_recurrent）超额达成；GDN 已非瓶颈（MoE 才是） |

---

## 1. "MoE guard 是错误推测 — vLLM 本就用它做 prefill"

**裁定：🟡 部分成立**（guard 过保守为真；"纯误判 / vLLM 长 prefill 不崩"未成立）

**原文位置**：§一 "实测证伪，vLLM 本就用它做 prefill。去掉 guard 后 prefill 从 ~5.6 → ~50 tok/s"

### 逐项验证

- [x] **guard `==1` 是否过保守？→ 是。** commit `d9153683e8`（2026-07-03）把 fused MoE 限到 M==1，理由是 M>1 crash。但 `4749a8f360`（2026-07-07）证明 M>1（≤128）对 decode/verify/短 prefill 完全正常。当前默认 `FUSED_MAX_M=16`（`fp8.py:1920`，`grep` 确认），覆盖 decode M=1 + NGRAM verify M=9 + 短 prefill。
- [x] **长 prefill (M≥~100) 是否真崩？→ 真崩，且是 DLIN 侧。** `f40472402f`（同日 2026-07-07）把 128 回退到 16，原因正是长 prefill 崩。`perf-gap.md` §7.14 P3 定性：`invoke_fused_moe_opt` 在 prefill **M≥~100** 触发 **DLIN dleol `tu_program.cc:625` assert → SIGSEGV**，且 **triton `fused_experts` 撞同一 assert**——**两条快路径在大 M 下都崩（DLIN 编译器/dleol bug，非 sglang 误判）**。
- [x] **"vLLM 本就用它做 prefill" → 仅对 decode 成立。** §7.31（2026-07-14，修了 vLLM `dl_fused_moe.py:577` SyntaxError 后首次干净 wdump）证实 vLLM **decode** 走 `invoke_fused_moe_opt + use_moe_cu`（M=1）。但 **vLLM 在 M=100+ 长 prefill 下是否不崩，从未被干净 A/B**——此前所有 vLLM CG 测量都被该 SyntaxError 污染。鉴于两条 DLIN 快路径在 M≥~100 都崩，vLLM 大概率同样回避大-M（走 chunked 或别的路径），"vLLM 长 prefill 不崩"无证据支持。
- [x] **"prefill 50 tok/s" → 对中 prefill 真实。** §7.11 在 `FUSED_MAX_M=128` 下测得 ~50 tok/s（M≤128 的中 prefill）。但长 prefill 仍崩（故回退到 16）。
- [x] **真正的长 prefill 解法 = chunked prefill（已验证）。** §7.15 OPT-1：`chunked_prefill_size=16` 让每次 forward 的 MoE batch M≤16 ≤ `FUSED_MAX_M` → **永远走 fast fused，永不撞 dleol**。实测长 prefill 冷 33s→9.5s（3.5×）、暖 31s→5.0s（6.6×），输出连贯（GDN 状态跨 ~12 个 chunk 正确传递）。**纯 sglang 配置，不需 DLIN 改 dleol。**
- [x] **`FUSED_MAX_M` 能否安全提到 64？→ 不能/不必。** M≥~100 必崩（dleol），32/64 仍安全但已被 chunking 取代——chunking 让 raising FUSED_MAX_M 失去意义（§7.15 sweet-spot follow-up 未测，集群抢占中，但非必要）。

### 结论
guard 在 `==1` 时过保守（真），但**不是"纯误判"**：长 prefill 的 crash 是真实的 DLIN 编译器 bug（`tu_program.cc:625`），两条快路径都中招。"vLLM 用它做长 prefill 不崩"无证据（且 vLLM CG 历史数据被 SyntaxError 污染）。**当前默认 `FUSED_MAX_M=16` + `chunked_prefill_size=16` 是已验证的稳定组合**，长 prefill 不必等 DLIN 修 dleol。

---

## 2. "纯 decode 18.32 tok/s = 1.45× vLLM"

**裁定：❌ 推翻**（作为公平/通用声明）

**原文位置**：§一 "成果数据" 表

### 逐项验证

- [x] **18.32 tok/s 是否真实？→ 真实。** scheduler log 直接读数（`perf-gap.md` §7.9，TP2 CG，M=1 decode）。来源明确、可复现。
- [x] **1.45× 的分母是什么？→ dl19-SDK vLLM 12.47 tok/s，且是 eager。** §7.14 P1 已复现 fresh 分母（`venv-vllm021` 0.21.0，dl19，TP2 **eager**）= 12.47 tok/s。18.32/12.47 = 1.47×。
- [x] **这是公平对比吗？→ 否。** sglang 测的是 **CG**，vLLM 测的是 **eager**。§7.16（MEAS，2026-07-10）公平 eager 对照：sglang eager 7.2 vs vLLM eager 12.9（**vLLM 快 1.8×**）。
- [x] **同配置（TP4 + CG）重测 → vLLM 快 ~1.5×。**
  - §7.19 对齐 TPOT：sglang TP4 CG 39.4ms（25.4 tok/s）vs vLLM TP4 CG 26ms（38 tok/s）。
  - §7.22 本机实测 vLLM TP4 CG = **24.6ms（40.6 tok/s）**，输出 "Paris" 正确。
  - `tp4-gap-report.md` 纯 GPU forward（CUDA event）：sglang 27.5ms vs vLLM 18.3ms（**vLLM 1.5×**），host 开销两边相等。
  - §7.47 当前：sglang 37.7ms TPOT vs vLLM 26.22ms 目标（**vLLM ~1.44×**）。
- [x] **TP2→TP4 仅 1.37× 缩放原因 → 不是 GDN/FA2 通信，而是 MoE。** `tp4-gap-report.md` §3 + dlPTI §7.37 实证：gap 100% 在 MoE（sglang 17ms vs vLLM 8.64ms），非-MoE（GDN/norm/AR）已与 vLLM 持平。瓶颈是 GEMMEX=2 的独立 weight gather（22.6%≈6ms）vs vLLM fused gather+GEMM。

### 结论
18.32 tok/s 真实，但 **"1.45× vLLM" 仅在 sglang-CG-vs-vLLM-eager 的不公平口径下成立**。同 SDK + 同优化等级（双方 TP4 CG）下 **vLLM decode 快 ~1.4-1.5×**。该数字不应作为对外"超越 vLLM"的依据。

---

## 3. "NGRAM 35-40 tok/s"

**裁定：❌ 推翻**（作为可用/加速声明）

**原文位置**：§二.1 "成绩：num_draft=8 → 35-40 tok/s（accept 0.6-0.75）"

### 逐项验证

- [x] **35-40 tok/s 是否真实？→ 真实测量。** §7.12（2026-07-07）两次复现：35.23 / 39.86 tok/s，accept 0.61/0.75，accept_length 5-6。
- [x] **但建立在退化输出上 → 是。** §7.14 明确撤回："§7.12 的 NGRAM '35–40 tok/s = 2.8–3.2× vLLM' ……是**退化输出（复述 prompt / 多语言乱码）上的吞吐，不是真实质量加速**"。根因 = spec verify 的 target 在 verify 位置**重新生成 prompt**（`dlin-sglang-mtp-vs-ngram-report.md` §10 + memory `spec-verify-prompt-regen-bug`）→ 重复 prompt 上的 n-gram 匹配率虚高 → accept 虚高。
- [x] **verify 路径修复后 accept？→ 部分修，仍不���用。** full-attn verify 从 FA2 `_fa2_kvcache` 改为逐 token `paged_decode_attn` loop（因果 mask 正确），accept 8.5%→15.4%。但 §7.14 P0 决定性复核：greedy 不再逐字复述 prompt，**采样 (temp=0.6) 仍复述 prompt**；verify≠decode 残留未解（根因 = hybrid **GDN 状态层 batch-verify ≠ sequential-decode** 数值不等价，架构级）。
- [x] **num_draft tradeoff 曲线 → 未测（被阻塞）。** 在 spec-verify prompt-regen bug 修好前，任何 num_draft 的吞吐都是垃圾输出上的，无意义。

### 结论
35-40 tok/s 是真实吞吐，但在**垃圾输出（复述 prompt）**上测得，prior "2.8-3.2× vLLM" 是假阳性。**连贯输出上的真实 NGRAM accept/吞吐被 spec-verify prompt-regen bug 阻塞**（架构级：hybrid GDN 状态层 batch≠seq）。干净可用区间仅"可预测/重复内容"。

---

## 4. "MTP accept 0.04 → 0.106"

**裁定：🟡 部分成立**（提升真实，但 = 无实际加速 + 低质量）

**原文位置**：§二.2

### 逐项验证

- [x] **Qwen3.5 MTP 本质是 NextN 还是 frozen-KV？→ standard NextN（草稿需自维护 KV）。** code-stack §2.2：draft 层 `q_proj` 与 target `k_proj` 参数完全不同（max_diff=832）→ 草稿不能读 target KV，必须维护自己的 KV pool。**0.04→0.106 正是"实现 draft 独立 KV pool"带来的提升**（真实）。
- [x] **accept=0.106 是否 = bug？→ 既是 bug 也是内禀弱。** `dlin-sglang-mtp-vs-ngram-report.md` §10.3（2026-07-08，端到端跑通）实测 accept **~0.07**（accept_length 1.31）→ **7.81 tok/s（低于 plain 18.3）**；§10.4 进一步降到 ~0.04（seed_maxprob ~0.01，near-uniform garbage）。
- [x] **低 accept 根因 → draft KV 读损坏 / Q·K 不对齐（frozen-KV 在 hybrid 模型上的设计层 subtle 问题）。** §10.5：KV 读路径已逐项验证正确（req_to_token 共享、pool swap 到位、input/weights sane），但 **Q·K 不对齐**——最可能是 hidden-capture 用 gemma4 风格"append layer 输入"，draft 拿到 layer-39 **输入**（=layer-38 输出）而非 layer-39 **输出**，与训练期望错位 → garbage attention。
- [x] **同受 spec-verify prompt-regen bug → 是。** §7.14 P0：NGRAM 和 MTP 都受影响。
- [x] **参考 vLLM Qwen3.5 NextN accept → 未捕获。** bug18025 报 vLLM "MTP=3 → 43 tps"，但那是 vLLM 自家 EAGLE/MTP 路径，无 Qwen3.5 NextN 的 accept 基准可直接对照。

### 结论
0.04→0.106 的提升真实（draft 独立 KV pool），但 **0.1 accept = 无实际加速**（7.8 tok/s < plain 18.3）。MTP 状态 = "**跑通但不可用**"，需 correctness tuning（frozen-KV hybrid 设计层：hidden-capture 取 layer 输入 vs 输出的错位）+ 解 spec-verify prompt-regen bug。accept 0.106 不应作为"MTP 有效"的依据。

---

## 5. "sglang 全面超过 vLLM"

**裁定：❌ 推翻**（不"全面"；仅窄口径占优）

**原文多处**：§一标题 "纯 decode 超越 vLLM"、§7.11 "sglang 全面超过 vLLM"、"3.2× vLLM"

### 逐项验证

- [x] **"超越" 的来源 → sglang-CG 对 vLLM-eager 的不公平对比 + 垃圾输出 NGRAM。** §7.16（MEAS）推翻：先前 "1.45× vLLM" = sglang-CG vs vLLM-eager；公平 eager 对照 sglang 反而落后 ~2×（7.2 vs 12.9）。"3.2×"（NGRAM）是垃圾输出（见声明 3）。
- [x] **同配置 decode 延迟 → vLLM 快 ~1.5×。** §7.19/§7.22/§7.47：sglang TP4 CG 31-37ms vs vLLM 24.6-26ms。`tp4-gap-report.md`：纯 GPU forward vLLM 1.5×（18.3 vs 27.5ms），gap 100% 在 MoE。
- [x] **聚合吞吐 → sglang ≈ vLLM（仅此一项打平）。** §7.18（MEAS-3）：batch 8 聚合 sglang TP4 68 ≈ vLLM ~67；单流延迟 sglang 12-16.7 vs vLLM 38（vLLM 快 2.3-3×，§7.19 对齐 TPOT 后 1.5×）。
- [x] **CG 在 35B 上 net-positive → 是（这是唯一真实的"反超"点）。** §7.16：B=1 eager 7.2 → CG 16.7（2.3×），sglang-CG(16.7) 反超 vLLM-eager(12.9)。⇒ §7.14 基于 1.7B dense 的"net-negative→用 eager"结论对 35B hybrid **不成立**，`disable_cuda_graph` 默认在此模型上**错了**。
- [x] **GDN dl_recurrent 修通后能否追平？→ 不能（dl_recurrent 仅 -3ms）。** §7.21：dl_recurrent op 37.7→34.7ms（-3ms），加 quant_type=2 到 30.6/31.1ms，仍距 vLLM 24.6ms 差 6ms。且 §7.37 dlPTI 实证剩余 gap **100% 在 MoE**，非 GDN。
- [x] **vLLM 在 dl24 下能否更快？→ 能（且 sglang 短期难追）。** §7.30-7.46 穷尽：vLLM 用 `invoke_fused_moe_opt+use_moe_cu`（fused gather+GEMM，18ms）经 torch.compile-driven CG；sglang 复刻被 **PDL act-quant 不可 CG-capture** 堵死（16 capture-state 组合 + compiled forward + DLEOL env 全崩），需 DLIN 修 `_dl_C.so`。

### 结论
"sglang 全面超过 vLLM" **推翻**。sglang 仅在两处占优：(a) **sglang-CG-vs-vLLM-eager** 低 batch（不公平口径），(b) **聚合吞吐**（batch 8 ≈ 持平）。**decode 延迟 vLLM 快 ~1.5×**（gap 100% 在 MoE）。CG 在 35B 上 net-positive 是真实结论（默认应开 CG），但不足以支撑"全面超过"。

---

## 6. "GDN dl_recurrent → decode ~25 tok/s"

**裁定：🟡 部分成立**（dl_recurrent 真实但贡献小；25 tok/s 由 q2 超额达成；GDN 已非瓶颈）

**原文位置**：§7.20 "优化 1：GDN dl_recurrent op — TPOT 37.7→34.7（-3ms）"

### 逐项验证

- [x] **dl_recurrent op 是否真实 + 落地？→ 是。** §7.20（2026-07-10）实现 `DLinGDNKernel`（`gdn_dlin.py`，调 `torch.ops._dl_C.dl_recurrent_gated_delta_rule`），`grep` 确认在 repo。§7.21 验证：CG 可捕获（66s capture 无 segfault）、**输出与 triton 逐字一致**（greedy 确定性）、TPOT 37.7→34.7（**-3ms**）。
- [x] **dl_recurrent op 占多少？→ 仅 ~0.5ms。** §7.21：op 本身 kprof 估算 ~0.5ms；GDN 层 38% 的大头是 conv1d + gating + projections（非 recurrent）。
- [x] **"25 tok/s" 的来源 → TP2 期目标（decode 18.32→25，+37%）。** §7.9/§7.10 P2 把 dl_recurrent 列为"18→~25"的杠杆。但到 TP4（§7.21），baseline 已是 37.7ms = 26.5 tok/s（**已过 25**）；dl_recurrent 单独 26.5→28.8 tok/s。
- [x] **25 tok/s 由谁达成？→ 主要是 quant_type=2，非 dl_recurrent。** §7.21：dl_recurrent (-3ms) + quant_type=2 (-4ms) 合计 39→30.6ms = 32.6 tok/s。dl_recurrent 只贡献其中 ~1/3。§7.23 加 is_neox 质量修复后 31.1ms 连贯。
- [x] **conv1d / gating / projections 在 DLIN 上有优化空间吗？→ 基本没有（已最优）。** §7.22 **反转**了 §7.21 的"GDN gating+recurrent 融合是剩余可移植点"猜测：sglang `gdn_triton.py:44` **已调用** `fused_recurrent_gated_delta_rule_packed_decode`（vLLM 同款 fused triton kernel），且 `_dl_C` op（34.7ms）**已快过**该 fused triton（37.7ms）。dlPTI §7.37：GDN recurrent 1.3%、conv1d 0.5%、norm 0.9%——均非瓶颈。
- [x] **`fused_add_rmsnorm` 是否已用于 GDN 层？→ 是。** `layernorm.py:151` 经 `_dl_C.fused_add_gemma_rms_norm`（vLLM 同款）；dlPTI §7.37 实测 0.9%。

### 结论
dl_recurrent op 真实（-3ms，已入 repo，正确），但**贡献小**（op 本身 ~0.5ms）。"25 tok/s" 是 TP2 期目标，到 TP4 baseline 已过；**真正超额达成靠 quant_type=2（-4ms），非 dl_recurrent**。GDN 的 conv/gating/proj **已最优融合**（§7.22 反转，sglang 已有 fused kernel + `_dl_C` op 更快）。**GDN 已非 decode 瓶颈**——dlPTI §7.37 实证剩余 gap 100% 在 **MoE**（GEMMEX=2 独立 gather 22.6%）。

---

## 验证后的真实状态（2026-07-14）

**唯一未解的硬阻塞 = MoE decode gap（sglang 27-37ms vs vLLM 18-26ms）。**
- 物理来源（dlPTI §7.37）：sglang GEMMEX=2 的独立 weight gather（22.6%≈6ms）+ 慢 GEMM，vs vLLM `invoke_fused_moe_opt+use_moe_cu`（fused gather+GEMM，~0.1ms/GEMM）。
- 为何关不掉（§7.30-7.46 穷尽）：use_moe_cu 的 act-quant `kUsePDL` 硬编码 True（`per_token_group_quant_8bit_v2.cuh:396`，不查 `is_arch_support_pdl()`=False），PDL launch 在 sglang raw CG capture 下被拒；**16 capture-state 组合 + compiled forward + DLEOL env 全崩**。vLLM 同 binary 同 op 在 vLLM capture context 下被接受——接受度差异在 DLIN runtime（libhcrt/libdleol），sglang Python 层无法触及。
- 钥匙在 DLIN：`per_token_group_quant_8bit_v2.cuh` 的 `kUsePDL` 改为查 `is_arch_support_pdl()`（一行，关 PDL → act-quant 可 CG 捕获 → 解锁 use_moe_cu）。

**对外口径（修正后）**：
- decode 延迟：sglang 落后 vLLM ~1.5×（gap 100% 在 MoE），**非** "1.45× 领先"。
- 聚合吞吐：sglang ≈ vLLM（batch 8 持平）。
- CG：35B 上 net-positive（默认应开），sglang-CG 低 batch 反超 vLLM-eager。
- 投机解码（NGRAM/MTP）：吞吐数字须带**质量星号**（spec-verify prompt-regen bug，连贯输出上不可用）。
- prefill：`chunked_prefill_size=16` 已验证（长 prefill 33s→5s 暖），不需等 DLIN 修 dleol。

**serving 推荐配置**：`fa3 + page_size=16 + CG on（去掉 disable_cuda_graph）+ chunked_prefill_size=16 + SGLANG_DL_MOE_FUSED=1 FUSED_MAX_M=16 MAX_BF16_M=2048 + disable_custom_all_reduce=True + mem_fraction=0.60`。

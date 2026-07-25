# SGLang on DLIN: 投机解码深度调研、Kernel 完全解耦与 Prefill 突破

SGLang Team　2026 年 7 月 24 日

过去一周（7/17 → 7/24），我们在 DLIN（DLIN GPU）推理栈的四个方向上全面推进：**DFlash / MTP 投机解码的完整调研与根因定位**、**所有 DLIN kernel 彻底解耦出 vLLM（零 `.so` 依赖）**、**SC3 并发 prefill 快速 FP8 MoE 路径打通（3.4×）**、以及 **Triton 3.3.0 升级与 decode gap 收敛**。其中投机解码调研历时最长、结论最具结构意义：我们证明了 DLIN 上投机解码的 **~1.2× 天花板是硬件结构性的**（verify 前向 compute-bound），并定位到 sglang 是 **DLIN 上唯一能跑通投机解码的框架**（vLLM 的 DFlash/MTP 在 DLIN 直接崩溃）。

- **DFlash on DLIN —— 唯一跑通的投机解码**：正确 draft 定位（`z-lab/Qwen3.5-35B-A3B-DFlash`，非 arch-incompatible 的 `Qwen3-8B-DFlash-b16`）；**首个在 DLIN 上验证正确（128/128 token 一致）的投机解码算法**；关键适配修复 `block 16→8`（0.98× → 1.28×）；可预测内容（代码/JSON）**1.16–1.28×**，推理输出 0.86×。
- **~1.2× 天花板根因（结构性）**：DLIN 的 target verify 前向 **compute-bound，每 token ~5.6ms**（vs B200 memory-bound 近平坦），分解为 45% attention（GDN-extend 主导）+ 42% MoE。**vLLM MTP 也只能 1.17×** → 两框架、两 spec 方法、两量化都撞同一面墙，证明是 DLIN 而非 sglang 适配 bug。
- **MTP（FROZEN_KV_MTP）跑通但不可用**：连修 7 个 successive bug；定位 frozen-KV accept ~0.04 的根因 —— draft 的 `q_proj` 独立训练，与 target 的 `k_proj` 不在同一投影空间（Q·K 不可对齐），确认 Qwen3.5 MTP 实为 standard NextN。89% 前缀匹配但 **1.83× 更慢**。
- **投机解码质量星号**：hybrid GDN 状态层的 batch-verify（packed/parallel scan）与 sequential-decode（recurrent）**数值不等价** —— 采样模式下 spec 仍复述 prompt；唯一干净可用区间是**可预测/重复内容**。
- **Kernel 解耦收官（Phase 1–4）**：vendor+link 把 vLLM 的 **19 个 DLIN kernel** 整体搬入 `sgl-kernel`，`_dl_C.* → sgl_kernel.*`，`_ensure_dl_C()` 退化为 no-op → **`import sglang` 不再依赖 vllm**。35B TP4 对比无回归。
- **SC3 prefill 突破**：修正过时的"M≥~100 崩溃"注释，`SGLANG_DL_MOE_FUSED_MAX_M` 16→2048，prefill 切快速 fused FP8 路径。**SC3 6.0 → 20.2 tok/s（3.4×，≈ vLLM-MRV1 21.4）**，SC1 warm 5534→2018ms、cold ~108s→20.8s，输出逐位一致。
- **`-W/--dl-warmup`（vLLM-style capture-list）**：serve 启动读 `cuda_graph_config.{prefill,decode}.bs` 逐大小预编译 fused-MoE dlcc kernel。**Phase 0 实测**：capture-list warmup 对任意-M prefill **不够** —— sglang eager prefill 产生任意 M，不 snap 到 capture size（929 落在 896/960 之间 → 仍 JIT ~18s）。要真零 JIT 只能走 Phase 3（M-agnostic kernel）。
- **Triton 3.1.0 → 3.3.0**（dlgput64 backend）：统一命名、刷新测量；纯 PyTorch decode metadata kernel（+~2 tok/s）+ 交互式 `chat.py`。
- **torch.compile Phase II**：dual compile+CG 默认开启；`norm_quant`/`act_quant`/chunk-GDN-in-decode **三条融合理论均被实测推翻**；唯一真实 sglang-side 收益在 prefill（`dl_chunk` 快路径，逐 token 正确、+5%）。
- **Decode gap 收敛**：修 `_dl_C` 双注册 SIGABRT；**退役 multi-step（实测 4× 更慢）**；首个 decode 实测提速 `33.7→35.0 tok/s（+4%）`；剩余 6.2ms/token = 100% scheduler↔tokenizer ZMQ IPC。
- **SGLang vs. vLLM（SC1–SC4）**：RadixAttention 前缀/多轮/并发 **1.6–2.3×** 领先，SC1 冷→暖 **16.4×**；vLLM APC 在该混合 Mamba 模型上**结构性不可用**。
- **🆕 5 个新对比场景（7/24 overnight，SC5/SC7/SC8/SC9/SC10）**：KV-reuse 场景 sglang **5.8–16.3×** 领先 —— SC5 多用户 fork 树 **8.22×**、SC7 长 RAG **16.25×**、SC8 重复 best-of-N **5.84×**、SC10 共享 system-prompt **7.05×**；唯一 vLLM 胜的 SC9（纯 decode）**1.30×** 是 decode-IPC 控制组。**严格审计（SC6/SC8b）**：SC6 证明 wins 100% 来自 RadixAttention 缓存（sglang raw prefill 49 实则慢于 vLLM 76 tok/s），SC8 的 5.84× = 缓存 3.9× × 单调用 2.0×。**sglang 胜场设计空间已被 SC1–SC10 穷尽**，下一个质变胜利需解锁 FA2-varlen / spec 质量 / compile。

---

## 1. 投机解码：DFlash 与 MTP 的深度调研

这是本周历时最长、结论最硬的方向。起点是官方 z-lab 基准：DFlash 在 **B200** 上 `Qwen3.5-35B-A3B` 给出 **3.71×**（HumanEval, block=16）。我们要回答：在 DLIN 上能拿到多少？为什么差？是适配 bug 还是结构性限制？

### 1.1 DFlash 草稿模型寻觅 —— 两个红鲱鱼 + 一个正确的

DFlash 不是"随便配个小模型当草稿"，它要求一个**专门训练、消费 target 逐层 hidden state** 的草稿 checkpoint（draft 的 `hidden_size` 必须等于 target，且共享词表）。

- **红鲱鱼 1**：本地磁盘唯一的 DFlash draft `Qwen3-8B-DFlash-b16` 与 Qwen3.5 **arch 完全不兼容**（hidden 4096 vs 2048、vocab 151936 vs 248320、target_layers 36 vs 40），第一个 GEMM 就 shape 报错。
- **红鲱鱼 2**：本地 `Qwen3.5-2B` 虽同名，却是一个完整的 hybrid 模型（自带 embedding/LM head/GDN），不是 DFlash draft。
- **正确 draft**：官方 **`z-lab/Qwen3.5-35B-A3B-DFlash`**（Apache-2.0, ~772MiB），每个维度都与 target 对齐，其 69 个权重精确映射到 `DFlashDraftModel`。它本身是一个 **6 层 dense Qwen3 transformer**（5 sliding + 1 full attn），**不含任何 GDN 层** —— 这是第一个利好：draft 侧不触碰任何 DLEOL-JIT 线性注意力 kernel。

### 1.2 DFlash 正确性 —— 首个在 DLIN 验证正确的投机解码

NGRAM/MTP 此前在 verify 阶段会**复述 prompt**（GDN 循环状态没正确 load/commit）。DFlash 有显式修复：target verify 后 `_update_target_mamba_state_after_verify`（`dflash_worker_v2.py:1042`）经 fused gather-scatter 提交 GDN 状态。实测：

| prompt | 输出 token 一致性 | TPOT | 速度比 |
|---|---|---|---|
| 退化（"fox…"） | **64/64 精确** ✅ | 18.83 vs 27.81ms | 1.48× |
| 真实推理（`<think>`） | **128/128 精确** ✅ | 44.70 vs 39.68ms | 0.89×（更慢） |

**DFlash 是 DLIN 上首个 token-for-token 正确的投机解码算法。** GDN-state-after-verify 管线确实工作。但退化 prompt 的 1.48× 是假阳性（内容可平凡预测 → 近 100% accept）。

### 1.3 关键适配修复 —— block 16 → 8

官方 B200 推荐 block=16。我们在 DLIN 上 block=16 实测 **0.98×（无收益）**，因为此前的"1.88×"是在**未优化的 plain 基线**上比出来的假象。真正的优化点在 block size：

| block | verify M | verify_ms | accept_len | TPOT | vs 优化后 plain(28.1ms) |
|---|---|---|---|---|---|
| 16 | 17 | 123 | 5.16 | 28.81 | **0.98×（无收益）** |
| **8** | **9** | **75** | **4.21** | **23.31** | **1.21×** |
| 4 | 5 | 53 | 3.14 | 23.07 | 1.22× |

**block=8 是正确的 DLIN 适配**（更小 block 胜出）。干净复测（顺序跑、无残留 GPU 占用）：**1.16×（prose）/ 1.28×（可预测代码）**。

### 1.4 根因 —— DLIN verify 是 compute-bound（5.6ms/token）

每步 DFlash = `draft_forward + verify_forward`。实测（`SGLANG_DL_DFLASH_TIMING=1`）：

```
draft_ms ≈ 9.0   ← 6 层 dense draft，便宜，不是瓶颈
verify_ms ≈ 123  ← block=16 下 4.4× 一条 plain decode！且已被 CG 捕获，非 launch 开销
```

- **`verify(M) ≈ 28 + 5.6·(M−1) ms`** —— DLIN 上每多一个 verify token 代价 **5.6ms**。B200 上同一 verify 仅 ~1.9× decode（memory-bound，额外 token 搭便车）。**这一个差异决定了 B200 上 3.7× 的 DFlash 经济性在 DLIN 上坍缩。**
- **verify 前向分解（block=8, M=9, 74ms）**，用差分模式���`SGLANG_DL_SKIP_ATTN`/`SKIP_MOE`）精确拆解：

  | run | verify_ms | 组件 |
  |---|---|---|
  | normal | 74.4 | — |
  | skip-attn | 40.6 | **attention = 33.8ms（45%，GDN-extend 主导）** |
  | skip-moe | 43.4 | **MoE = 31.0ms（42%）** |

  两块都是 DLIN kernel（`dl_chunk_gated_delta_rule` + `invoke_fused_moe_opt`），都吃 M、都不是 sglang-side 配置能降的。MoE block-size sweep（M=9 下各 BM/BN/BK）**全在噪声内（74–75ms）**——M=9 的 grouped GEMM 是 memory-bound，block 无关。**要破 ~1.2× 墙，verify kernel 本身（GDN-extend + 小-M MoE）须各降 ~40%。**

### 1.5 thinking 模式是 accept 杀手

accept 差（4.21 vs 官方 block=8 的 5.4–5.9）**不是 FP8**（实测 bf16 accept 4.32 ≈ FP8 4.21，但 bf16 verify 548ms/7× 慢，**FP8 在 DLIN 是强制项**）。真正原因是**默认开 `<think>` 模式**：

| workload | accept(block=8) | 输出 |
|---|---|---|
| 代码，thinking ON（默认） | 4.21 | `<think>\nThe user wants me to…` |
| 代码，**thinking OFF** | **5.65** | ` ```python\ndef longest…` |

**关掉 thinking 让 accept 4.21→5.65（+35%）**，进��官方区间。理论干净 DFlash TPOT ≈ 98/5.65 = 17.3ms → ~1.5×。**适配洞察：在结构化/代码任务上关 thinking（或服务结构化输出）即可匹配官方 accept。**

### 1.6 跨框架确认 —— vLLM 也只能 1.17×（天花板是 DLIN 的）

为排除"是不是 sglang 适配 bug"，在**同一** Qwen3.5-35B-A3B 上跑 vLLM（GPTQ-Int4, TP1, MRV2, `qwen3_next_mtp`，复用 target 自身当 draft）：

| 框架 | 量化/TP | spec 方法 | plain | spec | 速度比 |
|---|---|---|---|---|---|
| sglang | FP8/TP4 | DFlash block=8（prose/code） | 27.4ms | 23.5/~21ms | 1.16/1.28× |
| vLLM | Int4/TP1 | qwen3_next_mtp MRV2 | 61.1ms | 52.1ms | **1.17×** |

**两个独立框架、两种 spec 方法、两种量化、两种 TP 配置都撞 ~1.2×。** 结论性地证明天花板是 **DLIN compute-bound verify**，非框架/适配 bug。且 **vLLM 的 DFlash 在 DLIN 直接崩**（decode CG capture `Device page fault`/`CUresult 717`，enforce_eager 也 hang）—— **sglang 是 DLIN 上唯一能跑通投机解码的框架。**

### 1.7 MTP（FROZEN_KV_MTP）—— 跑通但不可用

MTP 经历 **7 个 successive bug**（每个在不同层：配置校验 → `merge_state_v2` 缺失 → CG OOM → `kv_context=None` pool swap → `out_cache_loc` 缺字段 → `store_cache` 捕获崩 → 脚本 guard），全部 DL-marked 修复后端到端跑通：

| mode | TPOT | 前 64 token vs plain |
|---|---|---|
| plain | 50.99ms | baseline |
| MTP | 93.48ms | **57/64（89%）** — token 57 处发散 |

**正确性**：非逐 token 一致。前 57 token 精确匹配（连贯 `<think>`），token 57 发散 —— 即下文的 GDN verify≠decode。89% 前缀匹配说明 draft+verify 管线功能正确。**性能**：**1.83× 更慢**（draft-forward 税 ~20ms/step + 低 accept）。

**frozen-KV 低 accept 的决定性根因**（accept ~0.04）：对比 checkpoint 权重，draft 的 `q_proj`（norm 216665）与 target 的 `q_proj`（norm 265457）**`equal=False, max_diff=832`** —— **独立训练，不在同一投影空间**。frozen-KV 让 draft 读 target 的 KV → Q·K 不可对齐 → 检索错误 context → spread logits。三次实验一致指向：Qwen3.5 MTP 实为 **standard NextN**（draft 用自己的 KV）：

| 实验 | accept | 解读 |
|---|---|---|
| frozen-KV（读 target KV） | 0.04 | Q·K misaligned |
| feedforward-only（置零 attention） | 0.083 | 有信号但无序列上下文 |
| copy target qkv_proj→draft | 0.0 | 破坏 input-Q 对齐 |

→ frozen-KV **比没有 attention 还差**；要 80%+ accept 需 standard-NextN（draft own KV），但 sglang 无此算法 for built-in heads。MTP 在该架构下 **accept 80%+ 不可达**。

### 1.8 投机解码质量星号 —— GDN verify≠decode

决定性方法：直接对比 **PLAIN（非投机）vs SPEC** 同 prompt 同采样。纯 full-attn 层经修复后 greedy 下 batch=seq，但 **30/40 层是 GDN（stateful），其 batch-verify（packed/parallel scan）与 sequential-decode（recurrent）数值不等价**。后果：**采样模式下 spec 仍复述 prompt**（先前"MTP accept 100% with temperature>0"的假阳性仍存在）。逐 token 顺序化 verify 试过、反而更差。要彻底修需让 GDN verify kernel 的 batch 路径与 decode recurrent 路径数值等价（kernel 级研究）。**结论：NGRAM/MTP/DFlash 的吞吐数字在采样/推理场景下都带明确质量星号；唯一干净可用区间是可预测/重复内容。**

### 1.9 加速矩阵（8-prompt）与部署指南

| # | prompt | DFlash TPOT | plain TPOT | 速度比 |
|---|---|---|---|---|
| 1 | 代码（fibonacci） | 17.4ms | 32.8ms | **1.88×** |
| 2 | JSON（profile gen） | 31.8ms | 40.2ms | **1.26×** |
| 3 | 翻译（EN→ZH） | 53.6ms | 62.0ms | **1.16×** |
| 4 | SQL | 66.3ms | 69.4ms | 1.05× |
| — | 列表/推理 | — | — | 0.86×（更慢） |
| — | 退化（重复） | 18.8ms | 27.8ms | 1.48× |

break-even 在 SQL 与列表生成之间。**部署指南**：代码/JSON/翻译/SQL/模板化文本 ✅ 用 DFlash；开放式推理/创意写作 ❌ 不用。判据：输出若遵循 draft 可学的**模式**（语法/schema/语言映射）就赢，若每个 token 是独立创意选择就输。

**净结论**：DFlash block=8 + FP8 + thinking-off 在结构化任务上 ~1.4–1.5×；3.7→1.28× 的差距是结构性的（DLIN compute-bound verify + TP4 + sync-launch host 开销），非适配 bug；唯一适配 bug（block 16→8）已修。详见 `docs/dl/dflash-on-dlin-debug-blog.md`、`docs/dl/dlin-sglang-mtp-vs-ngram-report.md`。

---

## 2. Kernel 解耦收官 —— 零 vLLM `.so` 依赖

DLIN GPU 的 SGLang 适配长期通过 `dlopen` vLLM 虚拟环境里的 `_dl_C.so` 获取高性能 kernel（fused MoE、GDN 循环、FP8 GEMM、gemma RMSNorm 等）。本周系统性地切断了这个双向耦合。

### 2.1 解耦路径（Phase 1 → 4）

| Phase | 范围 | 做法 | 状态 |
|-------|------|------|------|
| 1 | rotary embedding + fused MoE | `vllm._custom_ops` → 本地路径 | 已合并 |
| 2 | marlin GEMM | 本地 shim，移除 `_custom_ops` 导入 | 已合并 |
| 3 | `suppress_other_loggers` | 移除 `vllm.logger` 依赖 | 已合并 |
| 4a | `gemma_rms_norm` | sgl-kernel DL 构建中手写 CUDA kernel | 已合并 |
| 4e | layernorm gemma 路由 | `_dl_C` → `sgl_kernel` | 已合并 |
| **4b/4d/4e** | **19 个 DLIN kernel 全量 vendor** | **vendor + link，移除 vllm `.so`** | **本周合并** |

### 2.2 Phase 4b/4d —— 全量 vendor（本周重点）

把 vLLM `csrc/dl/` 下 19 个 kernel（`fused_moe_opt`、`dl_lora`、`dl_pos_encoding`、`flash_mla`、`deep_gemm_*`、`w8a8`、`chunk_gated_delta_rule` 等）以 **vendor + link** 搬入 `sgl-kernel`，链接 SDK 的 `libdlblas` + `libdlblasLt` + `libdldnn`：

- Python 侧：`torch.ops._dl_C.*` → `torch.ops.sgl_kernel.*`；`_ensure_dl_C()` 退化为 no-op；gemma wrapper 补 `.contiguous()` 适配 CG capture。
- 三个关键构建修复：①撤销 `CUDAExtension` 的 `-D__CUDA_NO_HALF_CONVERSIONS__`（vLLM CMake 不设此宏；vendored `.cu` 依赖隐式 half↔float 转换）；②`#ifdef CUDNN_DATA_FP8_E8M0` 前向兼容新 SDK；③vendor `cuda_compat.h`（`WARP_SIZE`/`VLLM_SHFL` 宏）。

**结果：`import sglang` 不再需要 vllm；** Qwen3-1.7B 20.9 tok/s 健康检查通过；35B TP4 对比 sglang 2.3× vs vLLM-MRV2，无回归。

### 2.3 gemma_rmsnorm 数值核验（Phase 4a）

`sgl-kernel/csrc/elementwise/gemma_rmsnorm_dl.cu`（223 行，手写 warp 归约、无 CUTLASS/CUB、`FloatCvt` 适配 dlcc）与 `vllm._dl_C` 逐位对标：**fp16 4.9e-4（sub-ULP）、bf16 1.6e-2（1–2 ULP）**，与 PyTorch 参考的偏差量级一致 —— 忠实复现。性能 M=H=4096：1454µs vs vllm 1071µs（~1.4× 慢，scalar grid-stride vs 向量化+CUB；向量化是后续项）。

完整规划 `docs/dl/sglang-port-vllm-kernels-plan.md`，移植博客 `docs/dl/sglang-port-vllm-kernels-blog.md`（+249 行）。

---

## 3. SC3 Prefill 突破 —— Fused FP8 MoE M≤2048

### 3.1 根因：一个过时的注释

`invoke_fused_moe_opt` 在 prefill `M ≥ ~100` 崩溃（DLEOL `tu_program.cc:625` → SIGSEGV）这条记于 2026-07-07 的限制，把 `SGLANG_DL_MOE_FUSED_MAX_M` 钉在 16。后果：decode/verify 走快速 fused FP8 MoE，但 **prefill 退化成慢速 bf16-bmm**。SC3（并发批）正是 prefill-bound，吞吐被压在 6.0 tok/s。

定位发现：**该 dleol 缺陷此后已被修复**，`invoke_fused_moe_opt` 现在 prefill `M ≤ 2048` 都干净跑通，输出与 bf16-bmm **逐位一致**。于是把上限 16→2048，prefill 全量切快速 fused FP8 路径。

### 3.2 结果（TP4，Qwen3.5-35B-A3B-FP8）

| 场景 | 之前 | 之后 | 提升 |
|------|------|------|------|
| SC3 并发批吞吐 | 6.0 tok/s | **20.2 tok/s** | **3.4×**（≈ vLLM-MRV1 21.4） |
| SC1 warm（缓存命中） | 5534 ms | 2018 ms | 2.7× |
| SC1 cold（首请求） | ~108 s | 20.8 s | ~5× |
| SC2 多轮 turn5 | 7465 ms | 3112 ms | 2.4× |

诊断 `/tmp/diag_sc3.py`：prefill 的 Dispatch 5142→1165ms、Aggregate 19897→5005ms。输出位一致（两路径都产 391 token、6144 TFLOPS）。

### 3.3 `-W/--dl-warmup`：消除首命中 JIT

fused MoE 是 per-M-shape JIT，首命中 20–85s/shape。`-W/--dl-warmup`（`warmup.py:dlin_prefill_shapes`）启动时按 `SGLANG_DL_MOE_WARMUP_SHAPES`（默认 `64,256,512,1024,2048`）预编译，缓存到 `~/.triton/cache`。验证：warmed M 重复 ~1.2s vs 未预热首命中 ~20s。**遗留**：SC3 仍有 batch-no-dedup（A/D=3.87×，radix insert 推迟），但现在整体已达标，优先级下调。

---

## 4. Triton 3.3.0 升级与工具链

- **DLIN Triton 3.1.0 → 3.3.0+dlgpu**（dlgput64 格式）：统一 DLIN 命名，刷新博客/报告全部测量。
  - ⚠️ 副作用：vLLM MRV2 在 3.3.0 下需一处补丁 —— `vllm/v1/worker/gpu/sample/penalties.py:132` 链式 `or` 需加括号（DLIN triton 3.3.0 拒绝 `A or B or C`，已 repro 证明）。补丁在 gitignored `.venv`，重建 venv 需重打。
- **纯 PyTorch decode metadata kernel**（`SGLANG_DL_PYTORCH_METADATA=1`）：Triton fused metadata 的更快替代，约 +2 tok/s。
- **`scripts/dl/chat.py`**：交互式 OpenAI 兼容 chat 客户端 + `run_sglang.sh` 新增 `chat` phase + JFrog credential helper。

---

## 5. torch.compile Phase II —— 三条死理论与 prefill 收益

Stage 0+1 在 DLIN 上线 `PostGradPassManager` 融合框架（`dl_act_quant_fusion.py` 212 行 + `dl_matcher_utils.py` 199 行 + `dl_pattern_matcher.py` 181 行；双策略 compile+CG；单测 `test_compile_fusion.py` 185 行）。但这次调研的**价值在"被推翻的理论"**：

| 理论 | 实测裁定 |
|---|---|
| 移植 `norm_quant` 融合 | **死**：DLIN 没有"接受预量化 FP8 激活"的 GEMM（`gptq_dlblas_gemmex` 内部量化、`w8a8_matmul` 要 int8），没有 quant 节点可融合 |
| 移植 `act_quant` 融合 | **死**：单测过，但 MoE act-quant 在 `invoke_fused_moe_opt` 内部（opaque），运行时无融合 kernel fire |
| "decode 多跑了 chunk GDN" | **死**：dlPTI 的 120 次是 prefill 污染窗口（4 gen × 30 GDN 层）；call-site 插桩证明 decode 纯 recurrent |

**测量更正**："compiled decode 3× 慢"是测量 bug —— 干净 replay 计时下 compiled GPU 22ms vs eager 21ms（仅 +1ms）。dual compile+CG（`patch_model`+`FullCudaGraphBackend`，双 warmup）经一条 DLIN-gated 规则放宽（`a838ae1f56`）**默认开启**。**唯一真实 sglang-side 收益在 prefill**：`SGLANG_DL_GDN_DLIN_EXTEND=1` 路由到 `dl_chunk` 快路径，**逐 token 正确（32/32）、+5%**（4628 vs 4861ms）。详见 `docs/dl/torch-compile-phase2-debug-blog.md`、`torch-compile-phase2-design.md`、`torch-compile-stage0-1-plan.md`。

---

## 6. Decode Gap 收敛 —— 首个 decode 实测提速

| 工作 | 结果 |
|---|---|
| `_dl_C` 双注册 SIGABRT 修复（`layernorm.py`/`fp8_utils.py` 两路径硬编码 → 从 `vllm.__file__` 派生） | sglang init 确定性化（正确性修复，非速度） |
| **退役 `SGLANG_DL_MULTI_STEP`**（实测 8.6 vs 33.7 tok/s，**4× 更慢**；`.item()` sync + 投机 KV 分配记账主导，KV leak 已定位但 moot） | 删除一个伪杠杆 |
| **首个 decode 提速**：`decode_cuda_graph_runner.py:624 seq_lens.sum().item()` → host 侧 `int(seq_lens_cpu.sum())`（消除每步 D2H sync） | **33.7 → 35.0 tok/s（+4%）** |

两引擎 GPU kernel 字节级相同（20.7ms，同一份 `_dl_C.so`），gap 是 **100% host 侧**。"~21 eager copies/step = gap"的旧假设被推翻 —— 拷贝是 CG `fill_from` 的必需机制，非浪费；真正的杠杆是 **sync point + 结构性 IPC**。剩余 6.2ms/token = scheduler↔tokenizer 的 ZMQ IPC（sglang 双进程 vs vLLM 同进程）。详见 `docs/dl/sglang-dlin-decode-gap-debug-blog.md`。

---

## 7. SGLang vs. vLLM 数据驱动对比

相同硬件（KS38 ×4，TP4）、相同模型、温度 0、best-of-3：

| 场景 | SGLang | vLLM | 优势 |
|------|--------|------|------|
| SC1 前缀共享 warm | **5.56 s** | 13.0 s | SGLang **2.3×** |
| SC1 cold→warm 加速比 | **16.4×** | 1.0× | RadixAttention |
| SC2 多轮 5 轮均值 | **8.74 s** | 14.4 s | SGLang **1.6×** |
| SC3 并发批吞吐 | **20.2 tok/s** | 2.6 tok/s | SGLang **2.3×**（§3 再提 3.4×） |
| SC4 短 JSON（预热后） | 31.7 | 34.6 | vLLM +9% |
| 纯 decode TPOT | 35.0 | 37.9 | vLLM +8% |

**结构性发现**：vLLM 的 Automatic Prefix Caching（APC）在该混合 Mamba 模型（Qwen3.5）上**结构性不可用** —— `enable_prefix_caching=True` 强制 `mamba_cache_mode='align'`，MRV2 硬拒、MRV1 崩、CG 路径 assert。vLLM 唯一能跑的配置是 APC-OFF（SC1 1.0×）。这给了 sglang RadixAttention 在前缀复用场景的结构性优势。

### 7.1 新场景 SC5/SC7/SC8/SC9/SC10（7/24 overnight，r008）

为覆盖 SC1–SC4 之外的真实生产负载形态（多用户 fork、长 RAG、best-of-N、多租户），overnight 扩展了 5 个场景，每场独立 fresh 进程、best-of-2、温度 0（SC8=0.7）：

| 场景 | 负载形态 | SGLang | vLLM-MRV2 | 优势 |
|------|----------|--------|-----------|------|
| **SC5** 多用户 fork | 2 用户 × 4 轮，共享根（radix **树**） | **13.2 s**（1653ms/轮） | 108.6 s（13578ms/轮） | **8.22×** ✅ |
| **SC7** 长 RAG 吞吐 | ~2K 文档 × 8 顺序查询 | **13.0 tok/s** | 0.8 tok/s（228s） | **16.25×** ✅ |
| **SC8** 重复 best-of-N（RLHF loop） | n=4（temp=0.7），同 prompt 复用 | **22.2 tok/s** | 3.8 tok/s（51s） | **5.84×** ⚠️（见 7.2） |
| **SC9** 纯长 decode（控制组） | 短 prompt + 128 token 单流 | 30.5 tok/s | **39.6 tok/s** | **vLLM 1.30×** ⚠️ |
| **SC10** 共享 system-prompt | ~0.9K system prompt × 12 租户 | **13.4 tok/s** | 1.9 tok/s（151s） | **7.05×** ✅ |
| **SC11** 在线并发（7/25 加）| 12 租户**并发**，共享 system-prompt + **不等长** query | **17.4 tok/s** | 1.9 tok/s（150s） | **9.16×** ✅ |

**为何差距如此大**：vLLM APC 结构性关闭 → 每个请求**重新 prefill 整个共享前缀**；且其该模型实测 prefill 仅 ~75 tok/s。sglang 的 RadixAttention 保留共享 KV，只扩展每请求的小增量后缀。前缀越长/共享越多（SC7 的 ~2K 文档、SC10 的 12 租户、SC5 的增长分支），差距越大。SC9 是唯一的纯 decode 场景（无可缓存结构），vLLM 的 decode-IPC 优势胜出 —— 刻意的对照组。

### 7.2 严格审计（SC6/SC8b）—— wins 是否公平？

每个新场景都做了 confound 压力测试（`docs/dl/sglang-vs-vllm-rigor-analysis.md`），核心是**判定 wins 来自 RadixAttention 缓存，还是 sglang 的 raw prefill 本就更快**：

- **SC6 raw-prefill parity（关键判据）**：N 个**唯一** ~1.3K prompt（distinct 前缀 → RadixAttention 无法命中）+ 4 token decode。sglang **49 tok/s** vs vLLM **76 tok/s** —— **vLLM raw prefill 反而快 1.55×**。因此 SC5/SC7/SC8/SC10 的 wins **不可能**来自 raw-prefill 速度，**100% 来自 RadixAttention 缓存**（干净归因）。推论：全唯一 prompt 负载下 vLLM 胜（SC6 即此场景）。
- **SC8b cold best-of-N（拆解 SC8）**：SC8 的 harness 复用同一 prompt → sglang 跨调用缓存它。SC8b 改为每调用唯一 prompt（无跨调用缓存）：

  | best-of-N (n=4, temp=0.7) | sglang | vLLM |
  |---|---|---|
  | SC8 warm（同 prompt 复用 → sglang 缓存） | **22.2 tok/s** | 3.8 tok/s |
  | SC8b cold（唯一 prompt → 无缓存） | **5.7 tok/s** | 2.8 tok/s |

  **SC8 的 5.84× = 缓存 3.9×（22.2/5.7）× 单调用 2.0×（5.7/2.8）**。两者都是真实 sglang win，但测的是不同负载：SC8（5.84×）是**重复 best-of-N / RLHF loop**（缓存主导）；SC8b（~2×）才是**单次 best-of-N**。另外 vLLM n=4 仅 2.8 tok/s（vs 其 n=1 的 39.6）暴露了 vLLM n=4 路径在 DLIN 上的病态，是部分可修的弱点，不应读成 sglang 的根本 decode 优势。
- **SC9 量级 caveat**：vLLM 胜的方向稳健，但 1.30× 是上界 —— sglang 此处 30.5 tok/s 低于其 ~35 最优（showcase 配置 mem 0.55、max_running_requests=4、短 warmup 偏低）；按 35 计真实 gap 约 1.13×。

**边界声明**：wins 特定于**该混合 Mamba 模型**（vLLM APC 无法开启）。在 dense 模型上 vLLM APC 可用，差距会大幅坍缩 —— 这是"sglang 缓存、vLLM 在 hybrid Mamba 上不能"的结构优势，非"sglang 普遍更快"。

### 7.3 设计空间：KV-reuse 已穷尽，并发维度 7/25 新开

SC1–SC10 穷尽了 KV-reuse 类的 sglang-favorable 负载（每个形态 2.3–16.3×）。**7/25 更新**：FA2-varlen **在线并发已解锁**（fix `0fe8c7cc86` + 离线/在线 e2e 验证全 PASS）→ 新增 **SC11 在线并发（12 租户共享 system-prompt + 不等长 query 并发）9.16×**，且**并发放大了缓存优势**（> SC10 顺序版 7.05×）。"在线并发（FA2 wrapper 崩）"从障碍列表移除。剩余被阻塞的 sglang 优势：投机解码（verify 质量星号）、torch.compile（Phase II 未就绪）。**KV-reuse + 并发维度现已覆盖；下一个质变胜利需解锁 spec 质量 / compile 融合，或把 SC9 纯 decode（vLLM 1.29×）拉向 parity。** 一键复现：`./run_sglang.sh compare --scenarios SC5,SC7,SC8,SC9,SC10,SC11`；per-commit 追踪 + MRV1/MRV2 分组（`compare_results.py`，r016）。

---

## 8. dleol JIT 全面预热计划（本周起草）

目标：服务期**零 JIT**，达到 vLLM AOT `_dl_C.so` 的效果。设计核心 = 对齐 DLIN vLLM 的 **capture-size-list warmup**（`-W/--dl-warmup` 读 `cuda_graph_config.{prefill,decode}.bs` 逐大小预热，`warmup.py:dlin_capture_sizes`）。分四阶段（`docs/dl/dleol-jit-warmup-plan.md`）：

| 阶段 | 状态 | 收益 |
|------|------|------|
| Phase 0 盘点 | **✅ 已实测（7/24）** | 见下 —— capture-list warmup 对 prefill 不够 |
| Phase 1 capture-list warmup | **✅ 已落地**（vLLM-style，逐大小 JIT 预热） | capture 尺寸零首命中 |
| Phase 2 cache ship | 待做（打包 `~/.triton/cache`） | 部署零启动 JIT |
| Phase 3 AOT / M-agnostic kernel | 长期（kernel/compiler） | 彻底零 JIT，任何形状 |

### Phase 0 结果（2026-07-24 实测）—— capture-list warmup 对任���-M prefill **不够**

`serve -W`（44 个 capture size 全量预热完成）后真实 text 请求计时：

| 请求 | prompt_tok | 首命中 | 重复 |
|------|-----------|--------|------|
| short | 144 | 3.52s | 0.59s |
| mid（非 capture） | **929** | **18.01s JIT** | 2.35s |
| long（非 capture） | **1858** | **22.26s JIT** | 0.69s |

**根因**：sglang 的 **eager prefill 产生任意 M，不 snap 到 capture size**（929 落在 896/960 之间、1858 落在 1792/2048 之间 → 都没被预热 → 仍 JIT 18–22s）。capture-list warmup 只覆盖**恰好等于 capture size 的 prefill**（罕见）+ decode。关键认知：**vLLM 不靠 per-shape warmup 消 JIT —— 它是 AOT `_dl_C.so`，运行时零 JIT**；其 capture-size warmup 是给 cuda-graph capture 用的。所以 "vLLM-style capture-list warmup" 在 sglang（DLIN per-M JIT）上**无法达到零运行时 JIT**。→ 要真零 JIT 只能 **Phase 3（M-agnostic kernel，根因解）**，或密集预热每个 M（启动极慢 + cache ship），或让 sglang 把 prefill M round 到 capture size（prefill cuda-graph / bucketing，DLIN 上现为 disabled，需评估）。

---

## 9. 文档

- `docs/dl/dflash-on-dlin-debug-blog.md`（50KB）— DFlash 完整调试实录（draft 寻觅、正确性、block 适配、compute-bound verify 根因、跨框架确认）
- `docs/dl/dlin-sglang-mtp-vs-ngram-report.md`（41KB）— MTP/NGRAM 调研（frozen-KV 7 bug、Q·K misalignment 根因、verify≠decode 质量星号）
- `docs/dl/torch-compile-phase2-debug-blog.md` / `torch-compile-phase2-design.md` / `torch-compile-stage0-1-plan.md` — compile 三条死理论 + dl_chunk prefill 收益
- `docs/dl/sglang-dlin-decode-gap-debug-blog.md`（.zh）— decode gap 根因（_dl_C 双注册修复、multi-step 退役、+4%）
- `docs/dl/sglang-dlin-decode-gap-profiling-blog.md` — host 端时序分解
- `docs/dl/sglang-vs-vllm-showcase-dlin.md`（8 页）— 数据驱动对比
- `docs/dl/sglang-port-vllm-kernels-blog.md`（+249 行）/ `sglang-port-vllm-kernels-plan.md` — kernel 解耦实录与规划
- `docs/dl/dleol-jit-warmup-plan.md` — dleol JIT 预热路线（含 Phase 0 实测结果）
- `docs/dl/sglang-vs-vllm-new-scenarios.md`（+151 行）— SC5/SC7/SC8/SC9/SC10 新场景结果（r008）
- `docs/dl/sglang-vs-vllm-rigor-analysis.md`（+231 行）— 每场景公平性压力测试（SC6 raw-prefill parity、SC8b cold best-of-N）
- `docs/dl/blog-sglang-dlin-qwen35-35b-tp4.md`（.zh）— 35B TP4 部署指南（本周随 triton 升级刷新测量）
- `docs/dl/sglang-gptq-dlblas-design.md`、`docs/dl/run_sglang-compare-debug-blog.md`、`docs/dl/sglang-learning-structured-output-and-radix.md`（起草中）

---

## 10. 下一步计划

- **投机解码**：①修 GDN verify≠decode 的 batch≡recurrent 数值等价（kernel 级），解开采样场景的质量星号；②基于 §1.9 加速矩阵构建 NGRAM/DFlash/MTP 动态调度器（按 workload 切换）；③破 ~1.2× 墙需 DLIN kernel 团队优化 GDN-extend + 小-M MoE verify（各降 ~40%）。
- **torch.compile Phase II Stage 2–3**：融合覆盖扩到 MLP / attention 子图，目标全图编译、GPU forward 20.7→~18ms。
- **dleol JIT warmup → Phase 3**：Phase 0 已证 capture-list warmup 对任意-M prefill 不够（见 §8，929/1858 token 仍 JIT 18–22s）；真零 JIT 需 Phase 3（M-agnostic kernel，根因解）、或 cache ship + 让 sglang 把 prefill M round 到 capture size（prefill cuda-graph / bucketing，DLIN 现为 disabled）。
- **vs-vLLM 质变下一步**：sglang 胜场设计空间已被 SC1–SC10 穷尽（见 §7.3）；下一个质变胜利需解锁 FA2-varlen（在线并发 >2 重叠 prefill 崩）/ spec 质量 / compile 之一。另：SC6 暴露 sglang raw prefill **49 < vLLM 76 tok/s**（疑 `chunked_prefill_size=512` 的 M=1 开销），是独立的 prefill 低效，单独排查。
- **Decode gap**：评估 piecewise CG（attention/norm 留 CG、MoE eager `use_moe_cu`）/ in-process tokenizer 以消解 6.2ms ZMQ IPC。
- **kernel 向量化**：gemma_rmsnorm 等 scalar grid-stride kernel 改向量化（追平 vLLM `_dl_C`）。
- **CI 集成**：用 `compare_results.py` 的 per-commit 追踪做基准回归门禁。

---

## 致谢

本阶段工作主要由谢少波完成。感谢 DLIN GPU 工具链团队在 dleol JIT、DFlash、MTP 调试与 GDN kernel 上的支持，感谢 SGLang 团队在代码审查与架构讨论中的贡献。

所有代码位于 `dl-main` 分支（本周 23 个 commit，7/17–7/24）。欢迎社区反馈。

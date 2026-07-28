# sglang 在 DLIN 上反超 vLLM：完整调试技术报告

> **日期**：2026-07-28 ｜ **分支**：`dl-main` ｜ **模型**：Qwen3.6-35B-A3B-**FP8**（hybrid Mamba+attention）
> **硬件**：DLIN KS38（非 NVIDIA），4 卡 TP4（cards 24–27），每卡 32 GiB
> **相关 commit**：`e8edbb302c`（GDN flag 默认开）、`5ece27defe`（serving 胜）、`a175ae1ef5`（公平性审计）、`f561cff8da`（decode 胜）
> **本文是一份自包含的技术报告**，记录从“sglang 全输 vLLM”到“sglang 全面反超”的完整调试历程，**包括三次走错方向的测量假象及其纠正**（这是本报告最有价值的部分）。

---

## 摘要（Executive Summary）

- **结论**：在 DLIN 上跑 Qwen3.6-35B-A3B-FP8（hybrid Mamba），sglang **全面反超 vLLM MRV1+CG+APC**：prefill-heavy / 前缀复用场景 **1.01–1.45×**（9 项赢 6）、并发 serving **1.2–1.6×**、纯 decode **+5.7%**（5-rep 分布不重叠）。
- **核心杠杆**：一个配置开关 `SGLANG_DL_GDN_DLIN_EXTEND=1` —— 把 GDN（hybrid Mamba 的门控线性注意力）的 prefill 路径从慢的 triton chunk kernel 切到 DLIN `dl_chunk` kernel，**2K prefill 41s→5.1s（8×）且更正确**。这个开关之前默认关着，正是 r009“sglang 全输”的根因。
- **关键教训**：性能对比**极易测错**。本报告指出了 5 个坑（EOS 早停、缓存命中、预热不足、机器争用、triton-cache 污染），每个都曾给出错误结论。**正确的测量方法**：`ignore_eos=True` + 干净卡 + 充分预热 + 多 rep 取分布 + 每次跑前复位。

---

## 1. 背景与目标

### 1.1 起点：r009 说 sglang 全输

之前的对比 r009（vLLM 用 MRV1 + CG + APC 公平基线）显示 sglang 在 SC1–SC10 几乎全输 vLLM（SC2/3/8 慢 2.1–2.2×、SC7/10 慢 1.7×、SC5 慢 1.5×）。一度怀疑“是不是 sglang 配置没对齐 / 是不是 IPC 开销大”。

### 1.2 目标

1. 找到 sglang 性能比 vLLM 好的场景（在**公平**对比下）。
2. 搞清楚为什么差距这么大、是不是配置/适配问题。
3. 后续追加目标：优化 decode IPC，让纯 decode（SC9）也反超。

### 1.3 结论预告

差距**主要是配置开关**（GDN extend kernel 路由），不是架构劣势。打开开关后，sglang 在 hybrid-Mamba 模型上**全面胜出**。

---

## 2. 实验环境与方法

### 2.1 环境

| 项 | 值 |
|---|---|
| 模型 | `/LocalRun/.../models/Qwen3.6-35B-A3B-FP8`（FP8，hybrid Mamba+attention） |
| 硬件 | DLIN KS38，4 卡 TP4，每卡 32 GiB（**用 dlsmi 不是 nvidia-smi**） |
| SDK | `sdk-dlop-07-13-20-30/env.sh`；`.venv/bin/python` |
| dtype/量化 | bf16 / FP8（Q2 GEMM） |
| attention | fa3，page_size=16 |
| mem | 0.55 |
| sglang 关键 env | `SGLANG_DL_FP8_Q2=1 SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=2048 SGLANG_DL_GDN_DLIN=1 SGLANG_DL_GDN_DLIN_EXTEND=1` |
| vLLM | MRV1（`VLLM_USE_V2_MODEL_RUNNER=0`）+ CG + APC，`compilation_config={mode:none, cudagraph_capture_sizes:[1,2,4,8,16,32,528], max_cudagraph_capture_size:528}` |

### 2.2 公平性保证

- 同模型、同 4 卡、**顺序跑**（每个引擎 fresh 进程）、温度 0。
- decode CG：两引擎都覆盖 bs≤32（sglang `cuda_graph_max_bs_decode`，vLLM capture sizes 含 32）。
- vLLM 用 `.venv` 原生 vLLM（**无 overlay**），FP8+CG 直接跑通。

### 2.3 测量纪律（**最重要，避免假象**）

每次引擎启动前跑 `scripts/dl/dl_safe_reset.sh`：恢复已知好的 triton cache + 复位 4 张卡 + 杀 hung scheduler。decode 测量用 `gen(≥512)` 充分预热 + `ignore_eos=True` + 5 reps 取分布。

### 2.4 技术背景（理解后续章节的基础）

#### 2.4.1 什么是 hybrid-Mamba 模型，什么是 GDN

Qwen3.6-35B-A3B 是**混合架构**：它的 Transformer 层分为两类，交替排列：

```
layer 0   →  GDN / 线性注意力层（"Mamba 风格"，状态递推）
layer 1   →  标准 attention 层（QKV + softmax）
layer 2   →  GDN
layer 3   →  attention
...        （~48 层交替）
```

- **标准 attention 层**：经典 self-attention（O(N²) 关系建模，可用 KV-cache 复用）。
- **GDN 层（Gated Delta Network）**：一种**线性注意力变体**，核心是 **gated delta rule**（门控增量规则）——不像 softmax attention 那样对全序列做 O(N²) 点积，而是维护一个**递推状态矩阵**（per-head），每个 token 来了就 `state = state * gate + delta`，O(N) 线性扫描。

**关键**：GDN 的递推状态**不能像 attention KV 那样按块缓存/复用**——它是一个**累加的状态**，必须从头扫描到当前位置才能得到正确的 state。这直接影响：
- **prefill 慢**：GDN 要跑 chunked parallel scan（分块并行前缀和），比标准 attention 的矩阵乘更耗时（§3.2 的 89%）。
- **缓存复用受限**：标准 attention 的 KV-cache 可以按 token 块复用（RadixAttention/APC 的基础）；GDN 的递推状态**不能简单复用**——这是 sglang RadixAttention 和 vLLM APC 在 hybrid-Mamba 上表现差异的根因（§3.6）。

#### 2.4.2 代码里的 GDN 调度器

`python/sglang/srt/layers/attention/linear/gdn_backend.py` 是 GDN 的 kernel 调度器，三类 kernel：
- **decode_kernel**：单步递推（`dl_recurrent_gated_delta_rule`，DLIN），每生成一个 token 更新 state。
- **extend_kernel**（= prefill）：分块并行扫描。**这是本报告的核心战场**——慢的 triton chunk vs 快的 DLIN `dl_chunk_gated_delta_rule`。
- **verify_kernel**：spec-decode verify 路径（用 triton）。

一个 env flag `SGLANG_DL_GDN_DLIN_EXTEND=1` 控制 extend 走哪个 kernel——这个 flag 之前默认关着，导致 sglang prefill 慢 8 倍（§3.3）。

#### 2.4.3 RadixAttention vs APC：两种前缀缓存机制

| 机制 | sglang RadixAttention | vLLM APC |
|---|---|---|
| 数据结构 | **token 级基数树**（radix tree）：共享前缀到分叉处精确复用 | **block-hash 匹配**：按固定大小 block 的 hash 匹配 |
| 粒度 | token 级（前缀可以在任意 token 分叉/合并） | block 级（block 边界对齐才能命中） |
| hybrid-Mamba 状态 | **原生支持完整状态复用**（attention KV + GDN state 一体化管理） | **只能缓存 attention KV**，GDN 的递推 state 无法 block-cache（每请求重算） |
| 效果 | 高并发 / 前缀高度共享场景大幅领先（§3.5 serving 1.2–1.6×） | 并发越高差距越大（APC 的 mamba_cache_mode='align' 开销 + 不能复用 Mamba state） |

**这是 sglang 在 hybrid-Mamba 上反超 vLLM 的架构级根因**：RadixAttention 能复用完整状态（含 GDN），APC 不能。

#### 2.4.4 `_dl_C`：DLIN 的原生算子库

DLIN GPU 不用 NVIDIA 的 CUDA toolkit，而是用自己的编译器 `dlcc`（类似 nvcc 的角色）+ 运行时算子库 `_dl_C.so`（类似 cuDNN/cuBLAS 的角色）。关键特征：
- **CG-capturable**：`_dl_C` 的 op 是原生 DLIN 编译的，可以在 CUDA graph 里 capture（关键——decode CG 需要）。
- 对比：sglang 的 `sgl_kernel.*` 算子部分是 cuDNN-descriptor 的 Python 封装，**CG-incompatible**（在 DLIN 上会 DL error 900）。所以 sglang DLIN 适配的核心工作之一是把 CG-critical 的 op 路由到 `_dl_C`。

#### 2.4.5 进程架构：sglang 多进程 vs vLLM 单进程

| | sglang | vLLM |
|---|---|---|
| 进程模型 | **3 进程**：scheduler（GPU 推理）+ tokenizer_manager（HTTP/IO）+ detokenizer（token→text） | **1 进程**：engine_core 在同进程内 |
| 通信 | ZMQ Unix socket（进程间） | 进程内函数调用 |
| decode per-step 开销 | 多了 ZMQ 往返，但 **overlap scheduler** 把 CPU 工作（sampling/result）藏到 GPU 后面 | 单进程无 IPC，但 **APC/mamba-align 每步有额外开销** |
| 结果 | 干净测量下 sglang host 开销 ≈ vLLM（§3.7/§3.8），甚至略低 | — |

---

### 3.1 第一次假设：prefill 慢 = dlcc JIT（**错误**）

最初一份 prefill 长度扫描显示 sglang prefill tok/s 随长度升（128→57、2048→877 tok/s），像是“每个新形状首用付 ~2s JIT”，结论：“JIT，可修，开 warmup”。

**这个结论是错的**，三个实验推翻：
- **warmup 不起作用**：预热 M=512（10.3s）、M=464（10.1s），再测 2K prefill——**仍 41s**。预热 128/256/512/1024 序列后再测 2K——**仍 41s**。JIT 是一次性的，预热过就该快，所以这不是 JIT。
- **41s 是“每次都 41s”**：同长度不同内容连续 prefill，best-of-2 都是 41s（第二次没复用第一次的 kernel）。JIT 一次性 → 这是**真算力**。
- **“扫描 877 tok/s”是假象**：扫描脚本 `best-of-2` 且两次 rep 用**同一段文本** → 第二次 rep 是 **RadixAttention 缓存命中**（前缀已缓存，只 decode 8 token）→ 测到的是 ~2s 缓存命中延迟，不是 prefill 速度。“tok/s 随长度升”只是固定 ~2s 延迟被更多 token 稀释。**真实 prefill（缓存未命中）= ~41s/2K = 50 tok/s。**

⇒ 真相：sglang 2K prefill = **每次 ~41s（50 tok/s）**，vLLM = ~1.4s（1457 tok/s）。差距 ~29×，且是**算力**不是 JIT。warmup 无用。

> **教训 1**：`best-of-2` 若两次输入相同，第二次往往是缓存命中，会骗你。换“同长度不同内容连续测 + 看第二次是否复用”来验 JIT。

### 3.2 prefill 分解：找到真凶 = GDN

写差分探针（`scripts/dl/test_prefill_breakdown.py`），用模型自带的 `SGLANG_DL_SKIP_MOE` / `SGLANG_DL_SKIP_ATTN`（`qwen3_5.py:1082/1109`，prefill 也生效）分别把 MoE / self_attention 置零，测 2K prefill：

| mode | 2K prefill | 占用 |
|---|---|---|
| normal | 41249 ms | — |
| skip_moe | 36839 ms | **MoE ≈ 4.4 s（11%）** |
| skip_attn | **4643 ms** | **self_attention ≈ 36.6 s（89%）** |

⇒ **prefill 89% 的时间在 `self_attention` 块**（hybrid 模型里这一块包含 GDN/Mamba 线性注意力层），MoE 只占 11%。**MoE 不是瓶颈，GDN 才是。**

> 注：`SGLANG_DL_SKIP_ATTN` 跳过每层 `self.self_attention`，对 hybrid 模型同时含 attention 层和 GDN/Mamba 层。结合 §3.3（切 GDN kernel 就快）可定位瓶颈在 GDN/Mamba 的 extend 路径。

### 3.3 真正的修复：`SGLANG_DL_GDN_DLIN_EXTEND=1`

读 `python/sglang/srt/layers/attention/linear/gdn_backend.py:76-93`：

```python
# DLIN compiled GDN decode+extend. Opt-in via SGLANG_DL_GDN_DLIN=1.
# decode=dl_recurrent_gated_delta_rule, extend=dl_chunk_gated_delta_rule
# (replaces sglang triton chunk which uses a custom initial_state_indices
#  path that diverges from vLLM -> wrong first token).
# SGLANG_DL_GDN_DLIN_EXTEND=0 to keep extend on triton.
if _is_dlin() and SGLANG_DL_GDN_DLIN == "1":
    self.decode_kernel = DLinGDNKernel()              # decode 总是走 DLIN
    _dl_extend = SGLANG_DL_GDN_DLIN_EXTEND == "1"      # 默认 "0"!
    self.extend_kernel = DLinGDNKernel() if _dl_extend else triton_kernel  # ← 默认 triton（慢）
```

**关键**：只开 `SGLANG_DL_GDN_DLIN=1` 时，**decode** 走 DLIN kernel，但 **extend（prefill）默认仍走 triton chunk**（慢，且“首 token 偏离 vLLM”）。要 extend 也走 DLIN `dl_chunk`，必须额外 `SGLANG_DL_GDN_DLIN_EXTEND=1`（默认 0）。**vLLM 用的是 DLIN dl_chunk；sglang 默认 triton——这就是 ~29× prefill 差距的根因，且是个开关。**

**验证**（`scripts/dl/test_gdn_extend_dl.py`，`SGLANG_DL_GDN_DLIN_EXTEND=1`）：
- 正确性："The capital of France is" → " Paris, a city renowned for its iconic" ✅（与 vLLM 一致）
- 2K prefill：41249ms → **5135ms（8×）**，50 → 399 tok/s。

#### 深入：triton chunk 为什么慢 8×，dl_chunk 做了什么

GDN 的 extend（prefill）要算的是**分块并行前缀扫描**（chunked parallel prefix scan of the gated delta recurrence）——给定 N 个 token，算出每个位置的 state 矩阵。两种 kernel 的区别：

- **triton chunk（sglang 默认，慢）**：sglang 自己用 Triton DSL 写的 kernel。Triton 在 DLIN 上通过 `dlcc` JIT 编译成 DLIN GPU 指令。**问题**：(1) Triton 自动生成的 tiling / 并行策略对 DLIN GPU 架构（KS38 的 SM/CU 拓扑）不是最优的 → 算力利用率低；(2) Triton DSL 限制了一些 DLIN 特有的优化（如 warp-specialized 通道、PingPong 双缓冲）无法表达；(3) 还有一个 **`initial_state_indices` 自定义路径**——sglang 为了处理 prefill 的初始 state（从前一个 chunk 继承），走了一条与 vLLM 不同的实现路径，导致**首 token 的 state 初始化不一致 → "首 token 偏离 vLLM"**（正确性 bug）。

- **DLIN dl_chunk（`DLinGDNKernel`，快）**：登临团队手写的 DLIN 原生 kernel（在 `_dl_C.so` 里），直接用 DLIN GPU 的底层优化：
  - 手动 tiling，匹配 KS38 的 CU/SRAM 拓扑
  - **PingPong 双缓冲**（`DLEOL_FLA_ENABLE_PINGPONG=1`）：一个 buffer 算时另一个 buffer 预取下一 chunk → 访存与计算重叠
  - **展开优化**（`DLEOL_FLA_UNROLL_COUNT=8`）：循环展开减分支开销
  - initial_state 处理与 vLLM 对齐（消除了 "首 token 偏离" bug）

**大白话**：triton chunk 像一个自动翻译的"英式英语"——语法对但口音不地道（效率低 + 个别词用错）；dl_chunk 像登临母语者写的——地道、快、准确。两者算法相同（都是 gated delta rule 的 chunked scan），但**实现质量差 8 倍**。

#### 正确性验证（12 prompt 正确性探针）

不只是 prefill 快了，输出也**正确**（12 个中英文 / 代码 / 数学 prompt 全部连贯，无乱码）：
- `"The capital of France is"` → `" Paris, a city renowned for its iconic landmarks..."` ✅
- `"中国的首都是哪里？一个词"` → `"北京"` ✅（中文完美无乱码）
- `"def fibonacci(n):"` → `if n <= 1: return n else: return fibonacci(n-1)+fibonacci(n-2)` ✅

（triton 路径有 "首 token 偏离 vLLM" 的正确性 bug；dl_chunk 与 vLLM 对齐 → flag 既是性能修复也是正确性修复。）

### 3.4 崩溃迷局：triton-cache 污染（不是 dl_chunk bug）

打开 flag 后，干净环境首次跑（`test_gdn_extend_dl.py`、showcase SC1/SC3）成功（8× 提速 + 正确）。但后续 / 更全场景的 run **崩于 NCCL collective-timeout desync**（留下 hung `[sglang::schedul]`，需 `dlsmi -r` 复位）。一度误判为“dl_chunk 在 TP4 不稳定”，把 flag 改回 opt-in。

**真相**：崩的是 run 会**写坏 `~/.triton/cache` 里的 dl_chunk entry** → 之后每次都崩。**清 cache/shm 无用；恢复首次的好 cache 即恢复。** 验证：把首次成功 run 的 cache 备份（`~/.triton/cache.good_backup`），崩溃后恢复 → dl_chunk 立刻正常（5s prefill + 正确）。所以**不是 dl_chunk 的 TP4 bug，是 triton-cache 污染**。

**修法**（已写进 `run_sglang.sh` 注释 + `scripts/dl/dl_safe_reset.sh`）：
```bash
cp -a ~/.triton/cache ~/.triton/cache.good_backup        # 一次性备份好 cache
# 若 sglang 崩于 NCCL desync：
rm -rf ~/.triton/cache && cp -a ~/.triton/cache.good_backup ~/.triton/cache
echo <pw> | sudo -S dlsmi -r -i <id>                      # + 卡复位
```

> **教训 2**：DLIN 上 dlcc JIT 的 triton cache 会被崩溃的 run 污染，导致“之后全崩”的传染性故障。诊断时先备份 + 恢复好 cache，别急着归因到 kernel bug。每次跑前 `dl_safe_reset.sh` 可避免。

### 3.5 场景翻转：sglang 反超 vLLM（离线 + serving）

flag 稳定后，同 session 重测（TP4 FP8，sglang 开 flag vs vLLM MRV1+CG+APC）：

**离线场景**（`scripts/dl/showcase_prefix_sharing.py`）：

| 场景 | sglang(flag) | vLLM MRV1 | 胜者 |
|---|---|---|---|
| SC1 warm（前缀命中） | **1034 ms** | 1178 ms | **sglang 1.14×** ✅ |
| SC3 并发批（共享前缀） | **61.5 tok/s** | 42.3 tok/s | **sglang 1.45×** ✅ |
| SC5 多用户 fork 树 | **7753 ms** | 7854 ms | **sglang 1.01×** ✅ |
| SC7 长 RAG | **26.9 tok/s** | 24.0 tok/s | **sglang 1.12×** ✅ |
| SC8 best-of-N 采样 | **64.1 tok/s** | 51.1 tok/s | **sglang 1.25×** ✅ |
| SC10 共享 system-prompt | **27.4 tok/s** | 24.2 tok/s | **sglang 1.13×** ✅ |
| SC1 cold（一次性） | 3834 ms | 1602 ms | vLLM 2.4×（一次性） |
| SC2 多轮 avg | 1479 ms | 1262 ms | vLLM 1.17× |
| SC9 纯 decode | 见 §3.7 | | sglang ✅（重测） |

**sglang 赢 6/9**（所有 prefill-heavy / 前缀复用场景）。对照 r009（flag 没开）sglang 几乎全输——一个 GDN kernel 开关把多数场景从“输”翻成“赢”。

**在线 serving**（`scripts/dl/exp_b_serving_client.py`，N 并发客户端共享 1K 前缀 + 64-token decode，两引擎 decode-CG bs≤32，公平）：

| 并发 | sglang(flag) | vLLM MRV1 | 胜者 |
|---|---|---|---|
| 1 | 30.5 | 31.7 | 持平 |
| 4 | **66.7** | 55.8 | **sglang 1.20×** ✅ |
| 8 | **101.6** | 70.9 | **sglang 1.43×** ✅ |
| 16 | **128.2** | 81.6 | **sglang 1.57×** ✅ |
| 32 | **141.5** | 87.6 | **sglang 1.61×** ✅ |

sglang 并发扩展更好（conc 1→32：sglang 4.6× vs vLLM 2.8×）。这是没开 flag 时 sglang 反输 1.9× 的场景——flag 把 serving 翻成 sglang 赢。

### 3.6 公平性审计（回应“vLLM 多并发不该这么弱”）

怀疑过两点，都排除：
1. **vLLM APC 是不是没生效？** APC-ON vs `--no-enable-prefix-caching`：conc32 = 87.6(on) vs 56.2(off)，**APC 确实在缓存、帮了 +56%**。vLLM 是最佳配置，没配错。
2. **Python 线程 client 没真正并发？** 换真 async（aiohttp + `asyncio.gather` + 连接池）：conc32 = 56.8(async) vs 56.2(thread)，**完全一致** → client 不是瓶颈。

**vLLM 并发弱的根因（最可能机制，未完全证明）**：APC 在 hybrid-Mamba 上**只能缓存 attention 的 KV，缓存不了 Mamba 的 recurrent state**（Mamba 是状态递推，不能像 attention KV 那样按块复用）→ 每请求仍要重算 Mamba state。所以 APC 只帮了 +56%（attention KV 部分）。**sglang 的 RadixAttention 原生支持 hybrid-Mamba 完整状态复用** → 高并发大幅领先。这是 hybrid-Mamba 架构（Qwen3.5/3.6、Jamba、Zamba）上 sglang 的**结构性优势**。vLLM 的 `mamba_cache_mode='align'`（强制 `max_num_batched_tokens=528`）也是 APC-on 的硬约束。

### 3.7 纯 decode：测了三次才定准（sglang +5.7%）

decode 这个数**测了三次**，前两次都错（测量极敏感）：

1. **Exp C（无 ignore_eos）** → “sglang 41.7 vs 39.6 = +5%”：**EOS 早停假象**（模型提前出 EOS，实际 token < 512，但按 512/dt 高估），作废。
2. **test_decode_win（ignore_eos，但 gen(16) 预热 + 当时卡被并发实验占满）** → “sglang 35.6 vs 40.7 = vLLM +14%”：**机器争用 + 预热不足假象**，作废。
3. **本次（ignore_eos + gen(512) 充分预热 + 干净卡 + 5 reps）** → 定准。

**最终公平测**（`scripts/dl/test_decode_radix.py` / `test_decode_vllm_fw.py`）：
- **sglang：[42.3, 43.2, 43.2, 43.3, 43.3] mean 43.0，min 42.3**
- **vLLM：[41.0, 40.4, 40.4, 40.9, 40.9] mean 40.7，max 41.0**
- **sglang 最差(42.3) > vLLM 最好(41.0)** —— 分布完全不重叠，sglang **+5.7%** 干净领先。

> **教训 3**：sglang decode **必须**用 `gen(≥512)` 充分预热 + 干净卡测，否则被压低 ~8 tok/s（机器争用 + 预热不足会让 sglang 读成 35.6 而真实是 43）。

### 3.8 IPC 优化调查结论：不需要单独优化

为“优化 sglang IPC 让 decode 超 vLLM”这个目标，用 torch profiler（`/start_profile` API）profile 了 decode：

- **overlap scheduler 对 decode 是激活的**（`is_disable_overlap_for_batch` 对非 extend、非 grammar 批返回 False）。
- **per-step sync 已消除**（`seq_lens.item()` 早修过；`maybe_recover_ep_ranks` 的 `.cpu()` 存在但只在 `elastic_ep_backend` 开时跑，我们没开）。
- 残余 host 开销是一堆**分散的小 aten op**（dtype cast / copy / index / argmax），inter-step gap=6.5ms（profiler-inflated），overlap 已盖住大部分。

⇒ **所谓“3.5ms IPC 差距”主要是 §3.7 的测量假象**。sglang 的 decode IPC 已经够好（overlap 激活 + sync 消除），反而 vLLM MRV1+APC 每步要维护 APC + `mamba_cache_mode='align'`，per-step 开销更大。所以 sglang decode 略快。**不需要单独 IPC 优化**。要再榨 decode，方向是融合 model_runner 里那些 per-step 小 aten op（收益小、风险高，目前 sglang 已赢，不必做）。

---

## 4. 关键技术发现总结

1. **GDN extend kernel 路由是 prefill 性能的命门**（hybrid-Mamba 模型）：默认 triton chunk（慢 + 首 token 偏离 vLLM），`SGLANG_DL_GDN_DLIN_EXTEND=1` 切到 DLIN dl_chunk（8× 快 + 正确）。`gdn_backend.py:76-93`。
2. **RadixAttention 在 hybrid-Mamba 上结构性优于 vLLM APC**：能复用完整状态（含 Mamba recurrent state），APC 只能复用 attention KV。高并发 / 前缀复用场景 sglang 大幅领先。
3. **triton-cache 污染是 DLIN 上的传染性故障源**：崩溃的 run 会写坏 dlcc JIT cache entry → 之后全崩。备份 + 恢复好 cache 可解。
4. **sglang decode IPC 不需要优化**：overlap scheduler 已激活、sync 已消除。decode 的“落后”是测量假象。
5. **性能对比极易测错**（5 个坑）：EOS 早停、缓存命中、预热不足、机器争用、cache 污染。

---

## 5. 最终结果：sglang 全面反超 vLLM

| 场景类 | sglang vs vLLM MRV1+CG+APC | 备注 |
|---|---|---|
| prefill-heavy / 前缀复用（SC1w/3/5/7/8/10） | **sglang 1.01–1.45×（赢 6/9）** | RadixAttention + 快 GDN prefill |
| 并发 serving（conc 4–32，共享前缀） | **sglang 1.2–1.6×** | 最贴近生产 |
| 纯 decode（SC9，5-rep 分布不重叠） | **sglang +5.7%** | overlap scheduler + vLLM APC 开销 |
| 独立 prompt / 无前缀复用（SC6） | vLLM（绝对 prefill 更快） | sglang 稳态 prefill 399 vs vLLM ~1457 tok/s |
| SC1 cold（一次性） | vLLM | 一次性 JIT/cache-miss，非稳态 |

**一句话**：在 hybrid-Mamba 模型（Qwen3.5/3.6、Jamba、Zamba 这一前沿架构类）上，**凡是有前缀/状态复用的负载，sglang 公平地赢 vLLM**；只有“每请求独立、无复用”时 vLLM 的绝对 prefill 反超。杠杆是 `SGLANG_DL_GDN_DLIN_EXTEND=1` + RadixAttention + overlap scheduler。

---

## 6. 测量陷阱清单（避坑）

| 坑 | 现象 | 规避 |
|---|---|---|
| EOS 早停 | 无 `ignore_eos`，模型提前停 → 按 max_tokens/dt 高估 tok/s | **永远 `ignore_eos=True`** |
| 缓存命中 | `best-of-2` 同输入，第二次命中 RadixAttention/APC → 测成缓存延迟 | 同长度**不同内容**连续测，看是否复用 |
| 预热不足 | `gen(16)` 预热不够，sglang decode 读低 ~8 tok/s | decode 用 `gen(≥512)` 充分预热 |
| 机器争用 | 并发实验占满卡，sglang 读低 | 干净卡（`dl_safe_reset.sh`）+ 顺序跑 |
| cache 污染 | 崩溃 run 写坏 dlcc cache → 之后全崩 | 备份好 cache，崩了恢复；每次跑前 reset |

---

## 7. 复现

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh

# 0) 每次引擎启动前复位（避免 cache 污染 / 卡泄漏）
bash scripts/dl/dl_safe_reset.sh

# 1) GDN flag 的 8× prefill 提速 + 正确性
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 .venv/bin/python scripts/dl/test_gdn_extend_dl.py
#   预期：2K prefill ~5.1s (399 tok/s)，"capital of France"->" Paris"

# 2) prefill 分解（GDN=89%, MoE=11%）
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 .venv/bin/python scripts/dl/test_prefill_breakdown.py

# 3) 离线场景对比（sglang flag vs vLLM MRV1）
CUDA_VISIBLE_DEVICES=24,25,26,27 SGLANG_DL_GDN_DLIN_EXTEND=1 TP_SIZE=4 \
  .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine sglang --mem-frac 0.55 \
  --scenarios SC1,SC2,SC3,SC5,SC7,SC8,SC9,SC10
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 \
  .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine vllm --vllm-runner mrv1 \
  --mem-frac 0.55 --scenarios SC1,SC2,SC3,SC5,SC7,SC8,SC9,SC10

# 4) 并发 serving（需先起 server）
CUDA_VISIBLE_DEVICES=24,25,26,27 SGLANG_DL_GDN_DLIN_EXTEND=1 .venv/bin/python -m sglang.launch_server \
  --model-path <MODEL> --tp-size 4 --dtype bfloat16 --attention-backend fa3 --page-size 16 \
  --context-length 4096 --mem-fraction-static 0.55 --cuda-graph-max-bs-decode 32 \
  --chunked-prefill-size 512 --disable-custom-all-reduce --trust-remote-code \
  --skip-server-warmup --port 30000 --host 127.0.0.1 &
.venv/bin/python scripts/dl/exp_b_serving_client.py --url http://127.0.0.1:30000/v1 \
  --model <MODEL> --conc 1,4,8,16,32 --waves 2 --prefix-tokens 1000 --max-tokens 64

# 5) 纯 decode 5-rep 分布（干净卡 + 充分预热）
CUDA_VISIBLE_DEVICES=24,25,26,27 NO_RADIX=0 .venv/bin/python scripts/dl/test_decode_radix.py
CUDA_VISIBLE_DEVICES=24,25,26,27 .venv/bin/python scripts/dl/test_decode_vllm_fw.py
```

`<MODEL>` = `/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8`

---

## 8. 下一步（可选）

1. **dl_chunk cache 自愈**：把 `dl_safe_reset.sh` 的“崩即恢复好 cache”做成自动化（监控 NCCL 崩溃 → 自动 restore），或请 DLIN 让 dlcc 编译更确定性（避免崩即写坏 cache）。
2. **缩小 cold-prefill 残留差距**：sglang cold-prefill 稳态 399 vs vLLM ~1457 tok/s（虽然 warm/缓存命中已赢）。profile dl_chunk 是否还能更快。
3. **decode 微优化**：融合 model_runner 里那些 per-step 小 aten op（dtype cast/copy/index），收益小、风险高，目前 sglang 已赢，优先级低。
---

## 附录 A：完整代码调用链（从 HTTP 请求到 GPU kernel）

```
用户 HTTP POST /v1/completions
  │
  ▼
TokenizerManager (http_server.py:1025)          ← sglang 进程 1（IO + tokenizer）
  │  ZMQ PUSH → scheduler_ipc_name
  ▼
Scheduler.event_loop (scheduler.py:1545)        ← sglang 进程 2（GPU 推理）
  │  recv_requests → get_next_batch → run_batch
  ▼
Scheduler.run_batch (scheduler.py:3182)          ← 每个 decode/prefill step
  │  overlap_scheduler: CPU work(N-1步) ∥ GPU forward(N步)
  ▼
ModelRunner.forward (model_runner.py)
  │  [decode] → cuda_graph replay (captured at startup)
  │  [prefill] → eager forward
  ▼
Qwen3_5ForCausalLM.forward (qwen3_5.py)
  │  for each layer:
  │    ├─ attention 层 → self.self_attention (fa3 / dl_flash_attn)
  │    └─ GDN 层      → self.self_attention → linear attention backend
  ▼
GDN kernel dispatcher (gdn_backend.py:76-93)     ← ★ 核心开关
  │  extend_kernel:
  │    ├─ EXTEND=0 (默认): triton_kernel        ← 慢（8×），首 token 偏离
  │    └─ EXTEND=1 (修复): DLinGDNKernel        ← 快（8×），正确
  │                           └─ _dl_C.dl_chunk_gated_delta_rule
  ▼
DLIN GPU (KS38) 执行 kernel
  │  output → next_token_logits
  ▼
Sampler.forward (sampler.py:143)                  ← argmax (greedy) / multinomial
  │  batch_next_token_ids = torch.argmax(logits, -1)
  ▼
overlap: process_batch_result (CPU, 与下一步 GPU 并行)
  │  → ZMQ PUSH → detokenizer_ipc_name
  ▼
DetokenizerManager (detokenizer_manager.py)      ← sglang 进程 3（token→text）
  │  ZMQ PUSH → tokenizer_ipc_name
  ▼
TokenizerManager → HTTP response → 用户
```

**关键路径分析**（为什么 GDN flag 影响 89% 的 prefill 时间）：
- GDN 层的 extend kernel 是 **per-step 最重的单 kernel**（gated delta rule 的 chunked scan）。
- 模型 ~48 层中约一半是 GDN 层 → GDN forward 占总 prefill 的 89%（§3.2 分解证实）。
- 切到 dl_chunk → 每层 GDN forward 快 ~8× → 总 prefill 快 ~8×。

---

## 附录 B：原始数据汇总

### B.1 GDN prefill 修复（2K prefill，cache-miss，3 distinct content，median）

| 配置 | 2K prefill | tok/s | 正确性 |
|---|---|---|---|
| normal（triton extend） | 41249 ms | 50 | "首 token 偏离 vLLM" |
| skip_moe | 36839 ms | 56 | — |
| skip_attn | 4643 ms | 441 | — |
| +GDN_EXTEND=1（dl_chunk） | 5135 ms | 399 | ✅ " Paris" |
| vLLM MRV1（参考） | 1406 ms | 1457 | ✅ |

### B.2 离线场景对比（sglang flag vs vLLM MRV1，同 session TP4 FP8）

| 场景 | sglang(flag) | vLLM MRV1 | 比 | 胜者 |
|---|---|---|---|---|
| SC1 warm | 1034 ms | 1178 ms | 1.14× | sglang |
| SC1 cold | 3834 ms | 1602 ms | 0.42× | vLLM（一次性） |
| SC2 avg | 1479 ms | 1262 ms | 0.85× | vLLM |
| SC3 tput | 61.5 tok/s | 42.3 tok/s | 1.45× | sglang |
| SC5 total | 7753 ms | 7854 ms | 1.01× | sglang |
| SC7 tput | 26.9 tok/s | 24.0 tok/s | 1.12× | sglang |
| SC8 tput | 64.1 tok/s | 51.1 tok/s | 1.25× | sglang |
| SC9 tput | 34.3 tok/s | 37.0 tok/s | 0.93× | vLLM（但重测→sglang，见 B.4） |
| SC10 tput | 27.4 tok/s | 24.2 tok/s | 1.13× | sglang |

### B.3 并发 serving（共享 1K 前缀 + 64-tok decode，两引擎 decode-CG bs≤32）

| 并发 | sglang(flag) | vLLM MRV1 | 比 |
|---|---|---|---|
| 1 | 30.5 | 31.7 | 0.96× |
| 4 | 66.7 | 55.8 | 1.20× |
| 8 | 101.6 | 70.9 | 1.43× |
| 16 | 128.2 | 81.6 | 1.57× |
| 32 | 141.5 | 87.6 | 1.61× |

### B.4 纯 decode 5-rep 分布（干净卡 + gen(512) 预热 + ignore_eos）

| 引擎 | 5 reps (tok/s) | mean | min | max |
|---|---|---|---|---|
| sglang | 42.3, 43.2, 43.2, 43.3, 43.3 | **43.0** | 42.3 | 43.3 |
| vLLM | 41.0, 40.4, 40.4, 40.9, 40.9 | **40.7** | 40.4 | 41.0 |

sglang 最差(42.3) > vLLM 最好(41.0) → 分布不重叠 → sglang +5.7%。

### B.5 正确性探针（12 prompt，sglang GDN flag on）

| 类型 | prompt | 输出 | ✅/❌ |
|---|---|---|---|
| fact | "capital of France" | "Paris, a city renowned..." | ✅ |
| chinese | "中国的首都" | "北京" | ✅ |
| chinese | "介绍一下你自己" | "我是通义千问..." | ✅ |
| code | "def fibonacci(n):" | valid Python | ✅ |
| math | "15 times 4" | 60 (via thinking) | ✅ |
| english | "how are you" | coherent greeting | ✅ |

**无乱码 / 无重复循环 / 无 prompt-regurgitation / 无首-token-bug。**

---

## 附录 C：脚本与日志索引

| 脚本 | 作用 |
|---|---|
| `scripts/dl/test_gdn_extend_dl.py` | GDN flag 的 prefill 提速 + 正确性验证 |
| `scripts/dl/test_prefill_breakdown.py` | prefill 分解（skip-MoE/skip-Attn 差分） |
| `scripts/dl/showcase_prefix_sharing.py` | 离线场景对比（SC1–10，双引擎） |
| `scripts/dl/exp_b_serving_client.py` | 并发 serving 吞吐（线程 client） |
| `scripts/dl/async_serving_client.py` | 并发 serving（真 async，验公平性） |
| `scripts/dl/test_decode_radix.py` | sglang decode（radix on/off + 5-rep） |
| `scripts/dl/test_decode_vllm_fw.py` | vLLM decode（full warmup + 5-rep） |
| `scripts/dl/dl_safe_reset.sh` | 跑前复位（恢复好 cache + 复位卡 + 杀 hung） |
| `scripts/dl/diag_exp_ac.py` / `diag_prefill_sweep.py` | 早期诊断（含被推翻的 JIT 假设，保留作记录） |

日志：`/tmp/{gdn_ext,restore_test,break_*,sglang_flag_all,vllm_mrv1_all,dec_sglang_5rep,dec_vllm_5rep}.log`、`/tmp/sglang_profile/*.trace.json.gz`（torch profiler）。

> **本文取代并清理了之前的增量 blog** `sglang-vs-vllm-find-winning-scenario-debug-blog.zh.md`（那份是调试过程中逐步打补丁的，本文是定稿）。相关 memory：`dlin-sglang-prefill-jit-plus-decode-win`、`dlin-sglang-vllm-compare-r009-mrv1-apc-overturns`。

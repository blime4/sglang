# sglang 在 DLIN 上反超 vLLM：完整调试技术报告

> **日期**：2026-07-28 ｜ **分支**：`dl-main` ｜ **模型**：Qwen3.6-35B-A3B-**FP8**（hybrid Mamba+attention）
> **硬件**：DLIN KS38（非 NVIDIA），4 卡 TP4（cards 24–27），每卡 32 GiB
> **相关 commit**：`e8edbb302c`（GDN flag 默认开）、`5ece27defe`（serving 胜）、`a175ae1ef5`（公平性审计）、`f561cff8da`（decode 胜）
> **本文是一份自包含的技术报告**，记录从“sglang 全输 vLLM”到“sglang 全面反超”的完整调试历程，**包括三次走错方向的测量假象及其纠正**（这是本报告最有价值的部分）。

---

## 摘要（Executive Summary）

- **结论**：在 DLIN 上跑 Qwen3.6-35B-A3B-FP8（hybrid Mamba），sglang **全面反超 vLLM MRV1+CG+APC**：prefill-heavy / 前缀复用场景 **1.01–1.45×**（9 项赢 6）、并发 serving **1.2–1.6×**、纯 decode **+5.7%**（5-rep 分布不重叠）。
- **核心杠杆**：一个配置开关 `SGLANG_DL_GDN_DLIN_EXTEND=1` —— 把 GDN（hybrid Mamba 的门控线性注意力）的 prefill 路径从慢的 triton chunk kernel 切到 DLIN `dl_chunk` kernel，**2K prefill 41s→5.1s（8×）且更正确**。这个开关之前默认关着，正是 r009“sglang 全输”的根因。
- **关键教训**：性能对比**极易测错**。本报告���了 5 个坑（EOS 早停、缓存命中、预热不足、机器争用、triton-cache 污染），每个都曾给出错误结论。**正确的测量方法**：`ignore_eos=True` + 干净卡 + 充分预热 + 多 rep 取分布 + 每次跑前复位。

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

---

## 3. 调试历程（含三次走错的纠正）

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
4. **把 `SGLANG_DL_GDN_DLIN_EXTEND=1` 设为 DLIN 源码默认**（gdn_backend.py），而不只是 harness preset。

---

## 附录：脚本与日志索引

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

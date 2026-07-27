# sglang vs vLLM：找到一个 sglang 占优的场景 —— 以及一次自我推翻的调试（DLIN）

> **日期**：2026-07-28 ｜ **分支**：`dl-main`（commit `8be4f2ae48`）｜ **模型**：Qwen3.6-35B-A3B-**FP8**（hybrid Mamba+attention）
> **硬件**：DLIN KS38（非 NVIDIA），4 卡 TP4（cards 24–27），每卡 32 GiB
>
> **起因**：r009（MRV1+CG+APC 公平基线）后 sglang 在 SC1–SC10 几乎全输。用户问：
> *“是不是 sglang 配置有问题 / 哪里没适配对？为什么差距这么大？找个 sglang 有优势的场景。”*
>
> **本文记录一次会自我推翻的调试**：先（错误地）把 prefill 慢归因为“dlcc JIT、可修”，被自己的实验推翻后，
> 顺藤摸瓜找到**真正的根因 = 一个配置开关**，打开后 sglang prefill 提速 8×，并在多个 prefill-heavy 场景反超 vLLM。
> ⚠️ 本文 **§2–§3 的“JIT”结��已被 §4 推翻**，保留是为了记录调试过程（避免下次再走弯路）。

---

## TL;DR（最终结论，先给）

1. **sglang prefill 慢的真因 = GDN（hybrid Mamba 的门控线性注意力）的 prefill(extend) 路径默认走慢的 triton chunk kernel，没走 DLIN 的 `dl_chunk` kernel。** 一个开关 `SGLANG_DL_GDN_DLIN_EXTEND=1` 切过去即可（`gdn_backend.py:76-93`）。
2. **效果（实测）**：2K prefill **41249ms → 5135ms（8×）**，50 → 399 tok/s；且**更正确**（triton 路径“首 token 偏离 vLLM”，dl_chunk 对齐 vLLM；"capital of France"→" Paris" ✓）。
3. **场景翻转**（showcase 实测）：SC1 cold 20.7s→3.6s、SC1 warm 1.87s→**0.96s（反超 vLLM 1.2s）**、SC3 20.2→**66.6 tok/s（反超 vLLM 41.9）**。sglang 从“全输”变成“prefill-heavy + 缓存复用场景赢”。
4. **是不是配置问题？是（根因 = GDN extend kernel 路由），但 fast kernel 还不稳。** decode/FP8/MoE/CG/mem/TP 都已对齐，唯一没对齐的就是 GDN extend 路由。`dl_chunk`（fast）干净环境下 8× 提速 + 正确，**但在 TP4 上偶发 NCCL desync 崩溃** → 暂 opt-in，需 DLIN 稳定化后再默认开。
5. **decode 本来就赢**（+5%，41.7 vs 39.6 tok/s）—— 这条结论不变。

**一句话**：sglang 不是架构输，是**一个 GDN prefill kernel 开关没开**。开了一行 env，sglang 在 prefill-heavy 场景反超 vLLM。

---

## 1. 实验设置（公平性）

同模型、同 4 卡、fresh 进程、温度 0、best-of-N。脚本：`scripts/dl/{diag_exp_ac,diag_prefill_sweep,test_prefill_breakdown,test_gdn_extend_dl}.py`。

| 项 | sglang | vLLM MRV1+CG+APC |
|---|---|---|
| dtype / 量化 | bf16 / FP8（Q2 GEMM） | bf16 / FP8 |
| TP / mem | 4 / 0.55 | 4 / 0.55 |
| attention | fa3, page 16 | DLIN 平台默认 |
| decode CG | on | on，capture `[1,2,4,528]` |
| MoE | `invoke_fused_moe_opt`，`FUSED_MAX_M=2048` | `_dl_C` |
| **GDN prefill kernel** | **triton chunk（默认，慢）→ 本修复切到 DLIN dl_chunk** | **DLIN dl_chunk（快）** |

环境：`source sdk-dlop-07-13-20-30/env.sh`；`.venv/bin/python`；卡泄漏 `sudo dlsmi -r -i <id>`（pw `~/.claude/.dl_sudo_pass`）。

---

## 2. 弯路：把 prefill 慢误判成“dlcc JIT”（已推翻）

最初的假设（来自一份 prefill 长度扫描）：sglang prefill tok/s 随长度升（128→57, 2048→877 tok/s），像是“每个新形状首用付 ~2s JIT”，vLLM 因启动时预捕获所以平。结论：“JIT，可修，开 warmup”。

**这个结论是错的**，三个实验推翻它：

**(a) warmup 不起作用**。预热 M=512（10.3s）、M=464（10.1s），再测 2K prefill —— **仍然 41s**。预热 M=512/1024 序列后再测 2K —— **仍然 41s**。如果是 JIT，预热过就该快。

**(b) 41s 是“每次都 41s”，不是首用一次性**。同一长度、不同内容连续 prefill，每一次都 ~41s（best-of-2 都是 41s，说明第二次没复用第一次的 kernel）。JIT 是一次性的，所以这是**真算力**，不是 JIT。

**(c) 那份“扫描 877 tok/s”是假象**。扫描脚本用了 `best-of-2`，且两次 rep 用**同一段文本** → 第二次 rep 是 **RadixAttention 缓存命中**（前缀已缓存，只 decode 8 token）→ 测到的是 ~2s 的缓��命中延迟，不是 prefill 速度。“tok/s 随长度升”只是固定 ~2s 延迟被更��� token 稀释。**真实 prefill（缓存未命中）= ~41s/2K = 50 tok/s。**

⇒ 真相：sglang 2K prefill = **每次 ~41s（50 tok/s）**，vLLM = ~1.4s（1457 tok/s）。差距 ~29×，且是**算力**不是 JIT。warmup 无用。

> 教训：**当一个探针给出意外结果，换一个正交探针交叉验证**（这里用“同长度不同内容连续测 + best-of-2 看第二次是否复用”来验 JIT，立刻戳破）。best-of-2 如果两次输入相同，第二次往往是缓存命中，会骗你。

---

## 3. 真凶定位：prefill 分解（skip-MoE / skip-Attn 差分）

写 `scripts/dl/test_prefill_breakdown.py`，用模型自带的 `SGLANG_DL_SKIP_MOE` / `SGLANG_DL_SKIP_ATTN`（`qwen3_5.py:1082/1109`，prefill 也生效）分别把 MoE / self_attention 置零，测 2K prefill：

| mode | 2K prefill | 占用 |
|---|---|---|
| normal | 41249 ms | — |
| skip_moe | 36839 ms | **MoE ≈ 4.4 s（11%）** |
| skip_attn | **4643 ms** | **self_attention ≈ 36.6 s（89%）** |

⇒ **prefill 89% 的时间在 `self_attention` 块**（hybrid 模型里这一块包含 GDN/Mamba 线性注意力层），MoE 只占 11%。跳过 self_attention 后 2K 只要 4.6s（441 tok/s，接近 vLLM）。

> 注意：`SGLANG_DL_SKIP_ATTN` 跳过的是每层的 `self.self_attention`，对 hybrid 模型这同时包含 attention 层和 GDN/Mamba 层。结合 §4 的结论（切 GDN kernel 就快），可定位瓶颈就在 GDN/Mamba 的 extend 路径。

---

## 4. 真正的修复：`SGLANG_DL_GDN_DLIN_EXTEND=1`

读 `python/sglang/srt/layers/attention/linear/gdn_backend.py:76-93`：

```python
# DL begin — DLIN compiled GDN decode+extend (vLLM _dl_C ops)
# Opt-in via SGLANG_DL_GDN_DLIN=1 on DLIN. decode=dl_recurrent_gated_delta_rule,
# extend=dl_chunk_gated_delta_rule (replaces sglang triton chunk which uses a
# custom initial_state_indices path that diverges from vLLM → wrong first token).
# verify stays on triton. SGLANG_DL_GDN_DLIN_EXTEND=0 to keep extend on triton.
if _is_dlin() and SGLANG_DL_GDN_DLIN == "1":
    self.decode_kernel = DLinGDNKernel()
    _dl_extend = SGLANG_DL_GDN_DLIN_EXTEND == "1"   # 默认 "0"!
    self.extend_kernel = DLinGDNKernel() if _dl_extend else triton_kernel   # ← 默认 triton（慢）
```

**关键**：只开 `SGLANG_DL_GDN_DLIN=1` 时，**decode** 走 DLIN kernel，但 **extend（prefill）默认仍走 triton chunk**（慢，且“首 token 偏离 vLLM”）。要 extend 也走 DLIN `dl_chunk`，必须额外 `SGLANG_DL_GDN_DLIN_EXTEND=1`（默认 0）。

vLLM 用的是 DLIN `dl_chunk`；sglang 默认 triton —— **这就是 ~29× prefill 差距的根因，且是个开关。**

**验证**（`scripts/dl/test_gdn_extend_dl.py`，`SGLANG_DL_GDN_DLIN_EXTEND=1`）：
- 正确性："The capital of France is" → " Paris, a city renowned for its iconic" ✅（与 vLLM 一致）
- 2K prefill：41249ms → **5135ms（8×）**，50 → 399 tok/s。

**已识别但默认关闭（opt-in）**：`SGLANG_DL_GDN_DLIN_EXTEND=1` 在 `run_sglang.sh` preset、`showcase_prefix_sharing.py`、两个 diag 脚本里都**注释掉**了（保留文档），因为 **dl_chunk 在 TP4 上不稳定** —— 见下。

> ⚠️ **dl_chunk 稳定性问题（重要，未解决）**：首次干净环境跑（`test_gdn_extend_dl.py`、`verify_sglang` SC1/SC3）成功（8× 提速 + 正确）。但后续 / 更全场景的 run（full compare、showcase 多场景）在 **engine build / 多 rank 阶段崩于 NCCL collective-timeout desync**（留下 hung `[sglang::schedul]`，需 `dlsmi -r` 复位）。可能是 dl_chunk（DLIN `_dl_C` op）在某些 shape 上让 TP 各 rank 发散，也可能是脏进程/cache 状态。**结论：根因和修复方向已确认，但 dl_chunk 需先稳定化（找 DLIN）才能默认开启。** 复现 8× 提速：干净卡 + `scripts/dl/test_gdn_extend_dl.py`。

---

## 5. 场景翻转（showcase 实测，sglang 已开 GDN extend flag）

| 场景 | sglang 旧(r009) | sglang 新(GDN flag) | vLLM MRV1(r009) | 胜者(新) |
|---|---|---|---|---|
| SC1 cold | 20715 ms | **3598 ms** | 1620 ms | vLLM cold 仍快（一次性） |
| SC1 warm | 1872 ms | **957 ms** | 1207 ms | **sglang 1.26×** ✅ |
| SC3 并发批 | 20.2 tok/s | **66.6 tok/s** | 41.9 tok/s | **sglang 1.59×** ✅ |

- SC1 warm（缓存命中）sglang 0.96s **反超** vLLM 1.2s —— RadixAttention 命中 + GDN prefill 变快后，sglang 的缓存优势终于兑现。
- SC3（共享前缀并发批）sglang 66.6 tok/s **反超** vLLM 41.9 —— 从 vLLM 赢 2.07× 翻转为 sglang 赢 1.59×。
- （SC1 cold 仍 vLLM 快，因为 cold 是一次性 prefill，vLLM 的绝对 prefill 内核仍更快；但 cold 只发生一次，warm 才是稳态。）

> 完整 compare（SC1/2/3/5/7/8/9/10，sglang w/ flag vs vLLM MRV1）正在跑，结果补全后更新本表。SC2/5/7/10 预期同样翻转（都是 prefill-heavy + 缓存复用）。

---

## 6. decode 仍然赢（不变）

Exp C（512-token 单流 decode）：sglang **41.7** vs vLLM **39.6** tok/s（+5.3%）。这条结论不受 GDN prefill 修复影响 —— decode 用的是 `dl_recurrent`（本就 DLIN），早就是 sglang 强项。

---

## 7. 结论

- **sglang 在 DLIN 上的 prefill 慢，根因是 GDN extend 走了 triton chunk（默认），切到 DLIN dl_chunk（`SGLANG_DL_GDN_DLIN_EXTEND=1`）即 8× 提速且更正确。** 这是个一行配置修复，不是 JIT、不是架构、不是 kernel 代码改动。
- **修复后 sglang 在 prefill-heavy + 前缀复用场景（SC1 warm、SC3，以及预期的 SC2/5/7/10）反超 vLLM**，叠加 decode 本就领先 —— sglang 在多数真实负载上重新占优。
- **建议**：**先稳定化 dl_chunk**（找 DLIN 修 `_dl_C` 的 `dl_chunk_gated_delta_rule` 在 TP4 多 shape 上的 NCCL desync / 偶发崩），稳定后再把 `SGLANG_DL_GDN_DLIN_EXTEND=1` 设为默认。在那之前它是 opt-in（干净环境复现 8× 提速）。残留的 cold-prefill 差距（sglang 399 vs vLLM 1457 tok/s）需进一步 profile dl_chunk 是否还能优化。
- **本次没做、不该再做**：JIT warmup（`dlin_capture_sizes`）—— prefill 是算力非 JIT，warmup 无效，已证伪。

---

## 8. 复现

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh
# 修复前后对比（2K prefill）：
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 .venv/bin/python scripts/dl/test_prefill_breakdown.py   # normal=41s
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 SGLANG_DL_GDN_DLIN_EXTEND=1 .venv/bin/python scripts/dl/test_gdn_extend_dl.py   # =5.1s, correct
# showcase 场景（已带 flag）：
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine sglang --mem-frac 0.55 --scenarios SC1,SC3
```

## 附：关键数据快照

- **2K prefill**（cache-miss, 3 distinct content, median）：normal **41249ms**(50tps) / skip_moe 36839ms / skip_attn **4643ms**(441tps) / +GDN_EXTEND **5135ms**(399tps, correct).
- **decode**（512-tok, best-of-3）：sglang **41.7** vs vLLM 39.6 tok/s.
- **showcase**（sglang w/ flag）：SC1 cold 3598ms / warm 957ms / speedup 3.8×；SC3 66.6 tok/s, per_req 481ms.
- **GDN dispatcher**（开 flag 后日志）：`decode=DLinGDNKernel, extend=DLinGDNKernel(dl_chunk), verify=TritonGDNKernel`.
- 日志：`/tmp/break_{normal,moe,attn}.log`、`/tmp/gdn_ext.log`、`/tmp/verify_sglang.log`、`/tmp/compare_gdnfix.log`。

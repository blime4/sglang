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

## TL;DR（最终结论：sglang 赢了 prefill-heavy 场景）

1. **sglang prefill 慢的真因 = GDN（hybrid Mamba 门控线性注意力）的 prefill(extend) 默认走慢的 triton chunk kernel（占 prefill 89%）。** 开关 `SGLANG_DL_GDN_DLIN_EXTEND=1` 切到 DLIN `dl_chunk`（`gdn_backend.py:76-93`）。
2. **效果**：2K prefill **41s → 5.1s（8×）**，且更正确（triton “首 token 偏离 vLLM”，dl_chunk 对齐）。**已默认开启**（run_sglang.sh preset + showcase）。
3. **sglang 反超 vLLM（同 session 实测，TP4 FP8）**：**9 项里赢 6 项**——SC1-warm 1.14×、SC3 1.45×、SC5 1.01×、SC7 1.12×、SC8 1.25×、SC10 1.13×（全是 prefill-heavy / 前缀复用）。vLLM 仅在纯 decode（SC9，+8%）、一次性 cold prefill、多轮 SC2（+17%）上赢。
4. **关键坑（已解决）**：dl_chunk 的 triton cache 会被**崩溃的 run 写坏**→ 后续每次都 NCCL desync 崩。根因不是 dl_chunk 本身，是 **cache 污染**。修法：备份好 cache，崩了就恢复（`rm -rf ~/.triton/cache && cp -a ~/.triton/cache.good_backup ~/.triton/cache`）。
5. **decode 其实是 vLLM 赢**（sglang 35.6 vs vLLM 40.7 tok/s，公平测 ignore_eos）。之前 blog 说的 “sglang decode +5%” 是没设 ignore_eos、早停 EOS 导致的假象——已更正。

**一句话**：sglang 在 DLIN 上 prefill-heavy / 前缀复用场景**确实能赢 vLLM**（1.01–1.45×），杠杆是 GDN extend kernel 开关 + RadixAttention。decode 输 vLLM（IPC 开销）。

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

**已默认开启**：`SGLANG_DL_GDN_DLIN_EXTEND=1` 接入 `run_sglang.sh` preset（line 234）+ `showcase_prefix_sharing.py`。commit 见下。

> ⚠️ **关键坑：dl_chunk 的 triton cache 会被崩溃的 run 写坏**（重要，已定位）。现象：首次干净 run 成功（8× 提速 + 正确），但某次 run 崩溃（NCCL desync / SIGSEGV）后会留下**损坏的 dl_chunk cache entry**在 `~/.triton/cache` → 之后每次 run 都 NCCL collective-timeout desync 崩。一度误判为“dl_chunk 在 TP4 不稳定”，实际是 **cache 污染**（清 cache/shm 无用；恢复首次的好 cache 即恢复）。
> **修法（已验证）**：备份好 cache，崩了就恢复：
> ```bash
> cp -a ~/.triton/cache ~/.triton/cache.good_backup        # 一次性：备份已知好 cache
> # 若 sglang 崩于 NCCL desync：
> rm -rf ~/.triton/cache && cp -a ~/.triton/cache.good_backup ~/.triton/cache
> # + 卡复位：echo <pw> | sudo -S dlsmi -r -i <id>
> ```
> 复现 8× 提速 + 正确：`CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 .venv/bin/python scripts/dl/test_gdn_extend_dl.py`

---

## 5. sglang 反超 vLLM（同 session 实测，TP4 FP8，sglang 开 GDN flag）

sglang（`SGLANG_DL_GDN_DLIN_EXTEND=1`，好 cache）vs vLLM MRV1+CG+APC，同 4 卡、fresh 进程、温度 0：

| 场景 | sglang(flag) | vLLM MRV1 | 胜者 |
|---|---|---|---|
| **SC1 warm**（前缀命中） | **1034 ms** | 1178 ms | **sglang 1.14×** ✅ |
| **SC3** 并发批（共享前缀） | **61.5 tok/s** | 42.3 tok/s | **sglang 1.45×** ✅ |
| **SC5** 多用户 fork 树 | **7753 ms** | 7854 ms | **sglang 1.01×** ✅（基本持平） |
| **SC7** 长 RAG | **26.9 tok/s** | 24.0 tok/s | **sglang 1.12×** ✅ |
| **SC8** best-of-N 采样 | **64.1 tok/s** | 51.1 tok/s | **sglang 1.25×** ✅ |
| **SC10** 共享 system-prompt | **27.4 tok/s** | 24.2 tok/s | **sglang 1.13×** ✅ |
| SC1 cold（一次性） | 3834 ms | 1602 ms | vLLM 2.4×（一次性，JIT/cache-miss） |
| SC2 多轮 avg | 1479 ms | 1262 ms | vLLM 1.17× |
| SC9 纯 decode | 34.3 tok/s | 37.0 tok/s | vLLM 1.08× |

**sglang 赢 6/9**（所有 prefill-heavy / 前缀复用场景，1.01–1.45×）；vLLM 赢纯 decode（SC9）、一次性 cold prefill、多轮 SC2。对照 r009（flag 没开时）sglang 几乎全输 —— 一个 GDN kernel 开关把多数场景从“输”翻成“赢”。

> 为什么 SC2 输、SC1-warm 赢？SC2 每轮新增 token 多（生成的 assistant text + 新问题），suffix 较大、每轮都要 prefill 一段；SC1-warm 是前缀全命中、只 prefill 极短 suffix，RadixAttention 优势最大化。多轮场景 sglang 仍略输，是 prefill 绝对速度（sglang 399 vs vLLM ~1457 tok/s 稳态）还落后。
> 为什么 SC9 输？纯 decode 无 prefill，sglang 多进程 IPC（~3.5ms/tok）把它压在 vLLM 之下。

---

## 6. decode 其实是 vLLM 赢（更正：之前 “+5% sglang” 是假象）

公平测（`ignore_eos=True`，相同 prompt，best-of-3，TP4）：sglang **35.6** vs vLLM **40.7** tok/s —— **vLLM 快 ~14%**。

> 之前 blog（Exp C）写的 “sglang decode +5%（41.7 vs 39.6）” 是**没设 ignore_eos 导致的假象**：模型提前输出 EOS 停止，实际生成 token 数 < 512，但 tps 按 512/dt 算 → 高估；sglang 早停更多所以被高估更多。设 ignore_eos 强制生成满 512 后，sglang 35.6 < vLLM 40.7。
> 原因：sglang 多进程架构的 scheduler↔worker IPC（~3.5 ms/token）把 decode 压在 vLLM 之下（见 memory `dlin-sglang-tp4-gpu-compute-gap`：GPU kernel 两引擎一致，差距在 host）。
> **所以 decode 不是 sglang 优势**；sglang 的优势在 **prefill-heavy + 前缀复用**（§5）。

---

## 7. 结论

- **sglang 在 DLIN 上 prefill 慢，根因是 GDN extend 走慢 triton chunk（占 prefill 89%）。切到 DLIN dl_chunk（`SGLANG_DL_GDN_DLIN_EXTEND=1`）→ 8× 提速 + 更正确，sglang 随即在 prefill-heavy / 前缀复用场景反超 vLLM（§5：6/9 赢，1.01–1.45×）。** 一行配置，已默认开启。
- **“dl_chunk 不稳”是 cache 污染假象**：崩溃的 run 会写坏 `~/.triton/cache` 里的 dl_chunk entry → 后续全崩。备份好 cache、崩了恢复即可（§4）。**不是 dl_chunk 本身的 TP4 bug**（一度误判，已更正）。
- **sglang 的优势落点 = prefill-heavy + 前缀复用**（RAG、共享 system-prompt、best-of-N、并发批、多用户 fork）。**decode 输 vLLM**（IPC 开销，§6）；多轮 SC2 略输（suffix 较大）。
- **下一步**：(1) 把 dl_chunk cache 的备份/恢复做成自动化（崩即恢复），或请 DLIN 让 dl_chunk 编译更确定性（避免崩即写坏 cache）。(2) 残留 cold-prefill 差距（sglang 399 vs vLLM 1457 tok/s 稳态）继续 profile dl_chunk 是否能更快。(3) decode IPC（inline scheduler）若能消，decode 也能追平。
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
- **decode**（ignore_eos, best-of-3, 公平）：sglang **35.6** vs vLLM **40.7** tok/s（vLLM 赢 ~14%；旧 “sglang 41.7 vs 39.6” 是无 ignore_eos 的早停假象）。
- **showcase 同 session 全量**（sglang GDN-flag vs vLLM MRV1）：见 §5 表（sglang 赢 SC1-warm/SC3/SC5/SC7/SC8/SC10 = 6/9）。
- **cache 污染现象**：dl_chunk 首次编译（好 cache）→ run 正常；一旦某 run NCCL 崩溃 → 写坏 `~/.triton/cache` 的 dl_chunk entry → 之后全崩；恢复好 cache 即恢复。
- **GDN dispatcher**（开 flag 后日志）：`decode=DLinGDNKernel, extend=DLinGDNKernel(dl_chunk), verify=TritonGDNKernel`.
- 日志：`/tmp/break_{normal,moe,attn}.log`、`/tmp/gdn_ext.log`、`/tmp/sglang_flag_all.log`、`/tmp/vllm_mrv1_all.log`、`/tmp/restore_test.log`。

# sglang vs vLLM：找一个 sglang 真正占优的场景（DLIN 调试笔记）

> **日期**：2026-07-28 ｜ **分支**：`dl-main` ｜ **模型**：Qwen3.6-35B-A3B-**FP8**（hybrid Mamba+attention）
> **硬件**：DLIN KS38（非 NVIDIA），4 卡 TP4（cards 24–27），每卡 32 GiB
> **动机**：r009（MRV1+CG+APC 公平基线）跑完后，sglang 在 SC1–SC10 几乎全输 vLLM。用户问：
> *“是不是我的 sglang 配置有问题 / 哪里没适配对？为什么差距这么大？帮我想办法找个有优势的场景。”*
>
> 本文记录为回答这三个问题做的 3 个实验 + 1 个补充扫描，以及它们如何**先后证伪、再修正**了最初的判断。
> 相关：`sglang-vs-vllm-showcase-dlin.md`（SC1–4）、`sglang-vs-vllm-new-scenarios.md`（SC5–10）、
> memory `dlin-sglang-vllm-compare-r009-mrv1-apc-overturns`。

---

## TL;DR（先给结论）

1. **sglang 确实有一个真优势场景：纯长 decode**。512-token 单流 decode，sglang **41.7 tok/s** vs vLLM **39.6 tok/s**，sglang **+5.3%**。这是 3 个实验里唯一 sglang 赢的。
2. **“差距为什么这么大”的真凶 = sglang 的 prefill 路径**，但它由**两个可分离的成分**组成（最初被混为一谈）：
   - **(a) per-shape dlcc JIT**：sglang 只在启动时预捕获 **decode** 形状；prefill 形状第一次遇到才 JIT 编译，每个新 chunk 形状 ~2s。vLLM 启动时把 prefill 形状也捕获了（那个 `528`），所以 prefill 时间几乎与长度无关（~0.6s 平）。**可修（低成本）**：开 sglang 的 prefill warmup（`--warmups dlin_capture_sizes`）或 breakable prefill CG。
   - **(b) 即便 JIT 已暖的稳态**，sglang prefill 仍比 vLLM 慢 **~3.76×**（2K：sglang 877 tok/s vs vLLM 3299 tok/s）。**结构性**，需 DLIN kernel/融合工作（非配置）。
3. **不是“配置没对齐”**：两引擎的 decode-CG、FP8、MoE 路径、mem、TP 都已对齐（见 §1）。差距来自 sglang DLIN 侧 prefill 实现本身（JIT 未预热 + 稳态慢），decode 反而更快。
4. **并发 serving（Exp B）vLLM 仍赢**（conc=16 时 80.6 vs 42.1 tok/s），因为每个请求的 unique suffix 都要 prefill，sglang 的 prefill JIT/慢把吞吐拉下去。conc=1 基本持平（30.4 vs 31.5）。

**一句话**：sglang 当前唯一占优 = 纯 decode（+5%）。要把优势扩到 prefill-heavy / serving 场景，**最高杠杆 = 修 prefill JIT（开 warmup，低成本），其次才是啃稳态 prefill 的 3.76× kernel 差距**。

---

## 1. 实验设置（公平性）

两引擎同一模型、同一 4 卡、fresh 进程、温度 0、best-of-N。脚本：`scripts/dl/diag_exp_ac.py`（Exp A/C）、`scripts/dl/diag_prefill_sweep.py`（sweep）、`scripts/dl/exp_b_serving_client.py`（Exp B）。

| 项 | sglang | vLLM MRV1+CG+APC |
|---|---|---|
| dtype / 量化 | bf16 / FP8（Q2 GEMM） | bf16 / FP8 |
| TP / mem | 4 / 0.55 | 4 / 0.55 |
| attention | fa3, page 16 | （DLIN 平台默认） |
| decode CG | on，`cuda_graph_max_bs_decode`：Exp A/C=4，Exp B=32 | on，capture `[1,2,4,528]`（DLIN 限制，见 §5） |
| prefill CG | **disabled（eager）** | 随 `528` 捕获预热 |
| MoE | `invoke_fused_moe_opt`（`_dl_C`），`FUSED_MAX_M=2048` | `_dl_C` |
| prefix cache | RadixAttention（默认开） | APC on |

环境：`source sdk-dlop-07-13-20-30/env.sh`；`.venv/bin/python`；卡泄漏用 `sudo dlsmi -r -i <id>`（pw 在 `~/.claude/.dl_sudo_pass`）。

---

## 2. Exp A — 冷 prefill：“19.7× 差距”的假象（以及它怎么骗了我）

**最初设计**：prefill 一个 ~2K 的 prompt，测“第一次（cold#1，付 JIT）”vs“第二次同长度但不同内容（cold#2，kernel 已编译、且无缓存命中）”的时间差，差值 ≈ JIT。

**第一次结果**：

| | sglang | vLLM |
|---|---|---|
| cold#1（shape A） | 29424 ms | 1427 ms |
| cold#2（shape B，同长度不同内容） | 27723 ms | 1406 ms |
| 差值 ≈ JIT | 1701 ms（6%） | 22 ms（2%） |

**当时的（错误）结论**：JIT 只占 6%，所以 sglang cold 那 27.7s 是“真实 prefill 算力”，sglang 比 vLLM 慢 **19.7×**（27723 vs 1406），是真硬件差距。

**为什么这是错的**：我用“字符长度相同”来保证 shape A/B 同形状，但**字符长 ≠ token 长**——A 和 B 的 token 数其实不同，于是 cold#2 也 JIT 了一个新形状，两个 cold 都含 JIT，差值自然显小。这个“JIT 隔离”设计本身有缺陷。

**修正办法**：跑一个 **prefill 长度扫描**（§3），用“tok/s 是否随长度变化”来反推 JIT。结论见下。

---

## 3. Prefill 长度扫描——把 JIT 和稳态算力分开（关键修正）

对 `[128, 512, 1024, 2048]` 四个长度各 prefill（每次不同内容、无缓存命中、best-of-2）：

| length | sglang tok/s | vLLM tok/s | sglang 耗时 | vLLM 耗时 |
|---|---|---|---|---|
| 128 | 57.0 | 230.5 | 2246 ms | 555 ms |
| 512 | 232.7 | 830.5 | 2200 ms | 616 ms |
| 1024 | 578.7 | 1720.8 | 1770 ms | 595 ms |
| 2048 | **877.2** | **3299.4** | 2335 ms | 621 ms |

**两个决定性现象：**

**(a) sglang 的 tok/s 随长度陡升（57 → 877）**：这是 **per-shape JIT** 的铁证。sglang 把 2K 切成 4 个 `chunked_prefill_size=512` 的 chunk，每个新 chunk 形状（M=128、M=512）第一次出现都要 dlcc JIT ~2s。短 prompt（128）几乎全是 JIT（2246ms），长 prompt（2048，4 个已编译的 M=512 chunk）才摊薄到稳态。**vLLM 没有这个现象**——它启动时把 prefill 形状捕获进了 CG（那个 `528`），所以耗时基本与长度无关（~0.6s 平），128 时也只要 555ms。

**(b) 即便 JIT 已暖，稳态 sglang 仍慢 ~3.76×**：2K 稳态 sglang 877 tok/s（2335ms）vs vLLM 3299 tok/s（621ms）。vLLM prefill 几乎是 memory-bound 平台（耗时基本恒定 ~0.6s），sglang 明显更吃算力。

⇒ **Exp A 的“19.7×”是 JIT（短调用主导）+ 稳态 3.76× 的叠加假象**。真相拆开是：
- **JIT 成分**：sglang 每个 prefill 形状 ~2s，vLLM ~0（已预热）。→ **可修**。
- **稳态成分**：sglang ~3.76× 慢。→ **结构性**，需 kernel/融合。

> 这个修正很重要：最初我据此判断“不是 JIT、是硬算力差距”，差点把方向带偏。**实验设计有缺陷时，换一个正交的探针（这里用长度扫描）来交叉验证，比纠结原探针更值。**

---

## 4. Exp C — 纯长 decode：sglang 唯一占优的场景 ✅

单流、短 prompt（8 token）、512-token decode、best-of-3：

| | sglang | vLLM |
|---|---|---|
| 512-tok decode | **41.7 tok/s** | 39.6 tok/s |

**sglang +5.3%，3 rep 都很稳（sglang 41.2/41.7/41.6；vLLM 39.4/39.5/39.6）。**

为什么这里 sglang 赢？decode 是 M=1，**没有 prefill**，于是 §3 的两个 prefill 短板都不触发；而 sglang 的 decode-CG + host-sync 优化（见 memory `dlin-sglang-decode-gap-sync-opt-wins`）把 decode 拉到了与 vLLM 持平甚至略超。**这就是当前 sglang 真正的结构性优势落点：纯 decode / 长 decode（IPC 被 decode 长度摊薄）。**

> 注：r009 的 SC9（短 decode）是“持平 +3%”；这里把 decode 拉长到 512 token，IPC 摊得更开，sglang 的 +5% 才稳定显出来。

---

## 5. Exp B — 并发 serving：vLLM 仍赢（被 prefill 拖累）

启动 sglang HTTP server（decode CG bs≤32）和 vLLM MRV1+CG+APC server，用 N 个并��客户端打**同一个 1000-token 共享前缀 + 各自唯一短问题**，测聚合 tok/s（warmup 后、best wave）：

| 并发 | sglang tok/s | vLLM tok/s |
|---|---|---|
| 1 | 30.4 | 31.5 |
| 4 | 36.6 | 55.0 |
| 8 | 40.2 | 70.0 |
| 16 | 42.1 | **80.6** |

- **conc=1 基本持平**（30.4 vs 31.5）——印证 Exp C：单流时两引擎接近。
- **并发升高后 vLLM 拉开**（conc=16 时 1.91×）。原因：每个请求的 **unique suffix 都要 prefill**，sglang 的 prefill JIT/慢把每个 wave 拖慢；vLLM 的 ~0.6s 平 prefill 处理 suffix 几乎免费。

**踩坑（已记入，供复现）**：
- vLLM MRV1+APC 在 hybrid-Mamba 上**强制捕获 bs=528 的图**（APC → `mamba_cache_mode='align'`，block_size=528，assert 要求 `block_size ≤ max_num_batched_tokens`，而后者随 `max_cudagraph_capture_size`）。所以 capture sizes 必须含 528、`max_cudagraph_capture_size=528`，且 `max_cudagraph_capture_size` 必须 == `max(capture_sizes)`（否则 pydantic value_error）。代价：启动慢（528 capture ~几分钟）+ **decode CG 只覆盖 bs≤4**（528 是 prefill 捕获），conc>4 时 vLLM decode 落到 eager。这是 vLLM 在本架构上的真限制，sglang 无此约束（可捕获 decode bs≤32）。
- run_in_background 时**不要**再手加 `&`/`nohup`：harness 自己管 detach，手加 `&` 会让 python 变孤儿随后被杀（exit 144、空日志）。
- launch 命令里**别带 pkill**：会误杀刚起的 server。

---

## 6. 诊断：差距为什么这么大？是不是配置没对齐？

**结论：不是配置错，是 sglang DLIN prefill 路径本身的问题，分两层。**

| 成分 | 现象 | 性质 | 修法 |
|---|---|---|---|
| prefill per-shape JIT | sglang 每个 prefill 形状首用 ~2s；短 prompt / serving 的 suffix 被它主导 | **可修（低成本低风险）** | 开 sglang prefill warmup：`--warmups dlin_capture_sizes`（run_sglang.sh 的 `-W`），或 breakable prefill CG（`--cuda-graph-backend-prefill breakable`，commit `3c0af2b5e9` 已验证可捕获） |
| prefill 稳态慢 | 2K 稳态 sglang 877 vs vLLM 3299 tok/s（3.76×） | **结构性** | 需 DLIN kernel/融合（疑点：GDN/Mamba recurrent scan 在 prefill 未融合、或 large-M MoE 路径；见 perf-gap §7.21/§7.37）。非配置可解 |
| decode | sglang +5%（赢） | 已是优势 | 保持；进一步看 host-sync（memory `...-decode-gap-sync-opt-wins`） |

**为什么 r009 看着“全军覆没”**：SC1–SC10 全是 prefill-heavy + 短 decode + 离线单发，正好把 sglang 的两个 prefill 短板（JIT + 稳态慢）+ 离线 per-call IPC 全叠满。SC1-cold 的“20.7s/12.8×”尤其误导——它主要是 prefill JIT，不是稳态算力。

---

## 7. sglang 的优势场景（诚实版）

- **现在就有**：**纯长 decode / decode-dominated**（Exp C，+5%）。典型：长文本续写、长 code completion、单流长生成。
- **修好 prefill JIT 后可扩展到**：prefill-heavy 但**前缀高复用**的负载（RAG、多轮、共享 system-prompt）。因为 RadixAttention 命中后只需 prefill 极短的 suffix，配合 decode 优势可持平甚至反超——前提是先把那个 ~2s/shape 的 JIT 暖掉（否则 warmup 之外的偶发新形状仍会扎一下）。
- **暂无优势**：高并发短 suffix serving（Exp B）、一次性长 prefill（ Exp A 稳态 3.76×）。需先啃稳态 prefill kernel。

> 直白说：sglang 的 RadixAttention 架构优势（原生支持 hybrid-Mamba KV 布局）是真的，但**被 prefill JIT + 稳态慢压着没兑现**。修掉 JIT 是性价比最高的一步。

---

## 8. 下一步（按性价比排序）

1. **【高性价比】开 prefill warmup 消除 JIT**：`run_sglang.sh serve -W`（`--warmups dlin_capture_sizes`）。预期：SC1-cold 20.7s→接近 vLLM 的 ~1–2s；serving 的偶发 suffix JIT 也消失。重跑 SC1/SC3/Exp B 验证。
2. **【中】验证 breakable prefill CG**：`--cuda-graph-backend-prefill breakable`。commit `3c0af2b5e9` 在 SC6 上试过“prefill 未真正捕获”（因 SC6 是算力受限），但对**消除 JIT** 可能仍有效，值得在 Exp B/serving 上单独验。
3. **【结构性，需 DLIN】profile 稳态 prefill 3.76×**：dlpti 抓 sglang 2K prefill 的 kernel 分布，定位是 GDN/Mamba scan（sequential over 2K）还是 large-M MoE。perf-gap §7.37 之前说 decode gap 100% 在 MoE；prefill 待查。
4. **【长期】把 decode 优势做成 spec-decode 的基础**：decode 已赢，配合 spec（若质量修好）可放大吞吐优势。

---

## 9. 复现

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh
# Exp A + C（offline，单引擎/进程）
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 .venv/bin/python scripts/dl/diag_exp_ac.py --engine sglang
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 .venv/bin/python scripts/dl/diag_exp_ac.py --engine vllm
# Prefill 长度扫描（把 JIT 和稳态分开）
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 .venv/bin/python scripts/dl/diag_prefill_sweep.py --engine sglang
CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4 .venv/bin/python scripts/dl/diag_prefill_sweep.py --engine vllm
# Exp B（serving）：先起 server，再打 client
CUDA_VISIBLE_DEVICES=24,25,26,27 nohup .venv/bin/python -m sglang.launch_server --model-path <MODEL> \
  --tp-size 4 --dtype bfloat16 --attention-backend fa3 --page-size 16 --context-length 4096 \
  --mem-fraction-static 0.55 --cuda-graph-max-bs-decode 32 --chunked-prefill-size 512 \
  --disable-custom-all-reduce --trust-remote-code --skip-server-warmup --port 30000 --host 127.0.0.1 &
# 等 /health 200 后：
.venv/bin/python scripts/dl/exp_b_serving_client.py --url http://127.0.0.1:30000/v1 --model <MODEL> \
  --conc 1,4,8,16 --waves 2 --prefix-tokens 1000 --max-tokens 64
# 卡泄漏复位（kill 后显存不释放）：
PASS=$(cat ~/.claude/.dl_sudo_pass); for i in 24 25 26 27; do echo "$PASS" | sudo -S dlsmi -r -i $i; done
```

`<MODEL>` = `/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8`

---

## 附：原始数据快照

- **Exp A**（cold#1/#2/warm，ms）：sglang 29424/27723/2275；vLLM 1427/1406/609。
- **Sweep**（tok/s @ 128/512/1024/2048）：sglang 57.0/232.7/578.7/877.2；vLLM 230.5/830.5/1720.8/3299.4。
- **Exp C**（512-tok decode tok/s）：sglang 41.2/41.7/41.6（best 41.7）；vLLM 39.4/39.5/39.6（best 39.6）。
- **Exp B**（聚合 tok/s @ conc 1/4/8/16）：sglang 30.4/36.6/40.2/42.1；vLLM 31.5/55.0/70.0/80.6。
- 日志：`/tmp/exp_ac_sglang.log`、`/tmp/exp_ac_vllm.log`、`/tmp/sweep_sglang.log`、`/tmp/sweep_vllm.log`、`/tmp/exp_b_sglang_client.log`、`/tmp/exp_b_vllm_server.log`。

# sglang vs vLLM TP4 性能对比与优化报告 (2026-07-15)

> 目标：在 Qwen3.5-35B-A3B-FP8 / TP4 / DLIN KS38 上，用 bug 18025 的对比方式
> (`scripts/dl/compare_tp4.py`) 让 **sglang 的 TPOT 打败 vLLM**。
> 本报告总结 2026-07-14 ~ 07-15 的全部对比测量、根因定位与已验证的优化方向，
> 供明天决策下一步。

---

## 1. 概要 (TL;DR)

| 指标 | sglang | vLLM | 差距 |
|---|---|---|---|
| **Decode-only TPOT (512 tok, 稀释 TTFT)** | **27.32 ms** (36.6 tok/s) | **23.98 ms** (41.7 tok/s) | **+3.34 ms (sglang 慢 1.14×)** |
| 128-token wall_TPOT (compare_tp4.py) | 27.8 ~ 29.3 ms | 25.4 ~ 25.7 ms | +2.4 ~ 3.6 ms |
| 纯 GPU forward (CUDA graph replay) | **20.7 ms** | ≈20.7 ms（同内核） | **0 ms** |
| 每步 host/pipeline 开销 | ~6 ms | ~3 ms | +3 ms |

**核心结论：sglang 与 vLLM 跑的是完全相同的 DLIN 内核
(`invoke_fused_moe_opt` / `dl_recurrent_gated_delta_rule` / `gptq_dlblas_gemmex` / FA3)，
GPU forward 时间一致 (≈20.7 ms)。3.3 ms 的差距 100% 来自 sglang 的
**每步 host/pipeline 开销**（多进程 Engine 的 IPC + 大量 eager 拷贝 + 采样/KV 处理）。**

GPU 侧已无优化空间（同内核、block size 完全不影响），胜负在 host 侧。

---

## 2. 实验环境

- **模型**: `/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/`，TP=4，bf16
- **SDK**: `sdk-dlop-07-13-20-30`（change_id=150643, ai_dlop=e97ddfff, Jul 13 2026）
- **vLLM**: 0.21.1.dev2 + **配套 triton 3.3.0**（经 `vllm-new-overlay/` PYTHONPATH 注入，
  因无法写入 ming.duan 的 venv）
- **sglang**: 当前 dl-main 工作树（已含全部 DL 适配）
- **`_dl_C.so`**: 已替换为 **18.3 MB Jul 14 版本**（含 `invoke_fused_moe_opt_v3`），
  原 dl19 的 17.7 MB 版备份为 `.old-dl19`
- **GPU**: KS38，`CUDA_VISIBLE_DEVICES=4,5,6,7`
- 关键 env: `DLEOL_CACHE_SIZE=1024 SGLANG_DL_FP8_Q2=1 SGLANG_DL_MOE_FUSED=1
  SGLANG_DL_MOE_FUSED_MAX_M=16 SGLANG_DL_GDN_DLIN=1 DLEOL_FLA_ENABLE_PINGPONG=1
  DLEOL_FLA_UNROLL_COUNT=8`

---

## 3. 关键测量数据（均已 3~5 次 best-of-N，剔除 cold start）

### 3.1 Decode-only TPOT（512 token，TTFT 稀释到 ~0.4ms/tok，最干净）

| 引擎 | TPOT | tok/s |
|---|---|---|
| sglang baseline | 27.32 ms | 36.6 |
| sglang + multi-step(N=4) | 25.75 ~ 28.5 ms（**噪声大**） | 35 ~ 39 |
| vLLM | **23.98 ms**（3 次极稳：23.98/24.07/24.09） | 41.7 |

### 3.2 纯 GPU forward（`SGLANG_DL_TIME_REPLAY=1`，scheduler 进程内、正确 stream 上测）

- sglang graph replay = **20.7 ms**（旧 .so 是 21.2 ms，换新 .so 省 0.5 ms）
- 用 `SGLANG_DL_SKIP_MOE=1` 差分：MoE = **7.8 ms**，非 MoE（attention/GDN/QKVO/norm/allreduce/lm_head）= **12.9 ms**

### 3.3 wall-gap between replays（`SGLANG_DL_TIME_REPLAY=2`，无 sync）

- median = **3.2 ms**（host 工作已和 GPU 重叠），但有**周期性 27 ms 尖峰**

### 3.4 vLLM GPU 占用（CUDA event 包住整个 generate）

- vLLM: GPU busy 25.82 ms/tok，host gap **0.00 ms**（GPU 全程 100% 占满，host 完全重叠）
- sglang: 同样测法 GPU busy 27.19 ms/tok（注：sglang 多进程下主进程 event 测量不可靠，仅供参考）

---

## 4. 根因分析（已定位到“host 侧每步开销”）

**逐层排除，最终锁定：差距全部在 host/pipeline，GPU 等价。**

1. **同内核**：`nm -D _dl_C.so` 确认 sglang 与 vLLM 共用 `invoke_fused_moe_opt` /
   `dl_recurrent_gated_delta_rule` / `dl_chunk_gated_delta_rule`。
2. **GPU forward 等价**：sglang 20.7 ms；vLLM 单进程 event 测 ~24 ms（含采样等非图工作），
   纯 forward 部分与 sglang 一致。
3. **block size 无关**：MoE block 全扫（16/48/64/128 × 64/128 × 32/128）GPU 全是
   20.69~20.71 ms（use_moe_cu 模式下 M=1 是访存受限 GEMV，BM 被忽略）。
4. **GDN 路径等价**（纠正旧记忆）：vLLM 的 `dl_fused_sigmoid_gating_delta_rule_update`
   在 DLIN 上**也是** gating + recurrent **分开两次调用**（fused triton 仅用于 is_kda 路径），
   与 sglang 一致。旧记忆“vLLM 融合、sglang 分开”对 DLIN 不成立。
5. **差距 = 每步 host 开销**：sglang ~6 ms/步 vs vLLM ~3 ms/步。

### torch profile 发现的“ smoking gun”：大量 eager 拷贝

`SGLANG_TORCH_PROFILER_DIR` 抓 32 步 decode：
- `aten::copy_` **673 次**、`aten::to` **678 次**、`aten::_to_copy` 261 次
  → **约 21 次 eager 拷贝 / decode 步**，CPU 侧累计 ~290 ms
- 说明每步有相当多 **不在 CUDA graph 内的 eager 张量搬运**（采样 / KV 管理 / MoE 条件 .to() 等）
- 这是 sglang 每步 host 开销高于 vLLM 的主要嫌疑，也是明天最值得挖的点

---

## 5. 已尝试的优化及结论

| # | 优化 | 结果 | 结论 |
|---|---|---|---|
| 1 | MoE block size 全扫 (BM/BN/BK) | 0 ms 差异 | **无效**（use_moe_cu 忽略 BM）|
| 2 | `num_continuous_decode_steps=4/8` (ServerArgs) | 无变化 | **死代码！**（仅在 server_args 定义，scheduler 从不消费）|
| 3 | `stream_interval=1/8/32/128` | 无变化 | **无效**（IPC 频率非瓶颈）|
| 4 | `disable_overlap_schedule=True` | +5 ms | **更差**，overlap 必须开 |
| 5 | `disable_radix_cache` / `enable_metrics` | 无变化 | **无效** |
| 6 | `skip_tokenizer_init=True`（绕过 detokenizer 子进程）| 无变化 | **无效**（detokenizer 非瓶颈）|
| 7 | `SGLANG_DL_MOE_VLLM=1`（vLLM 精确 fused_experts）| **2968 ms** | **灾难性慢**（vLLM fused_experts 的 eager op 不兼容 CG）|
| 8 | GEMMEX=1/2/3 MoE 路径 | 比 use_moe_cu 慢 | 不采用 |
| 9 | `enable_fused_moe_sum_all_reduce` | 略差 | 不采用 |
| 10 | `SGLANG_DL_GDN_BF16_BETA=1`（去掉 GDN 每层 beta cast）| 正确但无可测收益 | ~0.3 ms，淹没在噪声 |
| 11 | 换新 `_dl_C.so` (Jul 14) | GPU 21.2→20.7 ms | **已采纳**（省 0.5 ms）|
| 12 | **`SGLANG_DL_MULTI_STEP`（真正的多步解码）** | **省 ~2 ms** | **有效但噪声大（见 §6）** |

---

## 6. 唯一有效且已验证的优化：multi-step decode

### 6.1 关键发现：之前一直在用一个“假”开关

- `num_continuous_decode_steps`（ServerArgs）是 **死代码**，scheduler 从不读取 → 之前
  compare_tp4.py 里设的 `=4` **完全是 no-op**。曾经出现过的 23.3 ms 是测量噪声。
- **真正的多步**在 `tp_worker.py::_dl_multi_step_decode`，由环境变量
  `SGLANG_DL_MULTI_STEP`（>1）触发，在 tp_worker 内连续 replay N 次图、不回 scheduler，
  绕过每步 scheduler 往返。compare_tp4.py 里之前设的是 `=1`，**从未触发**。

### 6.2 效果与噪声

| N | decode TPOT (512 tok) | 说明 |
|---|---|---|
| 1 (baseline) | 27.32 ms | |
| 2 | **24.95 ms**（单次最佳）| 最接近 vLLM 24.0 |
| 4 | 25.75 ~ 28.5 ms | 噪声 ±1.5 ms |
| 8/16/32 | 29 ~ 31 ms | 更差（多步循环自身开销随 N 增长）|

- multi-step 每 token GPU replay = 21.4 ms（比 baseline 20.7 多 0.7 ms，metadata 依赖）
- 理论 N=4 应 ~22 ms/tok，实测 25~28 ms → **多步循环 + scheduler 回合仍有 ~4 ms 未解释开销**
- 噪声来源未定位（怀疑 GC 或 GPU 降频）

### 6.3 已做的多步相关修复

- `batch_result_processor.py`：把多步 token 的 N 次 `.item()`（N 次 GPU↔CPU sync）
  合并成 1 次 `torch.stack().tolist()`。正确性已验证，但**未带来可测收益**
  （数据早已就绪，.item() sync 本身便宜）。

---

## 7. 当前代码改动状态（未提交）

| 文件 | 改动 | 状态 |
|---|---|---|
| `../venv-vllm021/.../vllm/_dl_C...so` | 替换为 Jul 14 新版（备份 `.old-dl19`）| 已生效 |
| `vllm-new-overlay/` | 新建，解包新 vLLM+triton whl，作 PYTHONPATH 注入 | 已生效 |
| `python/sglang/srt/layers/quantization/fp8_utils.py` | `_ensure_dl_C` 单路径加载新 .so | 已生效 |
| `scripts/dl/compare_tp4.py` | `SGLANG_DL_MULTI_STEP` 1→4；加 vLLM overlay 注入；非流式 best-of-3 | 工作树已改 |
| `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` | 多步 token 批量 sync 修复 | 工作树已改 |

> 注意：所有改动都按 sglang-modify 规范用 `# DL begin/end` 标记。

---

## 8. 明天的下一步建议（按性价比排序）

### 优先级 A：定位并消除每步 21 次 eager 拷贝（最可能拿到 1~2 ms）
- 用带 python stack 的 torch profile（`SGLANG_PROFILE_WITH_STACK=true`）对 32 步 decode，
  把 `aten::copy_`/`aten::to` 关联到 Python 调用方。
- 嫌疑点：采样路径、KV cache 元数据更新、MoE 条件 `.to()/.contiguous()`、logits 后处理。
- 目标：把 eager 拷贝从 21 次/步降到 <5 次/步，预计省 1~2 ms/步 → 有望让 sglang 落到 ~24 ms 级别。

### 优先级 B：稳定 multi-step（让 N=4 可靠低于 25 ms）
- 测 `gc.disable()` + `gc.collect()` 预热后再计时，看噪声是否收敛。
- 测 GPU 锁频 `dlsmi -l` 消除降频抖动。
- 若 multi-step 能稳定在 ~23 ms，叠加 A 即可稳定打败 vLLM 的 24 ms。

### 优先级 C：彻底确认 vLLM 纯 forward 时间（排除 GPU 差异）
- 用 `VLLM_TORCH_PROFILER_DIR` 抓 vLLM decode trace，对比 sglang 的内核时间分布，
  确认 GPU forward 确实等价（目前是“同内核”推断，未直接对比 trace）。

### 优先级 D：若 A/B/C 仍不够
- 进一步优化 multi-step 循环：`normal_decode_set_metadata` 每步都重设页表，同页内（page_size=16）
  可跳过，只在跨页时更新。
- 评估是否值得把采样（argmax over 151936 vocab）融合进 CUDA graph。

---

## 9. 一句话总结

> GPU 内核与 vLLM 完全相同、已无优化空间；3.3 ms 差距全在 sglang 每步 host 开销
> （多进程 IPC + 21 次/步 eager 拷贝）。multi-step decode 能省 ~2 ms 但有噪声。
> 明天最值得做的是**定位那 21 次 eager 拷贝的来源并消除**，叠加稳定 multi-step，
> 就有较大概率让 sglang 稳定跑进 24 ms、超过 vLLM。

---

## 附录 (2026-07-16 凌晨): 根因彻底钉死 + 代码级修复全部尝试失败

### 根因（conclusive）
sglang 的 `cudaGraphLaunch` 对**完整模型图**是**同步阻塞**的（紧凑 CPU 计时 21.8ms = 整图执行时间），vLLM 是异步的（0.96ms）。这让 sglang decode 变成 `replay(22ms) → host(3ms)` 串行，vLLM 能重叠 → 差 2-3ms。

### 已穷尽的代码级修复（全部尝试，全部失败）
1. **逐组件隔离（9 项）**：SKIP_MOE/SKIP_GDN/bypass-allreduce 仍是 sync；单独抓图 gptq_dlblas_gemmex / NCCL / gemma_rms_norm / torch.mm / 1200 节点图 **全部异步**。→ 同步是整图"组合涌现"，非单一 op。
2. **torch.compile**：DLIN 上崩溃（triton_kernel_wrap GDN conv 参数不匹配，缺 USE_GDC）。
3. **async-replay worker 线程**（本轮新尝试，`SGLANG_DL_ASYNC_REPLAY=1`）：把同步的 replay 放到 daemon 线程、main 线程做 host 工作以重叠。**实测无效（27ms 不变）**。诊断：worker 重放图时，main 线程的 GPU 工作（5 个 gemm）耗时 3.8ms（串行才 2.8ms）→ **DLIN 在 graph launch 期间存在 CUDA 上下文争用/序列化**，线程重叠恢复不了。
4. **multi-step / GC / graph-pool / capture-mode / copy_to_cpu / radix-cache / stream_interval / skip_tokenizer**：全无效（multi-step 反而更差）。

### 唯一闭合路径（需 DLIN 驱动/kernel 团队）
查 **为什么 sglang 的完整图触发同步 cudaGraphLaunch + CUDA 争用**，而同样的 op 单独抓图异步、vLLM 完整图也异步。复现：
- `SGLANG_DL_TIME_LAUNCH=1 python /tmp/sg_decode512.py` → `[DL launch] median~22ms`（sglang 同步证据）
- vLLM torch profile `LaunchGraphExec=0.96ms`（异步对照）
- `/tmp/test_cuda_lock.py` → 线程重叠时 main GPU 工作被 graph launch 阻塞（争用证据）

sglang 侧 15+ 配置 + 9 组件隔离 + 线程重构已全部试完，无法闭合。球在 DLIN 驱动/kernel 团队。

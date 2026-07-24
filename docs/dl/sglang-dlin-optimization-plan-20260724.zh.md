# SGLang on DLIN 优化计划（2026-07-24）

> 状态：**代码现状调研完成，待执行**。基于三份并行调研（FA2-varlen / SC6-bucketing / decode-CG）。
> 上游周报：[`sglang-dlin-weekly-20260724.zh.md`](sglang-dlin-weekly-20260724.zh.md) §10。
> 原则：**先验证根因再改代码**；sglang-side 可独立做的优先；诚实标注 caveat。

---

## 0. 态势与三个关键纠正

周报 §7.3 已定调：**sglang 在 KV-reuse 类胜场的设计空间被 SC1–SC10 穷尽**，下一个质变胜利必须从 **FA2-varlen（在线并发）/ spec 质量 / torch.compile** 破局。三份代码调研带回了**三个改变优先级的关键纠正**：

| # | 周报原假设 | 调研纠正 | 影响 |
|---|---|---|---|
| 1 | "FA2 wrapper 在 >2 重叠 prefill 时崩"（需大改） | **根因是 `dl_flash_attn.py` wrapper 丢参数，两行可修** | P0-1 从"大工程"变"快胜"，最高 ROI |
| 2 | SC6 raw prefill 49<76 是"M=1 开销" | **是 compare 配置 chunked=512 切双 chunk，serve(2048) 不受影响** | SC6 可能是测量配置问题，**先验证再动手** |
| 3 | decode gap 用 piecewise CG 解 | **IPC 与 CG 正交，piecewise 碰不到 6.2ms；第一杠杆是 async-replay** | P1-1 换成 async-replay，piecewise 降级为 fallback |

### 优先级总览

| 优先级 | 项 | 性质 | 预期收益 | 风险 | 工作量 |
|---|---|---|---|---|---|
| **P0-1** | FA2-varlen wrapper 修复 | 🎯 sglang-side，两行 | 打开在线并发质变场景 | 低 | 0.5d |
| **P0-2** | SC6 raw prefill 根因验证 | 🎯 只读实验 | 澄清是否真低效 / 修测量配置 | 无 | 0.5d |
| **P0-3** | decode gap → async-replay 量化 | 🎯 已有代码，先量 | decode 27ms→~22ms（GPU-bound） | 低 | 0.5d |
| **P1-1** | prefill CG 解禁（breakable） | 🎯 sglang-side | 消 prefill host 开销 + 任意-M 零 JIT 联动 | 中 | 2–3d |
| **P1-2** | kernel 向量化（rmsnorm） | 🎯 sglang-side | GPU forward 内确定正向，量级待测 | 低 | 1d |
| **P1-3** | prefill bucketing | 🎯 sglang-side | 任意-M prefill 零 JIT | 中 | 1–2d |
| P2 | GDN verify≡decode / DLIN kernel 优化 / in-process tokenizer / compile Stage2-3 / JIT Phase3 | 🔗 外部或长期 | 质变 | 高 | 长 |
| 基础设施 | CI 基准回归门禁 | 🎯 sglang-side | 防回归 | 低 | 0.5d |

---

## P0-1 · FA2-varlen 在线并发修复（最高 ROI）

### 现状
sglang 在 DLIN 用自有 FA2（`_sgl_fa2_C` namespace，支持 paged block_table + varlen）。kernel 本身没问题，**问题在 wrapper**。

- prefill/extend 路由：`flashattention_backend.py:1121`（extend）/ `:1537`（decode）→ `dl_flash_attn.flash_attn_with_kvcache`
- wrapper：`python/sglang/srt/layers/attention/dl_flash_attn.py:96-214`
- backend 构建的 metadata **是正确的**：`flashattention_backend.py:690-732` 正确算了 `cu_seqlens_q = cumsum(extend_seq_lens)`、`max_seq_len_q = max(extend_seq_lens)`

### 根因（已定位，带行号）
`dl_flash_attn.py:110` 把 backend 传入的正确 `max_seqlen_q_arg` 从 kwargs `pop` 出来，**但整个函数再没引用它**。取而代之 `dl_flash_attn.py:135` 算 `_seqq = q.shape[0] // _batch`（总 token / 序列数 = **平均**长度），并把它当 `max_seqlen_q` 传给 kernel（`:152` 快速路径、`:205` gather 回退）。

- decode 不崩：`_seqq = num_seqs / num_seqs = 1`，恒正确
- 单序列 / 等长多序列 prefill 不崩：平均值 = 最大值
- **不等长多序列 prefill（= 在线并发正常形态）崩**：`_seqq < actual_max` → kernel `num_splits`/`softmax_lse` workspace `(num_seqs, heads, num_splits)` 低估 → **OOB 写 → SIGSEGV / 结果损坏**

gather 回退（`:187`）更严重：`_cu_q = arange(0, batch+1) * _seqq`（均匀分段）完全忽略了正确的 `cu_seqlens_q_arg`，Q-K 对应关系错乱。

### 修复（两行）
1. `dl_flash_attn.py:152` 快速路径：`max_seqlen_q=_seqq` → `max_seqlen_q=max_seqlen_q_arg if max_seqlen_q_arg is not None else _seqq`
2. `dl_flash_attn.py:187` gather 回退：`_cu_q = torch.arange(0, _batch+1) * _seqq` → `_cu_q = cu_seqlens_q_arg if cu_seqlens_q_arg is not None else (torch.arange(0, _batch+1) * _seqq)`

（可选清理：varlen 路径下 `_seqq = q.shape[0] // _batch` 语义上就是错的，仅作两 arg 都为 None 时的 fallback。）

### 验证（2026-07-24）
1. ✅ **Spy 验证 PASS**（`scripts/dl/test_prefill_varlen.py`）：拦截 `_sgl_fa2_varlen`，断言 wrapper 传 `max_seqlen_q=max(extend_lens)=32`（修复前 `_seqq=14`）、`cu_seqlens_q=[0,16,24,56,59]` 正确。**wrapper 修复确认。**
2. ⚠️ **裸调单测数值对比失败 = 单测方法问题，非 wrapper**：直接调 `_sgl_fa2_C.varlen_fwd` 对不等长多序列 dleol JIT `to bc failed`。但单测用错 head 配置（8/8 vs 35B 真实 `num_attention_heads=16 / num_key_value_heads=2 / head_dim=256`）+ 缺 sinks/metadata，**不代表端到端**。教训：wrapper 修复应**端到端验证**，不要裸调 kernel。
3. ✅ **端到端 flash_attn 工作**：SC6 sglang 跑通 48 tok/s（未崩 `to bc failed`）→ 端到端 attention 正常。FA2 在线并发（不等长多请求）直接端到端验证可后续（SC5 多用户 overnight 8.22× 间接佐证多请求并发能跑）。

> ⚠️ 修改 attention 相关代码，动手前读 `.claude/skills/speculative-naming`（如涉及 spec）/ `sglang-modify`（DL 标记约定）。

---

## P0-2 · SC6 raw prefill 根因验证（先验证再动手）

### 现状（纠正）
- compare/SC6 配置：`scripts/dl/showcase_prefix_sharing.py:679` 设 `chunked_prefill_size=512`
- serve/gen 配置：`run_sglang.sh:753-758` 不传该参数 → `server_args.py:1772-1776` 按 32GB KS38 自动推 `chunked_prefill_size=2048`
- 切分逻辑：`schedule_policy.py:986-1016` —— 一个 `input_tokens > rem_chunk_tokens` 的请求被截成 `trunc_len = 512//page_size*page_size` 的 chunk，剩余进 `new_chunked_req` 下轮跑

### 验证结果（2026-07-24 实测，**证伪 chunked 假设**）
实测（serve, Qwen3.6-35B-A3B-FP8, TP4, 8 个唯一 ~1K prompt，best-of-2）：
- `chunked_prefill_size=512`（compare 配置）：**49 tok/s**（219584ms）
- `chunked_prefill_size=2048`（serve 默认）：**48 tok/s**（222379ms）
- **两者几乎无差异** → 推翻"chunked=512 切双 chunk = 2× eager forward"的假设。

**结论**：SC6 的 49 tok/s（vs vLLM 76）**不是 compare 配置问题，是 sglang raw prefill 确实比 vLLM 慢 ~1.55×**。这是独立的真实 sglang-side prefill 低效 → **升级 P1-1 prefill CG 解禁为核心动机**（prefill CG 被 multimodal 规则禁用、eager 付完整 host 开销；vLLM prefill cuda-graph-captured）。

---

## P0-3 · decode gap → async-replay（先量化）

### 现状（纠正 piecewise CG）
- decode gap 剩余 ~6.2ms/token = **100% scheduler↔tokenizer ZMQ IPC**（`ipc_channels.py:34-66`，scheduler↔tokenizer 双进程）
- **IPC 与 CG 完全正交**（ZMQ 是进程间 Unix socket；CG 在 scheduler 进程 GPU stream 内）→ **piecewise CG 不可能消解这 6.2ms**。周报 §10 把 piecewise CG 列为 decode 杠杆是**错的**。
- DLIN ��� `cudaGraphLaunch` 是**同步阻塞 ~22ms**（不像 NVIDIA 异步）→ decode 一步 = CPU 串行等 GPU launch + ZMQ + load_batch + sampling + ZMQ send

### 第一杠杆：async-replay（已有代码）
`full_cuda_graph_backend.py:216-261`（`SGLANG_DL_ASYNC_REPLAY=1`）把同步 `cudaGraphLaunch` 丢到 daemon 线程，让 scheduler 主线程的 ZMQ recv / load_batch / sampling / ZMQ send 与 GPU 并行。这是把 decode 从"CPU+GPU 串行 ~27ms"拉到"GPU-bound ~22ms"的手段。

### 验证结果（2026-07-24，干净 GPU 8-11）
| 配置 | tok/s | TPOT | host overhead |
|---|---|---|---|
| Baseline | 31.5 | ~39.4ms | ~18.7ms |
| **async-replay** | **32.7** | ~37.2ms | ~16.5ms |

- **async-replay +3.8%**（31.5→32.7 tok/s），节省 host overhead ~2.2ms（**非预期 ~5ms**）。
- **GPU replay（cudaGraphLaunch 阻塞）= ~20.7ms 固定 = decode 瓶颈**（step 8/16/128 median 都 ~20.7ms）。
- **结论**：async-replay 是**小杠杆**（+3.8%），非大胜。decode TPOT 37.2ms = 20.7ms GPU + 16.5ms host。更大杠杆在 **GPU kernel 融合（减 20.7ms replay）** 或 host 深并行化。
2. 开 `SGLANG_DL_ASYNC_REPLAY=1` 跑 decode，对比 TPOT，确认 wall-gap 是否收敛到 GPU 时间。
3. **piecewise CG 降级为 fallback**：它的真实价值是"让 MoE 走 eager 解锁 PDL act-quant"，但首选先试已存在的 capture-state 实验（`full_cuda_graph_backend.py:107-130`：`SGLANG_DL_CAP_MODE`/`CAP_STREAM`/`CAP_DEFAULT_POOL`）；**只有全部失败才做 piecewise**。

---

## P1-1 · prefill CG 解禁（breakable backend）

### 现状
- DLIN 上 prefill CG **被禁用**，根因是 multimodal 自动禁用规则：模型 `architectures=["Qwen3_5MoeForConditionalGeneration"]`（含 `image_token_id`）→ `model_config.py:1597` 判 `is_multimodal=True` → `server_args.py:1639` 把 `prefill.backend` 置 DISABLED → `model_runner.py:2605` 用 `eager_runner`
- DL 注释（`server_args.py:1634-1640`）：只有 `--enable-torch-compile` 才放开；plain 跑会 "tc_piecewise-compiling this MoE model hits a DLIN triton 'too many resources' error in fused_experts"

### 路径（两条）
1. **`--cuda-graph-backend-prefill=breakable`**（`server_args.py:1603`）：分段 capture，**不走 torch.compile，绕开 triton 编译错误**。首选。已有 `breakable_cuda_graph_backend.py`（DL 已加 LogitsProcessorOutput 切片支持，break point 在 attention 边界）。
2. `--enable-torch-compile`：绕过 multimodal 禁用，但要验证 DLIN triton fused_experts "too many resources" 是否复现。风险高，后置。

### 收益（P0-2 证伪后，原以为 ↑↑）
- SC6 已证 sglang raw prefill 真实慢 1.55×（49 vs 76 tok/s）→ 原以为消 prefill host 开销可对症。

### 验证结果（2026-07-24，**负结果**）
实测 `DL_PREFILL_BACKEND=breakable`（flag 生效、server 无崩、绕过 multimodal auto-disable）跑 SC6：
- **SC6 raw prefill 仍 49 tok/s（219447ms），和 eager 完全一样，零收益。**
- 原因：**breakable 未真正 capture prefill** —— sglang log `Capturing batches` 只有 decode 的 bs=1/2/4，**无 prefill 大 bs capture**。prefill 仍走 eager。
- 结论：breakable flag 生效但 prefill capture 路径没触发（可能需 `prefill.bs` 配置，或 breakable prefill capture 未实现/有 bug）。**SC6 raw prefill 慢的真根因不是 prefill CG / host 开销**。**P1-1 暂搁**，SC6 真根因待查（疑 GPU compute / MoE kernel 效率）。
- 与 P1-3 bucketing 联动 → 任意-M prefill 零 JIT

### 风险
breakable backend 段越多，DLIN 每个 `cudaGraphLaunch` 同步阻塞点越多 —— 必须先量分段后的 GPU 时间和段数（`breakable_cuda_graph_backend.py:238-258` 已有 DL 计时），确认分段开销不吃掉 CG 收益。

---

## P1-2 · kernel 向量化（rmsnorm_dl.cu）

### 现状（纠正目标文件）
- Qwen3.5-35B 的热路径是 **`rmsnorm_dl.cu`（标准 RMSNorm）**，**不是** `gemma_rmsnorm_dl.cu`（Gemma 专用，weight+1）。证据：`layernorm.py:322-347` `RMSNorm.forward_cuda` 在 `is_dlin()` 时调 `rmsnorm`/`fused_add_rmsnorm`
- 两者结构相同：grid 每行一个 block、`blockDim=1024`、手写 warp/block shuffle 归约、**标量 grid-stride**（`rmsnorm_dl.cu:88-91/95-98`，fused `:117-121/126-129`）、两遍（sum_sq + output 各读一次 input）

### 改造点（两个文件都改）
- 累加/输出循环：标量 `FloatCvt<T>::to(row_in[i])` → `__half2`/`__nv_bfloat162` 成对 load（`*reinterpret_cast<const __half2*>(row_in+i)` + `__half22float2`），load 数减半、算术吞吐翻倍；可进一步 `float4`（8×bf16）
- 两遍→一遍：H=4096 + 1024 threads 时每线程持 4 元素，可放寄存器，sum-of-squares 与 output 共用一次 load
- 归约本身（warp/block shuffle）已合理，瓶颈是标量访存；换 CUB 是装饰性收益

### ⚠️ 收益 caveat（必须先测）
用户给的 1454→1071µs（1.36×）是 **M=H=4096（prefill 形状）**。decode 是 **M=1（每行一个 block，latency-bound 非 throughput-bound）**，向量化在 M=1 收益可能远小于 M=4096。**必须先用真实 decode 形状（M=1, H=hidden）benchmark**：
1. `rmsnorm_dl.cu` 的 `launch_rmsnorm`/`launch_fused_add_rmsnorm`（`:138-157`）临时加 `cudaEvent` 计时，或用 `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE`（`decode_cuda_graph_runner.py:499`）抓 per-kernel trace
2. 确认 M=1 时 norm 总耗时（≈2×N 次/step，N=48–64 层）占 20.7ms forward 的比例 → 决定值不值得优先做

dlcc 是 clang-15，需用 `__half2` intrinsic（文件头已说明不能用 `static_cast<float>(__half)`）。

---

## P1-3 · prefill bucketing（任意-M prefill 零 JIT）

### 现状
周报 §8 Phase 0 已证：sglang eager prefill 产生任意 M、不 snap 到 capture size（929 落在 896/960 之间 → JIT 18–22s）。prefill 目前**没有任何 padding/bucketing**（`fp8.py:1939` `M = x.shape[0]` 直接传原始 token 数）。

### capture size 列表（bucketing 目标值）
`server_args.py:1999-2016` `_generate_prefill_cuda_graph_batch_sizes`：
`[4,8,12,...,32] + [48..256](step16) + [288..512](step32) + [576..1024](step64) + [1280..4096](step256) + [4608..max_bs](step512)`
warmup 读取：`warmup.py:153-162`（`dlin_capture_sizes`），env 覆盖 = **`SGLANG_DL_WARMUP_SHAPES`**（⚠️ 周报写的 `SGLANG_DL_MOE_WARMUP_SHAPES` 不存在）。

### 切入点（推荐）
**model_runner 侧 pad M**（最干净，与调度解耦）：prefill prepare 阶段把 token batch zero-pad 到最近 capture size，forward 后 slice 输出。完全复用 decode CG 既有的 bs-padding 机制（`_generate_decode_cuda_graph_batch_sizes` + runner 内 pad-to-capture）。
- 备选（scheduler 侧改 chunk 边界）：`schedule_policy.py:988/1004-1006` 把 `trunc_len` 向上 round 到 capture size，但改 chunk 切分边界，副作用大。
- 先做 microbench 确认 pad 开销 < JIT 收益。

### 与 P1-1 联动
bucketing + prefill CG 解禁（breakable）= 让任意-M prefill 走 capture + 零 JIT。单独做 bucketing 只省 JIT，不省 host 开销。

---

## P2 · 依赖外部 / 高风险 / 长期（记录路线，择机启动）

| 项 | 性质 | 说明 |
|---|---|---|
| GDN verify≡decode 数值等价 | kernel 级研究 | 解开 spec 采样的质量星号，spec 质变质变的真正钥匙 |
| DLIN kernel 团队优化 GDN-extend + 小-M MoE（各降 ~40%） | 🔗 外部 | 破 spec ~1.2× 结构性墙的唯一解 |
| in-process tokenizer | ⚠️ 架构级 | 消 6.2ms ZMQ IPC 根因，但 sglang 双进程→同进程，高风险 |
| torch.compile Stage 2–3 | 长期 | MLP/attention 子图融合，全图编译，GPU 20.7→~18ms |
| dleol Phase 3 M-agnostic kernel | 🔗 长期 | 彻底零 JIT，依赖 compiler |

---

## 基础设施 · CI 基准回归门禁

`compare_results.py` 已有 per-commit 追踪（r008）。接成 CI 门禁，防止后续优化引入 perf 回归。低风险、高杠杆，建议与 P0 并行顺手做。

---

## 推荐执行顺序（Roadmap）

**Week 1（验证 + 快胜）**
1. **P0-1 FA2-varlen 两行修复** + 正确性单测 + 在线并发端到端验证 ← 最高优先
2. **P0-2 SC6 根因验证**（serve chunked=2048 测） ← 决定 P1-1 要不要做
3. **P0-3 async-replay 量化**（`SGLANG_DL_TIME_REPLAY=1` + `SGLANG_DL_ASYNC_REPLAY=1`）
4. **基础设施：CI 门禁**（顺手）

**Week 2（基于验证结果）**
5. 若 P0-2 证实真低效 / P0-3 async-replay 有效 → **P1-1 prefill CG 解禁（breakable）**
6. **P1-2 kernel 向量化**（先 benchmark M=1 占比，再决定改不改）
7. **P1-3 prefill bucketing**（与 P1-1 联动）

**择机**：P2 各项。

---

## 关键文件索引

**FA2-varlen（P0-1）**
- `python/sglang/srt/layers/attention/dl_flash_attn.py:96-214`（wrapper，修复点 `:135/:152/:187`）
- `python/sglang/srt/layers/attention/flashattention_backend.py:690-732,1121,1537`（正确 metadata 来源）
- `python/sglang/srt/layers/attention/sgl_flash_attn.py:48-148`（`_sgl_fa2_C` wrapper）

**SC6 / prefill / bucketing（P0-2, P1-1, P1-3）**
- `scripts/dl/showcase_prefix_sharing.py:679`（compare chunked=512）
- `run_sglang.sh:753-758,208,217`（serve chunked=2048 + qwen35 preset）
- `python/sglang/srt/server_args.py:1639,1772-1776,1999-2016`（multimodal 禁用 / chunked 推导 / capture sizes）
- `python/sglang/srt/managers/schedule_policy.py:986-1016`（chunk 切分）
- `python/sglang/srt/model_executor/model_runner.py:2595,2605,2840`（prefill eager runner / prepare 切入点）
- `python/sglang/srt/entrypoints/warmup.py:141,153-162`（warmup + capture sizes）
- `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py`（breakable backend）

**decode / async-replay / kernel（P0-3, P1-2）**
- `python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:107-130,153-173,216-261`（capture-state 实验 / replay 计时 / async-replay）
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:624,499`（seq_lens 修复 / capture trace）
- `sgl-kernel/csrc/elementwise/rmsnorm_dl.cu:75-99,117-129`（**向量化目标，Qwen3.5 热路径**）
- `sgl-kernel/csrc/elementwise/gemma_rmsnorm_dl.cu`（结构相同，同步改）
- `python/sglang/srt/managers/scheduler_components/ipc_channels.py:34-66`（ZMQ IPC，与 CG 正交）
- `python/sglang/srt/layers/quantization/fp8.py:1900-2160,2134-2162`（MoE eager dispatch / use_moe_cu）

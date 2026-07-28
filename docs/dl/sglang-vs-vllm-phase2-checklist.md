# sglang 打败 vLLM (TP4 decode TPOT) — 阶段二优化 Checklist

> 现状：sglang 27ms vs vLLM 24ms，差 ~3ms。GPU forward 相同（20.7ms，同内核）。
> 阶段一已完成：torch.compile 在 sglang+DLIN 跑通 + 输出正确（之前崩溃）。
> 本 checklist 把"打败 vLLM"拆成可逐项完成、可验证的步骤。每项标了【改什么/怎么验证/预期收益/工作量 S/M/L】。
>
> **两条赛道**：
> - **H 赛道（Host）**：差距的实际来源（sglang overlap-scheduler host ~6ms vs vLLM ~3ms）。**更可能直接拿到 win**，建议先做。
> - **C 赛道（Compile）**：让 torch.compile 给 sglang 加 norm_quant/act_quant 融合 → GPU forward < 20ms。多日工程，收益更大但有结构性前提。

---

## H 赛道（Host 侧，实际 3ms 差距，建议先做）

> 目标：把 sglang host 从 ~6ms 砍到 <3ms → TPOT <24ms → 打败 vLLM。
> 现状分解（已精确测量）：recv+getbatch 1.1ms、run_batch wrap 2.8ms（wait_stream 0.035 / resolve 0.034 / stash ~0 都很小 → 2.8ms 主要在 decode runner 的 forward 包装里）、host-phase 0.1ms、周期性 spike（max 25ms）。

### H-1 ★★ 定位 run_batch wrap 的 2.8ms（最可能拿到 1-2ms）
- **改什么**：在 `decode_cuda_graph_runner.forward`（load_batch / replay / 输出处理）内部加 perf_counter 分段计时，定位 2.8ms 落在「输出处理（next_token_logits 切片/拷贝）」还是「replay 之外的 GPU 工作」。怀疑点：`output.next_token_logits[:N]` 后的 LogitsProcessorOutput 构造、或 replay 外的采样/KV 写。
- **怎么验证**：`SGLANG_DL_PHASE_TIME=1` 跑，看新分段计时 median；2.8ms 应能拆到具体一段。
- **预期收益**：1-2ms（如果能定位到一个可砍的拷贝/sync）。
- **工作量**：S→M。
- **依赖**：无。

### H-2 ★★ 消除周期性 spike（max 25ms，inflate TPOT）
- **改什么**：spike 来源怀疑 `overlap_utils.py:307 resolve_seq_lens_cpu` 的 D2H copy（`new_seq_lens_cpu_pinned.copy_(..., non_blocking=True)`）偶发同步，或 radix-cache 周期操作。加 event 计时确认 spike 与哪个操作重合。
- **怎么验证**：spike 发生时打印当前在执行的段；或把 `resolve_seq_lens_cpu` 的 copy 改纯 async + event 延迟 sync，看 spike 是否消失、TPOT 是否降到 ~24-25ms。
- **预期收益**：1-2ms（spike 是 TPOT 27 vs steady-state ~25 的主因之一）。
- **工作量**：M。
- **依赖**：H-1（共用计时框架）。

### H-3 ★ recv+getbatch 1.1ms 瘦身
- **改什么**：单请求 decode 时 `recv_requests`（ZMQ poll 空）+ `get_next_batch_to_run`（1 请求）应 ~0；实测 1.1ms 偏高。查是否有冗余遍历 / Python 开销。
- **怎么验证**：单请求 decode 时打点；目标是 <0.3ms。
- **预期收益**：0.5-1ms。
- **工作量**：M。
- **依赖**：无。

### H-4 ★ H-1/H-2/H-3 叠加 + best-of-N 复测
- **改什么**：把 H-1/H-2/H-3 的改动叠加，跑 `compare_tp4.py`（128 tok best-of-3）+ 512 tok decode-only，确认 sglang < vLLM。
- **怎么验证**：`scripts/dl/compare_tp4.py`，看 Gap 行 sglang 是否 < vLLM。
- **预期收益**：如果 H-1/H-2/H-3 各拿 1ms，叠加 → host ~3ms → TPOT ~24ms（持平）；叠加好运气 → 打败。
- **工作量**：S。
- **依赖**：H-1, H-2, H-3。

---

## C 赛道（Compile 侧，让 GPU forward <20ms，多日）

> 目标：torch.compile 给 sglang 加 norm_quant/act_quant 融合 → GPU forward ~18ms → 打败 vLLM。
> 现状：compile 跑通但 decode 82ms（CG 抓的是编译后的 model.forward，inductor 没融合时通用 kernel 比 DLIN eager 慢）。
> decode 即使开 compile 也走 `full_cuda_graph_backend`（不是 tc_piecewise，decode_cuda_graph_runner.py:22 确认）。

### C-1 ★ 修 inductor 子进程 shim（sitecustomize）【S】
- **改什么**：在 sglang venv 的 site-packages 加 `sitecustomize.py`，启动时给 `triton.language.extra.cuda` 注入 `gdc_wait`/`gdc_launch_dependents` 的 `@triton.jit` no-op（当前只在主进程生效，子进程 re-import triton 丢失 → 必须配 `TORCHINDUCTOR_COMPILE_THREADS=1` 才能编译）。
- **怎么验证**：不带 `TORCHINDUCTOR_COMPILE_THREADS=1` 也能 `enable_torch_compile=True` 跑通编译。
- **预期收益**：0（基础设施，解锁后续）。
- **依赖**：阶段一已写好 shim，只需挪到 sitecustomize。

### C-2 ★★ flash_attn 注册 FakeTensor（消 graph-break）【M】
- **改什么**：给 `torch.ops._vllm_fa2_C.varlen_fwd`、`_vllm_fa3_C.fwd`（+ fwd_kvcache）注册 register_fake（输出 = q 形状 + softmax_lse）。参考已在 `dl_compile_meta.py` 给 `_dl_C` ops 注册的写法。
- **怎么验证**：`TORCH_LOGS=graph_breaks` 跑编译，flash_attn 相关 graph-break 消失。
- **预期收益**：减少 eager fallback，编译图更连续。
- **依赖**：C-1。

### C-3 ★ fullgraph 审计（消剩余 graph-break）【M】
- **改什么**：C-2 后再跑 `TORCH_LOGS=graph_breaks`，逐个消剩余 break（可能是某个 Python 控制流 / 另一个无 fake 的 op），直到 fullgraph（无 graph break 警告）。
- **怎么验证**：日志里无 `Graph break`。
- **预期收益**：编译图连续 → decode TPOT 从 82ms 降到 ~27ms（≈ baseline，因为 DLIN op 已 opaque 但还没融合）。
- **依赖**：C-2。

### C-4 ★★★ 移植 vLLM norm_quant 融合 pass【L】
- **改什么**：把 `vllm-new-overlay/vllm/compilation/passes/fusion/rms_quant_fusion.py`（`RMSNormQuantPattern` / `FusedAddRMSNormStaticQuantPattern` 等）移植到 sglang 的 `compilation/passes/` 框架（sglang 有自己的 pass_manager，`python/sglang/srt/compilation/pass_manager.py`），让 inductor 把 `RMSNorm + FP8 per-token quant` 融合成一个 kernel。
- **怎么验证**：编译日志出现融合；GPU forward 从 20.7ms 降到 ~18-19ms。
- **预期收益**：1.5-2.5ms GPU（这是 vLLM 快的核心）。
- **依赖**：C-3。
- **⚠ 结构性前提**：见 C-5。

### C-5 ★★ FP8 路径重构（norm_quant 能否应用的关键）【L】
- **改什么**：确认 sglang 的 FP8 线性路径。当前 `gptq_dlblas_gemmex` 接受 **bf16 输入、内部量化**（fp8_utils.py），而 vLLM 的 `w8a8_matmul` 接受 **预量化 FP8 输入**。`norm_quant` 融合（norm+per-token-quant）只在"有独立 quant 步骤"时才有意义。
  - 如果 sglang 是内部量化 → norm_quant 无法直接应用 → 需要把线性路径改成 `norm → per_token_quant → w8a8_matmul(FP8 输入)`，融合 norm+quant。
  - `.so` 里有 `w8a8_matmul`，可直接用。
- **怎么验证**：改完后输出仍正确（greedy 逐 token 比对）；融合能匹配上新结构。
- **预期收益**：解锁 C-4 的融合收益（没这步，C-4 白做）。
- **依赖**：C-3，且决定 C-4 是否有效。

### C-6 ★★ 移植 act_quant 融合 pass【L】
- **改什么**：类似 C-4，移植 vLLM 的 `fuse_act_quant`（silu/activation + FP8 quant 融合），用在 MoE 的中间激活→量化。
- **怎么验证**：MoE 中间步骤融合；decode TPOT 进一步下降。
- **预期收益**：0.5-1ms。
- **依赖**：C-4, C-5。

### C-7 ★ compile + host 叠加 + 复测打败【S】
- **改什么**：C 赛道（GPU forward ~18ms）+ H 赛道（host ~3ms）叠加，跑 `compare_tp4.py`。
- **怎么验证**：sglang wall_TPOT < vLLM（24ms）。
- **预期收益**：**打败 vLLM**（GPU 18 + host 3 = 21ms < 24ms）。
- **依赖**：C-6 + H-4。

---

## 推荐执行顺序

1. **先做 H 赛道**（H-1 → H-2 → H-3 → H-4）：实际差距在 host，更可能直接拿到 win 或持平。每项独立可验证。
2. **同时/之后做 C 赛道**（C-1 → C-2 → C-3 → C-5 → C-4 → C-6 → C-7）：多日，但能拿到 GPU forward 的真正提速（<20ms），叠加 host 后稳定打败。
3. **关键风险点**：C-5（FP8 路径重构）—— 如果 sglang 内部量化结构让 norm_quant 融合无法应用，C 赛道的收益会大打折扣，此时主攻 H 赛道。

---

## 每项的"完成定义"

每项完成后：
- 跑 `scripts/dl/compare_tp4.py`（128 tok）+ 512 tok decode-only（`/tmp/sg_decode512.py`）记录 TPOT。
- greedy 输出逐 token 比对 baseline（确保没改坏）。
- 在本 checklist 里打勾 + 记录实测 TPOT。
- 目标：sglang wall_TPOT < vLLM（当前 24-26ms）。

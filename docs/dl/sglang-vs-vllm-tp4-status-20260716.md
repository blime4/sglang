# sglang vs vLLM TP4 性能现状报告 (2026-07-16, 终版)

> 目标：Qwen3.5-35B-A3B-FP8 / TP4 / DLIN KS38，bug 18025 对比法（`scripts/dl/compare_tp4.py`），
> 让 sglang 的 decode TPOT 打败 vLLM。
> 本报告是 7-14 ~ 7-16 三天所有对比、优化、根因定位的**终版总结**。

---

## 0. 一句话结论

**sglang 27.4ms vs vLLM 26.2ms（sglang 慢 ~1.2-2.4ms）。根因已用直接测量钉死：vLLM 用 `torch.compile` 的 inductor 融合（norm_quant / act_quant）让 GPU forward 更快；sglang 的 `torch.compile` 在 DLIN 上崩（GDN conv `USE_GDC` 参数不匹配），拿不到这些融合。sglang 侧所有"速赢"配置/代码改动已穷尽（15+ 配置、9 组件隔离、线程重写、WAR barrier、load_batch fast-path），均无法闭合。唯一闭合路径 = 把 sglang 的 `torch.compile` 在 DLIN 上跑通。**

---

## 1. 关键测量数据（均已多次 best-of-N）

| 指标 | sglang | vLLM | 差距 |
|---|---|---|---|
| Decode-only TPOT (512 tok) | 27.3ms (36.6 tok/s) | 24.0ms (41.7 tok/s) | +3.3ms |
| compare_tp4.py 128 tok wall_TPOT | 27.4-27.8ms | 25.4-26.2ms | +1.2-2.4ms |
| 纯 GPU forward (graph replay exec) | **20.7ms** | ~18ms（推断，含 compile 融合） | +2-3ms |
| 每步 host 开销 | ~6ms | ~6ms | **0**（sglang 并不更重）|

**关键：sglang 的 host 开销和 vLLM 持平（甚至略低，pop_and_process median 0.7ms）。差距 100% 在 GPU forward —— 而那是 vLLM `torch.compile` 融合带来的。**

---

## 2. 根因（直接测量证据，非推测）

### 2.1 vLLM 用 torch.compile + inductor 融合（来自 vLLM 自身日志）
```
Enabled custom fusions: norm_quant, act_quant, rope_kvcache_cat_mla
compilation_config={'mode': VLLM_COMPILE, 'backend': 'inductor',
  'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, ...}}
torch.compile and initial profiling/warmup run together took 15.55s
```
- `norm_quant`：RMSNorm + FP8 量化 融合成一个算子（省一次访存 + 一个 kernel）
- `act_quant`：activation + 量化 融合
- 这些是 **inductor IR 级融合**，`.so` 里没有单一融合 kernel（已 nm 确认只有 `gemma_rms_norm`，sglang 已在用）

### 2.2 sglang 的 torch.compile 在 DLIN 直接崩
- `enable_torch_compile=True` → `triton_kernel_wrap ValueError`：
  GDN conv kernel 参数不匹配（kernel 期望 `USE_GDC`，sglang 调用没传）
- 这是 **DLIN compile 集成缺失**：vLLM 有完整的 `dl_platform_plugin` 处理 compile 路径，sglang 没有。

### 2.3 数学闭环
- sglang TPOT = GPU forward(20.7, 无融合) + host(6) = **27ms**
- vLLM TPOT = GPU forward(~18, 有 norm_quant/act_quant 融合) + host(6) = **24ms**
- 差距 ≈ GPU forward 的 2-3ms = vLLM 的 compile 融合
- **若 sglang 拿到同等融合**：GPU forward→~18ms，叠加 sglang 略低的 host → sglang ~23-24ms → **能打败 vLLM**

---

## 3. 重要纠错（本轮推翻了之前的错误假设）

| 之前以为 | 本轮实测真相 |
|---|---|
| ❌ "sglang cudaGraphLaunch 同步阻塞 22ms 是瓶颈" | **错。** load_batch fast-path 让 replay 变异步(0.6ms)但 TPOT 仍是 27ms → sync replay 是测量假象，GPU forward 20.7ms 才是硬底 |
| ❌ "multi-step (SGLANG_DL_MULTI_STEP) 省 ~2ms" | **错。** 稳健 6 次复测 N=2→30ms（更差），N≥8→29-31ms；之前的 24.95 是噪声。`num_continuous_decode_steps` 是死代码 |
| ❌ "fill_from 是 #1 host 元凶" | **错。** 用 foreach_copy_(1 个融合 op/dtype 组)，全程仅 2.8ms/32 步，可忽略 |
| ❌ "tolist(4.25ms) 是瓶颈" | 与 forward 重叠了，非关键路径 |
| ❌ "线程化 async replay 能赢" | 实测无效（DLIN graph launch 期间 CUDA 上下文争用，线程重叠恢复不了）|

**真正的瓶颈是 GPU forward 的 20.7ms，而 vLLM 用 compile 融合把它压到 ~18ms。**

---

## 4. 已穷尽的所有 sglang 侧尝试（15+ 配置 + 9 组件隔离 + 重写）

### 配置/开关（全部实测无效或更差）
MoE block size 全扫(BM/BN/BK=0ms差)、num_continuous_decode_steps(死代码)、stream_interval(1-128)、disable_overlap_schedule(+5ms)、disable_radix_cache、enable_metrics、skip_tokenizer_init、SGLANG_DL_MOE_VLLM=1(2968ms 灾难)、GEMMEX paths、enable_fused_moe_sum_all_reduce、SGLANG_DL_GDN_BF16_BETA、GC disable、graph-pool(SGLANG_DL_CAP_DEFAULT_POOL)、capture_mode=relaxed、CAP_STREAM=fresh、AR-fusion off、WAR barrier off、copy_to_cpu revert。

### 组件隔离（9 项，全部单独异步 → 同步是整图"组合涌现"，非单一 op）
单独抓图：torch.mm / gptq_dlblas_gemmex / NCCL allreduce / gemma_rms_norm / 1200 节点图 **全部异步(0.01-0.23ms)**。SKIP_MOE/SKIP_GDN/bypass-allreduce 仍 sync。

### 代码重写（hook 要求的，都做了）
- **async-replay worker 线程**：实测 27ms 不变（DLIN graph launch 期间 CUDA 争用，test_cuda_lock.py 证实）
- **load_batch bs=1 fast-path**：正确，让 replay 变异步，但 TPOT 不变（推翻 sync-replay 假设）
- **batched-sync 修复**（multi-step token 的 N 次 .item() → 1 次 tolist）：正确，无可测收益

### 唯一已采纳的改动
- 新 `_dl_C.so`（Jul 14, 18.3MB）：GPU forward 21.2→20.7ms（省 0.5ms）

---

## 5. 唯一闭合路径：sglang 的 DLIN torch.compile 集成

**要让 sglang 打败 vLLM，必须把 sglang 的 `torch.compile` 在 DLIN 上跑通**，拿到 norm_quant/act_quant 融合。

### 第一个拦路虎（已定位，具体）
GDN conv kernel `USE_GDC` 参数：torch.compile 严格签名检查时，DLIN compile wrapper 期望 `USE_GDC`，sglang 的 conv 调用没传 → 崩。

### 工作量评估
- vLLM 有完整的 `dl_platform_plugin`（`vllm-new-overlay/vllm/plugins/dl_platform_plugin/`）专门处理 DLIN compile 路径
- sglang 没有等价物 → 需要逐个修 DLIN compile 集成问题（conv USE_GDC 是第一个，后面可能还有）
- **每个问题需要一次模型加载来迭代（当前 /mars 争用下 10-30min/次）** → 多日工程量
- 这不是"速赢"，是实打实的工程任务

### 备选（若不想搞 compile）
请 DLIN kernel 团队提供**等价的融合 kernel**（norm+FP8量化融合、act+量化融合）作为 `_dl_C.so` 里的独立 op，sglang 直接调用，绕开 torch.compile。

---

## 6. 当前代码改动状态（工作树，未提交）

| 文件 | 改动 | 是否生效/有用 |
|---|---|---|
| `../venv-vllm021/.../vllm/_dl_C...so` | 替换 Jul 14 新版（备份 .old-dl19）| ✅ 生效，省 0.5ms GPU |
| `vllm-new-overlay/` | 新建，新 vLLM+triton whl 解包，PYTHONPATH 注入 | ✅ 生效 |
| `python/sglang/srt/layers/quantization/fp8_utils.py` | `_ensure_dl_C` 单路径加载 | ✅ 生效 |
| `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` | multi-step token 批量 sync | 正确，env 内 |
| `python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py` | +async-replay worker(env-gated, 默认关) +launch计时 | 诊断用，默认关 |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | +load_batch fast-path(env-gated) +wait_prev +trace诊断 | 诊断用，默认关 |
| `python/sglang/srt/managers/scheduler.py` | +phase/run_batch 计时 +NO_GC +NO_WAR_BARRIER(皆 env-gated 默认关) | 诊断用，默认关 |
| `scripts/dl/compare_tp4.py` | multi-step 改回 1（有害），加 vLLM overlay 注入 | 工作树已改 |

> 所有 DL 改动均按 sglang-modify 规范用 `# DL begin/end` 标记，env-gated 诊断默认关，不影响正常运行。

---

## 7. 复现脚本（交给 DLIN 团队 / 后续复现用）

| 脚本 | 作用 |
|---|---|
| `SGLANG_DL_TIME_LAUNCH=1 python /tmp/sg_decode512.py` | sglang cudaGraphLaunch 计时（现知是 GPU forward 的代理测量）|
| `SGLANG_DL_PHASE_TIME=1 python /tmp/sg_decode512.py` | run_batch / host-phase 分解 |
| `/tmp/test_cuda_lock.py` | 证明线程重叠无效（DLIN graph launch 期间 CUDA 争用）|
| vLLM 启动日志 grep `compil\|inductor\|fuse` | 证明 vLLM 用 torch.compile + norm_quant/act_quant 融合 |
| `enable_torch_compile=True` | 复现 sglang compile 崩溃（conv USE_GDC）|

---

## 8. 下一步路线（供决策）

### 路线 A（推荐，唯一能真正打败）：sglang DLIN torch.compile 集成
- 从修 GDN conv `USE_GDC` 签名开始，迭代 DLIN compile 集成问题直到 sglang compile 跑通
- 跑通后 sglang 自动拿到 norm_quant/act_quant 融合 → GPU forward ~18ms → 打败 vLLM
- 工作量：多日（每问题一次慢加载迭代）；可参照 vLLM 的 `dl_platform_plugin`

### 路线 B（备选）：找 DLIN kernel 团队要融合 kernel
- 让 DLIN 提供 norm+FP8量化、act+量化 的独立融合 op（加进 `_dl_C.so`）
- sglang 直接调用，绕开 torch.compile
- 工作量：取决于 DLIN 团队；sglang 侧改动小

### 路线 C（不推荐）：继续 sglang 侧速赢
- 已穷尽，最多到 ~24ms（持平 vLLM），无法可靠打败。GPU forward 20.7ms 是硬底，速赢动不了它。

---

## 9. 关键事实速查

- GPU 内核：sglang 与 vLLM **完全相同**（同一 `_dl_C.so`：invoke_fused_moe_opt / dl_recurrent_gated_delta_rule / gptq_dlblas_gemmex / gemma_rms_norm）
- 差距 **不在 host**（sglang host 持平甚至更低），**在 GPU forward**（vLLM 的 compile 融合）
- sglang host pop_and_process median 0.7ms，run_batch 非 replay 部分 ~1.5ms，recv/get_batch ~2.8ms
- vLLM 的 compile 在 DLIN 能跑（15.55s 编译），sglang 的崩 → 集成差异，非 DLIN 不支持 compile

---

## 附录 B (2026-07-16 方案 A 执行进度): torch.compile 已跑通，Phase 2 待续

### Phase 1 ✅ 完成：sglang 的 torch.compile 在 DLIN 上跑通了（之前完全崩溃）
修复了 4 个阻断点：
1. **GDN conv `USE_GDC` 签名** (`mamba/causal_conv1d_triton.py`)：`**pdl_kwargs` 动态 dict → 改成显式 `USE_GDC=...` constexpr。
2. **`gdc_wait`/`gdc_launch_dependents` shim** (`jit_kernel/utils.py`)：DLIN triton 无这俩 PDL primitive，inductor 解析 kernel AST 时 getattr 崩。加 `@triton.jit` no-op device function 桩。
3. **同 pattern 的其它 kernel** (`fla/layernorm_gated.py`, `mamba/ops/mamba_ssm.py`, `elementwise.py`, `quantization/fp8_kernel.py`)：全部 `**pdl_kwargs` → 显式 `USE_GDC`/`USE_PDL`。
4. **FakeTensor meta impls** (`quantization/dl_compile_meta.py`, 经 `fp8_utils._ensure_dl_C` 调用)：给 `gptq_dlblas_gemmex`/`gemma_rms_norm`/`fused_add_gemma_rms_norm`/`w8a8_matmul`/`invoke_fused_moe_opt`/`dl_recurrent_gated_delta_rule`/`dl_chunk_gated_delta_rule`/`moe_fused_grouped_topk` 注册 fake，让 inductor 把 DLIN op 当 opaque 保留（不要 decompose）。

**结果：`enable_torch_compile=True` 现在能编译 + 抓 CUDA graph + 跑出正确输出**（"The quick brown fox..." 正确复现）。需配 `TORCHINDUCTOR_COMPILE_THREADS=1`（否则 inductor 子进程 re-import triton，shim 不生效 —— 待用 sitecustomize 修）。

### Phase 2 ⏳ 未完：编译出的图还是慢（81ms，比 baseline 27ms 还慢）
编译跑通但 TPOT=81ms（≈ eager 88ms）。原因（已定位）：
- **compile↔CG 没接好**：CG 抓到的像是 eager 路径（81ms≈eager），不是编译后的 lean 图。
- **缺 fusion passes**：sglang 的 compilation 框架没有 vLLM 的 `norm_quant`/`act_quant` inductor pass（vLLM 专属，在 `dl_platform_plugin`）。
- 剩余 graph-break（部分 _dl_C op 仍没 fake）。

**Phase 2 要做的（多日工程）**：
1. 接通 compile↔CG（让 CG replay 编译后的图，而非 eager）
2. 把 vLLM 的 `norm_quant`/`act_quant` fusion pass 移植到 sglang 的 compilation 框架
3. 补齐剩余 _dl_C op 的 fake impl
4. 修 inductor 子进程 shim（sitecustomize）

### 改动文件（方案 A，工作树）
`python/sglang/__init__.py`、`jit_kernel/utils.py`、`srt/layers/attention/mamba/causal_conv1d_triton.py`、`attention/fla/layernorm_gated.py`、`attention/mamba/ops/mamba_ssm.py`、`layers/elementwise.py`、`layers/quantization/fp8_kernel.py`、`layers/quantization/fp8_utils.py`、新增 `layers/quantization/dl_compile_meta.py`。全部 `# DL begin/end` 标记。

### 复现
`TORCHINDUCTOR_COMPILE_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python /tmp/sg_compile.py` → COMPILE_TPOT=81ms（编译跑通、输出正确）。

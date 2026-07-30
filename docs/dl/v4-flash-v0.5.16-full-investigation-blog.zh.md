# DeepSeek-V4-Flash on DLIN sglang v0.5.16 — 完整调试与性能优化记录

**日期：** 2026-07-29 ~ 2026-07-30
**分支：** `dl-dev-v0.5.16`（已合并自 `dl-dev-v0.5.16-support-dsv4`）
**模型：** DeepSeek-V4-Flash（149GB，43 层 MHC+MLA 混合，256 experts top-6，FP4 MoE）
**硬件：** DLIN KS38 × 8（32GB/卡），TP8
**成果：** eager 1.60→2.36 tok/s；full CG 2.03→**6.44 tok/s（4×）**，输出正确

---

## 目录

1. [为什么要移植（support-dsv4 漂移）](#part-0)
2. [正确性移植（8 处修复）](#part-1)
3. [indexer DL op 优化（4× 提速的关键）](#part-2)
4. [CUDA Graph 分析](#part-3)
5. [mabs bug 修复](#part-4)
6. [精确 profile（deferred-sync 消除 sync 伪影）](#part-5)
7. [JIT FP8 GEMV kernel 开发](#part-6)
8. [MoE 瓶颈分析（所有路径测试）](#part-7)
9. [Speculative decoding 测试](#part-8)
10. [被阻塞的路径总结](#part-9)
11. [下一步优化方向](#part-10)

---

## <a id="part-0"></a>Part 0：为什么要移植（support-dsv4 漂移）

`support-dsv4`（07-27 暂停）在当前 `.venv` 上**整体漂移**，无法运行：

- 07-29 的 v0.5.16 "Phase 4e" 重构把算子 `torch.ops._dl_C.*` → `torch.ops.sgl_kernel.*`，rename `gemma_rms_norm`→`gemma_rmsnorm`。
- support-dsv4 的 7 个 `_dl_C.*` V4 算子引用全部 **MISSING**（静态+运行时确认）。
- eager smoke 掉到 triton 路径报 "Hidden size mismatch"。
- support-dsv4 的 DL MoE 块是死代码（runner 选了 triton）。
- 主线 `dl-dev-v0.5.16` 上**没有 DLIN-V4 适配**（只有上游 SM120/NVIDIA 的 V4）。

**结论：** 从 `dl-dev-v0.5.16`（bcf8795aa3）创建 `dl-dev-v0.5.16-support-dsv4`，把 support-dsv4 的 DLIN-V4 工作 port 过来。

---

## <a id="part-1"></a>Part 1：正确性移植（8 处修复，eager 输出 " Paris"）

fix-forward（跑→报错→修），将 support-dsv4 的 DL 逻辑落到 v0.5.16 的结构上：

| # | 文件 | 修复 | 关键细节 |
|---|---|---|---|
| 1 | `dsv4/indexer.py` | 失效 import `fp8_kernel`→`kernels.ops.quantization`（**首个阻塞**，导致 AutoModel 注册失败） | v0.5.16 把 fp8_kernel.py 从 `srt/layers/quantization/` 迁到 `kernels/ops/quantization/` |
| 2 | `deepseek_v4_backend.py` | `_is_dlin` + `_create_flashmla_metadata()` 返 None + MLA decode→`sgl_kernel.flash_mla_with_kvcache(causal=False)` + prefill→`sgl_kernel.flash_mla_sparse_prefill_fwd` | causal=False 是 **V4 正确性 bug #1**（sparse 模式下 causal=True 读 None 描述符→所有 MLA 层位置错误→乱码）。vLLM 和我们的 sm120/NVIDIA 路径都省略 causal（→False）。`_create_flashmla_metadata` 返 None 因为 `sgl_kernel.flash_mla`（Python wrapper）加载 NVIDIA-only flashmla_ops 扩展→ImportError。 |
| 3 | `kernels/ops/layernorm/mhc.py` | torch Sinkhorn port（tilelang 缺失） | **注意**：mhc.py 在 v0.5.16 从 `srt/layers/mhc.py` 迁到 `kernels/ops/layernorm/mhc.py`。v0.5.16 已有 `hc_pre_torch_impl` + `hc_post_torch_impl`（hc_post no-op fix 已在上游），只需补 Sinkhorn。 |
| 4 | `deep_gemm_wrapper/entrypoint.py` | `tf32_hc_prenorm_gemm` torch fallback | `not ENABLE_JIT_DEEPGEMM`（DLIN 无 deep_gemm）|
| 5 | `layernorm.py` | DL rmsnorm `else: x.contiguous()` | `sgl_kernel.rmsnorm` 要求 contiguous 输入，2D 非 contiguous 会报错 |
| 6 | `silu_and_mul_masked_post_quant.cuh` | 结构化绑定 lambda-capture 修复 | dlcc 拒绝 C++17 结构化绑定被 lambda 捕获（`auto [expert_id,...]=get_work()`→`auto [_expert_id,...]; auto expert_id=_expert_id;`）|
| 7 | `fp8.py` | DL MoE 默认 `_G()` 把 FP4 当 FP8（**bug #3，乱码根因**） | V4 MoE 是 mxfp4/FP4（w13 packed int8）。DL fused MoE 块的默认调用硬编码 FP8 tuple `(True,False,False,False)` = use_fp8_w8a8=True→FP4 字节当 FP8→9000× 残差爆炸→乱码。改为按 `w13.dtype==torch.int8` 选 mxfp4 tuple `(False,False,False,True)` |
| 8 | （主线已有）| DL MoE 块 `sgl_kernel.invoke_fused_moe_opt`（从 _dl_C 迁过来）| 只需修 FP4 flag |

**正确性门：** `"The capital of France is"` → **`" Paris"`** ✅

### 环境（V4 eager/CG，TP8 cards 0-7）

```bash
source sdk-dlop-07-13-20-30/env.sh  # ⚠️ 设好 LD_LIBRARY_PATH；别手动 export LD_LIBRARY_PATH=$SDK/lib（$SDK 未设→/lib/libcurt.so 是目录→torch import OSError）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8
export DLI_V2=ON TORCHDYNAMO_DISABLE=1 DLEOL_CACHE_SIZE=1024
export SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=2048 SGLANG_DL_FP8_Q2=1 SGLANG_DL_GDN_DLIN=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_HASH_TOPK=0 SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_TOPK_TRANSFORM_512_TORCH=1 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

---

## <a id="part-2"></a>Part 2：indexer DL op 优化（4× 提速的关键）

### 定位 indexer 是瓶颈

用**源码内 CUDA-event 计时**（spawn-safe——TP worker 是 spawn，父进程 monkeypatch 拿不到；直接在 `deepseek_v4.py`/`indexer.py` 源码里加 CUDA event）：

```
[ATTN_T] step attn=440ms   ← 投影(dlblas,快) + flash_mla(~18ms) + indexer
[MOE_T]  step MoE=85ms
```

ATTN=440ms（71%），投影快（dlblas_w8a8_block_fp8_linear = gptq_dlblas_gemmex），flash_mla ~18ms → **indexer 独占 ~400ms**。

### 为什么 indexer 慢

V4 indexer（c4/c128 MLA 路由 logits）在 `dsv4/indexer.py:forward_c4_indexer` 中：
- `use_fp4_indexer` 路径要 `from deep_gemm import fp8_fp4_paged_mqa_logits`——**deep_gemm 在 DLIN 缺失**。
- 所以 V4 落到非-FP4 的 `elif SGLANG_FP8_PAGED_MQA_LOGITS_TORCH:` → **纯 torch fallback `fp8_paged_mqa_logits_torch`**（每层一堆小 torch op × 43 层）= ~400ms。

### 参考 vLLM 修 indexer（DL op）

vLLM **monkeypatch `deep_gemm.fp8_fp4_paged_mqa_logits` → `torch.ops._dl_C.fp8_fp4_paged_mqa_logits`**（DL op）。FP8 路径 `q_scale=None`（V4 KV cache 是 fp8_e4m3）。

#### 3 个修复

**1. `dsv4/indexer.py`——DLIN 分支调 DL op**（在 torch fallback 之前）：
```python
elif is_dlin():
    def fn(q, kv_cache, weights, context_lens, block_tables, sched_meta, max_model_len, clean):
        _q = q if isinstance(q, torch.Tensor) else q[0]
        _qs = None if isinstance(q, torch.Tensor) else q[1]
        _sm = sched_meta if sched_meta is not None else torch.empty(0, ...)
        return torch.ops.sgl_kernel.fp8_fp4_paged_mqa_logits(
            _q.contiguous(), _qs, kv_cache, weights.float(), ...)
```

**2. `dsv4/metadata.py`——DLIN 上构造 `deep_gemm_metadata`**。第一次报 `cuDNN BAD_PARAM`——因为 `__post_init__` 在 torch env 下 `deep_gemm_metadata=None`，DL op 拿到空 metadata。修复：DLIN 上用 `sglang.jit_kernel.dsv4.get_paged_mqa_logits_metadata(c4, c4_page_size, num_sms)`。

**3. `paged_mqa_metadata.cuh`——smem 守卫**。第二次报 `paged_mqa_metadata.cuh:113: CUDA error: invalid argument`——`setup_kernel_smem_once` 里的 `cudaFuncSetAttribute(128KB smem)` 超过 DLIN 设备上限。修复：
```cpp
#ifdef SGL_ON_DLIN
    return ::cudaSuccess;  // metadata kernel 解码 batch=1 只用 ~8B smem
#else
    ... cudaFuncSetAttribute(128KB) ...
#endif
```
改完记得清 tvm-ffi 的 `*_metadata_*` JIT 缓存。

### 结果

| 配置 | TPOT | tok/s |
|---|---|---|
| 原始 eager（torch indexer）| 623ms | 1.60 |
| CG + torch indexer | 492ms | 2.03 |
| eager + **DL indexer** | 424ms | 2.36 |
| **CG + DL indexer** | **155ms** | **6.44** 🚀 |

**4× 提速**。那个慢 torch indexer（400ms、大量小 op）的 host launch 开销**同时也把 full CG 的 replay 压死了**——换 DL op 后 eager 和 CG 一起解锁。

---

## <a id="part-3"></a>Part 3：CUDA Graph 分析

- v0.5.16 的 `cuda_graph_backend_decode` 默认 = **FULL**（`cuda_graph_config.py:112`）。
- support-dsv4 的"整图捕获"在 32GB OOM。但 v0.5.16 的 breakable CG + "heavy capture-pool memory pressure; disabling prefill CG" 保护 → **decode full CG 在 mem=0.90 能跑**。
- full CG 在 mem=0.90 偶发 deadlock（capture pool 压力大，1 张卡 hang）——用 `~/.claude/skills/dl-gpu-hang-recover/dl_gpu_hang_recover.sh` 恢复。
- **CG + DL indexer = 155ms（稳定）**——DL indexer 的少量高效 op 让 CG graph 更精简。

---

## <a id="part-4"></a>Part 4：mabs bug 修复

### 发现

追 v3/GEMMEX MoE 路径时发现**所有替代路径都被 mabs bug 静默阻塞**：

- `Fp8MoEMethod.apply` 的 try/except（fp8.py:2602）捕获了 `DL_MOE_ERR` → 落到 triton → V4 FP4 "Hidden size mismatch"。
- DL_MOE_ERR 揭示根因：**`sgl_kernel::moe_align_block_size()` expected 8 args, received 9**。
- `kernels/ops/moe/__init__.py:74-84` 在 `ignore_invalid_expert=True` 时传 9 个参数（含 `ignore_invalid_expert`），但 v0.5.16 的 op 只接受 8 个——**签名漂移 bug**。
- 默认 use_moe_cu 路径避免 mabs（trivial dispatch），所以 6.44 tok/s 正常；所有更快路径被屏蔽。

### 修复（commit 4a85eed50d）

- `kernels/ops/moe/__init__.py`：移除 9-arg 分支的 `ignore_invalid_expert`。
- `fp8.py`：强制 `_need_mabs=False`——GEMMEX 路径用 topk_ids 直接调用，use_moe_cu 用 trivial dispatch，mabs 不需要。

### 效果

GEMMEX=4 FP4 路径**现在能到达 op**（不再落 triton），但遇到新的独立问题（FP4 dtype `UNKNOWN_SCALAR`——V4 的 int8 FP4 格式与 GEMMEX bit=4 期望的 vLLM Hadamard-mxfp4 不同）。

---

## <a id="part-5"></a>Part 5：精确 profile（deferred-sync 消除 sync 伪影）

### ⚠️ 关键纠正：indexer 不是 111ms——是 22ms

早期用**每调用 `torch.cuda.synchronize()`** 的计时测得 indexer = 5.3ms/call × 21 = 111ms。但 sync 本身串行化了 GPU 流水线，**人为膨胀了 5×**。

改用 **deferred-sync**（每调用 record event pair，每 21 calls sync 一次），测得 indexer 真实时间：

```
[IDX_T] calls=21 total=22.0ms per-call=1.05ms  ← 不是 5.3ms！
```

**indexer = 22ms（5%），不是 111ms。** 之前 5.3ms/call 是 sync 伪影。

### 准确 decode 拆解（eager，deferred-sync）

| 成分 | 时间 | 占比 | 性质 |
|---|---|---|---|
| **MoE（invoke_fused_moe_opt）** | 87ms | 56% | FP4 唯一正确路径 |
| **ATTN（投影+flash_mla+indexer）** | ~46ms | 30% | 投影 dlblas（快），indexer 22ms，flash_mla 18ms |
| **其余（norm/rope/MHC/all-reduce）** | ~22ms | 14% | |

**MoE（87ms）是解码瓶颈。**

---

## <a id="part-6"></a>Part 6：JIT FP8 GEMV kernel 开发

### 动机

投影用 dlblas GEMM（gptq_dlblas_gemmex），在 M=1 时 GEMM tiling 浪费 85%+（M-tile=16 只用 1 行）。目标是 memory-bound GEMV（~2µs vs ~0.6ms）。

### 实现

利用 sglang 的 JIT kernel 基础设施（`add-jit-kernel` skill），写了 `jit_kernel/csrc/elementwise/fp8_gemv.cuh`：
- 每个 warp 处理 1 个输出元素 N。
- 向量化 load（`AlignedVector`）+ warp reduce dot product。
- per-channel scale 应用。

### 测试结果

```
Correctness: max_rel_err=0.0000 mean_rel_err=0.0000  ← 完全正确
JIT fp8_gemv: 0.170ms/call (N=512 K=4096)
gptq_dlblas_gemmex: 0.391ms/call
Speedup: 2.3×
```

### 集成 + CG 效果

集成到 `dlblas_w8a8_block_fp8_linear`（M=1 时路由到 JIT GEMV）。但 **CG TPOT 不变（156ms）**——投影在 CG 里只占 ~28ms（小 fraction），MoE 87ms 主导。

**commit a15102673d。**

---

## <a id="part-7"></a>Part 7：MoE 瓶颈分析（所有路径测试）

### MoE = 87ms = 56% of CG TPOT。唯一正确路径 = `invoke_fused_moe_opt(use_mxfp4_w4a16)`。

| 路径 | 测试 | 结果 |
|---|---|---|
| v3 direct op (`invoke_fused_moe_opt_v3`) | weight_bits=4, M≥1 | ❌ `cannot infer quant_type from (BFloat16, Char, 4)` — v3 是 FP8(w8a8) 导向 |
| v3 via vLLM `fused_experts` | `mxfp4_w4a16_moe_quant_config` | ❌ `_vllm_fa2_C` SIGABRT（.so 双注册 vs sglang's `_sgl_fa2_C`）|
| GEMMEX=4 FP4 | `gptq_dlblas_gemmex(bit=4, quant_type=0)` | ❌ V4 FP4 格式不匹配（期望 vLLM Hadamard-mxfp4，V4 用 plain int8 FP4）→ `UNKNOWN_SCALAR` |
| JIT FP4 per-expert GEMV | 自己写 kernel | ❌ per-expert 151ms > fused 87ms（12 次 kernel invocation 开销 > 1 次 fused）|

### 关键发现

- **v3 理论上支持 FP4**（vLLM `dl_fused_moe.py:417`, `mxfp4_w4a16`）——但 v3 对 V4 的 w4a16（Char 权重 + bf16 激活 + weight_bits=4）无法推断 quant_type。
- **per-expert GEMV 不能打败 fused**——fused op 在一次 kernel 中处理所有 6 个 expert，per-expert 需要 12 次调用（即使 CG 去掉了 launch 开销，每次 kernel invocation 仍有 ~0.1ms setup 开销）。
- 要优化 MoE 需要 **fused FP4 MoE kernel 替换**（一个 memory-bound grouped FP4 GEMV kernel，所有 6 个 expert + silu + accumulate 在一次 kernel 调用中完成）——substantial dlcc kernel 开发。

---

## <a id="part-8"></a>Part 8：Speculative decoding 测试

| 方法 | 结果 |
|---|---|
| MTP/NEXTN | V4 不支持（`"Only EAGLE and DSPARK"`）|
| EAGLE3 | V4 不支持 |
| EAGLE topk=1（CG, mem=0.90）| OOM |
| EAGLE topk=1（CG, mem=0.78, ctx=512）| OOM |
| EAGLE 4-draft（eager, mem=0.90）| OOM |
| DSPARK | 需要外部 draft model（V4 MTP 是内置的）|

**EAGLE 在 32GB×8 上 OOM 的原因：** V4（149GB）+ EAGLE spec batch（draft+verify 的 KV + activations + draft model）+ CG workspace 在 32GB 上放不下。降低 mem_fraction → KV pool OOM。Catch-22。

---

## <a id="part-9"></a>Part 9：被阻塞的路径总结

| 路径 | 阻塞原因 | 解决方案 |
|---|---|---|
| 更高 TP（TP16）| `n_groups=8 // tp_size=16 = 0` | 架构限制（max TP=8）|
| EP（Expert Parallel）| M=1 decode 总计算量不变 + 增加 all-to-all | EP 对 M=1 无效 |
| 投影 Q1/Q2/w8a8 | 3 种 kernel variant CG TPOT 全部 ~155ms | dlblas kernel floor |
| 投影 JIT FP8 GEMV | 正确+2.3×快/call，但投影在 CG 只占 ~28ms | 已提交（a15102673d）|
| MoE v3 | quant_type inference 对 FP4 失败 | 需 v3 op 扩展支持 w4a16 |
| MoE GEMMEX=4 | V4 FP4 格式 ≠ vLLM Hadamard-mxfp4 | 需要 V4-specific FP4 格式适配 |
| MoE per-expert GEMV | 151ms > fused 87ms | 需要 fused FP4 kernel |
| Speculative EAGLE | OOM（32GB）| 需要 48GB+ 卡 |
| Speculative MTP/DSPARK | V4 不支持/需外部 draft | — |

---

## <a id="part-10"></a>Part 10：下一步优化方向

### 到 20 tok/s 的路径

**目标：** CG TPOT 155ms → 50ms（3× 提速）。

| 方向 | 预期效果 | 难度 |
|---|---|---|
| **Fused FP4 MoE kernel 替换** | MoE 87ms → ~3ms（batched GEMV，所有 expert 一次调用）→ CG ~71ms → ~14 tok/s | 高（200+ 行 dlcc CUDA，需正确 FP4 dequant + expert routing + silu + accumulate）|
| **48GB×8 硬件** | CG+EAGLE 能跑（spec ~2×）→ effective ~13 tok/s | 无代码（硬件升级）|
| **投影 dlblas GEMV 优化** | 投影 ~28ms → ~2ms → CG ~129ms → ~7.8 tok/s | 中（改进 JIT GEMV kernel，增加 warps/block + persistent kernel）|

**最有可能的路径：Fused FP4 MoE kernel**——这是唯一能把 MoE 从 87ms 降到个位数 ms 的方法。需要：
1. 修复 FP4 dequant 正确性（E2M1 nibble + e8m0 block scale）。
2. Batched kernel（所有 6 个 expert 在一个 grid 中并行处理）。
3. 集成 silu_and_mul + weighted accumulation。
4. 对 V4 实际 mxfp4 权重做正确性验证。

### 已提交的 commits

```
1bc613eb50 fix(dl): balance DL markers for JIT GEMV path in fp8_utils.py
a15102673d feat(dl): JIT FP8 GEMV kernel for M=1 decode projections
4a85eed50d fix(dl): bypass broken moe_align_block_size (9-vs-8 sig drift) + GEMMEX=4 FP4 MoE path
12b1a7f06e docs(dl): V4-Flash v0.5.16 indexer DL-op perf blog (1.6->6.44 tok/s, 4x)
be295fecef feat(dl): DeepSeek-V4-Flash on DLIN v0.5.16 — port + indexer DL op (1.6→6.44 tok/s)
```

### 教训

1. **deferred-sync vs per-call sync：** 每调用 sync 会串行化 GPU 流水线，人为膨胀 5×。用 deferred-sync（每 N 次 sync 一次）测真实 GPU 时间。
2. **fused > per-expert：** MoE 的 fused grouped GEMM 在一次 kernel 中处理所有 expert，per-expert GEMV 的 12 次调用 setup 开销无法打败它。
3. **DLIN 的 GEMM kernel 在 M=1 时有巨大 overhead（85-300× over memory floor）**——这不是 Python 能解决的，是 dlcc/dlablas kernel 优化问题。
4. **moe_align_block_size 9-vs-8 签名漂移**——v0.5.16 的 op 注册变了但 wrapper 没跟上。try/except 静默掩盖了 bug。遇到 "DL block falls to triton" 先检查 DL_MOE_ERR。
5. **source-level instrumentation is spawn-safe**——TP worker 是 spawn，父进程 monkeypatch 拿不到。直接在源码里加计时是唯一可靠的 per-component GPU 时间测量方法。

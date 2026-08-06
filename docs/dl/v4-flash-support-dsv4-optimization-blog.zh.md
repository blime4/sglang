# DeepSeek-V4-Flash on DLIN sglang — 从 0 到 15.83 tok/s 完整优化技术报告

**日期：** 2026-07-29 ~ 2026-08-03
**分支：** `dl-dev-v0.5.16`（基线）→ `dl-dev-v0.5.16-support-dsv4`（本报告对象）
**主要提交：** `be295fecef`～`4a85eed50`（阶段 A 移植）+ `a15102673`～`2ea25190a`（阶段 B 优化）+ 工作区未提交改动
**模型：** DeepSeek-V4-Flash（149GB，43 层 MHC+MLA 混合，256 experts top-6，FP4 MoE）
**硬件：** DLIN KS38 × 8（32GB/卡），TP8
**成果：** 无法运行 → 正确输出（" Paris"）→ eager 1.60 tok/s → CG **6.44 tok/s** → CG+EAGLE **15.83 tok/s（相对能跑基线 +145%）**

> 本文把 V4-Flash 在登临 sglang 上"从零到能跑对、再到跑得快"的全过程一次讲透。每个优化点都附**简明代码栈**（`文件:行号`）和**大白话解释**。
>
> **两个阶段：**
> - **阶段 A（0 → 6.44）：移植 + 正确性**——support-dsv4 分支整体漂移跑不起来，从 v0.5.16 重建，做 8 处正确性修复跑出正确输出，再用 indexer DL op 把 CG 提速 4×。
> - **阶段 B（6.45 → 15.83）：性能深挖**——移植 MHC Triton kernel、自写 indexer valid-only kernel、落地 EAGLE 投机解码，一路推到 15.83 tok/s。

---

## 目录

**阶段 A：从无到有（0 → 6.44 tok/s）**
1. [起点：support-dsv4 漂移，整体无法运行](#a-0)
2. [正确性移植：8 处修复（0 → " Paris"，1.60 tok/s）](#a-1)
3. [indexer DL op：4× CG 提速的关键（1.60 → 6.44）](#a-2)
4. [CUDA Graph 分析（让 6.44 稳定跑下来）](#a-3)
5. [解锁被静默阻塞的路径：mabs bug + JIT FP8 GEMV + MoE 摸底](#a-4)

**阶段 B：性能深挖（6.45 → 15.83 tok/s）**
6. [性能提升栈总览（6.45 → 15.83 一张图）](#b-0)
7. [优化一：MHC Triton kernel 移植（+32%）](#b-1)
8. [优化二：Indexer valid-only Triton kernel（+18%）](#b-2)
9. [优化三：EAGLE 投机解码落地（+21%）](#b-3)
10. [优化四：CUDA Graph 兼容性三连修](#b-4)
11. [优化五：deferred-sync 分组件 Profiling 基础设施](#b-5)
12. [优化六：模型层 DLIN 适配](#b-6)
13. [优化七：FP4 grouped-GEMV 探索性 kernel](#b-7)
14. [优化八：诊断与基准脚本矩阵](#b-8)

**附录**
15. [环境变量速查表](#env)
16. [踩坑总结与下一步](#summary)
17. [改动清单与提交历史](#changelog)

---

# 阶段 A：从无到有（0 → 6.44 tok/s）

## <a id="a-0"></a>1. 起点：support-dsv4 漂移，整体无法运行

老的 `support-dsv4` 分支（07-27 暂停）在当前 `.venv` 上**整体漂移**，一个 token 都跑不出来。原因：

- **算子大改名**：07-29 的 v0.5.16 "Phase 4e" 重构把算子命名空间 `torch.ops._dl_C.*` → `torch.ops.sgl_kernel.*`，还把 `gemma_rms_norm` rename 成 `gemma_rmsnorm`。
- **7 个算子引用全断**：support-dsv4 里 7 个 `_dl_C.*` 的 V4 算子引用全部 MISSING（静态 + 运行时双重确认）。
- **eager smoke 直接崩**：跑不起来 → 掉到 triton 路径 → 报 "Hidden size mismatch"。
- **DL MoE 是死代码**：runner 选了 triton，support-dsv4 写的 DL MoE 块根本没被调用。
- **主线没有登临 V4**：`dl-dev-v0.5.16` 主线上**没有任何 DLIN-V4 适配**，只有上游 SM120/NVIDIA 的 V4 实现。

**结论：** 不能在漂移的老分支上修，得**从干净的 `dl-dev-v0.5.16`（bcf8795aa3）创建新的 `dl-dev-v0.5.16-support-dsv4`**，把 support-dsv4 的 DLIN-V4 工作 port 过来，落到 v0.5.16 的新结构上。

> **大白话：** 老分支的代码调的算子名字已经被新版本全改了，等于"调用的 API 全不存在"，根本跑不起来。与其在烂摊子上修，不如在新地基上重建，把登临特有的逻辑一点点搬过来。

---

## <a id="a-1"></a>2. 正确性移植：8 处修复（0 → " Paris"，eager 1.60 tok/s）

用 **fix-forward**（跑 → 报错 → 修）的方式，把 support-dsv4 的 DL 逻辑落到 v0.5.16 结构上。共 8 处修复，按出现顺序：

| # | 文件 | 修复 | 关键细节 / 为什么炸 |
|---|---|---|---|
| 1 | `dsv4/indexer.py` | 失效 import：`fp8_kernel` → `kernels.ops.quantization` | **首个阻塞**，导致 AutoModel 注册失败。v0.5.16 把 `fp8_kernel.py` 从 `srt/layers/quantization/` 迁到了 `kernels/ops/quantization/`。 |
| 2 | `deepseek_v4_backend.py` | `_is_dlin` + `_create_flashmla_metadata()` 返 None + MLA decode→`sgl_kernel.flash_mla_with_kvcache(causal=False)` + prefill→`flash_mla_sparse_prefill_fwd` | **正确性 bug #1**：sparse 模式下 `causal=True` 会读 None 描述符 → 所有 MLA 层位置全错 → 乱码。vLLM 和 sm120/NVIDIA 路径都省略 causal（→False）。`_create_flashmla_metadata` 返 None 是因为 `sgl_kernel.flash_mla`（Python wrapper）会加载 NVIDIA-only 的 flashmla_ops 扩展 → ImportError。 |
| 3 | `kernels/ops/layernorm/mhc.py` | torch Sinkhorn port（tilelang 缺失） | mhc.py 在 v0.5.16 从 `srt/layers/mhc.py` 迁到 `kernels/ops/layernorm/mhc.py`。v0.5.16 已有 `hc_pre_torch_impl` + `hc_post_torch_impl`，只需补 Sinkhorn。 |
| 4 | `deep_gemm_wrapper/entrypoint.py` | `tf32_hc_prenorm_gemm` torch fallback | `not ENABLE_JIT_DEEPGEMM`（登临没有 deep_gemm）。 |
| 5 | `layernorm.py` | DL rmsnorm 加 `else: x.contiguous()` | `sgl_kernel.rmsnorm` 要求 contiguous 输入，2D 非 contiguous 直接报错。 |
| 6 | `silu_and_mul_masked_post_quant.cuh` | 结构化绑定 lambda-capture 修复 | dlcc 拒绝 C++17 结构化绑定被 lambda 捕获：`auto [expert_id,...]=get_work()` → 拆成 `auto [_expert_id,...]; auto expert_id=_expert_id;`。 |
| 7 | `fp8.py` | DL MoE 默认 `_G()` 把 FP4 当 FP8（**乱码根因**） | **正确性 bug #3**：V4 MoE 是 mxfp4/FP4（w13 packed int8）。DL fused MoE 块默认调用硬编码 FP8 tuple `(True,False,False,False)` = `use_fp8_w8a8=True` → 把 FP4 字节当 FP8 解 → 残差爆炸 ~9000× → 乱码。改成按 `w13.dtype==torch.int8` 选 mxfp4 tuple `(False,False,False,True)`。 |
| 8 | （主线已有）| DL MoE 块 `sgl_kernel.invoke_fused_moe_opt`（从 `_dl_C` 迁过来）| 只需修上面的 FP4 flag。 |

**正确性门：** `"The capital of France is"` → **`" Paris"`** ✅ —— 此时 eager + torch indexer 跑通，**1.60 tok/s**。

代码栈示例（fix #7，乱码根因，`fp8.py`）：

```python
# V4 MoE 权重是 FP4(int8 packed)；按 dtype 选 quant flag，不能硬编码 FP8
_use_mxfp4 = (layer.w13_weight.dtype == torch.int8)
_qf = (False, False, False, True) if _use_mxfp4 else (True, False, False, False)
#            ^^^^^^^^^^^^^^^^^^^ mxfp4 tuple        ^^^^^^^^^^^^^^^^^^^ FP8 tuple
```

> **大白话：** 把模型从"跑不起来"修到"输出正确"。8 个坑里最阴的两个：① MLA attention 的 `causal` 参数在 sparse 模式必须关掉，否则位置全乱（输出乱码）；② MoE 权重其实是 4-bit 的，但代码默认按 8-bit 解，数值直接炸 9000 倍（也是乱码）���修完这 8 个，终于吐出正确的 " Paris"，但只有 1.6 tok/s——因为有个 indexer 在疯狂烧时间。

---

## <a id="a-2"></a>3. indexer DL op 优化：4× CG 提速的关键（1.60 → 6.44）

### 3.1 定位 indexer 是瓶颈（源码内 CUDA-event 计时）

TP worker 是 **spawn** 出来的，父进程 monkeypatch 拿不到。唯一可靠的办法是**直接在源码里加 CUDA event**（`deepseek_v4.py` / `indexer.py`）。测得：

```
[ATTN_T] step attn=440ms   ← 投影(dlblas,快) + flash_mla(~18ms) + indexer
[MOE_T]  step MoE=85ms
```

ATTN 440ms（占 71%），投影是快的（`dlblas_w8a8_block_fp8_linear`），flash_mla ~18ms → **indexer 独占 ~400ms**。

### 3.2 为什么 indexer 慢

V4 indexer（c4/c128 MLA 路由 logits）在 `dsv4/indexer.py:forward_c4_indexer`：
- `use_fp4_indexer` 路径要 `from deep_gemm import fp8_fp4_paged_mqa_logits`——**deep_gemm 在登临上不存在**。
- 于是 fallback 到 `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` → **纯 torch 实现 `fp8_paged_mqa_logits_torch`**（每层一堆小 torch op × 43 层）≈ 400ms。

### 3.3 参考 vLLM 修：换 DL op

vLLM 的登临插件做的是 **monkeypatch `deep_gemm.fp8_fp4_paged_mqa_logits` → `torch.ops._dl_C.fp8_fp4_paged_mqa_logits`**（现 `sgl_kernel.fp8_fp4_paged_mqa_logits`）。FP8 路径 `q_scale=None`（V4 KV cache 是 fp8_e4m3）。

**3 个修复（按踩坑顺序）：**

**修复 1 — `indexer.py` 加 DLIN 分支，在 torch fallback 之前调 DL op：**
```python
elif is_dlin():
    def fn(q, kv_cache, weights, context_lens, block_tables, sched_meta, max_model_len, clean):
        _q = q if isinstance(q, torch.Tensor) else q[0]
        _qs = None if isinstance(q, torch.Tensor) else q[1]
        _sm = sched_meta if sched_meta is not None else torch.empty(0, ...)
        return torch.ops.sgl_kernel.fp8_fp4_paged_mqa_logits(
            _q.contiguous(), _qs, kv_cache, weights.float(), ...)
```

**修复 2 — `metadata.py` 在 DLIN 上构造 `deep_gemm_metadata`。** 第一次跑报 `cuDNN BAD_PARAM`——因为 `__post_init__` 在 torch env 下把 `deep_gemm_metadata=None`，DL op 拿到空 metadata。修复：DLIN 上用 `sglang.jit_kernel.dsv4.get_paged_mqa_logits_metadata(c4, c4_page_size, num_sms)`。

**修复 3 — `paged_mqa_metadata.cuh` 加 smem 守卫。** 第二次报 `paged_mqa_metadata.cuh:113: CUDA error: invalid argument`——`setup_kernel_smem_once` 里 `cudaFuncSetAttribute(128KB smem)` 超过登临设备上限。修复：
```cpp
#ifdef SGL_ON_DLIN
    return ::cudaSuccess;  // metadata kernel 解码 batch=1 只用 ~8B smem
#else
    ... cudaFuncSetAttribute(128KB) ...
#endif
```
（改完要清 tvm-ffi 的 `*_metadata_*` JIT 缓存。）

### 3.4 结果：4× 提速

| 配置 | TPOT | tok/s |
|---|---|---|
| 原始 eager（torch indexer）| 623ms | 1.60 |
| CG + torch indexer | 492ms | 2.03 |
| eager + DL indexer | 424ms | 2.36 |
| **CG + DL indexer** | **155ms** | **6.44** 🚀 |

**关键洞察：** 那个慢 torch indexer（400ms、海量小 op）的 **host launch 开销同时也把 full CG 的 replay 压死了**——换 DL op 后，eager 和 CG **一起解锁**。所以这不是单纯的"算子快一点"，而是"换掉瓶颈算子后，整条流水线（含 CG）才跑得起来"。

> **大白话：** indexer 这个部件，登临上本来该用一个高效算子，但代码因为"deep_gemm 不存在"退化成一堆零碎的 torch 小操作，光这些小操作的 CPU 启动开销就把 GPU 流水线打断了，连 CUDA Graph 都录不进去。参考 vLLM 把它换成登临自己的高效算子后，eager 和 CG 双双解锁，直接 4×。这就是 6.44 tok/s 的由��。

---

## <a id="a-3"></a>4. CUDA Graph 分析（让 6.44 稳定跑下来）

- v0.5.16 的 `cuda_graph_backend_decode` 默认 = **FULL**（`cuda_graph_config.py:112`）。
- support-dsv4 的"整图捕获"在 32GB 上 OOM。但 v0.5.16 的 **breakable CG** + "heavy capture-pool memory pressure; disabling prefill CG" 保护机制 → **decode full CG 在 mem=0.90 能跑**。
- full CG 在 mem=0.90 **偶发 deadlock**（capture pool 压力大，1 张卡 hang）——用 `dl-gpu-hang-recover` skill 的脚本恢复。
- **CG + DL indexer = 155ms（稳定）**——DL indexer 的少量高效 op 让 CG graph 更精简，反而更稳。

> **大白话：** CUDA Graph（把一整步 decode 录成一张"回放图"，省掉每步的 CPU 启动开销）是 decode 提速的大杀器，但在 32GB 卡上录整张图容易爆显存或卡死。靠 v0.5.16 的 breakable CG（录不下就自动降级）+ 关掉 prefill 的图（只录 decode）+ DL indexer 让图变精简，才把 6.44 稳定跑下来。

---

## <a id="a-4"></a>5. 解锁被静默阻塞的路径：mabs bug + JIT FP8 GEMV + MoE 摸底

到 6.44 tok/s 后，团队开始摸下一个瓶颈。这一节是阶段 A 的"侦察"——发现了几个被静默吞掉的 bug、试了几条更快的路径、并用 deferred-sync 纠正了 profile 数据。结论是：**6.44–6.45 是阶段 A 的稳定基线，瓶颈在 MoE（当时判断 87ms），但后续被纠正。**

### 5.1 mabs bug——所有更快路径被静默阻塞

追 v3/GEMMEX MoE 路径时，发现**所有替代路径都被一个 bug 静默挡住**：

- `Fp8MoEMethod.apply` 的 try/except（`fp8.py:2602`）把 `DL_MOE_ERR` 吞掉 → 退回 triton → V4 FP4 报 "Hidden size mismatch"。
- `DL_MOE_ERR` 揭示真因：**`sgl_kernel::moe_align_block_size()` 期望 8 个参数，实际收到 9 个**——签名漂移 bug。
- `kernels/ops/moe/__init__.py:74-84` 在 `ignore_invalid_expert=True` 时传 9 个参数，但 v0.5.16 的 op 只收 8 个。
- 默认的 use_moe_cu 路径绕开 mabs（trivial dispatch），所以 6.44 正常；**所有更快的路径都被这个 bug 屏蔽了**。

**修复（commit `4a85eed50`）：** 移除 9-arg 分支的 `ignore_invalid_expert`；强制 `_need_mabs=False`（GEMMEX 路径用 topk_ids 直接调用，use_moe_cu 用 trivial dispatch，根本不需要 mabs）。

**效果：** GEMMEX=4 FP4 路径**能到达 op 了**（不再退回 triton），但遇到新问题（FP4 dtype `UNKNOWN_SCALAR`——V4 的 int8 FP4 格式 ≠ GEMMEX bit=4 期望的 vLLM Hadamard-mxfp4）。

> **大白话：** 有个 try/except 把底层报错吞了，表象是"某条路不行"，真凶是个 8 参数 vs 9 参数的签名不匹配。默认路径碰巧绕开了它所以能跑，但所有更快的实验路径全被这个 bug 悄悄挡死。教训：遇到"DL 路径莫名退回 triton"，先去翻 `DL_MOE_ERR`。

### 5.2 JIT FP8 GEMV kernel（投影加速，commit `a15102673`）

**动机：** decode 投影用 dlblas GEMM（`gptq_dlblas_gemmex`），M=1 时 GEMM 的 tiling 浪费 85%+（M-tile=16 只用了 1 行）。目标是访存最优的 GEMV。

**实现：** 用 sglang 的 JIT kernel 基础设施（`add-jit-kernel` skill），写 `jit_kernel/csrc/elementwise/fp8_gemv.cuh`——每个 warp 处理 1 个输出 N，向量化 load + warp reduce 算点积 + per-channel scale。

**测试：** 正确性 `max_rel_err=0.0000`；性能 `0.170ms/call` vs dlblas 的 `0.391ms/call` = **2.3×**。

**集成效果：** 集成到 `dlblas_w8a8_block_fp8_linear`（M=1 路由到 JIT GEMV），但 **CG TPOT 不变（156ms）**——投影在 CG 里只占 ~28ms，不是大头。

### 5.3 精确 profile（deferred-sync）+ 关键纠正

**⚠️ 纠正一：indexer 不是 111ms——是 22ms。** 早期用"每调用 `torch.cuda.synchronize()`"测得 indexer 5.3ms/call × 21 = 111ms。但 sync 本身串行化了 GPU 流水线，**人为膨胀了 5×**。改用 deferred-sync（每调用只记 event，攒 21 次 sync 一次），测得真实 `1.05ms/call`、共 **22ms（5%）**。

**阶段 A 当时的 decode 拆解（eager，deferred-sync）：**

| 成分 | 时间 | 占比（当时）|
|---|---|---|
| MoE（`invoke_fused_moe_opt`）| 87ms | 56% |
| ATTN（投影 + flash_mla 18ms + indexer 22ms）| ~46ms | 30% |
| 其余（norm/rope/MHC/all-reduce）| ~22ms | 14% |

> **⚠️ 阶段 B 的后续纠正（重要）：** 上表 MoE 的 **87ms 后来被证明仍是 sync 伪影**，真实值约 **17ms**（[[dsv4-moe-true-cost]]，cuDNN 算子已是其能做到的接近上限）。这意味着阶段 A 末尾定下的"MoE 4× 优化"计划**不成立**——这也是阶段 B 转向 EAGLE 而非死磕 MoE 的根本原因（见 [§16.2](#summary)）。

### 5.4 MoE 所有路径测试（全部失败）

| 路径 | 测试 | 结果 |
|---|---|---|
| v3 direct op（`invoke_fused_moe_opt_v3`）| weight_bits=4, M≥1 | ❌ `cannot infer quant_type from (BFloat16, Char, 4)`——v3 是 FP8(w8a8) 导向 |
| v3 via vLLM `fused_experts` | `mxfp4_w4a16_moe_quant_config` | ❌ `_vllm_fa2_C` SIGABRT（.so 双注册冲突）|
| GEMMEX=4 FP4 | `gptq_dlblas_gemmex(bit=4, quant_type=0)` | ❌ V4 FP4 格式不匹配 → `UNKNOWN_SCALAR` |
| JIT FP4 per-expert GEMV | 自写 kernel | ❌ per-expert 151ms > fused 87ms（12 次 invocation 开销 > 1 次 fused）|

**关键发现：fused > per-expert。** fused op 一次 kernel 处理全部 6 个 expert；per-expert GEMV 要 12 次调用，每次 invocation 仍有 ~0.1ms setup 开销，打不过 fused。要优化 MoE 必须**fused FP4 MoE kernel 整体替换**——这是大工程（200+ 行 dlcc CUDA）。

### 5.5 投机解码初测（全部 OOM）

| 方法 | 结果 |
|---|---|
| MTP/NEXTN | V4 不支持（`"Only EAGLE and DSPARK"`）|
| EAGLE3 | V4 不支持 |
| EAGLE topk=1（CG, mem=0.90）| OOM |
| EAGLE topk=1（CG, mem=0.78, ctx=512）| OOM |
| EAGLE 4-draft（eager, mem=0.90）| OOM |
| DSPARK | 需要外部 draft model |

**Catch-22：** V4（149GB）+ EAGLE spec batch（draft+verify 的 KV + activations + draft model）+ CG workspace 在 32GB 上放不下；降低 mem_fraction → KV pool OOM。→ 阶段 A 结论"EAGLE 需要 48GB 卡才能跑"。

> **大白话（阶段 A 收尾）：** 6.44 tok/s 拿到后，侦察了一圈：① 发现一个吞报错的 bug 挡死了所有更快路径（已修，但下游还有 FP4 格式问题）；② 投影写了个 2.3× 的 GEMV，但只占零头，整体没变；③ 摸了 MoE 4 条加速路径，全失败——fused op 难以整体替换；④ 投机解码在 32GB 上全 OOM。当时的判断是"MoE 是 87ms 大头，得花大力气写 fused FP4 kernel"。**但这个判断在阶段 B 被 deferred-sync 的进一步纠正推翻了**——真瓶颈和真路径都换了。这就是阶段 B 的起点。

---

# 阶段 B：性能深挖（6.45 → 15.83 tok/s）

> 阶段 B 的全部内容是 support-dsv4 相对基线 `dl-dev-v0.5.16` 的净差异（commit `2ea25190a` + 工作区改动，19 文件 +3298/-115）。

## <a id="b-0"></a>6. 性能提升栈总览（6.45 → 15.83）

```
                          tok/s     相对前一步     相对能跑基线
  ─────────────────────────────────────────────────────────────
  阶段 A 终点（CG + DL indexer）     6.44        ——           ——
       │
       │  ① MHC Triton 移植（vLLM DL plugin PRE/POST kernel）
       ▼
                                     8.54      +32%         +32%
       │
       │  ② Indexer valid-only Triton kernel（替换 DL op）
       ▼
                                    10.16      +18%         +58%
       │
       │  （CG/ctx/mem 调参，为 EAGLE 腾显存）
       ▼
                                    13.05       ——         +102%
       │
       │  ③ EAGLE 投机解码（V4 自带 mtp.0 当 draft）
       ▼
                                    15.83      +21%        +145%
  ─────────────────────────────────────────────────────────────
```

> **怎么读：** ①②③ 是三条独立优化链路，每档标注实测增益。① 是上个 session 完成、随 commit `2ea25190a` 一起提交的；②③ 是该 commit 核心。中间 10.16→13.05 不是算子优化，是为 EAGLE 腾显存做的 `context_length=512` / `mem_fraction_static=0.95` 调参（见 [§9.3](#b-3)）。
>
> **反复踩的坑（写在前面的最重要结论）：** 投机解码的**接受率必须直接读 `num_correct_drafts`**（[§11.3](#b-5)），**绝不能从 tok/s 反推**——短生成的 warmup 会严重扭曲 tok/s，曾导致"EAGLE 被阻塞"的错误结论（推翻了阶段 A §5.5 的 OOM 判断），实际它 work（[[dsv4-eagle-breakthrough]]）。

---

## <a id="b-1"></a>7. 优化一 — MHC Triton kernel 移植（+32%，最大代码量）

**文件：** `python/sglang/jit_kernel/dsv4/dl_mhc_triton.py`（新增 1934 行，全分支最大单文件）
**背景：** V4 的 43 层里每层 attention 前后各有一个 MHC（Multi-Head Combine）变换。登临上 `tilelang` 缺失，DeepGEMM/torch fallback 约 2ms/层，占 attention 25–30%。vLLM 的登临插件有一套验证过的 Triton 实现，本优化把它移植进 sglang。

### 7.1 做了什么

MHC 分两个 kernel，每层各调一次：
- **PRE（`dl_mhc_pre_triton`）**：GEMM 后做 RMSNorm → sigmoid 门控 `post_mix` → 4×4 Sinkhorn 归一化得 `comb_mix` → 把 `pre_mix·residual` 归约成喂给 MoE 的 `layer_input`。
- **POST（`dl_mhc_post_triton`）**：头合并 `x = post⊗x_in + Σ comb·residual`。

### 7.2 代码栈（集成点）

`deepseek_v4.py:1451`（PRE，在 `_mhc_pre_impl` 内）：
```python
from sglang.srt.utils.common import is_dlin
if is_dlin():
    from sglang.jit_kernel.dsv4.dl_mhc_triton import dl_mhc_pre_triton
    post_mix, comb_mix, layer_input = dl_mhc_pre_triton(
        x.contiguous(), hc_fn.float().contiguous(),
        hc_scale.float().contiguous(), hc_base.float().contiguous(),
        self.rms_norm_eps, self.hc_eps, self.hc_eps,
        _MHC_POST_MULT_VALUE, self.hc_sinkhorn_iters, 1,
    )
    return layer_input, post_mix.squeeze(-1), comb_mix, False
```

`deepseek_v4.py:1579`（POST，在 `hc_post` 内）：
```python
from sglang.srt.utils.common import is_dlin as _is_dlin_post
if _is_dlin_post():
    from sglang.jit_kernel.dsv4.dl_mhc_triton import dl_mhc_post_triton
    return dl_mhc_post_triton(x, residual, post, comb)
```

### 7.3 移植时踩的三个 bug（关键）

| # | bug | 现象 | 根因 | 修复 | 位置 |
|---|---|---|---|---|---|
| 1 | `static_range` 编译挂死 | `dlgput` 编译 11+ 分钟出 0 个 kernel | Sinkhorn 循环用 `tl.static_range(BLOCK_SINKHORN≈20)`，20× 展开 → 登临 Triton IR 爆炸 | 改成普通 `range()`，`sinkhorn_repeat` 改成**运行时**参数 | `dl_mhc_triton.py:221/261/326/483` |
| 2 | `n_splits` 导致全乱码 | 输出全 0 / 乱码 | vLLM 用 split-K（`compute_num_split`），sglang torch 路用 `num_splits=None`；split 数不同 → 浮点求和顺序不同 → 误差经 43 层放大 | 强制 `n_splits = 1`，与 sglang 无 split-K 路径**逐位对齐** | `dl_mhc_triton.py:1563` |
| 3 | CG capture OOM（exit -9） | CUDA Graph capture 期间被 SIGKILL | Triton kernel 编译吃显存，capture 时和静态 KV 池抢内存 | `mem_fraction_static` 0.90 → **0.88** | 启动参数 |

**Fix 1 细节** —— 文件里所有 split/H/hc/j/k 循环仍是 `tl.static_range`，唯独 Sinkhorn 的 4 处改成动态 `range`：
```python
# dl_mhc_triton.py:221（全核并行 H 路径）
for iter_idx in range(sinkhorn_repeat - 1):   # ← 动态循环，不再编译期展开
```

**Fix 2 细节** —— `compute_num_split`（65 行）仍被调用以保持与 vLLM 签名一致，但返回值丢弃：
```python
# dl_mhc_triton.py:1563
n_splits = 1  # DL: force no split-K to match sglang torch path (num_splits=None)
_ = compute_num_split(...)   # 仅为 API 对齐，结果丢弃
```

> **大白话：** 每层 attention 进出时，原来用一段慢吞吞的 fallback 算 Sinkhorn 和头合并；现在换成 vLLM 在登临上调好的 Triton kernel。三个坑：① 登临编译器怕展开（20 步循环展开会卡死，改动态循环）；② 浮点 split-K 顺序不同会逐层放大成乱码（强制不拆）；③ 编译 kernel 时抢显存被杀（留 2% 余量）。修完三层 attention 快一截，43 层累加 +32%。

---

## <a id="b-2"></a>8. 优化二 — Indexer valid-only Triton kernel（+18%）

**文件：** `python/sglang/jit_kernel/dsv4/dl_mqa_logits_triton.py`（新增 147 行）
**门控：** `SGLANG_DL_IDX_TRITON=1`（默认关，方便和 DL op 做 A/B）

### 8.1 为什么 DL op 还能再优化（呼应阶段 A §3）

阶段 A 把 torch fallback（400ms）换成 DL op 拿到 6.44——但 DL op **本身仍按 padding 后的 `max_c4_seq_len` 来 launch grid**：
- 模型默认 1M context → `max_c4_seq_len ≈ 16384`（pad 到上限）
- decode 时实际有效条目只有 **3~68 个**
- DL op 对着 16384 个槽全跑一遍 → 单次 ~1700us，21 层 indexer ≈ **34ms = decode TPOT 的 29%**

这是 [[dsv4-indexer-bottleneck]]：DL op 比 torch 快（400→34ms），但仍在按���限白干。

### 8.2 自写 Triton kernel 怎么破

**核心：grid 只开 `M` 个 program，每个 program 在运行时按真实 `seq_len` 循环有效页**，grid 大小与 `max_c4_seq_len` 无关。

`dl_mqa_logits_triton.py:56`（kernel 主体）：
```python
m = tl.program_id(0)                 # grid = (M,)，固定大小 → CG 安全
sl = tl.load(seq_lens_ptr + m)
num_valid_pages = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE   # 运行时循环上界
num_valid_pages = tl.minimum(num_valid_pages, NUM_PAGES)
...
for page_slot in range(num_valid_pages):   # ← 只跑有效页，不碰 padding
    page = tl.load(block_table_ptr + ...).to(tl.int64)
    kv = tl.load(kv_data_ptr + d_offs).to(tl.float32)
    dots = tl.dot(kv, q_t).to(tl.float32)   # scores[e,h] = relu(kv·q)
    dots = tl.maximum(dots, 0.0)             # relu
    per_entry = tl.sum(dots * w[None, :], axis=1)
    logit_page = per_entry * kv_scale        # × scale
    tl.store(logits_ptr + ..., logit_page, mask=valid)
```

计算公式（和 torch ref `fp8_paged_mqa_logits_torch` 逐位一致）：
```
scores[e,h] = relu(kv_e · q_h)
logits[e]   = (Σ_h weight[h] · scores[e,h]) · kv_scale[e]
```

### 8.3 三个工程细节

1. **KV buffer layout**（从 vLLM `SetKAndS` 反推）：每页 8448 字节 = `[64×128 fp8 data][64×4 fp32 scale]`，scale 起始偏移 `2048` 个 fp32 位。
2. **输出 buffer 缓存复用**（`dl_mqa_logits_triton.py:122`）：第一次（eager warmup）分配后复用，**不在 capture 期间分配新 tensor**，否则触发 `cudaErrorStreamCaptureInvalidated`。
3. **不用 `zero()`**：下游 `topk_transform_512_pytorch_vectorized` 把 `≥ seq_len` 的无效槽 mask 成 `-inf`，未写入的垃圾槽天然被忽略，省一次全量清零。

### 8.4 集成 + A/B 校验

`indexer.py:651`（路由分发，DL 分支前置）：
```python
elif is_dlin() and os.environ.get("SGLANG_DL_IDX_TRITON"):
    def fn(q, kv_cache, weights, context_lens, block_tables, sched_meta, max_model_len, clean):
        from sglang.jit_kernel.dsv4.dl_mqa_logits_triton import dl_mqa_logits_triton
        logits = dl_mqa_logits_triton(_q, kv_u8, weights, block_tables,
                                      context_lens.reshape(-1), int(max_model_len))
        if os.environ.get("SGLANG_DL_IDX_TRITON_VERIFY"):   # 和 DL op 逐位比对前 5 次
            ref = torch.ops.sgl_kernel.fp8_fp4_paged_mqa_logits(...)
            md = (logits.float() - ref.float()).abs()
            print(f"[DL_IDX_AB] max_diff={md.max():.4f} ...")
        return logits
```

**实测：** 隔离微基准 1700us → 150us（**11×**）；端到端 8.54 → 10.16 tok/s（**+18%**）；A/B `max_diff ≈ 0`。

> **大白话：** 阶段 A 的 DL op 虽然比 torch 快，但仍是"按最大可能长度铺开干活"，decode 时 99% 白干。自写的 Triton kernel 只算真正有数据的几页，运行时间跟着真实序列长度走。一句"少干活"再换 +18%。**注意 indexer 优化是个接力赛：torch(400ms) → DL op(34ms) → Triton(3ms)。**

---

## <a id="b-3"></a>9. 优化三 — EAGLE 投机解码落地（+21%）

**配置点：** `scripts/dl/v4_decode_profile.py:76`（env-gated）
**核心：** 复用 V4 自带的 `mtp.0` NEXTN 层当 draft head，**不需要单独的 draft 模型**。

### 9.1 翻案阶段 A 的 OOM 判断

阶段 A §5.5 测出"EAGLE 在 32GB×8 上全 OOM，需要 48GB 卡"。这个结论**是错的**——根因不是显存绝对不够，而是当时没把 context 收窄 + mem 调到顶。阶段 B 用 `ctx=512 + mem=0.95` 让它跑起来了。更关键的是：之前从短生成的 tok/s 反推"接受率为 0 / 被阻塞"也是 warmup 伪影（[[dsv4-eagle-breakthrough]]）。

### 9.2 配置方式

`v4_decode_profile.py:76`：
```python
**({"speculative_algorithm": os.environ["SPEC_ALGO"],
   "speculative_num_steps":    int(os.environ.get("SPEC_NUM_STEPS", "2")),
   "speculative_eagle_topk":   int(os.environ.get("SPEC_TOPK", "1")),
   "speculative_num_draft_tokens": int(os.environ.get("SPEC_NUM_DRAFT", "4")),
  } if os.environ.get("SPEC_ALGO") else {}),
```

跑法：
```bash
SPEC_ALGO=EAGLE SPEC_NUM_STEPS=1 SPEC_TOPK=1 \
PROFILE_MEM_FRAC=0.95 PROFILE_CONTEXT_LEN=512 \
CUDA_VISIBLE_DEVICES=0,...,7 TP_SIZE=8 python scripts/dl/v4_decode_profile.py
```

### 9.3 关键调参结论

| 参数 | 取值 | 结论 |
|---|---|---|
| `SPEC_NUM_STEPS` | **1** | 最优：100% 接受，每步出 2 token |
| `SPEC_NUM_STEPS` | 2 | 14.53 tok/s（可接受，但不如 1） |
| `SPEC_NUM_STEPS` | 3 | 11.43 tok/s（接受率掉，反而更慢） |
| `SPEC_TOPK` | 1 | topk=1 即可，加大无益 |
| `mem_fraction_static` | 0.95 | ctx=512 下 32GB 卡刚好放得下 spec 验证显存 |

> **为什么需要 ctx=512 / mem=0.95（即 10.16→13.05 那一档）：** spec 验证步把 batch 从 M=1 变 M=2（draft + target），显存压力上来。默认大 ctx + 默认 mem 装不下 → 必须收窄 `context_length=512` 并把静态池顶到 0.95 才腾得出空间。这一步本身不是"算子提速"，而是"让 EAGLE 能跑起来"的前提。

### 9.4 接受率必须直接测（踩坑纠正）

早期结论"EAGLE 在登临上被阻塞"是**错的**——从短生成的 tok/s 反推接受率，被 warmup 骗了。本优化加了**直接读 `num_correct_drafts`** 的两条日志（[§11.3](#b-5)）：
```
[DL_SPEC] bs=1 block_size=2 correct_len(per-req)=[1] mean=1.00
[DL_SPEC_METRICS] bs=1 num_correct_drafts=1 accept_length=2.00
```
`accept_length = 1 + num_correct_drafts / bs`，真实可测、不被 warmup 干扰。

> **大白话：** V4 自己就带了个"小抄生成器"（mtp.0 层），EAGLE 让它先猜 1 个 token，主模型一次性验证 draft+target 两个位置，猜对就一步出 2 token（+21%）。阶段 A 说它"显存不够跑不了"是冤枉的——收窄 ctx + 顶满 mem 就能跑；而且接受率别用 tok/s 猜，直接数对了几个 draft。

---

## <a id="b-4"></a>10. 优化四 — CUDA Graph 兼容性三连修（解锁 spec CG）

这组不是"提速算子"，而是**修掉阻碍 CG（进而阻碍 spec CG）的三个 bug**。没它们，EAGLE + CG 跑不起来。

### 10.1 Fix A — `maybe_flush` 在 capture 期间同步（最关键）

**根因：** profiler 的 `maybe_flush()` 调 `torch.cuda.synchronize()`。CUDA Graph capture 期间任何 host-side sync 都**非法**，触发 `cudaErrorStreamCaptureInvalidated` 直接废掉整张图。

**杀伤力：** 它曾让团队误判"缩小 ctx 会破坏 CG"和"EAGLE CG capture 失败"，反复怀疑 kernel 本身。**真凶是这个 profiler 的 sync**。

**修复：** `dl_moe_profile.py:68`，capture 期间直接 return，事件攒着等下次 eager 调用再 flush：
```python
def maybe_flush():
    if not _ENABLE:
        return
    # host sync 在 capture 期间非法 → 跳过，事件攒到下次 eager 调用再 flush
    if torch.cuda.is_current_stream_capturing():
        return
    ...
```

### 10.2 Fix B — `dsa_backend.py` 的 `deep_gemm` 导入炸

**根因：** `dsa_backend.py:82` 顶层 `import deep_gemm`，登临上**没有 deep_gemm 包**，import 直接抛 `ImportError` 让整个 attention 后端起不来。

**修复：** `dsa_backend.py:81` 包成 try/except：
```python
if is_cuda():
    # DL begin — deep_gemm not available on DLIN; guard import.
    try:
        import deep_gemm
    except ImportError:
        deep_gemm = None
    # DL end
```

### 10.3 Fix C — 多流 overlap 与 CG capture 冲突

**根因：** indexer 的 multi-stream overlap（把 indexer 丢到备用 stream 和主 attention 并行）在 **M>1（spec 验证步）** 时与 CG capture 不兼容（`cudaErrorStreamCaptureUnsupported`）。

**修复：** `deepseek_v4.py:602`，登临上把多流阈值压到 1——decode（M=1）保留快路径，spec 验证（M>1）退回单流（CG 安全）：
```python
self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64
# DL: on DLIN, multi-stream overlap is incompatible with CG capture at M>1.
# Limit to M=1 so decode keeps the fast path, spec verify (M>1) falls back to
# single-stream (CG-safe). This unblocks EAGLE spec CG capture.
from sglang.srt.utils.common import is_dlin as _is_dlin_ms
if _is_dlin_ms():
    self._multi_stream_bs_limit = 1
```

> **大白话：** 三个挡在 CG 前面的小妖精：① 我们自己的测时工具每次会"喊 GPU 停一下读数"，录影时一喊就废（录制中不喊，录完再读）；② 登临没装 deep_gemm，硬 import 直接崩（加 try/except）；③ 多流并行和录影在 M>1 时打架（M=1 照常用，M>1 退单流）。三个都修了，EAGLE 的 CG 才录得成。

---

## <a id="b-5"></a>11. 优化五 — deferred-sync 分组件 Profiling 基础设施

**文件：** `python/sglang/srt/layers/quantization/dl_moe_profile.py`（新增 88 行）
**门控：** `SGLANG_DL_DECODE_PROFILE=1`，**关闭时零开销**（所有 `with dl_timer(...)` 在 `_ENABLED=False` 时是 no-op）

### 11.1 为什么要 deferred-sync

传统 profiler 每次 `record()` 都 `synchronize()` → 把 GPU 流水线**强制串行化**，测出来的是"被自己打断后的时间"，不是真实流水线时间。阶段 A 的 indexer "111→22ms = 5×"虚假膨胀、MoE "87ms（实为 17ms）"的可疑值，根子都是**逐次同步伪影**。

**deferred-sync：** 每次只记一对 CUDA event（不 sync），攒够 `FLUSH_EVERY`（默认 50）次才统一 sync 一次算平均。

```python
# dl_moe_profile.py（核心）
class _Region:
    def __enter__(self):
        if _ENABLED:
            self._s = torch.cuda.Event(enable_timing=True); self._s.record()
    def __exit__(self, *exc):
        if _ENABLED:
            e = torch.cuda.Event(enable_timing=True); e.record()
            a["starts"].append(self._s); a["ends"].append(e)   # 只记事件，不同步

def maybe_flush():        # 每 50 次才 sync 一次
    if torch.cuda.is_current_stream_capturing(): return      # Part 10 Fix A
    ...
    torch.cuda.synchronize()
    print(f"[DL_DECODE_PROF] {name}={avg_us:8.1f}us/call (n={a['n']})")
```

### 11.2 覆盖的组件（每个用 `with dl_timer("name")` 包住）

| 组件名 | 位置 | 测什么 |
|---|---|---|
| `qkv_proj` | `deepseek_v4.py:1171` | Q/K/V 投影 |
| `flash_mla` | `deepseek_v4.py:1224` | MLA attention 主算子 |
| `o_proj` | `deepseek_v4.py:1283` | 输出投影 wo_b |
| `attn` | `deepseek_v4.py:1671` | 整个 attention block（含 indexer alt-stream 等待） |
| `hc_pre` / `hc_post` | `deepseek_v4.py:1645/1727` | MHC 前后处理 |
| `flash_kernel` | `deepseek_v4_backend.py:1743` | 隔离 MLA kernel vs prep |
| `idx_prep` / `idx_logits` / `idx_topk` | `indexer.py:604/766/800` | indexer 三段 |
| `moe_w13` / `moe_w2` | `fp8.py:2561/2576` | MoE 两个 GEMM |
| `mlp_total` | `deepseek_v4.py:1859`（未提交） | 整个 MoE（GEMM+routing+comm） |

### 11.3 投机解码接受率直读（两条日志）

EAGLE 接受率不靠 tok/s 推，直接读 `num_correct_drafts`：

**`metrics_reporter.py:364`：**
```python
if _os.environ.get("SGLANG_DL_SPEC_DEBUG"):
    print(f"[DL_SPEC_METRICS] bs={bs} num_correct_drafts={num_correct_drafts} "
          f"accept_length={1 + num_correct_drafts / max(bs, 1):.2f}")
```

**`dflash_utils.py:585`：**
```python
if _os.environ.get("SGLANG_DL_SPEC_DEBUG"):
    _cl = correct_len.tolist()
    print(f"[DL_SPEC] bs={bs} block_size={block_size} correct_len(per-req)={_cl} "
          f"mean={sum(_cl)/max(len(_cl),1):.2f}")
```

### 11.4 差分 profile（逐组件跳过）

用一组 `SGLANG_DL_SKIP_*` 开关跳过某组件（置零），看总时间降多少 → 反推该组件真实占比：

| 开关 | 位置 | 跳过 |
|---|---|---|
| `SGLANG_DL_SKIP_QKV` | `deepseek_v4.py:1170` | q/k/v 投影 |
| `SGLANG_DL_SKIP_FLASH` | `deepseek_v4.py:1221` | MLA kernel |
| `SGLANG_DL_SKIP_OPROJ` | `deepseek_v4.py:1280` | 输出投影 |
| `SGLANG_DL_SKIP_ATTN` | `deepseek_v4.py:1669` | 整个 attention |
| `SGLANG_DL_SKIP_MOE` | `deepseek_v4.py:1856`（未提交） | 整个 MoE |
| `SGLANG_DL_SKIP_INDEXER` | `deepseek_v4.py:760` | indexer |

> **大白话：** 想知道"这 10ms 谁占大头"，光看绝对时间会被流水线骗。两个办法：(1) deferred-sync，攒 50 次再统一读，不打断 GPU；(2) 差分法——把某组件直接置零，看总时间掉多少，掉多少就是它的真实净占比。这套工具是阶段 B 所有判断的根基。

---

## <a id="b-6"></a>12. 优化六 — 模型层 DLIN 适配

### 12.1 `wo_a` 投影换掉 deep_gemm

V4 的 `wo_a`（FP8 weight-only）投影原来调 `deep_gemm.fp8_einsum`，登临没这个包。换成登临 port 的 `sgl_kernel.fp8_einsum`（和 vLLM 的 `dl deep_gemm_patch` 委托同一个）。

`deepseek_v4.py:1276`：
```python
# 旧：import deep_gemm; deep_gemm.fp8_einsum("bhr,hdr->bhd", ...)
# 新：sgl_kernel.fp8_einsum(o_fp8, o_s, wo_a, wo_scale, "bhr,hdr->bhd", list(recipe))
torch.ops.sgl_kernel.fp8_einsum(
    o_fp8, o_s,
    self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data,
    output, "bhr,hdr->bhd", list(recipe),
)
```

### 12.2 一句话改动矩阵（其余小改）

| 位置 | 改动 | 原因 |
|---|---|---|
| `deepseek_v4.py:602` | `_multi_stream_bs_limit = 1`（DLIN） | 见 [§10.3](#b-4) Fix C |
| `deepseek_v4.py:760` | indexer 受 `SGLANG_DL_SKIP_INDEXER` 门控 | 差分 profile |
| `deepseek_v4.py:1615` | 每步调 `maybe_flush()` | profiler 攒够 50 次统一打印 |
| `deepseek_v4.py:1669`（未提交） | `[DL_VERIFY_M]` 打印 verify 批 M | 排查 EAGLE 验证步批大小 |
| `deepseek_v4_backend.py:1743` | `flash_mla_with_kvcache` 包 `dl_timer("flash_kernel")` | 隔离 MLA kernel |

> **大白话：** 把"只有 NVIDIA/deep_gemm 才有"的调用全换成登临 port 好的 `sgl_kernel.*` 版本；再把所有热点用 profile 工具包起来，留一堆开关方便随时跳过某段做差分。

---

## <a id="b-7"></a>13. 优化七 — FP4 grouped-GEMV 探索性 kernel

**文件：** `python/sglang/jit_kernel/csrc/deepseek_v4/fp4_grouped_gemv.cuh`（新增 144 行）
**状态：** ⚠️ **探索性 / 微基准验证，尚未接入生产 decode 路径**（仅在 `scripts/dl/v4_fp4_gemv_test.py` 里测正确性+性能；生产 MoE 仍走 cuDNN `invoke_fused_moe_opt`）

### 13.1 动机（呼应阶段 A §5.4 的"需要 fused FP4 kernel"）

阶段 A §5.4 结论：要优化 MoE 必须**fused FP4 MoE kernel 整体替换**。这个 `.cuh` 就是朝这个方向迈的第一步——一个访存最优的 grouped FP4 GEMV。decode（M=1）时 MoE 是**访存密集型**，基线 cuDNN 在 TP8 per-rank 形状下**偏离访存地板 ~40×**（有效带宽 ~23 GB/s vs 峰值 ~1 TB/s）。但注意 [[ks38-bandwidth-starved]]：KS38 实测 HBM 只有 ~40-70 GB/s，自写 kernel 天花板是 ~2×，不是纸面的几十倍。

### 13.2 kernel 设计要点（CUDA）

```cpp
// fp4_grouped_gemv.cuh:28 — 每个 warp 负责一个输出 (slot, n)：grid = ceil(S*N/8)
__global__ void fp4_grouped_gemv_kernel(bf16* out, const bf16* x,
    const uint8* w, const uint8* sc, const int32* topk_ids,
    uint32 N, uint32 K, uint32 num_outputs) {
  const float kFp4Table[16] = {0,.5,1,1.5,2,3,4,6, 0,-.5,-1,-1.5,-2,-3,-4,-6};
  ...
  // 每个 lane 一次读 16 字节(uint4) = 32 个 FP4 = 恰好一个 e8m0 scale block
  // 32 lanes × 16B = 512 连续字节/warp-iter → 完全合并访存
  for (uint32_t it = 0; it < n_iters; ++it) {
    const uint4 wp = *reinterpret_cast<const uint4*>(w_row + byte_base);
    const float scv = exp2f(float(sc_row[ebase/32]) - 127.0f);
    ...
  }
  // 8 个独立累加器打破 serial += 依赖链（原版慢 10× 是被 ILP 卡住，不是访存）
  fp32_t p[8] = {...};
  ...
  fp32_t dot = device::warp::reduce_sum<32>(partial);
}
```

三个关键技巧：
1. **uint4 向量化 + 合并访存**：每 lane 16 字节对齐，warp 内 512 字节连续 → 榨干带宽。
2. **8 个独立累加器**：打破 `partial +=` 串行依赖（注释："原版比只读地板慢 10× → 其实是 ILP 瓶颈不是访存"）。
3. **kernel-local 反量化表**：FP4 查表放寄存器，不放 `__constant__`（dlcc 在 tvm-ffi inline-JIT 模块里不初始化 `__constant__`）。

### 13.3 为什么还没接生产

- 仅在 `v4_fp4_gemv_test.py` 做了和 cuDNN 的正确性 + 性能微基准。
- 真实 decode 路径里 MoE 还要过 routing/topk/silu_and_mul/comm，单纯换 GEMM 不够；且 KS38 带宽天花板限死收益（~2× 而非 40×）。
- 属于"准备好但未上线"的储备优化，留待 MoE 成为下一瓶颈时启用。

> **大白话：** 这是给 MoE 准备的一把"未来用的刀"——理论上把读权重的效率拉满。但登临显存带宽本来就吃紧（实测才几十 GB/s），这刀砍下去最多 2×，不是纸面几十倍，所以先放着、验证好，等 MoE 真成了瓶颈再上。

---

## <a id="b-8"></a>14. 优化八 — 诊断与基准脚本矩阵

`scripts/dl/` 下新增的一组脚本，是上面所有优化的"陪跑工具"：

| 脚本 | 用途 |
|---|---|
| `v4_decode_profile.py` | **总入口**：起 TP8 服务、warmup、deferred-sync profile、EAGLE 跑分、并发 M 测试 |
| `v4_mqa_logits_microbench.py` | indexer kernel 隔离微基准（DL op vs Triton，11× 那个数） |
| `mhc_kernel_isolate.py` / `mhc_kernel_diff.py` | MHC kernel 隔离计时 + 与 torch ref 逐位 diff |
| `v4_fp4_gemv_test.py` | FP4 GEMV kernel 正确性 + 性能（vs cuDNN） |
| `v4_moe_cudnn_microbench.py` | MoE cuDNN 路径微基准（验证 17ms vs 87ms 之争） |
| `v4_layout_dump.py` | dump KV/权重 layout（反推 indexer KV 8448 字节布局） |
| `v4_vllm_bench.py` | sglang vs vLLM 同模型同卡对比 |

`v4_decode_profile.py` 里几个关键编排（解释性能栈的测量条件）：
- `PROFILE_USE_CG=1` 才开 CG；**profiler 默认 eager**（deferred-sync 在 capture 期会跳过，CG 下没数据）。
- `PROFILE_CONTEXT_LEN` / `PROFILE_MEM_FRAC`：调 ctx 和 mem（EAGLE 的腾显存开关）。
- `SPEC_ALGO=EAGLE SPEC_NUM_STEPS=1`：开 EAGLE。
- `SGLANG_DL_SPEC_DEBUG=1`：直读接受率。
- `PROFILE_CONCURRENT>1`：并发 M 测试，诊断"verify-M=2 的 gap 是模型本身慢还是框架开销"（通往 20 tok/s 的判断依据）。

> **大白话：** 没有这些微基准脚本，"+18%""+21%"全是空口白话。每个优化都配一个最小复现脚本，能单独跑、单独比，才敢拍板"就是它带来的提升"。

---

## <a id="env"></a>15. 环境变量速查表

### 阶段 A 基线环境（V4 eager/CG，TP8）

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

### 阶段 B 优化开关（影响行为）

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `SGLANG_DL_IDX_TRITON=1` | indexer 用自写 Triton kernel（替换 DL op） | off |
| `SGLANG_DL_IDX_TRITON_VERIFY=1` | indexer A/B 逐位比对（前 5 次） | off |
| `SGLANG_DL_IDX_TORCH=1` | indexer 走 torch ref（A/B 基准） | off |
| `SPEC_ALGO=EAGLE` | 开 EAGLE 投机解码 | off |
| `SPEC_NUM_STEPS=1` | EAGLE draft 步数（1 最优） | 2 |
| `SPEC_TOPK=1` | EAGLE topk（1 即可） | 1 |

### 阶段 B Profile 开关（零开销，关闭即 no-op）

| 环境变量 | 作用 |
|---|---|
| `SGLANG_DL_DECODE_PROFILE=1` | 开 deferred-sync 分组件 profile |
| `SGLANG_DL_DECODE_FLUSH=50` | 每 50 次统一 sync 打印 |
| `SGLANG_DL_SPEC_DEBUG=1` | 直读 EAGLE 接受率（`num_correct_drafts`） |
| `SGLANG_DL_SKIP_{QKV,FLASH,OPROJ,ATTN,MOE,INDEXER}=1` | 差分 profile：跳过某组件 |

### 跑 EAGLE 的推荐组合（阶段 B 终态）

```bash
source <sdk>/env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8
export DLI_V2=ON TORCHDYNAMO_DISABLE=1 DLEOL_CACHE_SIZE=1024
export SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=2048 \
       SGLANG_DL_FP8_Q2=1 SGLANG_DL_GDN_DLIN=1
export SGLANG_DL_IDX_TRITON=1                 # indexer Triton
export SGLANG_TOPK_TRANSFORM_512_TORCH=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

SPEC_ALGO=EAGLE SPEC_NUM_STEPS=1 SPEC_TOPK=1 \
PROFILE_USE_CG=1 PROFILE_MEM_FRAC=0.95 PROFILE_CONTEXT_LEN=512 \
SGLANG_DL_SPEC_DEBUG=1 \
python scripts/dl/v4_decode_profile.py
```

---

## <a id="summary"></a>16. 踩坑总结与下一步

### 16.1 六个最值钱的教训

1. **接受率直接测，别从 tok/s 推。** 短生成的 warmup 伪影骗了团队两次（阶段 A 判 EAGLE OOM、又判被阻塞），差点放弃。→ `num_correct_drafts` 是唯一可信源。
2. **profiler 自己的 sync 会废掉 CG。** "ctx 缩小破坏 CG"的锅，真凶是 `maybe_flush` 的 host sync，不是 kernel。→ deferred-sync + capture 期跳过。
3. **sync 伪影能把时间放大 5×。** indexer 111ms 实为 22ms；MoE 87ms 实为 17ms。逐次 `synchronize()` 测出来的时间一律存疑。
4. **浮点 split-K 不对齐会逐层放大成乱码。** vLLM 的 split-K 和 sglang 的 no-split 必须统一成 `n_splits=1` 才逐位一致。
5. **登临编译器怕展开。** `tl.static_range` 的大循环展开能卡死编译器（11 分钟），改动态 `range`。
6. **算子按"配置上限"铺开是大坑。** indexer DL op 按 `max_c4_seq_len` 跑，decode 时 99% 白干。自写 kernel 按 `seq_len` 跑 → 11×。

### 16.2 性能栈里"没提速"的真相（防误读）

- **MoE 实测 ~17ms/step**（不是阶段 A blog 的 87ms）——后者是逐次 sync 伪影，[[dsv4-moe-true-cost]]。所以阶段 A 末尾的"4× 优化 MoE"计划**不成立**（cuDNN 已接近其能做到的上限，自写天花板 ~2×）。**这正是阶段 B 转向 EAGLE、不死磕 MoE 的根本原因。**
- `flash_mla` 是 vendored dldnn kernel，和 vLLM 同源，[[dsv4-flashmla-tapped]]——没有可 port 的空间，topk=512 已最优。
- 因此通往 20 tok/s 的路不在"再优化某个算子"，而在 **EAGLE 提接受率 / 并发摊销**（`PROFILE_CONCURRENT` 测的就是这条）。

### 16.3 下一步

| 方向 | 动作 | 预期 |
|---|---|---|
| EAGLE 深化 | `SPEC_NUM_STEPS=2` 调优 + 更大 draft，提 `accept_length` 到 3 | 推向 18-20 tok/s |
| 并发摊销 | `PROFILE_CONCURRENT=2/4` 测 per-req tok/s 是否保持 | 确认是否模型缩放瓶颈 |
| FP4 GEMV 上线 | 把 `fp4_grouped_gemv.cuh` 接入 decode MoE 路径 | MoE ~2×（带宽天花板） |
| 工作区改动提交 | `deepseek_v4.py`（`mlp_total`/`DL_VERIFY_M`）/ `v4_decode_profile.py`（并发路径）尚未 commit | 收尾 |

---

## <a id="changelog"></a>17. 改动清单与提交历史

### 阶段 A 提交（0 → 6.44）

```
be295fecef  feat(dl): DeepSeek-V4-Flash on DLIN v0.5.16 — port + indexer DL op (1.6→6.44 tok/s)
12b1a7f06e  docs(dl): V4-Flash v0.5.16 indexer DL-op perf blog (1.6->6.44 tok/s, 4x)
4a85eed50d  fix(dl): bypass broken moe_align_block_size (9-vs-8 sig drift) + GEMMEX=4 FP4 MoE path
a15102673d  feat(dl): JIT FP8 GEMV kernel for M=1 decode projections
1bc613eb50  fix(dl): balance DL markers for JIT GEMV path in fp8_utils.py
d131e4375   docs(dl): V4-Flash v0.5.16 完整调试与性能优化中文 blog  ← merge-base（基线终点）
```

### 阶段 B 提交（6.45 → 15.83）

```
2ea25190a   feat(dl): V4-Flash indexer kernel + EAGLE + profiling — 8.54→15.83 tok/s (+85%)
```

### 阶段 B 新增文件

- `python/sglang/jit_kernel/dsv4/dl_mhc_triton.py`（1934，MHC 移植）
- `python/sglang/jit_kernel/dsv4/dl_mqa_logits_triton.py`（147，indexer kernel）
- `python/sglang/jit_kernel/csrc/deepseek_v4/fp4_grouped_gemv.cuh`（144，FP4 探索）
- `python/sglang/srt/layers/quantization/dl_moe_profile.py`（88，profiler）
- `scripts/dl/v4_decode_profile.py`、`v4_fp4_gemv_test.py`、`v4_layout_dump.py`、`v4_moe_cudnn_microbench.py`、`v4_mqa_logits_microbench.py`、`v4_vllm_bench.py`、`mhc_kernel_diff.py`、`mhc_kernel_isolate.py`

### 阶段 B 修改文件

- `python/sglang/srt/models/deepseek_v4.py`（fp8_einsum 替换、多流限制、profiler 包裹、差分开关）
- `python/sglang/srt/layers/attention/dsv4/indexer.py`（Triton 路由 + A/B）
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`（flash_kernel 计时）
- `python/sglang/srt/layers/attention/dsa_backend.py`（deep_gemm 导入守卫）
- `python/sglang/srt/layers/quantization/fp8.py`（MoE GEMM 计时）
- `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`（接受率日志）
- `python/sglang/srt/speculative/dflash_utils.py`（接受率日志）

### 未提交工作区改动

- `deepseek_v4.py`：`mlp_total` 计时 + `SGLANG_DL_SKIP_MOE` + `[DL_VERIFY_M]` 批 M 日志
- `v4_decode_profile.py`：`PROFILE_CONCURRENT` 并发测试路径

# DeepSeek-V4-Flash on DLIN sglang v0.5.16 — 从 1.6 到 6.44 tok/s（indexer DL op 优化）

**日期：** 2026-07-29
**分支：** `dl-dev-v0.5.16-support-dsv4`
**模型：** DeepSeek-V4-Flash（149GB，43 层 MHC+MLA 混合，256 experts top-6，FP4 MoE）
**硬件：** DLIN KS38 × 8（32GB/卡），TP8
**结论：** eager 1.60→2.36 tok/s；full CG 2.03→**6.44 tok/s（4×）**，输出正确（" Paris"）。

---

## TL;DR

`support-dsv4` 旧分支已对当前 `.venv` **整体漂移**（v0.5.16 的 Phase 4e 把算子从 `torch.ops._dl_C.*` 迁到 `torch.ops.sgl_kernel.*`，旧分支 7 个 V4 算子引用全部 MISSING，eager 直接报错）。把 DLIN-V4 工作移植到新分支 `dl-dev-v0.5.16-support-dsv4` 后，先修正确性（8 处），再用源码内 CUDA-event 计时定位到**真正的瓶颈是 indexer**（占解码 71%，~400ms，跑的是纯 torch fallback），参考登临 vLLM 的 `deep_gemm_patch` 把它路由到 DL op，配合 full CG 拿到 **4× 提速**。

最大的教训：**两次差点优化错地方**——先以为是 MLA 投影慢（实际 v0.5.16 主线已经用 `dlblas_w8a8_block_fp8_linear` 优化了），又以为是 MoE（实际只占 14%）。**不 profile 不动手。**

---

## Part 0：为什么不直接在 support-dsv4 上继续

`support-dsv4`（07-27 暂停）跑不起来了：

- 07-29 的 v0.5.16 "Phase 4e" 重构把算子 `torch.ops._dl_C.*` → `torch.ops.sgl_kernel.*`，还把 `gemma_rms_norm` 改名 `gemma_rmsnorm`。support-dsv4 里全部 7 个 `_dl_C.*` V4 算子引用在当前 `.venv` 上 **MISSING**（静态 + 运行时双重确认）。
- eager smoke 掉到 triton 路径报 "Hidden size mismatch"（FP4 权重形状）。
- 它的 DL MoE 块（fp8.py:1920）是**死代码**（runner 选了 triton）。
- 主线 `dl-dev-v0.5.16` 上**没有任何 DLIN-V4 适配**（只有上游 SM120/NVIDIA 的 V4）。所以 support-dsv4 的 DLIN-V4 工作有独特价值 → 按指示**移植到新分支**，而不是原地续命。

新分支 `dl-dev-v0.5.16-support-dsv4` 从 `dl-dev-v0.5.16`（bcf8795aa3）创建，support-dsv4 原分支保留。

---

## Part 1：正确性移植（8 处修复，eager 输出 " Paris"）

修法都是把 support-dsv4 的 DL 逻辑重新落到 v0.5.16 的结构上（namespace `_dl_C`→`sgl_kernel`、文件迁移、rename）。fix-forward（跑 → 看报错 → 修）：

| # | 文件 | 修复 |
|---|---|---|
| 1 | `dsv4/indexer.py` | 失效 import `fp8_kernel`→`kernels.ops.quantization`（**首个阻塞**，导致 AutoModel `_DeepseekV4ConfigAlias` 注册失败）+ seq_lens reshape |
| 2 | `deepseek_v4_backend.py` | `_is_dlin` + `_create_flashmla_metadata()` 返 None（sgl_kernel.flash_mla 加载 NVIDIA flashmla_ops→ImportError）+ MLA decode→`sgl_kernel.flash_mla_with_kvcache(causal=False)`（**bug #1**）+ prefill→`sgl_kernel.flash_mla_sparse_prefill_fwd` |
| 3 | `kernels/ops/layernorm/mhc.py` | torch Sinkhorn port（tilelang 缺失；v0.5.16 已有 `hc_pre/hc_post_torch_impl`，只需补 Sinkhorn）|
| 4 | `deep_gemm_wrapper/entrypoint.py` | `tf32_hc_prenorm_gemm` torch fallback（`not ENABLE_JIT_DEEPGEMM`）|
| 5 | `layernorm.py` | DL rmsnorm `else: x.contiguous()` |
| 6 | `silu_and_mul_masked_post_quant.cuh` | 结构化绑定 lambda-capture 修复（dlcc 拒绝；`auto [_expert_id,...]=get_work(); auto expert_id=_expert_id;`）|
| 7 | `fp8.py` | **DL MoE 默认 `_G()` 把 FP4 当 FP8**（**bug #3**，乱码根因）→ 按 `w13.dtype==int8` 选 mxfp4 tuple |
| 8 | （主线已有）| DL MoE 块（`sgl_kernel.invoke_fused_moe_opt`，从 `_dl_C` 迁过来）只需修 FP4 flag |

正确性门：`"The capital of France is"` → **`" Paris. The capital of France is Paris..."`**。eager 基线 **TPOT=623ms（1.60 tok/s）**。

---

## Part 2：性能定位——indexer 才是瓶颈（不是 MLA 投影，也不是 MoE）

### CG 先排除了 host 开销

full CG（默认 decode backend=FULL）= 492ms，只比 eager（623ms）快 131ms ⇒ eager 的 host launch 开销只有 ~131ms，**解码 GPU 内核时间本身就 ~490ms**。所以 CG 不是到 20 tok/s 的路，**内核本身慢**。perf-doc 的 eager CUDA-event 拆解指向 MLA 投影 ~407ms。

### 第一次差点错：MLA 投影其实已经优化了

去翻 vLLM 的 DL 插件 + sglang 代码，发现 v0.5.16 的 `_dispatch_auto_backend`（fp8_utils.py:684）**已经把 DLIN 的 MLA 投影路由到 `dlblas_w8a8_block_fp8_linear`**（gptq_dlblas_gemmex）——"correct AND fast"，triton fallback 才是 "~40× slower"。**MLA 投影不是瓶颈。**

### 拿真实拆解（源码内 CUDA-event 计时，spawn-safe）

sglang TP worker 是 spawn，父进程 monkeypatch `DeepseekV4Model.forward` 拿到 0 次调用（白做）。改成**直接在 `deepseek_v4.py` 源码里加 CUDA-event 计时**（worker import 源码，spawn-safe），gate 在 `SGLANG_DL_MOE_TIME`，每解码步打印 `[ATTN_T]`/`[MOE_T]`：

```
ATTN_T  step attn=440ms   ← 投影(dlblas,快) + flash_mla(~18ms) + indexer
MOE_T   step MoE=85ms
其余(norm/rope/MHC) ~100ms
```

**ATTN=440ms（71%）。** 投影快、flash_mla ~18ms ⇒ **indexer 独占 ~400ms，是解码瓶颈。**

### 为什么 indexer 这么慢

V4 的 indexer（c4/c128 MLA 路由 logits）在 `dsv4/indexer.py:forward_c4_indexer` 里分派：
- `use_fp4_indexer` 路径要 `from deep_gemm import fp8_fp4_paged_mqa_logits`——**`deep_gemm` 在 DLIN 缺失**，会崩；
- 所以 V4 落到非-FP4 的 **`elif SGLANG_FP8_PAGED_MQA_LOGITS_TORCH:` → 纯 torch fallback `fp8_paged_mqa_logits_torch`**（每层一堆小 torch op × 43 层）= ~400ms。

---

## Part 3：参考 vLLM 修 indexer（DL op）

### vLLM 的做法

vLLM 不改 indexer 代码，而是在 `dl_platform_plugin/patch/deep_gemm_patch.py:69` **monkeypatch `deep_gemm.fp8_fp4_paged_mqa_logits` → `torch.ops._dl_C.fp8_fp4_paged_mqa_logits`**（DL op）。FP8 路径 `q_scale=None`（V4 KV cache 本来就是 fp8_e4m3）。

### 移植到 sglang（3 个修复）

**1. `dsv4/indexer.py`——DLIN 分支调 DL op**（在 torch fallback 之前）：
```python
elif is_dlin():
    def fn(q, kv_cache, weights, context_lens, block_tables, sched_meta, max_model_len, clean):
        _q = q if isinstance(q, torch.Tensor) else q[0]
        _qs = None if isinstance(q, torch.Tensor) else q[1]
        _sm = sched_meta if sched_meta is not None else torch.empty(0, dtype=torch.int32, device=_q.device)
        return torch.ops.sgl_kernel.fp8_fp4_paged_mqa_logits(
            _q.contiguous(), _qs, kv_cache, weights.float(),
            context_lens, block_tables, _sm, int(max_model_len), clean)
```
签名对齐 vLLM wrapper：`(q, kv_cache, weights, ctx_lens, page_table, sched_meta, max_len, clean)`。

**2. `dsv4/metadata.py`——DLIN 上构造 `deep_gemm_metadata`**。第一次跑报 `cuDNN error: CUDNN_STATUS_BAD_PARAM`——因为 `__post_init__` 在 torch env 下把 `deep_gemm_metadata=None`，DL op 拿到空 metadata。修复：DLIN 上用 `sglang.jit_kernel.dsv4.get_paged_mqa_logits_metadata(c4, c4_page_size, num_sms)` 构造（deep_gemm 缺失，用 JIT 路径）。

**3. `paged_mqa_metadata.cuh`——smem 守卫**。第二次跑报 `paged_mqa_metadata.cuh:113: CUDA error: invalid argument`——`setup_kernel_smem_once` 里的 `cudaFuncSetAttribute(128KB smem)` 超过 DLIN 设备上限（support-dsv4 当年给 topk 加过同类守卫，但没移植过来）。修复：
```cpp
#ifdef SGL_ON_DLIN
    return ::cudaSuccess;   // metadata kernel 解码 batch=1 只用 ~8B smem，远低于默认 48KB
#else
    ... cudaFuncSetAttribute(128KB) ...
#endif
```
改完记得清 tvm-ffi 的 `*_metadata_*` JIT 缓存（hash 可能只算顶层 .cu 不算 include 的 .cuh）。

### 结果

| 配置 | TPOT | tok/s |
|---|---|---|
| 原始 eager（torch indexer）| 623ms | 1.60 |
| CG + torch indexer | 492ms | 2.03 |
| eager + **DL indexer** | 424ms | 2.36 |
| **CG + DL indexer** | **155ms** | **6.44** 🚀 |

**4× 提速**，输出正确，full CG 现在干净捕获（不再 deadlock）。关键洞察：那个慢 torch indexer（400ms、大量小 op）的 host launch 开销**同时也把 full CG 的 replay 压死了**——换 DL op 后 eager 和 CG 一起解锁。

---

## Part 4：环境 & 复现

```bash
source sdk-dlop-07-13-20-30/env.sh   # 设好 LD_LIBRARY_PATH——别手动 export LD_LIBRARY_PATH=$SDK/lib（$SDK 未设→/lib/libcurt.so 是目录→torch import OSError）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8
export DLI_V2=ON TORCHDYNAMO_DISABLE=1 DLEOL_CACHE_SIZE=1024
export SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=2048 SGLANG_DL_FP8_Q2=1 SGLANG_DL_GDN_DLIN=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_HASH_TOPK=0 SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_TOPK_TRANSFORM_512_TORCH=1 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 正确性：  .venv/bin/python scripts/dl/v4_smoke.py
# eager TPOT：.venv/bin/python scripts/dl/v4_bench_tp.py
# CG TPOT：  USE_CG=1 CG_BACKEND=full MEM_FRACTION_STATIC=0.90 .venv/bin/python scripts/dl/v4_bench_tp.py
```

卡 hang（full CG 在 mem0.90 偶发 deadlock，或残留进程）：`~/.claude/skills/dl-gpu-hang-recover/dl_gpu_hang_recover.sh`（detect + `sudo dlsmi -r`）。

---

## Part 5：离 20 tok/s 还差什么（后续）

6.44 ≠ 20。剩下 ~155ms CG 解码里，DL indexer 本身在 CG 里可能还占大头（~100ms？），其次 MoE（eager 85ms）。可继续：
1. **DL indexer 能否再降**——vLLM 用同一个 op，看是否有更优调用/config（block 尺寸、schedule_metadata 复用）。
2. **MoE 换 v3**——参考 vLLM `dl_fused_moe`（`invoke_fused_moe_opt_v3`），记忆里 v3 比 non-v3 快 3.7×。需 `load_library(vllm/_dl_C.so)` + vLLM 的 mabs（sglang 的 mabs 在 M≥~100 OOB segfault）。
3. **更大显存**给 full CG 更多 headroom（48GB 卡或 TP16）。

教训复用：每次怀疑某个组件前，先**确认它是不是已经在快路径上**（v0.5.16 主线已经做了不少 DL 优化），再决定是否值得动。

# DLIN (DLIN) 算子与 CUDA Graph 优化：vLLM → sglang 差距分析

> 目的：对照 vLLM 的DLIN fork（`/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/vllm`，分支 `dl-main`，263 个 `# DL` 标记 + ~80 个 DL-only 文件），盘点 vLLM 已接入的DLIN算子与 CUDA Graph 优化，并标注 sglang 当前支持情况与接入点，作为后续把DLIN算子接入 sglang 的路线图。
>
> 范围：sglang fork（分支 `dl-main`，81 个 `# DL` 标记）。当前 DLIN 上已跑通 Qwen3-1.7B（FA2 路径，cuda graph 关闭）。
>
> 状态图例：✅ 已接入且可用 ｜ 🟡 部分接入 / 仅 fallback / stub ｜ ❌ 未接入

---

## 0. 两套DLIN底层库（vLLM 依赖）

vLLM 的DLIN算子统一落在两套库上，sglang 接入时也会面对同样的 API 面：

| 库 | 头文件 | 对标 NVIDIA | 覆盖算子 |
|---|---|---|---|
| **dldnn-ext**（暴露为 `cudnn*` 符号） | `dldnn_ext.h` | cuDNN | MLA（FlashMLA/DeepGemm MQA）、Gated Delta Rule、Fused MoE V3 |
| **dlblas-ext / dlblasLt-ext** | `dlblas_ext.h` / `dlblasLt_ext.h` | cuBLASLt | 量化 GEMM（GPTQ/AWQ/FP8/MXFP4/GGUF）、W8A8、LoRA GEMM、FP8 einsum |

> 注：vLLM 里 `cudnnFlashMLAWithKVcache` 这类 `cudnn*` 前缀是**DLIN dldnn-ext 的 cuDNN 风格符号**，不是 NVIDIA cuDNN。所有自定义算子注册在 `dl_ops` TORCH_LIBRARY 命名空间（`csrc/dl/torch_bindings.cpp`），Python 侧通过 `torch.ops._dl_C.*` 调用。

---

## 表 1：vLLM 接入的DLIN算子 + sglang 支持情况

按类别分组。约 40 个算子，13 个类别。

### 1.1 MLA（Multi-head Latent Attention）— dldnn FlashMLA 家族 ❌ 全缺

| DLIN算子 (API 符号) | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `cudnnFlashMLASparseFwd` (+ WorkspaceSize) | BF16 稀疏 prefill MLA，按索引 gather top-K KV chunk | `vllm/v1/attention/backends/mla/flashmla_sparse.py`；C++ `flash_mla_sparse_prefill_fwd` (`csrc/dl/flash_mla_interface.cu:88`) | ❌ | 新建 `layers/attention/dl_mla_backend.py`，在 `attention_registry.py` 按 `is_dlin()` 路由（仿 `dl_flash_attn.py`） |
| `cudnnFlashMLAWithKVcache` (+ WorkspaceSize) | 分页 MLA decode（支持 FP8 KV cache、block_tables、attention sink） | `flashmla.py:flash_mla_with_kvcache(_fp8)`；C++ `flash_mla_with_kvcache` (`flash_mla_interface.cu:155`) | ❌ | 同上；DeepSeek-V3.2/V4 的 decode 路径 |
| `cudnnDeepGemmMqaLogits` | FP8/FP4 MLA indexer 的 MQA logits GEMM（非分页） | `patch/deep_gemm_patch.py` → `sparse_attn_indexer.py`；C++ `fp8_fp4_mqa_logits` (`deep_gemm_mqa_logits.cu:161`) | ❌ | `model_executor/layers/sparse_attn_indexer.py`（sglang 无对应层，需先补 DeepSeek-V4 支持） |
| `cudnnDeepGemmPagedMqaLogits` | 同上，分页版（走 block_tables） | 同上；C++ `fp8_fp4_paged_mqa_logits` (`deep_gemm_mqa_logits.cu:221`) | ❌ | 同上 |

> **阻断点**：sglang 现有 8 个 MLA backend（FlashInfer/FlashMLA/Cutlass/TRTLLM/sm120/hip/tokenspeed）均依赖 NVIDIA FlashInfer/CUTLASS，DLIN 上一个都跑不了。要接DLIN MLA 必须先 vendor `dldnn_ext.h` 到 `sgl-kernel/3rdparty/`，并新建一个 `dl_mla_backend.py`。

### 1.2 Gated Delta Rule（Mamba/SSM 线性注意力）❌ 全缺

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `cudnnChunkGatedDeltaRule` (+ WS) | chunked prefill，gated delta rule（q,k,v,g,beta→out+更新 SSM state） | `ops/dl_gated_delta_rule.py`；C++ `dl_chunk_gated_delta_rule` | ❌ | `layers/attention/linear/gdn_backend.py` + `jit_kernel/cutedsl_gdn.py`，按 `is_dlin()` 路由 |
| `cudnnRecurrentGatedDeltaRule` (+ WS) | 逐 token decode，in-place 更新 ssm_state（支持 spec-decode） | 同上；C++ `dl_recurrent_gated_delta_rule` | ❌ | 同上 |

### 1.3 Fused MoE 🟡 部分（topk 有，fused GEMM 缺）

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `cudnnInvokeFusedMoeKernelV3` (+ Create/Set/Destroy descriptor, WS) | 端到端 fused grouped GEMM：反量化专家权重(fp8/int8/int4/mxfp4)+grouped GEMM+topk_weights+bias+zp | `ops/dl_fused_moe.py:dl_invoke_moe_v3`；patch `fused_experts = dl_inplace_fused_experts`；C++ `invoke_fused_moe_opt_v3` | ❌ | `layers/moe/moe_runner/` + `fused_moe_native.py`；sglang 当前 MoE 走 triton，无 DLIN fused GEMM |
| `invoke_fused_moe_opt` (V1 bool-flag 版) | 同上 V1 入口 | `_dl_ops.dl_invoke_fused_moe`；C++ `invoke_fused_moe_opt` | ❌ | 同上 |
| `moe_fused_grouped_topk` | fused grouped top-K 专家选择（带 bias + num_expert_group） | patch `grouped_topk = dl_grouped_topk`；C++ `moe_fused_grouped_topk` | 🟡 sglang 已编译 `topk_softmax`/`topk_sigmoid`/`moe_align_block_size`/`fast_topk`（真实 kernel）；但 `moe_fused_gate` 仅为 schema-only stub（`common_extension_dl.cc`），grouped_topk 变体未接 | `layers/moe/topk.py:TopKConfig.forward_cuda`；补 `csrc/moe/` + `setup_dl.py` |

### 1.4 Attention Backend（路由/封装层）✅/❌

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| DL FlashAttention backend | 封装 flash-attn FA2/FA3；强制 `_cudagraph_support=ALWAYS`、禁 cascade | `dl_flash_attn.py:FlashAttentionBackend` | ✅ **已接入且可用**（Qwen3-1.7B 走此路径） | `layers/attention/dl_flash_attn.py` + `flashattention_backend.py` DL 块 + `jit_kernel/flash_attention*.py` DL 块 |
| DL GDN Attention backend | Mamba/GDN 注意力的 DL builder（含 causal_conv1d metadata） | `dl_gdn_attn.py:GDNAttentionBackend` + `DlGDNAttentionMetadataBuilder` | ❌ | `layers/attention/linear/gdn_backend.py` |

### 1.5 RoPE（旋转位置编码）🟡 基础可用，DLIN 加速变体缺

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `deepseek_yarn_rotary_embedding` (+ batched) | DeepSeek YaRN scaled RoPE（支持多 LoRA 的 cos_sin_cache_offsets） | `ops/dl_deepseek_scaling_rope.py`；C++ kernel `csrc/dl/dl_pos_encoding_kernels.cu` | 🟡 sglang 基础 `rotary_embedding` ✅已编译可用；DeepSeek/Phi3 变体是纯 Python 类，DLIN 上走 torch（功能可用但无 DLIN kernel 加速） | `layers/rotary_embedding/factory.py:get_rope`；可选补 DLIN kernel 到 `csrc/elementwise/pos_enc.cu` |
| `longrope_rotary_embedding` (+ batched) | MiniCPM3/Phi-3 long-rope | `ops/dl_phi3_long_rope_scaled_rope.py` | 🟡 同上 | 同上 |

### 1.6 RMSNorm ✅ 已接入（2026-06-26）

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `rmsnorm` / `fused_add_rmsnorm` | 标准 RMSNorm + 残差融合（fp32 累加，bf16/fp16 输出） | vLLM `ops/dl_gemma_rms_norm.py`（gemma 变体） | ✅ **已接入并验证**：`sgl-kernel/csrc/elementwise/rmsnorm_dl.cu`（独立实现，无 FlashInfer 依赖），注册为 `torch.ops.sgl_kernel.{rmsnorm,fused_add_rmsnorm}`，`layernorm.py` 的 `is_dlin()` 分支路由到 kernel（边角情况回退 `forward_native`）。单元测试 rmsnorm 精确匹配、fused_add 在 bf16 精度内、residual 精确；`RMSNorm.forward_cuda` 集成测试通过 | `layers/layernorm.py:263`（DL 路由）、`sgl-kernel/csrc/elementwise/rmsnorm_dl.cu`、`common_extension_dl.cc`、`setup_dl.py`、`scripts/dl/test_rmsnorm_dl.py` |

> 注：vLLM 用的是 Gemma 变体（+1 offset），Qwen3 用标准 RMSNorm，故 sglang 自实现标准版而非照搬 vLLM。

### 1.7 MHC（Mixture of Heads Compute，gpt-oss 混合注意力）❌ 缺

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `tf32_hc_prenorm_gemm` + 3 个 Triton MHC kernel | 混合 chunked attention 的 pre-norm GEMM/head-fused/post-Sinkhorn | `ops/dl_mhc_triton.py`；C++ `deep_gemm_tf32_hc_prenorm_gemm.cu` | ❌ sglang 无 MHC 模型支持（优先级低） | — |

### 1.8 量化 GEMM（dlblas-ext 家族）❌ 全缺 — **最大阻断点**

底层统一入口 `gptq_dlblas_gemmex`（`csrc/dl/q_gemm_dlblas.cu`，调 `dlblasExtQuantParametersV2`），按 `quant_type`+`bit` 区分方案。

| DLIN算子 | 量化方案 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `gptq_dlblas_gemmex` | GPTQ int2/3/4/8 group-wise | `dl_quantization_plugin/gptq_dlblas.py` | ❌ | `layers/quantization/gptq/`；sglang `fp8_scaled_mm` 会在 DLIN 崩 |
| `gptq_dlblas_gemmex` (quant_type=2/1) | FP8 W8A8（block/channel-wise） | `fp8_dlblas.py` | ❌ | `layers/quantization/fp8.py:Fp8LinearMethod.apply` |
| `gptq_dlblas_gemmex` (AWQ→GPTQ) | AWQ int4/int8 | `awq_dblas.py` | ❌ | `layers/quantization/awq/` |
| `gptq_dlblas_gemmex` (mxfp4) | MXFP4（DeepSeek-V4 MoE） | `mxfp4_dlblas.py` | ❌ | `layers/quantization/mxfp4.py` |
| `gptq_dlblas_gemmex` (GGUF→GPTQ) | GGUF | `gguf_dlblas.py` | ❌ | `layers/quantization/gguf.py` |
| `gptq_dlblas_gemmex` (compressed-tensors) | W8A8/WNA16 | `compressed_tensors_dlblas.py` | ❌ | `layers/quantization/` |
| `w8a8_matmul` (`dlblasLtMatmul`) | W8A8 GEMM（per-token/per-channel scale） | `_dl_ops.w8a8_matmul`；C++ `w8a8_gemm_dlblas.cu` | ❌ | `layers/quantization/w8a8_*.py` |
| `fp8_einsum` (`cublasLt` batched strided) | 批量 FP8 GEMM（MLA down-proj） | `patch/deep_gemm_patch.py`；C++ `deep_gemm_mqa_logits.cu:289` | ❌ | DeepSeek-V4 attention |

> **阻断点**：全部依赖 FlashInfer/CUTLASS 头，sglang 的 `common_extension_dl.cc` 明确注释为 plan §5 未完成。不 vendor 这些头之前，任何量化模型在 DLIN 上都无法跑。

### 1.9 激活量化 ❌ 缺

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `per_token_group_quant_fp8_dl` | per-token-group FP8 (e4m3) 量化（支持 UE8M0 scale，MXFP4 用） | `csrc/dl/per_token_group_quant_dl.cu`（纯 CUDA，`__shfl` group-reduce） | ❌ | FP8 MoE expert 路径；sglang 无对应 DLIN kernel |

### 1.10 Sampler 🟡 部分

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `dl_top_k_sampling_from_probs` | Top-K 采样 | `ops/dl_flashinfer_sampler.py`（DL flashinfer-ext） | 🟡 sglang `sampling_backend="pytorch"` 走 torch fallback可用；无 DLIN 采样 kernel | `layers/sampler.py:Sampler.forward` |
| `dl_top_p_sampling_from_probs` | Top-P 采样 | 同上 | 🟡 同上 | 同上 |
| `dl_top_k_top_p_sampling_from_probs` | 组合采样 | 同上 | 🟡 同上 | 同上 |
| `dl_top_p_renorm_probs` / `dl_top_k_renorm_probs` | 概率重归一化 | 同上 | 🟡 `sgl_kernel.{top_k_renorm_prob,top_p_renorm_prob}` 在 DLIN 已 try/except guard | 同上 |

> 注：MoE 的 `topk_softmax`（门控）与最终 token sampler 的 top-k 是两个不同算子。前者已编译，后者无 DLIN kernel。

### 1.11 Conv（因果卷积，Mamba/线性注意力）❌ 缺

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `dl_causal_conv1d_fn` | depth-wise 因果 1D 卷积（+SiLU gating） | `ops/dl_causal_conv1d_fwd.py`（Triton JIT） | ❌ sglang 无 DLIN 分支，会走 triton conv1d | `layers/attention/mamba/causal_conv1d.py` |
| `fused_post_conv_prep` | fused post-conv 准备（transpose+pack） | `ops/dl_fused_post_conv.py` | ❌ | 同上 |

### 1.12 LoRA（punica）❌ 缺

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `dl_lora_shrink` / `dl_lora_expand` (`dlblasLoraGemm`) | 多 LoRA batched GEMM（shrink A·x → expand B·shrink+residual） | `lora/punica_wrapper/punica_dl.py`；C++ `dl_lora.cu` | ❌ sglang 无 DLIN LoRA 分支，走 `triton` backend（非 DLIN 优化） | `lora/backend/lora_registry.py`，新增 `dl` backend 或 `is_dlin()` 路由 |

### 1.13 Linear（非量化 + head padding）❌ 缺

| DLIN算子 | 作用 | vLLM 入口 | sglang 状态 | sglang 接入点 |
|---|---|---|---|---|
| `DlUnquantizedLinearMethod` (+ head padding weight loader) | 非量化 Linear，支持 DL head-padding（kernel 对齐） | `utils.py` + `linear.py:188` | ❌ | `layers/linear.py`（DeepSeek-V2/Qwen2 等有 head padding） |

### 表 1 汇总

| 类别 | 算子数 | sglang 状态 |
|---|---|---|
| MLA (dldnn) | 4 | ❌ 全缺 |
| Gated Delta Rule | 2 | ❌ 全缺 |
| Fused MoE | 3 | 🟡 topk 有 / fused GEMM 缺 |
| Attention backend | 2 | ✅ FlashAttn / ❌ GDN |
| RoPE | 4 | 🟡 基础可用 / 变体走 torch |
| RMSNorm | 2 | 🟡 stub→torch |
| MHC | 4 | ❌（低优先级） |
| 量化 GEMM | 8 | ❌ 全缺（最大阻断） |
| 激活量化 | 1 | ❌ |
| Sampler | 5 | 🟡 pytorch fallback |
| Conv1d | 2 | ❌（triton 兜底） |
| LoRA | 2 | ❌（triton 兜底） |
| Linear head-pad | 1 | ❌ |

---

## 表 2：vLLM 的 DLIN CUDA Graph 优化 + sglang 支持情况

> **基线**：sglang 当前在 DLIN 上**默认关闭 cuda graph**（`run_sglang.sh:80` `USE_CUDA_GRAPH=0` → `--disable-cuda-graph`）。所以下表所有优化在 sglang 上都尚未生效。vLLM 的核心思路是**把所有 backend 强制升到 FULL cudagraph 模式**，并把 capture 拆成 prefill/decode 两趟。

| # | 优化 | vLLM 机制（`# DL` 位置） | sglang 状态 | sglang 接入点 | 可移植性 |
|---|---|---|---|---|---|
| 1 | **强制 cudagraph_mode = FULL（否则降为 NONE）** | `config/compilation.py:1463` post-resolution 检查；`dl_config.py:192 dl_set_cuda_graph_config` 默认 FULL + compilation_mode=NONE | ❌ sglang 整体 graph 关闭 | `model_executor/cuda_graph_config.py`（`Backend.FULL/BREAKABLE/TC_PIECEWISE/DISABLED`）按 `is_dlin()` 选默认 | DL 相关（依赖各 backend 是否 graph-safe） |
| 2 | **独立的 decode capture-size 列表** `cudagraph_capture_decode_sizes` | `config/compilation.py:641,700` 新字段；`dl_config.py:110` 默认 `[1,2,3,4,6,8,16,24,32,48,64]`，spec-decode 时对齐 `1+num_speculative_tokens` | ❌ sglang 用单一 capture-size 集 | `model_executor/cuda_graph_config.py` + `runner/decode_cuda_graph_runner.py` | **通用，高价值** |
| 3 | **dispatcher 注册 decode-only FULL keys** | `patch/cudagraph_patch.py:14 _dl_initialize_cudagraph_keys`，在 prefill keys 之外补 decode×lora 的 FULL key | ❌ | sglang 无 cudagraph dispatcher 对应物（需新增或改造 capture 流程） | 通用 |
| 4 | **decode-only padding 表 + dispatch 阈值** | `patch/cudagraph_patch.py:37 _dl_compute_bs_to_padded_graph_size` + dispatch wrapper 按 `uniform_decode` 选 cap | ❌ | 同上 | 通用 |
| 5 | **multi-prefill 感知的 batch descriptor** | `patch/cudagraph_patch.py:160 _dl_create_padded_batch_descriptor`，按 `enable_dl_scheduler_for_multi_prefills` 决定 `num_reqs` | ❌ | sglang `runner/*_cuda_graph_runner.py` | 概念通用 |
| 6 | **把 FULL capture 拆成 prefill + decode 两趟** | `dl_gpu_model_runner.py:6425 capture_model` 拆非均匀(prefill)/均匀(decode)两组，中间 `synchronize()` | ❌ | `model_executor/model_runner.py:init_decode_cuda_graph` + `runner/decode_cuda_graph_runner.py:capture` | **通用，内存布局收益** |
| 7 | **GDN/Mamba attention 支持 prefill 的 FULL cudagraph**（固定 scratch buffer + 尾部 padding） | `dl_gdn_attn.py:43 _cudagraph_support=ALWAYS`；预分配 max-size 的 batch_ptr/offset_ptr/initial_state 等 buffer，把真实 metadata 拷到头部 | ❌ | `layers/attention/linear/gdn_backend.py` + `runner/*_cuda_graph_runner.py` | **技术通用**（stateful op 变 graph-safe 的标准手法） |
| 8 | **FlashAttention cudagraph 支持升为 ALWAYS** | `dl_flash_attn.py:284`（FA2 也强制 ALWAYS，非均匀 batch 走 FULL 而非 piecewise） | ❌ sglang FA backend 未声明 graph-support 级别 | `layers/attention/flashattention_backend.py` | DL 相关（依赖 DL FA 实现 graph-safe） |
| 9 | **默认关闭 cudagraph warmup，按需开启** | `dl_config.py:282 dl_extra_check_and_update_config`，默认 `cudagraph_num_of_warmups=0`，仅 CUBLAS/MiniMax-M2/LoRA/embedding 时 `=1` | ❌ | `model_executor/cuda_graph_config.py` | **通用，加速启动** |
| 10 | **capture 前预热 logits GEMM**（把 autotune 最优 plan 烤进 graph） | `dl_gpu_model_runner.py:5891` 在 `_dummy_sampler_run` 对 batch 1..`VLLM_MAX_LOGITS_WARMUP_BATCH`(16) 跑 `compute_logits`；运行时超限 `warning_once` | ❌ | `model_executor/model_runner.py` dummy-run 路径 | **通用，对任何 autotuned GEMM 有效** |
| 11 | **memory-pool tagging + cudagraph 内存 profiling** | `dl_gpu_worker.py:207,353,570` 用 `CuMemAllocator.use_memory_pool(tag=...)` 隔离 weights/kv_cache；profile 减去 cudagraph 内存 | ❌ | `model_executor/model_runner.py` 内存预算 | 概念通用 |
| 12 | encoder（视觉）cuda graph manager | 上游功能，DL 原样继承，**非 DL 新增** | — | sglang 已有 encoder graph 路径 | N/A |

### 表 2 相关 env/flag

| 名称 | 默认 | 作用 |
|---|---|---|
| `VLLM_BATCH_SIZE_CAPTURE` | None | 覆盖 capture batch-size 列表 |
| `VLLM_MAX_LOGITS_WARMUP_BATCH` | 16 | logits GEMM 预热上限 + 运行时告警阈值 |
| `VLLM_SKIP_DL_DEFAULT_CONFIG` | 0 | 所有 DL cudagraph patch 的总开关（=1 恢复上游） |
| `VLLM_MINIMAX_M2_USE_COMPILE_FUSION` | 1 | MiniMax-M2 时强制 warmup=1 |
| `enable_dl_scheduler_for_multi_prefills` | False | 决定 prefill graph 编码单/多 request |

---

## 建议的 sglang 接入顺序（路线图）

按"投入产出比 + 解锁后续工作"排序：

**P0 — 先把当前路径的性能补齐（低风险，无新依赖）**
1. **RMSNorm 真实 kernel**（🟡→✅）：补 `csrc/` norm kernel 到 `setup_dl.py` + `common_extension_dl.cc`，去掉 `layernorm.py` 的 `is_dlin()` guard。Qwen3 立即受益。
2. **Sampler DLIN kernel**（🟡→✅）：接 DL flashinfer sampler 或自实现，替换 pytorch fallback。
3. **moe_fused_gate 真实实现**（🟡→✅）：补齐 stub，让真实 MoE 模型能跑通。

**P1 — 解锁 cuda graph（性能质变，依赖 P0 完成且各 backend 通过 graph-safe 审计）**
4. 仿 vLLM 表 2 #1/#2/#6：在 `cuda_graph_config.py` 按 `is_dlin()` 选 FULL 模式 + 独立 decode capture 列表 + prefill/decode 两趟 capture。先把 graph 在 DLIN 上打开（去掉 `--disable-cuda-graph`）。
5. 表 2 #9/#10：默认关 warmup + capture 前预热 logits GEMM，降启动开销。

**P2 — 解锁量化模型（最大阻断，需 vendor 头）**
6. **vendor dldnn_ext.h / dlblas_ext.h** 到 `sgl-kernel/3rdparty/`，解决 FlashInfer/CUTLASS 头依赖（plan §5）。这是 MLA + 量化 GEMM 的共同前置。
7. 量化 GEMM（表 1.8）：FP8/W8A8 优先（DeepSeek/Qwen 量化版），接 `fp8_scaled_mm`。

**P3 — 解锁 MLA / DeepSeek**
8. MLA backend（表 1.1）：新建 `dl_mla_backend.py`，接 `cudnnFlashMLAWithKVcache`/`cudnnFlashMLASparseFwd`，在 `attention_registry.py` 按 `is_dlin()` 路由。

**P4 — 长尾**
9. LoRA、Conv1d、GDN/线性注意力、Linear head-padding（各加 `is_dlin()` 路由）。

---

## 附录：关键文件路径

**vLLM（参考源）**
- C++ 绑定：`csrc/dl/torch_bindings.cpp`、`csrc/dl/ops.h`
- Python op 面：`vllm/plugins/dl_platform_plugin/ops/_dl_ops.py`
- MLA 接口：`csrc/dl/flash_mla_interface.cu` + `vllm/v1/attention/backends/mla/flashmla_sparse.py`
- 平台调度根：`dl_platform.py`（attention/LoRA 路由）、`patch/fused_moe_related_patch.py`（MoE）、`patch/dl_patch_config.py`（patch 编排）、`patch/deep_gemm_patch.py`（monkey-patch deep_gemm）
- cuda graph：`dl_gpu_model_runner.py`、`patch/cudagraph_patch.py`、`dl_config.py`、`dl_worker.py`

**sglang（接入点）**
- DL build：`sgl-kernel/setup_dl.py`、`sgl-kernel/csrc/common_extension_dl.cc`、`sgl-kernel/pyproject_dl.toml`、`python/pyproject_dl.toml`
- 平台/检测：`python/sglang/srt/platforms/dlin.py`、`platforms/__init__.py`、`platforms/device_mixin.py`、`utils/common.py:is_dlin()`
- 已接 FA：`layers/attention/dl_flash_attn.py`、`flashattention_backend.py`、`jit_kernel/flash_attention*.py`
- 待接入：`layers/attention/attention_registry.py`（MLA/新 backend 路由）、`layers/moe/`、`layers/quantization/`、`layers/layernorm.py`、`layers/sampler.py`、`lora/backend/lora_registry.py`、`model_executor/cuda_graph_config.py`、`model_executor/runner/*_cuda_graph_runner.py`
- 约定：每次改 sglang 源先读 `.claude/skills/sglang-modify/SKILL.md`（`# DL begin/end` 标记），commit 时 `scripts/dl/check_dl_markers.py` 强制检查

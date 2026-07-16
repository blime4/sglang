# Qwen3.5-35B-A3B-FP8 DLIN 代码栈与设计原理

- **模型**：Qwen3.5-35B-A3B-FP8（FP8 blockwise [128,128]，35B MoE 256 experts×8 active，30/40 层 GatedDeltaNet linear-attn + 10/40 层 full-attn）
- **硬件**：DLIN KS38 QUAD 32GB
- **内容**：仅含已验证的代码栈、算子集成方式、原理说明

---

## 一、MoE Fused 代码栈

### 代码路径

```
Fp8MoEMethod.forward_cuda()                   # fp8.py:868  FusedMoEMethodBase
  └── is_dlin() → True, SGLANG_DL_MOE_FUSED=1
        └── invoke_fused_moe_opt               # torch.ops._dl_C （vLLM 的 _dl_C.so）
              └── cudnnInvokeFusedMoe*          # DLIN SDK <dldnn_ext.h>
```

### Guard 逻辑

```python
# fp8.py:1920
_DL_MOE_FUSED_MAX_M = int(os.environ.get("SGLANG_DL_MOE_FUSED_MAX_M", "16"))
if (
    is_dlin()
    and os.environ.get("SGLANG_DL_MOE_FUSED", "0") == "1"
    and x.shape[0] <= _DL_MOE_FUSED_MAX_M
):
```

- 默认 `FUSED_MAX_M=16`（覆盖 decode M=1 + NGRAM verify M=num_draft+1=9，不含长 prefill）
- M>16 走 bf16-bmm 路径（慢但稳定）
- M>128 会触发 DLIN JIT crash（`tu_program.cc:625 stride alignment`）

### 大白话

MoE 层收到 token，决定"每个 token 走哪 8 个 expert"的路由 + 每个 expert 做一次 FP8 GEMM。
DLIN 快路径是 `invoke_fused_moe_opt`（调 DLIN SDK `cudnnInvokeFusedMoe*`），
慢路径走 triton `fused_experts`。Guard 控制什么情况下走快路径。

---

## 二、投机解码代码栈

### 2.1 NGRAM

```
ScheduleBatch → prepare_for_draft()           # base_spec_worker.py
  └── NGRAMWorker.draft()                      # ngram_worker.py
        └── cpp_ngram 后缀树匹配（纯 CPU）
  └── prepare_for_verify()                     # 复用 EAGLE verify 骨架
        └── ForwardMode.TARGET_VERIFY          # forward_batch_info.py
```

**大白话**：NGRAM 没有草稿模型，草稿 token 来自已生成文本的 n-gram 匹配（"the capital of"后面大概率是"France"）。
草稿开销 ≈ 0，但命中率依赖 prompt 中的重复模式。
DLIN 缺 2 个 sgl_kernel op（`verify_tree_greedy`、`reconstruct_indices_from_tree_mask`）-> 用 torch fallback 补上。

### 2.2 MTP

**与 NGRAM 的区别**：
```
NGRAM: 草稿 = CPU 查 n-gram 表，耗时≈0
MTP:    草稿 = checkpoint 自带的 1 层 NextN 预测器前向（FP8 MoE + attn）
```

**代码栈**：
```
Qwen3_5ForCausalLMMTP                         # qwen3_5_mtp.py
  └── self.model = Qwen3_5ForCausalLM(         # 单层 NextN 预测器
        num_hidden_layers=1, is_nextn=True)

FrozenKVMTPWorkerV2(EAGLEWorkerV2)             # frozen_kv_mtp_worker_v2.py
  └── draft_forward → MTP 单层前向
        └── draft KV 来自 target 的 KV pool（复用的，不是自己分配的）
```

**关键发现**：Qwen3.5 MTP 草稿层的 `q_proj` 与 target 的 `k_proj` 参数完全不同（max_diff=832），
说明它是 **standard NextN**（草稿需自己维护 KV，不能读 target 的 KV），不是 frozen-KV。
实现 draft 独立 KV pool 后 accept 从 0.04 → 0.106。

---

## 三、GDN Kernel 路径

### 代码栈

```
GDNBackend.forward_decode()                   # gdn_backend.py
  └── GDNKernelRegistry → 选哪个 kernel 实现
        ├── (默认) triton chunk_gated_delta_rule
        └── (DLIN) DLinGDNKernel               # gdn_dlin.py
              └── torch.ops._dl_C.dl_recurrent_gated_delta_rule
              |     └── cudnnRecurrentGatedDeltaRule (<dldnn_ext.h>)
              └── torch.ops._dl_C.dl_chunk_gated_delta_rule
                    └── cudnnChunkGatedDeltaRule  (<dldnn_ext.h>)
```

### 大白话

Qwen3.5 30/40 层用 GatedDeltaNet（线性 attention），每层有个循环状态更新。
sglang 默认走 triton kernel（Hopper 专用 `gdc_*` 指令被 stub 了，跑得慢）。
vLLM 的 `_dl_C.so` 里有 `dl_recurrent_gated_delta_rule`（用 dleol JIT 编译），调它会快很多。

**注意事项**：GDN `_dl_C` op 的 `libdleol.so`（DLIN JIT 虚拟机）有 SDK 版本兼容问题——dl19 版和 dl24 版不通用，dl24 会 segfault。

**性能占比**：GDN `dl_recurrent` op 本身只占 ~0.5ms（kprof 估算），GDN 层 38% 的大头是 conv1d + gating + projections，非 recurrent 部分。

---

## 四、Attention 路径

### Decode attention

| 路径 | 实现 |
|---|---|
| Route A（default） | flashinfer 的 `_flashinfer_attention`（dlcc 编译时 stub） |
| Route B（DLIN） | 自编 `_vllm_fa2_C.so`（flash-attention 源码 + dlcc 编译） |
| Route C（sgl-kernel） | `paged_decode_attn_dl.cu`（sgl-kernel 自写 dlcc kernel） |

### Verify attention 修复

```python
# jit_kernel/flash_attention.py:290-327 — VERIFY paged_decode_attn loop
if _seqq > 1 and ... and causal:
    _base = cache_seqlens.long() - _seqq
    for _i in range(_seqq):
        _cs = (_base + _i + 1).to(torch.int32)
        torch.ops.sgl_kernel.paged_decode_attn(
            q_r[:, i], k_cache, v_cache, page_table, _cs, out_i, _scale)
```

原路径走 FA2 的 `_fa2_kvcache`，对多 query verify 树不精确。
改为逐 token loop 调 `paged_decode_attn`（sgl-kernel 自写 dlcc kernel，vs SDPA max_err=0），
query i 只看 KV 0..(cache_seq_len - seqq + i + 1)，因果 mask 正确。

---

## 五、质量退化问题（真实 blocker）

### 现象

投机解码（NGRAM 和 MTP 都受影响）的 verify 输出不正确：
- greedy 下输出复述 prompt（"Explain how..." → "Explain how Explain how"）
- 修了一段后变为 phrase-loop（"The neural network is The neural network is"）
- temperature>0 accept=1.0 但输出多语言乱码

### 已修复

full-attn verify 路径从 FA2 `_fa2_kvcache` 改为逐 token `paged_decode_attn` loop，因果 mask 正确。
效果：输出从复述 prompt → phrase-loop，target_predict[0] 'Ex'→'The'，accept 8.5%→15.4%。

### 仍未解

修完 verify 后，greedy 下 draft 仍然退化（预测 '16' → 循环 '22'）。
target 加 `rep_penalty=1.2` 后 draft 在可预测内容上能命中 3/4，但多数 verify 命中新颖内容 → 0 accept。

结构性疑点：
- DLIN 上唯一能跑的 `topk=1` 走的是 GDN target_verify 的 chain 路径（`retrieve_parent_token=None`），疑似此路径有问题
- `topk>1` 被另一个 DLIN triton 错误（`swa_out_cache_loc` NoneType）挡住，无法对照
- 采样/logit 修补（rep_penalty、logit bias）均验证无效

---

## 六、DLIN 算子集成方式一览

| 算子 | 集成方式 | 来源 |
|---|---|---|
| **Decode attention** | Route B：自编 `_vllm_fa2_C.so` | flash-attention 源码 + dlcc 编译 |
| **paged_decode_attn** | sgl-kernel 自写 dlcc kernel | `sgl-kernel/csrc/elementwise/paged_decode_attn_dl.cu` |
| **标准 RMSNorm** | sgl-kernel 自写 dlcc kernel | `sgl-kernel/csrc/elementwise/rmsnorm_dl.cu` |
| **Gemma RMSNorm** | 借 vLLM `_dl_C.so` → `gemma_rms_norm` | 纯 CUDA kernel，可自己编 |
| **FP8 GEMM (linear)** | 借 vLLM `_dl_C.so` → `gptq_dlblas_gemmex` | 调 DLIN SDK `dlblasLtMatmul` |
| **Fused MoE** | 借 vLLM `_dl_C.so` → `invoke_fused_moe_opt` | 调 DLIN SDK `cudnnInvokeFusedMoe` |
| **GDN decode** | 借 vLLM `_dl_C.so` → `dl_recurrent_gated_delta_rule` | 调 DLIN SDK `cudnnRecurrentGatedDeltaRule` |

`_dl_C.so` 从 vLLM venv dlopen 加载（`fp8_utils.py:_ensure_dl_C()`），
它的 C++ 源码在 vLLM `csrc/dl/` 下，SDK 头文件 `<dldnn_ext.h>` / `<dlblasLt_ext.h>` 来自 DLIN SDK 安装目录。

---

## 七、关键设计信息

### ForwardMode 分类（`forward_batch_info.py`）

| Mode | 用途 |
|---|---|
| EXTEND | prefill（首次 + 追加） |
| DECODE | 自回归生成 |
| MIXED | prefill + decode 混合 batch |
| TARGET_VERIFY | 投机解码 verify 阶段 |

### AttentionType 分类（`radix_attention.py`）

| Type | 含义 |
|---|---|
| DECODER | 标准 causal attention |
| DECODER_BIDIRECTIONAL | prefix 部分双向 |
| ENCODER_ONLY | encoder-decoder 中的 encoder |

### Dispatch 逻辑

`unified_attention_with_output()` 根据 `ForwardMode` + `AttentionType` 路由到具体 kernel 实现。

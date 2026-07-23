// DL begin: vendored from Denglin vLLM fork csrc/dl/flash_mla_interface.cu (port plan 4b/4d).
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Descriptors.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <dldnn_ext.h>

namespace {

inline cudnnDataType_t getDataType(const at::Tensor& t) {
  auto scalar_type = t.scalar_type();
  if (scalar_type == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (scalar_type == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (scalar_type == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (scalar_type == at::kBFloat16) {
    return CUDNN_DATA_BFLOAT16;
  } else if (scalar_type == at::kQInt8) {
    return CUDNN_DATA_INT8;
  } else if (scalar_type == at::kInt) {
    return CUDNN_DATA_INT32;
  } else if (scalar_type == at::kFloat8_e4m3fn) {
    return CUDNN_DATA_FP8_E4M3;
  } else if (scalar_type == at::kFloat8_e5m2) {
    return CUDNN_DATA_FP8_E5M2;
  } else if (scalar_type == at::kByte) {
    return CUDNN_DATA_UINT8;
  } else if (scalar_type == at::kChar) {
    return CUDNN_DATA_INT8;
  }
  TORCH_CHECK(false, "TensorDescriptor does not support ", scalar_type);
}

inline torch::Tensor expandTo3D(torch::Tensor t) {
  if (t.dim() == 1) {
    return t.unsqueeze(0).unsqueeze(0);
  } else if (t.dim() == 2) {
    return t.unsqueeze(0);
  } else {
    return t;
  }
}

constexpr int64_t operator"" _TiB(unsigned long long n) {
  return size_t(n) << 40;
}

inline at::Tensor allocate_workspace(size_t size, const at::Tensor& other) {
  TORCH_CHECK_WITH(OutOfMemoryError, size < 1_TiB,
                   "Not enough memory for workspace!");
  return at::empty({static_cast<int64_t>(size)}, other.options().dtype(at::kByte));
}

struct TensorInfo {
  void* ptr = nullptr;
  at::native::TensorDescriptor desc;

  TensorInfo() {
    desc.set(CUDNN_DATA_FLOAT, {1, 1, 1}, {1, 1, 1});
  }

  TensorInfo(const torch::Tensor& tensor)
    : ptr(tensor.data_ptr()) {
    torch::Tensor t = expandTo3D(tensor);
    desc.set(getDataType(t), t.sizes(), t.strides());
  }

  TensorInfo(const c10::optional<torch::Tensor>& tensor) {
    if (tensor) {
      torch::Tensor t = tensor.value();
      t = expandTo3D(t);
      ptr = t.data_ptr();
      desc.set(getDataType(t), t.sizes(), t.strides());
    } else {
      desc.set(CUDNN_DATA_FLOAT, {1, 1, 1}, {1, 1, 1});
    }
  }
};

} // namespace

std::vector<torch::Tensor>
flash_mla_sparse_prefill_fwd(
    const torch::Tensor& q,                              // [s_q, h_q, d_qk] bf16
    const torch::Tensor& kv,                             // [s_kv, h_kv, d_qk] bf16
    const torch::Tensor& indices,                        // [s_q, h_kv, topk] int32
    double sm_scale,
    int64_t d_v,
    const c10::optional<torch::Tensor>& attn_sink,       // optional [h_q] float32
    const c10::optional<torch::Tensor>& topk_length,     // optional [s_q] int32
    const c10::optional<torch::Tensor>& out) {           // optional [s_q, h_q, d_v] bf16
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto handle = at::native::getCudnnHandle();

  int64_t s_q = q.size(0);
  int64_t h_q = q.size(1);

  torch::Tensor output = out.has_value() ? out.value()
    : at::empty({s_q, h_q, d_v}, q.options());
  torch::Tensor max_logits = at::empty({s_q, h_q}, q.options().dtype(at::kFloat));
  torch::Tensor softmax_lse = at::empty({s_q, h_q}, q.options().dtype(at::kFloat));

  TensorInfo q_info(q);
  TensorInfo kv_info(kv);
  TensorInfo indices_info(indices);
  TensorInfo attn_sink_info(attn_sink);
  TensorInfo topk_length_info(topk_length);
  TensorInfo out_info(output);
  TensorInfo max_logits_info(max_logits);
  TensorInfo softmax_lse_info(softmax_lse);

  size_t workspace_size = 0;
  AT_CUDNN_CHECK(cudnnFlashMLASparseFwdWorkspaceSize(
      handle,
      q_info.desc.desc(), q_info.ptr,
      kv_info.desc.desc(), kv_info.ptr,
      indices_info.desc.desc(), indices_info.ptr,
      attn_sink_info.desc.desc(), attn_sink_info.ptr,
      topk_length_info.desc.desc(), topk_length_info.ptr,
      out_info.desc.desc(), out_info.ptr,
      max_logits_info.desc.desc(), max_logits_info.ptr,
      softmax_lse_info.desc.desc(), softmax_lse_info.ptr,
      static_cast<float>(sm_scale),
      static_cast<int>(d_v),
      &workspace_size
  ));

  auto workspace = allocate_workspace(workspace_size, output);
  AT_CUDNN_CHECK(cudnnFlashMLASparseFwd(
      handle,
      q_info.desc.desc(), q_info.ptr,
      kv_info.desc.desc(), kv_info.ptr,
      indices_info.desc.desc(), indices_info.ptr,
      attn_sink_info.desc.desc(), attn_sink_info.ptr,
      topk_length_info.desc.desc(), topk_length_info.ptr,
      out_info.desc.desc(), out_info.ptr,
      max_logits_info.desc.desc(), max_logits_info.ptr,
      softmax_lse_info.desc.desc(), softmax_lse_info.ptr,
      static_cast<float>(sm_scale),
      static_cast<int>(d_v),
      workspace.data_ptr(),
      workspace_size
  ));

  return {output, max_logits, softmax_lse};
}

std::vector<torch::Tensor>
flash_mla_with_kvcache(
    const torch::Tensor& q,                                    // [B, s_q, h_q, d]
    const torch::Tensor& k_cache,                              // [num_blocks, page_block_size, h_k, d]
    const c10::optional<torch::Tensor>& block_table,           // optional [B, max_blocks]
    const c10::optional<torch::Tensor>& cache_seqlens,         // optional [B]
    int64_t head_dim_v,
    double softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    const c10::optional<torch::Tensor>& indices,               // optional [B, s_q, topk]
    const c10::optional<torch::Tensor>& attn_sink,             // optional [h_q]
    const c10::optional<torch::Tensor>& extra_k_cache,         // optional
    const c10::optional<torch::Tensor>& extra_indices_in_cache,// optional
    const c10::optional<torch::Tensor>& topk_length,           // optional [B]
    const c10::optional<torch::Tensor>& extra_topk_length,     // optional [B]
    const c10::optional<torch::Tensor>& out) {                 // optional [B, s_q, h_q, head_dim_v]
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto handle = at::native::getCudnnHandle();

  int64_t B = q.size(0);
  int64_t s_q = q.size(1);
  int64_t h_q = q.size(2);

  torch::Tensor output = out.has_value() ? out.value()
    : at::empty({B, s_q, h_q, head_dim_v}, q.options());
  torch::Tensor softmax_lse = at::empty({B, h_q, s_q}, q.options().dtype(at::kFloat));

  TensorInfo q_info(q);
  TensorInfo k_cache_info(k_cache);
  TensorInfo block_table_info(block_table);
  TensorInfo cache_seqlens_info(cache_seqlens);
  TensorInfo indices_info(indices);
  TensorInfo attn_sink_info(attn_sink);
  TensorInfo extra_k_cache_info(extra_k_cache);
  TensorInfo extra_indices_in_cache_info(extra_indices_in_cache);
  TensorInfo topk_length_info(topk_length);
  TensorInfo extra_topk_length_info(extra_topk_length);
  TensorInfo out_info(output);
  TensorInfo softmax_lse_info(softmax_lse);

  size_t workspace_size = 0;
  AT_CUDNN_CHECK(cudnnFlashMLAWithKVcacheWorkspaceSize(
      handle,
      q_info.desc.desc(), q_info.ptr,
      k_cache_info.desc.desc(), k_cache_info.ptr,
      block_table_info.desc.desc(), block_table_info.ptr,
      cache_seqlens_info.desc.desc(), cache_seqlens_info.ptr,
      indices_info.desc.desc(), indices_info.ptr,
      attn_sink_info.desc.desc(), attn_sink_info.ptr,
      extra_k_cache_info.desc.desc(), extra_k_cache_info.ptr,
      extra_indices_in_cache_info.desc.desc(), extra_indices_in_cache_info.ptr,
      topk_length_info.desc.desc(), topk_length_info.ptr,
      extra_topk_length_info.desc.desc(), extra_topk_length_info.ptr,
      out_info.desc.desc(), out_info.ptr,
      softmax_lse_info.desc.desc(), softmax_lse_info.ptr,
      static_cast<int>(head_dim_v),
      softmax_scale,
      causal,
      is_fp8_kvcache,
      &workspace_size
  ));

  auto workspace = allocate_workspace(workspace_size, output);
  AT_CUDNN_CHECK(cudnnFlashMLAWithKVcache(
      handle,
      q_info.desc.desc(), q_info.ptr,
      k_cache_info.desc.desc(), k_cache_info.ptr,
      block_table_info.desc.desc(), block_table_info.ptr,
      cache_seqlens_info.desc.desc(), cache_seqlens_info.ptr,
      indices_info.desc.desc(), indices_info.ptr,
      attn_sink_info.desc.desc(), attn_sink_info.ptr,
      extra_k_cache_info.desc.desc(), extra_k_cache_info.ptr,
      extra_indices_in_cache_info.desc.desc(), extra_indices_in_cache_info.ptr,
      topk_length_info.desc.desc(), topk_length_info.ptr,
      extra_topk_length_info.desc.desc(), extra_topk_length_info.ptr,
      out_info.desc.desc(), out_info.ptr,
      softmax_lse_info.desc.desc(), softmax_lse_info.ptr,
      static_cast<int>(head_dim_v),
      softmax_scale,
      causal,
      is_fp8_kvcache,
      workspace.data_ptr(),
      workspace_size
  ));

  return {output, softmax_lse};
}
// DL end

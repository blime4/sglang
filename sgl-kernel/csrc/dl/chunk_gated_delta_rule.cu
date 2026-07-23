// DL begin: vendored from Denglin vLLM fork csrc/dl/chunk_gated_delta_rule.cu (port plan 4b/4d).
#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Descriptors.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
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
  } else if (scalar_type == at::kLong) {
    return CUDNN_DATA_INT64;
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

struct ChunkGatedDeltaRuleArgs {

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

  TensorInfo output;
  TensorInfo q;
  TensorInfo k;
  TensorInfo v;
  TensorInfo g;
  TensorInfo beta;
  TensorInfo initial_state;
  TensorInfo cu_seqlens;

  ChunkGatedDeltaRuleArgs(
      torch::Tensor& output_,
      torch::Tensor& q_,
      torch::Tensor& k_,
      torch::Tensor& v_,
      torch::Tensor& g_,
      torch::Tensor& beta_,
      torch::Tensor& initial_state_,
      torch::Tensor& cu_seqlens_)
    : output(output_),
      q(q_),
      k(k_),
      v(v_),
      g(g_),
      beta(beta_),
      initial_state(initial_state_),
      cu_seqlens(cu_seqlens_) {}
};

struct RecurrentGatedDeltaRuleArgs {

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

  TensorInfo output;
  TensorInfo q;
  TensorInfo k;
  TensorInfo v;
  TensorInfo g;
  TensorInfo beta;
  TensorInfo ssm_state;
  TensorInfo ssm_state_indices;
  TensorInfo cu_seqlens;
  TensorInfo num_accepted_tokens;

  RecurrentGatedDeltaRuleArgs(
      torch::Tensor& output_,
      torch::Tensor& q_,
      torch::Tensor& k_,
      torch::Tensor& v_,
      torch::Tensor& g_,
      torch::Tensor& beta_,
      torch::Tensor& ssm_state_,
      torch::Tensor& ssm_state_indices_,
      const c10::optional<torch::Tensor>& cu_seqlens_,
      const c10::optional<torch::Tensor>& num_accepted_tokens_)
    : output(output_),
      q(q_),
      k(k_),
      v(v_),
      g(g_),
      beta(beta_),
      ssm_state(ssm_state_),
      ssm_state_indices(ssm_state_indices_),
      cu_seqlens(cu_seqlens_),
      num_accepted_tokens(num_accepted_tokens_) {}
};

} // namespace

void dl_chunk_gated_delta_rule(
    torch::Tensor& output,      // [1, token, v_head, v_head_dim]
    torch::Tensor& q,           // [1, token, k_head, k_head_dim]
    torch::Tensor& k,           // [1, token, k_head, k_head_dim]
    torch::Tensor& v,           // [1, token, v_head, v_head_dim]
    torch::Tensor& g,           // [1, token, v_head]
    torch::Tensor& beta,        // [1, token, v_head]
    torch::Tensor& initial_state,   // [seq, v_head, v_head_dim, k_head_dim]
    torch::Tensor& cu_seqlens,      // [seq + 1]
    double scale,
    bool use_qk_l2norm_in_kernel) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto handle = at::native::getCudnnHandle();

  ChunkGatedDeltaRuleArgs args(output, q, k, v, g, beta, initial_state, cu_seqlens);

  size_t workspace_size = 0;
  AT_CUDNN_CHECK(cudnnGetChunkGatedDeltaRuleWorkspaceSize(
      handle,
      args.q.desc.desc(),
      args.k.desc.desc(),
      args.v.desc.desc(),
      args.g.desc.desc(),
      args.beta.desc.desc(),
      args.initial_state.desc.desc(),
      args.cu_seqlens.desc.desc(),
      args.output.desc.desc(),
      scale,
      use_qk_l2norm_in_kernel,
      &workspace_size
  ));

  auto workspace = allocate_workspace(workspace_size, output);
  AT_CUDNN_CHECK(cudnnChunkGatedDeltaRule(
      handle,
      args.q.desc.desc(),
      args.q.ptr,
      args.k.desc.desc(),
      args.k.ptr,
      args.v.desc.desc(),
      args.v.ptr,
      args.g.desc.desc(),
      args.g.ptr,
      args.beta.desc.desc(),
      args.beta.ptr,
      args.initial_state.desc.desc(),
      args.initial_state.ptr,
      args.cu_seqlens.desc.desc(),
      args.cu_seqlens.ptr,
      args.output.desc.desc(),
      args.output.ptr,
      scale,
      use_qk_l2norm_in_kernel,
      workspace.data_ptr(),
      workspace_size
  ));
}

void dl_recurrent_gated_delta_rule(
    torch::Tensor& output,      // [1, token, v_head, v_head_dim]
    torch::Tensor& q,           // [1, token, k_head, k_head_dim]
    torch::Tensor& k,           // [1, token, k_head, k_head_dim]
    torch::Tensor& v,           // [1, token, v_head, v_head_dim]
    torch::Tensor& g,           // [1, token, v_head]
    torch::Tensor& beta,        // [1, token, v_head]
    torch::Tensor& ssm_state,   // [num_cache_lines, v_head, v_head_dim, k_head_dim]
    torch::Tensor& ssm_state_indices, // [1, seq, 1 + spec_num]
    const c10::optional<torch::Tensor>& cu_seqlens,   // [seq + 1]
    const c10::optional<torch::Tensor>& num_accepted_tokens,  // [1, 1, seq]
    double scale,
    bool use_qk_l2norm_in_kernel) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto handle = at::native::getCudnnHandle();

  RecurrentGatedDeltaRuleArgs args(
    output, q, k, v, g, beta,
    ssm_state, ssm_state_indices, cu_seqlens, num_accepted_tokens);

  size_t workspace_size = 0;
  AT_CUDNN_CHECK(cudnnGetRecurrentGatedDeltaRuleWorkspaceSize(
      handle,
      args.q.desc.desc(),
      args.k.desc.desc(),
      args.v.desc.desc(),
      args.g.desc.desc(),
      args.beta.desc.desc(),
      args.ssm_state.desc.desc(),
      args.ssm_state_indices.desc.desc(),
      cu_seqlens ? args.cu_seqlens.desc.desc() : nullptr,
      num_accepted_tokens ? args.num_accepted_tokens.desc.desc() : nullptr,
      args.output.desc.desc(),
      scale,
      use_qk_l2norm_in_kernel,
      &workspace_size
  ));

  auto workspace = allocate_workspace(workspace_size, output);
  AT_CUDNN_CHECK(cudnnRecurrentGatedDeltaRule(
      handle,
      args.q.desc.desc(),
      args.q.ptr,
      args.k.desc.desc(),
      args.k.ptr,
      args.v.desc.desc(),
      args.v.ptr,
      args.g.desc.desc(),
      args.g.ptr,
      args.beta.desc.desc(),
      args.beta.ptr,
      args.ssm_state.desc.desc(),
      args.ssm_state.ptr,
      args.ssm_state_indices.desc.desc(),
      args.ssm_state_indices.ptr,
      cu_seqlens ? args.cu_seqlens.desc.desc() : nullptr,
      cu_seqlens ? args.cu_seqlens.ptr : nullptr,
      num_accepted_tokens ? args.num_accepted_tokens.desc.desc() : nullptr,
      num_accepted_tokens ? args.num_accepted_tokens.ptr : nullptr,
      args.output.desc.desc(),
      args.output.ptr,
      scale,
      use_qk_l2norm_in_kernel,
      workspace.data_ptr(),
      workspace_size
  ));
}
// DL end

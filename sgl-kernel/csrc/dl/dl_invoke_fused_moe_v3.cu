// DL begin: vendored from Denglin vLLM fork csrc/dl/dl_invoke_fused_moe_v3.cu (port plan 4b/4d).
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

inline cudnnDataType_t get_data_type(const at::Tensor& t) {
  auto scalar_type = t.scalar_type();
  if (scalar_type == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (scalar_type == at::kBFloat16) {
    return CUDNN_DATA_BFLOAT16;
  } else if (scalar_type == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (scalar_type == at::kByte) {
    return CUDNN_DATA_UINT8;
  } else if (scalar_type == at::kChar) {
    return CUDNN_DATA_INT8;
  } else if (scalar_type == at::kInt) {
    return CUDNN_DATA_INT32;
  } else if (scalar_type == at::kFloat8_e4m3fn) {
    return CUDNN_DATA_FP8_E4M3;
  } else if (scalar_type == at::kFloat8_e5m2) {
    return CUDNN_DATA_FP8_E5M2;
  } else {
    TORCH_CHECK(false, "Unsupported tensor dtype: ", scalar_type);
  }
}

inline torch::Tensor expand_to_3d(torch::Tensor t) {
  if (t.dim() == 1) {
    return t.unsqueeze(0).unsqueeze(0);
  } else if (t.dim() == 2) {
    return t.unsqueeze(0);
  } else {
    return t;
  }
}

inline bool is_fp8_dtype(at::ScalarType t) {
  return t == at::kFloat8_e4m3fn || t == at::kFloat8_e5m2;
}

inline cudnnInvokeFusedMoeDequantType_t infer_invoke_fused_moe_quant_type(
    const torch::Tensor& in,
    const torch::Tensor& w,
    const c10::optional<torch::Tensor>& w_scale,
    int64_t weight_bits) {
  constexpr auto kInvokeFusedMoeNoDequant =
      CUDNN_INVOKE_FUSED_MOE_NO_DEQUANT;
  constexpr auto kInvokeFusedMoeFp8W8A8 =
      CUDNN_INVOKE_FUSED_MOE_FP8_W8A8;
  constexpr auto kInvokeFusedMoeFp4W4A8 =
      CUDNN_INVOKE_FUSED_MOE_FP4_W4A8;
  constexpr auto kInvokeFusedMoeInt4W4A16 =
      CUDNN_INVOKE_FUSED_MOE_INT4_W4A16;
  constexpr auto kInvokeFusedMoeUint2W2A16 =
      CUDNN_INVOKE_FUSED_MOE_UINT2_W2A16;
  constexpr auto kInvokeFusedMoeUint3W3A16 =
      CUDNN_INVOKE_FUSED_MOE_UINT3_W3A16;
  constexpr auto kInvokeFusedMoeUint8W8A16 =
      CUDNN_INVOKE_FUSED_MOE_UINT8_W8A16;

  if (!w_scale.has_value() || weight_bits <= 0) {
    return kInvokeFusedMoeNoDequant;
  }

  const auto in_type = in.scalar_type();
  const auto w_type = w.scalar_type();
  const bool in_is_fp8 = is_fp8_dtype(in_type);
  const bool w_is_fp8 = is_fp8_dtype(w_type);

  switch (weight_bits) {
    case 8:
      if (w_is_fp8) {
        return kInvokeFusedMoeFp8W8A8;
      }
      if (w_type == at::kChar || w_type == at::kByte) {
        return kInvokeFusedMoeUint8W8A16;
      }
      break;
    case 4:
      if (w_type == at::kByte) {
        if (in_is_fp8) {
          return kInvokeFusedMoeFp4W4A8;
        }
        return kInvokeFusedMoeInt4W4A16;
      }
      break;
    case 3:
      if (w_type == at::kByte) {
        return kInvokeFusedMoeUint3W3A16;
      }
      break;
    case 2:
      if (w_type == at::kByte) {
        return kInvokeFusedMoeUint2W2A16;
      }
      break;
    default:
      break;
  }

  TORCH_CHECK(
      false,
      "invoke_fused_moe_opt_v3 cannot infer quant_type from dtype/weight_bits. "
      "in.dtype=", in.scalar_type(),
      ", w.dtype=", w.scalar_type(),
      ", weight_bits=", weight_bits);
}

struct InvokeFusedMoeArgs {
  struct TensorInfo {
    void* ptr = nullptr;
    at::native::TensorDescriptor desc;

    TensorInfo() {
      desc.set(CUDNN_DATA_FLOAT, {1, 1, 1}, {1, 1, 1});
    }

    explicit TensorInfo(const torch::Tensor& tensor) : ptr(tensor.data_ptr()) {
      torch::Tensor t = expand_to_3d(tensor);
      desc.set(get_data_type(t), t.sizes(), t.strides());
    }

    explicit TensorInfo(const c10::optional<torch::Tensor>& tensor) {
      if (tensor) {
        torch::Tensor t = expand_to_3d(tensor.value());
        ptr = t.data_ptr();
        desc.set(get_data_type(t), t.sizes(), t.strides());
      } else {
        desc.set(CUDNN_DATA_FLOAT, {1, 1, 1}, {1, 1, 1});
      }
    }
  };

  TensorInfo a;
  TensorInfo b;
  TensorInfo c;
  TensorInfo b_bias;
  TensorInfo b_scale;
  TensorInfo b_zp;
  TensorInfo topk_weights;
  TensorInfo topk_ids;
  TensorInfo sorted_token_ids;
  TensorInfo expert_ids;
  TensorInfo num_tokens_post_padded;

  InvokeFusedMoeArgs(
      const torch::Tensor& a_,
      const torch::Tensor& b_,
      const torch::Tensor& c_,
      const c10::optional<torch::Tensor>& b_bias_,
      const c10::optional<torch::Tensor>& b_scale_,
      const c10::optional<torch::Tensor>& b_zp_,
      torch::Tensor& topk_weights_,
      torch::Tensor& topk_ids_,
      torch::Tensor& sorted_token_ids_,
      torch::Tensor& expert_ids_,
      torch::Tensor& num_tokens_post_padded_)
      : a(a_),
        b(b_),
        c(c_),
        b_bias(b_bias_),
        b_scale(b_scale_),
        b_zp(b_zp_),
        topk_weights(topk_weights_),
        topk_ids(topk_ids_),
        sorted_token_ids(sorted_token_ids_),
        expert_ids(expert_ids_),
        num_tokens_post_padded(num_tokens_post_padded_) {}
};

constexpr int64_t operator"" _TiB(unsigned long long n) {
  return static_cast<int64_t>(n) << 40;
}

inline at::Tensor allocate_workspace(size_t size, const at::Tensor& other) {
  TORCH_CHECK_WITH(
      OutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
  return at::empty(
      {static_cast<int64_t>(size)}, other.options().dtype(at::kByte));
}

}  // namespace

void invoke_fused_moe_opt_v3(
    torch::Tensor& in,
    torch::Tensor& w,
    torch::Tensor& out,
    const c10::optional<torch::Tensor>& w_bias,
    const c10::optional<torch::Tensor>& w_scale,
    const c10::optional<torch::Tensor>& w_zp,
    torch::Tensor& topk_weights,
    torch::Tensor& topk_ids,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_post_padded,
    bool mul_routed_weight,
    int64_t topk,
    int64_t block_size_m,
    int64_t block_size_n,
    int64_t block_size_k,
    int64_t weight_bits,
    const std::vector<int64_t>& block_size,
    int64_t M) {
  (void)M;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(in));

  TORCH_CHECK(in.is_contiguous(), "in is not contiguous");
  TORCH_CHECK(w.is_contiguous(), "w is not contiguous");
  TORCH_CHECK(out.is_contiguous(), "out is not contiguous");
  TORCH_CHECK(topk_weights.is_contiguous(), "topk_weights is not contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids is not contiguous");
  TORCH_CHECK(
      sorted_token_ids.is_contiguous(), "sorted_token_ids is not contiguous");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids is not contiguous");
  TORCH_CHECK(
      num_tokens_post_padded.is_contiguous(),
      "num_tokens_post_padded is not contiguous");

  auto handle = at::native::getCudnnHandle();
  InvokeFusedMoeArgs args(
      in,
      w,
      out,
      w_bias,
      w_scale,
      w_zp,
      topk_weights,
      topk_ids,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded);

  bool has_zp = w_zp.has_value();
  bool has_bias = w_bias.has_value();
  size_t block_shape[2] = {0, 0};
  if (!block_size.empty()) {
    TORCH_CHECK(
        block_size.size() == 2,
        "block_size.size() should be 2 when provided");
    block_shape[0] = static_cast<size_t>(block_size[0]);
    block_shape[1] = static_cast<size_t>(block_size[1]);
  }

  cudnnInvokeFusedMoeDescriptor_t invoke_fused_moe_desc;
  auto quant_type =
      infer_invoke_fused_moe_quant_type(in, w, w_scale, weight_bits);

  AT_CUDNN_CHECK(cudnnCreateInvokeFusedMoeDescriptor(&invoke_fused_moe_desc));
  AT_CUDNN_CHECK(cudnnSetInvokeFusedMoeDescriptor(
      invoke_fused_moe_desc,
      topk,
      block_size_m,
      block_size_n,
      block_size_k,
      quant_type,
      has_zp,
      has_bias,
      mul_routed_weight,
      block_shape));

  size_t mem_size = 0;
  AT_CUDNN_CHECK(cudnnGetInvokeFusedMoeKernelWorkspaceSizeV3(
      handle,
      args.a.desc.desc(), args.a.ptr,
      args.b.desc.desc(), args.b.ptr,
      args.c.desc.desc(), args.c.ptr,
      args.b_scale.desc.desc(), args.b_scale.ptr,
      args.b_zp.desc.desc(), args.b_zp.ptr,
      args.topk_weights.desc.desc(), args.topk_weights.ptr,
      args.topk_ids.desc.desc(), args.topk_ids.ptr,
      args.sorted_token_ids.desc.desc(), args.sorted_token_ids.ptr,
      args.expert_ids.desc.desc(), args.expert_ids.ptr,
      args.num_tokens_post_padded.desc.desc(), args.num_tokens_post_padded.ptr,
      invoke_fused_moe_desc,
      &mem_size));

  auto workspace = allocate_workspace(mem_size, in);
  AT_CUDNN_CHECK(cudnnInvokeFusedMoeKernelV3(
      handle,
      args.a.desc.desc(), args.a.ptr,
      args.b.desc.desc(), args.b.ptr,
      args.c.desc.desc(), args.c.ptr,
      args.b_scale.desc.desc(), args.b_scale.ptr,
      args.b_zp.desc.desc(), args.b_zp.ptr,
      args.topk_weights.desc.desc(), args.topk_weights.ptr,
      args.topk_ids.desc.desc(), args.topk_ids.ptr,
      args.sorted_token_ids.desc.desc(), args.sorted_token_ids.ptr,
      args.expert_ids.desc.desc(), args.expert_ids.ptr,
      args.num_tokens_post_padded.desc.desc(), args.num_tokens_post_padded.ptr,
      args.b_bias.desc.desc(), args.b_bias.ptr,
      invoke_fused_moe_desc,
      workspace.data_ptr(),
      mem_size));

  AT_CUDNN_CHECK(cudnnDestroyInvokeFusedMoeDescriptor(invoke_fused_moe_desc));
}
// DL end

// DL begin: vendored from Denglin vLLM fork csrc/dl/fused_moe_opt.cu (port plan 4b/4d).
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

#include "../dispatch_utils.h"

namespace {

// Fixed constants common to both dynamic and static template versions:
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int MAX_VPT = 32;  // maximum VPT we support, > params.VPT = num_expert / num_expert_group
static constexpr int MAX_TOKENS_USE_WARP = 512;

} // namespace

namespace vllm {
namespace fused_moe {

// QQ NOTE: to handle the case for at::Half, error: more than one operator ">" matches these operands: built-in operator
// "arithmetic > arithmetic" function "operator>(const __half &, const __half &)"
template <typename T>
__device__ inline bool cmp_gt(const T& a, const T& b) {
  if constexpr (std::is_same<T, __half>::value) {
    // at::Half (or float16_t in our native case) causes ambiguity, so we cast to float.
    return static_cast<float>(a) > static_cast<float>(b);
  } else {
    // For types like float, at::BFloat16, or cutlass::half_t / cutlass::bfloat16_t, assume operator> works as expected.
    return a > b;
  }
}

template <typename T>
__device__ inline bool cmp_eq(const T& a, const T& b) {
  if constexpr (std::is_same<T, __half>::value) {
    return static_cast<float>(a) == static_cast<float>(b);
  } else {
    return a == b;
  }
}

#define SHFL_MASK 0xffffffff

template <typename T, int L>
struct TConvert {};

template <int L>
struct TConvert<float, L> {
  using type = float;
  using vvtype = alignas(2 * L) std::array<type, L>;
};

template <int L>
struct TConvert<c10::Half, L> {
  using type = __half;
  using vvtype = alignas(L) std::array<type, L>;
};

template <int L>
struct TConvert<c10::BFloat16, L> {
  using type = __nv_bfloat16;
  using vvtype = alignas(L) std::array<type, L>;
};

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
#ifdef CUDNN_DATA_FP8_E8M0
  } else if (scalar_type == at::kFloat8_e8m0fnu) {
    return CUDNN_DATA_FP8_E8M0;
#endif
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

} // anonymous namespace

struct InvokeFusedMoeArgs {

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

  TensorInfo A;
  TensorInfo B;
  TensorInfo C;
  TensorInfo B_bias;
  TensorInfo B_scale;
  TensorInfo B_zp;
  TensorInfo topk_weights;
  TensorInfo topk_ids;
  TensorInfo sorted_token_ids;
  TensorInfo expert_ids;
  TensorInfo num_tokens_post_padded;

  InvokeFusedMoeArgs(
      const torch::Tensor& A_,
      const torch::Tensor& B_,
      const torch::Tensor& C_,
      const c10::optional<torch::Tensor>& B_bias_,
      const c10::optional<torch::Tensor>& B_scale_,
      const c10::optional<torch::Tensor>& B_zp_,
      torch::Tensor& topk_weights_,
      torch::Tensor& topk_ids_,
      torch::Tensor& sorted_token_ids_,
      torch::Tensor& expert_ids_,
      torch::Tensor& num_tokens_post_padded_)
    : A(A_),
      B(B_),
      C(C_),
      B_bias(B_bias_),
      B_scale(B_scale_),
      B_zp(B_zp_),
      topk_weights(topk_weights_),
      topk_ids(topk_ids_),
      sorted_token_ids(sorted_token_ids_),
      expert_ids(expert_ids_),
      num_tokens_post_padded(num_tokens_post_padded_) {}
};

constexpr int64_t operator"" _TiB(unsigned long long n) {
  return size_t(n) << 40;
}

inline at::Tensor allocate_workspace(size_t size, const at::Tensor& other) {
  TORCH_CHECK_WITH(OutOfMemoryError, size < 1_TiB,
                   "Not enough memory for workspace!");
  return at::empty({static_cast<int64_t>(size)}, other.options().dtype(at::kByte));
}

// handle a row use a warp
template <typename T, typename Params>
__device__ void moe_fused_gate_small_batch_impl(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    Params params) {

  using FVType = typename TConvert<float, params.VPT>::vvtype;
  using VType = typename TConvert<T, params.VPT>::vvtype;
  using Type = typename TConvert<T, params.VPT>::type;
  constexpr int THREADS_PER_GROUP = WARP_SIZE / params.THREADS_PER_ROW;
  int tidx = threadIdx.x;
  int64_t thread_row =
      blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP;
  if (thread_row >= num_rows) {
    return;
  }

  // Cast pointers to type T:
  auto* input_ptr = reinterpret_cast<Type*>(input);
  auto* bias_ptr = reinterpret_cast<Type*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % WARP_SIZE;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;

  Type* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  FVType row_chunk;
  VType const* vec_thread_read_ptr = reinterpret_cast<VType const*>(thread_read_ptr);

  Type* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;
  FVType bias_chunk;
  VType const* vec_bias_thread_read_ptr = reinterpret_cast<VType const*>(bias_thread_read_ptr);

#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = static_cast<float>(vec_thread_read_ptr[0][ii]);
    bias_chunk[ii] = static_cast<float>(vec_bias_thread_read_ptr[0][ii]);
  }

#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    auto sigmoid_val = 1.0f / (1.0f + expf(-row_chunk[ii]));
    row_chunk[ii] = sigmoid_val;
    bias_chunk[ii] = sigmoid_val + bias_chunk[ii];
  }

  // local argmax
  float max_val = -FLT_MAX;
  float max_val_second = -FLT_MAX;
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    float val = bias_chunk[ii];

    if (cmp_gt(val, max_val)) {
      max_val_second = max_val;
      max_val = val;
    } else if (cmp_gt(val, max_val_second)) {
      max_val_second = val;
    }
  }

#pragma unroll
  for (int mask = 1; mask < THREADS_PER_GROUP; mask <<= 1) {
    auto other_max_val = __shfl_xor_sync(SHFL_MASK, max_val, mask);
    auto other_max_val_second = __shfl_xor_sync(SHFL_MASK, max_val_second, mask);
    if (cmp_gt(other_max_val, max_val)) {
      max_val_second = max_val;
      max_val = other_max_val;
    } else if (cmp_gt(other_max_val, max_val_second)) {
      max_val_second = other_max_val;
    }

    if (cmp_gt(other_max_val_second, max_val_second)) {
      max_val_second = other_max_val_second;
    }
  }

  float max_sum = max_val + max_val_second;

  // find topk_group
#pragma unroll
  for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group; ++k_idx) {
    bool is_minimum = true;
    float cur_max_sum = max_sum;
    int expert = first_elt_read_by_thread;

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_GROUP; mask >>= 1) {
      float other_max_sum = __shfl_xor_sync(SHFL_MASK, cur_max_sum, mask);
      int other_expert = __shfl_xor_sync(SHFL_MASK, expert, mask);

      if (cmp_gt(cur_max_sum, other_max_sum)
          || (cmp_eq(cur_max_sum, other_max_sum) && (other_expert > expert))) {
        is_minimum = false;
        cur_max_sum = other_max_sum;
        expert = other_expert;
      }
    }

    if (is_minimum) {
      max_sum = FLT_MAX;
    }
  }

  ////////////////////// Topk //////////////////////
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    // local argmax
    float max_val = bias_chunk[0];
    int expert = first_elt_read_by_thread;

    if (!cmp_eq(max_sum, FLT_MAX)) {
#pragma unroll
      for (int ii = 1; ii < params.VPT; ++ii) {
        float val = bias_chunk[ii];
        if (cmp_gt(val, max_val)) {
          max_val = val;
          expert = first_elt_read_by_thread + ii;
        }
      }
    } else {
      max_val = -FLT_MAX;
    }

    // argmax reduce
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      float other_max = __shfl_xor_sync(SHFL_MASK, max_val, mask);
      int other_expert = __shfl_xor_sync(SHFL_MASK, expert, mask);

      // lower indices to win
      if (cmp_gt(other_max, max_val) || (cmp_eq(other_max, max_val) && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    if (k_idx < topk) {
      int thread_to_clear_in_group = expert / params.VPT;
      int64_t idx = topk * thread_row + k_idx;

      if (thread_group_idx == thread_to_clear_in_group) {
        int expert_to_clear_in_thread = expert % params.VPT;

        // clear the max value in the thread
        bias_chunk[expert_to_clear_in_thread] = -FLT_MAX;
        // store output
        output_ptr[idx] = row_chunk[expert_to_clear_in_thread];
        indices_ptr[idx] = static_cast<int32_t>(expert);
      }

      // accumulate sum
      if (thread_group_idx == 0) {
        output_sum += output_ptr[idx];
      }
    }
  }

  // renormalize
  if (thread_group_idx == 0) {
    float output_scale = 1.f / output_sum;
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t const idx = topk * thread_row + ii;
      output_ptr[idx] = output_ptr[idx] * output_scale;
    }
  }
}

// from https://github.com/sgl-project/sglang/blob/v0.4.9.post2/sgl-kernel/csrc/moe/moe_fused_gate.cu
template <typename T, typename Params>
__device__ void moe_fused_gate_impl(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    Params params) {

  using VType = typename TConvert<T, MAX_VPT>::vvtype;
  using Type = typename TConvert<T, MAX_VPT>::type;
  int tidx = threadIdx.x;
  int64_t thread_row =
      blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP + tidx / params.THREADS_PER_ROW;
  if (thread_row >= num_rows) {
    return;
  }

  // Cast pointers to type T:
  auto* input_ptr = reinterpret_cast<Type*>(input);
  auto* bias_ptr = reinterpret_cast<Type*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;

  // Create local arrays for the row chunk and bias chunk and then reinterpret the address of row_chunk as a pointer to
  // AccessType.
  Type* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  VType row_chunk;
  VType const* vec_thread_read_ptr = reinterpret_cast<VType const*>(thread_read_ptr);

  Type* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;
  VType bias_chunk;
  VType const* vec_bias_thread_read_ptr = reinterpret_cast<VType const*>(bias_thread_read_ptr);

// QQ NOTE: doing the follow will be slower than loop assign and more importantly
// have misaligned address issue when params.VPT < 8 and mismatch with MAX_VPT
// AccessType<T>* row_chunk_vec_ptr = reinterpret_cast<AccessType<T>*>(&row_chunk);
// row_chunk_vec_ptr[0] = vec_thread_read_ptr[0];
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = vec_thread_read_ptr[0][ii];
    bias_chunk[ii] = vec_bias_thread_read_ptr[0][ii];
  }

  __syncthreads();

////////////////////// Sigmoid //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = static_cast<Type>(1.0f / (1.0f + expf(-float(row_chunk[ii]))));
  }
  __syncthreads();

////////////////////// Add Bias //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
  }

////////////////////// Exclude Groups //////////////////////
#pragma unroll
  for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group;
       ++k_idx) {  // QQ NOTE Here params.THREADS_PER_ROW = num_expert_group
    int expert = first_elt_read_by_thread;
    // local argmax
    Type max_val = static_cast<Type>(-FLT_MAX);
    Type max_val_second = static_cast<Type>(-FLT_MAX);
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      Type val = bias_chunk[ii];

      if (cmp_gt(val, max_val)) {
        max_val_second = max_val;
        max_val = val;
      } else if (cmp_gt(val, max_val_second)) {
        max_val_second = val;
      }
    }

    // QQ NOTE: currently fixed to pick top2 sigmoid weight value in each expert group and sum them as the group weight
    // to select expert groups
    Type max_sum = max_val + max_val_second;

// argmin reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      Type other_max_sum =
          static_cast<Type>(__shfl_xor_sync(SHFL_MASK, static_cast<float>(max_sum), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(SHFL_MASK, expert, mask, params.THREADS_PER_ROW);

      // higher indices win
      if (cmp_gt(max_sum, other_max_sum) || (cmp_eq(other_max_sum, max_sum) && other_expert > expert)) {
        max_sum = other_max_sum;
        expert = other_expert;
      }
    }

    // clear the max value in the thread
    if (k_idx < params.THREADS_PER_ROW - topk_group) {
      int const thread_to_clear_in_group = expert / params.VPT;

      if (thread_group_idx == thread_to_clear_in_group) {
#pragma unroll
        for (int ii = 0; ii < params.VPT; ++ii) {
          bias_chunk[ii] = static_cast<Type>(FLT_MAX);
        }
      }
    }
  }

  __syncthreads();

  ////////////////////// Topk //////////////////////
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    // local argmax
    Type max_val = bias_chunk[0];
    int expert = first_elt_read_by_thread;

    if (!cmp_eq(max_val, static_cast<Type>(FLT_MAX))) {
#pragma unroll
      for (int ii = 1; ii < params.VPT; ++ii) {
        Type val = bias_chunk[ii];
        if (cmp_gt(val, max_val)) {
          max_val = val;
          expert = first_elt_read_by_thread + ii;
        }
      }
    } else {
      max_val = static_cast<Type>(-FLT_MAX);
    }

// argmax reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      Type other_max =
          static_cast<Type>(__shfl_xor_sync(SHFL_MASK, static_cast<float>(max_val), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(SHFL_MASK, expert, mask, params.THREADS_PER_ROW);

      // lower indices to win
      if (cmp_gt(other_max, max_val) || (cmp_eq(other_max, max_val) && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    if (k_idx < topk) {
      int thread_to_clear_in_group = expert / params.VPT;
      int64_t idx = topk * thread_row + k_idx;

      if (thread_group_idx == thread_to_clear_in_group) {
        int expert_to_clear_in_thread = expert % params.VPT;

        // clear the max value in the thread
        bias_chunk[expert_to_clear_in_thread] = static_cast<Type>(-FLT_MAX);

        // store output
        output_ptr[idx] = static_cast<float>(row_chunk[expert_to_clear_in_thread]);

        indices_ptr[idx] = static_cast<int32_t>(expert);
      }

      // accumulate sum
      if (thread_group_idx == 0) {
        output_sum += output_ptr[idx];
      }
    }

    __syncthreads();
  }

  ////////////////////// Rescale Output //////////////////////
  if (thread_group_idx == 0) {
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t const idx = topk * thread_row + ii;
      output_ptr[idx] = static_cast<float>(
          static_cast<Type>(output_ptr[idx]) / static_cast<Type>(output_sum));
    }
  }
}

//------------------------------------------------------------------------------
// Templated Kernel Version (using compile-time constants)
//------------------------------------------------------------------------------
template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_, int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
  static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
  static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
  static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

template <
    typename T,
    int VPT,
    int NUM_EXPERTS,
    int THREADS_PER_ROW,
    int ROWS_PER_WARP,
    int ROWS_PER_CTA,
    int WARPS_PER_CTA>
__global__ void moe_fused_gate_kernel(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk) {
  KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> params;
  if constexpr (ROWS_PER_WARP == 1) {
    moe_fused_gate_small_batch_impl<T>(input, bias, output_ptr, indices_ptr,
                                       num_rows, topk_group, topk, params);
  } else {
    moe_fused_gate_impl<T>(input, bias, output_ptr, indices_ptr,
                           num_rows, topk_group, topk, params);
  }
}

// Macro to compute compile-time constants and launch the kernel.
#define LAUNCH_MOE_GATE_CONFIG(T, EXPERTS, EXPERT_GROUP)                                                   \
  do {                                                                                                     \
    if (num_rows > MAX_TOKENS_USE_WARP) {                                                                  \
      constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);                                                      \
      /* If EXPERT_GROUP > WARP_SIZE, fall back to 1 row per warp */                                       \
      constexpr int ROWS_PER_WARP = ((EXPERT_GROUP) <= WARP_SIZE) ? (WARP_SIZE / (EXPERT_GROUP)) : 1;      \
      constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;                                          \
      vllm::fused_moe::moe_fused_gate_kernel                                                               \
          <T, VPT, (EXPERTS), (EXPERT_GROUP), ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA>                  \
          <<<num_blocks, block_dim, 0, stream>>>(                                                          \
              input.data_ptr(),                                                                            \
              bias.data_ptr(),                                                                             \
              output.data_ptr<float>(),                                                                    \
              indices.data_ptr<int32_t>(),                                                                 \
              num_rows,                                                                                    \
              topk_group,                                                                                  \
              topk);                                                                                       \
    } else {                                                                                               \
      constexpr int VPT = (EXPERTS) / (WARP_SIZE);                                                         \
      /* If EXPERT_GROUP > WARP_SIZE, fall back to 1 row per warp */                                       \
      constexpr int ROWS_PER_WARP = 1;                                                                     \
      constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;                                          \
      vllm::fused_moe::moe_fused_gate_kernel                                                               \
          <T, VPT, (EXPERTS), (EXPERT_GROUP), ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA>                  \
          <<<num_blocks, block_dim, 0, stream>>>(                                                          \
              input.data_ptr(),                                                                            \
              bias.data_ptr(),                                                                             \
              output.data_ptr<float>(),                                                                    \
              indices.data_ptr<int32_t>(),                                                                 \
              num_rows,                                                                                    \
              topk_group,                                                                                  \
              topk);                                                                                       \
    }                                                                                                      \
    dispatched = true;                                                                                     \
  } while (0)

//------------------------------------------------------------------------------
// Dynamic Kernel Version (parameters computed at runtime)
//------------------------------------------------------------------------------
struct KernelParamsDynamic {
  int VPT;
  int NUM_EXPERTS;
  int THREADS_PER_ROW;
  int ROWS_PER_WARP;
  int ROWS_PER_CTA;
  int WARPS_PER_CTA;
};

template <typename T>
__global__ void moe_fused_gate_kernel_dynamic(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk) {
  KernelParamsDynamic params;
  params.NUM_EXPERTS = num_experts;             // e.g, for deepseek v3, this is 256
  params.VPT = num_experts / num_expert_group;  // e.g., for deepseek v3, this is 256 / 8 = 32
  params.THREADS_PER_ROW = num_expert_group;    // fixed as num_expert_group, e.g., for deepseek v3, this is 8
  params.WARPS_PER_CTA = WARPS_PER_CTA;         // fixed as 6
  params.ROWS_PER_WARP = std::max<int64_t>(1, WARP_SIZE / num_expert_group);  // WARP_SIZE is fixed as 32
  params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;

  moe_fused_gate_impl<T>(input, bias, output_ptr, indices_ptr, num_rows, topk_group, topk, params);
}

} // namespace fused_moe
} // vllm

void invoke_fused_moe_opt(
    torch::Tensor& in,      // [*, K]
    torch::Tensor& w,       // [E, N, K]
    torch::Tensor& out,     // [token, topk, N]
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
    bool use_fp8_w8a8,
    bool use_int8_w8a16,
    bool use_int4_w4a16,
    bool use_mxfp4_w4a16,
    const std::vector<int64_t>& block_size,
    int64_t M) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(in));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(in.is_contiguous(), "in is not contiguous");
  TORCH_CHECK(w.is_contiguous(), "w is not contiguous");
  TORCH_CHECK(out.is_contiguous(), "out is not contiguous");
  TORCH_CHECK(topk_weights.is_contiguous(), "topk_weights is not contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids is not contiguous");
  TORCH_CHECK(sorted_token_ids.is_contiguous(), "sorted_token_ids is not contiguous");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids is not contiguous");
  TORCH_CHECK(num_tokens_post_padded.is_contiguous(), "num_tokens_post_padded is not contiguous");

  auto handle = at::native::getCudnnHandle();

  vllm::fused_moe::InvokeFusedMoeArgs args(
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
  size_t block_shape[2] = {0};
  bool has_block_shape = false;

  if (use_fp8_w8a8 || use_int8_w8a16 || use_int4_w4a16 || use_mxfp4_w4a16) {
    TORCH_CHECK(block_size.size() == 2, "block_size.size() should be 2");
    block_shape[0] = block_size[0];
    block_shape[1] = block_size[1];
    has_block_shape = true;

    if ((use_int4_w4a16 || use_mxfp4_w4a16) && block_shape[0] == 0) {
      block_shape[0] = 1;
    }
  }

  cudnnInvokeFusedMoeDescriptor_t invoke_fused_moe_desc;
  cudnnInvokeFusedMoeDequantType_t quant_type = CUDNN_INVOKE_FUSED_MOE_NO_DEQUANT;
  if (use_fp8_w8a8) {
    quant_type = CUDNN_INVOKE_FUSED_MOE_FP8_W8A8;
  } else if (use_int8_w8a16) {
    quant_type = CUDNN_INVOKE_FUSED_MOE_INT8_W8A16;
  } else if (use_int4_w4a16) {
    quant_type = CUDNN_INVOKE_FUSED_MOE_INT4_W4A16;
  } else if (use_mxfp4_w4a16) {
    quant_type = CUDNN_INVOKE_FUSED_MOE_FP4_W4A8;
  }

  bool has_bias = w_bias.has_value();

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
      args.A.desc.desc(), args.A.ptr,
      args.B.desc.desc(), args.B.ptr,
      args.C.desc.desc(), args.C.ptr,
      args.B_scale.desc.desc(), args.B_scale.ptr,
      args.B_zp.desc.desc(), args.B_zp.ptr,
      args.topk_weights.desc.desc(), args.topk_weights.ptr,
      args.topk_ids.desc.desc(), args.topk_ids.ptr,
      args.sorted_token_ids.desc.desc(), args.sorted_token_ids.ptr,
      args.expert_ids.desc.desc(), args.expert_ids.ptr,
      args.num_tokens_post_padded.desc.desc(), args.num_tokens_post_padded.ptr,
      invoke_fused_moe_desc,
      &mem_size));

  auto workspace = vllm::fused_moe::allocate_workspace(mem_size, in);
  AT_CUDNN_CHECK(cudnnInvokeFusedMoeKernelV3(
      handle,
      args.A.desc.desc(), args.A.ptr,
      args.B.desc.desc(), args.B.ptr,
      args.C.desc.desc(), args.C.ptr,
      args.B_scale.desc.desc(), args.B_scale.ptr,
      args.B_zp.desc.desc(), args.B_zp.ptr,
      args.topk_weights.desc.desc(), args.topk_weights.ptr,
      args.topk_ids.desc.desc(), args.topk_ids.ptr,
      args.sorted_token_ids.desc.desc(), args.sorted_token_ids.ptr,
      args.expert_ids.desc.desc(), args.expert_ids.ptr,
      args.num_tokens_post_padded.desc.desc(), args.num_tokens_post_padded.ptr,
      args.B_bias.desc.desc(), args.B_bias.ptr,
      invoke_fused_moe_desc,
      workspace.data_ptr(),
      mem_size));

  AT_CUDNN_CHECK(cudnnDestroyInvokeFusedMoeDescriptor(invoke_fused_moe_desc));
}

void moe_fused_grouped_topk(
    torch::Tensor& input,
    torch::Tensor& bias,
    torch::Tensor& indices,
    torch::Tensor& output,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group) {
  int64_t num_rows = input.size(0);
  int32_t num_experts = input.size(1);
  int32_t computed_vpt = num_experts / num_expert_group;

  // Compute grid dimensions based on runtime value for num_expert_group.
  int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / num_expert_group);
  if (num_rows <= MAX_TOKENS_USE_WARP) {
    rows_per_warp = 1;
    computed_vpt = num_experts / WARP_SIZE;
  }

  int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
  int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

  TORCH_CHECK(bias.dtype() == input.dtype(), "bias and input must have the same dtype");

  // Check 1: Ensure that num_experts is a power of 2.
  TORCH_CHECK((num_experts & (num_experts - 1)) == 0, "num_experts must be a power of 2, but got ", num_experts);

  // Check 2: Ensure that num_experts is divisible by num_expert_group. (this also means num_expert_group is power of 2)
  TORCH_CHECK(
      num_experts % num_expert_group == 0,
      "num_experts must be divisible by num_expert_group, but got ",
      num_experts,
      " / ",
      num_expert_group);

  // Check 3: Ensure that num_experts/num_expert_group does not exceed MAX_VPT=32. Maximum VPT indicate max value per
  // threads we can process.
  TORCH_CHECK(
      computed_vpt <= MAX_VPT,
      "Per group experts: num_experts / num_expert_group = (",
      computed_vpt,
      ") exceeds the maximum supported (",
      MAX_VPT,
      ")");

  // Dispatch to templated kernel for known compile-time configurations.
  // We currently only support for:
  //   Case 1: 256 experts, with 8 or 16 groups.
  //   Case 2: 128 experts, with 4 or 8 groups.
  //   Case 3: other cases, require 8 <= num_experts / num_expert_group <= 32
  bool dispatched = false;
  switch (num_experts) {
    case 256: {
      if (num_expert_group == 8) {
        // This is deepseek v3 case. Here VPT = 256/8 = 32, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(c10::BFloat16, 256, 8);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(c10::Half, 256, 8);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float, 256, 8);
        }
      } else if (num_expert_group == 16) {
        // Here VPT = 256/16 = 16, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2 = 12.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(c10::BFloat16, 256, 16);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(c10::Half, 256, 16);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float, 256, 16);
        }
      }
      break;
    }
    case 128: {
      if (num_expert_group == 4) {
        // VPT = 128/4 = 32, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2 = 12.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(c10::BFloat16, 128, 4);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(c10::Half, 128, 4);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float, 128, 4);
        }
      } else if (num_expert_group == 8) {
        // VPT = 128/8 = 16, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(c10::BFloat16, 128, 8);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(c10::Half, 128, 8);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float, 128, 8);
        }
      }
      break;
    }
    default:
      break;
  }
  if (!dispatched) {
    // Fallback to the dynamic kernel if none of the supported combinations match.
    // currently only support num_experts / num_expert_group <= 32 for dynamic kernels
    if (input.scalar_type() == at::kBFloat16) {
      vllm::fused_moe::moe_fused_gate_kernel_dynamic<c10::BFloat16><<<num_blocks, block_dim, 0, stream>>>(
          input.data_ptr(),
          bias.data_ptr(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk);
    } else if (input.scalar_type() == at::kHalf) {
      vllm::fused_moe::moe_fused_gate_kernel_dynamic<c10::Half><<<num_blocks, block_dim, 0, stream>>>(
          input.data_ptr(),
          bias.data_ptr(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk);
    } else if (input.scalar_type() == at::kFloat) {
      vllm::fused_moe::moe_fused_gate_kernel_dynamic<float><<<num_blocks, block_dim, 0, stream>>>(
          input.data_ptr(),
          bias.data_ptr(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk);
    } else {
      TORCH_CHECK(false, "Unsupported data type for moe_fused_gate");
    }
  }
}
// DL end

// DL begin: vendored from Denglin vLLM fork csrc/dl/dl_lora.cu (port plan 4b/4d).
#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Descriptors.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../dispatch_utils.h"
#include "dlblas_ext.h"

#define SHFL_MASK 0xffffffff

template <typename T, int L>
struct alignas(L * sizeof(T)) _fvec {
  T data[L];

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      data[i] = T(0);
    }
  }
};

template <typename T>
struct _fvec<T, 1> {
  T data[1];

  __device__ __forceinline__ void zero() {
    data[0] = T(0);
  }
};

template <typename T>
__device__ __forceinline__ T zero() {
  T ret;
  ret.zero();
  return ret;
}

template <typename T, int L>
struct TArray {};

template <int L>
struct TArray<float, L> {
  using type = float;
  using vtype = _fvec<type, L>;

  __device__ __forceinline__ static vtype load(const vtype* ptr, int offset) {
    return ptr[offset];
  }
};

template <int L>
struct TArray<c10::Half, L> {
  using type = __half;
  using vtype = _fvec<type, L>;
  using fvtype = typename TArray<float, L>::vtype;

  __device__ __forceinline__ static
  fvtype load(const vtype* ptr, int offset) {
    vtype data = ptr[offset];
    fvtype ret;
    if constexpr (L == 1) {
      ret.data[0] = __half2float(data.data[0]);
    } else {
      float2* ret_ptr = reinterpret_cast<float2*>(ret.data);
#pragma unroll
      for (int i = 0; i < L / 2; ++i) {
        ret_ptr[i] = __half22float2(reinterpret_cast<half2*>(data.data)[i]);
      }
    }
    return ret;
  }
};

template <int L>
struct TArray<c10::BFloat16, L> {
  using type = __nv_bfloat16;
  using vtype = _fvec<type, L>;
  using fvtype = typename TArray<float, L>::vtype;

  template <int Width = L>
  __device__ __forceinline__ static
  std::enable_if_t<(Width != 8) && (Width != 4) && (Width != 16),
  // std::enable_if_t<true,
                   fvtype> load(const vtype* ptr, int offset) {
    vtype data = ptr[offset];
    fvtype ret;
    if constexpr (L == 1) {
      ret.data[0] = __bfloat162float(data.data[0]);
    } else {
      float2* ret_ptr = reinterpret_cast<float2*>(ret.data);
#pragma unroll
      for (int i = 0; i < L / 2; ++i) {
        ret_ptr[i] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(data.data)[i]);
      }
    }
    return ret;
  }

#if 1
  template <int Width = L>
  __device__ __forceinline__ static
  std::enable_if_t<Width == 4, fvtype> load(const vtype* ptr, int offset) {
    vtype data;
    asm ( "ldg64_l1l2_u64 %0, 0(%1, %2)" : "=l"(data) : "s"(ptr), "r"(offset) : "memory");
    fvtype ret;
    float2* ret_ptr = reinterpret_cast<float2*>(ret.data);
    ret_ptr[0] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(data.data)[0]);
    ret_ptr[1] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(data.data)[1]);
    return ret;
  }

  // !NOTE threads in the same warp must use same ptr
  template <int Width = L>
  __device__ __forceinline__ static
  std::enable_if_t<Width == 8, fvtype> load(const vtype* ptr, int offset) {
    vtype data;
    asm ( "ldg64_l1l2_u128 %0, 0(%1, %2)" : "=l"(data) : "s"(ptr), "r"(offset) : "memory");
    fvtype ret;
    float2* ret_ptr = reinterpret_cast<float2*>(ret.data);
    ret_ptr[0] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(data.data)[0]);
    ret_ptr[1] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(data.data)[1]);
    ret_ptr[2] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(data.data)[2]);
    ret_ptr[3] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(data.data)[3]);
    return ret;
  }

  template <int Width = L>
  __device__ __forceinline__ static
  std::enable_if_t<Width == 16, fvtype> load(const vtype* ptr, int offset) {
    _fvec<type, 8> data[2];
    asm ( "ldg64_l1l2_u128 %0, 0(%1, %2)" : "=l"(data[0]) : "s"(ptr), "r"(2 * offset) : "memory");
    asm ( "ldg64_l1l2_u128 %0, 0(%1, %2)" : "=l"(data[1]) : "s"(ptr), "r"(2 * offset + 1) : "memory");
    fvtype ret;
    float2* ret_ptr = reinterpret_cast<float2*>(ret.data);
    ret_ptr[0] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&data[0])[0]);
    ret_ptr[1] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&data[0])[1]);
    ret_ptr[2] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&data[0])[2]);
    ret_ptr[3] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&data[0])[3]);

    ret_ptr[4] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&data[1])[0]);
    ret_ptr[5] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&data[1])[1]);
    ret_ptr[6] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&data[1])[2]);
    ret_ptr[7] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&data[1])[3]);
    return ret;
  }
#endif
};

template <typename scalar_t, int K_BLOCKS, int WARPS_PER_BLOCK=8>
__global__ void
dl_lora_shrink_gemv_kernel(
    const scalar_t* __restrict__ inputs,  // shape [num_tokens, in_features]
    const int32_t* __restrict__ token_lora_mapping,  // shape [num_tokens]
    const int64_t* __restrict__ lora_ptr,  // shape [num_slices]
    int32_t lora_strides_d0,  // lora_rank * in_features
    int32_t lora_strides_d1,  // in_features
    int32_t lora_strides_d2,  // 1
    float* __restrict__ output,  // shape [num_slices, num_tokens, lora_rank]
    int32_t output_strides_d0,  // num_tokens * lora_rank
    int32_t output_strides_d1,  // lora_rank
    int32_t output_strides_d2,  // 1
    const int32_t* __restrict__ lora_ids,  // shape [max-loras + 1]
    float scaling,
    int32_t num_tokens,  // inputs.size(0)
    int32_t in_features  // inputs.size(1)
) {
  using Array = TArray<scalar_t, K_BLOCKS>;
  using Vec = typename TArray<scalar_t, K_BLOCKS>::vtype;
  using VecF = typename TArray<float, K_BLOCKS>::vtype;

  int32_t token_idx = blockIdx.x;
  int32_t lora_id = token_lora_mapping[token_idx];
  if (lora_id == -1) {
    // Early exit for the no-lora case.
    return;
  }

  int32_t lora_rank_idx = blockIdx.y;
  int32_t slice_idx = blockIdx.z;
  const Vec* in_ptr = (const Vec*)(inputs + token_idx * in_features);
  const Vec* w_ptr = (const Vec*)(reinterpret_cast<scalar_t*>(lora_ptr[slice_idx]) +
                      lora_id * lora_strides_d0 + lora_rank_idx * lora_strides_d1);
  float* out_ptr = output + slice_idx * output_strides_d0 +
                      token_idx * output_strides_d1 + lora_rank_idx * output_strides_d2;

  // Each block handles one output row.
  int32_t tid = threadIdx.x;
  float acc = 0.0f;
  const int32_t step = blockDim.x;
  const int32_t v_in_dim = in_features / K_BLOCKS;

  for (int32_t k = tid; k < v_in_dim; k += step) {
    VecF in = Array::load(in_ptr, k);
    VecF w = Array::load(w_ptr, k);
#pragma unroll
    for (int i = 0; i < K_BLOCKS; ++i) {
      acc += in.data[i] * w.data[i];
    }
  }

#pragma unroll
  for (int st = 1; st < 32; st <<= 1) {
    acc += __shfl_xor_sync(SHFL_MASK, acc, st);
  }

  __shared__ float shared[WARPS_PER_BLOCK];
  if (tid % 32 == 0) {
    shared[tid / 32] = acc;
  }
  __syncthreads();

  if (tid < WARPS_PER_BLOCK) {
    acc = shared[tid];
  }

#pragma unroll
  for (int st = 1; st < WARPS_PER_BLOCK; st <<= 1) {
    acc += __shfl_xor_sync(SHFL_MASK, acc, st);
  }

  if (tid == 0) {
    out_ptr[0] = acc * scaling;
  }
} // dl_lora_shrink_gemv_kernel

template <
    typename in_t,
    typename scalar_t,
    int LORA_RANK,
    int K_BLOCKS,
    bool ADD_INPUTS,
    int WARPS_PER_BLOCK=8>
__global__ void dl_lora_expand_gemv_kernel(
    const in_t* __restrict__ inputs,  // shape [num_slices, num_tokens, lora_rank]
    const int32_t* __restrict__ token_lora_mapping,  // shape [num_tokens]
    const int64_t* __restrict__ slice_start,  // shape [num_slices]
    const int64_t* __restrict__ lora_ptr,  // shape [num_slices]
    const int64_t* __restrict__ lora_strides_d0,  // shape [num_slices]
    const int64_t* __restrict__ lora_strides_d1,  // shape [num_slices]
    const int64_t* __restrict__ lora_strides_d2,  // shape [num_slices]
    const int64_t* __restrict__ hidden_sizes,   // shape [num_slices]
    scalar_t* __restrict__ output,  // shape [num_tokens, out_features]
    int32_t output_strides_d0,  // out_features
    int32_t output_strides_d1,  // 1
    const int32_t* __restrict__ lora_ids,  // shape [max-loras + 1]
    int32_t offset_start,
    bool add_inputs,
    int32_t slice_num,  // inputs.size(0)
    int32_t num_tokens,  // inputs.size(1)
    int32_t lora_rank  // inputs.size(2)
) {
  // constexpr int32_t N_PER_WARP = 32 / (LORA_RANK / K_BLOCKS);
  constexpr int32_t NUM_THREADS_PER_K = LORA_RANK / K_BLOCKS;
  constexpr int32_t N_NUM_PER_WARP = 32 / (LORA_RANK / K_BLOCKS);
  constexpr int32_t N_NUM_PER_BLOCK = N_NUM_PER_WARP * WARPS_PER_BLOCK;
  using InArray = TArray<in_t, K_BLOCKS>;
  using WArray = TArray<scalar_t, K_BLOCKS>;
  using VecIn = typename TArray<in_t, K_BLOCKS>::vtype;
  using VecW = typename TArray<scalar_t, K_BLOCKS>::vtype;
  using VecF = typename TArray<float, K_BLOCKS>::vtype;

  int32_t token_idx = blockIdx.y;
  int32_t lora_id = token_lora_mapping[token_idx];
  if (lora_id == -1) {
    // Early exit for the no-lora case.
    return;
  }

  int32_t n_block_idx = blockIdx.x;
  int32_t warp_id = threadIdx.y;
  // n index in warp
  int32_t n_start_warp = n_block_idx * N_NUM_PER_BLOCK + warp_id * N_NUM_PER_WARP;

  int32_t slice_idx = blockIdx.z;
  int32_t slice_total_hidden = hidden_sizes[slice_idx];

  if (n_start_warp >= slice_total_hidden) {
    // Early exit total warp threads for the out-of-range case.
    return;
  }

  int32_t lane_id = threadIdx.x;
  int32_t nidx_in_warp = lane_id / NUM_THREADS_PER_K;
  int32_t k_start_idx = lane_id % NUM_THREADS_PER_K;

  int32_t n_start_idx = n_start_warp + nidx_in_warp;
  int32_t output_offset = slice_start[slice_idx];  

  // int32_t cur_lora_strides_d1 = lora_strides_d1[slice_idx];
  int32_t cur_lora_strides_d0 = lora_strides_d0[slice_idx];
  const VecIn* in_ptr = (const VecIn*)(inputs + slice_idx * (num_tokens * LORA_RANK) +
                      token_idx * LORA_RANK);
  const VecW* w_ptr = (const VecW*)(reinterpret_cast<scalar_t*>(lora_ptr[slice_idx]) +
                      lora_id * cur_lora_strides_d0);
  scalar_t* out_ptr = output + token_idx * output_strides_d0 + output_offset;

  VecF in_data = InArray::load(in_ptr, k_start_idx);
  VecF w_data;

  float acc = 0.f;

  if (n_start_idx < slice_total_hidden) {
    w_data = WArray::load(w_ptr, n_start_idx * NUM_THREADS_PER_K + k_start_idx);
  }

#pragma unroll
  for (int j = 0; j < K_BLOCKS; ++j) {
    acc += in_data.data[j] * w_data.data[j];
  }

#pragma unroll
  for (int st = NUM_THREADS_PER_K / 2; st > 0 ; st >>= 1) {
    acc += __shfl_xor_sync(SHFL_MASK, acc, st);
  }

  if (k_start_idx == 0 && n_start_idx < slice_total_hidden) {
    if (ADD_INPUTS) {
      out_ptr[n_start_idx] = static_cast<scalar_t>(
            static_cast<float>(out_ptr[n_start_idx]) + acc);
    } else {
      out_ptr[n_start_idx] = static_cast<scalar_t>(acc);
    }
  }
}


inline cudaDataType_t ScalarTypeToCudaDataType(const c10::ScalarType& scalar_type) {
    switch (scalar_type) {
      case c10::ScalarType::Byte:
        return CUDA_R_8U;
      case c10::ScalarType::Char:
        return CUDA_R_8I;
      case c10::ScalarType::Int:
        return CUDA_R_32I;
      case c10::ScalarType::Half:
        return CUDA_R_16F;
      case c10::ScalarType::Float:
        return CUDA_R_32F;
      case c10::ScalarType::Double:
        return CUDA_R_64F;
      case c10::ScalarType::ComplexHalf:
        return CUDA_C_16F;
      case c10::ScalarType::ComplexFloat:
        return CUDA_C_32F;
      case c10::ScalarType::ComplexDouble:
        return CUDA_C_64F;
      case c10::ScalarType::Short:
        return CUDA_R_16I;
      case c10::ScalarType::Long:
        return CUDA_R_64I;
      case c10::ScalarType::BFloat16:
        return CUDA_R_16BF;
      case c10::ScalarType::Float8_e4m3fn:
	return (cudaDataType_t)CUDA_R_8F_E4M3;
      case c10::ScalarType::Float8_e5m2:
        return (cudaDataType_t)CUDA_R_8F_E5M2;
      default:
        TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", scalar_type, " to cudaDataType.")
    }
}

int vllm_lora_max_batch() {
  static auto lora_max_batch = []() -> int { 
    const char* p = std::getenv("VLLM_LORA_MAX_BATCH");
    if (p) {
      int batch = std::stoi(p);
      TORCH_CHECK(batch > 0, "VLLM_LORA_MAX_BATCH must be positive.");
      return batch;
    }
    return 32;
  }();
  return lora_max_batch;
}

#define LUANCH_LORA_SHRINK_KERNEL(K_BLOCKS)                                   \
  do {                                                                        \
    constexpr int WARPS_PER_BLOCK = 8;                                        \
    dim3 grid(num_tokens, lora_rank, slice_num);                              \
    dim3 block(32 * WARPS_PER_BLOCK, 1, 1);                                   \
    dl_lora_shrink_gemv_kernel                                                \
      <scalar_t, K_BLOCKS, WARPS_PER_BLOCK>                                   \
      <<<grid, block, 0, stream>>>(                                           \
        inputs.data_ptr<scalar_t>(),                                          \
        token_lora_mapping.data_ptr<int32_t>(),                               \
        lora_ptr_tensor.data_ptr<int64_t>(),                                  \
        lora_strides_d0,                                                      \
        lora_strides_d1,                                                      \
        lora_strides_d2,                                                      \
        output_tensor.data_ptr<float>(),                                      \
        output_strides_d0,                                                    \
        output_strides_d1,                                                    \
        output_strides_d2,                                                    \
        lora_ids.data_ptr<int32_t>(),                                         \
        (float)scaling,                                                       \
        inputs.size(0),                                                       \
        inputs.size(1)                                                        \
    );                                                                        \
  } while (0)

void dl_lora_shrink(
    const at::Tensor& inputs,  // shape [num_tokens, in_features]
    const at::Tensor& lora_ptr_tensor,  // shape [num_slices]
    const at::Tensor& lora_ptr_cpu_tensor,  // shape [num_slices]
    int64_t lora_strides_d0,  // lora_rank * in_features
    int64_t lora_strides_d1,  // in_features
    int64_t lora_strides_d2,  // 1
    at::Tensor& output_tensor,  // shape [num_slices, num_tokens, lora_rank]
    const at::Tensor& token_lora_mapping,  // shape [num_tokens]
    const at::Tensor& token_indices_sorted_by_lora_ids,  // shape [num_tokens]
    const at::Tensor& num_tokens_per_lora,  // shape [max-loras + 1]
    const at::Tensor& lora_token_start_loc,  // shape [max-loras + 1]
    const at::Tensor& lora_ids,  // shape [max-loras + 1]
    double scaling) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int32_t slice_num = output_tensor.size(0);
  const int32_t lora_rank = output_tensor.size(2);
  const int32_t num_tokens = inputs.size(0);
  const int32_t in_features = inputs.size(1);
  const int32_t total_loras = lora_ids.size(0);
  const int32_t max_loras = lora_ids.size(0) - 1;
  TORCH_CHECK(max_loras >= 0);

  int64_t output_strides_d0 = output_tensor.stride(0);
  int64_t output_strides_d1 = output_tensor.stride(1);
  int64_t output_strides_d2 = output_tensor.stride(2);
  auto is_power_of_2 = [](int64_t n) {
    return (n & (n - 1)) == 0;
  };

  TORCH_CHECK(is_power_of_2(lora_rank) && lora_rank <= 256,
              "Unsupport lora_rank: ", lora_rank);

  if (num_tokens <= vllm_lora_max_batch()) {
    VLLM_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "dl_lora_shrink", ([&] {
      // Launch the kernel.
      int k = in_features;
      if (k % 16 == 0) {
        LUANCH_LORA_SHRINK_KERNEL(16);
      } else if (k % 8 == 0) {
        LUANCH_LORA_SHRINK_KERNEL(8);
      } else if (k % 4 == 0) {
        LUANCH_LORA_SHRINK_KERNEL(4);
      } else if (k % 2 == 0) {
        LUANCH_LORA_SHRINK_KERNEL(2);
      } else {
        LUANCH_LORA_SHRINK_KERNEL(1);
      }
    }));
  } else {
    float f_scaling = (float)scaling;
    float f_beta = 0.f;
    std::vector<int32_t> n_array(slice_num, lora_rank);
    std::vector<const void*> b_array(slice_num, nullptr);
    for (int i = 0; i < slice_num; ++i) {
      b_array[i] = reinterpret_cast<const void*>(lora_ptr_cpu_tensor.data_ptr<int64_t>()[i]);
    }
    TORCH_CUDABLAS_CHECK(
      dlblasLoraGemm(at::cuda::getCurrentCUDABlasHandle(),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          num_tokens,
          n_array.data(),
          in_features,
          &f_scaling,
          inputs.data_ptr(),
          ScalarTypeToCudaDataType(inputs.scalar_type()),
          b_array.data(),
          ScalarTypeToCudaDataType(inputs.scalar_type()),
          &f_beta,
          output_tensor.data_ptr(),
          ScalarTypeToCudaDataType(output_tensor.scalar_type()),
          CUDA_R_32F,
          lora_ids.data_ptr<int32_t>(),
          slice_num,
          max_loras,
          kLoraShrink));
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

#define LUANCH_LORA_EXPAND_KERNEL(LORA_RANK, K_BLOCKS, ADD_INPUTS)            \
  do {                                                                        \
    constexpr int WARPS_PER_BLOCK = 8;                                        \
    constexpr int N_PER_WARP = 32 / (LORA_RANK / K_BLOCKS);                   \
    constexpr int N_PER_BLOCK = WARPS_PER_BLOCK * N_PER_WARP;                 \
    dim3 grid((max_n + N_PER_BLOCK - 1) / (N_PER_BLOCK),                      \
              num_tokens, slice_num);                                         \
    dim3 block(32, WARPS_PER_BLOCK, 1);                                       \
    if (inputs.scalar_type() == at::kFloat) {                                 \
      dl_lora_expand_gemv_kernel                                              \
        <float, scalar_t, LORA_RANK, K_BLOCKS,                                \
         ADD_INPUTS, WARPS_PER_BLOCK>                                         \
        <<<grid, block, 0, stream>>>(                                         \
          inputs.data_ptr<float>(),                                           \
          token_lora_mapping.data_ptr<int32_t>(),                             \
          slice_start_tensor.data_ptr<int64_t>(),                             \
          lora_ptr_tensor.data_ptr<int64_t>(),                                \
          lora_strides_d0_tensor.data_ptr<int64_t>(),                         \
          lora_strides_d1_tensor.data_ptr<int64_t>(),                         \
          lora_strides_d2_tensor.data_ptr<int64_t>(),                         \
          hidden_sizes_tensor.data_ptr<int64_t>(),                            \
          output_tensor.data_ptr<scalar_t>(),                                 \
          output_strides_d0,                                                  \
          output_strides_d1,                                                  \
          lora_ids.data_ptr<int32_t>(),                                       \
          offset_start,                                                       \
          add_inputs,                                                         \
          slice_num,                                                          \
          num_tokens,                                                         \
          lora_rank                                                           \
      );                                                                      \
    } else {                                                                  \
      TORCH_CHECK(                                                            \
          inputs.scalar_type() == output_tensor.scalar_type(),                \
          "Unsupported data type for LoRA expand: ",                          \
          inputs.scalar_type());                                              \
      dl_lora_expand_gemv_kernel                                              \
        <scalar_t, scalar_t, LORA_RANK, K_BLOCKS,                             \
         ADD_INPUTS, WARPS_PER_BLOCK>                                         \
        <<<grid, block, 0, stream>>>(                                         \
          inputs.data_ptr<scalar_t>(),                                        \
          token_lora_mapping.data_ptr<int32_t>(),                             \
          slice_start_tensor.data_ptr<int64_t>(),                             \
          lora_ptr_tensor.data_ptr<int64_t>(),                                \
          lora_strides_d0_tensor.data_ptr<int64_t>(),                         \
          lora_strides_d1_tensor.data_ptr<int64_t>(),                         \
          lora_strides_d2_tensor.data_ptr<int64_t>(),                         \
          hidden_sizes_tensor.data_ptr<int64_t>(),                            \
          output_tensor.data_ptr<scalar_t>(),                                 \
          output_strides_d0,                                                  \
          output_strides_d1,                                                  \
          lora_ids.data_ptr<int32_t>(),                                       \
          offset_start,                                                       \
          add_inputs,                                                         \
          slice_num,                                                          \
          num_tokens,                                                         \
          lora_rank                                                           \
      );                                                                      \
    }                                                                         \
  } while (0)

void dl_lora_expand(
    const at::Tensor& inputs,  // shape [num_slices, num_tokens, lora_rank]
    const at::Tensor& lora_ptr_tensor,  // shape [num_slices]
    const at::Tensor& lora_ptr_cpu_tensor,  // shape [num_slices]
    const at::Tensor& lora_strides_d0_tensor,  // shape [num_slices]
    const at::Tensor& lora_strides_d1_tensor,  // shape [num_slices]
    const at::Tensor& lora_strides_d2_tensor,  // shape [num_slices]
    at::Tensor& output_tensor,  // shape [num_tokens, out_features]
    const at::Tensor& slice_start_tensor,  // shape [num_slices]
    const at::Tensor& hidden_sizes_tensor,   // shape [num_slices]
    const at::Tensor& hidden_sizes_cpu_tensor,   // shape [num_slices]
    int64_t max_n,
    const at::Tensor& token_lora_mapping,  // shape [num_tokens]
    const at::Tensor& token_indices_sorted_by_lora_ids,  // shape [num_tokens]
    const at::Tensor& num_tokens_per_lora,  // shape [max-loras + 1]
    const at::Tensor& lora_token_start_loc,  // shape [max-loras + 1]
    int64_t offset_start,
    const at::Tensor& lora_ids,  // shape [max-loras + 1]
    bool add_inputs = false) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int32_t slice_num = inputs.size(0);
  const int32_t num_tokens = inputs.size(1);
  const int32_t lora_rank = inputs.size(2);
  const int32_t out_features = output_tensor.size(1);
  const int32_t total_loras = lora_ids.size(0);
  const int32_t max_loras = lora_ids.size(0) - 1;

  int64_t output_strides_d0 = output_tensor.stride(0);
  int64_t output_strides_d1 = output_tensor.stride(1);

  auto is_power_of_2 = [](int64_t n) {
    return (n & (n - 1)) == 0;
  };

  if (num_tokens <= vllm_lora_max_batch()) {
    VLLM_DISPATCH_FLOATING_TYPES(output_tensor.scalar_type(), "dl_lora_expand", ([&] {
      if (is_power_of_2(lora_rank) && lora_rank <= 256) {
        if (add_inputs) {
          if (lora_rank == 1) {
            LUANCH_LORA_EXPAND_KERNEL(1, 1, true);
          } else if (lora_rank == 2) {
            LUANCH_LORA_EXPAND_KERNEL(2, 2, true);
          } else if (lora_rank == 4) {
            LUANCH_LORA_EXPAND_KERNEL(4, 4, true);
          } else if (lora_rank == 8) {
            LUANCH_LORA_EXPAND_KERNEL(8, 8, true);
          } else if (lora_rank == 16) {
            LUANCH_LORA_EXPAND_KERNEL(16, 16, true);
          } else if (lora_rank == 32) {
            LUANCH_LORA_EXPAND_KERNEL(32, 16, true);
          } else if (lora_rank == 64) {
            LUANCH_LORA_EXPAND_KERNEL(64, 16, true);
          } else if (lora_rank == 128){
            LUANCH_LORA_EXPAND_KERNEL(128, 16, true);
          } else if (lora_rank == 256) {
            LUANCH_LORA_EXPAND_KERNEL(256, 16, true);
          }
        } else {
          if (lora_rank == 1) {
            LUANCH_LORA_EXPAND_KERNEL(1, 1, false);
          } else if (lora_rank == 2) {
            LUANCH_LORA_EXPAND_KERNEL(2, 2, false);
          } else if (lora_rank == 4) {
            LUANCH_LORA_EXPAND_KERNEL(4, 4, false);
          } else if (lora_rank == 8) {
            LUANCH_LORA_EXPAND_KERNEL(8, 8, false);
          } else if (lora_rank == 16) {
            LUANCH_LORA_EXPAND_KERNEL(16, 16, false);
          } else if (lora_rank == 32) {
            LUANCH_LORA_EXPAND_KERNEL(32, 16, false);
          } else if (lora_rank == 64) {
            LUANCH_LORA_EXPAND_KERNEL(64, 16, false);
          } else if (lora_rank == 128){
            LUANCH_LORA_EXPAND_KERNEL(128, 16, false);
          } else if (lora_rank == 256) {
            LUANCH_LORA_EXPAND_KERNEL(256, 16, false);
          }
        }
      } else {
        // Dynamic kernel launch.
        TORCH_CHECK(false, "Unsupport lora_rank: ", lora_rank);
      }
    }));
  } else {
    float f_scaling = 1.f;
    float f_beta = 1.f;
    std::vector<int> n_array(slice_num, 0);
    std::vector<const void*> b_array(slice_num, nullptr);
    for (int i = 0; i < slice_num; ++i) {
      n_array[i] = hidden_sizes_cpu_tensor.data_ptr<int64_t>()[i];
      b_array[i] = reinterpret_cast<const void*>(lora_ptr_cpu_tensor.data_ptr<int64_t>()[i]);
    }
    TORCH_CUDABLAS_CHECK(
      dlblasLoraGemm(at::cuda::getCurrentCUDABlasHandle(),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          num_tokens,
          n_array.data(),
          lora_rank,
          &f_scaling,
          inputs.data_ptr(),
          ScalarTypeToCudaDataType(inputs.scalar_type()),
          b_array.data(),
          ScalarTypeToCudaDataType(output_tensor.scalar_type()),
          &f_beta,
          output_tensor.data_ptr(),
          ScalarTypeToCudaDataType(output_tensor.scalar_type()),
          CUDA_R_32F,
          lora_ids.data_ptr<int32_t>(),
          slice_num,
          max_loras,
          kLoraExpand));
  }
  AT_CUDA_CHECK(cudaGetLastError());
}// DL end

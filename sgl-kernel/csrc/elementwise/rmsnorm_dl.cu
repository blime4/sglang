// DL begin
//
// Standalone RMSNorm kernels for DLIN — built by setup_dl.py via dlcc.
// No FlashInfer / CUTLASS dependency: the upstream sgl_kernel rmsnorm kernels
// include <flashinfer/norm.cuh>, which dlcc cannot build against yet, so this
// file provides an equivalent standard RMSNorm + fused(residual_add)_rmsnorm
// implemented with a plain warp/block reduction.
//
// Registered as the SAME torch ops the python wrappers in
// sgl-kernel/python/sgl_kernel/elementwise.py already call:
//   torch.ops.sgl_kernel.rmsnorm(out, input, weight, eps, enable_pdl)
//   torch.ops.sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)
// (enable_pdl is ignored on DLIN — PDL/cluster launch is disabled via SGL_ON_DLIN.)
//
// Math matches RMSNorm.forward_native (layernorm.py): accumulate in fp32,
//   out = x * rsqrt(mean(x^2) + eps) * weight
//   fused_add: residual = x + residual (stored back in orig dtype);
//              input    = rmsnorm(residual)
//
// DL end

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include "utils.h"

namespace sgl_kernel_dl {

// dlcc (clang-15) rejects static_cast<float>(__half/__nv_bfloat16); use the
// explicit fp16/bf16 conversion intrinsics instead.
template <typename T>
struct FloatCvt;
template <>
struct FloatCvt<__half> {
  static __device__ float to(__half x) { return __half2float(x); }
  static __device__ __half from(float x) { return __float2half_rn(x); }
};
template <>
struct FloatCvt<__nv_bfloat16> {
  static __device__ float to(__nv_bfloat16 x) { return __bfloat162float(x); }
  static __device__ __nv_bfloat16 from(float x) { return __float2bfloat16_rn(x); }
};

// Warp-level sum reduction via shuffle.
__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

// Block-level sum reduction. Assumes blockDim.x is a multiple of 32 and <= 1024.
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  val = warp_reduce_sum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  const int num_warps = blockDim.x >> 5;
  if (wid == 0) {
    val = (lane < num_warps) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[0] = val;
  }
  __syncthreads();
  return shared[0];
}

// One block per row. blockDim.x must be a multiple of 32 (launch uses 1024).
template <typename T>
__global__ void rmsnorm_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const float eps,
    const int hidden_size) {
  const int row = blockIdx.x;
  const T* row_in = input + static_cast<int64_t>(row) * hidden_size;
  T* row_out = out + static_cast<int64_t>(row) * hidden_size;

  __shared__ float smem[32];

  float local_sq = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float v = FloatCvt<T>::to(row_in[i]);
    local_sq += v * v;
  }
  const float sum_sq = block_reduce_sum(local_sq, smem);
  const float rms_inv = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float v = FloatCvt<T>::to(row_in[i]);
    row_out[i] = FloatCvt<T>::from(v * rms_inv * FloatCvt<T>::to(weight[i]));
  }
}

// One block per row. residual := input + residual (fp32), input := rmsnorm(residual).
template <typename T>
__global__ void fused_add_rmsnorm_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    const float eps,
    const int hidden_size) {
  const int row = blockIdx.x;
  T* row_in = input + static_cast<int64_t>(row) * hidden_size;
  T* row_res = residual + static_cast<int64_t>(row) * hidden_size;

  __shared__ float smem[32];

  // Pass 1: residual = input + residual; accumulate sum of squares.
  float local_sq = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float v = FloatCvt<T>::to(row_in[i]) + FloatCvt<T>::to(row_res[i]);
    row_res[i] = FloatCvt<T>::from(v);
    local_sq += v * v;
  }
  const float sum_sq = block_reduce_sum(local_sq, smem);
  const float rms_inv = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);

  // Pass 2: input = rmsnorm(residual). residual now holds the fp32 sum (as orig dtype).
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float v = FloatCvt<T>::to(row_res[i]);
    row_in[i] = FloatCvt<T>::from(v * rms_inv * FloatCvt<T>::to(weight[i]));
  }
}

// Block size: a multiple of 32. 1024 threads covers hidden sizes up to ~large
// via grid-stride; smaller hidden sizes just leave threads idle.
constexpr int kRMSBlock = 1024;

// Typed launch wrappers (avoid macros: dlcc + <<<a,b,c>>> commas don't play
// well with variadic macro args).
template <typename T>
void launch_rmsnorm(torch::Tensor& out, const torch::Tensor& input,
                    const torch::Tensor& weight, float eps, int hidden_size,
                    cudaStream_t stream) {
  rmsnorm_kernel<T><<<static_cast<int>(input.size(0)), kRMSBlock, 0, stream>>>(
      reinterpret_cast<T*>(out.data_ptr()),
      reinterpret_cast<const T*>(input.data_ptr()),
      reinterpret_cast<const T*>(weight.data_ptr()), eps, hidden_size);
}

template <typename T>
void launch_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual,
                              const torch::Tensor& weight, float eps,
                              int hidden_size, cudaStream_t stream) {
  fused_add_rmsnorm_kernel<T>
      <<<static_cast<int>(input.size(0)), kRMSBlock, 0, stream>>>(
          reinterpret_cast<T*>(input.data_ptr()),
          reinterpret_cast<T*>(residual.data_ptr()),
          reinterpret_cast<const T*>(weight.data_ptr()), eps, hidden_size);
}

void rmsnorm(
    torch::Tensor out,
    torch::Tensor input,
    torch::Tensor weight,
    double eps,
    bool /*enable_pdl*/) {
  CHECK_INPUT(input);
  CHECK_INPUT(out);
  CHECK_INPUT(weight);
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(weight.dim(), 1);
  CHECK_EQ(input.size(1), weight.size(0));
  CHECK_EQ(out.sizes(), input.sizes());
  const auto device = input.device();
  CHECK_EQ(out.device(), device);
  CHECK_EQ(weight.device(), device);

  const int batch_size = static_cast<int>(input.size(0));
  const int hidden_size = static_cast<int>(input.size(1));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::CUDAGuard guard(device);
  (void)batch_size;

  if (input.scalar_type() == at::ScalarType::Half) {
    launch_rmsnorm<__half>(out, input, weight, static_cast<float>(eps),
                           hidden_size, stream);
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    launch_rmsnorm<__nv_bfloat16>(out, input, weight, static_cast<float>(eps),
                                  hidden_size, stream);
  } else {
    TORCH_CHECK(false, "rmsnorm (DLIN) supports fp16/bf16 only, got ",
                input.scalar_type());
  }
}

void fused_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps,
    bool /*enable_pdl*/) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(residual.dim(), 2);
  CHECK_EQ(weight.dim(), 1);
  CHECK_EQ(input.size(0), residual.size(0));
  CHECK_EQ(input.size(1), residual.size(1));
  CHECK_EQ(input.size(1), weight.size(0));
  const auto device = input.device();
  CHECK_EQ(residual.device(), device);
  CHECK_EQ(weight.device(), device);

  const int batch_size = static_cast<int>(input.size(0));
  const int hidden_size = static_cast<int>(input.size(1));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::CUDAGuard guard(device);
  (void)batch_size;

  if (input.scalar_type() == at::ScalarType::Half) {
    launch_fused_add_rmsnorm<__half>(input, residual, weight,
                                     static_cast<float>(eps), hidden_size, stream);
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    launch_fused_add_rmsnorm<__nv_bfloat16>(input, residual, weight,
                                            static_cast<float>(eps), hidden_size,
                                            stream);
  } else {
    TORCH_CHECK(false, "fused_add_rmsnorm (DLIN) supports fp16/bf16 only, got ",
                input.scalar_type());
  }
}

}  // namespace sgl_kernel_dl
// DL end

// DL: M=1-optimized FP8 GEMV kernel for decode projections.
// Memory-bound: one warp per output N, coalesced weight-row read + dot product.
// Targets ~2µs/GEMV vs dlblas GEMM's ~600µs (M=1 tile waste).
#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/type.cuh>   // For DTypeTrait, fp8_e4m3_t, bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.h>    // For CHECK_HOST, div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, kWarpThreads
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::reduce_sum

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

// Each warp (32 threads) computes ONE output element N.
// Lane l handles K elements at stride 32: x[l], x[l+32], ... x[l+31*32].
// Partial dot: sum_l( x[l + 32*iter] * dequant(weight[N, l + 32*iter]) ).
// Warp reduce_sum → final dot(N). Multiply by scale[N].
template <int kKAlign, bool kUsePDL>
__global__ void fp8_gemv_kernel(
    bf16_t* __restrict__ out,         // [N]
    const bf16_t* __restrict__ x,     // [K] (M=1 input)
    const fp8_e4m3_t* __restrict__ w, // [N, K] (row-major, per-channel FP8)
    const fp32_t* __restrict__ scale, // [N] per-channel scale
    uint32_t N, uint32_t K) {
  const uint32_t n = blockIdx.x;     // one block per output element
  const uint32_t lane = threadIdx.x;  // 0..31 (one warp per block)
  const uint32_t warp_size = device::kWarpThreads;  // 32

  // Each thread accumulates a partial dot product over its K-slice
  fp32_t partial = 0.0f;
  const fp8_e4m3_t* w_row = w + (size_t)n * K;  // weight row for output N

  // Vectorized loop: process kVecN elements per iteration
  // bf16: 8 elems × 2B = 16B (128-bit). fp8: 8 elems × 1B = 8B. Both fit.
  constexpr int kVecN = 8;
  using w_vec_t = device::AlignedVector<fp8_e4m3_t, kVecN>;
  using x_vec_t = device::AlignedVector<bf16_t, kVecN>;

  const uint32_t k_vecs = K / kVecN;
  const uint32_t vec_stride = warp_size;

  for (uint32_t vi = lane; vi < k_vecs; vi += vec_stride) {
    w_vec_t wv;
    x_vec_t xv;
    wv.load(w_row, vi);
    xv.load(x, vi);
    #pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      fp32_t w_val = static_cast<fp32_t>(wv[i]);  // fp8 → fp32 dequant
      fp32_t x_val = static_cast<fp32_t>(xv[i]);   // bf16 → fp32
      partial += w_val * x_val;
    }
  }

  // Scalar tail for remaining K elements
  const uint32_t tail_base = k_vecs * kVecN;
  for (uint32_t ki = tail_base + lane; ki < K; ki += warp_size) {
    fp32_t w_val = static_cast<fp32_t>(w_row[ki]);
    fp32_t x_val = static_cast<fp32_t>(x[ki]);
    partial += w_val * x_val;
  }

  // Warp reduction: sum all 32 partials
  fp32_t dot = device::warp::reduce_sum<32>(partial);

  // Lane 0 writes the scaled result
  if (lane == 0) {
    out[n] = static_cast<bf16_t>(dot * scale[n]);
  }
}

template <bool kUsePDL>
void fp8_gemv(tvm::ffi::TensorView out, tvm::ffi::TensorView x,
              tvm::ffi::TensorView w, tvm::ffi::TensorView scale) {
  using namespace host;

  auto N = SymbolicSize{"N"};
  auto K = SymbolicSize{"K"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  // out: [N] bf16, x: [K] bf16, w: [N, K] fp8, scale: [N] fp32
  TensorMatcher({N}).with_dtype<bf16_t>().with_device<kDLCUDA>(device_).verify(out);
  TensorMatcher({K}).with_dtype<bf16_t>().with_device<kDLCUDA>(device_).verify(x);
  TensorMatcher({N, K}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device_).verify(w);
  TensorMatcher({N}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(scale);

  const uint32_t n = static_cast<uint32_t>(N.unwrap());
  const uint32_t k = static_cast<uint32_t>(K.unwrap());
  const DLDevice device = device_.unwrap();

  CHECK_HOST(n > 0 && k > 0) << "fp8_gemv: N=" << n << " K=" << k;

  constexpr uint32_t kBlockSize = 32;  // one warp per block (one output element)
  LaunchKernel(n, kBlockSize, device)(fp8_gemv_kernel<16, kUsePDL>,
      static_cast<bf16_t*>(out.data_ptr()),
      static_cast<const bf16_t*>(x.data_ptr()),
      static_cast<const fp8_e4m3_t*>(w.data_ptr()),
      static_cast<const fp32_t*>(scale.data_ptr()),
      n, k);
}

}  // namespace

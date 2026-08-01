// DL begin — V4 FP4 (E2M1 + e8m0, block=32) grouped GEMV for M=1 decode MoE.
// Memory-bound: one warp per output (slot, n). Each lane reads 16 bytes (32 FP4
// = exactly one e8m0 block) per iteration, fully coalesced across the warp.
// Replaces cuDNN invoke_fused_moe_opt which runs ~40x off the memory floor
// (~23 GB/s eff. vs ~1 TB/s peak) at TP8 per-rank decode shapes.
//
// Computes, for each routed expert slot s (topk_ids[s] -> expert e):
//   out[s, n] = sum_k x[k] * dequant_fp4(w[e, n, k]) * e8m0_scale(e, n, k/32)
// where w is int8-packed FP4 (low nibble = even k), scale is uint8 e8m0.
#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/type.cuh>   // For DTypeTrait, bf16_t, fp32_t
#include <sgl_kernel/utils.h>    // For CHECK_HOST, div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, kWarpThreads
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::reduce_sum

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

// E2M1 nibble (0..15) -> fp32 value. Matches DSV4_DEQUANT_FP4_TABLE (fp8.py:152).
// nibble & 0x7 selects magnitude {0,.5,1,1.5,2,3,4,6}; bit 3 is sign.
// NOTE: defined as a kernel-local const array (registers). A file-scope
// __constant__ array is NOT initialized in tvm-ffi inline-JIT modules on dlcc.

template <bool kUsePDL>
__global__ void fp4_grouped_gemv_kernel(
    bf16_t* __restrict__ out,            // [S, N]
    const bf16_t* __restrict__ x,        // [K]
    const uint8_t* __restrict__ w,       // [E, N, K/2]  int8 viewed as uint8 (FP4 packed)
    const uint8_t* __restrict__ sc,      // [E, N, K/32] e8m0 raw bytes
    const int32_t* __restrict__ topk_ids,  // [S]
    uint32_t N, uint32_t K, uint32_t num_outputs) {
  // E2M1 nibble dequant table (local -> registers, avoids __constant__ init issue).
  const float kFp4Table[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  // Multi-warp block: 8 warps/block (256 threads), one output per warp. Grid is
  // ceil(S*N / 8). This lifts occupancy vs 1-warp blocks so the bandwidth-starved
  // KS38 (~65 GB/s) keeps many weight reads in flight.
  constexpr uint32_t kWarpsPerBlock = 8;
  const uint32_t out_idx = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  if (out_idx >= num_outputs) return;
  const uint32_t slot = out_idx / N;
  const uint32_t n = out_idx % N;
  const uint32_t lane = threadIdx.x & 31u;

  const int32_t expert = topk_ids[slot];
  const size_t w_off = ((size_t)expert * N + n) * (K / 2);
  const size_t sc_off = ((size_t)expert * N + n) * (K / 32);
  const uint8_t* w_row = w + w_off;
  const uint8_t* sc_row = sc + sc_off;

  // Vectorized: each lane loads 16 weight bytes (uint4 = 128-bit) per iteration =
  // 32 FP4 elements = exactly one e8m0 scale block. 32 lanes x 16B = 512 contiguous
  // bytes per warp-iteration (coalesced). K/2 = 2048 bytes -> 4 iterations.
  // uint4 is used directly (AlignedVector<uint8_t> misbehaves on dlcc).
  constexpr uint32_t kBytesPerLane = 16;
  constexpr uint32_t kLaneStrideBytes = device::kWarpThreads * kBytesPerLane;  // 512
  const uint32_t n_iters = (K / 2) / kLaneStrideBytes;                          // 4

  // 8 independent accumulators to break the serial `partial +=` dependency chain
  // (kernel was ~10x slower than the read-only floor -> compute/ILP-bound, not memory).
  fp32_t p[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (uint32_t it = 0; it < n_iters; ++it) {
    const uint32_t byte_base = it * kLaneStrideBytes + lane * kBytesPerLane;
    const uint4 wp = *reinterpret_cast<const uint4*>(w_row + byte_base);
    const uint32_t words[4] = {wp.x, wp.y, wp.z, wp.w};
    const uint32_t ebase = byte_base * 2;                 // first element index
    const float scv = exp2f(static_cast<float>(sc_row[ebase / 32]) - 127.0f);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {      // each uint32 word = 4 bytes = 8 nibbles = 8 elements
      const uint32_t v = words[j];
      const uint32_t e0 = ebase + j * 8;
      p[0] += static_cast<float>(x[e0 + 0]) * (kFp4Table[(v >> 0) & 0xFu] * scv);
      p[1] += static_cast<float>(x[e0 + 1]) * (kFp4Table[(v >> 4) & 0xFu] * scv);
      p[2] += static_cast<float>(x[e0 + 2]) * (kFp4Table[(v >> 8) & 0xFu] * scv);
      p[3] += static_cast<float>(x[e0 + 3]) * (kFp4Table[(v >> 12) & 0xFu] * scv);
      p[4] += static_cast<float>(x[e0 + 4]) * (kFp4Table[(v >> 16) & 0xFu] * scv);
      p[5] += static_cast<float>(x[e0 + 5]) * (kFp4Table[(v >> 20) & 0xFu] * scv);
      p[6] += static_cast<float>(x[e0 + 6]) * (kFp4Table[(v >> 24) & 0xFu] * scv);
      p[7] += static_cast<float>(x[e0 + 7]) * (kFp4Table[(v >> 28) & 0xFu] * scv);
    }
  }
  fp32_t partial = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));

  // Warp reduction -> one dot per (slot, n)
  fp32_t dot = device::warp::reduce_sum<32>(partial);
  if (lane == 0) {
    out[slot * N + n] = static_cast<bf16_t>(dot);
  }
}

template <bool kUsePDL>
void fp4_grouped_gemv(tvm::ffi::TensorView out, tvm::ffi::TensorView x,
                      tvm::ffi::TensorView w, tvm::ffi::TensorView sc,
                      tvm::ffi::TensorView topk_ids) {
  using namespace host;

  auto S = SymbolicSize{"num_slots"};
  auto N = SymbolicSize{"N"};
  auto K = SymbolicSize{"K"};
  auto E = SymbolicSize{"E"};
  auto Kp = SymbolicSize{"K_packed"};  // K/2
  auto Ksc = SymbolicSize{"K_sc"};     // K/32
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  // out [S,N] bf16, x [K] bf16, w [E,N,K/2] uint8(FP4), sc [E,N,K/32] uint8(e8m0),
  // topk_ids [S] int32
  TensorMatcher({S, N}).with_dtype<bf16_t>().with_device<kDLCUDA>(device_).verify(out);
  TensorMatcher({K}).with_dtype<bf16_t>().with_device<kDLCUDA>(device_).verify(x);
  TensorMatcher({E, N, Kp}).with_dtype<uint8_t>().with_device<kDLCUDA>(device_).verify(w);
  TensorMatcher({E, N, Ksc}).with_dtype<uint8_t>().with_device<kDLCUDA>(device_).verify(sc);
  TensorMatcher({S}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk_ids);

  const uint32_t s = static_cast<uint32_t>(S.unwrap());
  const uint32_t n = static_cast<uint32_t>(N.unwrap());
  const uint32_t k = static_cast<uint32_t>(K.unwrap());
  const uint32_t kp = static_cast<uint32_t>(Kp.unwrap());
  const uint32_t ksc = static_cast<uint32_t>(Ksc.unwrap());
  const DLDevice device = device_.unwrap();

  CHECK_HOST(k % 32 == 0) << "fp4_grouped_gemv: K must be a multiple of 32 (block size), got K=" << k;
  CHECK_HOST(2 * kp == k) << "fp4_grouped_gemv: w K_packed must equal K/2";
  CHECK_HOST(32 * ksc == k) << "fp4_grouped_gemv: sc K_sc must equal K/32";

  constexpr uint32_t kWarpsPerBlock = 8;
  constexpr uint32_t kBlockSize = kWarpsPerBlock * 32;  // 256 threads = 8 warps
  const uint32_t num_outputs = s * n;
  const uint32_t grid = (num_outputs + kWarpsPerBlock - 1) / kWarpsPerBlock;
  LaunchKernel(grid, kBlockSize, device)(fp4_grouped_gemv_kernel<kUsePDL>,
      static_cast<bf16_t*>(out.data_ptr()),
      static_cast<const bf16_t*>(x.data_ptr()),
      static_cast<const uint8_t*>(w.data_ptr()),
      static_cast<const uint8_t*>(sc.data_ptr()),
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      n, k, num_outputs);
}

}  // namespace
// DL end

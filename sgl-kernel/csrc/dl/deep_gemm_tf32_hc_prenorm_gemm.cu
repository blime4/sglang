// DL begin: vendored from Denglin vLLM fork csrc/dl/deep_gemm_tf32_hc_prenorm_gemm.cu (port plan 4b/4d).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kBlockM = 64;
constexpr int kFastPathBlockM = 8;
constexpr int kBlockK = 64;
constexpr int kNumThreads = 256;
constexpr int kFastPathNumThreads = 128;

inline int align_to(const int value, const int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

__device__ __forceinline__ float tf32_round(const float value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t rounded;
  asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(rounded) : "f"(value));
  return __uint_as_float(rounded);
#else
  return value;
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void tf32_hc_prenorm_gemm_kernel(
    const int m, const int n, const int k, const int block_n,
    const int num_splits, const int has_split_output,
    const __nv_bfloat16 *__restrict__ a, const float *__restrict__ b,
    float *__restrict__ d, float *__restrict__ sqr_sum,
    const int64_t a_stride_m, const int64_t a_stride_k,
    const int64_t b_stride_n, const int64_t b_stride_k,
    const int64_t d_stride_split, const int64_t d_stride_m,
    const int64_t d_stride_n, const int64_t sqr_sum_stride_split,
    const int64_t sqr_sum_stride_m) {
  const int block_idx = static_cast<int>(blockIdx.x);
  const bool use_fast_path =
      has_split_output == 1 && n == 24 && block_n == 32 && a_stride_k == 1 &&
      b_stride_k == 1 && d_stride_n == 1 && sqr_sum_stride_m == 1;
  const int row_tile = use_fast_path ? kFastPathBlockM : kBlockM;
  const int m_block_idx = block_idx / num_splits;
  const int split_idx = block_idx % num_splits;
  const int m_begin = m_block_idx * row_tile;

  if (m_begin >= m || split_idx >= num_splits) {
    return;
  }

  const int num_k_blocks = k / kBlockK;
  const int blocks_per_split = num_k_blocks / num_splits;
  const int remain_blocks = num_k_blocks % num_splits;
  const int start_block =
      split_idx * blocks_per_split + min(split_idx, remain_blocks);
  const int block_count =
      blocks_per_split + (split_idx < remain_blocks ? 1 : 0);
  const int k_begin = start_block * kBlockK;
  const int k_end = (start_block + block_count) * kBlockK;

  if (use_fast_path) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_idx = static_cast<int>(threadIdx.x) >> 5;
    const int active_rows =
        (m - m_begin < kFastPathBlockM ? m - m_begin : kFastPathBlockM);
    const int warps_per_block = static_cast<int>(blockDim.x) / 32;
    constexpr int kNGroupSize = 8;
    constexpr int kNumNGroups = 24 / kNGroupSize;

    for (int local_m = warp_idx; local_m < active_rows;
         local_m += warps_per_block) {
      const int m_idx = m_begin + local_m;
      const int64_t a_base = m_idx * a_stride_m;

      for (int n_group = 0; n_group < kNumNGroups; ++n_group) {
        const int n_base = n_group * kNGroupSize;

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        float acc4 = 0.0f;
        float acc5 = 0.0f;
        float acc6 = 0.0f;
        float acc7 = 0.0f;
        float sqr_partial = 0.0f;

        const int64_t b_base0 = (n_base + 0) * b_stride_n;
        const int64_t b_base1 = (n_base + 1) * b_stride_n;
        const int64_t b_base2 = (n_base + 2) * b_stride_n;
        const int64_t b_base3 = (n_base + 3) * b_stride_n;
        const int64_t b_base4 = (n_base + 4) * b_stride_n;
        const int64_t b_base5 = (n_base + 5) * b_stride_n;
        const int64_t b_base6 = (n_base + 6) * b_stride_n;
        const int64_t b_base7 = (n_base + 7) * b_stride_n;
        const __nv_bfloat16 *a_ptr = a + a_base + k_begin + lane;
        const float *b_ptr0 = b + b_base0 + k_begin + lane;
        const float *b_ptr1 = b + b_base1 + k_begin + lane;
        const float *b_ptr2 = b + b_base2 + k_begin + lane;
        const float *b_ptr3 = b + b_base3 + k_begin + lane;
        const float *b_ptr4 = b + b_base4 + k_begin + lane;
        const float *b_ptr5 = b + b_base5 + k_begin + lane;
        const float *b_ptr6 = b + b_base6 + k_begin + lane;
        const float *b_ptr7 = b + b_base7 + k_begin + lane;

        if (n_group == 0) {
          for (int k_idx = k_begin + lane; k_idx < k_end; k_idx += 64) {
            const float a_value = __bfloat162float(*a_ptr);
            const float b0 = tf32_round(*b_ptr0);
            const float b1 = tf32_round(*b_ptr1);
            const float b2 = tf32_round(*b_ptr2);
            const float b3 = tf32_round(*b_ptr3);
            const float b4 = tf32_round(*b_ptr4);
            const float b5 = tf32_round(*b_ptr5);
            const float b6 = tf32_round(*b_ptr6);
            const float b7 = tf32_round(*b_ptr7);
            acc0 = fmaf(a_value, b0, acc0);
            acc1 = fmaf(a_value, b1, acc1);
            acc2 = fmaf(a_value, b2, acc2);
            acc3 = fmaf(a_value, b3, acc3);
            acc4 = fmaf(a_value, b4, acc4);
            acc5 = fmaf(a_value, b5, acc5);
            acc6 = fmaf(a_value, b6, acc6);
            acc7 = fmaf(a_value, b7, acc7);
            sqr_partial = fmaf(a_value, a_value, sqr_partial);
            const float a_value_next = __bfloat162float(a_ptr[32]);
            const float b0_next = tf32_round(b_ptr0[32]);
            const float b1_next = tf32_round(b_ptr1[32]);
            const float b2_next = tf32_round(b_ptr2[32]);
            const float b3_next = tf32_round(b_ptr3[32]);
            const float b4_next = tf32_round(b_ptr4[32]);
            const float b5_next = tf32_round(b_ptr5[32]);
            const float b6_next = tf32_round(b_ptr6[32]);
            const float b7_next = tf32_round(b_ptr7[32]);
            acc0 = fmaf(a_value_next, b0_next, acc0);
            acc1 = fmaf(a_value_next, b1_next, acc1);
            acc2 = fmaf(a_value_next, b2_next, acc2);
            acc3 = fmaf(a_value_next, b3_next, acc3);
            acc4 = fmaf(a_value_next, b4_next, acc4);
            acc5 = fmaf(a_value_next, b5_next, acc5);
            acc6 = fmaf(a_value_next, b6_next, acc6);
            acc7 = fmaf(a_value_next, b7_next, acc7);
            sqr_partial = fmaf(a_value_next, a_value_next, sqr_partial);
            a_ptr += 64;
            b_ptr0 += 64;
            b_ptr1 += 64;
            b_ptr2 += 64;
            b_ptr3 += 64;
            b_ptr4 += 64;
            b_ptr5 += 64;
            b_ptr6 += 64;
            b_ptr7 += 64;
          }
        } else {
          for (int k_idx = k_begin + lane; k_idx < k_end; k_idx += 64) {
            const float a_value = __bfloat162float(*a_ptr);
            const float b0 = tf32_round(*b_ptr0);
            const float b1 = tf32_round(*b_ptr1);
            const float b2 = tf32_round(*b_ptr2);
            const float b3 = tf32_round(*b_ptr3);
            const float b4 = tf32_round(*b_ptr4);
            const float b5 = tf32_round(*b_ptr5);
            const float b6 = tf32_round(*b_ptr6);
            const float b7 = tf32_round(*b_ptr7);
            acc0 = fmaf(a_value, b0, acc0);
            acc1 = fmaf(a_value, b1, acc1);
            acc2 = fmaf(a_value, b2, acc2);
            acc3 = fmaf(a_value, b3, acc3);
            acc4 = fmaf(a_value, b4, acc4);
            acc5 = fmaf(a_value, b5, acc5);
            acc6 = fmaf(a_value, b6, acc6);
            acc7 = fmaf(a_value, b7, acc7);
            const float a_value_next = __bfloat162float(a_ptr[32]);
            const float b0_next = tf32_round(b_ptr0[32]);
            const float b1_next = tf32_round(b_ptr1[32]);
            const float b2_next = tf32_round(b_ptr2[32]);
            const float b3_next = tf32_round(b_ptr3[32]);
            const float b4_next = tf32_round(b_ptr4[32]);
            const float b5_next = tf32_round(b_ptr5[32]);
            const float b6_next = tf32_round(b_ptr6[32]);
            const float b7_next = tf32_round(b_ptr7[32]);
            acc0 = fmaf(a_value_next, b0_next, acc0);
            acc1 = fmaf(a_value_next, b1_next, acc1);
            acc2 = fmaf(a_value_next, b2_next, acc2);
            acc3 = fmaf(a_value_next, b3_next, acc3);
            acc4 = fmaf(a_value_next, b4_next, acc4);
            acc5 = fmaf(a_value_next, b5_next, acc5);
            acc6 = fmaf(a_value_next, b6_next, acc6);
            acc7 = fmaf(a_value_next, b7_next, acc7);
            a_ptr += 64;
            b_ptr0 += 64;
            b_ptr1 += 64;
            b_ptr2 += 64;
            b_ptr3 += 64;
            b_ptr4 += 64;
            b_ptr5 += 64;
            b_ptr6 += 64;
            b_ptr7 += 64;
          }
        }

        acc0 = warp_reduce_sum(acc0);
        acc1 = warp_reduce_sum(acc1);
        acc2 = warp_reduce_sum(acc2);
        acc3 = warp_reduce_sum(acc3);
        acc4 = warp_reduce_sum(acc4);
        acc5 = warp_reduce_sum(acc5);
        acc6 = warp_reduce_sum(acc6);
        acc7 = warp_reduce_sum(acc7);
        if (n_group == 0) {
          sqr_partial = warp_reduce_sum(sqr_partial);
        }

        if (lane == 0) {
          const int64_t d_base = split_idx * d_stride_split +
                                 m_idx * d_stride_m + n_base * d_stride_n;
          d[d_base + 0 * d_stride_n] = acc0;
          d[d_base + 1 * d_stride_n] = acc1;
          d[d_base + 2 * d_stride_n] = acc2;
          d[d_base + 3 * d_stride_n] = acc3;
          d[d_base + 4 * d_stride_n] = acc4;
          d[d_base + 5 * d_stride_n] = acc5;
          d[d_base + 6 * d_stride_n] = acc6;
          d[d_base + 7 * d_stride_n] = acc7;

          if (n_group == 0) {
            const int64_t sqr_sum_offset =
                split_idx * sqr_sum_stride_split + m_idx * sqr_sum_stride_m;
            sqr_sum[sqr_sum_offset] = sqr_partial;
          }
        }
      }
    }
    return;
  }

  for (int elem_idx = static_cast<int>(threadIdx.x);
       elem_idx < kBlockM * block_n; elem_idx += kNumThreads) {
    const int local_m = elem_idx / block_n;
    const int n_idx = elem_idx % block_n;
    const int m_idx = m_begin + local_m;
    if (m_idx >= m || n_idx >= n) {
      continue;
    }

    float partial = 0.0f;
    for (int k_idx = k_begin; k_idx < k_end; ++k_idx) {
      const float a_value =
          __bfloat162float(a[m_idx * a_stride_m + k_idx * a_stride_k]);
      const float b_value =
          tf32_round(b[n_idx * b_stride_n + k_idx * b_stride_k]);
      partial = fmaf(a_value, b_value, partial);
    }
    const int64_t d_offset = has_split_output
                                 ? split_idx * d_stride_split +
                                       m_idx * d_stride_m + n_idx * d_stride_n
                                 : m_idx * d_stride_m + n_idx * d_stride_n;
    d[d_offset] = partial;
  }

  __shared__ float sqr_reduce_buffer[128];
  for (int local_m = 0; local_m < kBlockM; ++local_m) {
    const int sqr_m_idx = m_begin + local_m;
    float sqr_partial = 0.0f;
    if (threadIdx.x < 128 && sqr_m_idx < m) {
      for (int k_idx = k_begin + static_cast<int>(threadIdx.x); k_idx < k_end;
           k_idx += 128) {
        const float a_value =
            __bfloat162float(a[sqr_m_idx * a_stride_m + k_idx * a_stride_k]);
        sqr_partial = fmaf(a_value, a_value, sqr_partial);
      }
    }

    if (threadIdx.x < 128) {
      sqr_reduce_buffer[threadIdx.x] = sqr_partial;
    }
    __syncthreads();

    for (int offset = 64; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        sqr_reduce_buffer[threadIdx.x] +=
            sqr_reduce_buffer[threadIdx.x + offset];
      }
      __syncthreads();
    }

    if (threadIdx.x != 0 || sqr_m_idx >= m) {
      continue;
    }

    const int64_t sqr_sum_offset =
        has_split_output
            ? split_idx * sqr_sum_stride_split + sqr_m_idx * sqr_sum_stride_m
            : sqr_m_idx * sqr_sum_stride_m;
    sqr_sum[sqr_sum_offset] = sqr_reduce_buffer[0];
  }
}

void launch_tf32_hc_prenorm_gemm_raw(
    const int m, const int n, const int k, const int block_n,
    const int num_splits, const int has_split_output, const void *a,
    const float *b, float *d, float *sqr_sum, const int64_t a_stride_m,
    const int64_t a_stride_k, const int64_t b_stride_n,
    const int64_t b_stride_k, const int64_t d_stride_split,
    const int64_t d_stride_m, const int64_t d_stride_n,
    const int64_t sqr_sum_stride_split, const int64_t sqr_sum_stride_m,
    cudaStream_t stream) {
  const bool use_fast_path =
      has_split_output == 1 && n == 24 && block_n == 32 && a_stride_k == 1 &&
      b_stride_k == 1 && d_stride_n == 1 && sqr_sum_stride_m == 1;
  const int row_tile = use_fast_path ? kFastPathBlockM : kBlockM;
  const int grid_x = num_splits * ((m + row_tile - 1) / row_tile);
  const int num_threads = use_fast_path ? kFastPathNumThreads : kNumThreads;
  tf32_hc_prenorm_gemm_kernel<<<grid_x, num_threads, 0, stream>>>(
      m, n, k, block_n, num_splits, has_split_output,
      reinterpret_cast<const __nv_bfloat16 *>(a), b, d, sqr_sum, a_stride_m,
      a_stride_k, b_stride_n, b_stride_k, d_stride_split, d_stride_m,
      d_stride_n, sqr_sum_stride_split, sqr_sum_stride_m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

void tf32_hc_prenorm_gemm(const torch::Tensor &a, const torch::Tensor &b,
                          torch::Tensor &d, torch::Tensor &sqr_sum,
                          const c10::optional<int64_t> &num_splits) {
  TORCH_CHECK(a.is_cuda(), "tf32_hc_prenorm_gemm expects a to be CUDA");
  TORCH_CHECK(b.is_cuda(), "tf32_hc_prenorm_gemm expects b to be CUDA");
  TORCH_CHECK(d.is_cuda(), "tf32_hc_prenorm_gemm expects d to be CUDA");
  TORCH_CHECK(sqr_sum.is_cuda(),
              "tf32_hc_prenorm_gemm expects sqr_sum to be CUDA");
  TORCH_CHECK(a.dim() == 2, "tf32_hc_prenorm_gemm expects a to be 2D");
  TORCH_CHECK(b.dim() == 2, "tf32_hc_prenorm_gemm expects b to be 2D");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16,
              "tf32_hc_prenorm_gemm expects a dtype bfloat16");
  TORCH_CHECK(b.scalar_type() == at::kFloat,
              "tf32_hc_prenorm_gemm expects b dtype float32");
  TORCH_CHECK(d.scalar_type() == at::kFloat,
              "tf32_hc_prenorm_gemm expects d dtype float32");
  TORCH_CHECK(sqr_sum.scalar_type() == at::kFloat,
              "tf32_hc_prenorm_gemm expects sqr_sum dtype float32");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1,
              "tf32_hc_prenorm_gemm expects a and b to be K-major contiguous");
  TORCH_CHECK(sqr_sum.is_contiguous(),
              "tf32_hc_prenorm_gemm expects sqr_sum to be contiguous");

  const int64_t m64 = a.size(0);
  const int64_t k64 = a.size(1);
  const int64_t n64 = b.size(0);
  const int64_t k_from_b64 = b.size(1);
  TORCH_CHECK(k64 == k_from_b64,
              "tf32_hc_prenorm_gemm K mismatch: a.shape=", a.sizes(),
              ", b.shape=", b.sizes());

  const bool has_split_output = num_splits.has_value();
  if (has_split_output) {
    TORCH_CHECK(d.dim() == 3,
                "tf32_hc_prenorm_gemm split output expects d to be 3D");
    TORCH_CHECK(sqr_sum.dim() == 2,
                "tf32_hc_prenorm_gemm split output expects sqr_sum to be 2D");
    TORCH_CHECK(d.size(0) == *num_splits && sqr_sum.size(0) == *num_splits,
                "tf32_hc_prenorm_gemm split output size mismatch");
    TORCH_CHECK(d.size(1) == m64 && d.size(2) == n64 && sqr_sum.size(1) == m64,
                "tf32_hc_prenorm_gemm split output shape mismatch");
    TORCH_CHECK(
        d.stride(2) == 1,
        "tf32_hc_prenorm_gemm expects split d last dimension contiguous");
  } else {
    TORCH_CHECK(d.dim() == 2,
                "tf32_hc_prenorm_gemm non-split output expects d to be 2D");
    TORCH_CHECK(
        sqr_sum.dim() == 1,
        "tf32_hc_prenorm_gemm non-split output expects sqr_sum to be 1D");
    TORCH_CHECK(d.size(0) == m64 && d.size(1) == n64 && sqr_sum.size(0) == m64,
                "tf32_hc_prenorm_gemm non-split output shape mismatch");
    TORCH_CHECK(
        d.stride(1) == 1,
        "tf32_hc_prenorm_gemm expects non-split d last dimension contiguous");
  }

  TORCH_CHECK(n64 > 0 && k64 > 0, "tf32_hc_prenorm_gemm expects n,k > 0");
  if (m64 == 0) {
    return;
  }
  TORCH_CHECK(n64 <= 128, "tf32_hc_prenorm_gemm expects n <= 128");
  TORCH_CHECK(n64 % 8 == 0, "tf32_hc_prenorm_gemm expects n % 8 == 0");
  TORCH_CHECK(k64 % kBlockK == 0,
              "tf32_hc_prenorm_gemm expects k to be divisible by ", kBlockK);
  TORCH_CHECK(m64 <= std::numeric_limits<int>::max() &&
                  n64 <= std::numeric_limits<int>::max() &&
                  k64 <= std::numeric_limits<int>::max(),
              "tf32_hc_prenorm_gemm dimensions exceed int range");

  const int64_t num_splits64 = has_split_output ? *num_splits : 1;
  TORCH_CHECK(num_splits64 >= 1,
              "tf32_hc_prenorm_gemm expects num_splits >= 1");
  TORCH_CHECK(num_splits64 <= k64 / kBlockK,
              "tf32_hc_prenorm_gemm num_splits exceeds number of K blocks");
  TORCH_CHECK(num_splits64 <= std::numeric_limits<int>::max(),
              "tf32_hc_prenorm_gemm num_splits exceeds int range");

  const int m = static_cast<int>(m64);
  const int n = static_cast<int>(n64);
  const int k = static_cast<int>(k64);
  const int block_n = align_to(n, 16);
  const int num_splits_value = static_cast<int>(num_splits64);
  const int64_t d_stride_split = has_split_output ? d.stride(0) : 0;
  const int64_t d_stride_m = d.stride(has_split_output ? 1 : 0);
  const int64_t d_stride_n = d.stride(has_split_output ? 2 : 1);
  const int64_t sqr_sum_stride_split = has_split_output ? sqr_sum.stride(0) : 0;
  const int64_t sqr_sum_stride_m = sqr_sum.stride(has_split_output ? 1 : 0);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  launch_tf32_hc_prenorm_gemm_raw(
      m, n, k, block_n, num_splits_value, static_cast<int>(has_split_output),
      a.data_ptr(), b.data_ptr<float>(), d.data_ptr<float>(),
      sqr_sum.data_ptr<float>(), a.stride(0), a.stride(1), b.stride(0),
      b.stride(1), d_stride_split, d_stride_m, d_stride_n, sqr_sum_stride_split,
      sqr_sum_stride_m, at::cuda::getCurrentCUDAStream().stream());
}
// DL end

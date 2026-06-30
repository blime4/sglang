// DL begin
//
// Graph-safe paged-decode attention kernel for DLIN. Reads the paged KV cache
// DIRECTLY (no gather/scatter/packing) via page_table + seqlens -- all inputs
// are sglang's own fixed-shape tensors (already in the cuda-graph pool), so
// the single kernel launch captures + replays cleanly. Avoids every wall that
// packing-based approaches hit on DLIN's cuda graph runtime:
//   - gather (_gk[_mask]): boolean index -> graph-breaker.
//   - scatter: per-call alloc -> "CUDA error"; persistent buf -> replay corrupt.
// This kernel: ONE launch, no intermediate tensors, all graph-pool inputs.
//
// Math: standard online-softmax decode attention. For each (batch, q_head):
//   attend q to KV[0:seqlen] (paged via page_table), GQA head mapping.
//
// Registered as torch.ops.sgl_kernel.paged_decode_attn.
//
// DL end

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include "utils.h"

namespace sgl_kernel_dl {

// FloatCvt (dlcc rejects static_cast<float>(__half/__nv_bfloat16))
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

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Block reduce; returns total to ALL threads. smem must have >= 32 floats.
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

// One block per (batch, q_head). blockDim.x = D (head_dim).
template <typename T>
__global__ void paged_decode_attn_kernel(
    const T* __restrict__ q,          // [B, Hq, D]
    const T* __restrict__ k_cache,    // [num_blocks, Pg, Hkv, D]
    const T* __restrict__ v_cache,    // [num_blocks, Pg, Hkv, D]
    const int32_t* __restrict__ page_table,  // [B, max_blocks]
    const int32_t* __restrict__ seqlens,     // [B]
    T* __restrict__ out,              // [B, Hq, D]
    const float scale,
    const int Hq, const int Hkv, const int D,
    const int Pg, const int max_blocks)
{
    const int b = blockIdx.x / Hq;
    const int h = blockIdx.x % Hq;
    const int kv_group = h * Hkv / Hq;   // GQA: consecutive q heads share kv head
    const int seqlen = seqlens[b];
    const int tid = threadIdx.x;          // 0..D-1

    const float q_val = FloatCvt<T>::to(q[(int64_t)b * Hq * D + h * D + tid]);

    extern __shared__ float smem[];       // for block_reduce_sum (32 floats)

    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float out_val = 0.0f;

    for (int pos = 0; pos < seqlen; pos++) {
        int blk = page_table[b * max_blocks + pos / Pg];
        int off = pos % Pg;
        int64_t base = ((int64_t)blk * Pg + off) * Hkv * D + (int64_t)kv_group * D;

        // dot product q · k  (block-reduce across D)
        float partial = q_val * FloatCvt<T>::to(k_cache[base + tid]);
        float score = block_reduce_sum(partial, smem) * scale;

        // online softmax
        float new_max = fmaxf(max_score, score);
        float exp_old = __expf(max_score - new_max);
        float exp_new = __expf(score - new_max);
        sum_exp = sum_exp * exp_old + exp_new;
        out_val = out_val * exp_old + exp_new * FloatCvt<T>::to(v_cache[base + tid]);
        max_score = new_max;
    }

    out_val = (sum_exp > 0.0f) ? (out_val / sum_exp) : 0.0f;
    out[(int64_t)b * Hq * D + h * D + tid] = FloatCvt<T>::from(out_val);
}

void paged_decode_attn(
    torch::Tensor q,            // [B, Hq, D]
    torch::Tensor k_cache,      // [num_blocks, Pg, Hkv, D]
    torch::Tensor v_cache,      // [num_blocks, Pg, Hkv, D]
    torch::Tensor page_table,   // [B, max_blocks]
    torch::Tensor seqlens,      // [B]
    torch::Tensor out,          // [B, Hq, D]
    double softmax_scale)
{
    CHECK_INPUT(q); CHECK_INPUT(k_cache); CHECK_INPUT(v_cache);
    CHECK_INPUT(page_table); CHECK_INPUT(seqlens); CHECK_INPUT(out);
    const auto device = q.device();
    CHECK_EQ(k_cache.device(), device);
    const at::cuda::CUDAGuard guard(device);

    const int B = q.size(0);
    const int Hq = q.size(1);
    const int D = q.size(2);
    const int Hkv = k_cache.size(2);
    const int Pg = k_cache.size(1);
    const int max_blocks = page_table.size(1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int grid = B * Hq;
    const int block = D;            // one thread per head_dim element
    const int smem_bytes = 32 * sizeof(float);  // block_reduce_sum scratch

    if (q.scalar_type() == at::ScalarType::Half) {
        paged_decode_attn_kernel<__half><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<__half*>(q.data_ptr()),
            reinterpret_cast<const __half*>(k_cache.data_ptr()),
            reinterpret_cast<const __half*>(v_cache.data_ptr()),
            page_table.data_ptr<int32_t>(),
            seqlens.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(out.data_ptr()),
            static_cast<float>(softmax_scale), Hq, Hkv, D, Pg, max_blocks);
    } else if (q.scalar_type() == at::ScalarType::BFloat16) {
        paged_decode_attn_kernel<__nv_bfloat16><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
            page_table.data_ptr<int32_t>(),
            seqlens.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<float>(softmax_scale), Hq, Hkv, D, Pg, max_blocks);
    } else {
        TORCH_CHECK(false, "paged_decode_attn (DLIN) supports fp16/bf16 only, got ",
                    q.scalar_type());
    }
}

}  // namespace sgl_kernel_dl
// DL end

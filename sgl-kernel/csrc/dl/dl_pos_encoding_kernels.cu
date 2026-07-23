// DL begin: vendored from Denglin vLLM fork csrc/dl/dl_pos_encoding_kernels.cu (port plan 4b/4d).
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const scalar_t* cache_ptr, const int head_size, const int num_heads,
    const int num_kv_heads, const int rot_dim, const int token_idx,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}

// DL deepseek_yarn_rotary_embedding
template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_deepseek_yarn_rotary_embedding(
    scalar_t* __restrict__ query,  // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,    // [num_tokens, num_kv_heads, head_size]
    const scalar_t* cache_ptr, const int head_size, const int num_heads,
    const int num_kv_heads, const int rot_dim, const int token_idx,
    const int64_t query_token_stride, const int64_t query_head_stride,
    const int64_t key_token_stride, const int64_t key_head_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_token_stride + head_idx * query_head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_token_stride + head_idx * key_head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void deepseek_yarn_rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    scalar_t* __restrict__ query,           // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,             // [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim,
    const int64_t query_token_stride,
    const int64_t query_head_stride,
    const int64_t key_token_stride,
    const int64_t key_head_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_deepseek_yarn_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_token_stride, query_head_stride, key_token_stride, key_head_stride);
}


// DL longrope_rotary_embedding
template <typename scalar_t, bool IS_NEOX>
__global__ void longrope_rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t query_stride,
    const int64_t key_stride, const int64_t head_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int64_t pos_size, const int64_t k) {
  // read positions to judege if it need add k
  int64_t elements_per_thread = (pos_size - 1) / blockDim.x + 1;
  extern __shared__ bool condition_shared[];
  bool k_flag = false;
  for (int i=0; i<elements_per_thread && threadIdx.x + blockDim.x * i < pos_size; i++) {
    if (positions[threadIdx.x + blockDim.x * i] > k) {
      k_flag = true;
      break;
    }
  }
  condition_shared[threadIdx.x] = k_flag;
  __syncthreads();

  bool total_k_flag = true;
  for (int i=0; i<blockDim.x; i++) {
    if (!condition_shared[i]) {
      total_k_flag = false;
      break;
    }
  }

  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = total_k_flag ? positions[token_idx] + k : positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride, head_stride);
}

// DL batched_longrope_rotary_embedding
template <typename scalar_t, bool IS_NEOX>
__global__ void batched_longrope_rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2]
    const int64_t* __restrict__ cos_sin_cache_offsets,  // [batch_size, seq_len]
                                                        // or [num_tokens]
    const int rot_dim, const int64_t query_stride,
    const int64_t key_stride, const int64_t head_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int64_t pos_size, const int64_t k) {
  // read positions to judege if it need add k
  int64_t elements_per_thread = (pos_size - 1) / blockDim.x + 1;
  extern __shared__ bool condition_shared[];
  bool k_flag = false;
  for (int i=0; i<elements_per_thread && threadIdx.x + blockDim.x * i < pos_size; i++) {
    if (positions[threadIdx.x + blockDim.x * i] > k) {
      k_flag = true;
      break;
    }
  }
  condition_shared[threadIdx.x] = k_flag;
  __syncthreads();

  bool total_k_flag = true;
  for (int i=0; i<blockDim.x; i++) {
    if (!condition_shared[i]) {
      total_k_flag = false;
      break;
    }
  }

  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = total_k_flag ? positions[token_idx] + k : positions[token_idx];
  int64_t cos_sin_cache_offset = cos_sin_cache_offsets[token_idx];
  const scalar_t* cache_ptr =
      cos_sin_cache + (cos_sin_cache_offset + pos) * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride, head_stride);
}

// DL batched_deepseek_yarn_rotary_embedding
template <typename scalar_t, bool IS_NEOX>
__global__ void batched_deepseek_yarn_rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    scalar_t* __restrict__ query,           // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,             // [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int64_t* __restrict__ cos_sin_cache_offsets,  // [num_tokens]
    const int rot_dim,
    const int64_t query_token_stride, const int64_t query_head_stride,
    const int64_t key_token_stride, const int64_t key_head_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  int64_t cos_sin_cache_offset = cos_sin_cache_offsets[token_idx];
  const scalar_t* cache_ptr =
      cos_sin_cache + (cos_sin_cache_offset + pos) * rot_dim;

  apply_deepseek_yarn_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_token_stride, query_head_stride, key_token_stride, key_head_stride);
}
}  // namespace vllm

void longrope_rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size]
    torch::Tensor& key,    // [batch_size, seq_len, num_kv_heads * head_size] or
                           // [num_tokens, num_kv_heads * head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    int64_t k              // original_max_position_embeddings
  ) {
  int64_t num_tokens = query.numel() / query.size(-1);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);
  int pos_dim = positions.dim();
  int64_t pos_size = pos_dim == 1 ? positions.size(0) : positions.size(1) * positions.size(1);
  int64_t head_stride = head_size;

  int64_t block_dim = std::min<int64_t>(num_heads * rot_dim / 2, 512);
  dim3 grid(num_tokens);
  dim3 block(block_dim);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    vllm::longrope_rotary_embedding_kernel<scalar_t, true>
        <<<grid, block, block_dim, stream>>>(
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
            rot_dim, query_stride, key_stride, head_stride, num_heads, num_kv_heads,
            head_size, pos_size, k);
  });
}

/*
Batched version of rotary embedding, pack multiple LoRAs together
and process in batched manner.
*/
void batched_longrope_rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size]
    torch::Tensor& key,    // [batch_size, seq_len, num_kv_heads * head_size] or
                           // [num_tokens, num_kv_heads * head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    int64_t rot_dim,
    torch::Tensor& cos_sin_cache_offsets,  // [num_tokens]
    int64_t k              // original_max_position_embeddings
) {
  int64_t num_tokens = cos_sin_cache_offsets.size(0);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);
  int pos_dim = positions.dim();
  int64_t pos_size = pos_dim == 1 ? positions.size(0) : positions.size(1) * positions.size(1);
  int64_t head_stride = head_size;

  int64_t block_dim = std::min<int64_t>(num_heads * rot_dim / 2, 512);
  dim3 grid(num_tokens);
  dim3 block(block_dim);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    vllm::batched_longrope_rotary_embedding_kernel<scalar_t, true>
        <<<grid, block, block_dim, stream>>>(
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
            cos_sin_cache_offsets.data_ptr<int64_t>(), rot_dim, query_stride,
            key_stride, head_stride, num_heads, num_kv_heads, head_size, pos_size, k);
  });
}

void deepseek_yarn_rotary_embedding(
    torch::Tensor& positions,  // [num_tokens]
    torch::Tensor& query,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key,        // [num_tokens, 1, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  int64_t num_tokens = positions.numel();
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-2);
  int num_kv_heads = key.size(-2);
  int64_t query_head_stride = query.stride(-2);
  int64_t query_token_stride = query.stride(-3);
  int64_t key_head_stride = key.stride(-2);
  int64_t key_token_stride = key.stride(-3);

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    if (is_neox) {
      vllm::deepseek_yarn_rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(), rot_dim,
          query_token_stride, query_head_stride, key_token_stride, key_head_stride,
          num_heads, num_kv_heads, head_size);
    } else {
      vllm::deepseek_yarn_rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(), rot_dim,
          query_token_stride, query_head_stride, key_token_stride, key_head_stride,
          num_heads, num_kv_heads, head_size);
    }
  });
}

/*
Batched version of deepseek yarn rotary embedding, pack multiple LoRAs together
and process in batched manner.
*/
void batched_deepseek_yarn_rotary_embedding(
    torch::Tensor& positions,  // [num_tokens]
    torch::Tensor& query,  // [num_tokens, num_heads, head_size]
    torch::Tensor& key,    // [num_tokens, num_kv_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox, int64_t rot_dim,
    torch::Tensor& cos_sin_cache_offsets  // [num_tokens]
) {
  int64_t num_tokens = cos_sin_cache_offsets.size(0);
  int num_heads = query.size(-2);
  int num_kv_heads = key.size(-2);
  int64_t query_head_stride = query.stride(-2);
  int64_t query_token_stride = query.stride(-3);
  int64_t key_head_stride = key.stride(-2);
  int64_t key_token_stride = key.stride(-3);

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    if (is_neox) {
      vllm::batched_deepseek_yarn_rotary_embedding_kernel<scalar_t, true>
          <<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
              cos_sin_cache_offsets.data_ptr<int64_t>(), rot_dim,
              query_token_stride, query_head_stride,
              key_token_stride, key_head_stride,
              num_heads, num_kv_heads, head_size);
    } else {
      vllm::batched_deepseek_yarn_rotary_embedding_kernel<scalar_t, false>
          <<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
              cos_sin_cache_offsets.data_ptr<int64_t>(), rot_dim,
              query_token_stride, query_head_stride,
              key_token_stride, key_head_stride,
              num_heads, num_kv_heads, head_size);
    }
  });
}// DL end

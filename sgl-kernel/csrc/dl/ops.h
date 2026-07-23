#pragma once

#include <torch/all.h>


void gemma_rms_norm(
    torch::Tensor& out, torch::Tensor& input,
    torch::Tensor& weight,double epsilon);

void fused_add_gemma_rms_norm(
    torch::Tensor& input, torch::Tensor& residual,
    torch::Tensor& weight, double epsilon);

void dl_chunk_gated_delta_rule(
    torch::Tensor& output, torch::Tensor& q, torch::Tensor& k,
    torch::Tensor& v, torch::Tensor& g, torch::Tensor& beta,
    torch::Tensor& initial_state, torch::Tensor& cu_seqlens,
    double scale, bool use_qk_l2norm_in_kernel);

void dl_recurrent_gated_delta_rule(
    torch::Tensor& output,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    torch::Tensor& beta,
    torch::Tensor& ssm_state,
    torch::Tensor& ssm_state_indices,
    const c10::optional<torch::Tensor>& cu_seqlens,
    const c10::optional<torch::Tensor>& num_accepted_tokens,
    double scale, bool use_qk_l2norm_in_kernel);

torch::Tensor fp8_fp4_mqa_logits(
    const torch::Tensor& q_values, const c10::optional<torch::Tensor>& q_scale,
    const torch::Tensor& kv_packed, const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seqlen_ks, const torch::Tensor& cu_seqlen_ke,
    bool clean_logits = true,
    int64_t max_seqlen_k = 0,
    at::ScalarType logits_type = at::kFloat);

std::vector<torch::Tensor> flash_mla_sparse_prefill_fwd(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& indices,
    double sm_scale,
    int64_t d_v,
    const c10::optional<torch::Tensor>& attn_sink = c10::nullopt,
    const c10::optional<torch::Tensor>& topk_length = c10::nullopt,
    const c10::optional<torch::Tensor>& out = c10::nullopt);

std::vector<torch::Tensor> flash_mla_with_kvcache(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const c10::optional<torch::Tensor>& block_table = c10::nullopt,
    const c10::optional<torch::Tensor>& cache_seqlens = c10::nullopt,
    int64_t head_dim_v = 512,
    double softmax_scale = 0.0,
    bool causal = false,
    bool is_fp8_kvcache = false,
    const c10::optional<torch::Tensor>& indices = c10::nullopt,
    const c10::optional<torch::Tensor>& attn_sink = c10::nullopt,
    const c10::optional<torch::Tensor>& extra_k_cache = c10::nullopt,
    const c10::optional<torch::Tensor>& extra_indices_in_cache = c10::nullopt,
    const c10::optional<torch::Tensor>& topk_length = c10::nullopt,
    const c10::optional<torch::Tensor>& extra_topk_length = c10::nullopt,
    const c10::optional<torch::Tensor>& out = c10::nullopt);

torch::Tensor fp8_fp4_paged_mqa_logits(
    const torch::Tensor& q_values, const c10::optional<torch::Tensor>& q_scale,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& block_tables,
    const torch::Tensor& schedule_metadata,
    int64_t max_model_len, bool clean_logits = false,
    at::ScalarType logits_type = at::kFloat,
    const c10::optional<torch::Tensor>& indices = c10::nullopt);

void fp8_einsum(
    const torch::Tensor& a_values, const torch::Tensor& a_scale,
    const torch::Tensor& b_values, const torch::Tensor& b_scale,
    torch::Tensor& out,
    const std::string& equation,
    const std::vector<int64_t>& recipe);

void tf32_hc_prenorm_gemm(
    const torch::Tensor& a, const torch::Tensor& b,
    torch::Tensor& d, torch::Tensor& sqr_sum,
    const c10::optional<int64_t>& num_splits = c10::nullopt);

torch::Tensor w8a8_matmul(
    torch::Tensor a, torch::Tensor b_q_weight,
    std::optional<torch::Tensor> const& a_scales,
    torch::Tensor b_scales, bool a_is_quantized = false);

// DL custom longrope_rotary_embedding
void longrope_rotary_embedding(
    torch::Tensor& positions, torch::Tensor& query,
    torch::Tensor& key, int64_t head_size,
    torch::Tensor& cos_sin_cache, int64_t k);

void batched_longrope_rotary_embedding(
    torch::Tensor& positions, torch::Tensor& query,
    torch::Tensor& key, int64_t head_size,
    torch::Tensor& cos_sin_cache,
    int64_t rot_dim,
    torch::Tensor& cos_sin_cache_offsets, int64_t k);

// DL custom deepseek_yarn_rotary_embedding
void deepseek_yarn_rotary_embedding(
    torch::Tensor& positions, torch::Tensor& query,
    torch::Tensor& key, int64_t head_size,
    torch::Tensor& cos_sin_cache, bool is_neox);

void batched_deepseek_yarn_rotary_embedding(
    torch::Tensor& positions, torch::Tensor& query,
    torch::Tensor& key, int64_t head_size,
    torch::Tensor& cos_sin_cache, bool is_neox,
    int64_t rot_dim,
    torch::Tensor& cos_sin_cache_offsets);

torch::Tensor gptq_dlblas_gemmex(
    torch::Tensor a, torch::Tensor b_q_weight,
    torch::Tensor b_gptq_qzeros,
    torch::Tensor b_gptq_scales,
    int64_t quant_type,
    int64_t bit = 4);

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
    int64_t M);

// DL invoke fused moe optimized
void invoke_fused_moe_opt(
    torch::Tensor& x,
    torch::Tensor& w,
    torch::Tensor& y,
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
    int64_t M);

// DL grouped_topk optimized
void moe_fused_grouped_topk(
    torch::Tensor& gating,
    torch::Tensor& bias,
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weights,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group);

// DL, lora ops
void dl_lora_shrink(
    const at::Tensor& inputs,
    const at::Tensor& lora_ptr_tensor,
    const at::Tensor& lora_ptr_cpu_tensor,
    int64_t lora_strides_d0,
    int64_t lora_strides_d1,
    int64_t lora_strides_d2,
    at::Tensor& output_tensor,
    const at::Tensor& token_lora_mapping,
    const at::Tensor& token_indices_sorted_by_lora_ids,
    const at::Tensor& num_tokens_per_lora,
    const at::Tensor& lora_token_start_loc,
    const at::Tensor& lora_ids,
    double scaling);

void dl_lora_expand(
    const at::Tensor& inputs,
    const at::Tensor& lora_ptr_tensor,
    const at::Tensor& lora_ptr_cpu_tensor,
    const at::Tensor& lora_strides_d0_tensor,
    const at::Tensor& lora_strides_d1_tensor,
    const at::Tensor& lora_strides_d2_tensor,
    at::Tensor& output_tensor,
    const at::Tensor& slice_start_tensor,
    const at::Tensor& hidden_sizes_tensor,
    const at::Tensor& hidden_sizes_cpu_tensor,
    int64_t max_n,
    const at::Tensor& token_lora_mapping,
    const at::Tensor& token_indices_sorted_by_lora_ids,
    const at::Tensor& num_tokens_per_lora,
    const at::Tensor& lora_token_start_loc,
    int64_t offset_start,
    const at::Tensor& lora_ids,
    bool add_inputs);

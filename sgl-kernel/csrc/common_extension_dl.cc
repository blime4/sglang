// DL begin
// Denglin (登临/DLIN) curated pybind entry point for sgl_kernel.common_ops.
//
// DLIN analog of common_extension_rocm.cc. Registers ONLY ops whose backing
// sources are header-clean (no FetchContent deps): the rest of sgl-kernel needs
// libcudacxx `cuda/functional`, FlashInfer, CUTLASS, or pytorch_extension_utils.h
// which the setup_dl.py path does not fetch. This minimal set proves the dlcc
// build path end-to-end; coverage grows in later phases (fetch those headers,
// then re-enable activation/norm/quant/gemm/moe ops). See plan §5.
//
// Built via sgl-kernel/setup_dl.py (torch CUDAExtension; torch's dl-aware
// cpp_extension picks dlcc + --cuda-gpu-arch=dlgput64 automatically).

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"
// DL begin: vendored vllm dl/ kernel declarations (Phase 4b/4d bulk port).
#include "dl/ops.h"

// DL begin
// csrc/elementwise/rmsnorm_dl.cu — standalone RMSNorm (no FlashInfer dep).
// Schemas match the python wrappers in sgl_kernel/elementwise.py
// (torch.ops.sgl_kernel.{rmsnorm,fused_add_rmsnorm}.default).
namespace sgl_kernel_dl {
void rmsnorm(torch::Tensor out, torch::Tensor input, torch::Tensor weight,
             double eps, bool enable_pdl);
void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual,
                       torch::Tensor weight, double eps, bool enable_pdl);
// csrc/elementwise/gemma_rmsnorm_dl.cu — Gemma RMSNorm (weight+1).
void gemma_rmsnorm(torch::Tensor out, torch::Tensor input, torch::Tensor weight,
                   double eps);
void gemma_fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual,
                             torch::Tensor weight, double eps);
void paged_decode_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                       torch::Tensor page_table, torch::Tensor seqlens,
                       torch::Tensor out, double softmax_scale);
}  // namespace sgl_kernel_dl
// DL end

TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
  // csrc/elementwise/topk.cu
  m.def("fast_topk(Tensor score, Tensor indices, Tensor lengths, Tensor? row_starts) -> ()");
  m.impl("fast_topk", torch::kCUDA, &fast_topk_interface);

  // csrc/elementwise/pos_enc.cu
  m.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // csrc/moe/moe_align_kernel.cu
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // csrc/moe/moe_topk_{softmax,sigmoid}_kernels.cu (libcudacxx shim for cuda/functional)
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, float "
      "moe_softcapping, Tensor? correction_bias) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, Tensor? "
      "correction_bias) -> ()");
  m.impl("topk_sigmoid", torch::kCUDA, &topk_sigmoid);

  // csrc/memory/weak_ref_tensor.cpp
  m.def("weak_ref_tensor(Tensor tensor) -> Tensor");
  m.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

  // DL begin
  // csrc/elementwise/rmsnorm_dl.cu — standard RMSNorm + fused(residual_add).
  // Replaces the torch forward_native fallback; no FlashInfer header needed.
  m.def(
      "rmsnorm(Tensor! out, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &sgl_kernel_dl::rmsnorm);
  m.def(
      "fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, "
      "bool enable_pdl) -> ()");
  m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_kernel_dl::fused_add_rmsnorm);

  // csrc/elementwise/gemma_rmsnorm_dl.cu — Gemma RMSNorm (weight+1); replaces
  // vllm._dl_C.gemma_rms_norm / fused_add_gemma_rms_norm (plan Phase 4a).
  m.def("gemma_rmsnorm(Tensor! out, Tensor input, Tensor weight, float eps) -> ()");
  m.impl("gemma_rmsnorm", torch::kCUDA, &sgl_kernel_dl::gemma_rmsnorm);
  m.def("gemma_fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, "
        "float eps) -> ()");
  m.impl("gemma_fused_add_rmsnorm", torch::kCUDA, &sgl_kernel_dl::gemma_fused_add_rmsnorm);

  // csrc/dl/q_gemm_dlblas.cu — dlblas GPTQ GEMM (vendored from vllm); replaces
  // vllm._dl_C.gptq_dlblas_gemmex (plan 4a GEMM). Proves dlblas integration.
  m.def("gptq_dlblas_gemmex(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
        "Tensor b_gptq_scales, int quant_type, int bit=4) -> Tensor");
  m.impl("gptq_dlblas_gemmex", torch::kCUDA, &gptq_dlblas_gemmex);

  // DL begin: bulk-ported vllm _dl_C ops (Phase 4b/4d). Schemas match
  // vllm csrc/dl/torch_bindings.cpp; implementations in csrc/dl/*.cu.
  m.def(
      "dl_chunk_gated_delta_rule(Tensor! output, Tensor q, Tensor k, Tensor v, "
      "Tensor g, Tensor beta, Tensor initial_state, Tensor cu_seqlens, "
      "float scale, bool use_qk_l2norm_in_kernel) -> ()");
  m.impl("dl_chunk_gated_delta_rule", torch::kCUDA, &dl_chunk_gated_delta_rule);

  m.def(
      "dl_recurrent_gated_delta_rule(Tensor! output, Tensor q, Tensor k, Tensor v, "
      "Tensor g, Tensor beta, Tensor ssm_state, Tensor ssm_state_indices, "
      "Tensor? cu_seqlens, Tensor? num_accepted_tokens, "
      "float scale, bool use_qk_l2norm_in_kernel) -> ()");
  m.impl("dl_recurrent_gated_delta_rule", torch::kCUDA, &dl_recurrent_gated_delta_rule);

  m.def(
      "flash_mla_sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, "
      "float sm_scale, int d_v, Tensor? attn_sink=None, Tensor? topk_length=None, "
      "Tensor? out=None) -> Tensor[]");
  m.impl("flash_mla_sparse_prefill_fwd", torch::kCUDA, &flash_mla_sparse_prefill_fwd);

  m.def(
      "flash_mla_with_kvcache(Tensor q, Tensor k_cache, "
      "Tensor? block_table, Tensor? cache_seqlens, "
      "int head_dim_v, float softmax_scale, bool causal, bool is_fp8_kvcache, "
      "Tensor? indices=None, Tensor? attn_sink=None, "
      "Tensor? extra_k_cache=None, Tensor? extra_indices_in_cache=None, "
      "Tensor? topk_length=None, Tensor? extra_topk_length=None, "
      "Tensor? out=None) -> Tensor[]");
  m.impl("flash_mla_with_kvcache", torch::kCUDA, &flash_mla_with_kvcache);

  m.def(
      "fp8_fp4_mqa_logits(Tensor q_values, Tensor? q_scale, "
      "Tensor kv_packed, Tensor kv_scales, "
      "Tensor weights, Tensor cu_seqlen_ks, Tensor cu_seqlen_ke, "
      "bool clean_logits=True, int max_seqlen_k=0, "
      "ScalarType logits_type=float32) -> Tensor");
  m.impl("fp8_fp4_mqa_logits", torch::kCUDA, &fp8_fp4_mqa_logits);

  m.def(
      "fp8_fp4_paged_mqa_logits(Tensor q_values, Tensor? q_scale, "
      "Tensor kv_cache, Tensor weights, "
      "Tensor context_lens, Tensor block_tables, "
      "Tensor schedule_metadata, int max_model_len, "
      "bool clean_logits=False, ScalarType logits_type=float, "
      "Tensor? indices=None) -> Tensor");
  m.impl("fp8_fp4_paged_mqa_logits", torch::kCUDA, &fp8_fp4_paged_mqa_logits);

  m.def(
      "fp8_einsum(Tensor a_values, Tensor a_scale, "
      "Tensor b_values, Tensor b_scale, "
      "Tensor! out, str equation, int[] recipe) -> ()");
  m.impl("fp8_einsum", torch::kCUDA, &fp8_einsum);

  m.def(
      "tf32_hc_prenorm_gemm(Tensor a, Tensor b, Tensor! d, "
      "Tensor! sqr_sum, int? num_splits=None) -> ()");
  m.impl("tf32_hc_prenorm_gemm", torch::kCUDA, &tf32_hc_prenorm_gemm);

  m.def(
      "w8a8_matmul(Tensor a, Tensor b_q_weight, Tensor? a_scales, "
      "Tensor b_scales, bool a_is_quantized) -> Tensor");
  m.impl("w8a8_matmul", torch::kCUDA, &w8a8_matmul);

  m.def(
      "longrope_rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor! key, int head_size,"
      "                 Tensor cos_sin_cache, int k) -> ()");
  m.impl("longrope_rotary_embedding", torch::kCUDA, &longrope_rotary_embedding);

  m.def(
      "batched_longrope_rotary_embedding(Tensor positions, Tensor! query,"
      "                         Tensor! key, int head_size,"
      "                         Tensor cos_sin_cache,"
      "                         int rot_dim,"
      "                         Tensor cos_sin_cache_offsets, int k) -> ()");
  m.impl("batched_longrope_rotary_embedding", torch::kCUDA, &batched_longrope_rotary_embedding);

  m.def(
      "deepseek_yarn_rotary_embedding("
      "           Tensor positions, Tensor! query,"
      "           Tensor! key, int head_size,"
      "           Tensor cos_sin_cache, bool is_neox) -> ()");
  m.impl("deepseek_yarn_rotary_embedding", torch::kCUDA, &deepseek_yarn_rotary_embedding);

  m.def(
      "batched_deepseek_yarn_rotary_embedding("
      "       Tensor positions, Tensor! query,"
      "       Tensor! key, int head_size,"
      "       Tensor cos_sin_cache, bool is_neox,"
      "       int rot_dim,"
      "       Tensor cos_sin_cache_offsets) -> ()");
  m.impl("batched_deepseek_yarn_rotary_embedding", torch::kCUDA, &batched_deepseek_yarn_rotary_embedding);

  m.def(
      "invoke_fused_moe_opt(Tensor! x, Tensor! w, Tensor! y,"
      "  Tensor? w_bias, Tensor? w_scales, Tensor? w_zp, Tensor! topk_weights, Tensor! topk_ids,"
      "  Tensor! sorted_token_ids, Tensor! expert_ids,"
      "  Tensor! num_tokens_post_padded, bool mul_routed_weight,"
      "  int top_k, int block_size_m, int block_size_n, int block_size_k,"
      "  bool use_fp8_w8a8, bool use_int8_w8a16, bool use_int4_w4a16, bool use_mxfp4_w4a16,"
      "  int[] block_size, int M) -> ()");
  m.impl("invoke_fused_moe_opt", torch::kCUDA, &invoke_fused_moe_opt);

  m.def(
      "invoke_fused_moe_opt_v3(Tensor! x, Tensor! w, Tensor! y,"
      " Tensor? w_bias, Tensor? w_scales, Tensor? w_zp, Tensor! topk_weights, Tensor! topk_ids,"
      " Tensor! sorted_token_ids, Tensor! expert_ids,"
      " Tensor! num_tokens_post_padded, bool mul_routed_weight,"
      " int top_k, int block_size_m, int block_size_n, int block_size_k,"
      " int weight_bits, int[] block_size, int M) -> ()");
  m.impl("invoke_fused_moe_opt_v3", torch::kCUDA, &invoke_fused_moe_opt_v3);

  m.def(
      "moe_fused_grouped_topk(Tensor! gating, Tensor! bias,"
      " Tensor! topk_ids, Tensor! topk_weights,"
      " int topk, int num_expert_group, int topk_group) -> ()");
  m.impl("moe_fused_grouped_topk", torch::kCUDA, &moe_fused_grouped_topk);

  m.def(
      "dl_lora_shrink(Tensor! inputs, Tensor! lora_ptr_tensor, Tensor! lora_ptr_cpu_tensor,"
      "               int lora_strides_d0, int lora_strides_d1,"
      "               int lora_strides_d2, Tensor output_tensor,"
      "               Tensor! token_lora_mapping,"
      "               Tensor! token_indices_sorted_by_lora_id,"
      "               Tensor! num_tokens_per_lora, Tensor! lora_token_start_loc,"
      "               Tensor! lora_ids, float scaling) -> ()");
  m.impl("dl_lora_shrink", torch::kCUDA, &dl_lora_shrink);

  m.def(
      "dl_lora_expand(Tensor! inputs, Tensor! lora_ptr_tensor, Tensor! lora_ptr_cpu_tensor,"
      "               Tensor! lora_strides_d0, Tensor! lora_strides_d1,"
      "               Tensor! lora_strides_d2, Tensor output_tensor,"
      "               Tensor! slice_start_tensor, Tensor! hidden_size_tensor,"
      "               Tensor! hidden_size_cpu_tensor,"
      "               int max_n, Tensor! token_lora_mapping,"
      "               Tensor! token_indices_sorted_by_lora_id,"
      "               Tensor! num_tokens_per_lora, Tensor! lora_token_start_loc,"
      "               int offset_start, Tensor! lora_ids,"
      "               bool add_inputs) -> ()");
  m.impl("dl_lora_expand", torch::kCUDA, &dl_lora_expand);
  // DL end

  // csrc/elementwise/paged_decode_attn_dl.cu — graph-safe paged-decode attention
  // (no gather/scatter/packing; reads paged KV directly; single kernel launch).
  m.def(
      "paged_decode_attn(Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor page_table, Tensor seqlens, Tensor(a!) out, float softmax_scale) -> ()");
  m.impl("paged_decode_attn", torch::kCUDA, &sgl_kernel_dl::paged_decode_attn);
  // DL end

  // DL begin — custom allreduce ops (from common_extension.cc).
  // Source: csrc/allreduce/custom_all_reduce.cu (standard CUDA runtime APIs only —
  // cudaMemcpyAsync, CUDAStream; no NVIDIA P2P/IPC. Uses fake IPC pointers = pre-
  // allocated SHM). Enables sglang custom allreduce on DLIN (without this, sglang
  // falls back to slow NCCL = 48.5% of decode GPU time per kprof profiling).
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);
  m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool full_nvlink) -> int");
  m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  m.impl("all_reduce", torch::kCUDA, &all_reduce);
  // DL end

  // STUB (no impl): sglang registers a module-level
  // @torch.library.register_fake("sgl_kernel::moe_fused_gate") (topk.py) which
  // requires the op to EXIST. Dense models never call it; the schema-only m.def
  // satisfies the decorator. (Real MoE models need the cutlass-backed kernel —
  // plan §5.) Also stub the kimi_k2 variant for the same reason.
  m.def(
      "moe_fused_gate(Tensor input, Tensor bias, int num_expert_group, int topk_group, int topk, int "
      "num_fused_shared_experts, float routed_scaling_factor, bool apply_routed_scaling_factor_on_output) -> "
      "(Tensor[])");
  m.def(
      "kimi_k2_moe_fused_gate(Tensor input, Tensor bias, int topk, bool renormalize, "
      "float routed_scaling_factor, bool apply_routed_scaling_factor_on_output) -> "
      "(Tensor[])");
}

REGISTER_EXTENSION(common_ops)
// DL end

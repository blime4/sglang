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

// DL begin
// csrc/elementwise/rmsnorm_dl.cu — standalone RMSNorm (no FlashInfer dep).
// Schemas match the python wrappers in sgl_kernel/elementwise.py
// (torch.ops.sgl_kernel.{rmsnorm,fused_add_rmsnorm}.default).
namespace sgl_kernel_dl {
void rmsnorm(torch::Tensor out, torch::Tensor input, torch::Tensor weight,
             double eps, bool enable_pdl);
void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual,
                       torch::Tensor weight, double eps, bool enable_pdl);
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

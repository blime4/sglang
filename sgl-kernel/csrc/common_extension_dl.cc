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

  // csrc/memory/weak_ref_tensor.cpp
  m.def("weak_ref_tensor(Tensor tensor) -> Tensor");
  m.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

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

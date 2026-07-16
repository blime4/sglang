# DL begin — DLIN compiled GDN decode kernel (vLLM _dl_C.dl_recurrent_gated_delta_rule)
#
# Routes sglang GDN single-token decode through vLLM's compiled CUDA op
# torch.ops._dl_C.dl_recurrent_gated_delta_rule (the DLEOL-optimized recurrent
# kernel; pingpong/unroll via DLEOL_FLA_ENABLE_PINGPONG / DLEOL_FLA_UNROLL_COUNT
# env, read by libdleol.so at JIT time). This is the fast path vLLM uses to hit
# ~38 tok/s decode on Qwen3.5/3.6-35B-A3B (sglang's triton GDN is ~1.5x slower).
#
# Mirrors vllm/plugins/dl_platform_plugin/patch/gdn_linear_attn_patch.py +
# ops/_dl_ops.py::recurrent_gated_delta_rule: precompute gates (g, beta) via
# sglang's fused_gdn_gating, then call the op. State layout
# [num_slots, HV, V, K] K-innermost matches vLLM exactly (verified).
# Decode-only MVP; extend/verify stay on triton (separate kernels in the
# dispatcher), so this class intentionally only implements decode().
#
# CRITICAL (DL): the op is built into vLLM's _dl_C.so (dl19-built, loaded via
# fp8_utils._ensure_dl_C). Its DLEOL VM (libdleol.so) MUST be the dl19-matching
# build — i.e. source the DEFAULT `sdk/env.sh`, NOT sdk-0401 (dl24). With
# sdk-0401's dl24 libdleol, dl_recurrent_gated_delta_rule segfaults inside
# dleol::vm::CUInstExecutor::getCuFunction (kernel JIT load). The FP8 GEMM op
# tolerates the dl19/dl24 libdleol mismatch; this FLA VM does not.
# beta is cast to bf16 to match vLLM's fused_gdn_gating output dtype (the op is
# specialized for bf16 beta; sglang's helper returns fp32).
import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


class DLinGDNKernel(LinearAttnKernelBase):
    """GDN decode backed by vLLM's compiled dl_recurrent_gated_delta_rule op."""

    # DLIN uses the non-packed decode path (vLLM forces enable_packed_recurrent_decode
    # =False too): gates are precomputed outside, then the recurrent op runs.
    supports_packed_decode: bool = False

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # _dl_C.so is already loaded by fp8_utils._ensure_dl_C (for the FP8 GEMM);
        # load_library registers ALL ops in the .so, including the FLA ones. This
        # call is a cheap idempotent guard that also raises clearly if absent.
        from sglang.srt.layers.quantization.fp8_utils import _ensure_dl_C
        import os as _dl_os

        _ensure_dl_C()

        # DL: SKIP_GDN for profiling — measures non-GDN GPU time
        if _dl_os.environ.get("SGLANG_DL_SKIP_GDN") == "1":
            return torch.zeros_like(v)

        # Precompute gates exactly like vLLM's dl_fused_sigmoid_gating_delta_rule_update:
        #   g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)
        # sglang's fused_gdn_gating returns (g, beta) each shaped [1, B, HV] fp32.
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        # DL: vLLM's fused_gdn_gating returns beta as bf16 (sigmoid stored in b.dtype);
        # the compiled op is specialized for bf16 beta. sglang returns fp32 beta, which
        # triggers an unsupported DLEOL kernel specialization (getCuFunction segfault).
        # Cast to bf16 to match the op's expected dtype (g stays fp32, as in vLLM).
        beta = beta.to(torch.bfloat16)

        # Op writes output (aliased, empty_like v) and updates ssm_states in place
        # at ssm_state_indices — same in-place semantics as sglang's triton decode.
        output = torch.empty_like(v)  # [1, B, HV, V]

        # vLLM unsqueezes 1-D indices to 2-D (gdn_linear_attn_patch.py:55-56).
        ssm_state_indices = cache_indices
        if ssm_state_indices.ndim == 1:
            ssm_state_indices = ssm_state_indices.unsqueeze(-1)

        scale = k.shape[-1] ** -0.5

        torch.ops._dl_C.dl_recurrent_gated_delta_rule(
            output,
            q,
            k,
            v,
            g,
            beta,
            ssm_states,
            ssm_state_indices,
            query_start_loc,  # cu_seqlens
            None,  # num_accepted_tokens (decode, no spec)
            scale,
            True,  # use_qk_l2norm_in_kernel
        )
        return output

    def extend(self, q, k, v, g, beta, *, ssm_states, cache_indices,
               query_start_loc, **kwargs):
        # DL: route GDN extend (prefill) through vLLM's compiled dl_chunk op
        # instead of sglang's triton chunk_gated_delta_rule (which uses a custom
        # initial_state_indices path that diverges from vLLM → wrong first token).
        # dl_chunk has NO state_indices arg, so gather active states by
        # cache_indices, run, scatter the updated final state back in-place.
        from sglang.srt.layers.quantization.fp8_utils import _ensure_dl_C
        from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
        _ensure_dl_C()
        # beta bf16 to match op specialization (same as decode).
        beta = beta.to(torch.bfloat16)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        # DL: match vLLM DlChunkGatedDeltaRule.forward_native EXACTLY. The op's
        # INTERNAL l2norm path (use_qk_l2norm_in_kernel=True) raises
        # cuDNN_STATUS_NOT_SUPPORTED for the extend (large-T) case on DLIN (it
        # works for decode T=1). Fix: apply l2norm OUTSIDE (triton l2norm_fwd) and
        # pass use_qk_l2norm_in_kernel=False — exactly what vLLM does.
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
        # q/k/v: [1, T, H, D]; ssm_states: [num_slots, HV, V, K].
        active = ssm_states[cache_indices]  # [N, HV, V, K] (copy)
        # DL: vLLM casts initial_state to q.dtype (bf16) — fp32 state → dtype
        # mismatch in the op's internal matmul. cu_seqlens must be int64 (.long()).
        active = active.to(q.dtype)
        scale = k.shape[-1] ** -0.5
        output = torch.empty_like(v)
        cu_seqlens = query_start_loc.to(torch.int64)
        torch.ops._dl_C.dl_chunk_gated_delta_rule(
            output, q, k, v, g, beta, active, cu_seqlens, scale, False)  # False: l2norm done above
        ssm_states[cache_indices] = active.to(ssm_states.dtype)  # scatter back (pool dtype)
        return output, active, None
# DL end

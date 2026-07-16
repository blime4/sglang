# DL begin — FakeTensor (meta) impls for the DLIN _dl_C custom ops, so
# torch.compile/inductor treats them as OPAQUE fast ops instead of decomposing
# them (decomposition balloons the graph: 306s/capture, 82ms decode vs 27ms
# baseline). With these registered, inductor keeps the tuned DLIN kernels
# intact in the compiled graph. Call dl_register_meta() after _ensure_dl_C().
import torch  # noqa: F401


def dl_register_meta():
    from sglang.srt.layers.quantization.fp8_utils import _ensure_dl_C

    _ensure_dl_C()
    try:
        from torch.library import register_fake
    except ImportError:
        try:
            from torch.library import impl_abstract as register_fake
        except ImportError:
            return

    ns = "_dl_C"

    def _try(name, fn):
        try:
            register_fake(f"{ns}::{name}")(fn)
        except Exception:
            pass  # already registered or schema mismatch — skip

    # gptq_dlblas_gemmex(input[M,K], weight[K,N], scale_inv, scale_inv2,
    # quant_type:int, bit:int) -> output[M, N]
    def _gemmex(input, weight, scale_inv, scale_inv2, quant_type, bit):
        return input.new_empty(input.shape[0], weight.shape[1])

    # gemma_rms_norm(out, input, weight, eps) — in-place into out
    def _rms_norm(out, input, weight, eps):
        return None

    # fused_add_gemma_rms_norm(input, residual, weight, eps) — in-place input
    def _fused_add_rms_norm(input, residual, weight, eps):
        return None

    # w8a8_matmul(a, scale_a, b, scale_b, bias, out_dtype) -> [M, N]
    def _w8a8(a, scale_a, b, scale_b, bias, out_dtype):
        return a.new_empty(a.shape[0], b.shape[0])

    _try("gptq_dlblas_gemmex", _gemmex)
    _try("gemma_rms_norm", _rms_norm)
    _try("fused_add_gemma_rms_norm", _fused_add_rms_norm)
    _try("w8a8_matmul", _w8a8)

    # ---- in-place ops (mutate tensors per schema, return None) ----
    # invoke_fused_moe_opt: mutates x,w,y,topk_*,sorted/expert/npp; returns ()
    def _moe_opt(
        x, w, y, w_bias, w_scales, w_zp, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight, top_k, block_size_m, block_size_n, block_size_k,
        use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, use_mxfp4_w4a16,
        block_size, M,
    ):
        return None

    # dl_recurrent_gated_delta_rule: mutates output (and ssm_state in practice)
    def _recurrent_gdr(
        output, q, k, v, g, beta, ssm_state, ssm_state_indices,
        cu_seqlens, num_accepted_tokens, scale, use_qk_l2norm_in_kernel,
    ):
        return None

    def _chunk_gdr(
        output, q, k, v, g, beta, initial_state, cu_seqlens, scale,
        use_qk_l2norm_in_kernel,
    ):
        return None

    def _grouped_topk(
        gating, bias, topk_ids, topk_weights, topk, num_expert_group,
        topk_group,
    ):
        return None

    _try("invoke_fused_moe_opt", _moe_opt)
    _try("invoke_fused_moe_opt_v3", _moe_opt)  # similar signature (+1 arg)
    _try("dl_recurrent_gated_delta_rule", _recurrent_gdr)
    _try("dl_chunk_gated_delta_rule", _chunk_gdr)
    _try("moe_fused_grouped_topk", _grouped_topk)


# DL end

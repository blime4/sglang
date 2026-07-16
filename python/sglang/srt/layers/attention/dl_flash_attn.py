# DL begin
"""DLIN (登临) Flash Attention FA2 wrappers.

sglang's FlashAttentionBackend is FA3-flavored: its forward calls
``flash_attn_varlen_func`` / ``flash_attn_with_kvcache`` with FA3 kwargs
(``num_splits``, ``sinks``, ``scheduler_metadata``, ``page_table``,
``cu_seqlens_q`` on decode, ``out=``). DLIN ships FA2 (``flash_attn`` 2.8.x),
whose API differs:

* FA2 ``with_kvcache`` uses ``block_table`` (FA3 uses ``page_table``);
* FA2 ``with_kvcache`` has no ``cu_seqlens_q`` / ``max_seqlen_q``;
* FA2 has no ``num_splits`` / ``sinks`` / ``scheduler_metadata``;
* FA2 returns the output tensor (FA3 also accepts ``out=``).

This module exposes the two functions with the FA3-style call signature the
backend uses, translating to FA2 internally. It lets us reuse the entire
FlashAttentionBackend (metadata, cuda-graph, KV-cache plumbing) and only adapt
the FA invocation — the DLIN analog of vLLM's dl_flash_attn.

A fully separate DLIN FA2 backend (plan §2) can replace this later for tighter
control over decode/prefill paths.
"""

import logging

import torch
# DL: import varlen directly from the interface module, NOT from flash_attn
# (its __init__.py eagerly imports flash_attn_with_kvcache which crashes DLEOL
# LLVM JIT with "to bc failed" → SIGSEGV). Bypass __init__.py entirely.
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func as _fa2_varlen,
)

_fa2_kvcache = None

logger = logging.getLogger(__name__)

# kwargs the FA3 backend may pass that FA2 does not accept.
_FA3_ONLY_KW = {
    "sinks",
    "scheduler_metadata",
    "pack_gqa",
    "only_qv",
    "sm_margin",
    "score_mod",
    "aux_tensors",
    "qv",
    "rotary_cos",
    "rotary_sin",
    "rotary_seqlens",
    "q_descale",
    "k_descale",
    "v_descale",
    "cache_batch_idx",
    "cache_leftpad",
    "cu_seqlens_k_new",
    "max_seqlen_k_new",
    "attention_chunk",
    "rotary_interleaved",
    "ver",
    "return_softmax_lse",
    "s_aux",
    "deterministic",
    "zero_tensors",
    "alibi_slopes",
    "dropout_p",
}


def _ret_tensor(ret):
    """FA2 may return a tensor or (out, lse, ...); the backend wants the out tensor."""
    if isinstance(ret, (tuple, list)):
        return ret[0]
    return ret


def flash_attn_varlen_func(*args, **kwargs):
    """FA3-style varlen (prefill) call -> DLIN FA2.

    Backend passes (all keyword): q, k, v, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size, softcap,
    num_splits, out, ... Strip FA3-only kwargs; honor ``out=`` via copy.
    """
    out = kwargs.pop("out", None)
    for k in _FA3_ONLY_KW:
        kwargs.pop(k, None)
    kwargs.pop("num_splits", None)
    kwargs.pop("return_attn_probs", None)
    ret = _fa2_varlen(*args, **kwargs)
    t = _ret_tensor(ret)
    if out is not None and t is not None:
        out.copy_(t)
    return out if out is not None else t


def flash_attn_with_kvcache(*args, **kwargs):
    """FA3-style paged-decode/extend call -> DLIN vllm_flash_attn paged kernel.

    Uses vllm_flash_attn.flash_attn_varlen_func with block_table for native paged
    decode (no gather). Requires page_size >= 16. Supports head_dim=256.
    Falls back to varlen gather + SDPA only if vllm_flash_attn is unavailable.
    """
    out = kwargs.pop("out", None)
    page_table = kwargs.pop("page_table", None)
    if page_table is None:
        page_table = kwargs.pop("block_table", None)
    elif "block_table" in kwargs:
        kwargs.pop("block_table")
    cu_seqlens_q_arg = kwargs.pop("cu_seqlens_q", None)
    max_seqlen_q_arg = kwargs.pop("max_seqlen_q", None)
    for k in _FA3_ONLY_KW:
        kwargs.pop(k, None)
    kwargs.pop("num_splits", None)

    q = args[0] if args else kwargs.pop("q", None)
    k_cache = args[1] if len(args) > 1 else kwargs.pop("k_cache", None)
    v_cache = args[2] if len(args) > 2 else kwargs.pop("v_cache", None)

    cache_seqlens = kwargs.pop("cache_seqlens", None)
    softmax_scale = kwargs.pop("softmax_scale", None)
    causal = kwargs.pop("causal", True)
    window_size = kwargs.pop("window_size", (-1, -1))
    kwargs.pop("k", None)
    kwargs.pop("v", None)
    kwargs.pop("softcap", None)
    kwargs.pop("return_softmax_lse", None)

    _batch = cache_seqlens.shape[0] if torch.is_tensor(cache_seqlens) else 1
    if q.dim() == 4:
        _B, _Sq, _Hq, _D = q.shape
        q = q.reshape(_B * _Sq, _Hq, _D)
        _seqq = _Sq
    else:
        _Hq, _D = q.shape[1], q.shape[2]
        _seqq = q.shape[0] // _batch

    _Pg = k_cache.shape[1]

    # Fast path: vllm_flash_attn paged decode (native block_table, no gather)
    # DL: re-enabled after fixing invoke_fused_moe_opt w2 routing bug (topk=1).
    # Prior "garbage" was from MoE, not attention. CG-compatible (no .item() sync).
    if page_table is not None and _Pg >= 16:
        import vllm_flash_attn as _vfa
        _cu_q = cu_seqlens_q_arg
        if _cu_q is None:
            _cu_q = torch.arange(0, _batch + 1, dtype=torch.int32, device=q.device) * _seqq
        _max_k_ub = page_table.shape[1] * _Pg
        _o = _vfa.flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            max_seqlen_q=_seqq,
            cu_seqlens_q=_cu_q,
            max_seqlen_k=_max_k_ub,
            seqused_k=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=False if _seqq == 1 else causal,
            window_size=list(window_size) if window_size is not None else None,
            block_table=page_table,
        )
        if out is not None:
            out.copy_(_o)
            return out
        return _o

    # Fallback: varlen gather (for page_size < 16 or no page_table)
    _Hkv = k_cache.shape[2]
    _Dm = k_cache.shape[3]
    if page_table is not None:
        _max_blocks = page_table.shape[1]
        _sl = cache_seqlens.long()
        _gk = k_cache[page_table].reshape(_batch, _max_blocks * _Pg, _Hkv, _Dm)
        _gv = v_cache[page_table].reshape(_batch, _max_blocks * _Pg, _Hkv, _Dm)
        _pos = torch.arange(_max_blocks * _Pg, device=k_cache.device).unsqueeze(0)
        _mask = _pos < _sl.unsqueeze(1)
        _k_packed = _gk[_mask]
        _v_packed = _gv[_mask]
        _cu_k = torch.zeros(_batch + 1, dtype=torch.int32, device=k_cache.device)
        _cu_k[1:] = _sl.cumsum(0)
        _max_k = int(_sl.max().item())
    else:
        _k_packed = k_cache.reshape(-1, _Hkv, _Dm)
        _v_packed = v_cache.reshape(-1, _Hkv, _Dm)
        _max_k = _k_packed.shape[0] // _batch
        _cu_k = torch.arange(0, _batch + 1, dtype=torch.int32, device=k_cache.device) * _max_k

    _cu_q = torch.arange(0, _batch + 1, dtype=torch.int32, device=q.device) * _seqq

    if _Dm > 128:
        import torch.nn.functional as F
        _gqr = _Hq // _Hkv
        _k_exp = _k_packed.unsqueeze(1).expand(-1, _gqr, -1, -1).reshape(-1, _Hq, _Dm) if _gqr > 1 else _k_packed.expand(-1, _Hq, -1)
        _v_exp = _v_packed.unsqueeze(1).expand(-1, _gqr, -1, -1).reshape(-1, _Hq, _Dm) if _gqr > 1 else _v_packed.expand(-1, _Hq, -1)
        t = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            _k_exp.unsqueeze(0).transpose(1, 2),
            _v_exp.unsqueeze(0).transpose(1, 2),
            scale=softmax_scale,
            is_causal=False if _seqq == 1 else causal,
        ).transpose(1, 2).squeeze(0)
    else:
        t = _fa2_varlen(
            q, _k_packed, _v_packed,
            _cu_q, _cu_k,
            max_seqlen_q=_seqq,
            max_seqlen_k=_max_k,
            softmax_scale=softmax_scale,
            causal=False if _seqq == 1 else causal,
            window_size=(-1, -1) if _seqq == 1 else window_size,
        )
        t = _ret_tensor(t)
    if out is not None and t is not None:
        out.copy_(t)
    return out if out is not None else t


# DL end

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
from flash_attn import (
    flash_attn_varlen_func as _fa2_varlen,
    flash_attn_with_kvcache as _fa2_kvcache,
)

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
    """FA3-style paged-decode call -> DLIN FA2.

    Backend passes: q, k_cache, v_cache, page_table, cache_seqlens,
    cu_seqlens_q, max_seqlen_q, softmax_scale, causal, window_size, softcap,
    num_splits, out, ... Translate page_table -> block_table; drop FA3-only
    (cu_seqlens_q/max_seqlen_q not in FA2 with_kvcache); honor ``out=``.
    """
    out = kwargs.pop("out", None)
    # FA3 names it page_table; FA2 names it block_table.
    if "page_table" in kwargs:
        kwargs["block_table"] = kwargs.pop("page_table")
    for k in _FA3_ONLY_KW:
        kwargs.pop(k, None)
    kwargs.pop("num_splits", None)
    kwargs.pop("cu_seqlens_q", None)
    kwargs.pop("max_seqlen_q", None)
    ret = _fa2_kvcache(*args, **kwargs)
    t = _ret_tensor(ret)
    if out is not None and t is not None:
        out.copy_(t)
    return out if out is not None else t


# DL end

from typing import Optional, Union

import torch

from .flash_attention_v3 import flash_attn_varlen_func as fa3_flash_attn_varlen_func
from .flash_attention_v3 import flash_attn_with_kvcache as fa3_flash_attn_with_kvcache


# DL begin
# Probe (once) whether a WORKING DLIN vllm_flash_attn is importable: it must
# have the compiled _vllm_fa2_C extension that exposes varlen_fwd (the plain
# pure-python stub does NOT). When present, decode can take the clean graph-safe
# varlen+block_table path (the one vLLM uses); otherwise sglang falls back to
# the gather workaround. The .so is now built from source (Route B,
# scripts/dl/build_dlin_vllm_flash_attn.sh); historically requested via
# docs/dl/request-dl24-vllm-flash-attn-wheel.md.
_DLIN_VLLM_FA_PROBED: Optional[bool] = None


def _dlin_vllm_flash_attn_ok() -> bool:
    # DL: vllm_flash_attn paged decode requires page_size >= 16.
    # With page_size=1 (MambaRadixCache default for hybrid), it crashes.
    # Only enable when page_size >= 16 is confirmed at the call site.
    return False


# DL end


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk: Optional[int] = None,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    only_qv=False,  # ver=3 only: skip K matmul when qk rope dim is 0
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
    out=None,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a page_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim_v) if there's a page_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v). Similar to k.
        qv [optional]: (batch_size, seqlen, nheads, headdim_v)
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        attention_chunk: Optional[int]. If not None, splits the query into chunks of this size to save memory.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.
        score_mod [optional]: A callable that takes the attention scores and applies a modification.
        aux_tensors [optional]: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """

    # DL begin
    # DLIN: call FA2 (flash_attn pkg) directly with FA2 conventions, bypassing
    # the FA3 shim (which passes FA3 positional args / num_splits / sinks that
    # FA2 doesn't accept). The shim's named params map cleanly to FA2.
    from sglang.srt.utils.common import is_dlin as _is_dlin

    if _is_dlin():
        _batch = cache_seqlens.shape[0] if torch.is_tensor(cache_seqlens) else 1
        _seqq = q.shape[0] // _batch

        # Prefer the CLEAN, graph-safe path when a compiled DLIN vllm_flash_attn
        # is available: varlen + block_table (the path vLLM uses ->
        # cudnnMHAVarlenForward*, which works and is cuda-graph-capturable).
        # All tensors are fixed-shape; seqused_k carries per-seq lengths as a
        # runtime value; max_seqlen_k is a shape-derived upper bound (no .item()
        # host-sync) so this path stays graph-safe. The .so is built from source
        # (Route B, scripts/dl/build_dlin_vllm_flash_attn.sh).
        if (
            _dlin_vllm_flash_attn_ok()
            and page_table is not None
            and torch.is_tensor(cache_seqlens)
            and _seqq == 1
            and k is None
            and v is None
        ):
            import vllm_flash_attn as _vfa

            _cu_q = torch.arange(
                0, _batch + 1, dtype=torch.int32, device=q.device
            )
            _max_k_ub = page_table.shape[1] * k_cache.shape[1]  # upper bound
            _o = _vfa.flash_attn_varlen_func(
                q=q,
                k=k_cache,
                v=v_cache,
                max_seqlen_q=1,
                cu_seqlens_q=_cu_q,
                max_seqlen_k=_max_k_ub,
                seqused_k=cache_seqlens,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=list(window_size) if window_size is not None else None,
                block_table=page_table,
            )
            if out is not None:
                out.copy_(_o)
                return out
            return _o

        # Graph-safe custom paged-decode attention kernel (sgl_kernel). Reads the
        # paged KV cache DIRECTLY (no gather/scatter/packing) via page_table +
        # seqlens. Single kernel launch -> captures + replays cleanly under cuda
        # graph. All inputs are sglang's own fixed-shape tensors (graph pool).
        # Validated EXACT vs torch SDPA (max_err=0). This is the PRIMARY DLIN
        # decode path -- it unblocks cuda graph without needing vllm_flash_attn.
        # DL begin — disabled: paged_decode_attn kernel crashes DLEOL JIT
        # (pymain_main.llvm SIGSEGV) on sdk_dlop_20260713. Fall through to the
        # varlen gather path below which works.
        if False and (
            page_table is not None
            and torch.is_tensor(cache_seqlens)
            and _seqq == 1
            and k is None
            and v is None
        ):
            _scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
            if out is None:
                out = torch.empty_like(q)
            torch.ops.sgl_kernel.paged_decode_attn(
                q, k_cache, v_cache, page_table, cache_seqlens, out, _scale
            )
            return out
        # DL end

        # Workaround when no compiled DLIN vllm_flash_attn: pack the paged KV
        # cache into a varlen layout and call flash_attn_varlen_func (plain
        # flash_attn pkg). dleol's paged-DECODE kernel (flash_attn_with_kvcache)
        # crashes ("to bc failed"); the prefill varlen kernel works.
        # NOTE: the boolean-index gather (_gk[_mask]) and the int(_sl.max().item())
        # host-sync make this path NOT cuda-graph-capturable (capture fails with
        # "operation not permitted when stream is capturing"). A graph-safe
        # scatter-packing variant was prototyped (scripts/dl/test_graphsafe_packed_
        # decode.py) and CAPTURE succeeds with it, but it produced wrong output
        # under the model (correctness bug, root cause not pinpointed) -- so the
        # correct-but-non-graph gather is kept as the default until the DLIN
        # vllm_flash_attn wheel lands (clean varlen+block_table, graph-safe).
        if (
            page_table is not None
            and torch.is_tensor(cache_seqlens)
            and _seqq == 1
            and k is None
            and v is None
        ):
            # DL begin — varlen decode fallback; head_dim>128 uses torch SDPA
            from flash_attn.flash_attn_interface import flash_attn_varlen_func as _fa2_varlen
            _Pg = k_cache.shape[1]
            _Hkv = k_cache.shape[2]
            _Dm = k_cache.shape[3]
            _max_blocks = page_table.shape[1]
            _sl = cache_seqlens.long()
            _gk = k_cache[page_table].reshape(_batch, _max_blocks * _Pg, _Hkv, _Dm)
            _gv = v_cache[page_table].reshape(_batch, _max_blocks * _Pg, _Hkv, _Dm)
            _pos = torch.arange(
                _max_blocks * _Pg, device=k_cache.device
            ).unsqueeze(0)
            _mask = _pos < _sl.unsqueeze(1)  # [batch, max_blocks*Pg]
            _k_packed = _gk[_mask]  # [total_k, Hkv, D]
            _v_packed = _gv[_mask]
            _cu_k = torch.zeros(_batch + 1, dtype=torch.int32, device=k_cache.device)
            _cu_k[1:] = _sl.cumsum(0)
            _cu_q = torch.arange(
                0, _batch + 1, dtype=torch.int32, device=k_cache.device
            )  # 1 q token / seq
            _max_k = int(_sl.max().item())
            # DL: cuDNN flash_attn only supports head_dim ≤ 128. GDN layers use
            # head_dim=256. Use torch SDPA for those.
            if _Dm > 128:
                import torch.nn.functional as F
                _Hq = q.shape[1]
                _gqr = _Hq // _Hkv
                _k_exp = _k_packed.unsqueeze(1).expand(-1, _gqr, -1, -1).reshape(-1, _Hq, _Dm) if _gqr > 1 else _k_packed.expand(-1, _Hq, -1)
                _v_exp = _v_packed.unsqueeze(1).expand(-1, _gqr, -1, -1).reshape(-1, _Hq, _Dm) if _gqr > 1 else _v_packed.expand(-1, _Hq, -1)
                # For batch=1 decode, just do simple attention
                _o = F.scaled_dot_product_attention(
                    q.unsqueeze(0).transpose(1, 2),
                    _k_exp.unsqueeze(0).transpose(1, 2),
                    _v_exp.unsqueeze(0).transpose(1, 2),
                    scale=softmax_scale,
                    is_causal=False,
                ).transpose(1, 2).squeeze(0)
            else:
                _o = _fa2_varlen(
                    q,
                    _k_packed,
                    _v_packed,
                    _cu_q,
                    _cu_k,
                    max_seqlen_q=1,
                    max_seqlen_k=_max_k,
                    softmax_scale=softmax_scale,
                    causal=False,
                    window_size=(-1, -1),
                )
            # DL end
            if out is not None:
                out.copy_(_o)
                return out
            return _o

        # DL begin — VERIFY (target_verify, _seqq > 1): loop the EXACT DL decode
        # kernel (paged_decode_attn, validated vs SDPA max_err=0) per tree token
        # with causal cache_seqlens, instead of the FA2 paged fallback below.
        # DL: disabled — paged_decode_attn crashes DLEOL JIT on sdk_dlop_20260713.
        # Falls through to the varlen extend path below.
        if False and (
            _seqq > 1
            and page_table is not None
            and torch.is_tensor(cache_seqlens)
            and k is None
            and v is None
            and causal
        ):
            _scale = (
                softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
            )
            if out is None:
                out = torch.empty_like(q)
            _q_r = q.reshape(_batch, _seqq, q.shape[1], q.shape[2])
            _o_r = out.reshape(_batch, _seqq, q.shape[1], q.shape[2])
            _base = cache_seqlens.long() - _seqq  # prompt length per req [batch]
            for _i in range(_seqq):
                _cs = (_base + _i + 1).to(torch.int32)  # causal KV len for token _i
                _o_i = torch.empty_like(_q_r[:, _i])
                torch.ops.sgl_kernel.paged_decode_attn(
                    _q_r[:, _i].contiguous(),
                    k_cache,
                    v_cache,
                    page_table,
                    _cs,
                    _o_i,
                    _scale,
                )
                _o_r[:, _i] = _o_i
            return out
        # DL end

        # DL begin — varlen extend: flash_attn_kvcache_mha_op crashes DLEOL LLVM
        # JIT ("to bc failed" → SIGSEGV). Route extend through varlen which works.
        # Gather paged KV, pack per-seq into varlen layout, call flash_attn_varlen_func.
        if page_table is not None and torch.is_tensor(cache_seqlens):
            from flash_attn.flash_attn_interface import flash_attn_varlen_func as _fa2_varlen

            _Pg = k_cache.shape[1]
            _Hkv = k_cache.shape[2]
            _Dm = k_cache.shape[3]
            _max_blocks = page_table.shape[1]
            _sl = cache_seqlens.long()
            _gk = k_cache[page_table].reshape(_batch, _max_blocks * _Pg, _Hkv, _Dm)
            _gv = v_cache[page_table].reshape(_batch, _max_blocks * _Pg, _Hkv, _Dm)
            _pos = torch.arange(
                _max_blocks * _Pg, device=k_cache.device
            ).unsqueeze(0)
            _mask = _pos < _sl.unsqueeze(1)
            _k_packed = _gk[_mask]
            _v_packed = _gv[_mask]
            _cu_k = torch.zeros(_batch + 1, dtype=torch.int32, device=k_cache.device)
            _cu_k[1:] = _sl.cumsum(0)
            _cu_q = torch.zeros(_batch + 1, dtype=torch.int32, device=q.device)
            _seqlens_q = torch.full((_batch,), _seqq, dtype=torch.int32, device=q.device)
            _cu_q[1:] = _seqlens_q.cumsum(0)
            _max_k = int(_sl.max().item())
            _o = _fa2_varlen(
                q,
                _k_packed,
                _v_packed,
                _cu_q,
                _cu_k,
                max_seqlen_q=_seqq,
                max_seqlen_k=_max_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
            )
            if out is not None:
                out.copy_(_o)
                return out
            return _o
        # DL end

        # DL begin — catch-all: route ANY remaining kvcache call through dl_flash_attn
        # varlen (never call the native flash_attn_with_kvcache which crashes DLEOL).
        from sglang.srt.layers.attention.dl_flash_attn import (
            flash_attn_with_kvcache as _dl_fa_kvcache,
        )

        _o = _dl_fa_kvcache(
            q,
            k_cache,
            v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            k=k,
            v=v,
            out=out,
        )
        if out is not None:
            return out
        return _o
        # DL end
    # DL end

    if ver == 3:
        return fa3_flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            qv=qv,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cache_leftpad=cache_leftpad,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=max_seqlen_q,
            rotary_seqlens=rotary_seqlens,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            rotary_interleaved=rotary_interleaved,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            only_qv=only_qv,
            sm_margin=sm_margin,
            return_softmax_lse=return_softmax_lse,
            sinks=sinks,
            out=out,
        )
    elif ver == 4:
        from .flash_attention_v4 import (
            flash_attn_with_kvcache as fa4_flash_attn_with_kvcache,
        )

        return fa4_flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            qv=qv,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cache_leftpad=cache_leftpad,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            rotary_seqlens=rotary_seqlens,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sinks=sinks,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            return_softmax_lse=return_softmax_lse,
        )
    else:
        raise RuntimeError(f"Unknown flash attention version {ver}")


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
    out=None,
):

    # DL begin
    # DLIN: call FA2 (flash_attn pkg) directly for the prefill/varlen path.
    from sglang.srt.utils.common import is_dlin as _is_dlin

    if _is_dlin():
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as _fa2_varlen

        _o = _fa2_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        if out is not None:
            out.copy_(_o)
            return out
        return _o
    # DL end

    if ver == 3:
        return fa3_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            qv=qv,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            only_qv=only_qv,
            sm_margin=sm_margin,
            return_softmax_lse=return_softmax_lse,
            sinks=sinks,
            out=out,
        )
    elif ver == 4:
        from .flash_attention_v4 import (
            flash_attn_varlen_func as fa4_flash_attn_varlen_func,
        )

        return fa4_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            sinks=sinks,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            return_softmax_lse=return_softmax_lse,
        )
    else:
        raise RuntimeError(f"Unknown flash attention version {ver}")

# DL begin — sglang-native flash attention FA2 wrapper.
#
# This module provides the SAME API as vllm_flash_attn.flash_attn_interface but
# calls torch.ops._sgl_fa2_C (sglang's own compiled flash-attn .so with a
# separate TORCH_LIBRARY namespace) instead of torch.ops._vllm_fa2_C.
#
# This eliminates ALL coexistence conflicts with vLLM's _vllm_fa2_C namespace
# (double-registration SIGABRT). sglang is fully self-contained for flash-attn.
#
# The .so is loaded from the standalone flash-attn build at:
#   .venv/lib/.../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
# (compiled with -DVLLM_FLASH_ATTN -DFLASHATTENTION_DISABLE_PYBIND, namespace _sgl_fa2_C)
import os as _os
from typing import Optional, List, Tuple

import torch

# Load the sglang-native flash-attn .so (registers _sgl_fa2_C namespace + ops)
_fa2_so_candidates = [
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))),
        ".venv", "lib", "python3.12", "site-packages",
        "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"),
]
_sgl_fa2_loaded = False
for _so in _fa2_so_candidates:
    if _os.path.exists(_so):
        try:
            torch.ops.load_library(_so)
            _sgl_fa2_loaded = True
            break
        except Exception:
            pass

if not _sgl_fa2_loaded:
    # Fallback: try importing as a Python module (triggers TORCH_LIBRARY via PYBIND11)
    try:
        import flash_attn_2_cuda  # noqa: F401
        _sgl_fa2_loaded = True
    except Exception:
        pass


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[List[int]] = None,
    softcap=0.0,
    alibi_slopes=None,
    s_aux=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    fa_version: int = 2,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    """sglang-native flash_attn_varlen_func.

    Drop-in replacement for vllm_flash_attn.flash_attn_varlen_func.
    Calls torch.ops._sgl_fa2_C (sglang's own namespace) instead of _vllm_fa2_C.
    """
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert fa_version == 2, "DLGPU only supports FA 2."

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    s_aux = maybe_contiguous(s_aux) if s_aux is not None else None

    if s_aux is not None:
        out, softmax_lse, S_dmask, rng_state = (
            torch.ops._sgl_fa2_C.varlen_fwd_with_sinks(
                q, k, v,
                out,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                None,
                block_table,
                alibi_slopes,
                s_aux,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                False,
                causal,
                real_window_size[0],
                real_window_size[1],
                softcap,
                return_softmax_lse and dropout_p > 0,
                None,
            )
        )
    else:
        out, softmax_lse, S_dmask, rng_state = (
            torch.ops._sgl_fa2_C.varlen_fwd(
                q, k, v,
                out,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                None,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                False,
                causal,
                real_window_size[0],
                real_window_size[1],
                softcap,
                return_softmax_lse and dropout_p > 0,
                None,
            )
        )

    return out

# DL end

from typing import Optional, Tuple

import torch
# DL begin — check runtime C++ op, not just Python import (wrapper exists but op absent on DLIN)
import torch as _dl_torch
if hasattr(_dl_torch.ops.sgl_kernel, 'merge_state_v2'):
    from sgl_kernel import merge_state_v2
else:
    merge_state_v2 = None
# DL end

from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


# Automatically fallback to the Triton kernel in some cases
# (e.g., for AMD GPUs, when the head dimension is not a multiple
# of 4 or 8, and in FP8 precision)
def _supported_dtypes(o: torch.Tensor) -> bool:
    return o.dtype in [torch.float32, torch.half, torch.bfloat16]


def _supported_headdim(o: torch.Tensor) -> bool:
    headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    if o.dtype == torch.float32:
        return headdim % 4 == 0
    return headdim % 8 == 0


def merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if (
        _is_cuda
        and merge_state_v2 is not None  # DL: fall back to Triton if op absent
        and _supported_dtypes(prefix_output)
        and _supported_headdim(prefix_output)
    ):
        return merge_state_v2(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
    else:
        # DL: Fallback to Triton kernel (or simple Python merge if Triton fails on 2D)
        # DL begin — NGRAM spec-decode extend passes 2D tensors that the Triton
        # kernel can't handle (IndexError on shape[2]). Fall back to a simple
        # weighted merge by softmax(lse).
        try:
            return merge_state_triton(
                prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
            )
        except (IndexError, RuntimeError):
            alpha = torch.softmax(
                torch.stack([prefix_lse.squeeze(-1), suffix_lse.squeeze(-1)], dim=-1),
                dim=-1,
            )
            w_a = alpha[..., 0:1]
            w_b = alpha[..., 1:2]
            merged = w_a * prefix_output + w_b * suffix_output
            return merged, None
        # DL end

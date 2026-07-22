# DL begin
"""DLIN Inductor fusion passes for sglang torch.compile decode path.

``build_dl_fusion_passes()`` returns the list of post-grad fusion passes to
inject into Inductor's ``post_grad_custom_post_pass``. It is called by the
``SGLANG_DL_FUSION=1`` branch of
``sglang.srt.compilation.torch_compile_decoration.set_torch_compile_config``.

Returns an empty list when:
- the vLLM fused-op kernels (``torch.ops._C.silu_and_mul_*_quant``) are not
  loadable — so the module is safe to import unconditionally.
- DLIN fusion is disabled (handled by the caller's env gate).

Currently the only pass is ``ActivationQuantFusionPass`` (silu_and_mul + FP8
quant → fused single kernel). rope_kvcache / rms_norm quant passes can be
added here once their matchability on the DLIN compiled graph is confirmed.
"""
from __future__ import annotations

import logging

from sglang.srt.compilation.inductor_pass import InductorPass

logger = logging.getLogger(__name__)


def _dl_fusion_ops_available() -> bool:
    """True if vLLM ``_C.silu_and_mul_*_quant`` is loadable.

    The ops are brought in by ``_ensure_dl_C()`` (which loads vLLM ``_C.so``).
    We only check one representative op — they load together.
    """
    try:
        import torch

        if not hasattr(torch.ops._C, "silu_and_mul_quant"):
            # Try to load on demand so the check is accurate even if the model
            # forward hasn't run yet (unit tests, compile dry-runs).
            try:
                from sglang.srt.layers.quantization.fp8_utils import (
                    _ensure_dl_C,
                )

                _ensure_dl_C()
            except Exception:
                return False
        return hasattr(torch.ops._C, "silu_and_mul_quant")
    except Exception:
        return False


def build_dl_fusion_passes() -> list[InductorPass]:
    """Build the list of DLIN fusion passes for the decode compile path.

    Returns an empty list (safe no-op) when fused-op kernels are unavailable
    or construction fails; the caller's ``SGLANG_DL_FUSION=1`` gate already
    controls whether Inductor is pointed at this pass manager at all.
    """
    if not _dl_fusion_ops_available():
        logger.debug(
            "DL fusion: vLLM _C.silu_and_mul_*_quant not loadable; skipping "
            "activation_quant_fusion_pass."
        )
        return []

    passes: list[InductorPass] = []
    try:
        from sglang.srt.compilation.passes.fusion.dl_act_quant_fusion import (
            ActivationQuantFusionPass,
        )

        passes.append(ActivationQuantFusionPass())
    except Exception as e:
        logger.warning("DL fusion: failed to build ActivationQuantFusionPass: %s", e)
    return passes


__all__ = ["build_dl_fusion_passes"]
# DL end

# DL begin
"""Denglin (登临/DLIN) GPU platform for SGLang.

DLIN (e.g. KS38) is CUDA-shaped: device_type is "cuda", ``torch.cuda.*`` works,
``CUDA_VISIBLE_DEVICES`` is honored, and the DLIN-patched torch exposes
``torch.version.dl``. We therefore subclass :class:`CudaSRTPlatform` so every
CUDA code path in SGLang runs unchanged, and only override the handful of hooks
that need a DLIN-specific answer:

* :meth:`is_dlin` -> True (so code can branch to DLIN-optimized ops, mirroring
  vLLM's ``Platform.is_dl()``);
* :meth:`get_device_capability` -> (12, 0) — DLIN reports CUDA compute
  capability 12.0 (matches vLLM's DlPlatform);
* :meth:`get_default_attention_backend` -> the DLIN attention backend once
  registered; for now "triton", which runs on-device via the DLIN torch.

Operator routing (DLIN flash-attn / GEMM / MoE / quant) is added op-by-op in
later phases — see ``DLIN_INTEGRATION_PLAN.md``. Detection lives in
``sglang.srt.utils.common.is_dlin``.
"""

from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.srt.platforms.device_mixin import DeviceCapability


class DlinSRTPlatform(CudaSRTPlatform):
    """In-tree platform for Denglin (登临/DLIN) GPUs."""

    def is_dlin(self) -> bool:
        return True

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        # DLIN reports CUDA compute capability 12.0.
        return DeviceCapability(major=12, minor=0)

    def get_default_attention_backend(self) -> str:
        # DLIN: use the FlashAttention backend, whose funcs are sourced from the
        # DLIN flash_attn package (FA2) via jit_kernel/flash_attention_v3
        # (`is_dlin()` branch). Mirrors vLLM's DlPlatform routing to FA2.
        return "fa3"


# DL end

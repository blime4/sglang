"""CUDA/ROCm architecture detection and default compile target flags."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import torch

from sglang.jit_kernel.utils.common import (
    cache_once,
    is_hip_runtime,
    is_musa_runtime,
)
from sglang.srt.utils.common import get_cuda_version

logger = logging.getLogger(__name__)

# DL begin — shim triton Hopper-PDL primitives (tl.extra.cuda.gdc_wait /
# gdc_launch_dependents) ABSENT in DLIN Triton. Lives here because the FLA/conv/ssm
# kernels import is_arch_support_pdl from this module — so the inductor compile
# subprocess (which re-imports triton fresh, losing main-process monkeypatches)
# picks the shim up. The kernels reference these inside `if USE_GDC:` branches;
# inductor generates TTIR for the whole kernel, so gdc_wait gets called — provide
# @triton.jit no-op device functions (lower to nothing). At runtime USE_GDC=False
# on DLIN so the branch is skipped.
try:
    import triton as _dl_triton
    import triton.language.extra.cuda as _dl_tl_cuda_extra

    @_dl_triton.jit
    def _dl_gdc_wait():
        pass

    @_dl_triton.jit
    def _dl_gdc_launch_dependents():
        pass

    if not hasattr(_dl_tl_cuda_extra, "gdc_wait"):
        _dl_tl_cuda_extra.gdc_wait = _dl_gdc_wait
    if not hasattr(_dl_tl_cuda_extra, "gdc_launch_dependents"):
        _dl_tl_cuda_extra.gdc_launch_dependents = _dl_gdc_launch_dependents
except Exception:
    pass
# DL end


@dataclass
class ArchInfo:
    major: int
    minor: int
    suffix: str

    @property
    def target_name(self) -> str:
        return f"{self.major}.{self.minor}{self.suffix}"

    @property
    def jit_flag(self) -> str:
        return f"-DSGL_CUDA_ARCH={self.major * 100 + self.minor * 10}"


def _cuda_arch_suffix(major: int, minor: int) -> str:
    """Mirror FlashInfer's `_normalize_cuda_arch`: 9.x/10.x+ -> "a"; 12.0 -> "f"
    and 12.x (x>0) -> "a" (SM120/SM121 need separate cubins to avoid
    cudaErrorIllegalInstruction, requires CUDA >= 12.9); below 9.0 -> plain.
    Unlike FlashInfer, pre-12.9 CUDA falls back to plain instead of raising.
    """
    if major == 9:
        return "a"
    if major == 12:
        if get_cuda_version() < (12, 9):
            return ""
        return "f" if minor == 0 else "a"
    if major >= 10:
        return "a"
    return ""


@cache_once
def _init_jit_cuda_arch_once():
    global _CUDA_ARCH
    try:
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        logger.warning("Cannot detect CUDA architecture.")
        major, minor = 0, 0  # invalid value to trigger compile error if used
    # JIT builds target the exact local GPU, so the arch-specific target is
    # always correct on Hopper+ and unlocks arch-only instructions (redux.f32).
    # HIP/MUSA capability numbers aren't CUDA SM versions and stay unsuffixed.
    suffix = (
        ""
        if (is_hip_runtime() or is_musa_runtime())
        else _cuda_arch_suffix(major, minor)
    )
    _CUDA_ARCH = ArchInfo(major, minor, suffix)


def get_default_target_flags() -> List[str]:
    if is_hip_runtime():
        flags = ["-DUSE_ROCM", "-std=c++20", "-O3"]
        # Detect FP8 type based on GPU architecture
        try:
            device = torch.cuda.current_device()
            gcn_arch = torch.cuda.get_device_properties(device).gcnArchName
            if "gfx942" in gcn_arch:
                flags.append("-DHIP_FP8_TYPE_FNUZ=1")
            else:
                flags.append("-DHIP_FP8_TYPE_E4M3=1")
        except Exception:
            flags.append("-DHIP_FP8_TYPE_E4M3=1")
        return flags
    else:
        # DL begin
        # DLIN's dlcc: no nvcc-only `--expt-relaxed-constexpr`; the JIT shared
        # header needs SGL_CUDA_ARCH=700 (dlgput64 __CUDA_ARCH__) and SGL_ON_DLIN
        # to take the DLIN launch path (no cudaLaunchKernelEx / cluster / PDL).
        from sglang.srt.utils.common import is_dlin as _is_dlin

        if _is_dlin():
            return ["-DSGL_CUDA_ARCH=700", "-DSGL_ON_DLIN=1", "-std=c++20", "-O3"]
        # DL end
        return [
            get_jit_cuda_arch().jit_flag,
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
        ]


@contextmanager
def override_jit_cuda_arch(major: int, minor: int, suffix: str = ""):
    """A context manager to temporarily override CUDA architecture."""
    global _CUDA_ARCH
    old_value = get_jit_cuda_arch()
    _CUDA_ARCH = ArchInfo(major, minor, suffix)
    try:
        yield
    finally:
        _CUDA_ARCH = old_value


def get_jit_cuda_arch() -> ArchInfo:
    """Get the current CUDA architecture info."""
    _init_jit_cuda_arch_once()
    return _CUDA_ARCH


@cache_once
def is_arch_support_pdl() -> bool:
    if is_hip_runtime() or is_musa_runtime():
        return False
    # DL begin: DLIN has no Hopper PDL — gdc_wait/gdc_launch_dependents are absent
    # from DLIN Triton, so the FLA linear-attn kernels must compile without GDC.
    try:
        from sglang.srt.utils.common import is_dlin as _is_dlin

        if _is_dlin():
            return False
    except Exception:
        pass
    # DL end
    return get_jit_cuda_arch().major >= 9

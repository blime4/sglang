# DL begin
"""Matcher helpers for DLIN Inductor act_quant fusion.

Minimal port of vLLM's ``matcher_utils.py`` — only the two matchers needed by
``dl_act_quant_fusion.ActivationQuantFusionPass`` are included:

- ``MatcherSiluAndMul`` — matches ``torch.ops._C.silu_and_mul`` (custom) or its
  PyTorch-native decomposition ``F.silu(x[...,:d]) * x[...,d:]``.
- ``MatcherQuantFP8`` — matches the three FP8 quant ops used on the act-quant
  fusion path (static / dynamic-per-token / per-token-group).

The vLLM ``QuantKey`` dataclass system is collapsed to a plain string
(``QuantScheme``) covering only the FP8 schemes DLIN cares about. ROCm AITER
and nvfp4 branches from vLLM are intentionally dropped (not applicable).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._ops import OpOverload

# FP8 quant op registry, keyed by scheme name. Mirrors vLLM's QUANT_OPS dict
# (rms_quant_fusion.py / matcher_utils.py) but restricted to the FP8 ops that
# _ensure_dl_C() loads from vLLM's _C.so on DLIN.
QUANT_OPS: dict[str, OpOverload] = {
    "fp8_static": torch.ops._C.static_scaled_fp8_quant.default,
    "fp8_dynamic_per_token": torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,
    "fp8_dynamic_group": torch.ops._C.per_token_group_fp8_quant.default,
}

SILU_MUL_OP = torch.ops._C.silu_and_mul.default

FP8_DTYPE = torch.float8_e4m3fn


class MatcherCustomOp(ABC):
    """Base class for matchers that can present either a custom or native forward.

    Port of vLLM's ``MatcherCustomOp``. ``forward`` dispatches to
    ``forward_custom`` (custom-op form, what the compiled graph contains when
    the custom op is registered) or ``forward_native`` (PyTorch decomposition,
    what dynamo may have lowered the op into).
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.forward = self.forward_custom if enabled else self.forward_native

    @abstractmethod
    def forward_custom(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def forward_native(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @staticmethod
    def empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {"dtype": torch.bfloat16, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    @staticmethod
    def empty_f32(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {"dtype": torch.float32, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    def inputs(self) -> list[torch.Tensor]:
        raise NotImplementedError


class MatcherSiluAndMul(MatcherCustomOp):
    """Matcher for SiluAndMul: ``F.silu(x[...,:d]) * x[...,d:]`` (native) or
    ``torch.ops._C.silu_and_mul`` (custom)."""

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled)

    def inputs(self) -> list[torch.Tensor]:
        return [self.empty(5, 4)]

    def forward_custom(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        result = auto_functionalized(SILU_MUL_OP, result=out, input=x)
        return result[1]

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


class MatcherQuantFP8(MatcherCustomOp):
    """Matcher for FP8 quant, parameterized by scheme.

    Supports the three FP8 schemes used by act_quant fusion on DLIN:
    - ``fp8_static``: ``static_scaled_fp8_quant`` (scale is a parameter)
    - ``fp8_dynamic_per_token``: ``dynamic_per_token_scaled_fp8_quant``
    - ``fp8_dynamic_group``: ``per_token_group_fp8_quant`` (group_size 128/64)
    """

    def __init__(
        self,
        quant_scheme: str,
        group_size: int = 128,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled)
        assert quant_scheme in QUANT_OPS, f"unsupported scheme {quant_scheme}"
        self.quant_scheme = quant_scheme
        self.group_size = group_size
        self.QUANT_OP = QUANT_OPS[quant_scheme]

    def forward_custom(
        self, input: torch.Tensor, scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.empty(input.shape, device=input.device, dtype=FP8_DTYPE)

        if self.quant_scheme == "fp8_static":
            assert scale is not None
            _, result = auto_functionalized(
                self.QUANT_OP, result=result, input=input, scale=scale
            )
            return result, scale

        if self.quant_scheme == "fp8_dynamic_per_token":
            assert scale is None
            scale = torch.empty(
                (input.shape[0], 1), device=input.device, dtype=torch.float32
            )
            _, result, scale = auto_functionalized(
                self.QUANT_OP, result=result, input=input, scale=scale, scale_ub=None
            )
            return result, scale

        # fp8_dynamic_group
        if scale is None:
            scale = self._make_group_scale(input)
        finfo = torch.finfo(FP8_DTYPE)
        _, result, scale = auto_functionalized(
            self.QUANT_OP,
            input=input,
            output_q=result,
            output_s=scale,
            group_size=self.group_size,
            eps=1e-10,
            fp8_min=finfo.min,
            fp8_max=finfo.max,
            scale_ue8m0=False,
            dummy_is_scale_transposed=False,
            dummy_is_tma_aligned=False,
        )
        return result, scale

    def forward_native(
        self, input: torch.Tensor, scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Native reference decomposition (used when custom op is disabled).
        # Matches vLLM QuantFP8.forward_native per-token path closely enough
        # for pattern tracing; not numerically verified against the kernel.
        if self.quant_scheme == "fp8_static":
            assert scale is not None
            out = (input.float() * scale.float().reciprocal()).clamp(
                torch.finfo(FP8_DTYPE).min, torch.finfo(FP8_DTYPE).max
            ).to(FP8_DTYPE)
            return out, scale
        x_max = input.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        x_max = x_max.clamp(min=1e-12)
        scale = (x_max / torch.finfo(FP8_DTYPE).max).to(torch.float32)
        out = (input.float() * scale.reciprocal()).clamp(
            torch.finfo(FP8_DTYPE).min, torch.finfo(FP8_DTYPE).max
        ).to(FP8_DTYPE)
        return out, scale

    def _make_group_scale(self, input: torch.Tensor) -> torch.Tensor:
        gs = self.group_size
        rows = input.shape[0]
        cols = input.shape[-1]
        return torch.empty(
            (rows, cols // gs), device=input.device, dtype=torch.float32
        )

    def empty_f32(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {"dtype": torch.float32, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    def inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 16)
        if self.quant_scheme == "fp8_static":
            return [input, self.empty_f32(1, 1)]
        return [input]
# DL end

# DL begin
"""DLIN Inductor pass: fuse ``silu_and_mul → FP8 quant`` into one kernel.

Port of vLLM's ``act_quant_fusion.py`` (``ActivationQuantFusionPass``) onto the
sglang pattern-matcher base. When the post-grad FX graph contains an adjacent
pair ``silu_and_mul(x) → fp8_quant(...)`` the pass rewrites it to a single
``torch.ops._C.silu_and_mul_quant`` (static scale) or
``torch.ops._C.silu_and_mul_per_block_quant`` (per-token-group) node, removing
one kernel launch and one memory pass.

Two patterns are registered (mirror vLLM on ``is_cuda_alike``):

- ``SiluMulFp8StaticQuantPattern`` — ``silu_and_mul + static_scaled_fp8_quant``
  → ``silu_and_mul_quant``.
- ``SiluMulBlockQuantPattern`` — ``silu_and_mul + per_token_group_fp8_quant``
  → ``silu_and_mul_per_block_quant`` (group_size 128 and 64).

The per-block pattern is parameterized on ``is_scale_transposed`` and
``is_e8m0`` (vLLM also iterates ``is_tma_aligned``; DLIN has no TMA so that
axis is collapsed to ``False``). The custom-op (``MatcherCustomOp.enabled``)
form is what the compiled graph contains when the custom op is registered;
the native form covers the dynamo-lowered decomposition.
"""
from __future__ import annotations

import itertools
import logging
from typing import Any

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from sglang.srt.compilation.passes.fusion.dl_matcher_utils import (
    FP8_DTYPE,
    MatcherQuantFP8,
    MatcherSiluAndMul,
)
from sglang.srt.compilation.passes.fusion.dl_pattern_matcher import (
    SGLangFusionPatternMatcherPass,
    SGLangPatternReplacement,
)

logger = logging.getLogger(__name__)

FUSED_OPS: dict[str, Any] = {
    "fp8_static": torch.ops._C.silu_and_mul_quant.default,
    "fp8_dynamic_group": torch.ops._C.silu_and_mul_per_block_quant.default,
}


class ActivationQuantPattern(SGLangPatternReplacement):
    """Base class for silu_and_mul + FP8 quant fusions. Not used directly."""

    def __init__(self, quant_scheme: str, group_size: int = 128) -> None:
        self.quant_scheme = quant_scheme
        self.quant_dtype = FP8_DTYPE
        assert quant_scheme in FUSED_OPS, f"no fused op for scheme {quant_scheme}"
        self.FUSED_OP = FUSED_OPS[quant_scheme]
        self.group_size = group_size
        self.silu_and_mul_matcher = MatcherSiluAndMul()
        self.quant_matcher = MatcherQuantFP8(quant_scheme, group_size=group_size)

    def empty_quant(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {"dtype": self.quant_dtype, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)


class SiluMulFp8StaticQuantPattern(ActivationQuantPattern):
    """``silu_and_mul + static_scaled_fp8_quant`` → ``silu_and_mul_quant``."""

    def __init__(self) -> None:
        super().__init__("fp8_static")

    def get_inputs(self) -> list[torch.Tensor]:
        scale = self.quant_matcher.inputs()[1]
        return [*self.silu_and_mul_matcher.inputs(), scale]

    @property
    def pattern(self):
        def _pattern(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            result_silu_mul = self.silu_and_mul_matcher(input)
            result_quant = self.quant_matcher(result_silu_mul, scale)
            return result_quant[0]

        return _pattern

    @property
    def replacement(self):
        def _replacement(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            d = input.shape[-1] // 2
            output_shape = input.shape[:-1] + (d,)
            result = torch.empty(
                output_shape, device=input.device, dtype=self.quant_dtype
            )
            at = auto_functionalized(
                self.FUSED_OP, result=result, input=input, scale=scale
            )
            return at[1]

        return _replacement


class SiluMulBlockQuantPattern(ActivationQuantPattern):
    """``silu_and_mul + per_token_group_fp8_quant`` → ``silu_and_mul_per_block_quant``.

    Parameterized on ``is_scale_transposed`` and ``is_e8m0`` for the different
    scale layouts the upstream graph may emit. ``is_tma_aligned`` is pinned to
    False (DLIN has no TMA).
    """

    def __init__(
        self,
        group_size: int = 128,
        is_scale_transposed: bool = False,
        is_e8m0: bool = False,
    ) -> None:
        super().__init__("fp8_dynamic_group", group_size=group_size)
        # Rebuild quant_matcher with the right layout flags (the base matcher
        # in dl_matcher_utils always uses non-transposed ue8m0=False; the
        # layout only affects the replacement's scale allocation shape).
        self.is_scale_transposed = is_scale_transposed
        self.is_e8m0 = is_e8m0

    def get_inputs(self) -> list[torch.Tensor]:
        scale = self.quant_matcher.empty_f32(1, 1)
        return self.silu_and_mul_matcher.inputs() + [scale]

    @property
    def pattern(self):
        def _pattern(
            input: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            silu_out = self.silu_and_mul_matcher(input)
            result = torch.empty(
                silu_out.shape, device=silu_out.device, dtype=self.quant_dtype
            )
            assert scale is not None
            finfo = torch.finfo(self.quant_dtype)
            _, result, scale = auto_functionalized(
                self.quant_matcher.QUANT_OP,
                input=silu_out,
                output_q=result,
                output_s=scale,
                group_size=self.group_size,
                eps=1e-10,
                fp8_min=finfo.min,
                fp8_max=finfo.max,
                scale_ue8m0=self.is_e8m0,
                dummy_is_scale_transposed=self.is_scale_transposed,
                dummy_is_tma_aligned=False,
            )
            return result, scale

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            input: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            d = input.shape[-1] // 2
            output_shape = input.shape[:-1] + (d,)
            result = torch.empty(
                output_shape, device=input.device, dtype=self.quant_dtype
            )
            if self.is_scale_transposed:
                scale = torch.empty(
                    (d // self.group_size, input.shape[0]),
                    device=input.device,
                    dtype=torch.float32,
                ).permute(-1, -2)
            else:
                scale = torch.empty(
                    (input.shape[0], d // self.group_size),
                    device=input.device,
                    dtype=torch.float32,
                )
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                input=input,
                scales=scale,
                group_size=self.group_size,
                scale_ub=None,
                is_scale_transposed=self.is_scale_transposed,
            )
            return at[1], at[2]

        return _replacement


class ActivationQuantFusionPass(SGLangFusionPatternMatcherPass):
    """Fuse ``silu_and_mul + fp8_quant`` into ``silu_and_mul_*_quant``.

    Singleton-style: patterns can only be registered once per process (a
    torch pattern-matcher limitation vLLM also documents).
    """

    def __init__(self) -> None:
        super().__init__("dl_activation_quant_fusion_pass")
        self.register(SiluMulFp8StaticQuantPattern())
        for group_size, is_scale_transposed, is_e8m0 in itertools.product(
            [128, 64], [False, True], [False, True]
        ):
            self.register(
                SiluMulBlockQuantPattern(
                    group_size=group_size,
                    is_scale_transposed=is_scale_transposed,
                    is_e8m0=is_e8m0,
                )
            )
# DL end

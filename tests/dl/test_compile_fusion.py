# DL begin
"""Unit tests for DLIN Inductor fusion passes (no GPU required).

These tests run under FakeTensorMode so the ``device="cuda"`` tensors used by
the matcher helpers never allocate. They verify the pattern matcher rewrites
adjacent ``silu_and_mul → fp8_quant`` pairs into the single fused
``torch.ops._C.silu_and_mul_*_quant`` node — i.e. that the PORTED pass works
on a synthetic FX graph. They do NOT prove the pattern fires on sglang's
actual compiled decode graph (that requires a GPU + model run; see
scripts/dl/compile_check.py under SGLANG_DL_FUSION=1).

Run (no GPU):
  source sdk-dlop-07-13-20-30/env.sh
  python -m pytest tests/dl/test_compile_fusion.py -xvs
"""
from __future__ import annotations

import os

# FakeTensorMode + tracing must happen with the vLLM fused-op kernels loaded,
# otherwise torch.ops._C.silu_and_mul_quant is a bare OpOverload with no meta
# impl and tracing fails. Load before importing the pass module.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang.srt.layers.quantization.fp8_utils import _ensure_dl_C

_ensure_dl_C()

from sglang.srt.compilation.passes.fusion.dl_act_quant_fusion import (
    ActivationQuantFusionPass,
)


def _build_silu_quant_graph(
    input_shape: tuple[int, ...],
    quant_scheme: str,
    group_size: int = 128,
) -> torch.fx.GraphModule:
    """Trace a synthetic ``silu_and_mul → fp8_quant`` FX graph under fake mode.

    The graph is traced via ``pm.fwd_only`` so the result is canonicalized the
    same way the pattern is (``aten.empty.memory_format`` etc.). Direct
    ``torch.fx.symbolic_trace`` cannot trace ``auto_functionalized`` (a
    HigherOrderOperator), and hand-built FX graphs miss the canonicalization
    that ``fwd_only`` applies, so the matcher won't fire on them.

    ``quant_scheme`` selects which quant op to emit:
    - ``fp8_static``        → static_scaled_fp8_quant
    - ``fp8_dynamic_per_token`` → dynamic_per_token_scaled_fp8_quant
    - ``fp8_dynamic_group`` → per_token_group_fp8_quant
    """
    import torch._inductor.pattern_matcher as pm

    SILU_MUL_OP = torch.ops._C.silu_and_mul.default
    quant_ops = {
        "fp8_static": torch.ops._C.static_scaled_fp8_quant.default,
        "fp8_dynamic_per_token": torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,
        "fp8_dynamic_group": torch.ops._C.per_token_group_fp8_quant.default,
    }
    assert quant_scheme in quant_ops, quant_scheme
    QUANT_OP = quant_ops[quant_scheme]

    def target_fn(x: torch.Tensor, scale: torch.Tensor):
        d = x.shape[-1] // 2
        out_shape = x.shape[:-1] + (d,)
        silu_out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        silu_out = auto_functionalized(SILU_MUL_OP, result=silu_out, input=x)[1]

        fp8_out = torch.empty(
            silu_out.shape, dtype=torch.float8_e4m3fn, device=silu_out.device
        )
        if quant_scheme == "fp8_static":
            q = auto_functionalized(
                QUANT_OP, result=fp8_out, input=silu_out, scale=scale
            )[1]
            return q, scale
        if quant_scheme == "fp8_dynamic_per_token":
            dyn_scale = torch.empty(
                (silu_out.shape[0], 1), dtype=torch.float32, device=silu_out.device
            )
            at_q = auto_functionalized(
                QUANT_OP,
                result=fp8_out,
                input=silu_out,
                scale=dyn_scale,
                scale_ub=None,
            )
            return at_q[1], at_q[2]
        # fp8_dynamic_group
        finfo = torch.finfo(torch.float8_e4m3fn)
        group_scale = torch.empty(
            (silu_out.shape[0], silu_out.shape[-1] // group_size),
            dtype=torch.float32,
            device=silu_out.device,
        )
        _, q_out, s_out = auto_functionalized(
            QUANT_OP,
            input=silu_out,
            output_q=fp8_out,
            output_s=group_scale,
            group_size=group_size,
            eps=1e-10,
            fp8_min=finfo.min,
            fp8_max=finfo.max,
            scale_ue8m0=False,
            dummy_is_scale_transposed=False,
            dummy_is_tma_aligned=False,
        )
        return q_out, s_out

    with FakeTensorMode():
        x = torch.empty(input_shape, dtype=torch.bfloat16, device="cuda")
        if quant_scheme == "fp8_static":
            scale = torch.empty((1, 1), dtype=torch.float32, device="cuda")
        else:
            scale = torch.empty((1, 1), dtype=torch.float32, device="cuda")
        gm = pm.fwd_only(target_fn, (x, scale))
    return gm


def _count_auto_functionalized_targets(gm: torch.fx.GraphModule) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is auto_functionalized:
            inner = node.args[0] if node.args else None
            name = getattr(inner, "_schema", None)
            key = str(inner) if name is None else inner._schema.name
            counts[key] = counts.get(key, 0) + 1
    return counts


def test_static_quant_fusion_fires():
    """silu_and_mul + static_scaled_fp8_quant → silu_and_mul_quant (one node)."""
    gm = _build_silu_quant_graph((5, 4), "fp8_static")
    before = _count_auto_functionalized_targets(gm)
    assert before.get("_C::silu_and_mul", 0) == 1, before
    assert before.get("_C::static_scaled_fp8_quant", 0) == 1, before
    assert "_C::silu_and_mul_quant" not in before, before

    ActivationQuantFusionPass()(gm.graph)

    after = _count_auto_functionalized_targets(gm)
    assert after.get("_C::silu_and_mul_quant", 0) == 1, after
    assert "_C::silu_and_mul" not in after, after
    assert "_C::static_scaled_fp8_quant" not in after, after
    print(f"\n[test] static_quant fusion: before={before} after={after}")


def test_block_quant_fusion_fires():
    """silu_and_mul + per_token_group_fp8_quant → silu_and_mul_per_block_quant."""
    gm = _build_silu_quant_graph((5, 256), "fp8_dynamic_group", group_size=128)
    before = _count_auto_functionalized_targets(gm)
    assert before.get("_C::silu_and_mul", 0) == 1, before
    assert before.get("_C::per_token_group_fp8_quant", 0) == 1, before
    assert "_C::silu_and_mul_per_block_quant" not in before, before

    ActivationQuantFusionPass()(gm.graph)

    after = _count_auto_functionalized_targets(gm)
    assert after.get("_C::silu_and_mul_per_block_quant", 0) == 1, after
    assert "_C::silu_and_mul" not in after, after
    assert "_C::per_token_group_fp8_quant" not in after, after
    print(f"\n[test] block_quant fusion: before={before} after={after}")


def test_build_dl_fusion_passes_returns_pass():
    """build_dl_fusion_passes() returns ActivationQuantFusionPass when ops load."""
    from sglang.srt.compilation.passes.fusion import build_dl_fusion_passes

    passes = build_dl_fusion_passes()
    assert len(passes) == 1, passes
    assert isinstance(passes[0], ActivationQuantFusionPass), type(passes[0])
    print(f"\n[test] build_dl_fusion_passes -> {[type(p).__name__ for p in passes]}")


if __name__ == "__main__":
    test_static_quant_fusion_fires()
    test_block_quant_fusion_fires()
    test_build_dl_fusion_passes_returns_pass()
    print("\nALL TESTS PASSED")
# DL end

# DL begin
"""Pattern-matcher infrastructure for DLIN Inductor fusion passes.

Minimal port of vLLM's ``VllmFusionPatternMatcherPass`` / ``VllmPatternReplacement``
(``vllm/compilation/passes/vllm_inductor_pass.py``) onto sglang's
``SGLangInductorPass`` base class. This module provides:

- ``enable_fake_mode``: decorator that runs pattern tracing under a FakeTensorMode
  (copied from vLLM's ``inductor_pass.py``; sglang does not export one).
- ``SGLangPatternReplacement``: ABC holding a pattern/replacement pair +
  example inputs, registered into a ``PatternMatcherPass``.
- ``SGLangFusionPatternMatcherPass``: subclass of sglang's ``SGLangInductorPass``
  that owns a ``torch._inductor.pattern_matcher.PatternMatcherPass`` and applies
  it to the post-grad FX graph. Subclasses register patterns in ``__init__``.

Only what act_quant fusion needs is ported — no match-counting / debug-dump of
pattern source (vLLM's ``dump_patterns``) since sglang has no equivalent config.
"""
from __future__ import annotations

import functools
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, ParamSpec, TypeVar

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._dynamo.utils import lazy_format_graph_code
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily

from sglang.srt.compilation.inductor_pass import SGLangInductorPass

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def enable_fake_mode(fn: Callable[P, R]) -> Callable[P, R]:
    """Apply a FakeTensorMode around ``fn`` so tracing doesn't allocate real tensors.

    Verbatim port of vLLM's ``inductor_pass.enable_fake_mode`` — sglang does not
    export its own. Used when registering patterns with the Inductor pattern
    matcher (which traces pattern/replacement callables into FX subgraphs).
    """

    @functools.wraps(fn)
    def fn_new(*args: P.args, **kwargs: P.kwargs) -> R:
        with torch._guards.tracing(None), unset_fake_temporarily(), FakeTensorMode():
            result = fn(*args, **kwargs)
        return result

    return fn_new


def _fx_view_to_reshape(gm: fx.GraphModule) -> None:
    from torch._inductor.fx_passes.post_grad import view_to_reshape

    view_to_reshape(gm)


def _remove_noop_permute(gm: fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        if node.target != torch.ops.aten.permute.default:
            continue
        dims = node.args[1]
        if any(dim != i for i, dim in enumerate(dims)):
            continue
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)


class SGLangPatternReplacement(ABC, Generic[P, R]):
    """A pattern/replacement pair for FX graph fusion.

    Port of vLLM's ``VllmPatternReplacement``. Implement the three abstract
    members, then register instances with
    ``SGLangFusionPatternMatcherPass.register()``. The pass finds every
    occurrence of ``pattern`` and substitutes ``replacement``.
    """

    @property
    @abstractmethod
    def pattern(self) -> Callable[P, R]:
        """Closure defining the FX subgraph to search for."""
        ...

    @property
    @abstractmethod
    def replacement(self) -> Callable[P, R]:
        """Closure defining the FX subgraph to substitute in place of each match."""
        ...

    @abstractmethod
    def get_inputs(self) -> list[torch.Tensor]:
        """Example tensors used to trace pattern and replacement."""
        ...

    # Helpers for get_inputs: uninitialized tensors of common dtypes.
    @staticmethod
    def empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, device="cuda", **kwargs)

    @staticmethod
    def empty_bf16(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.bfloat16, device="cuda", **kwargs)

    @staticmethod
    def empty_fp16(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.float16, device="cuda", **kwargs)

    @staticmethod
    def empty_fp32(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.float32, device="cuda", **kwargs)

    @staticmethod
    def empty_i32(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.int32, device="cuda", **kwargs)


class SGLangFusionPatternMatcherPass(SGLangInductorPass):
    """SGLangInductorPass that owns an Inductor ``PatternMatcherPass``.

    Port of vLLM's ``VllmFusionPatternMatcherPass``. Subclasses register
    ``SGLangPatternReplacement`` instances in their ``__init__`` via
    ``self.register(pr)``; ``__call__`` applies the accumulated patterns to the
    post-grad FX graph.
    """

    def __init__(self, pass_name: str) -> None:
        super().__init__()
        self.pass_name = pass_name
        self.pm_pass = PatternMatcherPass(pass_name=pass_name)
        self._pattern_replacements: list[SGLangPatternReplacement] = []

    @enable_fake_mode
    def register(self, pr: SGLangPatternReplacement) -> None:
        pm.register_replacement(
            pr.pattern,
            pr.replacement,
            pr.get_inputs(),
            self._trace_fn,
            self.pm_pass,
        )
        self._pattern_replacements.append(pr)

    def uuid(self) -> str:
        return SGLangInductorPass.hash_source(
            type(self),
            *[type(pr) for pr in self._pattern_replacements],
        )

    @staticmethod
    def _trace_fn(*args: Any, **kwargs: Any) -> fx.GraphModule:
        gm = pm.fwd_only(*args, **kwargs)
        _fx_view_to_reshape(gm)
        _remove_noop_permute(gm)
        return gm

    def __call__(self, graph: fx.Graph) -> None:
        self.begin()
        self.dump_graph(graph, f"before.{self.pass_name}")
        matched = self.pm_pass.apply(graph)
        self.dump_graph(graph, f"after.{self.pass_name}")
        self.end_and_log()
        logger.debug("%s matched %d patterns", self.pass_name, matched)
        self.matched_count = matched
        # Diagnostic: print match count so GPU runs (log_level=error) still
        # surface whether the pass fired. Gate behind SGLANG_DL_FUSION_DEBUG.
        import os

        if os.environ.get("SGLANG_DL_FUSION_DEBUG") == "1":
            print(
                f"[DL_FUSION] {self.pass_name}: matched={matched} patterns "
                f"(registered={len(self._pattern_replacements)})",
                flush=True,
            )
# DL end

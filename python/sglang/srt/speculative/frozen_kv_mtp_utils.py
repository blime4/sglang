# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPContext

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


# DL begin — Frozen-KV MTP on hybrid (full/linear) models (e.g. Qwen3.5): the draft
# hybrid backend delegates full-attn layers to full_attn_backend and linear to
# linear_attn_backend, each with its OWN token_to_kv_pool. Swapping only the
# top-level draft_attn_backend.token_to_kv_pool leaves the sub-backends reading
# the draft pool (whose full_attention_layer_id_mapping is the draft's {0}, not the
# target's), so a remapped layer_id (target_phys) fails _transfer_full_attention_id.
# These helpers swap/restore token_to_kv_pool on the backend AND its sub-backends.
# See docs/dl/dlin-sglang-mtp-vs-ngram-report.md §10.2.


def _swap_draft_kv_pool(draft_attn_backend, target_pool):
    """Swap token_to_kv_pool on the draft backend AND its hybrid sub-backends."""
    backends = [draft_attn_backend]
    for sub_attr in ("full_attn_backend", "linear_attn_backend"):
        sub = getattr(draft_attn_backend, sub_attr, None)
        if sub is not None:
            backends.append(sub)
    saved = []
    for b in backends:
        pool = getattr(b, "token_to_kv_pool", None)
        if pool is not None:
            saved.append((b, pool))
            b.token_to_kv_pool = target_pool
    return saved


def _restore_draft_kv_pool(saved):
    for b, pool in saved:
        b.token_to_kv_pool = pool


# DL end


@contextmanager
def frozen_kv_target_view(
    forward_batch: ForwardBatch,
    kv_context: FrozenKVMTPContext,
    draft_attn_backend: AttentionBackend,
):
    """Build attention metadata against committed target-prefix geometry.

    Swaps ``draft_attn_backend.token_to_kv_pool`` to the frozen target pool
    so any helper that reads ``get_token_to_kv_pool()`` during metadata init
    sees the frozen target pool. Pool refs are derived from
    ``get_attn_backend().token_to_kv_pool`` — the single backend-attribute
    swap is seen by both readers (``get_token_to_kv_pool()`` and the
    backend's own ``self.token_to_kv_pool``).
    """
    # DL begin — standard NextN: no frozen-KV pool swap needed
    if kv_context is None:
        yield
        return
    # DL end
    saved_spec_info = forward_batch.spec_info
    forward_batch.spec_info = None
    # DL begin — swap sub-backends too (see _swap_draft_kv_pool)
    saved_pools = _swap_draft_kv_pool(
        draft_attn_backend, kv_context.target_token_to_kv_pool
    )
    # DL end
    try:
        yield
    finally:
        # DL begin
        _restore_draft_kv_pool(saved_pools)
        # DL end
        forward_batch.spec_info = saved_spec_info


@contextmanager
def target_kv_pool_view(
    forward_batch: ForwardBatch,
    kv_context: FrozenKVMTPContext,
    draft_attn_backend: AttentionBackend,
):
    """Run the draft model's forward with the target's frozen KV pool.

    Swaps ``draft_attn_backend.token_to_kv_pool`` to the frozen target pool.
    The single backend-attribute swap is seen by both readers —
    ``get_token_to_kv_pool()`` (because it resolves through
    ``get_attn_backend()``) and the backend's own ``self.token_to_kv_pool``
    reads (because ``self is draft_attn_backend``).
    """
    # DL begin — standard NextN: no frozen-KV pool swap needed
    if kv_context is None:
        yield
        return
    # DL end
    # DL begin — swap sub-backends too (see _swap_draft_kv_pool)
    saved_pools = _swap_draft_kv_pool(
        draft_attn_backend, kv_context.target_token_to_kv_pool
    )
    # DL end
    try:
        yield
    finally:
        # DL begin
        _restore_draft_kv_pool(saved_pools)
        # DL end


def set_frozen_kv_positions(forward_batch: ForwardBatch, topk: int) -> None:
    """Rope phase = last written target slot, not advanced per draft step."""
    seq_lens = forward_batch.seq_lens
    positions = torch.clamp(seq_lens - 1, min=0).to(torch.int64)
    if (
        topk > 1
        and forward_batch.positions is not None
        and forward_batch.positions.numel() == positions.numel() * topk
    ):
        positions = positions.repeat_interleave(topk, dim=0)
    if forward_batch.positions is None:
        forward_batch.positions = positions
    else:
        if forward_batch.positions.shape == positions.shape:
            forward_batch.positions.copy_(positions)
        else:
            forward_batch.positions = positions


def expand_for_topk_draft(forward_batch: ForwardBatch, topk: int) -> None:
    """Repeat committed-prefix metadata for the active ``B * topk`` frontier."""
    if topk == 1 or forward_batch.batch_size == 0:
        return

    if forward_batch.batch_size != forward_batch.seq_lens.shape[0]:
        raise RuntimeError(
            "Frozen-KV MTP topk expansion expects an unexpanded forward "
            "batch where batch_size == len(seq_lens)."
        )

    forward_batch.batch_size *= topk
    forward_batch.req_pool_indices = forward_batch.req_pool_indices.repeat_interleave(
        topk, dim=0
    )
    forward_batch.seq_lens = forward_batch.seq_lens.repeat_interleave(topk, dim=0)
    if forward_batch.seq_lens_cpu is not None:
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu.repeat_interleave(
            topk, dim=0
        )
        forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
    else:
        forward_batch.seq_lens_sum = torch.sum(forward_batch.seq_lens).item()

    positions = torch.clamp(forward_batch.seq_lens - 1, min=0).to(torch.int64)
    forward_batch.positions = positions
    forward_batch.num_token_non_padded_cpu = positions.numel()
    if forward_batch.num_token_non_padded is not None:
        forward_batch.num_token_non_padded.fill_(positions.numel())
    if (
        forward_batch.mrope_positions is not None
        and forward_batch.mrope_positions.shape[-1] * topk == positions.numel()
    ):
        forward_batch.mrope_positions = forward_batch.mrope_positions.repeat_interleave(
            topk, dim=-1
        )


def position_for_batch(batch: ScheduleBatch) -> torch.Tensor:
    return torch.clamp(batch.seq_lens - 1, min=0).to(torch.int64)


def select_last_extend_hidden(
    batch: ScheduleBatch, hidden_states: torch.Tensor
) -> torch.Tensor:
    if hidden_states.shape[0] == batch.batch_size():
        return hidden_states
    lens = torch.tensor(batch.extend_lens, device=hidden_states.device)
    last_indices = torch.cumsum(lens, dim=0) - 1
    return hidden_states[last_indices.to(torch.long)]

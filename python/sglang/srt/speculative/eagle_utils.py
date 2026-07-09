from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu
from sglang.srt.utils.async_probe import maybe_detect_oob

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

# DL begin — DLIN sgl_kernel build omits verify_tree_greedy (greedy tree-verify
# sampler for spec decoding). Direct CPU-offloaded port of VerifyTreeGreedy
# (sgl-kernel/csrc/speculative/eagle_utils.cu). Tensors are tiny (bs*num_draft),
# so one batched host transfer + numpy walk beats per-element .item() GPU syncs.
_DL_TREE_GREEDY_PROBED = None


def _dl_tree_greedy_needs_fallback() -> bool:
    global _DL_TREE_GREEDY_PROBED
    if _DL_TREE_GREEDY_PROBED is None:
        try:
            _ = torch.ops.sgl_kernel.verify_tree_greedy
            _DL_TREE_GREEDY_PROBED = False
        except AttributeError:
            _DL_TREE_GREEDY_PROBED = True
    return _DL_TREE_GREEDY_PROBED


def _verify_tree_greedy_torch(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
):
    import numpy as _np

    bs, D = candidates.shape
    S = accept_index.shape[1]
    dev = predicts.device
    cand = candidates.reshape(-1).cpu().numpy()
    ri = retrive_index.reshape(-1).cpu().numpy()
    rnt = retrive_next_token.reshape(-1).cpu().numpy()
    rns = retrive_next_sibling.reshape(-1).cpu().numpy()
    tp = target_predict.reshape(-1).cpu().numpy()
    pred = predicts.reshape(-1).cpu().numpy().copy()
    ai = accept_index.cpu().numpy().copy()
    atn = accept_token_num.cpu().numpy().copy()
    for bx in range(bs):
        base = bx * D
        last = int(ri[base])
        ai[bx, 0] = last
        num_acc = 0
        cur = 0
        for _j in range(1, S):
            cur = int(rnt[base + cur])
            while cur != -1:
                draft_index = int(ri[base + cur])
                draft_token = int(cand[base + cur])
                target_token = int(tp[last])
                if draft_token == target_token:
                    pred[last] = target_token
                    num_acc += 1
                    ai[bx, num_acc] = draft_index
                    last = draft_index
                    break
                else:
                    cur = int(rns[base + cur])
            if cur == -1:
                break
        atn[bx] = num_acc
        pred[last] = int(tp[last])
    predicts.copy_(torch.as_tensor(pred, dtype=predicts.dtype, device=dev))
    accept_index.copy_(torch.as_tensor(ai, dtype=accept_index.dtype, device=dev))
    accept_token_num.copy_(
        torch.as_tensor(atn, dtype=accept_token_num.dtype, device=dev)
    )


# DL end

if _is_cuda or _is_hip or _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def per_step_draft_out_cache_loc(
    out_cache_loc: torch.Tensor,
    batch_size: int,
    topk: int,
    num_steps: int,
) -> torch.Tensor:
    """Per-step slice of the multi-step EAGLE draft out_cache_loc buffer.

    Single source of truth for the layout shared by EagleWorkerV2.draft_forward
    (per-step write target) and DeepseekV4AttnBackend (per-step compression
    write target baked into metadata).
    """
    expected = batch_size * topk * num_steps
    assert out_cache_loc.shape[0] == expected, (
        f"out_cache_loc.shape[0]={out_cache_loc.shape[0]} != "
        f"batch_size * topk * num_steps = {batch_size}*{topk}*{num_steps}={expected}"
    )
    return (
        out_cache_loc.view(batch_size, topk, num_steps)
        .permute(2, 0, 1)
        .reshape(num_steps, -1)
    )


def _eagle_prefill_tail_tokens(
    batch: ScheduleBatch, next_token_ids: torch.Tensor
) -> torch.Tensor:
    """Per-seq tail token for EAGLE prefill rotation; uses next prompt token for
    non-final chunks (chunked-prefill chain consistency, see PR #26329)."""
    tail_tokens = next_token_ids.to(batch.input_ids.dtype)
    next_prompt_token = batch.chunked_req_next_prompt_token
    if next_prompt_token is not None:
        for i, r in enumerate(batch.reqs):
            if r is batch.chunked_req:
                tail_tokens = tail_tokens.clone()
                tail_tokens[i] = next_prompt_token
                break
    return tail_tokens


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    maybe_detect_oob(
        top_scores_index,
        0,
        ss_token_list.shape[1],
        "organize_draft_results: top_scores_index OOB for gather on ss_token_list",
    )
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0, dtype=torch.long, device=parents_list[0].device
        )

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


# DL begin — build_tree_kernel_efficient torch fallback (DLIN sgl_kernel omits the
# C op). CPU-offloaded port of the build_tree_efficient kernel (FULL_MASK path; the
# MTP default). See sgl-kernel/csrc/speculative/eagle_utils.cu:34.
_dl_build_tree_probe = None


def _dl_build_tree_needs_fallback() -> bool:
    global _dl_build_tree_probe
    if _dl_build_tree_probe is None:
        try:
            _ = torch.ops.sgl_kernel.build_tree_kernel_efficient
            _dl_build_tree_probe = False
        except AttributeError:
            _dl_build_tree_probe = True
    return _dl_build_tree_probe


def _build_tree_efficient_torch(
    parent_list, selected_index, verified_seq_len, tree_mask, positions,
    retrieve_index, retrieve_next_token, retrieve_next_sibling,
    topk, depth, draft_token_num, tree_mask_mode,
):
    import numpy as _np

    bs = parent_list.shape[0]
    D = int(draft_token_num)
    dev = tree_mask.device
    pl = parent_list.reshape(-1).cpu().numpy()
    si = selected_index.reshape(-1).cpu().numpy()
    vsl = verified_seq_len.cpu().numpy()
    tm = tree_mask.cpu().numpy().copy()
    pos = positions.reshape(-1).cpu().numpy().copy()
    ri = retrieve_index.reshape(-1).cpu().numpy().copy()
    rnt = retrieve_next_token.reshape(-1).cpu().numpy().copy()
    rns = retrieve_next_sibling.reshape(-1).cpu().numpy().copy()
    pl_stride = int(topk * (depth - 1) + 1)
    FULL = int(TreeMaskMode.FULL_MASK)
    for bid in range(bs):
        seq_tree_idx = D * D * bid
        for k in range(bid):
            seq_tree_idx += int(vsl[k]) * D
        seq_len = int(vsl[bid])
        for tid in range(D):
            if tree_mask_mode == FULL:
                token_tree_idx = seq_tree_idx + (seq_len + D) * tid + seq_len + 1
            else:
                token_tree_idx = D * D * bid + D * tid + 1
            tm[token_tree_idx - 1] = True
            for k in range(D - 1):
                tm[token_tree_idx + k] = False
            if tid == 0:
                pos[bid * D] = seq_len
                for i in range(D - 1, 0, -1):
                    ri[bid * D + i] = bid * D + i
                    parent_tb_idx = int(si[bid * (D - 1) + i - 1]) // topk
                    parent_position = 0
                    if parent_tb_idx > 0:
                        parent_token_idx = int(pl[bid * pl_stride + parent_tb_idx])
                        for pp in range(D):
                            if int(si[bid * (D - 1) + pp]) == parent_token_idx:
                                parent_position = pp + 1
                                break
                    if parent_position == D:
                        continue
                    if rnt[bid * D + parent_position] == -1:
                        rnt[bid * D + parent_position] = i
                    else:
                        origin = rnt[bid * D + parent_position]
                        rnt[bid * D + parent_position] = i
                        rns[bid * D + i] = origin
                ri[bid * D] = bid * D
            else:
                cur_position = tid - 1
                position = 0
                while True:
                    position += 1
                    tm[token_tree_idx + cur_position] = True
                    parent_tb_idx = int(si[bid * (D - 1) + cur_position]) // topk
                    if parent_tb_idx == 0:
                        break
                    token_idx = int(pl[bid * pl_stride + parent_tb_idx])
                    nxt = -1
                    for cp in range(D):
                        if int(si[bid * (D - 1) + cp]) == token_idx:
                            nxt = cp
                            break
                    if nxt < 0:
                        break
                    cur_position = nxt
                pos[bid * D + tid] = position + seq_len
    tree_mask.copy_(torch.as_tensor(tm, dtype=tree_mask.dtype, device=dev))
    positions.copy_(torch.as_tensor(pos, dtype=positions.dtype, device=dev))
    retrieve_index.copy_(torch.as_tensor(ri, dtype=retrieve_index.dtype, device=dev))
    retrieve_next_token.copy_(torch.as_tensor(rnt, dtype=retrieve_next_token.dtype, device=dev))
    retrieve_next_sibling.copy_(torch.as_tensor(rns, dtype=retrieve_next_sibling.dtype, device=dev))


# DL end


def build_tree_kernel_efficient(
    bonus_tokens: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    # DL begin — DLIN sgl_kernel omits build_tree_kernel_efficient; use the torch port
    elif _dl_build_tree_needs_fallback():
        _build_tree_efficient_torch(
            parent_list.to(dtype=torch.int64),
            top_scores_index.to(dtype=torch.int64),
            seq_lens.to(dtype=torch.int64),
            tree_mask, positions,
            retrieve_index, retrieve_next_token, retrieve_next_sibling,
            int(topk), int(spec_steps), int(num_verify_tokens), int(tree_mask_mode),
        )
    # DL end
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    # DL begin — DLIN falls back to the torch port when the sgl_kernel op is absent.
    if _dl_tree_greedy_needs_fallback():
        _verify_tree_greedy_torch(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
        return predicts, accept_index, accept_token_num
    # DL end
    if _is_cuda or _is_hip or _is_musa:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num


def get_draft_hidden_dim(model_runner: ModelRunner) -> int:
    """Derive the hidden dimension of target hidden states fed to the draft model."""
    hf_config = model_runner.model_config.hf_config
    eagle_config = getattr(hf_config, "eagle_config", {})
    use_aux = eagle_config.get("use_aux_hidden_state", False)
    spec_algorithm = model_runner.spec_algorithm

    if spec_algorithm is not None and spec_algorithm.is_eagle3() and use_aux:
        base = getattr(hf_config, "target_hidden_size", None)
        if base is None:
            base = model_runner.model_config.hidden_size
        layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids", [])
        num_aux = max(len(layer_ids), 1)
        return base * num_aux
    return model_runner.model_config.spec_hidden_size


def eagle_prepare_for_verify(
    verify_input: EagleVerifyInput,
    req_to_token_pool: ReqToTokenPool,
    batch: ScheduleBatch,
    target_worker: TpModelWorker,
):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify
    from sglang.srt.speculative.triton_ops.cache_locs import (
        assign_extend_cache_locs_func,
    )

    if not batch.forward_mode.is_idle():
        # Assign cache locations
        bs = len(batch.req_pool_indices)
        batch.input_ids = verify_input.draft_token
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_verify input_ids",
        )
        device = batch.device
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + verify_input.draft_token_num,
            batch_size=bs,
            draft_token_num=verify_input.draft_token_num,
            device=device,
        )

        prepare_mamba_track_for_verify(batch)

        # TBO's split_spec_info reads these; no-verify-sync leaves both None.
        verify_input.seq_lens_cpu = batch.seq_lens_cpu
        verify_input.seq_lens_sum = (
            int(batch.seq_lens_cpu.sum()) if batch.seq_lens_cpu is not None else None
        )

    # Get a forward batch
    batch.forward_mode = (
        ForwardMode.IDLE if batch.forward_mode.is_idle() else ForwardMode.TARGET_VERIFY
    )
    capture_mode = (
        CaptureHiddenMode.NULL
        if target_worker.model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.FULL
    )
    batch.capture_hidden_mode = capture_mode
    verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

    # Run attention backend plan and cuda graph preparation
    can_run_cuda_graph = bool(
        target_worker.model_runner.decode_cuda_graph_runner
        and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
            verify_forward_batch
        )
    )
    if can_run_cuda_graph:
        target_worker.model_runner.decode_cuda_graph_runner.load_batch(
            verify_forward_batch
        )
        verify_forward_batch.mark_forward_metadata_ready()
    # Non-cuda-graph: defer init to forward_extend, which runs after
    # `_forward_raw -> prepare_mlp_sync_batch` pads the batch. Initing
    # here would use pre-pad shapes and trip DSv4 indexer shape match.

    return verify_forward_batch, can_run_cuda_graph


def eagle_sample(
    verify_input: EagleVerifyInput,
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    vocab_mask: torch.Tensor = None,
):
    """
    Verify and find accepted tokens based on logits output and batch
    (which contains spec decoding information).
    """
    import torch.nn.functional as F

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_group,
        is_dp_attention_enabled,
    )
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )
    from sglang.srt.server_args import get_global_server_args
    from sglang.srt.speculative.spec_utils import (
        SIMULATE_ACC_LEN,
        generate_simulated_accept_index,
    )
    from sglang.srt.utils.async_probe import maybe_detect_nan, sanitize_nan_logits

    device = batch.device
    if batch.forward_mode.is_idle():
        predict = torch.empty(0, dtype=torch.int32, device=device)
        num_correct_drafts = torch.empty(0, dtype=torch.int32, device=device)
        accept_index = torch.empty(0, dtype=torch.int32, device=device)
        return predict, num_correct_drafts, accept_index

    bs = len(batch.seq_lens)
    sampling_info = batch.sampling_info
    next_token_logits = logits_output.next_token_logits

    sanitize_nan_logits(next_token_logits, "verify: target model logits")

    # Apply penalty
    # This is a relaxed version of penalties for speculative decoding.
    if sampling_info.acc_additive_penalties is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.acc_additive_penalties,
                verify_input.draft_token_num,
                dim=0,
            )
        )
    if sampling_info.acc_scaling_penalties is not None:
        apply_scaling_penalties(
            next_token_logits,
            torch.repeat_interleave(
                sampling_info.acc_scaling_penalties, verify_input.draft_token_num, dim=0
            ),
        )
    if sampling_info.logit_bias is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.logit_bias, verify_input.draft_token_num, dim=0
            )
        )

    # Apply grammar mask if provided
    if vocab_mask is not None:
        assert verify_input.grammar is not None
        verify_input.grammar.apply_vocab_mask(
            logits=next_token_logits, vocab_mask=vocab_mask
        )

    candidates = verify_input.draft_token.reshape(bs, verify_input.draft_token_num)
    predict_shape = list(next_token_logits.shape)[:-1]
    predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
    accept_index = torch.full(
        (bs, verify_input.max_tree_depth), -1, dtype=torch.int32, device=device
    )
    num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)

    # Sample tokens
    # DL begin — On DLIN, the sampling verify path (tree_speculative_sampling_target_only)
    # is None (sgl_kernel C++ op missing) and falls back to chain_speculative_sampling_triton
    # with draft_probs=zeros → always-accept → output = draft's garbage tokens. When this
    # fallback would produce garbage (DLIN + no rejection sampling), force greedy verify
    # (argmax match) for correct output. Accept stays at the draft-capacity limit (~10-21%)
    # but the output is coherent (target's argmax tokens), not garbage.
    _dl_force_greedy = False
    if not sampling_info.is_all_greedy:
        try:
            torch.ops.sgl_kernel.top_k_renorm_probs  # raises if missing (DLIN)
        except AttributeError:
            if not get_global_server_args().speculative_use_rejection_sampling:
                _dl_force_greedy = True
    # DL end
    # DL begin — DLIN fallback forces greedy verify to avoid always-accept garbage output.
    if sampling_info.is_all_greedy or _is_npu or _is_hip or _dl_force_greedy:
    # DL end
        target_predict = torch.argmax(next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, verify_input.draft_token_num)
        # DL begin — one-shot debug: dump draft candidates vs target verify argmax.
        # Answers "does the DRAFT propose coherent tokens, and does the TARGET
        # verify agree?" — the decisive evidence for greedy low-accept root cause.
        # Guarded by SGLANG_DL_MTP_DEBUG_VERIFY; fires for the first N verify calls.
        import os as _os
        _dl_dbg = _os.environ.get("SGLANG_DL_MTP_DEBUG_VERIFY", "")
        if _dl_dbg:
            _n = int(_dl_dbg)
            if not hasattr(eagle_sample, "_dl_verify_ct"):
                eagle_sample._dl_verify_ct = 0
            if eagle_sample._dl_verify_ct < _n:
                eagle_sample._dl_verify_ct += 1
                _cand = candidates[0].tolist()
                _tpred = target_predict[0].tolist()
                _match = [int(a == b) for a, b in zip(_cand, _tpred)]
                _pos = (
                    verify_input.positions.tolist()
                    if getattr(verify_input, "positions", None) is not None
                    else None
                )
                # Tree mask check: for topk=1 chain, bonus (tree token 0) should
                # attend to prefix (all True) + itself only (tree row = [1,0,0,...]).
                # Layout (FULL_MASK): each row = [seq_len prefix + D tree] entries.
                _cm = getattr(verify_input, "custom_mask", None)
                _cm_info = None
                if _cm is not None:
                    _D = verify_input.draft_token_num
                    _row_len = None
                    try:
                        _cm_cpu = _cm.detach().cpu()
                        _ntot = _cm_cpu.numel()
                        _nrows = bs * _D
                        _row_len = _ntot // _nrows if _nrows else 0
                        # bonus row = row 0; tree part = last D entries
                        _bonus_tree = _cm_cpu[_row_len - _D:_row_len].tolist() if _row_len else None
                        _d0_tree = _cm_cpu[_row_len + _row_len - _D : 2*_row_len].tolist() if _row_len else None
                    except Exception:
                        _bonus_tree = _d0_tree = None
                    _cm_info = f"mask_nelem={_cm.numel()} row_len={_row_len} bonus_tree_row={_bonus_tree} d0_tree_row={_d0_tree}"
                import sys as _sys
                print(
                    f"[DL-MTP-VERIFY#{eagle_sample._dl_verify_ct}] "
                    f"draft_candidates={_cand} "
                    f"target_predict={_tpred} "
                    f"match={_match} "
                    f"verify_positions={_pos} "
                    f"{_cm_info} "
                    f"acc_scaling_penalties={'None' if sampling_info.acc_scaling_penalties is None else 'set'}",
                    file=_sys.stderr, flush=True,
                )
        # DL end
        predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            retrieve_index=verify_input.retrieve_index,
            retrieve_next_token=verify_input.retrieve_next_token,
            retrieve_next_sibling=verify_input.retrieve_next_sibling,
            target_predict=target_predict,
            topk=verify_input.tree_topk,
        )
    else:
        # DL begin — DLIN sgl_kernel's top_k_renorm_prob Python wrapper imports fine
        # but calls torch.ops.sgl_kernel.top_k_renorm_probs (C++ op, missing on DLIN).
        # Probe the C++ op; if absent, use inline torch fallbacks (NOT sampler.py's
        # which also re-exports the sgl_kernel wrapper).
        _dl_use_renorm_fallback = None
        try:
            _ = torch.ops.sgl_kernel.top_k_renorm_probs
            _dl_use_renorm_fallback = False
        except AttributeError:
            _dl_use_renorm_fallback = True

        if _dl_use_renorm_fallback:
            def top_k_renorm_prob(probs, top_ks):
                if not isinstance(top_ks, torch.Tensor):
                    top_ks = torch.tensor([top_ks] * probs.shape[0], device=probs.device, dtype=torch.int64)
                out = probs.clone()
                for i in range(probs.shape[0]):
                    k = int(top_ks[i].item())
                    if k <= 0 or k >= probs.shape[1]:
                        continue
                    topk_vals, topk_idx = probs[i].topk(k)
                    mask = torch.zeros_like(probs[i])
                    mask[topk_idx] = 1.0
                    out[i] = probs[i] * mask
                    out[i] = out[i] / out[i].sum()
                return out

            def top_p_renorm_prob(probs, top_ps):
                if not isinstance(top_ps, torch.Tensor):
                    top_ps = torch.tensor([top_ps] * probs.shape[0], device=probs.device, dtype=probs.dtype)
                out = probs.clone()
                for i in range(probs.shape[0]):
                    p = float(top_ps[i].item())
                    if p >= 1.0:
                        continue
                    sorted_vals, sorted_idx = probs[i].sort(descending=True)
                    cumsum = sorted_vals.cumsum(dim=-1)
                    mask_vals = (cumsum - sorted_vals) < p
                    mask = torch.zeros_like(probs[i])
                    mask[sorted_idx[mask_vals]] = 1.0
                    out[i] = probs[i] * mask
                    s = out[i].sum()
                    if s > 0:
                        out[i] = out[i] / s
                return out

            tree_speculative_sampling_target_only = None
        else:
            from sgl_kernel import (
                top_k_renorm_prob,
                top_p_renorm_prob,
                tree_speculative_sampling_target_only,
            )
        # DL end

        from sglang.srt.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        use_rejection_sampling = (
            get_global_server_args().speculative_use_rejection_sampling
        )

        # Apply temperature and get target probs
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, verify_input.draft_token_num, dim=0
        )  # (bs * num_draft_tokens, 1)

        target_probs = F.softmax(
            next_token_logits / expanded_temperature, dim=-1
        )  # (bs * num_draft_tokens, vocab_size)
        maybe_detect_nan(target_probs, "v2 verify: target_probs after softmax")
        target_probs = top_k_renorm_prob(
            target_probs,
            torch.repeat_interleave(
                sampling_info.top_ks, verify_input.draft_token_num, dim=0
            ),
        )  # (bs * num_draft_tokens, vocab_size)
        maybe_detect_nan(target_probs, "v2 verify: target_probs after top_k_renorm")
        target_probs = top_p_renorm_prob(
            target_probs,
            torch.repeat_interleave(
                sampling_info.top_ps, verify_input.draft_token_num, dim=0
            ),
        )
        maybe_detect_nan(target_probs, "v2 verify: target_probs after top_p_renorm")
        target_probs = target_probs.reshape(bs, verify_input.draft_token_num, -1)
        draft_probs = (
            verify_input.draft_probs
            if use_rejection_sampling
            else torch.zeros_like(target_probs)
        )
        # Defense-in-depth behind the spec_hook startup allowlist: validate the
        # actual kernel inputs (catches draft_probs plumbing regressions or a
        # startup guard bypassed by a worker subclass) before the Triton kernel.
        if use_rejection_sampling and (
            draft_probs is None or draft_probs.shape[-1] != target_probs.shape[-1]
        ):
            raise ValueError(
                "Rejection sampling requires a target-vocab draft proposal "
                "distribution; the current speculative algorithm/draft worker "
                "does not produce one (draft_probs missing or vocab-mismatched)."
            )

        # coins for rejection sampling
        coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
        # coins for final sampling
        coins_for_final_sampling = torch.rand((bs,), dtype=torch.float32, device=device)

        sampling_fn = (
            chain_speculative_sampling_triton
            if use_rejection_sampling
            else tree_speculative_sampling_target_only
        )
        # DL begin — if tree_speculative_sampling_target_only is None (DLIN),
        # fall back to rejection sampling (chain_speculative_sampling_triton).
        if sampling_fn is None:
            sampling_fn = chain_speculative_sampling_triton
        # DL end
        sampling_fn(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=verify_input.retrieve_index,
            retrive_next_token=verify_input.retrieve_next_token,
            retrive_next_sibling=verify_input.retrieve_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=get_global_server_args().speculative_accept_threshold_single,
            threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
            deterministic=True,
        )

        # Sync sampling results across TP ranks: different GPUs may
        # produce slightly different target_probs due to floating-point
        # non-determinism in softmax/top_k/top_p, causing different
        # sampled tokens. Broadcast from rank 0 to ensure consistency.
        tp_group = (
            get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
        )
        if tp_group.world_size > 1:
            tp_group.broadcast(predict, src=0)
            tp_group.broadcast(accept_index, src=0)
            tp_group.broadcast(num_correct_drafts, src=0)

    if SIMULATE_ACC_LEN > 0:
        # Do simulation. The helper builds (and returns) a replacement
        # accept_index of width spec_steps + 1, so pass max_tree_depth - 1
        # to keep the simulated width identical to the real one.
        accept_index = generate_simulated_accept_index(
            accept_index=accept_index,
            predict=predict,  # mutable
            num_correct_drafts=num_correct_drafts,  # mutable
            simulate_acc_len=SIMULATE_ACC_LEN,
            bs=bs,
            spec_steps=verify_input.max_tree_depth - 1,
        )

    # `num_correct_drafts` stays drafts-only inside this function; the returned
    # tensor includes the trailing/bonus token via out-of-place +1 so the
    # name no longer flips semantics mid-function (naming doc C2).
    return predict, num_correct_drafts + 1, accept_index

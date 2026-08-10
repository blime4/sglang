from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


# DL begin — DLIN capture-size-list warmup (vLLM-style) for SGLANG_DL_MOE_FUSED.
# invoke_fused_moe_opt (fast fused FP8 MoE prefill kernel) is JIT-compiled by dlcc
# PER prefill-M shape (~20-85s one-time each, cached at ~/.triton/cache). Without
# warmup the first real request at each new M pays that spike. DLIN vLLM warms by
# iterating its capture-size lists (vllm/v1/worker/gpu_worker.py:575 builds
# warmup_sizes = compile_sizes + cg_capture_sizes; cudagraph_utils.py:134 does
# `for num_tokens in capture_sizes`). This mirrors that: read the engine's OWN
# capture-size lists (server_args.cuda_graph_config.{prefill,decode}.bs — the same
# sizes cuda-graph capture / compile uses) and JIT-warm each, so the sizes the
# engine actually serves are pre-compiled. Decode-batch sizes are also covered by
# cuda-graph capture at startup; this warmup reinforces them and covers the
# prefill sizes (prefill CG is disabled on DLIN, so capture doesn't warm them).
# Override the list with SGLANG_DL_WARMUP_SHAPES (csv) for a subset. Enable via
# `--warmups=dlin_capture_sizes`. NOTE: warming the full capture list is one-time
# but slow (~20-85s/shape); use the env override or cache-shipping for faster startup.
@warmup("dlin_capture_sizes")
async def dlin_capture_sizes(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    import os

    sa = tokenizer_manager.server_args
    cfg = getattr(sa, "cuda_graph_config", None)
    prefill_bs = (
        list(getattr(getattr(cfg, "prefill", None), "bs", []) or [])
    )  # e.g. [4,8,...,2048]
    decode_bs = list(getattr(getattr(cfg, "decode", None), "bs", []) or [])  # e.g. [1,2]
    env_shapes = os.environ.get("SGLANG_DL_WARMUP_SHAPES")
    if env_shapes:
        sizes = [int(x) for x in env_shapes.split(",") if x.strip()]
        logger.info("DL dlin_capture_sizes: env override SGLANG_DL_WARMUP_SHAPES=%s", sizes)
    else:
        sizes = sorted(set(prefill_bs + decode_bs))  # both capture lists, deduped
    logger.info(
        "DL dlin_capture_sizes warmup: capture lists prefill.bs=%s decode.bs=%s -> "
        "sweeping %d sizes %s (~20-85s/shape first time; dlcc JIT cached at ~/.triton/cache)",
        prefill_bs, decode_bs, len(sizes), sizes[:12] + (["..."] if len(sizes) > 12 else []),
    )
    for size in tqdm.tqdm(sizes):
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            # max_new_tokens=1: the prefill forward through the MoE is what triggers
            # the M-shape dlcc compile; one decode step is enough and keeps it fast.
            sampling_params={"max_new_tokens": 1, "temperature": 0, "min_p": 0.0},
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST
        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()
    logger.info("DL dlin_capture_sizes warmup done.")
    # DL end

# Copyright 2023-2026 SGLang Team
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
"""FullCudaGraphBackend — captures the entire model forward as one
torch.cuda.CUDAGraph per shape.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.cuda.CUDAGraph()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = self._device_module.graph

        # DL begin — capture-state experiments (env-gated, default off) to unblock
        # invoke_fused_moe_opt's PDL act-quant under full CG (see docs §7.30).
        #   SGLANG_DL_CAP_MODE: cudaStreamCaptureMode ("global"|"relaxed"|"thread_local").
        #     NOTE: torch.cuda.graph's `capture_error_mode` IS the capture mode.
        #   SGLANG_DL_CAP_DEFAULT_POOL=1: skip the self-managed graph pool (use torch
        #     default) — tests whether sglang's set_graph_pool_id address space breaks
        #     the PDL kernel (vLLM uses torch's default pool).
        import os as _dl_os
        _dl_cap_mode = _dl_os.environ.get("SGLANG_DL_CAP_MODE", "global")
        _dl_pool = None if _dl_os.environ.get("SGLANG_DL_CAP_DEFAULT_POOL") == "1" else self._pool
        # SGLANG_DL_CAP_STREAM=fresh: capture on a freshly-created stream (default
        # priority) instead of graph_capture()'s stream — tests whether the capture
        # stream's properties affect use_moe_cu's PDL act-quant capturability.
        _dl_cap_stream = self._capture_stream
        if _dl_os.environ.get("SGLANG_DL_CAP_STREAM") == "fresh":
            _dl_cap_stream = self._device_module.Stream()
        with graph_ctx(
            cuda_graph=graph,
            pool=_dl_pool,
            stream=_dl_cap_stream,
            capture_error_mode=_dl_cap_mode,
        ):
            out = forward_fn()
        # DL end

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        # DL begin — measure pure GPU forward time per decode step (opt-in
        # SGLANG_DL_TIME_REPLAY=1). synchronize() serializes so the elapsed_time
        # is the true GPU duration of the graph, not async-overlapped wall time.
        # SGLANG_DL_TIME_REPLAY=2: wall-clock gap between replays (no sync).
        import os as _dl_os
        _dl_mode = _dl_os.environ.get("SGLANG_DL_TIME_REPLAY", "0")
        if _dl_mode == "1":
            if not hasattr(self, "_dl_replay_times"):
                self._dl_replay_times = []
                self._dl_s = torch.cuda.Event(enable_timing=True)
                self._dl_e = torch.cuda.Event(enable_timing=True)
            self._dl_s.record()
            self._graphs[shape_key].replay()
            self._dl_e.record()
            self._dl_e.synchronize()
            self._dl_replay_times.append(self._dl_s.elapsed_time(self._dl_e))
            if len(self._dl_replay_times) % 8 == 0:
                ts = self._dl_replay_times[-8:]
                ts_sorted = sorted(ts)
                med = ts_sorted[len(ts_sorted) // 2]
                print(
                    f"[DL replay GPU] step={len(self._dl_replay_times)} "
                    f"median8={med:.2f}ms mean8={sum(ts)/len(ts):.2f}ms",
                    flush=True,
                )
            return self._outputs[shape_key]
        elif _dl_mode == "2":
            import time as _dl_time
            if not hasattr(self, "_dl_wall_times"):
                self._dl_wall_times = []
                self._dl_last_replay = None
            now = _dl_time.perf_counter()
            if self._dl_last_replay is not None:
                gap_ms = (now - self._dl_last_replay) * 1000
                self._dl_wall_times.append(gap_ms)
                if len(self._dl_wall_times) % 16 == 0:
                    ts = self._dl_wall_times[-16:]
                    ts.sort()
                    med = ts[len(ts) // 2]
                    print(
                        f"[DL replay wall-gap] step={len(self._dl_wall_times)} "
                        f"median16={med:.2f}ms min={ts[0]:.2f} max={ts[-1]:.2f}",
                        flush=True,
                    )
            self._graphs[shape_key].replay()
            self._dl_last_replay = _dl_time.perf_counter()
            return self._outputs[shape_key]
        # DL end
        # DL begin — mode 3: tight CPU timing of cudaGraphLaunch ONLY (no sync,
        # no profiler). Settles whether launch blocks the CPU (11.3ms seen in
        # torch profile could be per-node profiler overhead, not real launch time).
        import os as _dl_os
        import time as _dl_time
        if _dl_os.environ.get("SGLANG_DL_TIME_LAUNCH") == "1":
            if not hasattr(self, "_dl_launch_times"):
                self._dl_launch_times = []
            _lt0 = _dl_time.perf_counter()
            self._graphs[shape_key].replay()
            self._dl_launch_times.append((_dl_time.perf_counter() - _lt0) * 1000)
            if len(self._dl_launch_times) % 32 == 0:
                _ts = sorted(self._dl_launch_times[-32:])
                print(
                    f"[DL launch] step={len(self._dl_launch_times)} "
                    f"median32={_ts[16]:.3f}ms p90={_ts[28]:.3f}ms max={_ts[-1]:.3f}ms",
                    flush=True,
                )
            return self._outputs[shape_key]
        # DL end
        # DL begin — async-replay worker thread (SGLANG_DL_ASYNC_REPLAY=1).
        # DLIN cudaGraphLaunch blocks the calling thread for the full graph exec
        # (~22ms). Running it on a daemon worker lets the scheduler main thread
        # do host work (pop_and_process / recv / get_next_batch) concurrently,
        # making decode GPU-bound (~22ms) instead of CPU+GPU-serial (~27ms).
        # 1-in-flight + buffer safety: wait_prev() is called by the runner
        # BEFORE load_batch so the previous replay (which reads the static input
        # buffers) is done before load_batch overwrites them.
        import os as _dl_os
        if _dl_os.environ.get("SGLANG_DL_ASYNC_REPLAY") == "1":
            if not hasattr(self, "_dl_async_init"):
                import threading as _dl_th, queue as _dl_q
                self._dl_async_q = _dl_q.Queue()
                self._dl_async_prev_done = _dl_th.Event()
                self._dl_async_prev_done.set()  # no pending job initially

                def _dl_async_worker():
                    while True:
                        key = self._dl_async_q.get()
                        if key is None:
                            break
                        # blocks ~22ms on DLIN (synchronous cudaGraphLaunch)
                        self._graphs[key].replay()
                        self._dl_async_cur_done.set()

                w = _dl_th.Thread(target=_dl_async_worker, daemon=True)
                w.start()
                self._dl_async_worker = w
                self._dl_async_init = True
            import threading as _dl_th2
            # fresh done-event for THIS job; the NEXT replay()'s wait_prev will
            # wait on it (1-in-flight + prev output ready before next load_batch)
            self._dl_async_cur_done = _dl_th2.Event()
            self._dl_async_prev_done = self._dl_async_cur_done
            self._dl_async_q.put(shape_key)
            return self._outputs[shape_key]
        # DL end
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    # DL begin — wait for the previous async replay to finish. Called by the
    # decode runner BEFORE load_batch so the static input buffers are free.
    def dl_wait_prev_replay(self):
        if getattr(self, "_dl_async_init", False):
            self._dl_async_prev_done.wait()
    # DL end

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None

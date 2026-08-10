import os, time, triton, triton.language.extra.cuda as _tlc
if not hasattr(_tlc, "gdc_wait"):
    @triton.jit
    def _w(): pass
    _tlc.gdc_wait = _w
if not hasattr(_tlc, "gdc_launch_dependents"):
    @triton.jit
    def _d(): pass
    _tlc.gdc_launch_dependents = _d

import torch, sglang

# Monkey-patch torch to count ops
_orig_call = torch._C._TensorBase.__torch_function__
_op_count = [0]
def _count_call(cls, *args, **kwargs):
    _op_count[0] += 1
    return _orig_call(cls, *args, **kwargs)
# Can't easily patch __torch_function__, but we can count CUDA launches instead

# Count via CUDA event pairs
_count = [0]
_orig_matmul = torch.matmul
def _wrapped(*a, **k):
    _count[0] += 1
    return _orig_matmul(*a, **k)
torch.matmul = _wrapped

e = sglang.Engine(
    model_path=os.environ["MODEL_PATH"], dtype="bfloat16", tp_size=2,
    attention_backend="fa3", page_size=16, mem_fraction_static=0.82,
    disable_cuda_graph=True, context_length=4096, max_running_requests=16)
p = ["The capital of France is"]
e.generate(p, sampling_params={"max_new_tokens": 8, "temperature": 0})
_count[0] = 0
e.generate(p, sampling_params={"max_new_tokens": 1, "temperature": 0})
print(f"torch.matmul calls per decode step: {_count[0]}")

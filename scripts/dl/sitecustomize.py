# DL begin — applied at EVERY Python interpreter startup (including inductor
# worker subprocesses, which re-import triton fresh and so miss the shim that
# sglang/__init__.py installs only in the main process). Mirrors the shim at
# python/sglang/__init__.py:29-53.
#
# Install by copying (or symlinking) this file into the venv site-packages:
#     SITE=$(python -c "import site; print(site.getsitepackages()[0])")
#     cp scripts/dl/sitecustomize.py "$SITE/sitecustomize.py"
# Python auto-imports sitecustomize at startup, so inductor subprocess workers
# get the gdc_wait / gdc_launch_dependents no-op stubs — removing the need for
# TORCHINDUCTOR_COMPILE_THREADS=1.
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

"""JAX backend selection (spec section 1.3). CPU is the portable default;
--backend gpu selects CUDA (requires jax[cuda12], falls back to CPU with a
logged warning); jax-metal is experimental, opt-in via --backend metal, and
falls back to CPU with a logged warning if it cannot initialise. Must be
called BEFORE anything imports jax. An externally-set JAX_PLATFORMS always
wins over the CPU default (setdefault), but explicit --backend gpu/metal
overrides it."""

import os
import sys


def _preload_pip_cuda_libs():
    """Preload the venv's pip-installed NVIDIA libs (RTLD_GLOBAL) so they win
    over stale copies reachable via LD_LIBRARY_PATH. Concretely: this machine's
    shell profile exports conda-env lib dirs holding libnvJitLink 12.1, which
    shadows pip's 12.9 and breaks libcusparse with an undefined-symbol error
    inside jax's CUDA plugin init."""
    import ctypes
    import glob
    import site
    for sp in site.getsitepackages():
        for pattern in ("nvidia/nvjitlink/lib/libnvJitLink.so*",
                        "nvidia/cusparse/lib/libcusparse.so*"):
            for lib in sorted(glob.glob(os.path.join(sp, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass  # select_backend falls back to CPU with a warning


def select_backend(name: str = "cpu") -> str:
    assert "jax" not in sys.modules, "select_backend must run before jax is imported"
    if name == "metal":
        os.environ["JAX_PLATFORMS"] = "metal,cpu"
        try:
            import jax
            platform = jax.devices()[0].platform
            if platform.lower() != "metal":
                raise RuntimeError(f"got platform {platform!r}")
            print("[backend] EXPERIMENTAL jax-metal active. If you see NaNs or "
                  "unsupported-op errors, rerun with --backend cpu.")
            return "metal"
        except Exception as e:
            print(f"[backend] WARNING: jax-metal unavailable ({e}); "
                  "falling back to CPU.")
            os.environ["JAX_PLATFORMS"] = "cpu"
            return "cpu"
    if name == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
        _preload_pip_cuda_libs()
        try:
            import jax
            platform = jax.devices()[0].platform
            if platform.lower() != "gpu" and "cuda" not in platform.lower():
                raise RuntimeError(f"got platform {platform!r}")
            dev = jax.devices()[0]
            print(f"[backend] CUDA active on {getattr(dev, 'device_kind', dev)}.")
            return "gpu"
        except Exception as e:
            print(f"[backend] WARNING: CUDA unavailable ({e}); "
                  "falling back to CPU. Install with: pip install 'jax[cuda12]<0.10'")
            os.environ["JAX_PLATFORMS"] = "cpu"
            return "cpu"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    return "cpu"

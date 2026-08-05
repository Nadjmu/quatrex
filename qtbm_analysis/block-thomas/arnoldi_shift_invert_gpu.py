#!/usr/bin/env python3
"""
The GPU counterpart of arnoldi_shift_invert_cpu.py: same quantities, same
methods, same CLI, cuDSS instead of MUMPS.

Everything except the factorization is imported from the CPU script, so the
two cannot drift apart -- a fix to the shift-invert transform, the residual
check or the PROPACK fallback lands in both. What differs here is only:

    --backend   cudss (default) | gmres_cupy      instead of mumps | ...
    [gpu]       device report instead of [threads]

WHERE THE GPU ACTUALLY HELPS. The factorization and the triangular solves run
on the device; the outer Krylov method (ARPACK / PROPACK) stays on the host,
because scipy owns the restart and orthogonalization logic and there is no
cupy equivalent for non-Hermitian eigs or for PROPACK. Each application
therefore costs a host->device copy of one vector, a cuDSS solve, and a copy
back. At n ~ 1.7e5 that is ~2.7 MB per transfer against a solve that dominates
it, so the split is not the bottleneck -- but it does mean --end largest,
which needs no factorization at all, does no GPU work whatsoever and is
better run from the CPU script.

MEMORY IS THE BINDING CONSTRAINT, not time. --quantity singular --end
smallest factorizes A and A^H, and both live on the device at once. If that
does not fit, the CPU script with --backend mumps will.

gmres_cupy is iterative: its solves are only as exact as its rtol, which the
outer Krylov method assumes they are not. Useful for a timing comparison, not
for a converged eigenvalue.

Usage is identical to the CPU script -- see its --help for the full matrix of
quantities and ends:

    python arnoldi_shift_invert_gpu.py MATRIX --quantity condition
    python arnoldi_shift_invert_gpu.py MATRIX --quantity singular \\
        --method propack -k 5
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))

from arnoldi_shift_invert_cpu import GPU_BACKENDS, main as _main


def report_gpu():
    """
    Device and memory report, the GPU analogue of [threads]. A factorization
    that does not fit is the failure mode here, so the free-memory figure is
    the one to read against the [factor] lines.
    """
    try:
        import cupy as cp
    except ImportError:
        print("[gpu] cupy not importable -- cudss will fail at construction",
              flush=True)
        return

    try:
        n_dev = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        print(f"[gpu] no CUDA device visible ({exc})", flush=True)
        return

    print(f"[gpu] devices = {n_dev}", flush=True)
    for i in range(n_dev):
        try:
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"]
            name = name.decode() if isinstance(name, bytes) else name
            with cp.cuda.Device(i):
                free, total = cp.cuda.runtime.memGetInfo()
            print(f"[gpu]   {i}: {name}, "
                  f"{free / 2**30:.1f} / {total / 2**30:.1f} GiB free",
                  flush=True)
        except Exception as exc:
            print(f"[gpu]   {i}: could not query ({exc})", flush=True)

    try:
        import nvmath
        print(f"[gpu] nvmath {getattr(nvmath, '__version__', '?')}", flush=True)
    except ImportError:
        print("[gpu] nvmath-python not installed -- cudss unavailable "
              "(pip install nvmath-python[cu12])", flush=True)


if __name__ == "__main__":
    _main(backends=GPU_BACKENDS, default_backend="cudss",
          report_env=report_gpu)

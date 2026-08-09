#!/usr/bin/env python3
"""
GPU counterpart of arnoldi_shift_invert_cpu.py: the same quantities, methods
and command-line interface, with cuDSS in place of MUMPS.

Everything except the choice of factorization backend is imported from the CPU
script, so the two cannot diverge: a correction to the shift-invert transform,
the residual check or the PROPACK fallback applies to both. Only two things
differ:

    --backend   cudss (default) or gmres_cupy, instead of mumps and the other
                CPU backends
    [gpu]       a device report in place of the [threads] report

Division of work
----------------
The factorization and the triangular solves execute on the device; the outer
Krylov method, ARPACK or PROPACK, remains on the host, because SciPy owns the
restart and orthogonalization logic and CuPy provides no equivalent for
non-Hermitian eigs or for PROPACK. Each application of the operator therefore
costs a host-to-device copy of one vector, a cuDSS solve, and a copy back. At
n of order 1.7e5 that is about 2.7 MB per transfer against a solve that
dominates it, so the split is not the limiting factor.

It does follow that --end largest, which requires no factorization, performs no
device work at all and should be run from the CPU script.

Constraint
----------
Memory, not time, is the binding constraint. --quantity singular --end smallest
factorizes both A and A^H and holds both on the device simultaneously. Where
that does not fit, the CPU script with --backend mumps will.

gmres_cupy is iterative, so its solves are accurate only to its rtol, whereas
the outer Krylov method assumes an exact solve. It is suitable for a timing
comparison, not for a converged eigenvalue.

Usage
-----
Identical to the CPU script; see its --help for the full set of quantities and
spectrum ends.

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

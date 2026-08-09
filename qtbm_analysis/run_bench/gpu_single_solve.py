#!/usr/bin/env python3
"""
Timing and memory of a single cuDSS solve on the GPU, with per-stage progress.

Input
-----
One fixed matrix and right-hand-side pair, at M_PATH and RHS_PATH: a CSR .npz
triplet and a .npy array.

Purpose
-------
The GPU counterpart of single_solve.py, restricted to cuDSS. Progress is
reported at every stage, which matters more here than on the CPU: device work
is enqueued asynchronously, so a stage that appears instantaneous may not have
executed yet.

Output
------
Timings, the relative residual and the memory figures on stdout. Nothing is
written to disk.

Usage
-----
    python gpu_single_solve.py
"""

import sys
import time
import resource
import subprocess
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp

from solver_classes import CuDSS, gpu_available

M_PATH = "/scratch/yimili/matrices/dev_12_sorted_BENCH/M_E_0.npz"
RHS_PATH = "/scratch/yimili/matrices/dev_12_sorted_BENCH/rhs_E_0.npy"

DTYPE = np.complex128


def load_matrix(path):
    """Load a CSR matrix from an .npz triplet of data, indices, indptr, shape."""
    print(f"[load] reading {path} ...", flush=True)
    d = np.load(path)
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    print(f"[load] done -- shape={A.shape}, nnz={A.nnz}", flush=True)
    return A


def cpu_peak_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0


def gpu_mem_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def timed(label, fn, *args, **kwargs):
    print(f"[run] starting: {label} ...", flush=True)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"[run] finished: {label} ({dt*1e3:.2f} ms)", flush=True)
    return out, dt


def main():
    A = load_matrix(M_PATH)

    print(f"[load] reading {RHS_PATH} ...", flush=True)
    b = np.load(RHS_PATH)
    print(f"[load] done -- rhs shape={b.shape}", flush=True)

    print(f"dtype used = {np.dtype(DTYPE).name}", flush=True)

    if not gpu_available():
        print("cuDSS: skipped (no GPU visible)")
        return

    try:
        nrhs = b.shape[1] if b.ndim == 2 else 1
        print(f"nrhs = {nrhs}", flush=True)

        cpu_before = cpu_peak_mb()
        gpu_before = gpu_mem_used_mb()
        print(f"[mem] before: CPU peak RSS = {cpu_before:.1f} MB, "
              f"GPU used = {gpu_before} MB", flush=True)

        cud, t_f = timed("cuDSS factor (plan+factorize)", CuDSS, A, DTYPE, nrhs)
        x, t_s = timed("cuDSS solve", cud.solve, b)

        cpu_after = cpu_peak_mb()
        gpu_after = gpu_mem_used_mb()

        res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)

        print("\n--- cuDSS result ---")
        print(f"  factor time      : {t_f*1e3:10.2f} ms")
        print(f"  solve time       : {t_s*1e3:10.2f} ms")
        print(f"  residual         : {res:.3e}")
        print(f"  factor mem (est) : {cud.factor_nbytes()/1e6:10.1f} MB (solver-reported)")
        print(f"  CPU peak RSS     : {cpu_before:10.1f} -> {cpu_after:10.1f} MB "
              f"(delta {cpu_after - cpu_before:+.1f} MB)")
        if gpu_before is not None and gpu_after is not None:
            print(f"  GPU mem used     : {gpu_before:10.1f} -> {gpu_after:10.1f} MB "
                  f"(delta {gpu_after - gpu_before:+.1f} MB)")
        print(f"  cuDSS metadata   : {cud.get_metadata()}")

        cud.free()
        print("[done] cuDSS handle freed", flush=True)

    except Exception as e:
        print(f"\n--- cuDSS ---\n  FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Single-solve timing + memory check (CPU). Choose which solvers to run
via --solvers. Prints progress at every stage.

Usage:
    python single_solve_cpu.py --solvers superlu mumps
    python single_solve_cpu.py --solvers block_thomas --bs 104
    python single_solve_cpu.py /path/to/M.npz /path/to/rhs.npy --solvers superlu umfpack mumps gmres
"""

import sys
import time
import argparse
import resource
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp

from solver_classes import (
    SparseLU, UMFPACK, MUMPS, GMRES, BlockThomas, extract_blocks_sparse,
)

DEFAULT_M_PATH = "/scratch/yimili/matrices/dev_12_sorted_BENCH/M_E_0.npz"
DEFAULT_RHS_PATH = "/scratch/yimili/matrices/dev_12_sorted_BENCH/rhs_E_0.npy"

DTYPE = np.complex128


def load_matrix(path):
    print(f"[load] reading {path} ...", flush=True)
    d = np.load(path)
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    print(f"[load] done -- shape={A.shape}, nnz={A.nnz}", flush=True)
    return A


def cpu_peak_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0


def timed(label, fn, *args, **kwargs):
    print(f"[run] starting: {label} ...", flush=True)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"[run] finished: {label} ({dt*1e3:.2f} ms)", flush=True)
    return out, dt


def report(label, x, A, b, t_factor, t_solve, mem_factor_bytes,
           cpu_before, cpu_after, extra=""):
    res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    print(f"\n--- {label} ---")
    print(f"  factor time      : {t_factor*1e3:10.2f} ms")
    print(f"  solve time       : {t_solve*1e3:10.2f} ms")
    print(f"  residual         : {res:.3e}")
    if mem_factor_bytes is not None:
        print(f"  factor mem       : {mem_factor_bytes/1e6:10.1f} MB (solver-reported)")
    print(f"  CPU peak RSS     : {cpu_before:10.1f} -> {cpu_after:10.1f} MB "
          f"(delta {cpu_after - cpu_before:+.1f} MB){extra}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", nargs="?", default=DEFAULT_M_PATH)
    parser.add_argument("rhs", nargs="?", default=DEFAULT_RHS_PATH)
    parser.add_argument("--solvers", nargs="+", default=["superlu"],
                        choices=["superlu", "umfpack", "mumps", "gmres", "block_thomas"])
    parser.add_argument("--bs", type=int, default=None,
                        help="block size, required for block_thomas")
    args = parser.parse_args()

    A = load_matrix(args.matrix)

    print(f"[load] reading {args.rhs} ...", flush=True)
    b = np.load(args.rhs)
    print(f"[load] done -- rhs shape={b.shape}", flush=True)
    print(f"dtype used = {np.dtype(DTYPE).name}", flush=True)
    print(f"solvers requested: {args.solvers}", flush=True)

    if "superlu" in args.solvers:
        cpu_before = cpu_peak_mb()
        slu, t_f = timed("SuperLU factor", SparseLU, A, DTYPE)
        x, t_s = timed("SuperLU solve", slu.solve, b)
        cpu_after = cpu_peak_mb()
        report("SuperLU", x, A, b, t_f, t_s, slu.factor_nbytes(), cpu_before, cpu_after)

    if "umfpack" in args.solvers:
        try:
            cpu_before = cpu_peak_mb()
            umf, t_f = timed("UMFPACK factor", UMFPACK, A, DTYPE)
            x, t_s = timed("UMFPACK solve", umf.solve, b)
            cpu_after = cpu_peak_mb()
            report("UMFPACK", x, A, b, t_f, t_s, umf.factor_nbytes(), cpu_before, cpu_after)
        except (ImportError, TypeError) as e:
            print(f"\n--- UMFPACK ---\n  skipped ({e})")

    if "mumps" in args.solvers:
        try:
            cpu_before = cpu_peak_mb()
            mmp, t_f = timed("MUMPS factor", MUMPS, A, DTYPE)
            x, t_s = timed("MUMPS solve", mmp.solve, b)
            cpu_after = cpu_peak_mb()
            report("MUMPS", x, A, b, t_f, t_s, mmp.factor_nbytes(), cpu_before, cpu_after,
                   extra="  (no L/U exposed)")
        except ImportError as e:
            print(f"\n--- MUMPS ---\n  skipped ({e})")

    if "gmres" in args.solvers:
        cpu_before = cpu_peak_mb()
        gm, t_f = timed("GMRES (scipy) factor (ILU)", GMRES, A, DTYPE)
        x, t_s = timed("GMRES (scipy) solve", gm.solve, b)
        cpu_after = cpu_peak_mb()
        report("GMRES (scipy)", x, A, b, t_f, t_s, gm.factor_nbytes(), cpu_before, cpu_after,
               extra=f"  (it~{gm.last_iters}, info={gm.last_info})")

    if "block_thomas" in args.solvers:
        if args.bs is None:
            print("\n--- Block Thomas ---\n  skipped (pass --bs <block_size>)")
        else:
            print(f"[prep] extracting blocks (bs={args.bs}) ...", flush=True)
            D, L, U = extract_blocks_sparse(A, args.bs)
            print(f"[prep] done -- {len(D)} blocks", flush=True)
            cpu_before = cpu_peak_mb()
            bt, t_f = timed("Block Thomas factor", BlockThomas, D, L, U, DTYPE)
            x, t_s = timed("Block Thomas solve", bt.solve, b)
            cpu_after = cpu_peak_mb()
            report("Block Thomas", x, A, b, t_f, t_s, bt.factor_nbytes(), cpu_before, cpu_after)


if __name__ == "__main__":
    main()
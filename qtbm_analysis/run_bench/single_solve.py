#!/usr/bin/env python3
"""
Timing and memory of a single solve on the CPU, with per-stage progress.

Input
-----
    matrix    a CSR .npz triplet holding data, indices, indptr and shape
    rhs       a .npy right-hand side
    --solvers which solvers to run, canonical names from solvers/cli.py

The block solvers take no partition argument: it is always detected from the
sparsity pattern (block_sizes_from_matrix), since the exported matrices are
generally non-uniform block-tridiagonal and a fixed block size would either
misdetect that structure or require the caller to already know it.

Purpose
-------
The diagnostic entry point. Where run_benchmarks.py sweeps every material
silently, this runs one matrix with one right-hand side and reports progress at
every stage, so that a failure or an unexpected cost can be localized.

Algorithm
---------
Per requested solver: construct, which performs the whole factorization and is
what the reported factorization time measures, then solve, then report the
relative residual, the solver-reported factor footprint, and the peak resident
set size before and after.

The two memory figures answer different questions. factor_nbytes is what the
solver believes it stores; the peak RSS delta is what the process actually
consumed, including workspace and fill-in the solver does not account for.

Block partitions are validated with offband_nnz before any block solver runs,
and the script aborts rather than returning a solution computed from a
partition that discards real couplings.

Output
------
A per-solver report on stdout. Nothing is written to disk.

Usage
-----
    python single_solve.py --solvers superlu mumps
    python single_solve.py --solvers block-thomas block-thomas-inv
    python single_solve.py --solvers block-thomas-inv-fp16 --inv-dtype float16
    python single_solve.py /path/M.npz /path/rhs.npy --solvers superlu umfpack
"""

import sys
import time
import argparse
import resource
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp

import cli
from cli import BLOCK_SOLVERS
from solver_classes import (
    SparseLU, UMFPACK, MUMPS, GMRES, extract_blocks_sparse,
    BlockThomas, BlockThomasExplicitInv,
    BlockThomasFP16, BlockThomasExplicitInvFP16,
    block_sizes_from_matrix, offband_nnz,
)

# Solvers this script can drive: the CPU set, since it has no device handling.
CPU_SOLVERS = ("superlu", "umfpack", "mumps", "gmres") + BLOCK_SOLVERS

DEFAULT_M_PATH = str(cli.EXPORT_DIR / "dev_12_sorted_BENCH/M_E_0.npz")
DEFAULT_RHS_PATH = str(cli.EXPORT_DIR / "dev_12_sorted_BENCH/rhs_E_0.npy")

DTYPE = np.complex128


def load_matrix(path):
    """Load a CSR matrix from an .npz triplet of data, indices, indptr, shape."""
    print(f"[load] reading {path} ...", flush=True)
    d = np.load(path)
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    print(f"[load] done -- shape={A.shape}, nnz={A.nnz}", flush=True)
    return A


def cpu_peak_mb():
    """Peak resident set size of this process so far, in MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0


def timed(label, fn, *args, **kwargs):
    """Call fn with progress reporting; return (result, elapsed seconds)."""
    print(f"[run] starting: {label} ...", flush=True)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"[run] finished: {label} ({dt*1e3:.2f} ms)", flush=True)
    return out, dt


def report(label, x, A, b, t_factor, t_solve, mem_factor_bytes,
           cpu_before, cpu_after, extra=""):
    """
    Report one solver result: the two timings, the relative residual, the
    solver-reported factor footprint, and the peak RSS before and after.
    """
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
    ap = cli.new_parser(__doc__)
    ap.add_argument("matrix", nargs="?", default=DEFAULT_M_PATH,
                    help="CSR .npz triplet (data, indices, indptr, shape)")
    ap.add_argument("rhs", nargs="?", default=DEFAULT_RHS_PATH,
                    help=".npy right-hand side")
    cli.add_solver_selection(ap, choices=CPU_SOLVERS, default=("superlu",))
    cli.add_inv_dtype(ap)
    args = ap.parse_args()

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

    wanted_blocks = [s for s in args.solvers if s in BLOCK_SOLVERS]
    if wanted_blocks:
        # Always auto-detected: the exported matrices generally have a
        # non-uniform block-tridiagonal structure, so a fixed --block-size
        # would either misdetect it or require the caller to already know it.
        print("[prep] detecting block partition from the sparsity pattern ...",
              flush=True)
        partition = block_sizes_from_matrix(A)
        print(f"[prep] {len(partition)} blocks, "
              f"sizes {min(partition)}..{max(partition)}", flush=True)

        # A partition that cuts a real coupling yields a wrong solution
        # without raising, so it is verified before any solve.
        bad = offband_nnz(A, partition)
        print(f"[prep] off-band nnz = {bad}", flush=True)
        if bad:
            raise SystemExit(
                f"partition leaves {bad} nonzeros outside the "
                f"block-tridiagonal band -- refusing to solve with it")
        print("[prep] extracting blocks ...", flush=True)
        D, L, U = extract_blocks_sparse(A, partition)
        print(f"[prep] done -- {len(D)} blocks", flush=True)

        inv_dtype = getattr(np, args.inv_dtype)
        variants = [
            ("block-thomas",          "Block Thomas (LU)",
             lambda: BlockThomas(D, L, U, DTYPE)),
            ("block-thomas-inv",      "Block Thomas (explicit inv)",
             lambda: BlockThomasExplicitInv(D, L, U, DTYPE)),
            ("block-thomas-fp16",     "Block Thomas fp16 (LU)",
             lambda: BlockThomasFP16(D, L, U)),
            ("block-thomas-inv-fp16", f"Block Thomas fp16 (inv in {args.inv_dtype})",
             lambda: BlockThomasExplicitInvFP16(D, L, U, inv_dtype=inv_dtype)),
        ]
        for key, label, ctor in variants:
            if key not in args.solvers:
                continue
            cpu_before = cpu_peak_mb()
            bt, t_f = timed(f"{label} factor", ctor)
            x, t_s = timed(f"{label} solve", bt.solve, b)
            cpu_after = cpu_peak_mb()
            report(label, x, A, b, t_f, t_s, bt.factor_nbytes(),
                   cpu_before, cpu_after)


if __name__ == "__main__":
    main()
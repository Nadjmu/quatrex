#!/usr/bin/env python3
"""
Smallest-magnitude eigenvalue by shift-invert (sigma = 0) inverse iteration.

The linear solve at each step used to be ILU-preconditioned GMRES, which is
single-threaded: scipy's spilu is a serial SuperLU routine and the GMRES
sweep is dominated by applying it. Both MUMPS and Block Thomas replace that
with a *direct* factorization, so the structure changes:

    factor ONCE, outside the loop -> apply the same factors every iteration

MUMPS parallelizes inside its own multifrontal factorization (OpenMP, plus
threaded BLAS on the frontal matrices). Block Thomas turns the whole solve
into dense LAPACK calls on the diagonal blocks, so it inherits whatever
threading OpenBLAS is configured for. Either way the work is no longer
serial, and there is no preconditioner accuracy to trade off -- each solve is
exact to working precision, so the iteration count is the pure inverse-power
convergence rate.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp

from solver_classes import (
    BlockThomas,
    BlockThomasExplicitInv,
    MUMPS,
    SparseLU,
    block_sizes_from_matrix,
    extract_blocks_sparse,
    offband_nnz,
)

MATRIX_PATH = "/scratch/yimili/matrices/WS2-hBN-25_benchmark-QUATREX-DZ/M_E_0.npz"

BACKENDS = ("mumps", "blockthomas", "blockthomas_inv", "superlu")


def load_csr_npz(path):
    with np.load(path) as f:
        return sp.csr_matrix(
            (f["data"], f["indices"], f["indptr"]),
            shape=tuple(f["shape"]),
        )


def report_threads():
    """
    Print the effective thread counts, so a run that is accidentally serial is
    visible in the log rather than only in the wall time.
    """
    print(f"[threads] os.cpu_count()   = {os.cpu_count()}", flush=True)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        print(f"[threads] {var:<20} = {os.environ.get(var, '<unset>')}", flush=True)
    try:
        from threadpoolctl import threadpool_info

        for info in threadpool_info():
            print(
                f"[threads] {info['user_api']}/{info['internal_api']}: "
                f"{info['num_threads']} ({Path(info['filepath']).name})",
                flush=True,
            )
    except ImportError:
        print("[threads] threadpoolctl not installed, skipping BLAS probe",
              flush=True)


def build_solver(A, backend, dtype=np.complex128, block_sizes=None):
    """
    Factorize A once and return an object with a .solve(b) method.

    Block Thomas needs a block-tridiagonal partition; it is detected from the
    sparsity pattern and verified with offband_nnz -- a partition that cuts a
    real coupling does not fail loudly, extract_blocks_sparse just drops those
    entries and the solve returns a plausible, wrong answer.
    """
    if backend == "mumps":
        return MUMPS(A, dtype=dtype)

    if backend == "superlu":
        return SparseLU(A.tocsc(), dtype=dtype)

    if backend in ("blockthomas", "blockthomas_inv"):
        Ac = A.tocsr()
        Ac.sort_indices()
        sizes = block_sizes or block_sizes_from_matrix(Ac)
        bad = offband_nnz(Ac, sizes)
        print(
            f"[blocks] {len(sizes)} blocks, "
            f"min/max/mean = {min(sizes)}/{max(sizes)}/{np.mean(sizes):.1f}, "
            f"off-band nnz = {bad}",
            flush=True,
        )
        if bad:
            raise ValueError(
                f"partition is not block-tridiagonal ({bad} off-band nnz); "
                "Block Thomas would silently discard those couplings"
            )
        D, L, U = extract_blocks_sparse(Ac, sizes)
        cls = BlockThomas if backend == "blockthomas" else BlockThomasExplicitInv
        return cls(D, L, U, dtype=dtype)

    raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")


def smallest_eigenvalue(
    A,
    backend="mumps",
    tol=1e-8,
    maxiter=100,
    block_sizes=None,
):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Use complex arithmetic.
    A = A.astype(np.complex128).tocsr()
    n = A.shape[0]

    print(f"[factor] {backend}...", flush=True)
    t0 = time.perf_counter()
    solver = build_solver(A, backend, dtype=np.complex128,
                          block_sizes=block_sizes)
    t_factor = time.perf_counter() - t0
    print(f"[factor] done in {t_factor:.2f} s", flush=True)

    rng = np.random.default_rng(1234)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x /= np.linalg.norm(x)

    t_solve = 0.0
    for iteration in range(1, maxiter + 1):
        t0 = time.perf_counter()
        y = np.asarray(solver.solve(x)).reshape(n)
        t_solve += time.perf_counter() - t0

        y_norm = np.linalg.norm(y)
        if y_norm == 0 or not np.isfinite(y_norm):
            raise RuntimeError(f"Invalid vector returned by {backend}")

        x = y / y_norm

        Ax = A @ x
        eigenvalue = np.vdot(x, Ax) / np.vdot(x, x)

        residual = np.linalg.norm(Ax - eigenvalue * x)
        scale = np.linalg.norm(Ax) + abs(eigenvalue) * np.linalg.norm(x)
        relative_residual = residual / max(scale, np.finfo(float).tiny)

        print(
            f"[{iteration:3d}] "
            f"lambda={eigenvalue.real:.12e} "
            f"{eigenvalue.imag:+.12e}j, "
            f"residual={relative_residual:.3e}",
            flush=True,
        )

        if relative_residual < tol:
            print(
                f"[time] factor {t_factor:.2f} s, "
                f"{iteration} solves {t_solve:.2f} s "
                f"({t_solve / iteration:.3f} s each)",
                flush=True,
            )
            return eigenvalue, x

    raise RuntimeError("Inverse iteration did not converge")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", nargs="?", default=MATRIX_PATH,
                    help="CSR .npz triplet (data/indices/indptr/shape)")
    ap.add_argument("--backend", choices=BACKENDS, default="mumps",
                    help="linear solver used for the shift-invert step "
                         "(default: mumps)")
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="relative eigenpair residual to stop at")
    ap.add_argument("--maxiter", type=int, default=100)
    args = ap.parse_args()

    report_threads()

    print(f"[load] {args.matrix}", flush=True)
    A = load_csr_npz(args.matrix)

    print(
        f"[load] shape={A.shape}, nnz={A.nnz}, dtype={A.dtype}",
        flush=True,
    )

    eigenvalue, _ = smallest_eigenvalue(
        A, backend=args.backend, tol=args.tol, maxiter=args.maxiter,
    )

    print(
        "\nSmallest-magnitude eigenvalue: "
        f"{eigenvalue.real:.12e} "
        f"{eigenvalue.imag:+.12e}j"
    )


if __name__ == "__main__":
    main()

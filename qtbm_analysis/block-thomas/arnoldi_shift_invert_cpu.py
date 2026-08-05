#!/usr/bin/env python3
"""
Smallest-magnitude eigenvalues by shift-invert (sigma = 0) Arnoldi.

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
exact to working precision, so the iteration count is the pure convergence
rate of the outer eigensolver.

Two outer solvers sit on top of that factorization:

  --method arpack  (default)  scipy's eigs in shift-invert mode. Genuinely
      Arnoldi: builds a Krylov subspace of A^-1, so it returns k eigenvalues
      at once and needs far fewer solves than a power iteration to reach the
      same accuracy -- power iteration converges linearly at |lam_1/lam_2|,
      which is slow exactly when the wanted eigenvalues are clustered.
  --method power              the plain one-vector inverse iteration this
      script used to run. Kept because it is trivially auditable: one vector,
      one Rayleigh quotient, no restart logic.

Both cost (number of solves) x (per-solve time) + one factorization, so the
solve count printed at the end is the figure to compare, not the iteration
count.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigs

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
METHODS = ("arpack", "power")


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


class CountingSolve:
    """
    Wraps solver.solve so the number of applications and the time spent in
    them can be reported. ARPACK decides internally how many solves it wants
    (restart cycles, deflation), so counting them is the only honest way to
    compare its cost against the power iteration's.
    """

    def __init__(self, solver, n, dtype=np.complex128):
        self.solver = solver
        self.n = n
        self.dtype = dtype
        self.count = 0
        self.seconds = 0.0

    def __call__(self, b):
        t0 = time.perf_counter()
        y = np.asarray(self.solver.solve(b)).reshape(self.n).astype(self.dtype)
        self.seconds += time.perf_counter() - t0
        self.count += 1
        if not np.all(np.isfinite(y)):
            raise RuntimeError("solver returned a non-finite vector")
        return y

    def report(self, t_factor):
        each = self.seconds / max(self.count, 1)
        print(
            f"[time] factor {t_factor:.2f} s, "
            f"{self.count} solves {self.seconds:.2f} s ({each:.3f} s each)",
            flush=True,
        )


def eig_residuals(A, vals, vecs):
    """
    Relative residual ||A x - lam x|| / (||A x|| + |lam| ||x||) per column.

    Computed against the ORIGINAL A, not the shift-inverted operator ARPACK
    actually worked with -- its `tol` refers to the transformed problem, so
    this is the number that says whether the eigenpair is good.
    """
    AX = A @ vecs
    num = np.linalg.norm(AX - vecs * vals, axis=0)
    den = np.linalg.norm(AX, axis=0) + np.abs(vals) * np.linalg.norm(vecs, axis=0)
    return num / np.maximum(den, np.finfo(float).tiny)


def smallest_arpack(A, apply_inv, k=1, tol=0.0, maxiter=None, ncv=None):
    """
    Shift-invert Arnoldi at sigma = 0 via ARPACK.

    Passing OPinv is what keeps our factorization in play: without it eigs
    would factor A - sigma*I itself with SuperLU, which is the serial path
    this script exists to avoid. sigma=0 makes OPinv exactly A^-1, and
    which="LM" selects the largest |1/lam|, i.e. the smallest |lam|.
    """
    n = A.shape[0]
    OPinv = LinearOperator((n, n), matvec=apply_inv, dtype=np.complex128)
    vals, vecs = eigs(A, k=k, sigma=0.0, OPinv=OPinv, which="LM",
                      tol=tol, maxiter=maxiter, ncv=ncv)
    order = np.argsort(np.abs(vals))
    return vals[order], vecs[:, order]


def smallest_power(A, apply_inv, tol=1e-8, maxiter=100):
    """
    One-vector inverse iteration. Each printed lambda is the Rayleigh
    quotient of the current iterate -- successive estimates of the SAME
    eigenvalue, not a list of different ones.
    """
    n = A.shape[0]
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x /= np.linalg.norm(x)

    for iteration in range(1, maxiter + 1):
        y = apply_inv(x)
        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            raise RuntimeError("inverse iteration produced a zero vector")
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
            return np.array([eigenvalue]), x.reshape(n, 1)

    raise RuntimeError("Inverse iteration did not converge")


def smallest_eigenvalues(
    A,
    backend="mumps",
    method="arpack",
    k=1,
    tol=1e-8,
    maxiter=None,
    ncv=None,
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

    apply_inv = CountingSolve(solver, n)

    print(f"[solve] method={method}, k={k}", flush=True)
    if method == "arpack":
        vals, vecs = smallest_arpack(A, apply_inv, k=k, tol=tol,
                                     maxiter=maxiter, ncv=ncv)
    elif method == "power":
        if k != 1:
            raise ValueError("--method power carries a single vector and can "
                             "only return k=1; use --method arpack for k > 1")
        vals, vecs = smallest_power(A, apply_inv, tol=tol,
                                    maxiter=maxiter or 100)
    else:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")

    apply_inv.report(t_factor)
    return vals, vecs, eig_residuals(A, vals, vecs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", nargs="?", default=MATRIX_PATH,
                    help="CSR .npz triplet (data/indices/indptr/shape)")
    ap.add_argument("--backend", choices=BACKENDS, default="mumps",
                    help="linear solver used for the shift-invert step "
                         "(default: mumps)")
    ap.add_argument("--method", choices=METHODS, default="arpack",
                    help="outer eigensolver (default: arpack)")
    ap.add_argument("-k", type=int, default=1,
                    help="number of smallest-magnitude eigenvalues; "
                         "arpack only (default: 1)")
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="convergence tolerance; for arpack this is ARPACK's "
                         "tol on the shift-inverted problem, 0 means machine "
                         "precision")
    ap.add_argument("--ncv", type=int, default=None,
                    help="arpack Krylov subspace size (default: scipy's "
                         "max(2k+1, 20)); raise it if convergence stalls")
    ap.add_argument("--maxiter", type=int, default=None,
                    help="arpack restart cycles, or power iterations "
                         "(default: 100 for power)")
    args = ap.parse_args()

    report_threads()

    print(f"[load] {args.matrix}", flush=True)
    A = load_csr_npz(args.matrix)

    print(
        f"[load] shape={A.shape}, nnz={A.nnz}, dtype={A.dtype}",
        flush=True,
    )

    vals, _, res = smallest_eigenvalues(
        A, backend=args.backend, method=args.method, k=args.k,
        tol=args.tol, maxiter=args.maxiter, ncv=args.ncv,
    )

    print(f"\n{len(vals)} smallest-magnitude eigenvalue(s):")
    print(f"{'#':>3}  {'real':>22}  {'imag':>22}  {'|lambda|':>12}  {'residual':>10}")
    for i, (lam, r) in enumerate(zip(vals, res)):
        print(f"{i:>3}  {lam.real:>22.12e}  {lam.imag:>+22.12e}  "
              f"{abs(lam):>12.6e}  {r:>10.3e}")


if __name__ == "__main__":
    main()

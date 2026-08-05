#!/usr/bin/env python3
"""
Smallest-magnitude eigenvalues or singular values of a QTBM matrix.

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

--quantity singular switches to the smallest SINGULAR values, where the
choice of method matters more:

  --method propack  (default)  Lanczos bidiagonalization with implicit
      restarts. PROPACK's IRL mode targets the small end directly and keeps
      the bidiagonal form, so it never squares the condition number.
  --method arpack              svds via eigsh on the normal equations. Small
      singular values sit at the BOTTOM of the spectrum of A^H A, whose
      condition number is cond(A)^2 -- sigma_min below sqrt(eps)*sigma_max is
      simply not recoverable this way. Included for comparison.
  --shift-invert               run the chosen method on A^-1 instead of A and
      reciprocate: sigma_min(A) = 1 / sigma_max(A^-1). The largest singular
      values of A^-1 are the easy end of the spectrum, so this is by far the
      best-conditioned route -- and it is the only singular mode that uses
      the factorization at all.

WITHOUT --shift-invert the singular modes never touch the factorization, so
--backend is ignored and no factorization is built. With it, TWO
factorizations are needed (A and A^H), because svds needs both matvec and
rmatvec and neither MUMPS's nor Block Thomas's python API exposes a transpose
solve.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

# scipy ships PROPACK but keeps svds(solver="propack") opt-in, and the check
# runs at import time -- setting this after `import scipy` is too late and
# svds raises instead of running. setdefault so an explicit 0 in the
# environment still wins.
os.environ.setdefault("SCIPY_USE_PROPACK", "1")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigs, svds

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
QUANTITIES = ("eigenvalue", "singular")
EIG_METHODS = ("arpack", "power")
SVD_METHODS = ("propack", "arpack")
METHODS = tuple(dict.fromkeys(EIG_METHODS + SVD_METHODS))


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

    def __init__(self, solver, n, dtype=np.complex128, label="A"):
        self.solver = solver
        self.n = n
        self.dtype = dtype
        self.label = label
        self.count = 0
        self.seconds = 0.0

    def __call__(self, b):
        t0 = time.perf_counter()
        y = np.asarray(self.solver.solve(b)).reshape(self.n).astype(self.dtype)
        self.seconds += time.perf_counter() - t0
        self.count += 1
        if not np.all(np.isfinite(y)):
            raise RuntimeError(f"solver returned a non-finite vector ({self.label})")
        return y


def report_solves(t_factor, *counters):
    """One [time] line per factorization, so the A and A^H halves of a
    shift-inverted SVD can be read separately."""
    print(f"[time] factor {t_factor:.2f} s total", flush=True)
    for c in counters:
        each = c.seconds / max(c.count, 1)
        print(
            f"[time] {c.label:<4} {c.count} solves {c.seconds:.2f} s "
            f"({each:.3f} s each)",
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


def svd_residuals(A, s, u, v):
    """
    Relative residual ||A v - sigma u|| / (||A v|| + sigma ||u||) per triplet.
    u and v are (n, k); this is checked against the original A whatever
    operator the method actually ran on.
    """
    AV = A @ v
    num = np.linalg.norm(AV - u * s, axis=0)
    den = np.linalg.norm(AV, axis=0) + s * np.linalg.norm(u, axis=0)
    return num / np.maximum(den, np.finfo(float).tiny)


def _sorted_triplets(s, u, vh):
    """svds' output order is not contractual across solvers and versions, so
    sort ascending by singular value here rather than trusting it."""
    order = np.argsort(s)
    return s[order], u[:, order], vh[order, :].conj().T


def smallest_singular_direct(A, k=1, method="propack", tol=0.0, maxiter=None,
                             ncv=None):
    """
    Smallest singular values of A directly, via svds(which="SM").

    No factorization is involved, so --backend plays no part here. PROPACK
    reaches the small end through its implicitly restarted Lanczos
    bidiagonalization; ARPACK gets there through A^H A and therefore through
    cond(A)^2 -- the reason the two disagree on ill-conditioned matrices is
    the squaring, not a bug.
    """
    kw = dict(k=k, which="SM", solver=method, tol=tol,
              return_singular_vectors=True)
    if maxiter is not None:                # propack maps this to its kmax
        kw["maxiter"] = maxiter
    if method == "arpack" and ncv is not None:   # propack has no ncv
        kw["ncv"] = ncv
    u, s, vh = svds(A, **kw)
    return _sorted_triplets(s, u, vh)


def smallest_singular_shift_invert(A, apply_inv, apply_inv_H, k=1,
                                   method="propack", tol=0.0, maxiter=None,
                                   ncv=None):
    """
    Smallest singular values of A as the LARGEST of A^-1, reciprocated:
    A = U S V^H  =>  A^-1 = V S^-1 U^H, so sigma(A) = 1 / sigma(A^-1).

    The large end is the well-conditioned end for every Krylov SVD method, so
    this converges fast and stays accurate where which="SM" on A struggles.
    The cost is a second factorization: svds needs rmatvec (a solve with A^H)
    and neither MUMPS's nor Block Thomas's python API offers a transpose
    solve, so A^H is factored separately by the caller.

    Note the vector swap. svds on A^-1 returns its left vectors (= V of A) and
    right vectors (= U of A), so they come back exchanged relative to A.
    """
    n = A.shape[0]
    Ainv = LinearOperator((n, n), matvec=apply_inv, rmatvec=apply_inv_H,
                          dtype=np.complex128)
    kw = dict(k=k, which="LM", solver=method, tol=tol,
              return_singular_vectors=True)
    if maxiter is not None:                # propack maps this to its kmax
        kw["maxiter"] = maxiter
    if method == "arpack" and ncv is not None:   # propack has no ncv
        kw["ncv"] = ncv
    u_inv, s_inv, vh_inv = svds(Ainv, **kw)

    if np.any(s_inv <= 0):
        raise RuntimeError("A^-1 has a zero singular value; A is numerically "
                           "singular and sigma_min is not recoverable this way")

    # 1/s reverses the ordering, and the vectors swap roles.
    s = 1.0 / s_inv
    u = vh_inv.conj().T
    v = u_inv
    order = np.argsort(s)
    return s[order], u[:, order], v[:, order]


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


def _factor(A, backend, block_sizes, label):
    print(f"[factor] {backend} ({label})...", flush=True)
    t0 = time.perf_counter()
    solver = build_solver(A, backend, dtype=np.complex128,
                          block_sizes=block_sizes)
    dt = time.perf_counter() - t0
    print(f"[factor] {label} done in {dt:.2f} s", flush=True)
    return CountingSolve(solver, A.shape[0], label=label), dt


def smallest(
    A,
    quantity="eigenvalue",
    backend="mumps",
    method="arpack",
    k=1,
    tol=1e-8,
    maxiter=None,
    ncv=None,
    shift_invert=False,
    block_sizes=None,
):
    """
    Returns (values, residuals). `values` are the k smallest eigenvalues
    (complex) or singular values (real), ascending in magnitude; `residuals`
    are always measured against the original A.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Use complex arithmetic.
    A = A.astype(np.complex128).tocsr()

    if quantity == "eigenvalue":
        if method not in EIG_METHODS:
            raise ValueError(f"--method {method} is not an eigenvalue method; "
                             f"choose from {EIG_METHODS}")
        # Eigenvalue mode is shift-invert by construction: it is the
        # factorization that makes sigma=0 the accessible end of the spectrum.
        apply_inv, t_factor = _factor(A, backend, block_sizes, "A")

        print(f"[solve] {quantity}, method={method}, k={k}", flush=True)
        if method == "arpack":
            vals, vecs = smallest_arpack(A, apply_inv, k=k, tol=tol,
                                         maxiter=maxiter, ncv=ncv)
        else:
            if k != 1:
                raise ValueError("--method power carries a single vector and "
                                 "can only return k=1; use --method arpack "
                                 "for k > 1")
            vals, vecs = smallest_power(A, apply_inv, tol=tol,
                                        maxiter=maxiter or 100)

        report_solves(t_factor, apply_inv)
        return vals, eig_residuals(A, vals, vecs)

    if quantity == "singular":
        if method not in SVD_METHODS:
            raise ValueError(f"--method {method} is not a singular-value "
                             f"method; choose from {SVD_METHODS}")

        if not shift_invert:
            # Nothing here uses a factorization, so do not pay for one.
            print(f"[solve] {quantity}, method={method}, k={k}, "
                  f"which=SM on A (no factorization; --backend ignored)",
                  flush=True)
            s, u, v = smallest_singular_direct(A, k=k, method=method, tol=tol,
                                               maxiter=maxiter, ncv=ncv)
            return s, svd_residuals(A, s, u, v)

        apply_inv, t_A = _factor(A, backend, block_sizes, "A")
        AH = A.conj().T.tocsr()
        apply_inv_H, t_AH = _factor(AH, backend, block_sizes, "A^H")

        print(f"[solve] {quantity}, method={method}, k={k}, "
              f"which=LM on A^-1", flush=True)
        s, u, v = smallest_singular_shift_invert(
            A, apply_inv, apply_inv_H, k=k, method=method, tol=tol,
            maxiter=maxiter, ncv=ncv,
        )
        report_solves(t_A + t_AH, apply_inv, apply_inv_H)
        return s, svd_residuals(A, s, u, v)

    raise ValueError(f"unknown quantity {quantity!r}, expected one of {QUANTITIES}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", nargs="?", default=MATRIX_PATH,
                    help="CSR .npz triplet (data/indices/indptr/shape)")
    ap.add_argument("--quantity", choices=QUANTITIES, default="eigenvalue",
                    help="compute smallest eigenvalues or smallest singular "
                         "values (default: eigenvalue)")
    ap.add_argument("--backend", choices=BACKENDS, default="mumps",
                    help="linear solver behind the shift-invert step "
                         "(default: mumps). Ignored by --quantity singular "
                         "unless --shift-invert is given")
    ap.add_argument("--method", choices=METHODS, default=None,
                    help=f"outer solver. eigenvalue: {EIG_METHODS} "
                         f"(default arpack). singular: {SVD_METHODS} "
                         f"(default propack)")
    ap.add_argument("--shift-invert", action="store_true",
                    help="singular only: run the method on A^-1 and "
                         "reciprocate. Best conditioned, and the only "
                         "singular mode that uses --backend; costs a second "
                         "factorization for A^H")
    ap.add_argument("-k", type=int, default=1,
                    help="how many smallest values to return (default: 1)")
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="convergence tolerance on the TRANSFORMED problem "
                         "the method actually runs on; 0 means machine "
                         "precision")
    ap.add_argument("--ncv", type=int, default=None,
                    help="arpack Krylov subspace size (default: scipy's "
                         "max(2k+1, 20)); raise it if convergence stalls. "
                         "Not used by propack")
    ap.add_argument("--maxiter", type=int, default=None,
                    help="arpack restart cycles, or power iterations "
                         "(default: 100 for power)")
    args = ap.parse_args()

    if args.method is None:
        args.method = "arpack" if args.quantity == "eigenvalue" else "propack"
    if args.shift_invert and args.quantity != "singular":
        ap.error("--shift-invert applies to --quantity singular; the "
                 "eigenvalue path is always shift-inverted")

    report_threads()

    print(f"[load] {args.matrix}", flush=True)
    A = load_csr_npz(args.matrix)

    print(
        f"[load] shape={A.shape}, nnz={A.nnz}, dtype={A.dtype}",
        flush=True,
    )

    vals, res = smallest(
        A, quantity=args.quantity, backend=args.backend, method=args.method,
        k=args.k, tol=args.tol, maxiter=args.maxiter, ncv=args.ncv,
        shift_invert=args.shift_invert,
    )

    if args.quantity == "eigenvalue":
        print(f"\n{len(vals)} smallest-magnitude eigenvalue(s):")
        print(f"{'#':>3}  {'real':>22}  {'imag':>22}  "
              f"{'|lambda|':>12}  {'residual':>10}")
        for i, (lam, r) in enumerate(zip(vals, res)):
            print(f"{i:>3}  {lam.real:>22.12e}  {lam.imag:>+22.12e}  "
                  f"{abs(lam):>12.6e}  {r:>10.3e}")
    else:
        print(f"\n{len(vals)} smallest singular value(s):")
        print(f"{'#':>3}  {'sigma':>22}  {'residual':>10}")
        for i, (sv, r) in enumerate(zip(vals, res)):
            print(f"{i:>3}  {sv:>22.12e}  {r:>10.3e}")


if __name__ == "__main__":
    main()

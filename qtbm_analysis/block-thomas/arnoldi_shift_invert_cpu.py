#!/usr/bin/env python3
"""
Extreme eigenvalues, singular values and the condition number of a QTBM
matrix, on CPU.

TWO ORTHOGONAL CHOICES
----------------------
  --quantity {eigenvalue, singular, condition}   WHAT to compute
  --end      {smallest, largest, both}           WHICH end of the spectrum

Everything else follows from those two. In particular there is no
--shift-invert flag any more: which end you ask for decides whether a
factorization is needed at all.

  --end largest    the EASY end. Krylov methods converge on the dominant
                   part of the spectrum naturally, so this is plain
                   matrix-vector products: no factorization, and --backend
                   is unused.
  --end smallest   the HARD end, and the reason this script exists. The
                   spectrum is transformed so the wanted end becomes the
                   dominant one:
                     eigenvalues     eigs(A, sigma=0, OPinv=A^-1)
                     singular values svds(A^-1, which="LM"), then 1/sigma
                   Both need A^-1 applied repeatedly, which is what
                   --backend factorizes once and reuses. Singular values
                   need a SECOND factorization for A^H, because svds wants
                   rmatvec and neither MUMPS's nor Block Thomas's python API
                   exposes a transpose solve.

  --quantity condition   sigma_max / sigma_min, i.e. --quantity singular
                         --end both plus the ratio. The 2-norm condition
                         number, the number that decides whether a
                         mixed-precision iterative refinement scheme can
                         converge on these matrices.

METHODS
-------
  eigenvalue: arpack (default)   Arnoldi. The one that works.
              power              one-vector power / inverse iteration. Kept
                                 because it is trivially auditable and
                                 because its failure to converge on a
                                 clustered spectrum is worth showing rather
                                 than asserting.
  singular:   arpack (default)   svds via eigsh on the normal equations.
              propack            Lanczos bidiagonalization, never forms
                                 A^H A.

WHY ARPACK BEATS PROPACK HERE, AND WHY THAT IS NOT PROPACK'S FAULT
------------------------------------------------------------------
scipy's svds hardcodes maxiter=None in its propack branch, so PROPACK never
receives a basis-size limit and falls back to its own default kmax = 10*k.
At k=1 that is a Lanczos basis of 10, against ARPACK's default ncv of 20 --
and on the WS2-hBN matrix a basis of 20 converged while 10 did not. Raising
-k is the only lever the public API leaves: kmax scales with it. Under
--end smallest a PROPACK non-convergence therefore falls back to ARPACK on
the already-paid-for factorizations rather than discarding them
(--no-fallback to disable).

WHAT --no-shift-invert IS FOR
-----------------------------
It attacks the small end of A directly (which="SM") instead of transforming
it. This is the slow, badly-conditioned path -- for eigenvalues ARPACK has
no way to target small magnitudes, and for singular values ARPACK reaches
them through A^H A and hence cond(A)^2. It is kept so the comparison can be
run rather than asserted. Expect it to be slow or to fail.
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
    CuDSS,
    GMRESCuPy,
    MUMPS,
    SparseLU,
    block_sizes_from_matrix,
    extract_blocks_sparse,
    normalize_block_sizes,
    offband_nnz,
)

CPU_BACKENDS = ("mumps", "blockthomas", "blockthomas_inv", "superlu")
GPU_BACKENDS = ("cudss", "gmres_cupy")
ALL_BACKENDS = CPU_BACKENDS + GPU_BACKENDS

# Backends whose solve() wants an (n, nrhs) column rather than a flat (n,).
COLUMN_RHS_BACKENDS = {"cudss"}

QUANTITIES = ("eigenvalue", "singular", "condition")
ENDS = ("smallest", "largest", "both")
EIG_METHODS = ("arpack", "power")
SVD_METHODS = ("arpack", "propack")
METHODS = tuple(dict.fromkeys(EIG_METHODS + SVD_METHODS))


# ===========================================================================
# loading and environment
# ===========================================================================
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


# ===========================================================================
# factorization
# ===========================================================================
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

    if backend == "cudss":
        return CuDSS(A, dtype=dtype)

    if backend == "gmres_cupy":
        # Iterative: solves are only as exact as rtol, which the outer Krylov
        # method assumes they are not. Fine for a timing comparison, not for
        # a converged eigenvalue.
        return GMRESCuPy(A, dtype=dtype)

    if backend in ("blockthomas", "blockthomas_inv"):
        Ac = A.tocsr()
        Ac.sort_indices()
        # block_sizes may be an int (uniform) from --block-sizes; normalize so
        # the report below and the partition itself see the same tuple.
        sizes = (normalize_block_sizes(Ac.shape[0], block_sizes)
                 if block_sizes else block_sizes_from_matrix(Ac))
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

    raise ValueError(f"unknown backend {backend!r}, expected one of {ALL_BACKENDS}")


class CountingSolve:
    """
    Wraps solver.solve so the number of applications and the time spent in
    them can be reported. ARPACK decides internally how many solves it wants
    (restart cycles, deflation), so counting them is the only honest way to
    compare its cost against another method's.
    """

    def __init__(self, solver, n, dtype=np.complex128, label="A", column=False):
        self.solver = solver
        self.n = n
        self.dtype = dtype
        self.label = label
        self.column = column
        self.count = 0
        self.seconds = 0.0

    def __call__(self, b):
        b = np.asarray(b, dtype=self.dtype)
        if self.column:
            b = b.reshape(self.n, 1)
        t0 = time.perf_counter()
        y = np.asarray(self.solver.solve(b)).reshape(self.n).astype(self.dtype)
        self.seconds += time.perf_counter() - t0
        self.count += 1
        if not np.all(np.isfinite(y)):
            raise RuntimeError(f"solver returned a non-finite vector ({self.label})")
        return y


def factorize(A, backend, block_sizes, label):
    print(f"[factor] {backend} ({label})...", flush=True)
    t0 = time.perf_counter()
    solver = build_solver(A, backend, dtype=np.complex128,
                          block_sizes=block_sizes)
    dt = time.perf_counter() - t0
    print(f"[factor] {label} done in {dt:.2f} s", flush=True)
    return CountingSolve(solver, A.shape[0], label=label,
                         column=backend in COLUMN_RHS_BACKENDS), dt


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


# ===========================================================================
# residuals -- always measured against the original A
# ===========================================================================
def eig_residuals(A, vals, vecs):
    """
    Relative residual ||A x - lam x|| / (||A x|| + |lam| ||x||) per column.

    Computed against the ORIGINAL A, not whatever transformed operator the
    method ran on -- ARPACK's `tol` refers to the transformed problem, so
    this is the number that says whether the eigenpair is good.
    """
    AX = A @ vecs
    num = np.linalg.norm(AX - vecs * vals, axis=0)
    den = np.linalg.norm(AX, axis=0) + np.abs(vals) * np.linalg.norm(vecs, axis=0)
    return num / np.maximum(den, np.finfo(float).tiny)


def svd_residuals(A, s, u, v):
    """
    Relative residual ||A v - sigma u|| / (||A v|| + sigma ||u||) per triplet.
    u and v are (n, k); checked against the original A whatever operator the
    method actually ran on.
    """
    AV = A @ v
    num = np.linalg.norm(AV - u * s, axis=0)
    den = np.linalg.norm(AV, axis=0) + s * np.linalg.norm(u, axis=0)
    return num / np.maximum(den, np.finfo(float).tiny)


def _sorted_triplets(s, u, vh, descending=False):
    """svds' output order is not contractual across solvers and versions, so
    sort here rather than trusting it."""
    order = np.argsort(s)
    if descending:
        order = order[::-1]
    return s[order], u[:, order], vh[order, :].conj().T


# ===========================================================================
# eigenvalues
# ===========================================================================
def eig_arpack(A, end, k=1, tol=0.0, maxiter=None, ncv=None, apply_inv=None):
    """
    ARPACK for either end.

    end="smallest" with apply_inv: shift-invert at sigma=0. Passing OPinv is
    what keeps our factorization in play -- without it eigs would factor
    A - sigma*I itself with SuperLU, the serial path this script exists to
    avoid. sigma=0 makes OPinv exactly A^-1, and which="LM" then selects the
    largest |1/lam|, i.e. the smallest |lam|.

    end="smallest" without apply_inv: which="SM" straight at A. ARPACK builds
    a Krylov space of A, whose dominant directions are the LARGE eigenvalues,
    so the wanted ones are the last to appear. Slow, often stagnant.
    """
    n = A.shape[0]
    kw = dict(k=k, tol=tol, maxiter=maxiter, ncv=ncv)
    if end == "largest":
        vals, vecs = eigs(A, which="LM", **kw)
    elif apply_inv is not None:
        OPinv = LinearOperator((n, n), matvec=apply_inv, dtype=np.complex128)
        vals, vecs = eigs(A, sigma=0.0, OPinv=OPinv, which="LM", **kw)
    else:
        vals, vecs = eigs(A, which="SM", **kw)

    order = np.argsort(np.abs(vals))
    if end == "largest":
        order = order[::-1]
    return vals[order], vecs[:, order]


def eig_power(A, apply_op, tol=1e-8, maxiter=100, label="power"):
    """
    One-vector power iteration. apply_op is A for the largest eigenvalue and
    A^-1 for the smallest; the Rayleigh quotient is taken against A either
    way, so the printed lambda is always an eigenvalue estimate of A.

    Each printed lambda is the Rayleigh quotient of the CURRENT iterate --
    successive estimates of the SAME eigenvalue, not a list of different
    ones. Convergence is linear in the ratio of the two dominant eigenvalues
    of whatever operator is being applied, so a clustered spectrum stalls it.
    """
    n = A.shape[0]
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x /= np.linalg.norm(x)

    for iteration in range(1, maxiter + 1):
        y = apply_op(x)
        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            raise RuntimeError(f"{label} iteration produced a zero vector")
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

    raise RuntimeError(
        f"{label} iteration did not converge in {maxiter} iterations "
        f"(this is the expected outcome on a clustered spectrum -- "
        f"use --method arpack)"
    )


# ===========================================================================
# singular values
# ===========================================================================
def _svds_kwargs(k, which, method, tol, maxiter, ncv):
    kw = dict(k=k, which=which, solver=method, tol=tol,
              return_singular_vectors=True)
    if maxiter is not None:
        kw["maxiter"] = maxiter
    if method == "arpack" and ncv is not None:   # propack has no ncv
        kw["ncv"] = ncv
    return kw


def svd_direct(A, end, k=1, method="arpack", tol=0.0, maxiter=None, ncv=None):
    """
    Singular values of A directly. No factorization, so --backend is unused.

    end="largest" is the easy, well-conditioned case and needs nothing else.
    end="smallest" is the hard one: PROPACK reaches it through its implicitly
    restarted Lanczos bidiagonalization, ARPACK through A^H A and therefore
    through cond(A)^2 -- sigma_min below sqrt(eps)*sigma_max is simply not
    recoverable that way.
    """
    which = "LM" if end == "largest" else "SM"
    u, s, vh = svds(A, **_svds_kwargs(k, which, method, tol, maxiter, ncv))
    return _sorted_triplets(s, u, vh, descending=(end == "largest"))


def svd_shift_invert(A, apply_inv, apply_inv_H, k=1, method="arpack", tol=0.0,
                     maxiter=None, ncv=None, fallback=True):
    """
    Smallest singular values of A as the LARGEST of A^-1, reciprocated:
    A = U S V^H  =>  A^-1 = V S^-1 U^H, so sigma(A) = 1 / sigma(A^-1).

    The large end is the well-conditioned end for every Krylov SVD method, so
    this converges fast and stays accurate where which="SM" on A struggles.

    Note the vector swap. svds on A^-1 returns its left vectors (= V of A) and
    right vectors (= U of A), so they come back exchanged relative to A.

    `fallback` retries with ARPACK if PROPACK fails to converge. Reaching this
    point cost two factorizations, and PROPACK's basis size is not tunable
    from here -- svds hardcodes maxiter=None in its propack branch, so kmax
    stays at PROPACK's own default of 10*k however the CLI is invoked.
    Throwing the factorizations away over that would be pure waste; ARPACK
    gets the same operator and its ncv/maxiter DO get through.
    """
    n = A.shape[0]
    Ainv = LinearOperator((n, n), matvec=apply_inv, rmatvec=apply_inv_H,
                          dtype=np.complex128)
    kw = _svds_kwargs(k, "LM", method, tol, maxiter, ncv)

    try:
        u_inv, s_inv, vh_inv = svds(Ainv, **kw)
    except np.linalg.LinAlgError as exc:
        if not (fallback and method == "propack"):
            raise
        print(f"[warn] propack did not converge ({exc}); retrying with arpack "
              f"on the SAME factorizations -- pass --no-fallback to let it "
              f"fail instead", flush=True)
        kw = _svds_kwargs(k, "LM", "arpack", tol, maxiter, ncv)
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


# ===========================================================================
# drivers
# ===========================================================================
def compute_eigenvalues(A, end, backend, method, k, tol, maxiter, ncv,
                        shift_invert, block_sizes):
    if end == "largest":
        print(f"[solve] eigenvalue, largest, method={method}, k={k} "
              f"(no factorization; --backend unused)", flush=True)
        if method == "power":
            vals, vecs = eig_power(A, lambda x: A @ x, tol=tol,
                                   maxiter=maxiter or 100, label="power")
        else:
            vals, vecs = eig_arpack(A, "largest", k=k, tol=tol,
                                    maxiter=maxiter, ncv=ncv)
        return vals, eig_residuals(A, vals, vecs)

    if not shift_invert:
        print(f"[solve] eigenvalue, smallest, method={method}, k={k}, "
              f"which=SM on A (no factorization; --backend unused)", flush=True)
        if method == "power":
            raise ValueError("power iteration converges to the LARGEST "
                             "eigenvalue; reaching the smallest one needs "
                             "A^-1, so --no-shift-invert cannot be combined "
                             "with --method power --end smallest")
        vals, vecs = eig_arpack(A, "smallest", k=k, tol=tol, maxiter=maxiter,
                                ncv=ncv, apply_inv=None)
        return vals, eig_residuals(A, vals, vecs)

    apply_inv, t_factor = factorize(A, backend, block_sizes, "A")
    print(f"[solve] eigenvalue, smallest, method={method}, k={k}, "
          f"shift-invert at sigma=0", flush=True)
    if method == "power":
        vals, vecs = eig_power(A, apply_inv, tol=tol, maxiter=maxiter or 100,
                               label="inverse")
    else:
        vals, vecs = eig_arpack(A, "smallest", k=k, tol=tol, maxiter=maxiter,
                                ncv=ncv, apply_inv=apply_inv)
    report_solves(t_factor, apply_inv)
    return vals, eig_residuals(A, vals, vecs)


def compute_singular(A, end, backend, method, k, tol, maxiter, ncv,
                     shift_invert, fallback, block_sizes):
    if end == "largest" or not shift_invert:
        which = "LM" if end == "largest" else "SM"
        print(f"[solve] singular, {end}, method={method}, k={k}, "
              f"which={which} on A (no factorization; --backend unused)",
              flush=True)
        s, u, v = svd_direct(A, end, k=k, method=method, tol=tol,
                             maxiter=maxiter, ncv=ncv)
        return s, svd_residuals(A, s, u, v)

    apply_inv, t_A = factorize(A, backend, block_sizes, "A")
    AH = A.conj().T.tocsr()
    apply_inv_H, t_AH = factorize(AH, backend, block_sizes, "A^H")

    print(f"[solve] singular, smallest, method={method}, k={k}, "
          f"which=LM on A^-1", flush=True)
    s, u, v = svd_shift_invert(A, apply_inv, apply_inv_H, k=k, method=method,
                               tol=tol, maxiter=maxiter, ncv=ncv,
                               fallback=fallback)
    report_solves(t_A + t_AH, apply_inv, apply_inv_H)
    return s, svd_residuals(A, s, u, v)


def run(args):
    """
    Returns a dict with whichever of "eig_smallest", "eig_largest",
    "svd_smallest", "svd_largest" were requested, each a (values, residuals)
    pair, plus "cond" when both singular ends were computed.
    """
    A = load_csr_npz(args.matrix)
    print(f"[load] shape={A.shape}, nnz={A.nnz}, dtype={A.dtype}", flush=True)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    A = A.astype(np.complex128).tocsr()

    quantity = "singular" if args.quantity == "condition" else args.quantity
    end = "both" if args.quantity == "condition" else args.end
    ends = ("largest", "smallest") if end == "both" else (end,)

    shift_invert = not args.no_shift_invert
    out = {}

    for e in ends:
        if quantity == "eigenvalue":
            vals, res = compute_eigenvalues(
                A, e, args.backend, args.method, args.k, args.tol,
                args.maxiter, args.ncv, shift_invert, args.block_sizes,
            )
            out[f"eig_{e}"] = (vals, res)
        else:
            vals, res = compute_singular(
                A, e, args.backend, args.method, args.k, args.tol,
                args.maxiter, args.ncv, shift_invert, not args.no_fallback,
                args.block_sizes,
            )
            out[f"svd_{e}"] = (vals, res)

    if "svd_largest" in out and "svd_smallest" in out:
        s_max = out["svd_largest"][0][0]
        s_min = out["svd_smallest"][0][0]
        out["cond"] = s_max / s_min

    return out


# ===========================================================================
# output
# ===========================================================================
def print_eigenvalues(vals, res, label):
    print(f"\n{len(vals)} {label} eigenvalue(s):")
    print(f"{'#':>3}  {'real':>22}  {'imag':>22}  "
          f"{'|lambda|':>12}  {'residual':>10}")
    for i, (lam, r) in enumerate(zip(vals, res)):
        print(f"{i:>3}  {lam.real:>22.12e}  {lam.imag:>+22.12e}  "
              f"{abs(lam):>12.6e}  {r:>10.3e}")


def print_singular(vals, res, label):
    print(f"\n{len(vals)} {label} singular value(s):")
    print(f"{'#':>3}  {'sigma':>22}  {'residual':>10}")
    for i, (sv, r) in enumerate(zip(vals, res)):
        print(f"{i:>3}  {sv:>22.12e}  {r:>10.3e}")


def print_results(out):
    for e, label in (("largest", "largest-magnitude"),
                     ("smallest", "smallest-magnitude")):
        if f"eig_{e}" in out:
            print_eigenvalues(*out[f"eig_{e}"], label)
    for e, label in (("largest", "largest"), ("smallest", "smallest")):
        if f"svd_{e}" in out:
            print_singular(*out[f"svd_{e}"], label)

    if "cond" in out:
        cond = out["cond"]
        s_max = out["svd_largest"][0][0]
        s_min = out["svd_smallest"][0][0]
        print("\ncondition number (2-norm)")
        print(f"  sigma_max = {s_max:.12e}")
        print(f"  sigma_min = {s_min:.12e}")
        print(f"  cond(A)   = {cond:.6e}   (~1e{np.log10(cond):.1f})")
        # The precision an iterative-refinement scheme has to beat: a working
        # precision u with cond(A)*u >= 1 cannot resolve the solution at all.
        for name, u in (("fp16", 4.9e-4), ("fp32", 6.0e-8), ("fp64", 1.1e-16)):
            verdict = "unusable" if cond * u >= 1 else f"cond*u = {cond * u:.2e}"
            print(f"  {name:<5} {verdict}")


# ===========================================================================
# CLI
# ===========================================================================
EPILOG = """\
what you can ask for
--------------------
  --quantity eigenvalue --end smallest   lambda_min, shift-invert  [factorizes]
  --quantity eigenvalue --end largest    lambda_max, plain Arnoldi
  --quantity eigenvalue --end both       both
  --quantity singular   --end smallest   sigma_min, on A^-1     [factorizes x2]
  --quantity singular   --end largest    sigma_max, plain svds
  --quantity singular   --end both       both
  --quantity condition                   sigma_max/sigma_min     [factorizes x2]

examples   (MATRIX is the required path to a CSR .npz triplet)
--------
  # smallest eigenvalue, the default
  %(prog)s MATRIX

  # 10 smallest eigenvalues with Block Thomas instead of MUMPS
  %(prog)s MATRIX -k 10 --backend blockthomas

  # show that the power method stalls where Arnoldi does not
  %(prog)s MATRIX --method power --maxiter 200

  # largest eigenvalue -- no factorization, --backend unused
  %(prog)s MATRIX --end largest

  # smallest singular value
  %(prog)s MATRIX --quantity singular

  # PROPACK: needs -k 5 to get kmax = 10*k = 50; falls back to arpack if it
  # still does not converge, reusing the factorizations
  %(prog)s MATRIX --quantity singular --method propack -k 5

  # the condition number, and what it means for mixed-precision IR
  %(prog)s MATRIX --quantity condition

  # the badly-conditioned comparison paths, expected to be slow or to fail
  %(prog)s MATRIX --quantity singular --no-shift-invert
"""


def build_parser(backends=CPU_BACKENDS, default_backend="mumps", prog=None):
    ap = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("matrix", type=Path,
                    help="path to the matrix: a CSR .npz triplet "
                         "(data/indices/indptr/shape), as written by "
                         "export_qtbm_systems._save_csr_npz")

    ap.add_argument("--quantity", choices=QUANTITIES, default="eigenvalue",
                    help="what to compute (default: eigenvalue). "
                         "'condition' means singular values at both ends "
                         "plus the ratio")
    ap.add_argument("--end", choices=ENDS, default="smallest",
                    help="which end of the spectrum (default: smallest). "
                         "'smallest' is the end that needs a factorization; "
                         "'largest' never does. Ignored by "
                         "--quantity condition, which needs both")
    ap.add_argument("--method", choices=METHODS, default=None,
                    help=f"algorithm. eigenvalue: {EIG_METHODS} "
                         f"(default arpack). singular: {SVD_METHODS} "
                         f"(default arpack)")
    ap.add_argument("--backend", choices=backends, default=default_backend,
                    help=f"direct solver factorized for the shift-invert "
                         f"(default: {default_backend}). Unused whenever no "
                         f"factorization is needed -- see --end")

    ap.add_argument("-k", type=int, default=1,
                    help="how many values to return (default: 1). Note "
                         "propack ties its basis size to this: kmax = 10*k, "
                         "and svds does not forward --maxiter to propack, so "
                         "raising -k is the only way to give it more room")
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="convergence tolerance on the TRANSFORMED problem "
                         "the method actually runs on; 0 means machine "
                         "precision. Reported residuals are always against "
                         "the original A")
    ap.add_argument("--ncv", type=int, default=None,
                    help="arpack Krylov subspace size (default: scipy's "
                         "max(2k+1, 20)); raise it if convergence stalls on "
                         "a clustered spectrum. Not used by propack")
    ap.add_argument("--maxiter", type=int, default=None,
                    help="arpack restart cycles, or power iterations "
                         "(default: 100 for power). NOT forwarded to propack "
                         "by scipy")

    ap.add_argument("--no-shift-invert", action="store_true",
                    help="attack the small end of A directly instead of "
                         "transforming it. The badly-conditioned comparison "
                         "path -- expect it to be slow or to fail")
    ap.add_argument("--no-fallback", action="store_true",
                    help="do not retry with arpack when propack fails to "
                         "converge (default: retry, reusing the "
                         "factorizations already paid for)")
    ap.add_argument("--block-sizes", type=int, default=None,
                    help="uniform Block Thomas block size; default is to "
                         "detect a custom partition from the sparsity pattern")
    return ap


def resolve_args(ap, args):
    """Fill in per-quantity defaults and reject impossible combinations."""
    quantity = "singular" if args.quantity == "condition" else args.quantity

    if args.method is None:
        args.method = "arpack"
    valid = EIG_METHODS if quantity == "eigenvalue" else SVD_METHODS
    if args.method not in valid:
        ap.error(f"--method {args.method} is not valid for "
                 f"--quantity {args.quantity}; choose from {valid}")

    if args.quantity == "condition" and args.end != "smallest":
        ap.error("--quantity condition computes both ends itself; drop --end")
    if args.method == "power" and args.k != 1:
        ap.error("--method power carries a single vector and can only return "
                 "k=1; use --method arpack for k > 1")
    return args


def main(argv=None, backends=CPU_BACKENDS, default_backend="mumps",
         report_env=report_threads):
    ap = build_parser(backends, default_backend)
    args = resolve_args(ap, ap.parse_args(argv))

    report_env()
    print(f"[load] {args.matrix}", flush=True)

    print_results(run(args))


if __name__ == "__main__":
    main()

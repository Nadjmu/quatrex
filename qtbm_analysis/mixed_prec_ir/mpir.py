#!/usr/bin/env python3
"""
Mixed-precision iterative refinement on QTBM material data.

Input
-----
    h5path, --idx      the system A = E_<idx>/M and b = E_<idx>/rhs from a
                       material HDF5 file
    --solver           which solver family provides the low-precision
                       factorization: superlu, umfpack, mumps, block_thomas or
                       cudss
    --factor-dtype     the factorization precision, u_f below
    --inner            the inner correction solve, direct or gmres
    --tol, --max-iter  the outer convergence criterion

The file is opened read-only; nothing is written back.

Background
----------
Iterative refinement solves A x = b by computing a solution in a low precision
and correcting it using residuals computed in a higher one. Three precisions
appear in the modern analysis (Carson and Higham, 2017 and 2018):

    u_f   the precision of the factorization, --factor-dtype here
    u     the working precision, in which x and the corrections are stored,
          complex128 here
    u_r   the precision in which the residual is computed, complex128 here

The classical result of Wilkinson and of Moler is that when the residual is
computed more accurately than the factorization, refinement recovers a forward
error governed by u rather than by u_f, provided the factorization is accurate
enough for the correction equation to be solved usefully. This is the entire
motivation for the scheme: a factorization is O(n^3) and a residual is O(nnz),
so accuracy characteristic of a high-precision factorization is obtained at the
cost of a low-precision one.

The condition under which this holds is the practical question. For the
classical variant, in which the correction equation is solved by a triangular
substitution using the low-precision factors, convergence requires roughly

    kappa_inf(A) u_f  <  1,

so at u_f = 2^-24, single precision, the method is limited to kappa about 1e7,
and at u_f = 2^-11, half precision, to kappa about 1e3. QTBM matrices near a
band edge exceed both. The GMRES-based variant replaces that inner solve with
GMRES applied to the preconditioned system, which relaxes the requirement to
approximately kappa u_f^{1/2} or better depending on the variant analysed, and
is what makes refinement applicable at these condition numbers at all. This is
why both inner solvers are implemented here and why the condition number is
reported alongside every result.

Algorithms
----------
Both variants share the outer loop and differ only in step 4.

    1. Build the solver at u_f. This is the one-time factorization cost, and
       the only step whose cost scales as a factorization.
    2. x = solver.solve(b), cast up to the working precision.
    3. r = b - A x, computed in complex128.
    4. Solve A dx = r for the correction:

       direct, LU-IR (Buttari et al., 2006)
           dx = solver.solve(r), a single low-precision triangular
           substitution reusing the same factorization. One triangular solve
           per outer iteration.

       gmres, GMRES-IR (Carson and Higham, 2017)
           dx is obtained by GMRES applied to A in complex128, left
           preconditioned by M^-1 v = solver.solve(v), where v is cast down to
           u_f and the result cast back up. GMRES itself runs at the working
           precision; only the preconditioner applications are cheap
           low-precision solves, and they reuse the same factorization.

    5. x = x + dx. Repeat from 3 until ||r|| / ||b|| < tol or max_iter is
       reached.

The preconditioner in the GMRES variant requires only the action of the
factorization as an operator, never explicit access to L and U. That is what
allows the same code to drive superlu, block_thomas, mumps and cudss
identically, including the two solvers that expose no factors at all.

Comparison
----------
Three variants of the same solver family are measured, so that the comparison
isolates the effect of precision and refinement rather than of the solver
implementation:

    1. <solver> at u_f with refinement           the method under test
    2. <solver> at complex128, no refinement     the accuracy reference
    3. <solver> at u_f, no refinement            the lower bound refinement
                                                 must improve upon

Limitations
-----------
UMFPACK has no single-precision build, so solver_classes.UMFPACK raises
TypeError for --factor-dtype complex64. This is a property of the library and
not of this script; select a different solver for a low-precision comparison.

The first cuDSS call in a process pays a fixed start-up cost for CUDA context
creation and kernel compilation that is independent of problem size, measured
at roughly 1.2 s. Left in place it would be charged in full to whichever
variant runs first, which is the refinement variant. An untimed warm-up solve
is therefore performed before any variant is measured; see _warm_up_gpu.

Output
------
A table of relative residual, forward error against a reference solution, wall
time and peak memory per variant, the outer convergence history, and, for the
GMRES variant, the inner iteration counts. Nothing is written to disk; for
sweeps and figures see c32_gmres_ir.py and plotting/plot_mixed_prec_ir.py.

Usage
-----
    python mpir.py .../carbon-nanotube.h5 --idx 5 --solver superlu \\
        --factor-dtype complex64

    python mpir.py .../carbon-nanotube.h5 --idx 5 --solver block-thomas \\
        --block-size 32 --factor-dtype complex64

    python mpir.py .../si-bulk.h5 --idx 254 --solver mumps \\
        --factor-dtype complex64 --inner gmres --gmres-tol 1e-8 \\
        --gmres-restart 30 --gmres-max-iter 50

    python mpir.py .../si-bulk.h5 --idx 254 --solver cudss \\
        --factor-dtype complex64

References
----------
J. H. Wilkinson, Rounding Errors in Algebraic Processes, 1963.
A. Buttari et al., Mixed precision iterative refinement techniques for the
    solution of dense linear systems, IJHPCA 21(4), 2007.
E. Carson and N. J. Higham, A new analysis of iterative refinement and its
    application to accurate solution of ill-conditioned sparse linear systems,
    SIAM J. Sci. Comput. 39(6), 2017.
E. Carson and N. J. Higham, Accelerating the solution of linear systems by
    iterative refinement in three precisions, SIAM J. Sci. Comput. 40(2), 2018.
"""

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))
import cli
from solver_classes import (
    SparseLU, UMFPACK, MUMPS, BlockThomas, CuDSS, extract_blocks_sparse,
)
from bench_all import backward_errors, _matrix_norm, NORMWISE_ORDS

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

# The working and residual precision, u and u_r in the analysis. Both are
# complex128; only the factorization precision u_f is varied.
HIGH_DTYPE = np.complex128


# ─────────────────────────────────────────────────────────────────────────────
# Solver registry. Each entry maps a --solver name to a builder with the
# signature builder(A, dtype, bs, b) returning an object exposing solve(b) and
# factor_nbytes().
#
# The right-hand side b is required only by the cuDSS builder, which must know
# the column count at construction time; every other builder ignores it. It is
# passed uniformly so that no call site has to special-case cuDSS.
# ─────────────────────────────────────────────────────────────────────────────

def _build_block_thomas(A, dtype, bs, b):
    if bs is None:
        raise ValueError("--block-size is required for --solver block-thomas")
    D, L, U = extract_blocks_sparse(A, bs)
    return BlockThomas(D, L, U, dtype=dtype)


def _solve_columns(solve_one, b):
    """Apply a single-RHS solve function column by column to a 1-D or 2-D b."""
    b = np.asarray(b)
    if b.ndim == 1:
        return solve_one(b)
    return np.column_stack(
        [solve_one(np.ascontiguousarray(b[:, j])) for j in range(b.shape[1])]
    )


class _CuDSSSolver:
    """
    Adapter around solver_classes.CuDSS for repeated solves against a fixed
    factorization, which is what refinement requires.

    Right-hand-side binding
    -----------------------
    CuDSS binds the right-hand-side shape (n, nrhs) at construction, and
    nvmath's reset_operands rejects any later array whose shape or strides
    differ from that binding. One Fortran-ordered (n, nrhs) buffer is therefore
    allocated once and each right-hand side copied into it, so that every solve
    presents a byte-layout-identical array and the rebuild-and-refactorize
    fallback inside CuDSS.solve is never triggered. Without this the
    factorization would be recomputed at every refinement step, which would
    defeat the method.

    Block against column solves
    ---------------------------
    A whole (n, k) right-hand side is solved in a single cuDSS call whenever k
    matches the nrhs the solver was built with, which is the case for LU-IR and
    for both direct variants. GMRES-IR is the exception: SciPy's gmres applies
    the preconditioner to one vector at a time, so a second solver with nrhs=1
    is built lazily and only when that path is taken. Looping over columns
    unconditionally would multiply the number of device round trips by k
    without benefit.
    """

    def __init__(self, A, dtype, nrhs):
        self.n = A.shape[0]
        self.dtype = np.dtype(dtype)
        self.nrhs = max(int(nrhs), 1)
        self._A = A
        self._solver = CuDSS(A, dtype=self.dtype, nrhs=self.nrhs)
        self._buf = np.empty((self.n, self.nrhs), dtype=self.dtype, order="F")
        self._one = None        # built lazily, for GMRES-IR only
        self._one_buf = None
        # Symbolic (reordering) and numeric factorization seconds, timed
        # separately inside solver_classes.CuDSS; see factor_breakdown().
        self.symbolic_s = self._solver.plan_seconds
        self.numeric_s = self._solver.factor_seconds

    def factor_breakdown(self):
        """(symbolic_s, numeric_s), or None where a solver fuses the two."""
        return (self.symbolic_s, self.numeric_s)

    def _solve_block(self, b2d):
        self._buf[:, :] = b2d
        return self._solver.solve(self._buf)

    def _solve_one(self, v):
        if self._one is None:
            self._one = CuDSS(self._A, dtype=self.dtype, nrhs=1)
            self._one_buf = np.empty((self.n, 1), dtype=self.dtype, order="F")
        self._one_buf[:, 0] = v
        return self._one.solve(self._one_buf)[:, 0]

    def solve(self, b):
        b = np.asarray(b)
        if b.ndim == 1:
            return self._solve_one(b)
        if b.shape[1] == self.nrhs:
            return self._solve_block(b)
        return _solve_columns(self._solve_one, b)

    def factor_nbytes(self):
        return self._solver.factor_nbytes()

    def free(self):
        self._solver.free()
        if self._one is not None:
            self._one.free()


# Keyed by canonical solver name; see solvers/cli.py.
SOLVER_BUILDERS = {
    "superlu":      lambda A, dtype, bs, b: SparseLU(A, dtype=dtype),
    "umfpack":      lambda A, dtype, bs, b: UMFPACK(A, dtype=dtype),
    "mumps":        lambda A, dtype, bs, b: MUMPS(A, dtype=dtype),
    "block-thomas": _build_block_thomas,
    "cudss":        lambda A, dtype, bs, b: _CuDSSSolver(
        A, dtype, b.shape[1] if np.asarray(b).ndim == 2 else 1),
}

# Solvers whose first call in a process pays a fixed device start-up cost that
# must be excluded from the measurement. See _warm_up_gpu.
_GPU_SOLVERS = ("cudss",)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_system(h5path, idx):
    """Load A (E_<idx>/M) and b (E_<idx>/rhs) from a material's HDF5 file."""
    with h5py.File(h5path, "r") as f:
        path = f"E_{idx}"
        if path not in f:
            raise SystemExit(f"{path} not found in {h5path}")
        g = f[path]
        gm = g["M"]
        shape = tuple(gm.attrs["shape"]) if "shape" in gm.attrs else None
        A = sp.csc_matrix((gm["data"][:], gm["indices"][:], gm["indptr"][:]), shape=shape)
        b = g["rhs"][:]
    if b.shape[-1] == 0:
        raise SystemExit(f"E_{idx}/rhs has zero columns -- pick a different --idx")
    return A, b


def load_energy_metadata(h5path):
    """
    (indices, energies, valence_band_edge, conduction_band_edge) from a
    material file's metadata group. Edge values are None if not recorded.
    """
    with h5py.File(h5path, "r") as f:
        indices = f["metadata/indices"][:]
        energies = f["metadata/energies"][:]
        attrs = f["metadata"].attrs
        valence = float(attrs["valence_band_edge"]) if "valence_band_edge" in attrs else None
        conduction = float(attrs["conduction_band_edge"]) if "conduction_band_edge" in attrs else None
    return indices, energies, valence, conduction


def energy_of_idx(indices, energies, idx):
    """Energy in eV recorded for one E_<idx>, or None if idx is not present."""
    hit = np.flatnonzero(indices == idx)
    return float(energies[hit[0]]) if hit.size else None


def idx_of_energy(indices, energies, energy):
    """Index whose recorded energy is nearest the requested one, in eV."""
    nearest = int(np.argmin(np.abs(energies - energy)))
    return int(indices[nearest])


def load_condition_number(h5path, idx):
    """
    kappa_2(A) for one energy index, from the material's own condition-estimate
    file, cli.CONDITION_DIR/<material>.h5 -- a separate file from h5path,
    written by the condition-est pipeline, not by run_benchmarks.py. Its
    /condition/indices holds the same energy indices as h5path's
    metadata/indices, and /condition/cond_2 the corresponding kappa_2 (from
    sigma_max/sigma_min); /condition/valid marks entries where the SVD
    estimate succeeded. Returns None if the file, the index, or a valid
    estimate for it isn't present.
    """
    cond_path = cli.CONDITION_DIR / f"{Path(h5path).stem}.h5"
    if not cond_path.exists():
        return None
    with h5py.File(cond_path, "r") as f:
        if "condition/cond_2" not in f:
            return None
        indices = f["condition/indices"][:]
        cond2 = f["condition/cond_2"][:]
        valid = f["condition/valid"][:] if "condition/valid" in f else None
    hit = np.flatnonzero(indices == idx)
    if hit.size == 0:
        return None
    i = hit[0]
    if valid is not None and not valid[i]:
        return None
    return float(cond2[i])


# ─────────────────────────────────────────────────────────────────────────────
# The refinement variants
# ─────────────────────────────────────────────────────────────────────────────

def solve_mixed_ir(solver_name, A, b, bs, low_dtype, tol, max_iter, x_true=None):
    """
    LU-IR: iterative refinement whose correction solve is a single
    low-precision triangular substitution.

    The factorization is computed once at low_dtype and reused at every outer
    iteration; the residual is formed in complex128. Convergence requires
    approximately kappa_inf(A) * u_f < 1, so this variant is the one that fails
    first as the matrix becomes ill-conditioned. See the module docstring.

    Parameters
    ----------
    x_true : complex128 reference solution, optional. When supplied, the
        relative forward error is recorded at each iteration, at the point
        where the residual has already been computed. It is used for reporting
        only: it enters neither the convergence test nor any other control
        flow, so it cannot perturb the residual, timing or memory figures.

    Returns
    -------
    (x, extra) with extra carrying the residual history, the forward-error
    history and the factor footprint.
    """
    b_high = np.asarray(b, dtype=HIGH_DTYPE)
    A_high = A.tocsc().astype(HIGH_DTYPE)
    norm_b = np.linalg.norm(b_high)
    norm_x_true = np.linalg.norm(x_true) if x_true is not None else None
    history = []
    true_err_history = []

    t0 = time.perf_counter()
    solver = SOLVER_BUILDERS[solver_name](A, low_dtype, bs, b)
    factor_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = solver.solve(b_high.astype(low_dtype)).astype(HIGH_DTYPE)

    for _ in range(max_iter):
        r = b_high - A_high @ x
        rel = np.linalg.norm(r) / norm_b
        history.append(rel)
        if x_true is not None:
            true_err_history.append(np.linalg.norm(x - x_true) / norm_x_true)
        if rel < tol:
            break
        x = x + solver.solve(r.astype(low_dtype)).astype(HIGH_DTYPE)
    inner_s = time.perf_counter() - t0

    extra = {
        "history": history,
        "true_err_history": true_err_history,
        "mem_bytes": solver.factor_nbytes(),
        "factor_s": factor_s,
        "inner_s": inner_s,
        "factor_breakdown": getattr(solver, "factor_breakdown", lambda: None)(),
    }
    if hasattr(solver, "free"):
        solver.free()
    return x, extra


def _gmres_solve(A_op, rhs, M_op, tol, restart, maxiter, callback):
    """
    Call scipy.sparse.linalg.gmres compatibly across SciPy versions.

    The tol keyword was renamed to rtol in SciPy 1.12; the fallback covers
    older installations.
    """
    try:
        return spla.gmres(A_op, rhs, M=M_op, rtol=tol, atol=0.0, restart=restart,
                          maxiter=maxiter, callback=callback, callback_type="pr_norm")
    except TypeError:
        return spla.gmres(A_op, rhs, M=M_op, tol=tol, atol=0.0, restart=restart,
                          maxiter=maxiter, callback=callback, callback_type="pr_norm")


def solve_gmres_ir(solver_name, A, b, bs, low_dtype, tol, max_iter, x_true=None,
                   gmres_tol=1e-8, gmres_restart=30, gmres_max_iter=50):
    """
    GMRES-IR: iterative refinement whose correction solve is preconditioned
    GMRES at the working precision.

    The correction equation A dx = r is solved by GMRES applied to A in
    complex128, left preconditioned by M^-1 v = solver.solve(v), with v cast
    down to low_dtype and the result cast back up before it is returned to
    GMRES. Only the preconditioner applications are performed in low precision.

    Because GMRES requires only the action of the factorization as an operator,
    and never L and U themselves, this variant is available for every solver in
    the registry, including MUMPS and cuDSS, which expose no factors.

    Relative to LU-IR this relaxes the condition-number requirement
    substantially, which is what makes refinement usable on the ill-conditioned
    QTBM systems. The cost is the inner iteration count, recorded per outer
    step in extra["gmres_iters_history"].

    SciPy's gmres accepts a single right-hand-side vector only, so a
    multi-column b is handled by looping over columns. The initial
    low-precision solve and the residual and error bookkeeping remain
    vectorized.

    Parameters
    ----------
    x_true : as in solve_mixed_ir; recorded for reporting only.

    Returns
    -------
    (x, extra) with extra carrying the residual history, the forward-error
    history, the inner GMRES iteration counts and the factor footprint.
    """
    b_high = np.asarray(b, dtype=HIGH_DTYPE)
    orig_ndim = b_high.ndim
    b2 = b_high if orig_ndim == 2 else b_high[:, None]
    A_high = A.tocsc().astype(HIGH_DTYPE)
    n, k = A_high.shape[0], b2.shape[1]
    norm_b = np.linalg.norm(b_high)

    x_true2 = None
    norm_x_true = None
    if x_true is not None:
        x_true2 = x_true if x_true.ndim == 2 else x_true[:, None]
        norm_x_true = np.linalg.norm(x_true)

    history = []
    true_err_history = []
    gmres_iters_history = []   # list (per outer iter) of lists (per rhs column)

    t0 = time.perf_counter()
    solver = SOLVER_BUILDERS[solver_name](A, low_dtype, bs, b)
    factor_s = time.perf_counter() - t0

    def precond_apply(v):
        return solver.solve(v.astype(low_dtype)).astype(HIGH_DTYPE)

    A_op = spla.LinearOperator((n, n), matvec=lambda v: A_high @ v, dtype=HIGH_DTYPE)
    M_op = spla.LinearOperator((n, n), matvec=precond_apply, dtype=HIGH_DTYPE)

    t0 = time.perf_counter()
    x2 = solver.solve(b2.astype(low_dtype)).astype(HIGH_DTYPE)   # x0: low-precision direct solve

    for _ in range(max_iter):
        r2 = b2 - A_high @ x2
        rel = np.linalg.norm(r2) / norm_b
        history.append(rel)
        if x_true2 is not None:
            true_err_history.append(np.linalg.norm(x2 - x_true2) / norm_x_true)
        if rel < tol:
            break

        d2 = np.zeros_like(x2)
        iters_this_round = []
        for j in range(k):
            counter = [0]

            def _cb(_res, counter=counter):
                counter[0] += 1

            dj, info = _gmres_solve(A_op, r2[:, j], M_op, gmres_tol, gmres_restart,
                                    gmres_max_iter, callback=_cb)
            if info != 0:
                warnings.warn(
                    f"GMRES-IR: inner GMRES did not fully converge for rhs "
                    f"column {j} (info={info}); using its last iterate anyway."
                )
            d2[:, j] = dj
            iters_this_round.append(counter[0])
        gmres_iters_history.append(iters_this_round)
        x2 = x2 + d2
    inner_s = time.perf_counter() - t0

    x = x2 if orig_ndim == 2 else x2[:, 0]

    extra = {
        "history": history,
        "true_err_history": true_err_history,
        "gmres_iters_history": gmres_iters_history,
        "mem_bytes": solver.factor_nbytes(),
        "factor_s": factor_s,
        "inner_s": inner_s,
        "factor_breakdown": getattr(solver, "factor_breakdown", lambda: None)(),
    }
    if hasattr(solver, "free"):
        solver.free()
    return x, extra


def solve_direct(solver_name, A, b, bs, dtype):
    """
    One factorization and one solve at `dtype`, with no refinement.

    Used for both reference variants: at complex128 it provides the accuracy
    ceiling, and at the low precision it provides the lower bound that
    refinement must improve upon.
    """
    t0 = time.perf_counter()
    solver = SOLVER_BUILDERS[solver_name](A, dtype, bs, b)
    factor_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = solver.solve(np.asarray(b, dtype=dtype)).astype(HIGH_DTYPE)
    inner_s = time.perf_counter() - t0

    mem_bytes = solver.factor_nbytes()
    factor_breakdown = getattr(solver, "factor_breakdown", lambda: None)()
    if hasattr(solver, "free"):
        solver.free()
    return x, {"mem_bytes": mem_bytes, "factor_s": factor_s, "inner_s": inner_s,
              "factor_breakdown": factor_breakdown}


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────

def _warm_up_gpu(solver_name, A, b, bs, low_dtype):
    """
    Perform one discarded factorization and solve outside the timed region.

    The first cuDSS call in a process pays a fixed start-up cost, for CUDA
    context creation and kernel compilation, that is essentially independent of
    problem size: measured at about 1.2 s for both n = 768 with 71k nonzeros
    and n = 2080 with 847k nonzeros. Without this warm-up that cost is charged
    in full to whichever variant runs first, which is the refinement variant,
    against subsequent variants doing comparable device work in 20 to 110 ms.
    Consuming it here leaves all three variants in the same warmed state, so
    their wall times are comparable.

    A failure here is ignored. The warm-up affects measurement only, not
    correctness; if the solver cannot be built, for want of a device or a
    package, the measurement loop reports it through its own skip path.
    """
    if solver_name not in _GPU_SOLVERS:
        return
    print("  Warming up GPU (untimed: CUDA context + cuDSS kernel JIT) ...",
          flush=True)
    try:
        warm = SOLVER_BUILDERS[solver_name](A, low_dtype, bs, b)
        warm.solve(np.asarray(b, dtype=low_dtype))
        if hasattr(warm, "free"):
            warm.free()
    except Exception as e:
        print(f"    warm-up skipped ({type(e).__name__}: {e})")
    gc.collect()


def _matrix_nbytes(A, dtype):
    """
    Bytes A occupies when held at `dtype`: its values at that precision, plus
    the index arrays, whose size does not depend on the precision.
    """
    values = A.nnz * np.dtype(dtype).itemsize
    return int(values + A.indices.nbytes + A.indptr.nbytes)


def _krylov_nbytes(n, restart):
    """
    Bytes of the Krylov basis one inner GMRES solve holds, at the working
    precision.

    SciPy's gmres takes a single right-hand side, so solve_gmres_ir loops over
    the columns and only one basis is resident at a time; the count does not
    scale with the number of right-hand sides. A restarted GMRES(m) holds m + 1
    basis vectors of length n.
    """
    return int((restart + 1) * n * np.dtype(HIGH_DTYPE).itemsize)


def _per_column(diff, denom):
    """
    Relative error per right-hand-side column.

    diff and denom are (n,) or (n, k). Returns a 1-D array of length k, with
    k = 1 for a single right-hand side.
    """
    if diff.ndim == 1:
        return np.array([np.linalg.norm(diff) / np.linalg.norm(denom)])
    return np.linalg.norm(diff, axis=0) / np.linalg.norm(denom, axis=0)


def benchmark_solver(fn, A_high, b_high, repeats, x_true=None, normA=None):
    """
    Run fn() `repeats` times and record accuracy, time and memory.

    Returns a list of dicts with the keys residual, true_err, eta1, eta2,
    etainf, omega, wall_s, extra and x, where

        residual = ||A x - b|| / ||b||, in the Frobenius norm when b has
                   several columns, so one aggregate number covers all of them
        true_err = ||x - x_true|| / ||x_true||, present only when x_true was
                   supplied through --reference-solver, otherwise None
        eta1/eta2/etainf = normwise backward error (Rigal-Gaches), in the 1-,
                   2- and infinity-norm; eta2 is None where normA[2] is None
                   (the spectral-norm estimate did not converge)
        omega    = componentwise backward error (Oettli-Prager)

    normA is {1: ..., 2: ..., inf: ...}, from bench_all._matrix_norm, computed
    once per problem in run_benchmarks and passed in here since it does not
    depend on the variant; see bench_all.backward_errors for the formulas.

    extra["per_rhs_residual"] and extra["per_rhs_true_err"] carry the same
    quantities broken out per right-hand-side column, so that a subset of
    columns solving markedly worse than the rest is visible; the aggregate
    figures conceal that.

    x is the solution of the last repeat, retained so that individual entries
    can be compared across variants.

    Memory is not instrumented here. The quantity the mixed-precision argument
    concerns is the size of the stored factorization, which every solver
    reports itself through factor_nbytes and which the drivers carry in
    extra["mem_bytes"]; see run_benchmarks for why the process-level figures
    were dropped.
    """
    norm_b = np.linalg.norm(b_high)
    norm_x_true = np.linalg.norm(x_true) if x_true is not None else None
    records = []
    for _ in range(repeats):
        gc.collect()

        t0 = time.perf_counter()
        x, extra = fn()
        wall = time.perf_counter() - t0

        res_vec = A_high @ x - b_high
        residual = np.linalg.norm(res_vec) / norm_b
        extra["per_rhs_residual"] = _per_column(res_vec, b_high)

        eta1 = eta2 = etainf = omega = None
        if normA is not None:
            _, etas, omega = backward_errors(A_high, x, b_high, normA, R=res_vec)
            eta1, eta2, etainf = etas[1], etas[2], etas[np.inf]

        true_err = None
        if x_true is not None:
            err_vec = x - x_true
            true_err = np.linalg.norm(err_vec) / norm_x_true
            extra["per_rhs_true_err"] = _per_column(err_vec, x_true)

        records.append({
            "residual":   residual,
            "true_err":   true_err,
            "eta1":       eta1,
            "eta2":       eta2,
            "etainf":     etainf,
            "omega":      omega,
            "wall_s":     wall,
            "extra":      extra,
            "x":          x,
        })
    return records


def run_benchmarks(h5path, idx, solver_name, bs, low_dtype, tol, max_iter, repeats,
                   reference_solver=None, inner="direct",
                   gmres_tol=1e-8, gmres_restart=30, gmres_max_iter=50):
    A, b = load_system(h5path, idx)
    A_high = A.tocsc().astype(HIGH_DTYPE)
    b_high = np.asarray(b, dtype=HIGH_DTYPE)

    # Matrix-only, so computed once and reused for every variant's backward
    # error; see bench_all._matrix_norm and bench_all.backward_errors.
    normA = {p: _matrix_norm(A_high, p) for p in NORMWISE_ORDS}
    if normA[2] is None:
        print(f"  [warning] svds did not converge for ||A||_2; eta2 is "
              f"skipped for every variant at this index")

    low_name = np.dtype(low_dtype).name
    indices, energies, valence, conduction = load_energy_metadata(h5path)
    energy = energy_of_idx(indices, energies, idx)
    energy_str = f"{energy:.4f} eV" if energy is not None else "unknown"
    edges = []
    if valence is not None:
        edges.append(f"valence={valence:.4f} eV")
    if conduction is not None:
        edges.append(f"conduction={conduction:.4f} eV")
    edge_str = f"  [{', '.join(edges)}]" if edges else ""

    print(f"Problem : {h5path.name}  E_{idx}  E={energy_str}{edge_str}  "
          f"n={A.shape[0]}  nnz={A.nnz}  b.shape={b.shape}")
    print(f"Solver  : {solver_name}   low_dtype={low_name}   high_dtype=complex128")

    if inner == "gmres":
        inner_label = "GMRES-IR"
        print(f"Inner   : GMRES(A) in complex128, preconditioned by {solver_name} "
              f"{low_name}   [gmres_tol={gmres_tol:.1e}  restart={gmres_restart}  "
              f"max_iter={gmres_max_iter}]")
    else:
        inner_label = "LU-IR"
        print(f"Inner   : single {low_name} triangular solve (classic LU-IR)")

    kappa = load_condition_number(h5path, idx)
    if kappa is not None:
        print(f"Condition number (kappa_2) at E_{idx}: {kappa:.3e}")
    else:
        cond_path = cli.CONDITION_DIR / f"{h5path.stem}.h5"
        print(f"Condition number: not available (no valid entry for idx={idx} "
              f"in {cond_path})")

    x_true = None
    if reference_solver is not None:
        print(f"Reference: x_true = {reference_solver} complex128", flush=True)
        try:
            ref = SOLVER_BUILDERS[reference_solver](A, HIGH_DTYPE, bs, b)
            x_true = ref.solve(b_high).astype(HIGH_DTYPE)
            if hasattr(ref, "free"):
                ref.free()
            ref_res = np.linalg.norm(A_high @ x_true - b_high) / np.linalg.norm(b_high)
            print(f"           ||A@x_true - b|| / ||b|| for x_true itself: {ref_res:.2e}")
            print(f"           x_true (shape {x_true.shape}, dtype {x_true.dtype}):")
        except (ImportError, TypeError, RuntimeError) as e:
            raise SystemExit(f"--reference-solver {reference_solver} failed: {e}")

    if repeats == 1:
        print("Runs    : 1 per variant\n")
    else:
        print(f"Runs    : {repeats} per variant (median reported)\n")

    # Consume the fixed device start-up cost before any variant is timed, so it
    # is not charged to whichever variant happens to run first.
    _warm_up_gpu(solver_name, A, b, bs, low_dtype)

    if inner == "gmres":
        ir_fn = lambda: solve_gmres_ir(solver_name, A, b, bs, low_dtype, tol, max_iter,
                                       x_true=x_true, gmres_tol=gmres_tol,
                                       gmres_restart=gmres_restart,
                                       gmres_max_iter=gmres_max_iter)
    else:
        ir_fn = lambda: solve_mixed_ir(solver_name, A, b, bs, low_dtype, tol, max_iter,
                                       x_true=x_true)

    # The third entry of each variant records what that method must hold
    # besides its factorization, for the working-set row of the report:
    #
    #   matrix_dtype  the precision A itself is held at. Refinement forms its
    #                 residual at the working precision, so it must keep A at
    #                 complex128 however low u_f is; a bare low-precision solve
    #                 needs A only at u_f. This is why the working set does not
    #                 halve even when the factorization does.
    #   krylov        True where the inner solve builds a Krylov basis, whose
    #                 size is added on top; see _krylov_nbytes.
    variants = [
        (f"{solver_name} {low_name} + {inner_label}", ir_fn,
         {"matrix_dtype": HIGH_DTYPE, "krylov": inner == "gmres"}),
        (f"{solver_name} complex128 (direct)",
         lambda: solve_direct(solver_name, A, b, bs, HIGH_DTYPE),
         {"matrix_dtype": HIGH_DTYPE, "krylov": False}),
        (f"{solver_name} {low_name} (no refine)",
         lambda: solve_direct(solver_name, A, b, bs, low_dtype),
         {"matrix_dtype": low_dtype, "krylov": False}),
    ]
    variant_info = {name: info for name, _fn, info in variants}

    all_records = {}
    for name, fn, _info in variants:
        print(f"  Benchmarking '{name}' x{repeats} ...", flush=True)
        try:
            all_records[name] = benchmark_solver(fn, A_high, b_high, repeats,
                                                 x_true=x_true, normA=normA)
        except (ImportError, TypeError, RuntimeError) as e:
            print(f"    skipped: {e}")
    print()

    if not all_records:
        raise SystemExit("No variant ran successfully -- see skip messages above.")

    names = list(all_records.keys())
    col = 32

    def med(name, key):
        return float(np.median([r[key] for r in all_records[name]]))

    def med_extra(name, key):
        vals = [r["extra"][key] for r in all_records[name] if key in r["extra"]]
        return float(np.median(vals)) if vals else None

    def med_opt(name, key):
        """Like med(), but None-safe for a top-level key that may be None
        (eta2 when svds did not converge, or omega/eta1/etainf when normA
        was never computed)."""
        vals = [r[key] for r in all_records[name] if r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    header = f"{'Metric':<{col}}" + "".join(f"  {nm:<28}" for nm in names)
    print(header)
    print("─" * len(header))

    have_x_true = all_records[names[0]][0]["true_err"] is not None

    # 1. Accuracy: residual, forward error, then the two backward errors.
    rows = [
        ("Relative residual ||Ax-b||/||b||", "residual",    "{:.2e}"),
    ]
    if have_x_true:
        rows.append(("Forward error (ferr) ||x-x_true||/||x_true||", "true_err", "{:.2e}"))
    for label, key, fmt in rows:
        vals = [fmt.format(med(nm, key)) if isinstance(fmt, str) else fmt(med(nm, key))
                for nm in names]
        print(f"  {label:<{col-2}}" + "".join(f"  {v:<28}" for v in vals))

    for label, key in [("Normwise backward error (nbe, eta_1)",   "eta1"),
                       ("Normwise backward error (nbe, eta_2)",   "eta2"),
                       ("Normwise backward error (nbe, eta_inf)", "etainf")]:
        vals = [f"{med_opt(nm, key):.2e}" if med_opt(nm, key) is not None
                else "n/a" for nm in names]
        print(f"  {label:<{col-2}}" + "".join(f"  {v:<28}" for v in vals))

    omega_vals = [f"{med_opt(nm, 'omega'):.2e}" if med_opt(nm, "omega") is not None
                  else "n/a" for nm in names]
    print(f"  {'Componentwise backward error (cbe: omega)':<{col-2}}" +
          "".join(f"  {v:<28}" for v in omega_vals))

    # 2. Timing: wall time, then factorization (with its symbolic/numeric
    # split where available), then solve/inner-loop time.
    wall_vals = [f"{med(nm, 'wall_s')*1e3:.1f}" for nm in names]
    print(f"  {'Wall time (ms)':<{col-2}}" + "".join(f"  {v:<28}" for v in wall_vals))

    # Factorization time is the solver construction call; solve/inner-loop time
    # is everything after it -- for the two solve_direct variants, the single
    # solve() call, and for the refinement variant, the initial low-precision
    # solve plus every outer iteration (residual, correction solve, update).
    # The two need not sum exactly to "Wall time" above: dtype casts and array
    # setup outside both timed blocks are not attributed to either.
    factor_vals = [f"{med_extra(nm, 'factor_s')*1e3:.1f}" if med_extra(nm, "factor_s") is not None
                   else "n/a" for nm in names]
    print(f"  {'Factorization time (ms)':<{col-2}}" +
          "".join(f"  {v:<28}" for v in factor_vals))

    # Symbolic (reordering) vs. numeric factorization -- a breakdown of the
    # row directly above, where the backend keeps the two separate; scipy's
    # splu, UMFPACK and python-mumps fuse both into one call and expose no
    # split, so this is only populated for cuDSS.
    breakdowns = {nm: all_records[nm][0]["extra"].get("factor_breakdown")
                 for nm in names}
    if any(bd is not None for bd in breakdowns.values()):
        for label, i in [("    symbolic (ms)", 0), ("    numeric (ms)", 1)]:
            vals = [f"{breakdowns[nm][i]*1e3:.1f}" if breakdowns[nm] is not None
                    else "n/a (fused)" for nm in names]
            print(f"  {label:<{col-2}}" + "".join(f"  {v:<28}" for v in vals))

    solve_vals = [f"{med_extra(nm, 'inner_s')*1e3:.1f}" if med_extra(nm, "inner_s") is not None
                  else "n/a" for nm in names]
    print(f"  {'Solve / inner-loop time (ms)':<{col-2}}" +
          "".join(f"  {v:<28}" for v in solve_vals))

    # 3. Memory: the size of the stored factorization, which is the quantity
    # the mixed-precision argument is about, and its ratio to the complex128
    # factorization refinement is meant to replace.
    #
    # Process-level figures (peak Python heap, peak RSS) are deliberately not
    # reported. Neither is comparable across the solvers benchmarked here: the
    # Block Thomas factors are NumPy arrays and so are visible to tracemalloc,
    # while SuperLU, UMFPACK and MUMPS hold theirs in compiled extensions and
    # cuDSS holds its on the device, none of which the Python heap observes.
    # Peak RSS is additionally order-dependent, since the allocator does not
    # return freed memory to the operating system: whichever variant runs first
    # is charged the whole growth and the rest measure zero.
    MIB = 1024.0**2
    mem = {nm: all_records[nm][0]["extra"].get("mem_bytes", 0) for nm in names}

    # The complex128 direct variant is the factorization refinement is meant to
    # replace, so it is the denominator that makes each saving readable.
    ref = next((nm for nm in names if "complex128 (direct)" in nm), None)

    def ratio_row(label, size):
        if ref is None or not size.get(ref):
            return
        vals = [f"{size[nm]/size[ref]:.2f}x" if size[nm] else "n/a"
                for nm in names]
        print(f"  {label:<{col-2}}" + "".join(f"  {v:<28}" for v in vals))

    print(f"  {'Factor memory (MiB, factor_nbytes)':<{col-2}}" +
          "".join(f"  {mem[nm]/MIB:<28.2f}" for nm in names))
    ratio_row("  relative to complex128", mem)

    # The factorization is not the whole footprint. Refinement computes its
    # residual at the working precision, so it holds A at complex128 however
    # low u_f is, and GMRES-IR additionally holds a Krylov basis. Reporting the
    # factor alone therefore overstates what mixed precision saves: the factor
    # halves, the working set does not.
    working = {}
    for nm in names:
        info = variant_info.get(nm, {"matrix_dtype": HIGH_DTYPE, "krylov": False})
        total = mem[nm] + _matrix_nbytes(A_high, info["matrix_dtype"])
        if info["krylov"]:
            total += _krylov_nbytes(A_high.shape[0], gmres_restart)
        working[nm] = total

    print(f"  {'Working set (MiB, factor + A + basis)':<{col-2}}" +
          "".join(f"  {working[nm]/MIB:<28.2f}" for nm in names))
    ratio_row("  relative to complex128", working)

    n_rhs = b_high.shape[1] if b_high.ndim == 2 else 1
    if n_rhs > 1:
        print(f"\n  Per-RHS-column breakdown (first run, {n_rhs} right-hand sides):")
        for nm in names:
            extra0 = all_records[nm][0]["extra"]
            per_res = extra0.get("per_rhs_residual")
            per_err = extra0.get("per_rhs_true_err")
            print(f"    {nm}")
            for j in range(n_rhs):
                line = f"      rhs {j}: ||Ax-b||/||b|| = {per_res[j]:.3e}"
                if per_err is not None:
                    line += f"   ||x-x_true||/||x_true|| = {per_err[j]:.3e}"
                print(line)

    ir_name = names[0]
    history = all_records[ir_name][0]["extra"].get("history", [])
    true_err_history = all_records[ir_name][0]["extra"].get("true_err_history", [])
    gmres_iters_history = all_records[ir_name][0]["extra"].get("gmres_iters_history", [])
    if history:
        print(f"\n  IR convergence history (first run, {ir_name}):")
        for i, rel in enumerate(history):
            tag = "  <- converged" if i == len(history) - 1 and rel < tol else ""
            line = f"    iter {i}: ||r||/||b|| = {rel:.3e}"
            if i < len(true_err_history):
                line += f"   ||x-x_true||/||x_true|| = {true_err_history[i]:.3e}"
            if i < len(gmres_iters_history):
                iters = gmres_iters_history[i]
                iters_str = iters[0] if len(iters) == 1 else iters
                line += f"   inner GMRES iters = {iters_str}"
            print(line + tag)

    probe_idx = 100
    if A.shape[0] > probe_idx:
        print(f"\n  x[{probe_idx}] comparison across variants (first run):")
        for nm in names:
            xv = all_records[nm][0]["x"]
            val = xv[probe_idx] if xv.ndim == 1 else xv[probe_idx, 0]
            print(f"    {nm:<35} x[{probe_idx}] = {val!r}")

    nnz = int(A.nnz)
    n = A.shape[0]
    mib64 = nnz * 16 / 1024**2   # complex128 = 16 bytes/entry
    mib32 = nnz * 8 / 1024**2    # complex64  = 8 bytes/entry
    idx_mib = (nnz + n + 1) * 4 / 1024**2
    print(f"\n  Theoretical matrix value storage ({nnz} nonzeros, indices excluded):")
    print(f"    complex128 : {mib64:.3f} MiB")
    print(f"    complex64  : {mib32:.3f} MiB  (50% saving on values)")
    print(f"    Index arrays (shared): ~{idx_mib:.3f} MiB")

    return all_records


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap)
    cli.add_index_selection(ap, default_all=True)
    ap.add_argument("--energy", type=float, nargs="+", default=None,
                    metavar="EV",
                    help="one or more energies in eV; each is resolved to "
                         "the index with the nearest recorded energy. "
                         "Mutually exclusive with --idx/--start/--end.")
    cli.add_solver_selection(ap, choices=tuple(SOLVER_BUILDERS),
                             default="superlu", multiple=False)
    cli.add_block_partition(ap, auto=False)
    cli.add_factor_dtype(
        ap, default="complex64",
        help="precision of the low-precision factorization, u_f "
             "(default: complex64)")
    ap.add_argument("--inner", choices=["direct", "gmres"], default="direct",
                    help="inner correction solve: 'direct' is classic LU-IR, "
                         "a single low-precision triangular solve; 'gmres' is "
                         "GMRES-IR, GMRES in complex128 preconditioned by the "
                         "low-precision factorization")
    ap.add_argument("--tol", type=float, default=1e-14,
                    help="outer convergence tolerance on ||r||/||b||")
    ap.add_argument("--max-iter", type=int, default=10, metavar="N",
                    help="maximum outer refinement iterations")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats per variant; the median is reported")
    ap.add_argument("--reference-solver", choices=["superlu", "mumps"],
                    default="superlu", metavar="NAME",
                    help="compute x_true with this solver at complex128 and "
                         "report ||x - x_true||/||x_true|| for every variant")
    ap.add_argument("--gmres-tol", type=float, default=1e-8,
                    help="relative tolerance of the inner GMRES solve "
                         "(--inner gmres only)")
    ap.add_argument("--gmres-restart", type=int, default=30,
                    help="inner GMRES restart parameter (--inner gmres only)")
    ap.add_argument("--gmres-max-iter", type=int, default=50,
                    help="maximum inner GMRES iterations per outer step "
                         "(--inner gmres only)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    if args.energy is not None:
        if args.idx is not None or args.start is not None:
            ap.error("--energy is mutually exclusive with --idx/--start/--end")
        file_indices, file_energies, _, _ = load_energy_metadata(h5path)
        indices = [idx_of_energy(file_indices, file_energies, e)
                  for e in args.energy]
    else:
        indices = cli.resolve_indices(ap, args)
    factor_dtype = np.dtype(args.factor_dtype)

    for idx in indices:
        if len(indices) > 1:
            print("=" * 78)
        run_benchmarks(h5path, idx, args.solver, args.block_size,
                       factor_dtype, args.tol, args.max_iter, args.repeats,
                       reference_solver=args.reference_solver,
                       inner=args.inner, gmres_tol=args.gmres_tol,
                       gmres_restart=args.gmres_restart,
                       gmres_max_iter=args.gmres_max_iter)


if __name__ == "__main__":
    main()
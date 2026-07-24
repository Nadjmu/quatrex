#!/usr/bin/env python3
"""
Mixed-Precision Iterative Refinement on real material data
============================================================
Generalizes the original random-data LU-IR benchmark in several ways:

  1. The matrix and RHS come from a real energy index in a material's HDF5
     file (E_<idx>/M, E_<idx>/rhs) instead of a synthetic random system.
  2. The low-precision factorization/solve step is any solver from
     solver_classes.py, selected via --solver, instead of hardcoded SuperLU.
  3. The inner correction solve can be either:
       - "direct"  : a single low-precision triangular solve (classic LU-IR,
                     Buttari et al. 2006)
       - "gmres"   : GMRES run in HIGH precision on the full operator A,
                     preconditioned by the low-precision factorization
                     applied via solver.solve() (GMRES-IR, Carson & Higham
                     2017-style). Selected with --inner gmres.

Compares three variants of the SAME chosen solver family (apples-to-apples,
unlike comparing across different solver implementations):

  1. <solver> at --low-dtype + fp64 iterative refinement   (this work;
     "direct" -> LU-IR, "gmres" -> GMRES-IR, per --inner)
  2. <solver> at complex128, direct (no refinement)         (baseline / reference)
  3. <solver> at --low-dtype, direct (no refinement)         (lower bound)

LU-IR algorithm (Buttari et al. 2006, generalized to complex data and to any
solver with a .solve(b) method and a factorization built once in __init__):
  1. Build the solver at low precision -- this is the one-time factor cost
  2. x0 = solver.solve(b)                     (low precision, cast to complex128)
  3. r  = b - A @ x                            (complex128 residual)
  4. dx = solver.solve(r)                      (low-precision correction solve,
                                                 reusing the SAME factorization)
  5. x += dx; repeat until ||r||/||b|| < tol or max_iter reached

GMRES-IR algorithm (same outer loop, different inner correction step):
  1-3. same as above
  4. dx = GMRES(A, r) in complex128, preconditioned by M^{-1}v = solver.solve(v)
     (v cast to low precision, result cast back to complex128) -- GMRES itself
     runs at full precision, only the preconditioner applications are cheap
     low-precision triangular solves reusing the SAME factorization.
  5. x += dx; repeat until ||r||/||b|| < tol or max_iter reached

  Note: GMRES-IR's preconditioner only needs the factorization's action as
  an operator (solver.solve), not explicit access to L/U factors. This means
  it works uniformly for superlu, block_thomas, mumps, AND cudss -- none of
  which need to expose their factors for GMRES to use them as a preconditioner.

Solvers usable via --solver (all need a persistent, reusable factorization):
  superlu, umfpack, mumps, block_thomas   -- CPU, always available
  cudss                                   -- GPU (needs nvmath-python + a
                                              visible CUDA device); skipped
                                              gracefully (like a missing
                                              package) if no GPU is present.

Note: UMFPACK has no single-precision build (solver_classes.UMFPACK raises
TypeError for --low-dtype complex64) -- this is a real library limitation,
not a bug here; pick a different --solver for a low-precision comparison.

GPU timing note: the first cuDSS solve in a process pays a large one-time
CUDA/cuDSS kernel-JIT + context cost that is independent of problem size.
Left alone it lands entirely on whichever variant runs first (LU-IR), making
it look ~1.2s slower than it really is. run_benchmarks() therefore does an
untimed warm-up solve before any variant is benchmarked -- see _warm_up_gpu.

Usage:
    python mpir.py /scratch/yimili/matrices/hdf5/carbon-nanotube.h5 \\
        --idx 5 --solver superlu --low-dtype complex64

    python mpir.py /scratch/yimili/matrices/hdf5/carbon-nanotube.h5 \\
        --idx 5 --solver block_thomas --bs 32 --low-dtype complex64

    # GMRES-IR: GMRES in complex128, preconditioned by the low-precision LU
    python mpir.py /scratch/yimili/matrices/hdf5/si-bulk.h5 \\
        --idx 254 --solver mumps --low-dtype complex64 --inner gmres \\
        --gmres-tol 1e-8 --gmres-restart 30 --gmres-maxiter 50

    # cuDSS on GPU
    python mpir.py /scratch/yimili/matrices/hdf5/si-bulk.h5 \\
        --idx 254 --solver cudss --low-dtype complex64
"""

import argparse
import gc
import sys
import threading
import time
import tracemalloc
import warnings
from pathlib import Path

import h5py
import numpy as np
import psutil
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))
from solver_classes import (
    SparseLU, UMFPACK, MUMPS, BlockThomas, CuDSS, extract_blocks_sparse,
)

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

HIGH_DTYPE = np.complex128   # "full precision" reference throughout


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers (unchanged from the original random-data script)
# ─────────────────────────────────────────────────────────────────────────────

def _rss_mb():
    return psutil.Process().memory_info().rss / 1024**2


class _PeakRSSTracker:
    """Polls process RSS every `interval` seconds in a background thread.
    Use as a context manager; read .peak_mb after exit."""
    def __init__(self, interval=0.005):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def _poll(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, _rss_mb())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_mb = _rss_mb()
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()


# ─────────────────────────────────────────────────────────────────────────────
# Solver registry -- maps --solver name to a builder(A, dtype, bs, b) -> instance
#
# `b` is only used by the cuDSS builder (which needs the RHS column count at
# construction time -- see _CuDSSSolver); every other builder ignores it.
# It's threaded through uniformly so the call sites don't special-case cuDSS.
# ─────────────────────────────────────────────────────────────────────────────

def _build_block_thomas(A, dtype, bs, b):
    if bs is None:
        raise ValueError("--bs is required for --solver block_thomas")
    D, L, U = extract_blocks_sparse(A, bs)
    return BlockThomas(D, L, U, dtype=dtype)


def _solve_columns(solve_one, b):
    """Apply a single-RHS solve function column-by-column to a 1-D or 2-D b."""
    b = np.asarray(b)
    if b.ndim == 1:
        return solve_one(b)
    return np.column_stack(
        [solve_one(np.ascontiguousarray(b[:, j])) for j in range(b.shape[1])]
    )


class _CuDSSSolver:
    """
    Wrapper around solver_classes.CuDSS for use inside this benchmark.

    Two things it handles that plain CuDSS doesn't:

    1. RHS SHAPE BINDING. CuDSS binds its RHS shape (n, nrhs) at construction,
       and nvmath's reset_operands() rejects any later array whose shape or
       strides differ from that original binding. This class allocates ONE
       Fortran-ordered (n, nrhs) buffer up front and copies each RHS into it,
       so every solve hands nvmath a byte-layout-identical array and the
       expensive rebuild-and-refactorize fallback inside CuDSS.solve() never
       triggers.

    2. BLOCK VS COLUMN SOLVES. The whole (n, k) RHS is solved in ONE cuDSS
       call whenever k matches the nrhs it was built with -- that is the
       common case for LU-IR and the direct variants, where every solve uses
       the same column count as the original b. Only GMRES-IR needs
       single-vector solves (scipy's gmres applies the preconditioner one
       vector at a time), so a second nrhs=1 solver is built LAZILY and only
       then. Column-looping every solve unconditionally would multiply the
       number of GPU round trips by k for no reason.
    """

    def __init__(self, A, dtype, nrhs):
        self.n = A.shape[0]
        self.dtype = np.dtype(dtype)
        self.nrhs = max(int(nrhs), 1)
        self._A = A
        self._solver = CuDSS(A, dtype=self.dtype, nrhs=self.nrhs)
        self._buf = np.empty((self.n, self.nrhs), dtype=self.dtype, order="F")
        self._one = None        # lazily built nrhs=1 solver (GMRES-IR only)
        self._one_buf = None

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


SOLVER_BUILDERS = {
    "superlu":      lambda A, dtype, bs, b: SparseLU(A, dtype=dtype),
    "umfpack":      lambda A, dtype, bs, b: UMFPACK(A, dtype=dtype),
    "mumps":        lambda A, dtype, bs, b: MUMPS(A, dtype=dtype),
    "block_thomas": _build_block_thomas,
    "cudss":        lambda A, dtype, bs, b: _CuDSSSolver(
        A, dtype, b.shape[1] if np.asarray(b).ndim == 2 else 1),
}

# Solvers whose first call in a process pays a large fixed GPU start-up cost.
_GPU_SOLVERS = ("cudss",)


# ─────────────────────────────────────────────────────────────────────────────
# Real-data loading (replaces make_problem)
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


def load_condition_number(h5path, idx):
    """
    global/condition_full_svd is an array indexed the same way as
    metadata/indices / metadata/energies -- one condition number per energy
    index, at the SAME level as E_<idx> (a sibling, not nested under it).
    Returns None if the dataset or this particular idx isn't present.
    """
    with h5py.File(h5path, "r") as f:
        if "global/condition_full_svd" not in f:
            return None
        kappa_arr = f["global/condition_full_svd"][:]
    return float(kappa_arr[idx]) if 0 <= idx < len(kappa_arr) else None


# ─────────────────────────────────────────────────────────────────────────────
# Solvers (generalized: any registered solver family, any low precision)
# ─────────────────────────────────────────────────────────────────────────────

def solve_mixed_ir(solver_name, A, b, bs, low_dtype, tol, max_iter, x_true=None):
    """
    Classic LU-IR: the correction solve dx = solver.solve(r) is a single
    low-precision triangular solve.

    x_true: optional complex128 reference solution. If given, the relative
    true error ||x - x_true|| / ||x_true|| is recorded at every iteration
    (same point where the residual is already computed) purely for
    reporting -- it does NOT feed into the convergence check or the loop's
    control flow, so it can't distort the residual/timing/memory metrics.
    """
    b_high = np.asarray(b, dtype=HIGH_DTYPE)
    A_high = A.tocsc().astype(HIGH_DTYPE)
    norm_b = np.linalg.norm(b_high)
    norm_x_true = np.linalg.norm(x_true) if x_true is not None else None
    history = []
    true_err_history = []

    solver = SOLVER_BUILDERS[solver_name](A, low_dtype, bs, b)
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

    extra = {
        "history": history,
        "true_err_history": true_err_history,
        "mem_bytes": solver.factor_nbytes(),
    }
    if hasattr(solver, "free"):
        solver.free()
    return x, extra


def _gmres_solve(A_op, rhs, M_op, tol, restart, maxiter, callback):
    """Thin wrapper around scipy.sparse.linalg.gmres that works across scipy
    versions (the tol -> rtol rename happened in scipy 1.12)."""
    try:
        return spla.gmres(A_op, rhs, M=M_op, rtol=tol, atol=0.0, restart=restart,
                          maxiter=maxiter, callback=callback, callback_type="pr_norm")
    except TypeError:
        return spla.gmres(A_op, rhs, M=M_op, tol=tol, atol=0.0, restart=restart,
                          maxiter=maxiter, callback=callback, callback_type="pr_norm")


def solve_gmres_ir(solver_name, A, b, bs, low_dtype, tol, max_iter, x_true=None,
                   gmres_tol=1e-8, gmres_restart=30, gmres_maxiter=50):
    """
    GMRES-IR: the correction solve A @ dx = r is no longer a single
    low-precision triangular solve -- it's GMRES running in HIGH precision
    on the full operator A, left-preconditioned by M^{-1}v = solver.solve(v)
    (v cast to low precision, result cast back to complex128 before being
    handed to GMRES). GMRES only needs the factorization's action as an
    operator, not explicit L/U factors, so this works for superlu,
    block_thomas, mumps, AND cudss alike.

    scipy's gmres only accepts a single RHS vector, so multi-column b is
    handled by looping GMRES over columns; the initial low-precision direct
    solve and the residual/error bookkeeping stay vectorized as before.

    x_true: see solve_mixed_ir -- same reporting-only true-error tracking.
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

    solver = SOLVER_BUILDERS[solver_name](A, low_dtype, bs, b)

    def precond_apply(v):
        return solver.solve(v.astype(low_dtype)).astype(HIGH_DTYPE)

    A_op = spla.LinearOperator((n, n), matvec=lambda v: A_high @ v, dtype=HIGH_DTYPE)
    M_op = spla.LinearOperator((n, n), matvec=precond_apply, dtype=HIGH_DTYPE)

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
                                    gmres_maxiter, callback=_cb)
            if info != 0:
                warnings.warn(
                    f"GMRES-IR: inner GMRES did not fully converge for rhs "
                    f"column {j} (info={info}); using its last iterate anyway."
                )
            d2[:, j] = dj
            iters_this_round.append(counter[0])
        gmres_iters_history.append(iters_this_round)
        x2 = x2 + d2

    x = x2 if orig_ndim == 2 else x2[:, 0]

    extra = {
        "history": history,
        "true_err_history": true_err_history,
        "gmres_iters_history": gmres_iters_history,
        "mem_bytes": solver.factor_nbytes(),
    }
    if hasattr(solver, "free"):
        solver.free()
    return x, extra


def solve_direct(solver_name, A, b, bs, dtype):
    solver = SOLVER_BUILDERS[solver_name](A, dtype, bs, b)
    x = solver.solve(np.asarray(b, dtype=dtype)).astype(HIGH_DTYPE)
    mem_bytes = solver.factor_nbytes()
    if hasattr(solver, "free"):
        solver.free()
    return x, {"mem_bytes": mem_bytes}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def _warm_up_gpu(solver_name, A, b, bs, low_dtype):
    """
    Do one THROWAWAY factor+solve outside the timed region.

    Why this exists: the first cuDSS call in a process pays a large fixed
    start-up cost (CUDA context creation, cuDSS kernel JIT/caching) that is
    essentially INDEPENDENT of problem size -- empirically ~1.2s whether
    n=768/nnz=71k or n=2080/nnz=847k. Without this warm-up that cost is
    silently charged to whichever variant happens to run first (LU-IR),
    which made LU-IR look ~1.2s slower than the direct variants even though
    the direct variants were doing comparable GPU work right afterwards at
    ~20-110ms. Burning it here makes all three variants start from the same
    warmed-up state, so their reported wall times are actually comparable.

    Failures are swallowed: a warm-up is an optimization, not a correctness
    step. If the solver can't be built (no GPU, missing package), the real
    benchmark loop below will report that properly via its own skip path.
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


def _per_column(diff, denom):
    """Relative error per RHS column. diff, denom: (n,) or (n, k) arrays.
    Returns a 1-D array of length k (k=1 for a single RHS)."""
    if diff.ndim == 1:
        return np.array([np.linalg.norm(diff) / np.linalg.norm(denom)])
    return np.linalg.norm(diff, axis=0) / np.linalg.norm(denom, axis=0)


def benchmark_solver(fn, A_high, b_high, repeats, x_true=None):
    """
    Run fn() `repeats` times. Returns list of dicts with keys:
    residual, true_err, wall_s, peak_py_mb, net_rss_mb, extra, x.

    residual = ||A@x - b|| / ||b||                (Frobenius norm if b has
               multiple RHS columns -- one aggregate number across all of them)
    true_err = ||x - x_true|| / ||x_true||         (only if x_true is given --
               see --reference-solver; None otherwise)

    extra["per_rhs_residual"] / extra["per_rhs_true_err"] hold the SAME
    errors broken out per RHS column (length-1 array if b is a single
    vector), for inspecting whether some right-hand sides solve much worse
    than others -- the aggregate numbers above can hide that.

    x is the full solution array of the LAST repeat (kept around so callers
    can inspect/compare individual entries, e.g. x[100], across variants).
    """
    norm_b = np.linalg.norm(b_high)
    norm_x_true = np.linalg.norm(x_true) if x_true is not None else None
    records = []
    for _ in range(repeats):
        gc.collect()
        baseline_rss = _rss_mb()

        tracemalloc.start()
        with _PeakRSSTracker() as tracker:
            t0 = time.perf_counter()
            x, extra = fn()
            wall = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        res_vec = A_high @ x - b_high
        residual = np.linalg.norm(res_vec) / norm_b
        extra["per_rhs_residual"] = _per_column(res_vec, b_high)

        true_err = None
        if x_true is not None:
            err_vec = x - x_true
            true_err = np.linalg.norm(err_vec) / norm_x_true
            extra["per_rhs_true_err"] = _per_column(err_vec, x_true)

        records.append({
            "residual":   residual,
            "true_err":   true_err,
            "wall_s":     wall,
            "peak_py_mb": peak_bytes / 1024**2,
            "net_rss_mb": tracker.peak_mb - baseline_rss,
            "extra":      extra,
            "x":          x,
        })
    return records


def run_benchmarks(h5path, idx, solver_name, bs, low_dtype, tol, max_iter, repeats,
                   reference_solver=None, inner="direct",
                   gmres_tol=1e-8, gmres_restart=30, gmres_maxiter=50):
    A, b = load_system(h5path, idx)
    A_high = A.tocsc().astype(HIGH_DTYPE)
    b_high = np.asarray(b, dtype=HIGH_DTYPE)

    low_name = np.dtype(low_dtype).name
    print(f"Problem : {h5path.name}  E_{idx}  n={A.shape[0]}  nnz={A.nnz}  "
          f"b.shape={b.shape}")
    print(f"Solver  : {solver_name}   low_dtype={low_name}   high_dtype=complex128")

    if inner == "gmres":
        inner_label = "GMRES-IR"
        print(f"Inner   : GMRES(A) in complex128, preconditioned by {solver_name} "
              f"{low_name}   [gmres_tol={gmres_tol:.1e}  restart={gmres_restart}  "
              f"maxiter={gmres_maxiter}]")
    else:
        inner_label = "LU-IR"
        print(f"Inner   : single {low_name} triangular solve (classic LU-IR)")

    kappa = load_condition_number(h5path, idx)
    if kappa is not None:
        print(f"Condition number (full SVD) at E_{idx}: {kappa:.3e}")
    else:
        print(f"Condition number: not available (no global/condition_full_svd "
              f"entry for idx={idx})")

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

    # Burn the one-time GPU start-up cost BEFORE any variant is timed, so it
    # doesn't get charged to whichever variant happens to run first.
    _warm_up_gpu(solver_name, A, b, bs, low_dtype)

    if inner == "gmres":
        ir_fn = lambda: solve_gmres_ir(solver_name, A, b, bs, low_dtype, tol, max_iter,
                                       x_true=x_true, gmres_tol=gmres_tol,
                                       gmres_restart=gmres_restart,
                                       gmres_maxiter=gmres_maxiter)
    else:
        ir_fn = lambda: solve_mixed_ir(solver_name, A, b, bs, low_dtype, tol, max_iter,
                                       x_true=x_true)

    variants = [
        (f"{solver_name} {low_name} + {inner_label}", ir_fn),
        (f"{solver_name} complex128 (direct)",
         lambda: solve_direct(solver_name, A, b, bs, HIGH_DTYPE)),
        (f"{solver_name} {low_name} (no refine)",
         lambda: solve_direct(solver_name, A, b, bs, low_dtype)),
    ]

    all_records = {}
    for name, fn in variants:
        print(f"  Benchmarking '{name}' x{repeats} ...", flush=True)
        try:
            all_records[name] = benchmark_solver(fn, A_high, b_high, repeats, x_true=x_true)
        except (ImportError, TypeError, RuntimeError) as e:
            print(f"    skipped: {e}")
    print()

    if not all_records:
        raise SystemExit("No variant ran successfully -- see skip messages above.")

    names = list(all_records.keys())
    col = 32

    def med(name, key):
        return float(np.median([r[key] for r in all_records[name]]))

    header = f"{'Metric':<{col}}" + "".join(f"  {nm:<28}" for nm in names)
    print(header)
    print("─" * len(header))

    have_x_true = all_records[names[0]][0]["true_err"] is not None
    rows = [
        ("Relative residual ||Ax-b||/||b||", "residual",    "{:.2e}"),
    ]
    if have_x_true:
        rows.append(("True error ||x-x_true||/||x_true||", "true_err", "{:.2e}"))
    rows += [
        ("Wall time (ms)",                   "wall_s",      lambda v: f"{v*1e3:.1f}"),
        ("Peak Python heap (MiB)",           "peak_py_mb",  "{:.2f}"),
        ("Peak RSS above baseline (MiB)",    "net_rss_mb",  "{:.1f}"),
    ]
    for label, key, fmt in rows:
        vals = [fmt.format(med(nm, key)) if isinstance(fmt, str) else fmt(med(nm, key))
                for nm in names]
        print(f"  {label:<{col-2}}" + "".join(f"  {v:<28}" for v in vals))

    mem_row = [f"{all_records[nm][0]['extra'].get('mem_bytes', 0)/1e6:.2f}"
               for nm in names]
    print(f"  {'Factor memory (MB, factor_nbytes)':<{col-2}}" +
          "".join(f"  {v:<28}" for v in mem_row))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("h5path", type=Path, help="material HDF5 file")
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--solver", choices=list(SOLVER_BUILDERS), default="superlu")
    parser.add_argument("--bs", type=int, default=None,
                        help="block size, required for --solver block_thomas")
    parser.add_argument("--low-dtype", choices=["complex64", "complex128"],
                        default="complex64")
    parser.add_argument("--tol", type=float, default=1e-14)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--reference-solver", choices=["superlu", "mumps"], default="superlu",
                        help="compute x_true via this solver at complex128 and report "
                             "||x-x_true||/||x_true|| for every variant (default: off, "
                             "residual-only)")
    parser.add_argument("--inner", choices=["direct", "gmres"], default="direct",
                        help="inner correction solve: 'direct' = classic LU-IR "
                             "(single low-precision triangular solve, default); "
                             "'gmres' = GMRES-IR (GMRES in complex128 preconditioned "
                             "by the low-precision factorization)")
    parser.add_argument("--gmres-tol", type=float, default=1e-8,
                        help="relative tolerance for the inner GMRES solve "
                             "(--inner gmres only)")
    parser.add_argument("--gmres-restart", type=int, default=30,
                        help="GMRES restart parameter (--inner gmres only)")
    parser.add_argument("--gmres-maxiter", type=int, default=50,
                        help="max GMRES (restart cycles/iterations) per outer IR "
                             "step (--inner gmres only)")
    args = parser.parse_args()

    run_benchmarks(args.h5path, args.idx, args.solver, args.bs,
                   np.dtype(args.low_dtype), args.tol, args.max_iter, args.repeats,
                   reference_solver=args.reference_solver, inner=args.inner,
                   gmres_tol=args.gmres_tol, gmres_restart=args.gmres_restart,
                   gmres_maxiter=args.gmres_maxiter)


if __name__ == "__main__":
    main()
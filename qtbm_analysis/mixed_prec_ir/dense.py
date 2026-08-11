"""
Mixed-precision iterative refinement on a dense system, using LAPACK directly.

Input
-----
A dense matrix and right-hand side stored as .npy arrays, at MATRIX_PATH and
RHS_PATH.

Variants compared
-----------------
    fp32-IR   sgetrf for the factorization, residuals in fp64, sgetrs for the
              corrections. This is the canonical mixed-precision algorithm, the
              one LAPACK exposes as dsgesv. SciPy does not wrap dsgesv, but
              sgetrf followed by sgetrs is the same computation.
    fp64      dgetrf and dgetrs, no refinement. The accuracy reference.
    fp64-IR   dgetrf with refinement, so that the overhead of the refinement
              loop is measured with the factorization precision held equal to
              the working precision. Any difference against fp64 is the cost of
              refinement alone.

Half precision is absent because CPU LAPACK provides no fp16 factorization; it
requires cuSOLVER or MAGMA on a device. The half-precision study is therefore
carried out with the Block Thomas implementations in solvers/solver_classes.py,
which simulate fp16 arithmetic in NumPy.

Algorithm
---------
The refinement loop is the classical one:

    factorize A in the low precision, once
    x  = solve(b) in the low precision, promoted to fp64
    repeat:
        r  = b - A x           computed in fp64
        dx = solve(r)          in the low precision, reusing the factorization
        x  = x + dx
    until ||r|| / ||b|| < tol or max_iter is reached

Residuals are always formed in fp64; this is what allows the refined solution
to attain an accuracy characteristic of fp64 despite an fp32 factorization.

Instrumentation
---------------
Memory is measured two ways, since neither is sufficient alone. tracemalloc
records the Python heap exactly, covering the NumPy and SciPy allocations, but
does not observe allocations made inside LAPACK. The RSS poller samples the
resident set size from a background thread every 5 ms and does observe them, at
the cost of missing peaks shorter than its sampling interval. A near-zero net
RSS delta between runs indicates that the operating system is reusing pages
already mapped, not that no memory was allocated.

Output
------
A table of relative error, wall time and peak memory per variant, on stdout.
"""

import gc
import threading
import time
import tracemalloc

import numpy as np
import psutil
import scipy.linalg as la

# ── paths ────────────────────────────────────────────────────────────────────
MATRIX_PATH = "/scratch/yimili/matrices/carbon-nanotube/M_E_50.npz"
RHS_PATH    = "/scratch/yimili/matrices/carbon-nanotube/rhs_E_50.npy"


# ── memory helpers ───────────────────────────────────────────────────────────

def _rss_mb():
    return psutil.Process().memory_info().rss / 1024**2


class _PeakRSSTracker:
    """Polls process RSS every `interval` s in a background thread."""
    def __init__(self, interval=0.005):
        self.interval = interval
        self.peak_mb  = 0.0
        self._stop    = threading.Event()

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


# ── solvers ──────────────────────────────────────────────────────────────────

def _ir_loop(A64, B64, lu, piv, getrs, tol, max_iter):
    """
    Generic IR loop given pre-computed LU factors.

    Parameters
    ----------
    A64     : (n, n) fp64 — used only for fp64 residual computation
    B64     : (n, k) fp64 — right-hand sides
    lu, piv : LU factors from [s|d]getrf
    getrs   : callable — the matching [s|d]getrs triangular solver
    tol     : convergence threshold on max_j ||r_j||/||b_j||
    max_iter: maximum refinement steps

    Returns
    -------
    X       : (n, k) fp64 solution
    history : list of max relative residual norms, one per IR iteration
    """
    dtype_lu = lu.dtype          # fp32 or fp64

    # initial solve — cast RHS to factorisation precision, result back to fp64
    X = getrs(lu, piv, B64.astype(dtype_lu))[0].astype(np.float64)

    norm_B  = np.linalg.norm(B64, axis=0)   # (k,)
    history = []

    for _ in range(max_iter):
        R   = B64 - A64 @ X                              # fp64 residual
        rel = float(np.max(np.linalg.norm(R, axis=0) / norm_B))
        history.append(rel)
        if rel < tol:
            break
        dX = getrs(lu, piv, R.astype(dtype_lu))[0].astype(np.float64)
        X  = X + dX                                      # fp64 accumulation

    return X, history


def solve_fp32_ir(A64, B64, tol, max_iter):
    """fp32 LU (sgetrf) + fp64 residual + fp32 correction solves (sgetrs)."""
    lu, piv, info = la.lapack.sgetrf(A64.astype(np.float32))
    if info != 0:
        raise RuntimeError(f"sgetrf failed: info={info}")
    X, history = _ir_loop(A64, B64, lu, piv, la.lapack.sgetrs, tol, max_iter)
    return X, history, lu


def solve_fp64(A64, B64):
    """Pure fp64 LU + solve, no refinement (LAPACK dgesv baseline)."""
    lu, piv, info = la.lapack.dgetrf(A64)
    if info != 0:
        raise RuntimeError(f"dgetrf failed: info={info}")
    X = la.lapack.dgetrs(lu, piv, B64)[0]
    return X, [], lu


def solve_fp64_ir(A64, B64, tol, max_iter):
    """fp64 LU (dgetrf) + fp64 residual + fp64 correction solves (dgetrs).
    Shows IR overhead when no precision is reduced."""
    lu, piv, info = la.lapack.dgetrf(A64)
    if info != 0:
        raise RuntimeError(f"dgetrf failed: info={info}")
    X, history = _ir_loop(A64, B64, lu, piv, la.lapack.dgetrs, tol, max_iter)
    return X, history, lu


# ── benchmark wrapper ────────────────────────────────────────────────────────

def benchmark(fn, repeats=3):
    """Run fn() `repeats` times; return list of result dicts."""
    records = []
    for _ in range(repeats):
        gc.collect()
        baseline_rss = _rss_mb()

        tracemalloc.start()
        with _PeakRSSTracker() as tracker:
            t0           = time.perf_counter()
            X, hist, lu  = fn()
            wall         = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        records.append({
            "X":          X,
            "history":    hist,
            "lu_dtype":   lu.dtype,
            "wall_ms":    wall * 1e3,
            "peak_py_mb": peak_bytes / 1024**2,
            "net_rss_mb": tracker.peak_mb - baseline_rss,
        })
    return records


# ── main ─────────────────────────────────────────────────────────────────────

def main(tol=1e-12, max_iter=30, repeats=5):
    # load
    print(f"Loading  {MATRIX_PATH}")
    A = np.asarray(np.load(MATRIX_PATH), dtype=np.float64)
    print(f"Loading  {RHS_PATH}")
    B = np.asarray(np.load(RHS_PATH),    dtype=np.float64)
    if B.ndim == 1:
        B = B[:, None]

    n, k = B.shape
    assert A.shape == (n, n), f"Expected square matrix, got {A.shape}"

    print(f"\nSystem   : n={n},  {k} RHS column(s)")
    print(f"Tolerance: {tol:.0e},  max IR iters: {max_iter},  repeats: {repeats}\n")

    # reference solution for true-error comparison
    X_ref  = la.solve(A, B)
    norm_X = np.linalg.norm(X_ref)

    # theoretical storage: n×n dense matrix
    mb = lambda dtype: A.size * np.dtype(dtype).itemsize / 1024**2

    solvers = [
        ("fp32 LU + fp64 IR",  lambda: solve_fp32_ir(A, B, tol, max_iter)),
        ("fp64  (no IR)",       lambda: (*solve_fp64(A, B),)),   # baseline
        ("fp64 LU + fp64 IR",  lambda: solve_fp64_ir(A, B, tol, max_iter)),
    ]

    records = {}
    for name, fn in solvers:
        print(f"  Running '{name}' × {repeats} ...", flush=True)
        records[name] = benchmark(fn, repeats)
    print()

    # ── summary table ────────────────────────────────────────────────────────
    names = [nm for nm, _ in solvers]
    W     = 26   # column width

    def med(name, key):
        return float(np.median([r[key] for r in records[name]]))

    def true_err(name):
        X = records[name][0]["X"]
        return np.linalg.norm(X - X_ref) / norm_X

    def lu_mb(name):
        dtype = records[name][0]["lu_dtype"]
        return mb(dtype)

    header = f"{'Metric':<36}" + "".join(f"  {nm:<{W}}" for nm in names)
    sep    = "─" * len(header)
    print(header)
    print(sep)

    rows = [
        ("LU factor precision",
            lambda nm: str(records[nm][0]["lu_dtype"])),
        ("Theoretical LU storage (MiB)",
            lambda nm: f"{lu_mb(nm):.3f}"),
        ("IR iterations (run 1)",
            lambda nm: str(len(records[nm][0]["history"])) if records[nm][0]["history"] else "0 (no IR)"),
        ("Final ||r||/||b|| (run 1)",
            lambda nm: f"{records[nm][0]['history'][-1]:.2e}" if records[nm][0]["history"] else "n/a"),
        ("True error vs fp64 ref",
            lambda nm: f"{true_err(nm):.2e}"),
        ("Wall time ms  [median]",
            lambda nm: f"{med(nm, 'wall_ms'):.2f}"),
        ("Peak Python heap MiB [median]",
            lambda nm: f"{med(nm, 'peak_py_mb'):.3f}"),
        ("Peak RSS delta MiB  [median]",
            lambda nm: f"{med(nm, 'net_rss_mb'):.2f}"),
    ]

    for label, fn_val in rows:
        vals = [fn_val(nm) for nm in names]
        print(f"  {label:<34}" + "".join(f"  {v:<{W}}" for v in vals))

    print(sep)

    # ── per-solver convergence histories ─────────────────────────────────────
    print()
    for name, _ in solvers:
        hist = records[name][0]["history"]
        print(f"  Convergence — {name}  (max ||r_j||/||b_j|| over {k} RHS)")
        if not hist:
            print("    [direct solve, no IR iterations]")
        else:
            for i, rel in enumerate(hist):
                converged = (i == len(hist) - 1 and rel < tol)
                tag = "  ← converged" if converged else ""
                print(f"    iter {i:2d}: {rel:.4e}{tag}")
        print()

    # ── storage summary ──────────────────────────────────────────────────────
    print(f"  Dense n×n matrix storage ({n}×{n}):")
    print(f"    fp32 LU factors : {mb(np.float32):.3f} MiB")
    print(f"    fp64 LU factors : {mb(np.float64):.3f} MiB")
    print(f"    Saving with fp32: {mb(np.float64) - mb(np.float32):.3f} MiB  "
          f"({100*(1 - mb(np.float32)/mb(np.float64)):.0f}%)")
    print()
    print("  Note: fp16 LU is not available in CPU LAPACK.")
    print("  Use MAGMA or cuSOLVER (dsgesv_mp / cusolverDnDSgesv) for fp16 on GPU.")


if __name__ == "__main__":
    main(tol=1e-12, max_iter=30, repeats=5)
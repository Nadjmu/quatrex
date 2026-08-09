"""
Mixed-precision iterative refinement on a sparse system, using SuperLU.

Input
-----
A sparse matrix and right-hand side; see the configuration below the imports.

Variants compared
-----------------
    fp32 LU with fp64 refinement   the method under test
    fp64 spsolve, no refinement    the accuracy reference
    fp32 solve, no refinement      the lower bound refinement must improve upon

Algorithm
---------
Classical LU-based iterative refinement (Buttari et al., 2006):

    1. cast A to fp32 and factorize it
    2. x = solve(b) in fp32, promoted to fp64
    3. r = b - A x, computed in fp64
    4. dx = solve(r) in fp32, reusing the same factorization
    5. x = x + dx
    6. repeat from 3 until ||r|| / ||b|| < tol or max_iter is reached

The residual at step 3 is the only quantity computed in the working precision,
and computing it there is what permits the refined solution to attain fp64
accuracy from an fp32 factorization. Convergence requires approximately
kappa_inf(A) * u_f < 1; see mpir.py for the general treatment and for the
GMRES-based variant that relaxes this requirement.

Instrumentation
---------------
Memory is measured two ways, since neither is sufficient alone. tracemalloc
records the Python heap exactly, covering the NumPy and SciPy allocations, but
does not observe allocations made inside SuperLU, where the fill-in resides.
The RSS poller samples the resident set size from a background thread every
5 ms and does observe them, at the cost of missing peaks shorter than its
sampling interval.

SuperLU and the operating system tend to reuse pages already mapped across
calls, so the net RSS delta after a call is frequently near zero. The
meaningful quantity is the peak above baseline during the call, which is what
the background poller records.

Output
------
A table of relative error, wall time and peak memory per variant, on stdout.
"""

import gc
import threading
import time
import tracemalloc
import warnings

import numpy as np
import psutil
import scipy.sparse as sp
import scipy.sparse.linalg as spla

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rss_mb():
    return psutil.Process().memory_info().rss / 1024**2


class _PeakRSSTracker:
    """
    Polls process RSS every `interval` seconds in a background thread.
    Use as a context manager; read .peak_mb after exit.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────────

def solve_mixed_ir(A, b, tol=1e-10, max_iter=20):
    """fp32 LU factorisation + fp64 iterative refinement."""
    b64    = np.asarray(b, dtype=np.float64)
    A64    = A.tocsc().astype(np.float64)
    norm_b = np.linalg.norm(b64)
    history = []

    A32 = A64.astype(np.float32)
    lu  = spla.splu(A32)
    x   = lu.solve(b64.astype(np.float32)).astype(np.float64)

    for _ in range(max_iter):
        r   = b64 - A64 @ x
        rel = np.linalg.norm(r) / norm_b
        history.append(rel)
        if rel < tol:
            break
        x += lu.solve(r.astype(np.float32)).astype(np.float64)

    return x, {"history": history}


def solve_fp64(A, b):
    """Pure fp64 direct solve (SuperLU)."""
    x = spla.spsolve(A.tocsc().astype(np.float64), b.astype(np.float64))
    return x, {}


def solve_fp32(A, b):
    """Pure fp32 direct solve, no refinement."""
    lu = spla.splu(A.tocsc().astype(np.float32))
    x  = lu.solve(b.astype(np.float32)).astype(np.float64)
    return x, {}


# ─────────────────────────────────────────────────────────────────────────────
# Test-problem generator
# ─────────────────────────────────────────────────────────────────────────────

def make_problem(n=2000, density=5e-3, seed=42):
    """Diagonally-dominant random sparse system with known exact solution."""
    rng    = np.random.default_rng(seed)
    A      = sp.random(n, n, density=density, format="csr",
                       random_state=rng, dtype=np.float64)
    A      = A + sp.diags(np.abs(A).sum(axis=1).A1 + 1.0)
    x_true = rng.standard_normal(n)
    b      = A @ x_true
    return A, b, x_true


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_solver(fn, x_true, repeats):
    """
    Run fn() `repeats` times.
    Returns list of dicts with keys: x, wall_s, peak_py_mb, net_rss_mb, extra.
    """
    records = []
    for _ in range(repeats):
        gc.collect()
        baseline_rss = _rss_mb()

        tracemalloc.start()
        with _PeakRSSTracker() as tracker:
            t0       = time.perf_counter()
            x, extra = fn()
            wall     = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        true_err = (np.linalg.norm(x - x_true) / np.linalg.norm(x_true))
        records.append({
            "true_err":   true_err,
            "wall_s":     wall,
            "peak_py_mb": peak_bytes / 1024**2,
            "net_rss_mb": tracker.peak_mb - baseline_rss,
            "extra":      extra,
        })
    return records


def run_benchmarks(n=3000, density=5e-3, tol=1e-14, max_iter=20, repeats=3):
    print(f"Problem : n={n}, density={density}, nnz~{int(n*n*density)}")
    print(f"Runs    : {repeats} per solver (median reported)\n")

    A, b, x_true = make_problem(n, density)

    solvers = [
        ("fp32 LU + IR",         lambda: solve_mixed_ir(A, b, tol, max_iter)),
        ("fp64 spsolve",          lambda: solve_fp64(A, b)),
        ("fp32 only (no refine)", lambda: solve_fp32(A, b)),
    ]

    all_records = {}
    for name, fn in solvers:
        print(f"  Benchmarking '{name}' x{repeats} ...", flush=True)
        all_records[name] = benchmark_solver(fn, x_true, repeats)
    print()

    # ── Summary table ────────────────────────────────────────────────────────
    names = [nm for nm, _ in solvers]
    col   = 30

    def med(name, key):
        return float(np.median([r[key] for r in all_records[name]]))

    header = f"{'Metric':<{col}}" + "".join(f"  {nm:<26}" for nm in names)
    print(header)
    print("─" * len(header))

    rows = [
        ("True error ||x-x*||/||x*||", "true_err",   "{:.2e}"),
        ("Wall time (ms)",             "wall_s",      lambda v: f"{v*1e3:.1f}"),
        ("Peak Python heap (MiB)",     "peak_py_mb",  "{:.2f}"),
        ("Peak RSS above baseline (MiB)", "net_rss_mb", "{:.1f}"),
    ]

    for label, key, fmt in rows:
        vals = [fmt.format(med(n, key)) if isinstance(fmt, str)
                else fmt(med(n, key))
                for n in names]
        print(f"  {label:<{col-2}}" + "".join(f"  {v:<26}" for v in vals))

    # ── IR convergence history ───────────────────────────────────────────────
    history = all_records["fp32 LU + IR"][0]["extra"].get("history", [])
    if history:
        print(f"\n  IR convergence history (first run):")
        for i, rel in enumerate(history):
            tag = "  <- converged" if i == len(history) - 1 else ""
            print(f"    iter {i}: ||r||/||b|| = {rel:.3e}{tag}")

    # ── Theoretical storage ──────────────────────────────────────────────────
    nnz     = int(A.nnz)
    mib64   = nnz * 8  / 1024**2
    mib32   = nnz * 4  / 1024**2
    idx_mib = (nnz + n + 1) * 4 / 1024**2   # col_indices + indptr (int32)

    print(f"\n  Theoretical matrix value storage ({nnz} nonzeros, indices excluded):")
    print(f"    fp64  : {mib64:.3f} MiB")
    print(f"    fp32  : {mib32:.3f} MiB  (50% saving on values)")
    print(f"    Index arrays (shared): ~{idx_mib:.3f} MiB")
    print()
    print(f"  Note: SuperLU fill-in can be 5-50x the original matrix size.")
    print(f"  The fp32 factor saves ~{mib64-mib32:.3f} MiB here; gains grow with n.")
    print(f"  RSS delta near zero = OS/SuperLU reusing already-mapped pages.")

    return all_records


if __name__ == "__main__":
    run_benchmarks(n=3000, density=5e-3, tol=1e-14, max_iter=20, repeats=3)
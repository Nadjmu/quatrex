#!/usr/bin/env python3
"""
Runtime of mixed-precision iterative refinement, across solvers.

The companion of mpir.py. mpir.py asks whether refinement converges and to
what accuracy; this script asks what it costs. Two variants are timed for
every (index, solver):

    c64_ir    complex64 factorization + LU-IR      (the method)
    c128      complex128 direct solve, no refinement   (what it replaces)

and each is split into three stages that are drawn as one stacked bar:

    symbolic_s        the fill-reducing ordering and symbolic factorization,
                      or for Block Thomas the detection and extraction of the
                      blocks. No floating-point arithmetic, so it costs the
                      same at both precisions and bounds the achievable
                      speedup. SuperLU fuses it into the numerical phase and
                      reports nothing; its bar has no symbolic segment and
                      phases_split is 0.
    factorization_s   the rest of the factorization: the numerical phase, plus
                      the cast of A into the factorization precision and, for
                      cuDSS, the host-to-device transfer. This is the stage a
                      lower u_f makes cheaper.
    solve_s           every triangular solve plus every complex128 residual
                      b - Ax. One of each for c128; for c64_ir, one solve per
                      outer refinement step and one residual per iterate.

The three sum to total_s, the bar height, which is the end-to-end algorithmic
cost of the variant.

What is not timed
-----------------
Nothing that exists only to diagnose convergence is inside a timed region.
Every stage above is bracketed by its own perf_counter pair around the solver
call, the matrix-vector product or the builder, so the forward errors the
stopping rule watches, the backward errors, the convergence factors mu_hat and
rho and the phi_* terms of Corollary 3.3 all fall between the brackets and are
charged to nothing. The convergence diagnostics of mpir.refinement_metrics are
not computed here at all -- this script never calls it.

The one thing this does leave out is the O(n) vector update x + d, which is not
bracketed by any timer. It is the same work in both variants and orders of
magnitude below a solve.

The refinement loop is nonetheless run with a reference solution, through
--reference-solver, so that it stops on the same rule and after the same
number of outer steps as it would in mpir.py. Timing a loop that ran to
--max-iter because it had nothing to stop on would measure the safety net.

Precision
---------
u_f is complex64. complex32 is rejected: those factorizations are hand-written
NumPy kernels that embed each complex block into a real one of twice the
dimension, so timing them against LAPACK, MUMPS and cuDSS would compare
implementations rather than precisions. Their accuracy is the subject of
mpir.py, where the comparison is meaningful.

Fairness
--------
Every variant of one index is measured in one process, back to back, against
the same A and b, after an untimed warm-up of that solver at both precisions --
see _warm_up, which exists so that the one-time cost of the first LAPACK, MUMPS
or cuDSS call in the process is not charged to the first bar of the figure. The
reference solution is computed once per index and shared, so its cost is
charged to no solver. --repeats runs of each variant are made and the median of each stage
reported; total_s is the sum of those three medians, so the bar equals the sum
of its parts. total_s_min and total_s_max are the smallest and largest
per-repeat totals, the spread the median hides.

Solvers are compared as installed on the machine the script runs on. MUMPS and
cuDSS are libraries with their own threading and, for cuDSS, their own device;
Block Thomas is NumPy calling LAPACK on dense blocks. Running all four
together requires a node with a GPU, and the comparison is then between a GPU
solver and three CPU solvers.

Memory
------
Recorded but not yet drawn: factor_mb, the stored factorization from each
solver's factor_nbytes, matrix_mb, A at the precision the variant holds it
(complex128 for both, since LU-IR forms its residual there), and their sum
working_mb. factor_nbytes returns 0 where a backend exposes no size -- MUMPS
where INFOG(3) is unreachable, cuDSS where the factorization info carries no
lu_nnz -- and factor_mb_reported is 0 there. A figure must drop such a row
rather than draw it at zero.

Output
------
One numbered experiment appended per invocation to

    <outdir>/<material>/<material>_perf.h5
    └── experiments/0001/     attrs: the run configuration
        └── runs              one row per (index, solver, variant)

beside the convergence file mpir.py writes, so one material directory holds
both studies and every figure drawn from either.

Usage
-----
    python mpperf.py .../graphene.h5 --idx 84 254 601
    python mpperf.py .../graphene.h5 --start 0 --end 800 --stride 200
    python mpperf.py .../graphene.h5 --idx 84 --solvers mumps block-thomas
    python mpperf.py .../graphene.h5 --list-experiments

Then

    python ../plotting/mixed_prec_ir/plot_mpperf.py \\
        .../mixed-precision-IR/graphene/graphene_perf.h5
"""

import datetime
import gc
import math
import os
import platform
import socket
import sys
from pathlib import Path

import numpy as np
import scipy

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))
import cli
from factor_io import save_table, load_table, material_metadata
from solver_classes import block_sizes_from_matrix

from mpir import (
    C32, HIGH_DTYPE, EXPERIMENTS_GROUP, EXTENDED_REFERENCE,
    DEFAULT_INV_DTYPE, DEFAULT_MAX_ITER, FACTORIZATION_FAILURES,
    SOLVER_BUILDERS,
    build_reference, dtype_label, energy_of_idx, experiment_attrs,
    experiment_names, idx_of_energy, load_condition_numbers,
    load_energy_metadata, load_system, new_experiment, resolve_experiment,
    solve_direct, solve_mixed_ir, solver_dtypes, unit_roundoff,
    _inf_norm, _matrix_nbytes,
)

MIB = 1024.0 ** 2
PERF_SUFFIX = "_perf"

# The four families the summary figure compares. Every one of them has both a
# complex128 and a complex64 factorization, which is what the two variants
# need. UMFPACK has no single-precision build and the two -fp16 families are
# excluded for the reason given under Precision in the module docstring.
DEFAULT_PERF_SOLVERS = ("block-thomas", "mumps", "cudss", "superlu")

# The stacked segments, in the order they are drawn from the axis upwards.
PHASES = ("symbolic_s", "factorization_s", "solve_s")

# The two bars of every pair. c64_ir is drawn on the left of c128 because it is
# the method under test and c128 the baseline it is read against.
VARIANTS = ("c64_ir", "c128")

VARIANT_LABEL = {
    "c64_ir": "{low} + LU-IR",
    "c128":   "complex128 direct",
}

RUN_COLUMNS = [
    # identity
    "idx", "energy", "n", "nnz", "n_rhs", "n_blocks",
    "solver", "variant", "factor_dtype",
    # conditioning: kappa_inf is the x axis of the summary figure
    "u_f", "u", "kappa_2", "kappa_inf", "lu_ir_bound",
    # the three drawn stages and their sum
    "symbolic_s", "factorization_s", "solve_s", "total_s",
    # provenance of the split: phases_split is 0 where the backend fuses the
    # symbolic and numeric phases, in which case symbolic_s is nan and the
    # whole factorization sits in factorization_s. numeric_reported_s is what
    # the backend called the numerical phase and factor_s the whole builder
    # call, so factor_s - symbolic_s - numeric_reported_s is the setup that
    # neither phase claimed and that factorization_s absorbs.
    "phases_split", "numeric_reported_s", "factor_s",
    "total_s_min", "total_s_max",
    # what the timing is worth: a c64_ir bar that did not converge did not
    # deliver the answer c128 did, and its height is not a speedup
    "outer_iters", "n_solves", "converged", "ferr_ref", "ferr_tol",
    "stop_reason",
    # memory, recorded but not yet drawn
    "factor_mb", "factor_mb_reported", "matrix_mb", "working_mb",
    "repeats", "reduce", "reference_solver",
]


def perf_path(outdir, material):
    """
    <outdir>/<material>/<material>_perf.h5, beside the convergence file
    mpir.analysis_path returns for the same material.
    """
    return Path(outdir) / material / f"{material}{PERF_SUFFIX}.h5"


def load_experiment(path, experiment=None):
    """
    (name, attrs, runs) for one experiment, the last by default. `runs` is the
    column dict factor_io.load_table returns; `experiment` accepts either the
    padded name or the bare number.
    """
    name = resolve_experiment(path, experiment)
    attrs = experiment_attrs(path, name)
    runs, _ = load_table(path, f"{EXPERIMENTS_GROUP}/{name}/runs")
    return name, attrs, runs


# ─────────────────────────────────────────────────────────────────────────────
# Environment
#
# A wall-clock comparison is a statement about a machine, and every one of the
# quantities below changes the answer. They are recorded on the experiment so
# that a figure drawn a year later still says what it was measured on, and so
# that two experiments can be checked for comparability rather than assumed
# comparable.
#
# The one that dominates: OpenBLAS defaults to one thread per core. On a 72-core
# node factoring dense blocks 128 to 352 wide, that was measured costing 68x --
# a complex128 Block Thomas factorization of one si-bulk index took 254 ms with
# the thread count capped and 17 s without. It is not a second-order effect and
# a run that does not record it cannot be reproduced.
# ─────────────────────────────────────────────────────────────────────────────

# Read rather than set. mpperf never sets a thread count itself: OpenBLAS fixes
# its pool size at the first call, so a value set from inside the process is
# unreliable, and silently overriding what the user asked for would make the
# recorded environment a lie. Cap it in the shell, as the README says.
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _blas_info():
    """(name, version, build configuration) of the BLAS NumPy is linked against."""
    try:
        cfg = np.show_config(mode="dicts") or {}
        blas = (cfg.get("Build Dependencies") or {}).get("blas") or {}
        return (str(blas.get("name", "")), str(blas.get("version", "")),
                str(blas.get("openblas configuration", "")))
    except Exception:
        return ("", "", "")


def _threadpools():
    """
    What each loaded native threadpool says its size is, as 'api=n' strings, or
    [] where threadpoolctl is not installed.

    This is the only source that reports the size a library actually chose, as
    opposed to what the environment asked for. The two differ whenever no
    variable is set, which is the case this exists to catch.

    Sampled twice: once in environment(), and again after the sweep. The first
    sample sees only what is loaded at start-up, which is NumPy and SciPy --
    python-mumps and nvmath dlopen their libraries, and any OpenMP runtime
    they bring with them, when their first solver is built. A pool that only
    appears in the second sample is one belonging to a solver, and it is the
    one a start-up-only check would miss.
    """
    try:
        import threadpoolctl
    except ImportError:
        return []
    try:
        return [f"{d.get('internal_api', '?')}={d.get('num_threads', '?')}"
                for d in threadpoolctl.threadpool_info()]
    except Exception:
        return []


def environment():
    """
    Everything a timing comparison depends on that cannot be recovered from the
    result table afterwards.

    `cpu_affinity` rather than `cpu_count`: under taskset the two differ, and it
    is the affinity mask that bounds how many cores the run could actually use.
    """
    name, version, config = _blas_info()
    env = dict(
        host=socket.gethostname(),
        platform=platform.platform(),
        processor=platform.machine(),
        cpu_count=int(os.cpu_count() or 0),
        cpu_affinity=len(os.sched_getaffinity(0))
                     if hasattr(os, "sched_getaffinity") else 0,
        blas_name=name,
        blas_version=version,
        blas_config=config,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
    )
    for var in THREAD_VARS:
        env[f"env_{var}"] = os.environ.get(var, "")
    pools = _threadpools()
    env["threadpools"] = ", ".join(pools) if pools else "(threadpoolctl absent)"
    return env


def loadavg(suffix):
    """The 1/5/15-minute load averages, tagged with when they were taken."""
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        return {}
    return {f"loadavg_1_{suffix}": float(one),
            f"loadavg_5_{suffix}": float(five),
            f"loadavg_15_{suffix}": float(fifteen)}


def check_environment(env, load):
    """
    Print the environment, and warn where it is one a timing comparison should
    not be made in. Returns the number of warnings, which main() reports again
    at the end -- the header scrolls away in a long run and a silently
    unusable measurement is the thing most likely to be misread.

    Nothing is refused. The user may have a reason to measure on a busy node;
    the requirement is that the run says so.
    """
    print(f"Machine : {env['host']}  {env['processor']}  "
          f"{env['cpu_affinity']} of {env['cpu_count']} cores available")
    print(f"BLAS    : {env['blas_name']} {env['blas_version']}")
    threads = {v: env[f"env_{v}"] for v in THREAD_VARS if env[f"env_{v}"]}
    print(f"Threads : "
          + (", ".join(f"{k}={v}" for k, v in threads.items()) if threads
             else "no thread variable set")
          + f"   [{env['threadpools']}]")
    if load:
        print(f"Load    : {load.get('loadavg_1_start', float('nan')):.1f} "
              f"(1 min), {load.get('loadavg_15_start', float('nan')):.1f} "
              f"(15 min)")

    warnings_ = []
    # OpenBLAS with no cap takes one thread per core. Harmless on 8 cores,
    # catastrophic on 72 with blocks this size; 16 is where it starts to hurt.
    if not threads and env["cpu_affinity"] > 16:
        warnings_.append(
            f"no thread limit is set and {env['cpu_affinity']} cores are "
            f"available, so OpenBLAS will use one thread per core. On blocks "
            f"of a few hundred rows this has been measured 68x slower than a "
            f"capped pool. Set OPENBLAS_NUM_THREADS (see the README).")
    # Someone else's job makes every CPU number here a measurement of the node
    # rather than of the solver. A quarter of the cores busy is generous.
    busy = load.get("loadavg_1_start", 0.0)
    if env["cpu_affinity"] and busy > 0.25 * env["cpu_affinity"]:
        warnings_.append(
            f"load average is {busy:.1f} on {env['cpu_affinity']} cores: the "
            f"node is running other work, and CPU timings will measure that "
            f"as much as the solver. The GPU solver is unaffected.")
    for message in warnings_:
        print(f"\n  [WARNING] {message}")
    if warnings_:
        print()
    return len(warnings_)


# How the repeats of one stage are reduced to the number drawn.
#
# median is the default because it is what a reader expects and what the study
# was specified with. min is the defensible choice on a shared machine:
# contention, page reclaim and scheduling only ever ADD time, never remove it,
# so the fastest repeat is the observation least contaminated by the node and
# the closest to the cost of the solver itself. It is the standard estimator
# for exactly this reason (timeit.Timer.repeat documents the same argument).
#
# It is not a way to make a bad run look good: the stability check below reads
# total_s_min and total_s_max whichever reducer is used, so a contaminated run
# is still reported as one. min changes which number is drawn, not whether the
# run is admitted.
REDUCERS = {"median": np.median, "min": np.min}


# The spread above which a row is not a measurement. Contention only ever adds
# time, so a solver whose slowest repeat is more than this multiple of its
# fastest was interrupted during at least one of them; cuDSS on an idle GPU
# holds 1.014 across five indices, and a capped, quiet CPU run should be well
# inside 1.5.
STABILITY_LIMIT = 2.0


def check_stability(rows, limit=STABILITY_LIMIT):
    """
    Report every row whose repeats disagree by more than `limit`, and return
    how many there were.

    This is the check that decides whether a run is usable. The median hides
    the spread by construction -- that is what a median is for -- so without it
    a bar drawn from five repeats that ranged over two orders of magnitude
    looks exactly like a bar drawn from five that agreed.
    """
    bad = []
    for row in rows:
        lo, hi = row["total_s_min"], row["total_s_max"]
        if lo > 0 and np.isfinite(lo) and np.isfinite(hi) and hi / lo > limit:
            bad.append((hi / lo, row))
    if not bad:
        return 0
    print(f"\n{'!' * 92}")
    print(f"UNSTABLE: {len(bad)} of {len(rows)} measurements have repeats "
          f"disagreeing by more than {limit:g}x.")
    print("These are not measurements of the solver. Do not draw them.\n")
    print(f"{'idx':>6} {'solver':<16} {'variant':<10} "
          f"{'min':>10} {'median':>10} {'max':>10} {'ratio':>7}")
    for ratio, row in sorted(bad, key=lambda p: -p[0]):
        print(f"{int(row['idx']):>6} {row['solver']:<16} {row['variant']:<10} "
              f"{row['total_s_min'] * 1e3:>9.1f}m {row['total_s'] * 1e3:>9.1f}m "
              f"{row['total_s_max'] * 1e3:>9.1f}m {ratio:>7.1f}x")
    print(f"\nUsual cause: another job on the node, or an uncapped OpenBLAS "
          f"thread pool.\nSee the Reproducibility section of "
          f"mixed_prec_ir/README.md.")
    print("!" * 92)
    return len(bad)


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────

# Histories the refinement loop retains for mpir.py's convergence figures. They
# are O(n * outer_iters * n_rhs) per repeat and of no use here, so they are
# dropped after every repeat rather than accumulated across a sweep.
_HISTORY_KEYS = ("x_history", "d_history", "r_history", "history",
                 "true_err_history")


def _warm_up(solver_name, A, b, low_dtype, inv_dtype):
    """
    One untimed factorization and solve at each precision, discarded.

    The first call into a numerical library in a process pays costs no later
    call pays: OpenBLAS builds its thread pool on the first LAPACK call, SciPy
    resolves its lazy imports, MUMPS and cuDSS initialise their own state --
    cuDSS additionally creates a CUDA context and JITs its kernels, about 1.2 s
    and essentially independent of problem size -- and the allocator faults in
    pages it afterwards reuses. Left to fall inside a measurement, all of it
    lands on whichever variant runs first, which is the first solver of the
    first index. Its bar is then not comparable with any other bar in the
    figure, and the effect survives the median: it is one large outlier only
    when it is confined to one repeat, and thread-pool and allocator warm-up
    are not.

    Both precisions are warmed, not just u_f: the two variants factor at
    different ones, and a library's first call at each can differ -- cuDSS JITs
    per dtype.

    This is done per (index, solver) rather than once per process so that every
    measurement is equally warm, including the first solver of a later index
    whose matrix has a different size or pattern. It costs two solves against
    2 * --repeats measured ones.

    Failures are ignored. The warm-up affects measurement only; a solver that
    cannot be built is reported by measure_variant through its own skip path.
    """
    for dtype in (low_dtype, HIGH_DTYPE):
        try:
            solve_direct(solver_name, A, b, None, dtype, inv_dtype)
        except Exception:
            pass
    gc.collect()


def _driver(variant, solver_name, A, b, low_dtype, inv_dtype, opts):
    """
    The zero-argument callable one repeat times. Both are mpir drivers, so the
    cost study measures the same code the convergence study does.
    """
    if variant == "c128":
        return lambda: solve_direct(solver_name, A, b, None, HIGH_DTYPE,
                                    inv_dtype)
    return lambda: solve_mixed_ir(
        solver_name, A, b, None, low_dtype, opts["max_iter"],
        x_true=opts["x_true"], normA=None, inv_dtype=inv_dtype,
        ferr_tol=opts["ferr_tol"])


def _phases(extra):
    """
    (symbolic_s, factorization_s, solve_s, numeric_reported_s, factor_s) of
    one repeat.

    factorization_s is factor_s minus the symbolic phase, not the numerical
    phase the backend reports: the builder also casts A into the factorization
    precision and, for cuDSS, moves it to the device, and that work is
    precision-dependent and belongs on the same side of the split as the
    arithmetic. Attributing it this way is also what makes the three segments
    sum to the true cost of the variant rather than to slightly less than it.

    Where the backend fuses the two phases the symbolic segment is nan and the
    whole factorization is factorization_s; the caller records phases_split=0
    so a figure can say so rather than draw a missing stage as zero.
    """
    breakdown = extra.get("factor_breakdown")
    factor_s = float(extra["factor_s"])
    symbolic_s = float(breakdown[0]) if breakdown else float("nan")
    numeric_reported_s = float(breakdown[1]) if breakdown else float("nan")
    factorization_s = factor_s - (symbolic_s if breakdown else 0.0)
    solve_s = float(extra["solve_s"]) + float(extra.get("residual_s", 0.0))
    return (symbolic_s, factorization_s, solve_s, numeric_reported_s, factor_s)


def measure_variant(variant, solver_name, A, b, low_dtype, inv_dtype, opts,
                    repeats, reduce="median"):
    """
    One (solver, variant) measurement, as a partial row, or None where the
    variant could not run -- an absent package, no CUDA device, a partition the
    solver rejects, a low-precision factorization that overflowed. The reason
    is printed and the sweep continues, so one missing solver does not cost the
    whole index.
    """
    low_name = dtype_label(low_dtype)
    label = VARIANT_LABEL[variant].format(low=low_name)
    print(f"    {label:<28}", end="", flush=True)
    fn = _driver(variant, solver_name, A, b, low_dtype, inv_dtype, opts)

    stages, x, extra = [], None, None
    try:
        for _ in range(repeats):
            gc.collect()
            x, extra = fn()
            stages.append(_phases(extra))
            for key in _HISTORY_KEYS:
                extra.pop(key, None)
    except (ImportError, TypeError, RuntimeError, ValueError,
            *FACTORIZATION_FAILURES) as e:
        print(f"skipped ({type(e).__name__}: {e})")
        return None

    # Each stage reduced over the repeats, so total_s is the sum of exactly the
    # three numbers drawn. It is therefore the reduction of each part rather
    # than the reduction of the totals; total_s_min and total_s_max carry the
    # spread of the per-repeat totals whichever reducer is in use.
    stages = np.asarray(stages, dtype=float)
    symbolic_s, factorization_s, solve_s, numeric_reported_s, factor_s = \
        REDUCERS[reduce](stages, axis=0)
    per_repeat_total = np.nansum(stages[:, :3], axis=1)
    total_s = float(np.nansum([symbolic_s, factorization_s, solve_s]))

    monitor = extra.get("monitor")
    summary = monitor.summary() if monitor is not None else {}

    # The forward error against the shared reference, from the last repeat.
    # An O(n) vector norm computed here, outside every timed region.
    ferr_ref = float("nan")
    if opts["x_true"] is not None:
        norm_true = _inf_norm(opts["x_true"])
        if norm_true > 0:
            ferr_ref = _inf_norm(x - opts["x_true"]) / norm_true

    # `converged` answers one question for both variants: did this variant
    # deliver the target accuracy, so that its bar height means something. For
    # c64_ir it is the verdict of mpir's stopping monitor. c128 runs no loop
    # and has nothing to converge; it is the accuracy the study is measured
    # against, so it is recorded as converged and ferr_ref sits beside it for a
    # figure to check rather than assume.
    converged = int(bool(summary.get("converged", False))) if monitor else 1

    mem_bytes = extra.get("mem_bytes", 0) or 0
    factor_mb = max(float(mem_bytes), 0.0) / MIB
    # complex128 for both variants: LU-IR forms its residual at the working
    # precision and so holds A there however low u_f is. This is why the
    # working set does not halve when the factorization does.
    matrix_mb = _matrix_nbytes(A, HIGH_DTYPE) / MIB

    row = dict(
        variant=variant,
        symbolic_s=float(symbolic_s),
        factorization_s=float(factorization_s),
        solve_s=float(solve_s),
        total_s=total_s,
        phases_split=int(math.isfinite(symbolic_s)),
        numeric_reported_s=float(numeric_reported_s),
        factor_s=float(factor_s),
        total_s_min=float(per_repeat_total.min()),
        total_s_max=float(per_repeat_total.max()),
        outer_iters=int(summary.get("outer_iters", 0)),
        n_solves=int(extra.get("n_solves", 0)),
        converged=converged,
        ferr_ref=ferr_ref,
        ferr_tol=float(summary.get("ferr_tol", opts["ferr_tol"] or float("nan"))),
        stop_reason=summary.get("stop_reason", "no refinement"),
        factor_mb=factor_mb,
        factor_mb_reported=int(bool(mem_bytes)),
        matrix_mb=matrix_mb,
        working_mb=factor_mb + matrix_mb,
        repeats=int(repeats),
        reduce=str(reduce),
    )

    tag = "" if monitor is None else \
        f"   {row['outer_iters']} outer, {row['n_solves']} solves" + \
        ("" if converged else "   NOT CONVERGED")
    split = "" if row["phases_split"] else "   (no symbolic split)"
    # The min-max spread of the per-repeat totals is printed beside the
    # median so a repeat still paying a one-time cost is visible while the
    # sweep runs, rather than only after the figure looks wrong. A spread of
    # more than about 2x on a warmed solver means _warm_up missed something.
    print(f"{total_s * 1e3:9.1f} ms  "
          f"[{per_repeat_total.min() * 1e3:.1f}-{per_repeat_total.max() * 1e3:.1f}]   "
          f"sym {symbolic_s * 1e3:7.1f}  fact {factorization_s * 1e3:8.1f}  "
          f"solve {solve_s * 1e3:8.1f}   ferr {ferr_ref:.2e}{tag}{split}")

    del extra, x
    gc.collect()
    return row


def measure_index(h5path, idx, solvers, low_dtype, inv_dtype, max_iter,
                  repeats, reference_solver, ferr_tol, reduce="median"):
    """
    Every (solver, variant) at one energy index, measured in one process
    against byte-identical inputs.
    """
    A, b = load_system(h5path, idx)
    n = A.shape[0]
    n_rhs = b.shape[1] if np.asarray(b).ndim == 2 else 1
    u_f = unit_roundoff(low_dtype)
    u = unit_roundoff(HIGH_DTYPE)

    indices, energies, _, _ = load_energy_metadata(h5path)
    energy = energy_of_idx(indices, energies, idx)
    kappa = load_condition_numbers(h5path, idx)

    print(f"E_{idx}  " +
          (f"E={energy:.4f} eV  " if energy is not None else "") +
          f"n={n}  nnz={A.nnz}  n_rhs={n_rhs}" +
          (f"  kappa_inf={kappa['inf']:.2e}  kappa_inf*u_f="
           f"{kappa['inf'] * u_f:.2e}" if kappa["inf"] is not None else
           "  kappa: not available (run condition-est/condition_est.py)"))

    # The convergence level, resolved exactly as mpir.run_benchmarks resolves
    # it so that the same run stops at the same place in both studies:
    # Corollary 3.3's own limiting accuracy cond(A,x) u where the condition-est
    # file has it, sqrt(n) u otherwise, an explicit --ferr-tol over both.
    if ferr_tol is not None:
        ferr_level = float(ferr_tol)
    elif kappa["skeel_x"] is not None:
        ferr_level = kappa["skeel_x"] * u
    else:
        ferr_level = math.sqrt(n) * u

    # x_true, shared by every solver at this index, so the reference is charged
    # to none of them. It is what the refinement loop's stopping rule watches;
    # without it the loop would run to max_iter and the timing would measure
    # the safety net.
    x_true = None
    if reference_solver is not None:
        ref = build_reference(reference_solver, A, None, b)
        x_true = ref.solve(np.asarray(b, dtype=HIGH_DTYPE)).astype(HIGH_DTYPE)
        if hasattr(ref, "free"):
            ref.free()
        del ref
        gc.collect()
    opts = dict(x_true=x_true, max_iter=max_iter, ferr_tol=ferr_level)

    common = dict(
        idx=int(idx), n=int(n), nnz=int(A.nnz), n_rhs=int(n_rhs),
        energy=float(energy) if energy is not None else float("nan"),
        factor_dtype=dtype_label(low_dtype),
        u_f=float(u_f), u=float(u),
        kappa_2=kappa[2] if kappa[2] is not None else float("nan"),
        kappa_inf=kappa["inf"] if kappa["inf"] is not None else float("nan"),
        lu_ir_bound=(kappa["inf"] * u_f) if kappa["inf"] is not None
                    else float("nan"),
        reference_solver=reference_solver or "",
    )

    rows = []
    for solver_name in solvers:
        print(f"  {cli.label(solver_name)}")
        # Every one-time library start-up cost is consumed before either
        # variant of this solver is timed, so none of it is charged to
        # whichever runs first. See _warm_up.
        _warm_up(solver_name, A, b, low_dtype, inv_dtype)
        n_blocks = len(block_sizes_from_matrix(A)) \
            if solver_name.startswith("block-thomas") else -1
        for variant in VARIANTS:
            row = measure_variant(variant, solver_name, A, b, low_dtype,
                                  inv_dtype, opts, repeats, reduce)
            if row is not None:
                rows.append(dict(common, solver=solver_name,
                                 n_blocks=int(n_blocks), **row))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(rows):
    """
    One line per (index, solver): the two totals and the ratio between them.

    The ratio is the speedup the figure shows as two bar heights, printed so a
    run can be read without drawing it. It is left blank where refinement did
    not converge, since a method that did not produce the answer has no
    speedup over one that did.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row["idx"]), row["solver"]), {})[
            row["variant"]] = row
    if not grouped:
        return

    print("\n" + "=" * 92)
    print(f"{'idx':>6} {'kappa_inf':>11} {'solver':<18} "
          f"{'c64+LU-IR':>11} {'c128':>11} {'speedup':>9} {'outer':>6}  note")
    print("-" * 92)

    def _kappa(variants):
        row = variants.get("c64_ir") or variants.get("c128")
        kappa = row["kappa_inf"]
        # nan sorts unpredictably; indices with no condition number, which the
        # figure drops, go last here rather than into the middle of the table.
        return kappa if np.isfinite(kappa) else float("inf")

    for (idx, solver), variants in sorted(
            grouped.items(), key=lambda kv: (_kappa(kv[1]), kv[0][1])):
        ir, ref = variants.get("c64_ir"), variants.get("c128")
        kappa = (ir or ref)["kappa_inf"]
        t_ir = ir["total_s"] if ir else float("nan")
        t_ref = ref["total_s"] if ref else float("nan")
        speedup = (t_ref / t_ir) if (ir and ref and t_ir > 0
                                     and ir["converged"]) else float("nan")
        note = "" if (ir and ir["converged"]) else "LU-IR did not converge"
        print(f"{idx:>6} {kappa:>11.2e} {solver:<18} "
              f"{t_ir * 1e3:>9.1f}ms {t_ref * 1e3:>9.1f}ms "
              f"{speedup:>9.2f} {ir['outer_iters'] if ir else 0:>6}  {note}")
    print("=" * 92)


def list_experiments(path):
    """Print what a file holds, one line per experiment."""
    names = experiment_names(path)
    if not names:
        print(f"{path} holds no experiments")
        return
    print(f"{path}")
    for name in names:
        _, attrs, runs = load_experiment(path, name)
        print(f"  {name}  {attrs.get('timestamp', '?'):<26} "
              f"{attrs.get('factor_dtype', '?')}  "
              f"{len(attrs.get('indices', []))} idx  "
              f"{len(runs.get('idx', []))} runs  "
              f"repeats={attrs.get('repeats', '?')}  "
              f"{', '.join(attrs.get('solvers', []))}")


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap)
    cli.add_index_selection(ap, default_all=True)
    ap.add_argument("--energy", type=float, nargs="+", default=None,
                    metavar="EV",
                    help="one or more energies in eV; each is resolved to the "
                         "index with the nearest recorded energy. Mutually "
                         "exclusive with --idx/--start/--end")
    cli.add_solver_selection(
        ap, choices=tuple(SOLVER_BUILDERS), default=DEFAULT_PERF_SOLVERS,
        multiple=True,
        help=f"solver families to compare (default: "
             f"{', '.join(DEFAULT_PERF_SOLVERS)}). Each contributes one pair "
             f"of bars per index")
    cli.add_factor_dtype(
        ap, choices=("complex64", "complex128"), default="complex64",
        help="precision of the low-precision factorization, u_f (default: "
             "complex64). complex32 is rejected; see Precision in the module "
             "docstring. complex128 is accepted only to measure the harness "
             "against itself")
    cli.add_inv_dtype(ap, default=np.dtype(DEFAULT_INV_DTYPE).name)
    ap.add_argument("--repeats", type=int, default=5, metavar="N",
                    help="runs of each variant; the median of each stage is "
                         "reported (default: 5)")
    ap.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                    metavar="N",
                    help=f"safety net on the outer refinement steps; the run "
                         f"normally stops well before it, when the forward "
                         f"error increases (default: {DEFAULT_MAX_ITER})")
    ap.add_argument("--ferr-tol", type=float, default=None, metavar="TOL",
                    help="accuracy the refined solution must reach to count as "
                         "converged. Default is cond(A,x) u from the "
                         "condition-est file, falling back to sqrt(n) u; the "
                         "same resolution mpir.py uses, so a run stops in the "
                         "same place in both studies")
    ap.add_argument("--reference-solver",
                    choices=["superlu", "mumps", "cudss", EXTENDED_REFERENCE],
                    default=EXTENDED_REFERENCE, metavar="NAME",
                    help=f"solver providing x_true, which the refinement "
                         f"loop's stopping rule watches and against which "
                         f"ferr_ref is measured. Computed once per index, "
                         f"outside every timed region, and charged to no "
                         f"solver (default: {EXTENDED_REFERENCE})")
    cli.add_output(ap, outdir_default=str(cli.MIXED_PREC_DIR),
                   outdir_help=f"directory holding <material>/"
                               f"<material>{PERF_SUFFIX}.h5, to which each run "
                               f"appends one numbered experiment "
                               f"(default: {cli.MIXED_PREC_DIR})")
    ap.add_argument("--reduce", choices=tuple(REDUCERS), default="median",
                    help="how the repeats of each stage become the number "
                         "drawn (default: median). 'min' is the better "
                         "estimator on a shared machine: contention only ever "
                         "adds time, so the fastest repeat is the least "
                         "contaminated one. It does not suppress the "
                         "stability check, which reads the full spread either "
                         "way")
    ap.add_argument("--stability-limit", type=float,
                    default=STABILITY_LIMIT, metavar="R",
                    help=f"flag any measurement whose slowest repeat exceeds "
                         f"its fastest by more than this ratio (default: "
                         f"{STABILITY_LIMIT:g}). Contention only ever adds "
                         f"time, so a large spread means the run was "
                         f"interrupted, not that the solver is variable")
    ap.add_argument("--no-save", action="store_true",
                    help="print the report but append no experiment")
    ap.add_argument("--list-experiments", action="store_true",
                    help="list the experiments already in the file and exit")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    material = args.material or h5path.stem
    out_path = perf_path(args.outdir, material)

    if args.list_experiments:
        list_experiments(out_path)
        return

    if args.factor_dtype == C32:
        ap.error("complex32 is not timed here; see Precision in --help")

    if args.energy is not None:
        if args.idx is not None or args.start is not None:
            ap.error("--energy is mutually exclusive with --idx/--start/--end")
        file_indices, file_energies, _, _ = load_energy_metadata(h5path)
        indices = [idx_of_energy(file_indices, file_energies, e)
                   for e in args.energy]
    else:
        indices = cli.resolve_indices(ap, args)

    # Rejected here rather than by whichever library was asked for it, so an
    # unsupported pairing costs nothing.
    for solver_name in args.solvers:
        if args.factor_dtype not in solver_dtypes(solver_name):
            ap.error(f"--solvers {solver_name} has no {args.factor_dtype} "
                     f"factorization; it accepts "
                     f"{', '.join(solver_dtypes(solver_name))}")

    low_dtype = np.dtype(args.factor_dtype)
    inv_dtype = np.dtype(args.inv_dtype)

    env = environment()
    load = loadavg("start")
    print(f"Problem : {h5path}")
    print(f"Solvers : {', '.join(args.solvers)}")
    print(f"Variants: {dtype_label(low_dtype)} + LU-IR, complex128 direct")
    print(f"Runs    : {args.repeats} per variant, {args.reduce} of each "
          f"stage; one untimed warm-up solve per precision before each "
          f"solver")
    n_warnings = check_environment(env, load)
    print()

    all_rows, skipped = [], []
    for idx in indices:
        if len(indices) > 1:
            print("=" * 92)
        try:
            all_rows.extend(measure_index(
                h5path, idx, args.solvers, low_dtype, inv_dtype,
                args.max_iter, args.repeats, args.reference_solver,
                args.ferr_tol, args.reduce))
        except SystemExit as e:                  # a bad index, from load_system
            skipped.append((idx, str(e)))
            print(f"E_{idx}: skipped ({e})")

    if skipped:
        print(f"\n{len(skipped)} of {len(indices)} indices skipped: "
              f"{', '.join(str(i) for i, _ in skipped)}")

    print_summary(all_rows)
    load.update(loadavg("end"))
    # Sampled now that every solver library is loaded; see _threadpools.
    pools_end = _threadpools()
    env["threadpools_end"] = (", ".join(pools_end) if pools_end
                              else "(threadpoolctl absent)")
    if pools_end and env["threadpools_end"] != env["threadpools"]:
        print(f"\n[note] thread pools at start: {env['threadpools']}")
        print(f"       thread pools at end:   {env['threadpools_end']}")
        print(f"       A pool appearing only at the end belongs to a solver "
              f"library loaded during the run.")
        print(f"       One sized well above the cap above is a solver "
              f"ignoring it, and is worth chasing.")
    n_unstable = check_stability(all_rows, args.stability_limit)
    if n_warnings and not n_unstable:
        print(f"\n[note] the run completed with {n_warnings} environment "
              f"warning(s) above; the repeats were nonetheless stable.")

    if args.no_save or not all_rows:
        return

    attrs = dict(
        material=material,
        source=str(h5path),
        timestamp=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        solvers=list(args.solvers),
        variants=list(VARIANTS),
        phases=list(PHASES),
        factor_dtype=args.factor_dtype,
        inv_dtype=args.inv_dtype,
        working_dtype=np.dtype(HIGH_DTYPE).name,
        inner="direct",
        inner_label="LU-IR",
        working_u=unit_roundoff(HIGH_DTYPE),
        u_f=unit_roundoff(low_dtype),
        max_iter=int(args.max_iter),
        ferr_tol=float(args.ferr_tol) if args.ferr_tol is not None else -1.0,
        reference_solver=args.reference_solver,
        repeats=int(args.repeats),
        reduce=args.reduce,
        indices=np.asarray(indices, dtype=np.int64),
        n_requested=len(indices),
        n_skipped=len(skipped),
        stability_limit=float(args.stability_limit),
        n_unstable=int(n_unstable),
        **env,
        **load,
        **material_metadata(h5path),
    )
    if skipped:
        attrs["skipped_idx"] = np.asarray([i for i, _ in skipped],
                                          dtype=np.int64)
        attrs["skipped_reason"] = [message for _, message in skipped]

    name = new_experiment(out_path, attrs)
    save_table(out_path, f"{EXPERIMENTS_GROUP}/{name}/runs", all_rows,
               columns=RUN_COLUMNS)
    print(f"\nwrote {out_path}:/{EXPERIMENTS_GROUP}/{name}")
    print(f"  runs  {len(all_rows)} rows (one per index, solver and variant)")
    if n_unstable:
        print(f"  {n_unstable} of {len(all_rows)} rows are unstable; the "
              f"figure will mark them, but the run should be repeated on a "
              f"quiet node with a capped thread pool.")
    print(f"  plot with: python ../plotting/mixed_prec_ir/plot_mpperf.py "
          f"{out_path} --experiment {int(name)}")


if __name__ == "__main__":
    main()

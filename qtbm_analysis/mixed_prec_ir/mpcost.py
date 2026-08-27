#!/usr/bin/env python3
"""
Time and memory cost of mixed-precision iterative refinement, across solvers.

Input
-----
    h5path, --idx      the systems A = E_<idx>/M and b = E_<idx>/rhs from a
                       material HDF5 file
    --solvers          the solver families to compare, in one experiment
    --factor-dtype     the low precision u_f; complex64 only, see Precision
    --repeats          repeats per variant; the median is reported

The companion of mpir.py. mpir.py asks whether refinement converges and to
what accuracy; this script asks what it costs, on the assumption that it does.
The two are separate scripts because they are separate measurements with
incompatible requirements: the convergence study varies precision and inner
solver over one solver family, while a cost study must hold precision fixed and
vary the family, and its numbers are only comparable when every variant of one
index is measured back to back in one process.

The refinement loops, the solver registry and the stopping criteria are
imported from mpir.py rather than reimplemented, so the two studies measure the
same code.

Precision
---------
u_f is complex64 and is not swept. The complex32 factorizations are
hand-written NumPy kernels that embed each complex block into a real one of
twice the dimension; their run time reflects that implementation and not the
cost of half-precision arithmetic, so timing them against LAPACK, MUMPS and
cuDSS would compare implementations rather than precisions. complex32 is
therefore rejected by the parser. Its accuracy behaviour is the subject of
mpir.py, where the comparison is meaningful.

--factor-dtype is kept as an option so the fixed choice is recorded in the
output rather than implied, and so that complex128 can be requested to measure
the harness overhead itself.

Variants
--------
Four variants are measured for every (index, solver), the layout of Amestoy et
al. (2023), table 2:

    c128_direct   the solver at complex128, no refinement. The baseline every
                  ratio is taken against: the double-precision direct solve
                  that mixed precision is meant to replace.
    c64_direct    the solver at complex64, no refinement. Not a usable answer
                  at these condition numbers; it is measured because the ratio
                  of its factorization time to the baseline's is the
                  factorization speedup, the quantity Zounon et al. (2022)
                  report, isolated from any refinement cost.
    luir          complex64 factorization + LU-IR
    gmresir       complex64 factorization + GMRES-IR

The two refinement variants differ in cost mainly through the number of
low-precision solves they perform: LU-IR does one per outer step, GMRES-IR one
per inner GMRES iteration. n_solves records it.

A speedup is only meaningful for a variant that reached the target accuracy.
The refinement variants stop on mpir's criteria, condition 5 of which compares
the forward error against the reference solution, so a converged run is one
that reached the accuracy of the reference; `converged` and `ferr_ref` are
recorded beside every timing and a figure must gate on them.

Timing
------
Three totals are recorded per variant and they measure different things:

    wall_s      end to end around the driver call, including the harness's own
                setup: the refinement drivers hold A at the working precision
                and copy it to get there, which the two direct variants do not.
    total_s     factor_s + inner_s, the algorithmic total. The stage
                breakdown below sums to exactly this, and it is the quantity
                the speedup and stacked-bar figures use.
    setup_s     wall_s - total_s, the harness overhead just described, kept as
                a column so the difference between the two totals is visible
                rather than absorbed.

total_s is split into stages, each measured strictly inside the total so the
parts cannot exceed it:

    factor_symbolic_s   ordering and symbolic factorization, or for the Block
                        Thomas families the detection and extraction of the
                        blocks. No floating-point arithmetic, so this stage
                        costs the same at every precision and bounds the
                        achievable speedup. SuperLU and UMFPACK fuse it into
                        the numerical phase and report nan.
    factor_numeric_s    the numerical factorization, the stage a lower u_f
                        makes cheaper.
    solve_s             the low-precision solves, n_solves of them.
    residual_s          the working-precision residuals b - Ax.
    other_s             what remains of inner_s: the update, the stopping
                        monitor and, for GMRES-IR, the products with A and the
                        orthogonalization.

Memory
------
Reported as three components that sum to working_mb:

    factor_mb   the stored factorization, from each solver's factor_nbytes.
    matrix_mb   A at the precision that variant must hold it: complex128 for
                the baseline and for both refinement variants, which form
                their residuals there, and complex64 for c64_direct. This is
                why the working set does not halve when the factorization
                does.
    krylov_mb   the inner GMRES basis, (restart + 1) vectors of length n at
                the working precision. Zero except for gmresir.

factor_nbytes returns 0 rather than raising where a backend exposes no size:
MUMPS when INFOG(3) is unreachable through the installed python-mumps, cuDSS
when the factorization info carries no lu_nnz. Zero bytes of factors is not a
possible measurement, so factor_mb_reported flags it and working_mb then
counts only the components that were measured. A figure must drop such a row
rather than draw it at zero.

Process-level memory is not reported, for the reason given in mpir.py: the
Block Thomas factors are NumPy arrays and visible to tracemalloc, while
SuperLU, MUMPS and cuDSS hold theirs in compiled extensions or on the device,
and peak RSS is order-dependent because the allocator does not return freed
pages. Only the per-solver figure is comparable.

Fairness
--------
Every variant of one index is measured in one process, back to back, against
the same A and b, after an untimed warm-up of any GPU solver involved. The
reference solution is computed once per index and shared by every solver, so
its cost is charged to none of them.

Solvers are compared on wall-clock cost as installed and configured on the
machine the script runs on. MUMPS and cuDSS are libraries with their own
threading and, for cuDSS, their own device; the Block Thomas families are
NumPy code calling LAPACK on dense blocks. A difference between two solvers
here is a difference between those implementations on this machine, not a
property of the algorithms in isolation.

Output
------
One numbered experiment appended per invocation to

    <outdir>/<material>/<material>_cost.h5
    └── experiments/0001/     attrs: the whole run configuration
        ├── runs        one row per (index, solver, variant)
        └── speedups    one row per (index, solver)

beside the convergence file mpir.py writes, so that one material directory
holds both studies and every figure drawn from either.

`speedups` is derived from `runs` and adds no measurement. It exists so that
the definition of each ratio -- which variant is the numerator, which the
denominator, which total is used -- is fixed once here rather than
reconstructed in each figure, where two figures could disagree. Every column
of it is reproducible from `runs`.

Rows are keyed by `variant_key`, one of c128_direct, c64_direct, luir,
gmresir, and never by the human-readable `variant` label, which carries the
solver name and precision and is for figure legends only.

Usage
-----
    python mpcost.py .../si-bulk.h5 --start 0 --end 200 --stride 20 \\
        --solvers mumps cudss block-thomas block-thomas-inv

    python mpcost.py .../carbon-nanotube.h5 --idx 84 254 \\
        --solvers superlu mumps block-thomas --repeats 5

    python mpcost.py .../si-bulk.h5 --list-experiments

References
----------
M. Zounon, N. J. Higham, C. Lucas and F. Tisseur, Performance impact of
    precision reduction in sparse linear systems solvers, PeerJ Comput. Sci.
    8:e778, 2022. The factorization speedup and the stage breakdown that
    explains it.
P. Amestoy et al., Combining sparse approximate factorizations with
    mixed-precision iterative refinement, ACM TOMS 49(1), 2023. The variant
    layout, and the normalized time and memory breakdowns.
"""

import datetime
import gc
import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))
import cli
from factor_io import save_table, load_table, material_metadata
from solver_classes import block_sizes_from_matrix

from mpir import (
    C32, HIGH_DTYPE, EXPERIMENTS_GROUP, SOLVER_BUILDERS,
    DEFAULT_RHO_THRESH, DEFAULT_MAX_ITER, DEFAULT_FERR_THRESH,
    DEFAULT_INV_DTYPE,
    benchmark_solver, dtype_label, experiment_names,
    load_condition_numbers, load_energy_metadata, load_system, energy_of_idx,
    new_experiment, resolve_experiment, experiment_attrs,
    solve_direct, solve_gmres_ir, solve_mixed_ir, solver_dtypes,
    unit_roundoff, _krylov_nbytes, _matrix_nbytes, _warm_up_gpu,
)

MIB = 1024.0 ** 2

# Solvers compared by default: every family with both a complex128 and a
# complex64 factorization. UMFPACK is absent because it has no single-precision
# build (cli.SOLVERS), so it cannot supply the c64 half of any ratio here; the
# two -fp16 families are absent for the reason given under Precision above.
DEFAULT_COST_SOLVERS = ("mumps", "cudss", "block-thomas", "block-thomas-inv",
                        "superlu")

COST_SUFFIX = "_cost"


# ─────────────────────────────────────────────────────────────────────────────
# The four variants
#
# `key` is what every row is keyed and every figure switched on. The label is
# built per solver for legends and carries no meaning a reader should parse.
#
# `matrix_dtype` is the precision that variant must hold A at, which is not
# always u_f: both refinement variants form their residual at the working
# precision and so keep A there however low u_f is. It is the reason the
# working set does not halve when the factorization does, and it is read by
# _memory_components.
#
# `krylov` marks the one variant that additionally holds a Krylov basis.
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_KEYS = ("c128_direct", "c64_direct", "luir", "gmresir")

VARIANT_SPEC = {
    "c128_direct": dict(inner="", is_refined=0, high=True,  krylov=False,
                        short="complex128 direct"),
    "c64_direct":  dict(inner="", is_refined=0, high=False, krylov=False,
                        short="{low} direct"),
    "luir":        dict(inner="direct", is_refined=1, high=False, krylov=False,
                        short="{low} + LU-IR"),
    "gmresir":     dict(inner="gmres", is_refined=1, high=False, krylov=True,
                        short="{low} + GMRES-IR"),
}

RUN_COLUMNS = [
    # identity
    "idx", "energy", "n", "nnz", "n_rhs", "n_blocks",
    "solver", "factor_dtype", "variant_key", "variant", "inner", "is_refined",
    # conditioning, for gating and for the x-axis of a speedup-against-kappa
    # figure; nan where the condition-est pipeline has no valid row
    "u_f", "u", "kappa_2", "kappa_inf", "lu_ir_bound",
    # accuracy: a timing is only meaningful for a variant that got the answer
    "relres", "ferr_ref", "converged", "outer_iters", "gmres_total",
    "stop_reason",
    # time
    "wall_s", "wall_s_min", "wall_s_max", "total_s", "setup_s",
    "factor_s", "factor_symbolic_s", "factor_numeric_s", "phases_split",
    "inner_s", "solve_s", "residual_s", "other_s", "n_solves",
    # memory
    "factor_mb", "factor_mb_reported", "matrix_mb", "krylov_mb", "working_mb",
    # provenance
    "repeats", "reference_solver", "reference_nbe",
]

SPEEDUP_COLUMNS = [
    "idx", "energy", "n", "nnz", "solver",
    # factorization speedup, complex128 over complex64: Zounon et al., figs 2-7
    "factor_s_c128", "factor_s_c64", "factor_speedup",
    "factor_numeric_s_c128", "factor_numeric_s_c64", "factor_numeric_speedup",
    # the fraction of the baseline factorization spent in the analysis phase,
    # which no precision reduction can touch: Zounon et al., figs 8-9
    "analysis_frac_c128", "phases_split",
    # end-to-end speedup of each variant over the complex128 direct solve:
    # Amestoy et al., fig 2 inverted
    "total_s_c128", "total_s_c64", "total_s_luir", "total_s_gmresir",
    "speedup_c64", "speedup_luir", "speedup_gmresir",
    # memory against the same baseline: Amestoy et al., fig 3
    "factor_mb_c128", "factor_mb_c64", "factor_mb_ratio", "factor_mb_reported",
    "working_mb_c128", "working_mb_luir", "working_mb_gmresir",
    "working_mb_ratio_luir", "working_mb_ratio_gmresir",
    # what the timings are worth
    "n_solves_luir", "n_solves_gmresir",
    "converged_luir", "converged_gmresir",
    "ferr_c128", "ferr_c64", "ferr_luir", "ferr_gmresir",
]


def cost_path(outdir, material):
    """
    <outdir>/<material>/<material>_cost.h5, beside the convergence file
    mpir.analysis_path returns for the same material.

    One directory per material holds both studies and every figure drawn from
    either, so the material is one `scp -r` unit. The two are separate files
    rather than two groups of one file because they are appended to
    independently and hold different tables.
    """
    return Path(outdir) / material / f"{material}{COST_SUFFIX}.h5"


def load_experiment(path, experiment=None):
    """
    (name, attrs, runs, speedups) for one experiment, the last by default.

    runs and speedups are the column dicts factor_io.load_table returns.
    `experiment` accepts either the padded name or the bare number.
    """
    name = resolve_experiment(path, experiment)
    attrs = experiment_attrs(path, name)
    base = f"{EXPERIMENTS_GROUP}/{name}"
    runs, _ = load_table(path, f"{base}/runs")
    speedups, _ = load_table(path, f"{base}/speedups")
    return name, attrs, runs, speedups


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────

def _variant_label(solver_name, key, low_name):
    """Legend text for one variant of one solver. Not parsed anywhere."""
    return f"{solver_name} {VARIANT_SPEC[key]['short'].format(low=low_name)}"


def _driver(key, solver_name, A, b, low_dtype, inv_dtype, opts):
    """
    The zero-argument callable benchmark_solver times for one variant.

    Every one of them is an mpir driver, so the cost study measures the same
    refinement loops the convergence study does.
    """
    spec = VARIANT_SPEC[key]
    if spec["is_refined"] == 0:
        dtype = HIGH_DTYPE if spec["high"] else low_dtype
        return lambda: solve_direct(solver_name, A, b, None, dtype, inv_dtype)
    if key == "luir":
        return lambda: solve_mixed_ir(
            solver_name, A, b, None, low_dtype, opts["max_iter"],
            x_true=opts["x_true"], normA=None, inv_dtype=inv_dtype,
            rho_thresh=opts["rho_thresh"], ferr_thresh=opts["ferr_thresh"])
    return lambda: solve_gmres_ir(
        solver_name, A, b, None, low_dtype, opts["max_iter"],
        x_true=opts["x_true"], gmres_tol=opts["gmres_tol"],
        gmres_restart=opts["gmres_restart"],
        gmres_max_iter=opts["gmres_max_iter"], normA=None,
        inv_dtype=inv_dtype, rho_thresh=opts["rho_thresh"],
        k_max=opts["k_max"], ferr_thresh=opts["ferr_thresh"])


def _memory_components(key, A_high, mem_bytes, low_dtype, gmres_restart):
    """
    (factor_mb, matrix_mb, krylov_mb, working_mb, reported) for one variant.

    working_mb is the sum of the components that were measured. Where the
    backend reported no factor size, `reported` is 0 and factor_mb is left at
    0 rather than guessed, so working_mb is then a lower bound on the true
    footprint; see the Memory section of the module docstring.
    """
    spec = VARIANT_SPEC[key]
    matrix_dtype = HIGH_DTYPE if (spec["high"] or spec["is_refined"]) \
        else low_dtype
    factor_mb = max(float(mem_bytes or 0), 0.0) / MIB
    matrix_mb = _matrix_nbytes(A_high, matrix_dtype) / MIB
    krylov_mb = (_krylov_nbytes(A_high.shape[0], gmres_restart) / MIB
                 if spec["krylov"] else 0.0)
    return (factor_mb, matrix_mb, krylov_mb,
            factor_mb + matrix_mb + krylov_mb, int(bool(mem_bytes)))


def _reduce(records, key):
    """Median over repeats of a top-level record key."""
    values = [r[key] for r in records if r.get(key) is not None]
    return float(np.median(values)) if values else float("nan")


def _reduce_extra(records, key, default=float("nan")):
    """Median over repeats of an `extra` key."""
    values = [r["extra"][key] for r in records if key in r["extra"]
              and r["extra"][key] is not None]
    return float(np.median(values)) if values else default


def _reduce_breakdown(records):
    """
    Median over repeats of the factorization phase split, or None where the
    backend fuses the phases.

    Reduced the same way factor_s is, and not read off the first repeat: the
    two are drawn as parts of one bar, so taking one from the median run and
    the other from an arbitrary run makes the parts disagree with the whole by
    the difference between two repeats.
    """
    pairs = [r["extra"]["factor_breakdown"] for r in records
             if r["extra"].get("factor_breakdown") is not None]
    if not pairs:
        return None
    return (float(np.median([p[0] for p in pairs])),
            float(np.median([p[1] for p in pairs])))


def measure_variant(key, solver_name, A, A_high, b, b_high, low_dtype,
                    inv_dtype, opts, repeats):
    """
    One (index, solver, variant) measurement, as a partial row.

    Returns None where the variant could not run -- an absent package, no CUDA
    device, a partition the solver rejects -- so that one missing solver does
    not abort the sweep. The reason is printed.
    """
    label = _variant_label(solver_name, key, dtype_label(low_dtype))
    print(f"    {label:<48}", end="", flush=True)
    fn = _driver(key, solver_name, A, b, low_dtype, inv_dtype, opts)
    try:
        records = benchmark_solver(fn, A_high, b_high, repeats,
                                   x_true=opts["x_true"], normA=None)
    except (ImportError, TypeError, RuntimeError, ValueError) as e:
        print(f"skipped ({type(e).__name__}: {e})")
        return None

    walls = np.array([r["wall_s"] for r in records], dtype=float)
    extra0 = records[0]["extra"]
    monitor = extra0.get("monitor")
    summary = monitor.summary() if monitor is not None else {}
    breakdown = _reduce_breakdown(records)
    gmres_history = extra0.get("gmres_iters_history", [])

    factor_s = _reduce_extra(records, "factor_s")
    inner_s = _reduce_extra(records, "inner_s")
    solve_s = _reduce_extra(records, "solve_s")
    residual_s = _reduce_extra(records, "residual_s", 0.0)
    total_s = factor_s + inner_s
    wall_s = float(np.median(walls))

    factor_mb, matrix_mb, krylov_mb, working_mb, reported = _memory_components(
        key, A_high, extra0.get("mem_bytes", 0), low_dtype,
        opts["gmres_restart"])

    spec = VARIANT_SPEC[key]
    # `converged` answers one question for every variant: did this variant
    # deliver the target accuracy, so that its timing counts. For the two
    # refinement variants it is the verdict of mpir's stopping monitor, whose
    # condition 5 compares the forward error against the reference solution.
    # The two unrefined variants run no loop and have nothing to converge, so
    # it is set from what they are: the complex128 solve defines the target
    # accuracy and the complex64 solve does not reach it, that being the
    # premise of the whole study. The measured ferr_ref sits beside it in
    # either case, so a figure can check rather than assume.
    converged = int(bool(summary.get("converged", False))) if monitor \
        else int(spec["high"])
    row = dict(
        variant_key=key,
        variant=label,
        inner=spec["inner"],
        is_refined=spec["is_refined"],
        relres=_reduce(records, "residual"),
        ferr_ref=_reduce(records, "true_err"),
        converged=converged,
        outer_iters=int(summary.get("outer_iters", 0)),
        gmres_total=int(sum(sum(g) for g in gmres_history))
                    if gmres_history else -1,
        stop_reason=summary.get("stop_reason", "no refinement"),
        wall_s=wall_s,
        wall_s_min=float(walls.min()),
        wall_s_max=float(walls.max()),
        total_s=total_s,
        setup_s=wall_s - total_s,
        factor_s=factor_s,
        factor_symbolic_s=float(breakdown[0]) if breakdown else float("nan"),
        factor_numeric_s=float(breakdown[1]) if breakdown else float("nan"),
        phases_split=int(breakdown is not None),
        inner_s=inner_s,
        solve_s=solve_s,
        residual_s=residual_s,
        other_s=inner_s - solve_s - residual_s,
        n_solves=int(_reduce_extra(records, "n_solves", 0)),
        factor_mb=factor_mb,
        factor_mb_reported=reported,
        matrix_mb=matrix_mb,
        krylov_mb=krylov_mb,
        working_mb=working_mb,
        repeats=int(repeats),
    )

    tag = "" if monitor is None else \
        f"  {row['outer_iters']} outer, {row['n_solves']} solves"
    print(f"{total_s*1e3:9.1f} ms   {working_mb:7.2f} MiB   "
          f"ferr {row['ferr_ref']:.2e}{tag}")

    # The iterate, correction and residual histories are O(n) per outer step
    # per repeat and are of no use to a cost study. Dropped before the next
    # variant is built so a long sweep does not accumulate them.
    del records, extra0
    gc.collect()
    return row


def measure_index(h5path, idx, solvers, low_dtype, inv_dtype, opts, repeats,
                  reference_solver):
    """
    Every (solver, variant) at one energy index, measured in one process.

    The system, its condition numbers and the reference solution are loaded
    and computed once here and shared by every solver, so the reference is
    charged to none of them and every solver is measured against byte-identical
    inputs.
    """
    A, b = load_system(h5path, idx)
    A_high = A.tocsc().astype(HIGH_DTYPE)
    b_high = np.asarray(b, dtype=HIGH_DTYPE)
    n = A.shape[0]
    n_rhs = b_high.shape[1] if b_high.ndim == 2 else 1

    indices, energies, _, _ = load_energy_metadata(h5path)
    energy = energy_of_idx(indices, energies, idx)
    kappa = load_condition_numbers(h5path, idx)
    u_f = unit_roundoff(low_dtype)

    print(f"E_{idx}  " +
          (f"E={energy:.4f} eV  " if energy is not None else "") +
          f"n={n}  nnz={A.nnz}  n_rhs={n_rhs}" +
          (f"  kappa_inf={kappa['inf']:.2e}  kappa_inf*u_f="
           f"{kappa['inf'] * u_f:.2e}" if kappa["inf"] is not None else
           "  kappa: not available"))

    # x_true, shared by every solver at this index. Its own backward error is
    # not recomputed here; ferr_ref is only as meaningful as the reference is,
    # which mpir.py reports for the same reference solver.
    x_true = None
    reference_nbe = float("nan")
    if reference_solver is not None:
        ref = SOLVER_BUILDERS[reference_solver](A, HIGH_DTYPE, None, b,
                                                DEFAULT_INV_DTYPE)
        x_true = ref.solve(b_high).astype(HIGH_DTYPE)
        reference_nbe = float(np.linalg.norm(A_high @ x_true - b_high)
                              / np.linalg.norm(b_high))
        if hasattr(ref, "free"):
            ref.free()
        del ref
        gc.collect()
    opts = dict(opts, x_true=x_true)

    # k_max, stopping condition 4, defaults to ceil(0.1 n) and so is resolved
    # per index rather than at parse time; see mpir.main.
    if opts.get("k_max_requested") is None:
        opts["k_max"] = math.ceil(0.1 * n)
    elif opts["k_max_requested"] <= 0:
        opts["k_max"] = None
    else:
        opts["k_max"] = int(opts["k_max_requested"])

    common = dict(
        idx=int(idx), n=int(n), nnz=int(A.nnz), n_rhs=int(n_rhs),
        energy=float(energy) if energy is not None else float("nan"),
        factor_dtype=dtype_label(low_dtype),
        u_f=float(u_f), u=float(unit_roundoff(HIGH_DTYPE)),
        kappa_2=kappa[2] if kappa[2] is not None else float("nan"),
        kappa_inf=kappa["inf"] if kappa["inf"] is not None else float("nan"),
        lu_ir_bound=(kappa["inf"] * u_f) if kappa["inf"] is not None
                    else float("nan"),
        reference_solver=reference_solver or "",
        reference_nbe=reference_nbe,
    )

    rows = []
    for solver_name in solvers:
        print(f"  {cli.label(solver_name)}")
        # The fixed device start-up cost is consumed before any variant of
        # this solver is timed, so it is not charged to whichever runs first.
        _warm_up_gpu(solver_name, A, b, None, low_dtype, inv_dtype)
        n_blocks = len(block_sizes_from_matrix(A)) \
            if solver_name.startswith("block-thomas") else -1
        for key in VARIANT_KEYS:
            row = measure_variant(key, solver_name, A, A_high, b, b_high,
                                  low_dtype, inv_dtype, opts, repeats)
            if row is not None:
                rows.append(dict(common, solver=solver_name,
                                 n_blocks=int(n_blocks), **row))
    return rows


def _ratio(numerator, denominator):
    """
    numerator / denominator, or nan where either is missing or the denominator
    is not positive. A speedup is never reported as inf or as a negative
    number: both would be a missing measurement drawn as a result.
    """
    if not np.isfinite(numerator) or not np.isfinite(denominator) \
            or denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def speedup_rows(run_rows):
    """
    The derived per-(index, solver) ratio table. Adds no measurement; every
    column is reproducible from `runs`.

    A ratio whose numerator or denominator is missing -- a variant that was
    skipped, a solver that reported no factor size, a phase split the backend
    does not expose -- is nan rather than a substituted value, so a figure
    drops the point instead of drawing an invented one.
    """
    grouped = {}
    for row in run_rows:
        grouped.setdefault((int(row["idx"]), row["solver"]), {})[
            row["variant_key"]] = row

    out = []
    for (idx, solver), variants in sorted(grouped.items()):
        def field(key, name, default=float("nan")):
            row = variants.get(key)
            return row[name] if row is not None else default

        any_row = next(iter(variants.values()))
        symbolic = field("c128_direct", "factor_symbolic_s")
        numeric = field("c128_direct", "factor_numeric_s")
        out.append(dict(
            idx=idx, solver=solver,
            energy=any_row["energy"], n=any_row["n"], nnz=any_row["nnz"],

            factor_s_c128=field("c128_direct", "factor_s"),
            factor_s_c64=field("c64_direct", "factor_s"),
            factor_speedup=_ratio(field("c128_direct", "factor_s"),
                                  field("c64_direct", "factor_s")),
            factor_numeric_s_c128=numeric,
            factor_numeric_s_c64=field("c64_direct", "factor_numeric_s"),
            factor_numeric_speedup=_ratio(
                numeric, field("c64_direct", "factor_numeric_s")),
            # The share of the baseline factorization that no precision
            # reduction can accelerate, hence the ceiling 1/(1 - f) on the
            # factorization speedup.
            analysis_frac_c128=_ratio(symbolic, symbolic + numeric),
            phases_split=int(field("c128_direct", "phases_split", 0)),

            total_s_c128=field("c128_direct", "total_s"),
            total_s_c64=field("c64_direct", "total_s"),
            total_s_luir=field("luir", "total_s"),
            total_s_gmresir=field("gmresir", "total_s"),
            speedup_c64=_ratio(field("c128_direct", "total_s"),
                               field("c64_direct", "total_s")),
            speedup_luir=_ratio(field("c128_direct", "total_s"),
                                field("luir", "total_s")),
            speedup_gmresir=_ratio(field("c128_direct", "total_s"),
                                   field("gmresir", "total_s")),

            factor_mb_c128=field("c128_direct", "factor_mb"),
            factor_mb_c64=field("c64_direct", "factor_mb"),
            factor_mb_ratio=_ratio(field("c64_direct", "factor_mb"),
                                   field("c128_direct", "factor_mb")),
            # 1 only when both sides of the memory ratios were measured.
            factor_mb_reported=int(field("c128_direct", "factor_mb_reported", 0)
                                   and field("c64_direct",
                                             "factor_mb_reported", 0)),
            working_mb_c128=field("c128_direct", "working_mb"),
            working_mb_luir=field("luir", "working_mb"),
            working_mb_gmresir=field("gmresir", "working_mb"),
            working_mb_ratio_luir=_ratio(field("luir", "working_mb"),
                                         field("c128_direct", "working_mb")),
            working_mb_ratio_gmresir=_ratio(
                field("gmresir", "working_mb"),
                field("c128_direct", "working_mb")),

            n_solves_luir=int(field("luir", "n_solves", -1)),
            n_solves_gmresir=int(field("gmresir", "n_solves", -1)),
            converged_luir=int(field("luir", "converged", 0)),
            converged_gmresir=int(field("gmresir", "converged", 0)),
            ferr_c128=field("c128_direct", "ferr_ref"),
            ferr_c64=field("c64_direct", "ferr_ref"),
            ferr_luir=field("luir", "ferr_ref"),
            ferr_gmresir=field("gmresir", "ferr_ref"),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(speedups):
    """
    The medians over the swept indices, one line per solver.

    A median over indices, not a mean: a single index whose factorization is
    pathological would move a mean and says nothing about the sweep. Ratios
    that are nan at an index -- a skipped variant, an unreported factor size --
    are excluded from that solver's median, and the count of contributing
    indices is printed so a line drawn from two points is not read as one drawn
    from twenty.
    """
    if not speedups:
        return
    by_solver = {}
    for row in speedups:
        by_solver.setdefault(row["solver"], []).append(row)

    def med(rows, key):
        values = [r[key] for r in rows if np.isfinite(r[key])]
        return (float(np.median(values)) if values else float("nan"),
                len(values))

    print("\nMedian over the swept indices "
          "(speedup = complex128 direct / variant, on total_s):")
    header = (f"  {'solver':<20}{'n':>4}{'fact':>8}{'numeric':>9}"
              f"{'analysis':>10}{'LU-IR':>8}{'GMRES-IR':>10}"
              f"{'mem LU-IR':>11}{'mem GM-IR':>11}{'conv':>10}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for solver, rows in by_solver.items():
        fact, n_fact = med(rows, "factor_speedup")
        numeric, _ = med(rows, "factor_numeric_speedup")
        analysis, _ = med(rows, "analysis_frac_c128")
        luir, _ = med(rows, "speedup_luir")
        gmres, _ = med(rows, "speedup_gmresir")
        mem_l, _ = med(rows, "working_mb_ratio_luir")
        mem_g, _ = med(rows, "working_mb_ratio_gmresir")
        n_luir = sum(1 for r in rows if r["converged_luir"])
        n_gmres = sum(1 for r in rows if r["converged_gmresir"])

        def cell(value, width, suffix=""):
            return (f"{value:>{width - len(suffix)}.2f}{suffix}"
                    if np.isfinite(value) else f"{'n/a':>{width}}")

        print(f"  {solver:<20}{len(rows):>4}"
              f"{cell(fact, 8, 'x')}{cell(numeric, 9, 'x')}"
              f"{cell(analysis * 100 if np.isfinite(analysis) else analysis, 10, '%')}"
              f"{cell(luir, 8, 'x')}{cell(gmres, 10, 'x')}"
              f"{cell(mem_l, 11, 'x')}{cell(mem_g, 11, 'x')}"
              f"{f'{n_luir}/{n_gmres}':>10}")
    print("    fact/numeric  factorization speedup, whole call and its "
          "numerical phase alone")
    print("    analysis      share of the complex128 factorization spent in "
          "the analysis phase,")
    print("                  which no precision reduction accelerates; n/a "
          "where the backend fuses the phases")
    print("    mem           working set relative to the complex128 direct "
          "solve, below 1 is a saving")
    print("    conv          indices where LU-IR / GMRES-IR reached the "
          "reference accuracy")


def list_experiments(path):
    """Print what a cost file already holds, one line per experiment."""
    names = experiment_names(path)
    if not names:
        print(f"{path} holds no experiments")
        return
    print(path)
    for name in names:
        _, attrs, runs, speedups = load_experiment(path, name)
        solvers = attrs.get("solvers", [])
        solvers = [solvers] if isinstance(solvers, str) else list(solvers)
        print(f"  {name}  {attrs.get('timestamp', '?'):<26} "
              f"{attrs.get('factor_dtype', '?'):<11} "
              f"{len(attrs.get('indices', []))} idx  "
              f"{len(solvers)} solvers ({', '.join(str(s) for s in solvers)})  "
              f"{len(runs.get('idx', []))} runs  "
              f"{len(speedups.get('idx', []))} speedup rows")


# ─────────────────────────────────────────────────────────────────────────────
# Command line
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap)
    cli.add_index_selection(ap, default_all=True)
    cli.add_solver_selection(ap, choices=tuple(SOLVER_BUILDERS),
                             default=list(DEFAULT_COST_SOLVERS), multiple=True)
    cli.add_factor_dtype(
        ap, choices=("complex64", "complex128"), default="complex64",
        help="the low precision u_f (default: complex64). complex32 is "
             "rejected: the half-precision factorizations are hand-written "
             "NumPy kernels, so timing them against LAPACK, MUMPS and cuDSS "
             "would compare implementations rather than precisions")
    cli.add_inv_dtype(ap, default=np.dtype(DEFAULT_INV_DTYPE).name)
    ap.add_argument("--repeats", type=int, default=3, metavar="N",
                    help="repeats per variant; the median is reported and the "
                         "spread is kept as wall_s_min and wall_s_max "
                         "(default: 3)")
    ap.add_argument("--reference-solver", choices=["superlu", "mumps", "cudss"],
                    default="mumps", metavar="NAME",
                    help="compute x_true with this solver at complex128, once "
                         "per index and shared by every solver measured "
                         "(default: mumps). Its cost is charged to none of "
                         "them")
    # The stopping criteria are mpir's; they are exposed here because they
    # determine how many solves a refinement variant performs and so are part
    # of what is being timed.
    ap.add_argument("--rho-thresh", type=float, default=DEFAULT_RHO_THRESH,
                    metavar="RHO",
                    help=f"mpir stopping condition 2 (default: "
                         f"{DEFAULT_RHO_THRESH})")
    ap.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                    metavar="N",
                    help=f"mpir stopping condition 3, maximum outer "
                         f"refinement steps (default: {DEFAULT_MAX_ITER})")
    ap.add_argument("--k-max", type=int, default=None, metavar="K",
                    help="mpir stopping condition 4, on the inner GMRES "
                         "iterations of one outer step. Defaults to "
                         "ceil(0.1 n); 0 disables it")
    ap.add_argument("--ferr-thresh", type=float, default=DEFAULT_FERR_THRESH,
                    metavar="RATIO",
                    help=f"mpir stopping condition 5, on the forward error "
                         f"against the reference (default: "
                         f"{DEFAULT_FERR_THRESH}); 0 or negative disables it")
    ap.add_argument("--gmres-tol", type=float, default=1e-8,
                    help="relative tolerance of the inner GMRES solve")
    ap.add_argument("--gmres-restart", type=int, default=30,
                    help="inner GMRES restart parameter; also sets the size "
                         "of the Krylov basis charged to krylov_mb")
    ap.add_argument("--gmres-max-iter", type=int, default=50,
                    help="maximum inner GMRES iterations per outer step")
    cli.add_output(ap, outdir_default=str(cli.MIXED_PREC_DIR),
                   outdir_help=f"directory holding <material>/"
                               f"<material>{COST_SUFFIX}.h5, to which each run "
                               f"appends one numbered experiment "
                               f"(default: {cli.MIXED_PREC_DIR})")
    ap.add_argument("--no-save", action="store_true",
                    help="print the report but append no experiment")
    ap.add_argument("--list-experiments", action="store_true",
                    help="list the experiments already in the cost file and "
                         "exit, without running anything")
    args = ap.parse_args()

    material = args.material or Path(args.h5path).stem
    out_path = cost_path(args.outdir, material)
    if args.list_experiments:
        list_experiments(out_path)
        return

    h5path = Path(args.h5path)
    indices = cli.resolve_indices(ap, args)
    solvers = list(dict.fromkeys(args.solvers))   # de-duplicated, order kept

    # Rejected here rather than minutes into a run; see mpir.solver_dtypes.
    if args.factor_dtype == C32:
        ap.error("--factor-dtype complex32 is not measured here; see the "
                 "Precision section of the module docstring")
    unusable = [s for s in solvers if args.factor_dtype not in solver_dtypes(s)]
    if unusable:
        ap.error(f"{', '.join(unusable)} ha{'s' if len(unusable) == 1 else 've'}"
                 f" no {args.factor_dtype} factorization, so no ratio against "
                 f"the complex128 baseline can be formed; drop "
                 f"{'it' if len(unusable) == 1 else 'them'} from --solvers")

    low_dtype = np.dtype(args.factor_dtype)
    inv_dtype = np.dtype(args.inv_dtype)
    ferr_thresh = args.ferr_thresh if args.ferr_thresh > 0 else None
    opts = dict(max_iter=args.max_iter, rho_thresh=args.rho_thresh,
                ferr_thresh=ferr_thresh, gmres_tol=args.gmres_tol,
                gmres_restart=args.gmres_restart,
                gmres_max_iter=args.gmres_max_iter,
                k_max_requested=args.k_max, k_max=None, x_true=None)

    print(f"Material : {h5path}")
    print(f"Solvers  : {', '.join(solvers)}")
    print(f"Precision: u_f = {args.factor_dtype}, u = "
          f"{np.dtype(HIGH_DTYPE).name}")
    print(f"Variants : {', '.join(VARIANT_KEYS)}")
    print(f"Runs     : {args.repeats} per variant (median reported)")
    print(f"Reference: {args.reference_solver} complex128, once per index\n")

    all_runs, skipped = [], []
    for idx in indices:
        try:
            all_runs.extend(measure_index(
                h5path, idx, solvers, low_dtype, inv_dtype, opts,
                args.repeats, args.reference_solver))
        except SystemExit as e:              # a bad index, from load_system
            skipped.append((idx, str(e)))
            print(f"E_{idx}: skipped ({e})")
        print()

    if not all_runs:
        raise SystemExit("No variant ran successfully -- see the messages "
                         "above.")

    speedups = speedup_rows(all_runs)
    print_summary(speedups)

    if args.no_save:
        return

    attrs = dict(
        material=material,
        source=str(h5path),
        timestamp=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        study="computational cost",
        solvers=solvers,
        variants=list(VARIANT_KEYS),
        factor_dtype=args.factor_dtype,
        inv_dtype=args.inv_dtype,
        working_dtype=np.dtype(HIGH_DTYPE).name,
        residual_dtype=np.dtype(HIGH_DTYPE).name,
        working_u=unit_roundoff(HIGH_DTYPE),
        factor_u=unit_roundoff(low_dtype),
        repeats=int(args.repeats),
        reference_solver=args.reference_solver,
        rho_thresh=float(args.rho_thresh),
        max_iter=int(args.max_iter),
        k_max=int(args.k_max) if args.k_max is not None else -1,
        ferr_thresh=float(ferr_thresh) if ferr_thresh is not None else -1.0,
        gmres_tol=float(args.gmres_tol),
        gmres_restart=int(args.gmres_restart),
        gmres_max_iter=int(args.gmres_max_iter),
        indices=np.asarray(indices, dtype=np.int64),
        n_requested=len(indices),
        n_skipped=len(skipped),
        speedup_basis="total_s = factor_s + inner_s, of the c128_direct variant",
        variant_layout="Amestoy et al. 2023, table 2",
        stage_breakdown="Zounon et al. 2022, figures 8 and 9",
        **material_metadata(h5path),
    )
    if skipped:
        attrs["skipped_idx"] = np.asarray([i for i, _ in skipped],
                                          dtype=np.int64)
        attrs["skipped_reason"] = [message for _, message in skipped]

    name = new_experiment(out_path, attrs)
    base = f"{EXPERIMENTS_GROUP}/{name}"
    save_table(out_path, f"{base}/runs", all_runs, columns=RUN_COLUMNS)
    save_table(out_path, f"{base}/speedups", speedups,
               columns=SPEEDUP_COLUMNS)
    print(f"\nwrote {out_path}:/{EXPERIMENTS_GROUP}/{name}")
    print(f"  runs      {len(all_runs)} rows (one per index, solver and variant)")
    print(f"  speedups  {len(speedups)} rows (one per index and solver)")
    print(f"  plot with: python ../plotting/mixed_prec_ir/plot_mpir_cost.py "
          f"{out_path} --experiment {int(name)}")


if __name__ == "__main__":
    main()

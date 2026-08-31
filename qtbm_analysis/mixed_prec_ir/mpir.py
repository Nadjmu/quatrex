#!/usr/bin/env python3
"""
Mixed-precision iterative refinement on QTBM material data.

Input
-----
    h5path, --idx      the system A = E_<idx>/M and b = E_<idx>/rhs from a
                       material HDF5 file
    --solver           which solver family provides the low-precision
                       factorization: superlu, umfpack, mumps, block-thomas,
                       block-thomas-inv or cudss
    --factor-dtype     the factorization precision, u_f below: complex128,
                       complex64 or complex32
    --inv-dtype        for block-thomas-inv at complex32 only, the precision
                       its explicit block inverses are formed in
    --inner            the inner correction solve, direct or gmres
    --max-iter         safety net on the outer steps; the stopping rule itself
                       takes no option, see Stopping below
    --ferr-tol         the accuracy that counts as converged, default cond(A,x) u
                       (falls back to sqrt(n) u where cond(A,x) is unavailable)
    --outdir           where the analysis file is written

The solver family and the factorization precision are chosen independently:
--solver selects the implementation and --factor-dtype the precision it runs
at. That is what makes the three variants below a comparison of precision
rather than of implementation. Not every combination exists; see Precisions.

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

so at u_f = 2^-24, complex64, the method is limited to kappa about 1e7, and at
u_f = 2^-11, complex32, to kappa about 1e3. QTBM matrices near a band edge
exceed both. The GMRES-based variant replaces that inner solve with
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

    5. x = x + dx. Repeat from 3 until a stopping criterion fires.

Stopping
--------
The loop stops on one rule: the forward error against the reference solution
increased. An increase means the correction just applied made the answer
worse, which is rounding noise rather than refinement, so the previous iterate
was the best the method reached and is what gets returned. --max-iter is a
safety net behind it, and the only thing that ends the loop when no reference
solution is available.

The rule is checked before the pass spends a solve, so a run stops the moment
it stops improving and outer_iters counts the corrections that produced the
returned solution. It costs a vector norm, not a solve.

This replaced the five conditions of Oktay and Carson, section 2.1.1 (the
correction no longer moves the iterate; corrections stopped shrinking
geometrically; an iteration limit; inner GMRES too long; the forward error
improving by less than a threshold) together with the Demmel et al. psi
estimate that declared convergence. Each needed a constant whose right value
varied with u_f and kappa_inf, and every one of them could cut off a run that
was still genuinely converging -- which happens precisely at the
ill-conditioned end of a sweep, biasing the iteration count downward exactly
where it is most interesting. An increase in the measured forward error needs
no constant.

Convergence is a separate question from stopping. Stopping says the method got
as far as it was going to; converged says whether that was far enough, and is

    best ferr <= --ferr-tol,   defaulting to cond(A, x) u

Corollary 3.3's own limiting accuracy, read from the condition-est file's
cond_skeel_x column, rather than sqrt(n) u, the level the working precision
can represent in the abstract. The two can differ by many orders of magnitude
on an ill-conditioned system: judging convergence against sqrt(n) u mistakes
the theorem's own limit for a failure, calling a run that reached exactly the
accuracy it was ever going to reach "not converged". sqrt(n) u remains the
fallback where cond_skeel_x is unavailable -- an older condition-est file, or
none at all -- and an explicit --ferr-tol always wins over either. See
RefinementMonitor.

A residual tolerance is the wrong criterion here: on these systems the
residual reaches the working precision long before the forward error does,
so it would declare convergence while the solution is still wrong, and it
cannot distinguish slow convergence from divergence at all.

Metrics
-------
The refinement variant's convergence is reported through the quantities of
Carson and Higham's Corollary 3.3, the right-hand panels of their numerical
experiments:

    phi_i = 2 u_s min(cond(A), kappa_inf(A) mu_i) + u_s ||E_i||_inf

split into a conditioning term and a correction-solver term, together with the
contraction actually observed. Both halves of the min are evaluated:
kappa_inf(A) and cond(A) = || |A^-1| |A| ||_inf come from the condition-est
pipeline, as its cond_inf and cond_skeel columns, and mu_i from the iterates
themselves, so the term is the Corollary's rather than an upper bound on it.
Which of the two halves the min selected is recorded per step. All of them are
reconstructed after the run,
from the iterates, corrections and residuals the loop retained, and never
inside it: the loop is timed and its memory is measured, so a diagnostic solve
performed there would corrupt the figures the report exists to give. See
refinement_metrics, which also states which of these are estimates and why.

These are recorded but not plotted. The one quantity a
convergence-versus-conditioning study reads is

    outer_iters  the corrections that produced the returned solution

together with converged, which says whether that solution reached the accuracy
above. Those two, against kappa_inf(A), are what decide whether mixed-precision
refinement is worth using on a given system, and are the whole content of the
summary figure; see plotting/mixed_prec_ir/plot_mpir.py. The Corollary 3.3
quantities stay in the iterations table for the per-index figure and for the
thesis text, and nothing is derived from them automatically.

outer_iters is only as good as the reference solution it is measured against,
which is why --reference-solver defaults to extended: a complex128 reference
has a forward error of the same order as refinement's own limiting accuracy,
so the stopping rule would fire when the reference ran out rather than when
the method did, earliest exactly where kappa_inf is largest. See
_ExtendedReference.

The preconditioner in the GMRES variant requires only the action of the
factorization as an operator, never explicit access to L and U. That is what
allows the same code to drive superlu, block-thomas, mumps and cudss
identically, including the two solvers that expose no factors at all.

Precisions
----------
--factor-dtype takes complex128, complex64 or complex32, and which of them a
--solver accepts is read from cli.SOLVERS; see solver_dtypes.

    complex128  every solver
    complex64   every solver except umfpack, which has no single-precision
                build
    complex32   block-thomas and block-thomas-inv only

complex32 is a storage label and not a NumPy dtype. There is no complex half
format, so the two Block Thomas implementations embed each complex block into a
real one of twice the dimension and hold that in float16; solver_classes
documents the embedding and the power-of-two scalings it requires. The general
sparse direct solvers offer no half-precision factorization at all, which is
why complex32 is restricted to those two families rather than being rejected at
run time by whichever library was asked for it.

Two consequences are visible in the output and are not incidental.

First, vectors entering a complex32 solver are cast to complex128, not to some
narrower type: the cast is then lossless and every rounding happens inside the
solver, where the embedding and the rescaling can be applied to it. Casting
narrower first would insert a second, unrelated rounding ahead of the
half-precision one and misattribute its effect. See cast_dtype.

Second, complex32 does not halve the stored matrix relative to complex64. The
embedding writes each of the two real components of an entry twice, so one
complex entry occupies 8 bytes in both cases, and what the narrower format
saves the redundancy gives back. See ITEMSIZE.

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
Only block-thomas and block-thomas-inv have a complex32 implementation, and
UMFPACK has no single-precision build at all. Both restrictions are properties
of the libraries and not of this script; --factor-dtype is checked against
solver_dtypes before anything is factored, so an unsupported pairing is
rejected by the parser rather than by a TypeError several minutes in.

The first cuDSS call in a process pays a fixed start-up cost for CUDA context
creation and kernel compilation that is independent of problem size, measured
at roughly 1.2 s. Left in place it would be charged in full to whichever
variant runs first, which is the refinement variant. An untimed warm-up solve
is therefore performed before any variant is measured; see _warm_up_gpu.

Output
------
A console report: the per-variant table of residual, forward error against the
reference solution, backward errors, wall time and factor memory, then the
convergence history and the convergence-factor panel described above.

Each invocation appends one numbered experiment to
<outdir>/<material>/<material>.h5, holding its whole configuration and both
result tables, unless --no-save is given. Nothing is ever overwritten, so the
file records every run made:

    experiments/0001/runs        one row per (index, variant)
    experiments/0001/iterations  one row per (index, outer step), plus one terminal row per index for the actually-returned solution -- see solve_mixed_ir

--list-experiments prints what a file already holds. See the Result file
section below.

References
----------
E. Oktay and E. Carson, Multistage mixed precision iterative refinement,
    Numer. Linear Algebra Appl. 29(4), 2022; arXiv:2107.06200. Section 2.1.1
    is the source of the stopping criteria used here.
J. Demmel et al., Error bounds from extra-precise iterative refinement,
    ACM TOMS 32(2), 2006. The psi estimate the convergence test uses.

Usage
-----
    python mpir.py .../carbon-nanotube.h5 --idx 5 --solver superlu \\
        --factor-dtype complex64

    python mpir.py .../si-bulk.h5 --idx 254 --solver mumps \\
        --factor-dtype complex64 --inner gmres --gmres-tol 1e-8 \\
        --gmres-restart 30 --gmres-max-iter 50

    python mpir.py .../si-bulk.h5 --idx 254 --solver cudss \\
        --factor-dtype complex64

The complex32 factorization is too inaccurate for LU-IR at these condition
numbers, so it is the GMRES-preconditioned variant that is worth running:

    python mpir.py .../carbon-nanotube.h5 --idx 84 --solver block-thomas \\
        --factor-dtype complex32 --inner gmres

    python mpir.py .../carbon-nanotube.h5 --idx 84 --solver block-thomas-inv \\
        --factor-dtype complex32 --inner gmres --inv-dtype float16

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
import datetime
import gc
import itertools
import math
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
    SparseLU, UMFPACK, MUMPS, BlockThomas, BlockThomasExplicitInv,
    BlockThomasFP16, BlockThomasExplicitInvFP16, CuDSS,
    extract_blocks_sparse, block_sizes_from_matrix, offband_nnz,
)
from bench_all import backward_errors, _matrix_norm, NORMWISE_ORDS
# _attr_value is factor_io's own coercion of a Python value into something
# h5py will store, including the list-of-strings case that NumPy cannot hold.
# Imported rather than repeated so that both writers agree on what an attribute
# becomes on disk.
from factor_io import save_table, load_table, material_metadata, _attr_value

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

# The working and residual precision, u and u_r in the analysis. Both are
# complex128; only the factorization precision u_f is varied.
HIGH_DTYPE = np.complex128


# ─────────────────────────────────────────────────────────────────────────────
# Result file
#
# One invocation of this script is one experiment, and every experiment is kept.
# Each run appends a new numbered group rather than overwriting the last, so the
# analysis file becomes a record of what was actually run:
#
#   <outdir>/<material>/<material>.h5
#   └── experiments/
#       ├── 0001/          attrs: the whole run configuration; see main
#       │   ├── runs        one row per (index, variant)
#       │   └── iterations  one row per (index, outer step), plus one terminal row per index for the actually-returned solution -- see solve_mixed_ir
#       ├── 0002/
#       └── ...
#
# One directory per material, not the flat <outdir>/<material>.h5 of
# cli.analysis_h5: plot_mpir.py writes its figures into a subdirectory of that
# same material directory (exp0001/, exp0002/, ...), so the whole material --
# the data and every figure ever drawn from it -- is one directory, `scp -r`
# able as a unit. See analysis_path.
#
# The experiment name is zero-padded because HDF5 orders keys as strings:
# unpadded, "10" would sort before "2" and a listing would come out in the
# wrong order.
#
# Every other analysis in this pipeline rewrites its one group in place, which
# is right for a sweep that is reproducible from its inputs. Refinement is not:
# the interesting runs differ in precision, inner solver and stopping
# thresholds, and comparing them is the point, so each is kept beside the
# configuration that produced it.
#
# The two tables are split rather than joined because they have different
# lengths. A figure of final accuracy against energy reads runs alone; a
# convergence trajectory or a phi panel reads iterations alone. Both carry idx,
# so either joins back to the other. All three variants appear in runs; only
# the refinement variant performs outer steps, so only it appears in iterations.
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS_GROUP = "experiments"

RUN_COLUMNS = [
    "idx", "energy", "n", "nnz", "n_rhs", "n_blocks",
    "solver", "factor_dtype", "inner", "variant", "is_refined",
    "u_f", "u", "u_s", "kappa_2", "kappa_inf", "cond_skeel", "cond_skeel_x",
    "lu_ir_bound",
    "relres", "ferr_ref", "eta1", "eta2", "etainf", "omega",
    "outer_iters", "converged", "ferr_best", "ferr_tol", "stop_reason",
    "gmres_total", "gmres_avg", "wall_s", "factor_s", "factor_symbolic_s",
    "factor_numeric_s", "inner_s", "solve_s", "residual_s", "other_s",
    "n_solves", "factor_mb", "factor_mb_reported", "working_mb",
    "reference_solver", "reference_nbe", "reference_floor",
]

ITERATION_COLUMNS = [
    "idx", "energy", "n", "nnz", "solver", "factor_dtype", "inner", "variant",
    "outer_iteration", "relres", "residual_norm_inf", "ferr_ref", "rho",
    "etainf", "omega",
    "mu_hat", "phi_cond_hat", "phi_solve_hat", "phi_hat",
    "phi_cond_binding", "phi_cond_form",
    "ferr_ratio",
    "correction_norm_inf", "reference_correction_norm_inf",
    "gmres_inner_iterations", "gmres_inner_max", "note",
]


def analysis_path(outdir, material):
    """
    <outdir>/<material>/<material>.h5, the analysis file for one material.

    One directory per material rather than cli.analysis_h5's flat
    <outdir>/<material>.h5: an experiment also has figures (see
    plotting/mixed_prec_ir/plot_mpir.py), and every figure a run produces goes
    into a subdirectory of this one. The whole material -- the data and every
    experiment run against it -- is then one directory, `scp -r`-able as a
    unit instead of as a scatter of same-prefixed files.
    """
    outdir = Path(outdir)
    return outdir / material / f"{material}.h5"


def experiment_names(path):
    """Existing experiment group names, in order. [] if the file has none."""
    path = Path(path)
    if not path.exists():
        return []
    with h5py.File(path, "r") as f:
        if EXPERIMENTS_GROUP not in f:
            return []
        return sorted(f[EXPERIMENTS_GROUP].keys())


def new_experiment(path, attrs):
    """
    Create the next numbered experiment group, write `attrs` onto it, and
    return its name. The caller then writes its own tables under
    experiments/<name>/.

    The number is the lowest unused one, so an experiment deleted by hand is
    reused rather than leaving a gap. The configuration is attached to the
    experiment group itself and not to the tables beneath it, so that a reader
    looks in one place for what a run was.

    Separate from save_experiment so that mpperf.py, whose experiments hold a
    different table, shares the numbering and the attribute convention rather
    than reimplementing them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        root = f.require_group(EXPERIMENTS_GROUP)
        used = {int(k) for k in root.keys() if k.isdigit()}
        number = next(i for i in itertools.count(1) if i not in used)
        name = f"{number:04d}"
        g = root.create_group(name)
        for key, value in attrs.items():
            if value is not None:
                g.attrs[key] = _attr_value(value)
    return name


def experiment_attrs(path, name):
    """Attribute dict of one experiment group, strings decoded."""
    with h5py.File(path, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in f[f"{EXPERIMENTS_GROUP}/{name}"].attrs.items()}


def resolve_experiment(path, experiment=None):
    """
    The name of the requested experiment, the last one by default.

    `experiment` accepts either the padded name or the bare number.
    """
    names = experiment_names(path)
    if not names:
        raise SystemExit(f"{path} holds no experiments; run the script that "
                         f"writes them first")
    if experiment is None:
        return names[-1]
    name = f"{int(experiment):04d}" if str(experiment).isdigit() \
        else str(experiment)
    if name not in names:
        raise SystemExit(f"{path} has no experiment {name}; it holds "
                         f"{', '.join(names)}")
    return name


def save_experiment(path, attrs, run_rows, iter_rows):
    """
    Append one experiment to the analysis file and return its name.

    See new_experiment for the numbering and the attribute convention.
    """
    name = new_experiment(path, attrs)
    base = f"{EXPERIMENTS_GROUP}/{name}"
    save_table(path, f"{base}/runs", run_rows, columns=RUN_COLUMNS)
    save_table(path, f"{base}/iterations", iter_rows, columns=ITERATION_COLUMNS)
    return name


def load_experiment(path, experiment=None):
    """
    (name, attrs, runs, iterations) for one experiment, the last by default.

    runs and iterations are the column dicts load_table returns. `experiment`
    accepts either the padded name or the bare number.
    """
    name = resolve_experiment(path, experiment)
    attrs = experiment_attrs(path, name)
    base = f"{EXPERIMENTS_GROUP}/{name}"
    runs, _ = load_table(path, f"{base}/runs")
    iters, _ = load_table(path, f"{base}/iterations")
    return name, attrs, runs, iters


# ─────────────────────────────────────────────────────────────────────────────
# Factorization precisions
#
# --factor-dtype names a precision, not necessarily a NumPy dtype: complex32 is
# a storage label for the embedded-real float16 factorizations, exactly as in
# cli.COMPLEX_DTYPES. Everything in this section exists so that the label can
# travel through the drivers unchanged and be resolved only at the three points
# where its meaning actually differs -- the cast applied before a solve, the
# unit roundoff u_f, and the bytes an entry occupies.
# ─────────────────────────────────────────────────────────────────────────────

C32 = "complex32"

# --solver family -> the cli.SOLVERS entry carrying its complex32
# implementation. bench_all treats the half-precision Block Thomas variants as
# solvers in their own right, since it sweeps solvers rather than precisions;
# here the family is chosen by --solver and the precision by --factor-dtype, so
# the paired entries are merged and complex32 becomes one more precision of the
# family it belongs to.
C32_SIBLING = {
    "block-thomas":     "block-thomas-fp16",
    "block-thomas-inv": "block-thomas-inv-fp16",
}

# Bytes one complex matrix entry occupies at each precision, for the memory
# rows. complex32 is not 4: the half-precision solvers hold the real embedding
# of each block, in which a complex entry becomes a 2x2 float16 block, so each
# of its two real components is stored twice. The embedding gives back in
# redundancy exactly what the narrower format saves, and an entry costs the
# same 8 bytes it does at complex64. Stated here rather than inferred from an
# itemsize, because there is no dtype to infer it from and because the equality
# is a result worth being explicit about.
ITEMSIZE = {"complex128": 16, "complex64": 8, C32: 8}


def is_c32(dtype):
    """True for the complex32 label, which no NumPy dtype corresponds to."""
    return isinstance(dtype, str) and dtype == C32


def dtype_label(dtype):
    """
    Canonical precision name for a --factor-dtype value.

    Accepts either the complex32 label or anything np.dtype understands, so
    that call sites need not know which they hold.
    """
    return C32 if is_c32(dtype) else np.dtype(dtype).name


def cast_dtype(dtype):
    """
    NumPy dtype a vector is cast to before it enters a solver built at `dtype`.

    For complex32 this is complex128, not a narrower type. The half-precision
    solvers embed to real float16 and perform their own rounding and
    power-of-two rescaling on every solve, so a complex128 argument is cast
    losslessly and all precision loss occurs inside the solver, where it
    belongs. Casting to complex64 on the way in would insert a second rounding
    that is no part of the method and would be charged to it.
    """
    return np.dtype(HIGH_DTYPE) if is_c32(dtype) else np.dtype(dtype)


def unit_roundoff(dtype):
    """
    The unit roundoff of a precision: half of np.finfo(...).eps, since eps is
    the spacing between 1 and the next representable float and the unit
    roundoff -- the maximum relative rounding error under round-to-nearest --
    is half of that spacing. This is Carson and Higham's u (Table 1.2: half
    precision 2^-11, single 2^-24, double 2^-53), the convention already used
    elsewhere in this project (run_bench/sweep_fp16.py,
    plotting/block-thomas/plot_fp16_accuracy.py) and in the module docstring's
    own worked example above (u_f = 2^-24 at complex64, 2^-11 at complex32).

    Used both for u_f, the factorization precision, and for u, the working
    precision (dtype=HIGH_DTYPE) -- one function so the two can never drift
    apart in convention.
    """
    return float(np.finfo(np.float16 if is_c32(dtype) else dtype).eps) / 2.0


def solver_dtypes(solver_name):
    """
    Precisions --solver `solver_name` accepts, from cli.SOLVERS merged with its
    complex32 sibling; see C32_SIBLING. cli.SOLVERS stays the one place a
    solver's supported precisions are recorded.
    """
    dtypes = list(cli.SOLVERS[solver_name]["dtypes"])
    sibling = C32_SIBLING.get(solver_name)
    if sibling is not None:
        dtypes += [d for d in cli.SOLVERS[sibling]["dtypes"] if d not in dtypes]
    return tuple(dtypes)


# ─────────────────────────────────────────────────────────────────────────────
# Solver registry. Each entry maps a --solver name to a builder with the
# signature builder(A, dtype, bs, b, inv_dtype) returning an object exposing
# solve(b) and factor_nbytes().
#
# Two of those arguments concern one builder each and are ignored by the rest:
# b, which cuDSS needs because it binds the right-hand-side column count at
# construction, and inv_dtype, which only block-thomas-inv at complex32 reads.
# Both are passed uniformly so that no call site has to know which solver it is
# about to build.
#
# dtype is the --factor-dtype value, so it may be the complex32 label rather
# than a NumPy dtype. Only the two Block Thomas builders can receive it; the
# parser rejects the combination for every other solver. See solver_dtypes.
# ─────────────────────────────────────────────────────────────────────────────

# Precision in which BlockThomasExplicitInvFP16 forms its explicit block
# inverses before rounding them to float16 for storage and application. Its own
# default, and the standard mixed-precision split: explicit inversion is the
# least stable step in that algorithm, and performing it in float16 loses
# accuracy no later step recovers. Under this setting the factorization is not
# purely complex32; --inv-dtype float16 makes it so.
DEFAULT_INV_DTYPE = np.float32

def _detected_blocks(A):
    """
    Partition detected from the sparsity pattern, validated with offband_nnz.

    The exported matrices have a non-uniform block structure, so the partition
    is always detected and never uniform, as in run_bench/single_solve.py and
    run_benchmarks.py's resolve_partition. A partition that cuts a real
    coupling yields a silently wrong solution, so it is validated before
    anything is factored.
    """
    bs = block_sizes_from_matrix(A)
    bad = offband_nnz(A, bs)
    if bad:
        raise ValueError(
            f"block partition leaves {bad} nonzeros outside the "
            f"block-tridiagonal band; the solution would be wrong")
    return bs


def _extract_timed(A):
    """
    (D, L, U, structural_s): the block extraction and how long it took.

    Detecting the partition, validating it and slicing the blocks out of the
    sparse matrix is structural work on the sparsity pattern. It performs no
    factorization arithmetic and its cost does not depend on the factorization
    precision, which places it in the same role as the reordering and symbolic
    phase of a general sparse direct solver: it is the part of the
    factorization that lowering the precision cannot accelerate. It is timed
    separately for that reason; see _factor_breakdown.
    """
    t0 = time.perf_counter()
    D, L, U = extract_blocks_sparse(A, _detected_blocks(A))
    return D, L, U, time.perf_counter() - t0


def _tag_phases(solver, structural_s, numeric_s):
    """
    Attach a factorization phase split to a solver object built here.

    The two Block Thomas families are constructed in two steps by the builders
    below rather than inside one class, so the split cannot be recorded by the
    class itself the way MUMPS and cuDSS record theirs. _factor_breakdown reads
    either form.
    """
    solver.symbolic_s = structural_s
    solver.numeric_s = numeric_s
    return solver


def _build_block_thomas(A, dtype, bs, b, inv_dtype):
    """
    Implementation 1 (LU with substitution), at complex128, complex64 or
    complex32. `bs` is ignored; see _detected_blocks. `inv_dtype` is ignored;
    this implementation inverts nothing.
    """
    D, L, U, structural_s = _extract_timed(A)
    t0 = time.perf_counter()
    solver = BlockThomasFP16(D, L, U) if is_c32(dtype) \
        else BlockThomas(D, L, U, dtype=dtype)
    return _tag_phases(solver, structural_s, time.perf_counter() - t0)


def _build_block_thomas_inv(A, dtype, bs, b, inv_dtype):
    """
    Implementation 2 (explicit block inverses), at complex128, complex64 or
    complex32. `bs` is ignored; see _detected_blocks. `inv_dtype` applies only
    at complex32, where it sets the precision the inverses are formed in before
    being rounded to float16; see DEFAULT_INV_DTYPE.
    """
    D, L, U, structural_s = _extract_timed(A)
    t0 = time.perf_counter()
    solver = BlockThomasExplicitInvFP16(D, L, U, inv_dtype=inv_dtype) \
        if is_c32(dtype) else BlockThomasExplicitInv(D, L, U, dtype=dtype)
    return _tag_phases(solver, structural_s, time.perf_counter() - t0)


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
        self._one = None        # single-vector solver; see prepare_single_rhs
        self._one_buf = None

    def prepare_single_rhs(self):
        """
        Build the nrhs=1 solver now rather than on the first single-vector
        solve.

        cuDSS binds the right-hand-side shape at construction, so a
        single-vector solve needs a solver of its own -- a second full plan and
        factorization of the same matrix. GMRES applies the preconditioner one
        vector at a time, so that second factorization is certain to be needed
        under GMRES-IR, and building it lazily would charge it to the first
        preconditioner application: the solve stage would absorb a whole
        factorization while factor_s understated the factorization by the same
        amount. Callers that know the single-vector path will be taken build it
        inside their factorization timing instead; see solve_gmres_ir.

        Idempotent.
        """
        if self._one is None:
            self._one = CuDSS(self._A, dtype=self.dtype, nrhs=1)
            self._one_buf = np.empty((self.n, 1), dtype=self.dtype, order="F")
        return self._one

    def factor_breakdown(self):
        """
        (analysis_s, numeric_s), summed over the solvers actually built.

        The nrhs=1 solver is a second factorization of the same matrix, so it
        is counted where it exists and the two phases then account for every
        factorization factor_s paid for.
        """
        built = [self._solver] + ([self._one] if self._one is not None else [])
        return (sum(x.plan_seconds for x in built),
                sum(x.factor_seconds for x in built))

    def _solve_block(self, b2d):
        self._buf[:, :] = b2d
        return self._solver.solve(self._buf)

    def _solve_one(self, v):
        self.prepare_single_rhs()
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


def _factor_breakdown(solver):
    """
    (analysis_s, numeric_s) of a built solver's factorization, or None where
    the backend fuses the two phases and exposes no split.

    The analysis phase -- the fill-reducing ordering and the symbolic
    factorization, or for the Block Thomas families the detection and
    extraction of the blocks -- performs no floating-point arithmetic, so its
    cost is the same at every factorization precision. It therefore bounds the
    speedup a lower precision can produce: a factorization spending a fraction
    f of its time there cannot be accelerated by more than 1/f however cheap
    the arithmetic becomes. Zounon et al. (2022, figures 8 and 9) identify this
    as the main reason sparse mixed-precision speedups fall short of 2.

    Two forms are read. MUMPS and cuDSS time their own phases and expose
    factor_breakdown(); the Block Thomas builders here are two-step and tag the
    solver with symbolic_s and numeric_s instead. SuperLU (scipy's splu) and
    UMFPACK fuse both phases into one call and return None.
    """
    fn = getattr(solver, "factor_breakdown", None)
    if callable(fn):
        breakdown = fn()
        if breakdown is not None:
            return breakdown
    symbolic = getattr(solver, "symbolic_s", None)
    numeric = getattr(solver, "numeric_s", None)
    if symbolic is None or numeric is None:
        return None
    return (float(symbolic), float(numeric))


# Keyed by canonical solver name; see solvers/cli.py.
SOLVER_BUILDERS = {
    "superlu":      lambda A, dtype, bs, b, inv_dtype: SparseLU(A, dtype=dtype),
    "umfpack":      lambda A, dtype, bs, b, inv_dtype: UMFPACK(A, dtype=dtype),
    "mumps":        lambda A, dtype, bs, b, inv_dtype: MUMPS(A, dtype=dtype),
    "block-thomas":     _build_block_thomas,
    "block-thomas-inv": _build_block_thomas_inv,
    "cudss":        lambda A, dtype, bs, b, inv_dtype: _CuDSSSolver(
        A, dtype, b.shape[1] if np.asarray(b).ndim == 2 else 1),
}

# Solvers whose first call in a process pays a fixed device start-up cost that
# must be excluded from the measurement. See _warm_up_gpu.
_GPU_SOLVERS = ("cudss",)

# Exceptions that mean "this factorization cannot be formed in this precision",
# as opposed to a bug. Raised by the complex32 Block Thomas families in
# solvers/solver_classes.py:
#
#   FloatingPointError  a block reached inf or nan and no power-of-two scale
#                       brings it back into fp16 range (_pow2_scale), or the
#                       explicit block inverse overflowed fp16
#   ZeroDivisionError   a pivot underflowed to exactly zero in fp16
#
# Both are ordinary outcomes of half precision on an ill-conditioned block, not
# programming errors. run_benchmarks catches them per variant (alongside
# ImportError/TypeError/RuntimeError, for the same reason an uninstalled MUMPS
# or a GPU-less cuDSS is skipped gracefully there): when it is the REFINED
# variant that fails, a synthetic run row records it as converged=False,
# outer_iters=0 rather than dropping the index, since these are typically the
# hardest indices in a sweep and a kappa_inf-vs-iterations figure that simply
# ends where they start is silently biased at the point it matters most.
#
# main()'s own except FACTORIZATION_FAILURES, one level up, is now a fallback
# for a failure occurring somewhere run_benchmarks does not wrap (there is
# none known at present) rather than the primary path -- it still drops the
# whole index, which is the right behaviour only if every variant failed the
# same way, including the complex128 baseline.
#
# ValueError is deliberately absent, so a shape mistake still crashes rather
# than being recorded as a numerical failure.
FACTORIZATION_FAILURES = (FloatingPointError, ZeroDivisionError)


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


# ─────────────────────────────────────────────────────────────────────────────
# The reference solution
#
# Every "forward error" this script reports is measured against a reference
# solution, so it is only as good as that reference. A reference computed by a
# complex128 direct solver carries a forward error of order cond(A, x) u, which
# on these matrices is 1e-11 or worse -- the same order as the limiting
# accuracy of refinement itself (Carson and Higham, Corollary 3.3, eq. 3.10).
# Measured against such a reference, ferr_ref stops decreasing when the
# reference runs out, not when the method does, and both of the quantities a
# convergence study wants are corrupted by it:
#
#   the iteration count, because stopping condition 5 fires when ferr stops
#   improving, so a coarse reference stops the loop early -- and it is coarsest
#   exactly where kappa is largest, which flattens the very trend being
#   measured;
#
#   the contraction rate, because the last steps then sit on the reference's
#   plateau rather than the method's.
#
# _ExtendedReference removes that by refining the complex128 solution with the
# residual accumulated in double-double, exactly as
# block-thomas/forward_error.py does. See its class docstring for what the
# gain actually is; it is not u^2.
# ─────────────────────────────────────────────────────────────────────────────

# The residual has to be computed in more than the working precision or
# refinement cannot improve on the solve it starts from. np.clongdouble is the
# obvious way to do that and is what this used to use, but its width is a
# property of the machine rather than of the algorithm: 80-bit x87 on x86-64
# (eps = 1.1e-19, done by the FPU) and IEEE binary128 on aarch64
# (eps = 1.9e-34, which no common CPU implements -- numpy emulates every
# operation in software). On an aarch64 node that made the O(nnz) matvec
# below, called n_rhs * (steps + 1) times per index, run at software-emulation
# speed: minutes per index on a QTBM block, buying twenty orders of magnitude
# more accuracy than the measurement can use.
#
# Double-double replaces it. A value is an unevaluated pair of float64s,
# hi + lo with |lo| <= ulp(hi)/2, giving a 106-bit significand. Every
# operation below is plain float64 arithmetic, so it runs at hardware speed
# and vectorises on every platform, and the reference no longer depends on
# what np.longdouble happens to be -- the same run on x86-64 and on aarch64
# now produces the same x_true.
#
# The error-free transformations are the standard ones: Dekker (1971) for the
# exact product, Knuth (1969) for the exact sum. See Ogita, Rump and Oishi,
# "Accurate sum and dot product", SIAM J. Sci. Comput. 26(6), 2005.
EPS_EXT = 2.0 ** -106
_DD_SPLITTER = 2.0 ** 27 + 1.0


def _two_sum(a, b):
    """(s, e) with s = fl(a + b) and a + b = s + e exactly."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _two_prod(a, b):
    """
    (p, e) with p = fl(a * b) and a * b = p + e exactly.

    Dekker's splitting rather than an FMA because numpy exposes no fused
    multiply-add; the split halves have 26 significand bits each, so every
    partial product below is exact in float64.
    """
    p = a * b
    ca = _DD_SPLITTER * a
    ah = ca - (ca - a)
    al = a - ah
    cb = _DD_SPLITTER * b
    bh = cb - (cb - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _dd_add(ah, al, bh, bl):
    """(ah, al) + (bh, bl), double-double, renormalised."""
    s, e = _two_sum(ah, bh)
    e = e + (al + bl)
    h = s + e
    return h, e - (h - s)


def _row_length_groups(indptr):
    """
    Rows bucketed by exact nonzero count, as (rows, positions) pairs.

    positions is an (rows_in_bucket, row_length) array of offsets into the CSR
    data array, so one bucket's nonzeros gather into a rectangular block that
    can be reduced along its rows with no padding and no ragged bookkeeping.
    A block-tridiagonal matrix has one distinct row length per block row, so
    this is a handful of buckets however large the matrix is -- which is the
    case this exists for. A matrix with many distinct row lengths degrades to
    many buckets and a correspondingly slow Python loop; nothing here is wrong
    in that case, it is just no longer the fast path.
    """
    counts = np.diff(indptr)
    groups = []
    for m in np.unique(counts):
        if m == 0:
            continue
        rows = np.flatnonzero(counts == m)
        groups.append((rows, indptr[rows][:, None] + np.arange(m)[None, :]))
    return groups


def _dd_reduce_rows(rh, rl, ih, il):
    """
    Sum each row of a rectangular complex double-double block.

    Pairwise, so the reduction is log2(width) vectorised passes rather than
    `width` of them -- the numpy call count then does not grow with how wide a
    block row is, which is what keeps this ahead of the emulated-quad version
    it replaces. Double-double addition is not associative, but its error
    bound does not depend on the order, so pairing costs no accuracy.
    """
    m = rh.shape[1]
    while m > 1:
        half = m // 2
        nrh, nrl = _dd_add(rh[:, :half], rl[:, :half],
                           rh[:, half:2 * half], rl[:, half:2 * half])
        nih, nil = _dd_add(ih[:, :half], il[:, :half],
                           ih[:, half:2 * half], il[:, half:2 * half])
        if m & 1:
            # Odd width: fold the unpaired last column into the first result.
            nrh[:, 0], nrl[:, 0] = _dd_add(nrh[:, 0], nrl[:, 0],
                                           rh[:, m - 1], rl[:, m - 1])
            nih[:, 0], nil[:, 0] = _dd_add(nih[:, 0], nil[:, 0],
                                           ih[:, m - 1], il[:, m - 1])
        rh, rl, ih, il = nrh, nrl, nih, nil
        m = half
    return rh[:, 0], rl[:, 0], ih[:, 0], il[:, 0]


# Refinement steps taken to build the reference. Three is what
# forward_error.py uses and is ample: the iteration contracts by roughly
# kappa_inf * u per step from a complex128 solve, so it reaches the residual
# precision's own limit within two.
DEFAULT_REF_STEPS = 3


class _ExtendedReference:
    """
    x_true by iterative refinement with a double-double residual.

    The initial solve and every correction use one SuperLU complex128
    factorization; only the residual b - A x is accumulated in double-double,
    the pair-of-float64 format defined above. SciPy has no sparse type in that
    format, so the product is formed from the CSR arrays directly: exact
    products via _two_prod, summed per row by a pairwise double-double
    reduction over rows of equal length.

    How much better this is, and why not more
    ----------------------------------------
    Refinement in a residual precision u_r converges to a relative error of
    order kappa_inf(A) u_r, so this reference improves on a plain complex128
    solve by roughly eps_double / EPS_EXT, about 1e16. Unlike the
    np.clongdouble version this replaced, that factor is now the same on every
    machine: EPS_EXT is 2^-106 by construction rather than whatever the
    platform's long double happens to be (1.1e-19 on x86-64, 1.9e-34 on
    aarch64), so a reference computed on one node matches one computed on
    another.

    It is still not a reason to treat the reference as exact:
    reference_floor = kappa_inf(A) * EPS_EXT is recorded per index so that a
    measured forward error within an order of magnitude of it can be
    recognised as a measurement of the reference rather than of the method.

    The solution is returned at complex128. Storing it wider would not help:
    what limits the measurement is the reference's own error, not the 1e-16 of
    rounding it for storage, and returning complex128 keeps it usable by every
    caller including the sparse products in refinement_metrics.
    """

    def __init__(self, A, max_steps=DEFAULT_REF_STEPS):
        self._csr = A.tocsr().astype(HIGH_DTYPE)
        # Split once. The reduction works on real and imaginary parts
        # separately, and re-deriving them per call was the single largest
        # avoidable cost in the version this replaced.
        self._re = np.ascontiguousarray(self._csr.data.real)
        self._im = np.ascontiguousarray(self._csr.data.imag)
        self._indices = self._csr.indices
        self._groups = _row_length_groups(self._csr.indptr)
        self._normA_inf = float(abs(self._csr).sum(axis=1).max())
        self._lu = spla.splu(A.tocsc().astype(HIGH_DTYPE))
        self._max_steps = int(max_steps)
        # Diagnostics of the last solve, read by run_benchmarks for the report.
        self.steps = 0
        self.residual = float("nan")

    def _matvec_dd(self, xh, xl):
        """
        A @ (xh + xl) in double-double, one column, as (re_h, re_l, im_h, im_l).

        xl is a correction of relative size ~1e-16, so A @ xl only has to be
        right to working precision: it is one ordinary sparse matvec rather
        than a second double-double one, which nearly halves the cost for no
        loss that reaches the result.
        """
        n = self._csr.shape[0]
        re_h, re_l = np.zeros(n), np.zeros(n)
        im_h, im_l = np.zeros(n), np.zeros(n)
        xr = np.ascontiguousarray(xh.real)
        xi = np.ascontiguousarray(xh.imag)
        for rows, pos in self._groups:
            a, bb = self._re[pos], self._im[pos]
            j = self._indices[pos]
            c, d = xr[j], xi[j]
            # (a + bi)(c + di) = (ac - bd) + (ad + bc)i, every product exact.
            h1, l1 = _two_prod(a, c)
            h2, l2 = _two_prod(bb, d)
            h3, l3 = _two_prod(a, d)
            h4, l4 = _two_prod(bb, c)
            prh, prl = _dd_add(h1, l1, -h2, -l2)
            pih, pil = _dd_add(h3, l3, h4, l4)
            srh, srl, sih, sil = _dd_reduce_rows(prh, prl, pih, pil)
            re_h[rows], re_l[rows] = srh, srl
            im_h[rows], im_l[rows] = sih, sil
        if xl is not None:
            corr = self._csr @ xl
            re_h, re_l = _dd_add(re_h, re_l, corr.real, 0.0)
            im_h, im_l = _dd_add(im_h, im_l, corr.imag, 0.0)
        return re_h, re_l, im_h, im_l

    def _residual_dd(self, b, xh, xl):
        """b - A @ (xh + xl) in double-double, as a pair of complex128."""
        prh, prl, pih, pil = self._matvec_dd(xh, xl)
        rh, rl = _dd_add(b.real, 0.0, -prh, -prl)
        ih, il = _dd_add(b.imag, 0.0, -pih, -pil)
        return rh + 1j * ih, rl + 1j * il

    def _solve_one(self, b):
        b = np.asarray(b, dtype=HIGH_DTYPE)
        xh, xl = self._lu.solve(b), None
        tol = EPS_EXT * 10
        for _ in range(self._max_steps):
            r_h, r_l = self._residual_dd(b, xh, xl)
            # The correction solve is a working-precision solve either way, so
            # the residual is rounded to complex128 here; what mattered was
            # computing it without the cancellation error of a complex128
            # matvec, which is what the double-double product above avoids.
            d = self._lu.solve(r_h + r_l)
            # The iterate is carried as a double-double between steps: rounded
            # back to complex128 every step it could never become more
            # accurate than the solutions it exists to judge.
            nh, nl = _dd_add(xh.real, 0.0 if xl is None else xl.real,
                             d.real, 0.0)
            mh, ml = _dd_add(xh.imag, 0.0 if xl is None else xl.imag,
                             d.imag, 0.0)
            xh, xl = nh + 1j * mh, nl + 1j * ml
            self.steps += 1
            norm_x = float(np.max(np.abs(xh)))
            if norm_x and float(np.max(np.abs(d))) <= tol * norm_x:
                break
        r_h, r_l = self._residual_dd(b, xh, xl)
        denom = self._normA_inf * float(np.max(np.abs(xh))) \
            + float(np.max(np.abs(b)))
        if denom:
            self.residual = max(0.0 if np.isnan(self.residual) else self.residual,
                                float(np.max(np.abs(r_h + r_l)) / denom))
        return np.asarray(xh + xl, dtype=HIGH_DTYPE)

    def solve(self, b):
        self.steps = 0
        self.residual = float("nan")
        return _solve_columns(self._solve_one, b)

    def fast_solve(self, b):
        """
        One plain complex128 triangular solve against the same factorization,
        skipping the extended-precision residual loop entirely.

        Used for refinement_metrics' phi_solve term, and for nothing else.
        x_true still comes from solve(), so no convergence quantity -- the
        stopping rule, outer_iters, converged, ferr_best, and through x_true
        also rho, mu_hat and phi_cond -- is affected by this method existing.

        Why the cheaper solve is used at all: the loop in _solve_one is
        dominated by _matvec_dd, an error-free-transformation product over
        the CSR arrays with no BLAS path, run n_rhs * (max_steps + 1) times
        per call. On a QTBM block (nnz in the millions, tens of right-hand sides)
        that is seconds to minutes *per outer iteration*, and phi_solve is a
        diagnostic that never reaches the summary figure or a convergence
        verdict. Measured on a synthetic block-tridiagonal system with
        nnz = 9.3e4 and 26 right-hand sides, this is about 160x faster, and
        the correction it returns differs by ~1e-27.

        What it costs, and it is not nothing
        ------------------------------------
        phi_solve compares the method's own correction against this one, so
        the reference is only meaningful while it is clearly more accurate
        than the correction under test. This solve carries cond(A,x) u, so
        whether that holds depends on u_s, the precision the method's own
        correction was formed in:

            LU-IR      u_s = u_f (6e-8 at complex64, 5e-4 at complex32).
                       cond(A,x) u stays well below that up to the condition
                       numbers LU-IR survives at all, so phi_solve is sound.
            GMRES-IR   u_s = u, the same working precision this solve's own
                       error is measured in. Once cond(A,x) grows, the two
                       become comparable and phi_solve_hat degrades into
                       noise -- exactly in the high-kappa regime GMRES-IR
                       exists for.

        So a GMRES-IR phi_solve at large kappa_inf should not be quoted from
        a run made this way. To restore the accurate term for one experiment,
        drop the getattr in run_benchmarks that prefers this method and let
        ref_solve call solve() again; nothing else needs to change.
        """
        return self._lu.solve(np.asarray(b, dtype=HIGH_DTYPE))

    def factor_nbytes(self):
        return 0

    def free(self):
        self._lu = None


# The --reference-solver name of the extended-precision reference. Kept out of
# SOLVER_BUILDERS deliberately: it is not a solver under test and must never be
# selectable as --solver.
EXTENDED_REFERENCE = "extended"


def build_reference(name, A, bs, b):
    """
    The reference solver named by --reference-solver, as an object with
    solve(b) and free().
    """
    if name == EXTENDED_REFERENCE:
        return _ExtendedReference(A)
    return SOLVER_BUILDERS[name](A, HIGH_DTYPE, bs, b, DEFAULT_INV_DTYPE)


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


# Datasets read out of the condition-estimate file, and the key each becomes
# in the dict load_condition_numbers returns. cond_skeel and cond_skeel_x were
# added to condition_est.py after cond_2 and cond_inf, so a file written before
# then holds only the first two; each is read independently and a missing one
# becomes None rather than failing the whole lookup.
CONDITION_DATASETS = {
    "cond_2": 2,
    "cond_inf": "inf",
    "cond_skeel": "skeel",
    "cond_skeel_x": "skeel_x",
}


def load_condition_numbers(h5path, idx):
    """
    The condition numbers of one energy index, from the material's own
    condition-estimate file, cli.CONDITION_DIR/<material>.h5 -- a separate file
    from h5path, written by the condition-est pipeline, not by
    run_benchmarks.py. Its /condition/indices holds the same energy indices as
    h5path's metadata/indices and /condition/valid marks rows where the
    estimate succeeded; condition_est.py computes all norms of one row
    together, so validity is shared between them.

    Returns {2: kappa_2, "inf": kappa_inf, "skeel": cond(A),
    "skeel_x": cond(A, x)}, each None where the file, the index, a valid row
    for it, or that particular dataset is absent, or where the stored value is
    not finite -- cond_skeel_x is written as NaN at an index whose right-hand
    side has no columns.

    What each is for:

    kappa_2   = sigma_max / sigma_min, the general-purpose conditioning figure
                reported throughout this project.
    kappa_inf = ||A||_inf ||A^-1||_inf. Reported alongside kappa_2 because the
                classical LU-IR convergence requirement, kappa_inf(A) u_f < 1
                (see the module docstring), is stated in the infinity norm, not
                the 2-norm; the two can differ enough that only one of them
                crosses the threshold near a band edge.
    skeel     = cond(A) = || |A^-1| |A| ||_inf, Skeel's condition number. This
                is the cond(A) of Carson and Higham's Corollary 3.3, the left
                half of its min(cond(A), kappa_inf(A) mu_i); see
                refinement_metrics. cond(A) <= kappa_inf(A) always, since the
                latter is the former's worst case over row scalings, so where
                the min binds on this term phi_cond is strictly smaller than
                the kappa_inf form alone would give.
    skeel_x   = cond(A, x) = || |A^-1| |A| |x| ||_inf / ||x||_inf. Not part of
                the convergence factor: it sets the limiting accuracy
                refinement can reach, roughly cond(A, x) u, rather than the
                rate at which it gets there. Carried through to the run table
                so a figure can draw that floor beside the forward error.
    """
    missing = {key: None for key in CONDITION_DATASETS.values()}
    cond_path = cli.CONDITION_DIR / f"{Path(h5path).stem}.h5"
    if not cond_path.exists():
        return missing
    with h5py.File(cond_path, "r") as f:
        if "condition/cond_2" not in f or "condition/cond_inf" not in f:
            return missing
        indices = f["condition/indices"][:]
        hit = np.flatnonzero(indices == idx)
        if hit.size == 0:
            return missing
        i = hit[0]
        valid = f["condition/valid"][:] if "condition/valid" in f else None
        if valid is not None and not valid[i]:
            return missing
        out = dict(missing)
        for name, key in CONDITION_DATASETS.items():
            path = f"condition/{name}"
            if path not in f:
                continue
            value = float(f[path][i])
            out[key] = value if np.isfinite(value) else None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stopping criterion
#
# One rule: stop as soon as the forward error against the reference solution
# increases. An increase means the correction just applied made the answer
# worse, which is rounding noise rather than refinement -- there is nothing
# left to converge to, and the previous iterate was the best the method
# reached. See RefinementMonitor.
#
# This replaces the five conditions of Oktay and Carson, section 2.1.1, that
# this module used previously (correction too small, successive corrections
# stopped shrinking, iteration limit, inner GMRES too long, forward error
# stopped improving by a threshold) together with the Demmel et al. psi
# convergence estimate. They were dropped deliberately: each needed a constant
# whose right value varied with u_f and kappa_inf, and every one of them could
# cut off a run that was still genuinely converging, which happens exactly at
# the large kappa_inf end of a sweep and so biased the iteration count downward
# there -- the one number this module now exists to report. An increase in the
# measured forward error needs no constant and cannot be argued with.
# ─────────────────────────────────────────────────────────────────────────────

# Cap on outer refinement steps. A safety net, not a convergence criterion:
# without a reference solution there is no forward error to watch and this is
# the only thing that ends the loop. 30 rather than 15 because complex32 at a
# large kappa_inf contracts by only a factor of two or three per step --
# measured on carbon-nanotube E_1608, 29 corrections were still productive.
DEFAULT_MAX_ITER = 30


def _inf_norm(X):
    """
    max |x_i| over every entry, which is the vector infinity norm and, for a
    multi-column right-hand side, the largest of its columns' infinity norms.

    The criteria below are stated for a single right-hand side. Aggregating
    across columns this way makes the stopping decision the one the worst
    column would have made on its own, so no column is declared converged on
    the strength of the others.
    """
    X = np.asarray(X)
    return float(np.abs(X).max()) if X.size else 0.0


class RefinementMonitor:
    """
    The stopping rule -- stop when the forward error increases -- and the
    convergence verdict that follows it.

    Usage is two calls per pass of the refinement loop. At the top, once the
    current iterate's forward error is known:

        if monitor.check(ferr): break        # ferr rose: keep the previous x

    and after a correction has been applied:

        if monitor.count(): break            # max_iter reached

    check() is deliberately at the *top*, before the pass spends a solve on a
    correction. Every criterion this class used to carry looked backwards --
    whether step i was worth taking could only be judged once step i + 1 had
    been formed -- so a run always paid for one correction it had already
    decided not to want. Comparing the forward error of the iterate in hand
    against the previous one needs nothing from the future, so the loop stops
    the moment the answer stops improving, and `best_iter` is the number of
    corrections that produced the best iterate rather than that number plus
    one.

    Why an increase, and not a threshold on the rate: an increase means the
    last correction made the answer worse. That is rounding noise, not
    refinement, and no constant has to be chosen to recognise it. Every
    threshold this class used to carry (a ratio of successive corrections, a
    ratio of successive forward errors, a bound on inner GMRES iterations)
    needed a value that depended on u_f and on kappa_inf, and each of them
    could cut off a run still genuinely converging -- which happens precisely
    at the ill-conditioned end of a sweep, biasing the iteration count downward
    exactly where it is most interesting.

    The verdict is separate from the rule. Stopping says the method got as far
    as it was going to; `converged` says whether that was far enough, and is
    simply

        best_ferr <= ferr_tol

    This class's OWN default for ferr_tol, when constructed with ferr_tol=None
    as here, is sqrt(n) u -- the accuracy the working precision can represent
    in the abstract. run_benchmarks does not rely on that default: it resolves
    a sharper level, Corollary 3.3's own limiting accuracy cond(A, x) u, from
    the condition-est file before ever constructing this monitor, and always
    passes a concrete number. That level is larger than sqrt(n) u on an
    ill-conditioned system, sometimes by many orders of magnitude, so judging
    convergence against sqrt(n) u there mistakes the theorem's own limit for a
    failure. --ferr-tol overrides either.

    Without a reference solution there is no forward error to watch: check()
    then never fires, the loop runs to max_iter, and `converged` stays False
    because nothing was measured that could establish it.

    This object only decides and records. It never touches the iterate, so the
    accuracy of the returned solution does not depend on it.
    """

    def __init__(self, u, n, max_iter=DEFAULT_MAX_ITER, ferr_tol=None):
        # u is the working precision, complex128 here for every variant, and
        # is deliberately not u_s: the question is whether refinement reached
        # the accuracy the working precision can hold, and asking it at u_f
        # would pass a solution only as good as the factorization refinement
        # set out to improve on.
        self.u = float(u)
        self.ferr_tol = (math.sqrt(n) * float(u) if ferr_tol is None
                         else float(ferr_tol))
        self.max_iter = int(max_iter)
        # ferr_0 = inf so the first iterate can never count as an increase.
        self._ferr_prev = math.inf
        self.iter = 0            # corrections applied so far
        self.best_iter = 0       # corrections applied to reach the best iterate
        self.best_ferr = None
        self.converged = False
        self.reasons = []
        # Per-pass history, recorded whether or not the loop stopped.
        self.ferr_history = []
        self.ferr_ratio_history = []

    def check(self, ferr):
        """
        Record the forward error of the iterate now in hand and report whether
        the loop should stop.

        `ferr` is ||x_i - x_ref||_inf / ||x_ref||_inf, the same quantity
        refinement_metrics reports as ferr_ref, or None when no reference
        solver was given. Returns True when it rose above the previous pass's.
        """
        w = None
        if ferr is not None and self._ferr_prev > 0:
            w = ferr / self._ferr_prev
        self.ferr_history.append(ferr)
        self.ferr_ratio_history.append(w)

        # A non-finite forward error means the iterate itself is inf or nan --
        # a low-precision factorization that overflowed without raising, so
        # everything downstream is garbage. Stop here rather than let it run to
        # max_iter: nan fails every comparison, so the increase test below
        # would never fire and the run would report a full-length iteration
        # count for a solution that never existed.
        if ferr is not None and not math.isfinite(ferr):
            self.reasons = [f"forward error is not finite ({ferr}); the "
                            f"low-precision solve produced inf or nan"]
            return True

        if ferr is not None and ferr > self._ferr_prev:
            self.reasons = [f"forward error increased "
                            f"({self._ferr_prev:.3e} -> {ferr:.3e})"]
            return True

        if ferr is not None:
            self._ferr_prev = ferr
            self.best_ferr = ferr
            self.best_iter = self.iter
        return False

    def at_limit(self):
        """
        True once max_iter corrections have been applied.

        Checked before spending another solve, and after check() has already
        recorded the iterate in hand, so a run that ends here has still had
        its last iterate measured -- unlike a limit tested after the update,
        which would leave the final correction's result unrecorded.
        """
        if self.iter >= self.max_iter:
            self.reasons = [f"iteration limit (iter={self.iter} >= "
                            f"max_iter={self.max_iter})"]
            return True
        return False

    def count(self):
        """Record that one more correction has been applied."""
        self.iter += 1

    def finish(self):
        """Settle the convergence verdict once the loop has ended."""
        self.converged = bool(self.best_ferr is not None
                              and self.best_ferr <= self.ferr_tol)
        return self.converged

    def summary(self):
        """The stopping decision, for the report and the result table."""
        return {
            "converged": self.converged,
            # The corrections that produced the returned solution, which is the
            # best iterate -- not self.iter, which also counts the one that
            # made things worse when the loop stopped on an increase. Without a
            # reference there is no best iterate to point at, so every applied
            # correction counts; reporting best_iter there would say 0 steps
            # for a loop that ran to max_iter.
            "outer_iters": (self.best_iter if self.best_ferr is not None
                            else self.iter),
            "ferr_best": self.best_ferr,
            "ferr_tol": self.ferr_tol,
            "stop_reason": "; ".join(self.reasons) if self.reasons
                           else "loop ended without a condition firing",
        }


# ─────────────────────────────────────────────────────────────────────────────
# The refinement variants
# ─────────────────────────────────────────────────────────────────────────────

def solve_mixed_ir(solver_name, A, b, bs, low_dtype, max_iter, x_true=None,
                   normA=None, inv_dtype=DEFAULT_INV_DTYPE, ferr_tol=None):
    """
    LU-IR: iterative refinement whose correction solve is a single
    low-precision triangular substitution.

    The factorization is computed once at low_dtype and reused at every outer
    iteration; the residual is formed in complex128. Convergence requires
    approximately kappa_inf(A) * u_f < 1, so this variant is the one that fails
    first as the matrix becomes ill-conditioned. See the module docstring.

    The loop stops when the forward error against the reference increases, not
    on a residual tolerance; see RefinementMonitor. Its effective solve
    precision is u_s = u_f: the correction is whatever the low-precision
    factors return, so refinement is limited by the precision the
    factorization was computed at.

    Parameters
    ----------
    low_dtype : the --factor-dtype value, u_f. May be the complex32 label; the
        vectors handed to the solver are cast with cast_dtype(low_dtype), which
        is complex128 in that case and leaves every rounding to the solver.
    x_true : complex128 reference solution, optional. The relative forward
        error against it is recorded every iteration and is what the stopping
        rule watches: the reference is the best available estimate of the exact
        solution, so once the iterate stops approaching it there is nothing
        left to gain. This is a cheap O(n) vector-norm comparison, not a solve,
        so it does not perturb the timing or memory figures. Without it the
        loop can only run to max_iter.
    inv_dtype : passed to the builder; read only by block-thomas-inv at
        complex32.
    ferr_tol : accuracy the returned solution must reach to count as converged.
        None means sqrt(n) u; see RefinementMonitor.

    Returns
    -------
    (x, extra), where x is the best iterate the loop reached rather than the
    last one it computed. Besides the residual history and the factor
    footprint, extra carries the iterates, the corrections and the residual
    vectors the correction solver was handed, from which refinement_metrics
    reconstructs every convergence quantity afterwards. Nothing derived is
    computed inside the timed region; see the module docstring.
    """
    b_high = np.asarray(b, dtype=HIGH_DTYPE)
    A_high = A.tocsc().astype(HIGH_DTYPE)
    cast = cast_dtype(low_dtype)
    norm_b = np.linalg.norm(b_high)
    norm_x_true = np.linalg.norm(x_true) if x_true is not None else None
    # The infinity-norm counterpart of norm_x_true, for the stopping rule,
    # which is stated in the infinity norm; see RefinementMonitor and
    # refinement_metrics.ferr_ref, which is exactly this quantity recomputed.
    norm_x_true_inf = _inf_norm(x_true) if x_true is not None else None
    history = []
    true_err_history = []
    x_history = []
    d_history = []
    r_history = []

    # The convergence level is stated at the working precision, not at u_f:
    # refinement is asked whether it reached the accuracy u can represent, and
    # answering that at u_f would accept a solution good only to the
    # factorization precision, which is what refinement exists to improve on.
    # u_f enters the metrics as u_s, never the stopping test.
    monitor = RefinementMonitor(unit_roundoff(HIGH_DTYPE), A_high.shape[0],
                                max_iter=max_iter, ferr_tol=ferr_tol)

    t0 = time.perf_counter()
    solver = SOLVER_BUILDERS[solver_name](A, low_dtype, bs, b, inv_dtype)
    factor_s = time.perf_counter() - t0

    # Stage timers inside the refinement loop. Each is a pair of
    # perf_counter calls around work measured in milliseconds, so the
    # instrumentation sits several orders of magnitude below the resolution of
    # what it measures. solve_s and residual_s are the two stages the
    # mixed-precision argument is about -- the low-precision substitutions and
    # the working-precision residuals -- and whatever is left of inner_s is the
    # vector arithmetic of the update and of the stopping monitor. Both are
    # measured strictly inside the inner_s window, so their sum cannot exceed
    # it and the remainder is non-negative by construction.
    solve_s = 0.0
    residual_s = 0.0
    n_solves = 0

    t_inner = time.perf_counter()
    t0 = time.perf_counter()
    x = solver.solve(b_high.astype(cast)).astype(HIGH_DTYPE)
    solve_s += time.perf_counter() - t0
    n_solves += 1

    # Every iterate the loop produces is recorded at the TOP of its pass,
    # including the final one, so x_history is complete and needs no fix-up
    # afterwards: the pass that stops does so before applying a correction, so
    # there is never an unrecorded iterate left over. d_history is therefore
    # one shorter than x_history, which refinement_metrics already guards for.
    #
    # x_best is what the function returns. When the loop stops on an increase,
    # the iterate in hand is by definition worse than the one before it, and
    # handing that back would mean knowingly returning the second-best answer
    # the method computed; the previous iterate is kept instead.
    x_best = x
    for _ in range(max_iter + 1):
        t0 = time.perf_counter()
        r = b_high - A_high @ x
        residual_s += time.perf_counter() - t0
        history.append(np.linalg.norm(r) / norm_b)
        ferr_live = None
        if x_true is not None:
            true_err_history.append(np.linalg.norm(x - x_true) / norm_x_true)
            if norm_x_true_inf:
                ferr_live = _inf_norm(x - x_true) / norm_x_true_inf
        # No copy: the update below rebinds x to a new array rather than
        # writing into it, so the retained reference is never overwritten.
        # Keep it that way -- an in-place `x +=` here would silently corrupt
        # every stored iterate.
        x_history.append(x)
        # The residual is retained in complex128, before the cast into the
        # solver. That is the system the correction equation poses; how the
        # tested method rounds it on the way in is part of what is measured.
        r_history.append(r)

        if monitor.check(ferr_live):
            break
        x_best = x
        if monitor.at_limit():
            break

        t0 = time.perf_counter()
        d = solver.solve(r.astype(cast)).astype(HIGH_DTYPE)
        solve_s += time.perf_counter() - t0
        n_solves += 1
        d_history.append(d)
        x = x + d
        monitor.count()
    inner_s = time.perf_counter() - t_inner
    monitor.finish()
    x = x_best

    extra = {
        "history": history,
        "true_err_history": true_err_history,
        "x_history": x_history,
        "d_history": d_history,
        "r_history": r_history,
        "monitor": monitor,
        "mem_bytes": solver.factor_nbytes(),
        "factor_s": factor_s,
        "inner_s": inner_s,
        "solve_s": solve_s,
        "residual_s": residual_s,
        # No Krylov work: LU-IR's correction is one triangular solve. Reported
        # as zero rather than omitted so that every driver returns the same
        # stage keys and a cost study needs no per-variant special case.
        "krylov_s": 0.0,
        "n_solves": n_solves,
        "factor_breakdown": _factor_breakdown(solver),
    }
    if hasattr(solver, "free"):
        solver.free()
    return x, extra


def _backward_error_histories(A_high, b_high, x_history, normA):
    """
    eta_inf and omega of every outer iterate, as two lists.

    Called by benchmark_solver once the wall timer has stopped, never from
    inside a refinement loop: omega needs |A| |x|, a second sparse matvec per
    iteration, which charged to the timed region would inflate the very
    figures the timing rows exist to report. The loops therefore only retain
    their iterates, and the errors are reconstructed here afterwards.

    Returns ([], []) when normA or the iterate history is absent.
    """
    if normA is None or not x_history:
        return [], []
    etainf_history, omega_history = [], []
    for xi in x_history:
        _, etas, omega = backward_errors(A_high, xi, b_high, normA)
        etainf_history.append(etas[np.inf])
        omega_history.append(omega)
    return etainf_history, omega_history


def refinement_metrics(A_high, x_true, extra, u_s, kappa_inf, ref_solve=None,
                       cond_skeel=None):
    """
    The per-iteration convergence quantities of Carson and Higham, Corollary
    3.3, reconstructed from the arrays a refinement run retained.

    Every quantity here is computed after the fact, from x_history, d_history
    and r_history, and never inside the refinement loop: the loop is timed and
    the memory rows are read off it, so a diagnostic solve or an extra matrix
    product performed there would inflate the very figures the report exists to
    give. Nothing in this function can change the solution either.

    Corollary 3.3 states that the forward error contracts by roughly

        phi_i = 2 u_s min(cond(A), kappa_inf(A) mu_i)  +  u_s ||E_i||_inf
              = phi_cond                               +  phi_solve

    per iteration, and that refinement converges while phi_i is comfortably
    below 1. The two terms separate the two things that can stop it: the left
    one is conditioning together with the direction the current error points
    in, the right one is how accurately the correction equation was solved.

    What is estimated, and why
    --------------------------
    x_true is a numerical reference, not the exact solution, so every quantity
    below is an estimate and is named with a hat in the output. In particular:

    ferr_ref   ||x_true - x_i||_inf / ||x_true||_inf, the forward error
               relative to the reference solution. It is not a certified
               forward error.
    rho        ferr_ref[i+1] / ferr_ref[i], the contraction actually observed.
               This is the measurement phi_hat is a prediction of; the two
               need not agree closely, since phi_hat is built from estimated
               and directional quantities.
    mu_hat     ||A (x_true - x_i)||_inf / (||A||_inf ||x_true - x_i||_inf), the
               factor of Carson and Higham (3.1). It lies between
               1/kappa_inf(A) and 1 and is small exactly when the error points
               along the directions A damps, which is what lets refinement
               converge at condition numbers the plain bound forbids.
    phi_cond   2 u_s min(cond(A), kappa_inf(A) mu_hat), the Corollary's term
               as stated, with cond(A) = || |A^-1| |A| ||_inf read from the
               condition-est pipeline's cond_skeel column. Both halves are
               bounds on the same quantity, || |A^-1| |A| |e_i| ||_inf /
               ||e_i||_inf: the left one holds it worst-case over the direction
               of the error, the right one uses the direction this step's error
               actually took. Whichever is smaller is the one the Corollary
               asserts, and which of the two bound is recorded per step in
               phi_cond_binding, since a run in which the min is always taken
               on cond(A) is one where mu_hat carries no information.

               Only one of the two is used where the other is unavailable: an
               older condition-est file has no cond_skeel column, and mu_hat is
               undefined once the error reaches zero. The form actually
               evaluated is recorded as phi_cond_form, so a row is never
               ambiguous about which it holds. Dropping either half can only
               overstate phi_cond, so a partial estimate stays conservative.
    phi_solve  ||d_i - d_i_ref||_inf / ||d_i_ref||_inf, where d_i_ref solves
               A d = r_i for the same retained residual r_i with the reference
               solver. Under d_hat = (I + u_s E) d this measures
               u_s ||E_i d_i|| / ||d_i||, the error in the direction the
               correction actually took, rather than the worst case
               u_s ||E_i||_inf that the Corollary bounds with. It is therefore
               a lower estimate of the true term.

    Parameters
    ----------
    u_s : effective precision of the correction solve; u_f for LU-IR and u for
          GMRES-IR, set by whichever driver produced `extra`.
    kappa_inf : kappa_inf(A) from the condition-est pipeline, or None.
    cond_skeel : cond(A) = || |A^-1| |A| ||_inf from the same pipeline, or
          None. Supplying it is what makes phi_cond the Corollary's min rather
          than its kappa_inf(A) mu_hat half alone.
    ref_solve : callable solving A d = r at the working precision, used for
          phi_solve. None skips that term, and phi_hat is then phi_cond alone.

    Returns a list of per-iteration dicts, one per outer step.
    """
    x_history = extra.get("x_history", [])
    d_history = extra.get("d_history", [])
    r_history = extra.get("r_history", [])
    gmres_hist = extra.get("gmres_iters_history", [])
    residuals = extra.get("history", [])
    # Per-iterate backward errors, already reconstructed by benchmark_solver
    # from the same retained iterates; carried through here so that one row per
    # step holds everything a convergence figure needs.
    etainf_hist = extra.get("etainf_history", [])
    omega_hist = extra.get("omega_history", [])
    monitor = extra.get("monitor")
    if not x_history:
        return []

    normA_inf = _matrix_norm(A_high, np.inf)
    norm_x_true = _inf_norm(x_true) if x_true is not None else None

    rows = []
    for i, x_i in enumerate(x_history):
        row = {
            "outer_iteration": i,
            "relres": residuals[i] if i < len(residuals) else None,
            "residual_norm_inf": _inf_norm(r_history[i]) if i < len(r_history) else None,
            "correction_norm_inf": _inf_norm(d_history[i]) if i < len(d_history) else None,
            "etainf": etainf_hist[i] if i < len(etainf_hist) else None,
            "omega": omega_hist[i] if i < len(omega_hist) else None,
        }

        # Forward error against the reference, and the factor mu it implies.
        if x_true is not None and norm_x_true:
            err = x_true - (x_i if x_i.ndim == x_true.ndim else x_i[:, 0])
            ferr = _inf_norm(err) / norm_x_true
            row["ferr_ref"] = ferr
            norm_err = _inf_norm(err)
            if norm_err > 0 and normA_inf:
                row["mu_hat"] = _inf_norm(A_high @ err) / (normA_inf * norm_err)

        # The conditioning term, min(cond(A), kappa_inf(A) mu_hat), with each
        # half dropped where its input is missing rather than substituted for:
        # recorded as None rather than 0 when neither is available, so that a
        # missing input is never read as a small one.
        mu = row.get("mu_hat")
        candidates = {}
        if cond_skeel is not None:
            candidates["cond"] = cond_skeel
        if kappa_inf is not None and mu is not None:
            candidates["kappa_mu"] = kappa_inf * mu
        if candidates:
            binding = min(candidates, key=candidates.get)
            row["phi_cond_hat"] = 2.0 * u_s * candidates[binding]
            row["phi_cond_binding"] = binding
            row["phi_cond_form"] = ("2*u_s*min(cond_skeel, kappa_inf*mu_hat)"
                                    if len(candidates) == 2
                                    else "2*u_s*cond_skeel"
                                    if binding == "cond"
                                    else "2*u_s*kappa_inf*mu_hat")
        else:
            row["phi_cond_hat"] = None
            row["phi_cond_binding"] = "unavailable"
            row["phi_cond_form"] = "unavailable"

        # The correction-solver term, against a reference solve of the same
        # residual this step actually posed.
        if ref_solve is not None and i < len(r_history) and i < len(d_history):
            try:
                d_ref = ref_solve(r_history[i])
                norm_d_ref = _inf_norm(d_ref)
                row["reference_correction_norm_inf"] = norm_d_ref
                if norm_d_ref > 0:
                    row["phi_solve_hat"] = _inf_norm(d_history[i] - d_ref) / norm_d_ref
            except (ImportError, TypeError, RuntimeError, ValueError) as e:
                row["phi_solve_hat"] = None
                row["note"] = f"reference correction failed: {type(e).__name__}: {e}"

        parts = [row.get("phi_cond_hat"), row.get("phi_solve_hat")]
        row["phi_hat"] = sum(p for p in parts if p is not None) \
            if any(p is not None for p in parts) else None

        # The stopping rule's own quantity for this step, from the monitor that
        # made the decision; see RefinementMonitor. ferr_ratio > 1 is the step
        # the loop stopped on.
        if monitor is not None and i < len(monitor.ferr_ratio_history):
            row["ferr_ratio"] = monitor.ferr_ratio_history[i]

        if i < len(gmres_hist):
            iters = gmres_hist[i]
            row["gmres_inner_iterations"] = int(sum(iters))
            row["gmres_inner_max"] = int(max(iters)) if iters else 0

        rows.append(row)

    # Observed contraction, which needs the next step's forward error and so is
    # filled in once the whole history exists.
    for i, row in enumerate(rows):
        this, nxt = row.get("ferr_ref"), rows[i + 1].get("ferr_ref") \
            if i + 1 < len(rows) else None
        if this and nxt is not None:
            row["rho"] = nxt / this
    return rows


def _gmres_solve(A_op, rhs, M_op, tol, restart, max_inner_iters, callback):
    """
    Call scipy.sparse.linalg.gmres, correcting for a documented but easy to
    miss unit mismatch, and compatibly across SciPy versions.

    scipy.sparse.linalg.gmres's `maxiter` parameter counts INNER iterations
    only under callback_type='legacy' (its default when callback_type is not
    given). This module always requests callback_type='pr_norm', to get a
    per-iteration residual for --gmres-tol and for the diagnostic iteration
    counts this module reports -- and the documented effect of asking for
    'pr_norm' (or 'x') explicitly is that `maxiter` switches to counting
    RESTART CYCLES instead. This is stated in SciPy's own docstring for
    callback_type and is identical in every SciPy version checked (1.11
    through 1.13): it is not a version-compatibility issue, unlike the
    tol/rtol rename below, which is.

    Silently passing max_inner_iters=50 straight through as `maxiter` under
    'pr_norm' therefore does not cap the run at 50 iterations -- it caps it at
    50 restart cycles, i.e. up to 50 * restart individual iterations (1500 at
    the default restart=30). Measured on carbon-nanotube E_1605 at complex32:
    a single inner solve ran 1427 real iterations under a nominal
    --gmres-max-iter 50, and the resulting outer step took 4.5 minutes. The
    fix is to convert the caller's intended total into the restart-cycle count
    scipy actually wants: ceil(max_inner_iters / restart). The realised cap is
    then a multiple of `restart`, rounded up from what was asked for, never
    down, so an outer step still costs at most what --gmres-max-iter promises,
    to within one restart cycle.

    The tol keyword was renamed to rtol in SciPy 1.12; the fallback covers
    older installations. That rename is unrelated to the maxiter fix above,
    which applies identically in both branches.
    """
    restart_cycles = max(1, math.ceil(max_inner_iters / restart))
    try:
        return spla.gmres(A_op, rhs, M=M_op, rtol=tol, atol=0.0, restart=restart,
                          maxiter=restart_cycles, callback=callback,
                          callback_type="pr_norm")
    except TypeError:
        return spla.gmres(A_op, rhs, M=M_op, tol=tol, atol=0.0, restart=restart,
                          maxiter=restart_cycles, callback=callback,
                          callback_type="pr_norm")


def solve_gmres_ir(solver_name, A, b, bs, low_dtype, max_iter, x_true=None,
                   gmres_tol=1e-8, gmres_restart=30, gmres_max_iter=50,
                   normA=None, inv_dtype=DEFAULT_INV_DTYPE, ferr_tol=None):
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

    The loop stops when the forward error against the reference increases; see
    RefinementMonitor. Its effective solve precision is u_s = u, the working
    precision: GMRES iterates on the preconditioned system in complex128, so
    the correction is as accurate as the working precision allows however low
    u_f is. That is the whole reason this variant tolerates condition numbers
    LU-IR cannot.

    Parameters
    ----------
    low_dtype, inv_dtype : as in solve_mixed_ir. At complex32 the
        preconditioner applications are the only place the half-precision
        factorization is touched, and each vector reaches it cast losslessly to
        complex128; see cast_dtype.
    x_true : as in solve_mixed_ir: recorded for reporting, and what the
        stopping rule watches.
    ferr_tol : accuracy the returned solution must reach to count as
        converged. None means this monitor's own default, sqrt(n) u;
        run_benchmarks does not rely on that default and always passes a
        resolved value, cond(A,x) u where available. See RefinementMonitor.

    Returns
    -------
    (x, extra), where x is the best iterate the loop reached rather than the
    last one it computed, and extra carries the residual history, the iterates,
    the corrections, the residual vectors handed to the correction solve, the
    inner GMRES iteration counts and the factor footprint.
    """
    b_high = np.asarray(b, dtype=HIGH_DTYPE)
    orig_ndim = b_high.ndim
    b2 = b_high if orig_ndim == 2 else b_high[:, None]
    A_high = A.tocsc().astype(HIGH_DTYPE)
    cast = cast_dtype(low_dtype)
    n, k = A_high.shape[0], b2.shape[1]
    norm_b = np.linalg.norm(b_high)

    x_true2 = None
    norm_x_true = None
    norm_x_true_inf = None
    if x_true is not None:
        x_true2 = x_true if x_true.ndim == 2 else x_true[:, None]
        norm_x_true = np.linalg.norm(x_true)
        norm_x_true_inf = _inf_norm(x_true2)

    history = []
    true_err_history = []
    x_history = []
    d_history = []
    r_history = []
    gmres_iters_history = []   # list (per outer iter) of lists (per rhs column)

    # At the working precision, as in solve_mixed_ir; u_f never enters the
    # stopping test.
    monitor = RefinementMonitor(unit_roundoff(HIGH_DTYPE), n,
                                max_iter=max_iter, ferr_tol=ferr_tol)

    t0 = time.perf_counter()
    solver = SOLVER_BUILDERS[solver_name](A, low_dtype, bs, b, inv_dtype)
    # GMRES applies the preconditioner one vector at a time. Where a solver
    # binds the right-hand-side shape at construction, a single-vector solve
    # needs a second factorization of the same matrix; it is built here, inside
    # the factorization timing, rather than on the first preconditioner
    # application, where its cost would land in the solve stage and factor_s
    # would understate the factorization by a whole factorization. Only
    # _CuDSSSolver defines the hook; every other solver has nothing to prepare.
    prepare = getattr(solver, "prepare_single_rhs", None)
    if callable(prepare):
        prepare()
    factor_s = time.perf_counter() - t0

    # Stage timers, as in solve_mixed_ir. Here the low-precision solves are
    # the preconditioner applications, one per inner GMRES iteration rather
    # than one per outer step, so solve_s and n_solves count every call GMRES
    # makes. What remains of inner_s after solve_s and residual_s is the rest
    # of GMRES: the products with A, the orthogonalization and the least
    # squares problem, all at the working precision.
    # gmres_s is the Krylov work alone: the products with A, the
    # orthogonalization and the least squares problem. It is accumulated as the
    # wall time of each _gmres_solve call MINUS the preconditioner time that
    # call spent inside it, which precond_apply has already counted into
    # solve_s. Measured this way rather than as inner_s - solve_s - residual_s,
    # which would also sweep in the forward-error diagnostics the refinement
    # loop computes between the timed regions; see mpperf's docstring on what
    # is deliberately not timed.
    stage = {"solve_s": 0.0, "n_solves": 0, "gmres_s": 0.0}
    residual_s = 0.0

    def precond_apply(v):
        t = time.perf_counter()
        out = solver.solve(v.astype(cast)).astype(HIGH_DTYPE)
        stage["solve_s"] += time.perf_counter() - t
        stage["n_solves"] += 1
        return out

    A_op = spla.LinearOperator((n, n), matvec=lambda v: A_high @ v, dtype=HIGH_DTYPE)
    M_op = spla.LinearOperator((n, n), matvec=precond_apply, dtype=HIGH_DTYPE)

    t_inner = time.perf_counter()
    t0 = time.perf_counter()
    x2 = solver.solve(b2.astype(cast)).astype(HIGH_DTYPE)   # x0: low-precision direct solve
    stage["solve_s"] += time.perf_counter() - t0
    stage["n_solves"] += 1

    # Every iterate recorded at the top of its pass, and x2_best returned
    # rather than the last iterate computed; see solve_mixed_ir for why.
    x2_best = x2
    for _ in range(max_iter + 1):
        t0 = time.perf_counter()
        r2 = b2 - A_high @ x2
        residual_s += time.perf_counter() - t0
        history.append(np.linalg.norm(r2) / norm_b)
        ferr_live = None
        if x_true2 is not None:
            true_err_history.append(np.linalg.norm(x2 - x_true2) / norm_x_true)
            if norm_x_true_inf:
                ferr_live = _inf_norm(x2 - x_true2) / norm_x_true_inf
        # No copy; see the note in solve_mixed_ir. x2 = x2 + d2 below rebinds
        # rather than writing in place, so the retained reference stays valid.
        x_history.append(x2)
        # Retained in complex128, before any cast; see solve_mixed_ir.
        r_history.append(r2)

        if monitor.check(ferr_live):
            break
        x2_best = x2
        if monitor.at_limit():
            break

        d2 = np.zeros_like(x2)
        iters_this_round = []
        for j in range(k):
            counter = [0]

            def _cb(_res, counter=counter):
                counter[0] += 1

            t_g, s_g = time.perf_counter(), stage["solve_s"]
            dj, info = _gmres_solve(A_op, r2[:, j], M_op, gmres_tol, gmres_restart,
                                    gmres_max_iter, callback=_cb)
            stage["gmres_s"] += ((time.perf_counter() - t_g)
                                 - (stage["solve_s"] - s_g))
            if info != 0:
                warnings.warn(
                    f"GMRES-IR: inner GMRES did not fully converge for rhs "
                    f"column {j} (info={info}); using its last iterate anyway."
                )
            d2[:, j] = dj
            iters_this_round.append(counter[0])
        gmres_iters_history.append(iters_this_round)
        d_history.append(d2)

        # A progress line per outer step, not just per-column warnings.
        # Without it a slow step -- inner GMRES stagnating near a large
        # kappa_inf can cost gmres_max_iter iterations per column, per outer
        # step -- produces nothing on the terminal beyond scipy's own
        # convergence warnings, which look identical whether the run is
        # working through max_iter steps or actually stuck. The whole run is
        # still bounded (at most max_iter outer steps, each at most
        # gmres_max_iter inner iterations per column), just possibly slow;
        # this line is what tells the two apart while it happens rather than
        # only in the "Convergence history" table printed after it returns.
        ferr_str = f"{ferr_live:.3e}" if ferr_live is not None else "n/a"
        print(f"    outer {len(gmres_iters_history):>3}: ferr={ferr_str}   "
              f"inner gmres iters (per rhs) = {iters_this_round}", flush=True)

        x2 = x2 + d2
        monitor.count()
    inner_s = time.perf_counter() - t_inner
    monitor.finish()
    x2 = x2_best

    x = x2 if orig_ndim == 2 else x2[:, 0]

    extra = {
        "history": history,
        "true_err_history": true_err_history,
        "x_history": x_history,
        "d_history": d_history,
        "r_history": r_history,
        "monitor": monitor,
        "gmres_iters_history": gmres_iters_history,
        "mem_bytes": solver.factor_nbytes(),
        "factor_s": factor_s,
        "inner_s": inner_s,
        "solve_s": stage["solve_s"],
        "residual_s": residual_s,
        "krylov_s": stage["gmres_s"],
        "n_solves": stage["n_solves"],
        "factor_breakdown": _factor_breakdown(solver),
    }
    if hasattr(solver, "free"):
        solver.free()
    return x, extra


def solve_direct(solver_name, A, b, bs, dtype, inv_dtype=DEFAULT_INV_DTYPE):
    """
    One factorization and one solve at `dtype`, with no refinement.

    Used for both reference variants: at complex128 it provides the accuracy
    ceiling, and at the low precision it provides the lower bound that
    refinement must improve upon. `dtype` may be the complex32 label, in which
    case the right-hand side is cast with cast_dtype and the solver does its
    own rounding.
    """
    t0 = time.perf_counter()
    solver = SOLVER_BUILDERS[solver_name](A, dtype, bs, b, inv_dtype)
    factor_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = solver.solve(np.asarray(b, dtype=cast_dtype(dtype))).astype(HIGH_DTYPE)
    inner_s = time.perf_counter() - t0

    mem_bytes = solver.factor_nbytes()
    factor_breakdown = _factor_breakdown(solver)
    if hasattr(solver, "free"):
        solver.free()
    # The stage split of a direct solve is trivial -- one solve call and no
    # residual -- but it is reported in the same keys as the refinement
    # variants so that a stage-breakdown figure can read every variant the
    # same way rather than special-casing the two unrefined ones.
    return x, {"mem_bytes": mem_bytes, "factor_s": factor_s, "inner_s": inner_s,
              "solve_s": inner_s, "residual_s": 0.0, "krylov_s": 0.0,
              "n_solves": 1, "factor_breakdown": factor_breakdown}


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────

def _warm_up_gpu(solver_name, A, b, bs, low_dtype, inv_dtype=DEFAULT_INV_DTYPE):
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
        warm = SOLVER_BUILDERS[solver_name](A, low_dtype, bs, b, inv_dtype)
        warm.solve(np.asarray(b, dtype=cast_dtype(low_dtype)))
        if hasattr(warm, "free"):
            warm.free()
    except Exception as e:
        print(f"    warm-up skipped ({type(e).__name__}: {e})")
    gc.collect()


def _matrix_nbytes(A, dtype):
    """
    Bytes A occupies when held at `dtype`: its values at that precision, plus
    the index arrays, whose size does not depend on the precision.

    Per-entry sizes come from ITEMSIZE rather than from a NumPy itemsize,
    because complex32 has no dtype to ask and because its 8 bytes are a
    property of the real embedding rather than of a storage format.
    """
    values = A.nnz * ITEMSIZE[dtype_label(dtype)]
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

        # Per-iteration backward errors, from the iterates the refinement loop
        # retained. Done here rather than in the loop so that neither the wall
        # timer above nor the loop's own inner_s is charged for them. The
        # iterates are kept rather than popped: refinement_metrics reads them
        # again, also outside the timed region.
        extra["etainf_history"], extra["omega_history"] = \
            _backward_error_histories(A_high, b_high,
                                      extra.get("x_history", []), normA)

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


def run_benchmarks(h5path, idx, solver_name, bs, low_dtype, max_iter, repeats,
                   reference_solver=None, inner="direct",
                   gmres_tol=1e-8, gmres_restart=30, gmres_max_iter=50,
                   inv_dtype=DEFAULT_INV_DTYPE, ferr_tol=None):
    A, b = load_system(h5path, idx)
    A_high = A.tocsc().astype(HIGH_DTYPE)
    b_high = np.asarray(b, dtype=HIGH_DTYPE)

    # Matrix-only, so computed once and reused for every variant's backward
    # error; see bench_all._matrix_norm and bench_all.backward_errors.
    normA = {p: _matrix_norm(A_high, p) for p in NORMWISE_ORDS}
    if normA[2] is None:
        print(f"  [warning] svds did not converge for ||A||_2; eta2 is "
              f"skipped for every variant at this index")

    low_name = dtype_label(low_dtype)
    u_work_pre = unit_roundoff(HIGH_DTYPE)
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

    if solver_name.startswith("block-thomas"):
        part = block_sizes_from_matrix(A)
        print(f"Blocks  : {len(part)} blocks detected from the sparsity "
              f"pattern, sizes {min(part)}..{max(part)}")

    if is_c32(low_dtype):
        # What complex32 means for this solver, since it is a label rather than
        # a dtype and the two implementations differ in where they leave half
        # precision. See the module docstring.
        note = (f"explicit block inverses formed at "
                f"{np.dtype(inv_dtype).name}, stored and applied at float16"
                if solver_name == "block-thomas-inv"
                else "LU with substitution, float16 throughout")
        print(f"        : complex32 = real embedding at block size 2m in "
              f"float16; {note}")

    if inner == "gmres":
        inner_label = "GMRES-IR"
        print(f"Inner   : GMRES(A) in complex128, preconditioned by {solver_name} "
              f"{low_name}   [gmres_tol={gmres_tol:.1e}  restart={gmres_restart}  "
              f"max_iter={gmres_max_iter}]")
    else:
        inner_label = "LU-IR"
        print(f"Inner   : single {low_name} triangular solve (classic LU-IR)")

    kappa = load_condition_numbers(h5path, idx)
    if kappa[2] is not None:
        print(f"Condition number at E_{idx}: kappa_2={kappa[2]:.3e}  "
              f"kappa_inf={kappa['inf']:.3e}")
        # The two Skeel numbers, where the condition-est file carries them.
        # cond(A) is the left half of Corollary 3.3's min and so changes
        # phi_cond; cond(A, x) does not enter the rate at all and is printed
        # because cond(A, x) * u is the accuracy refinement can end at.
        if kappa["skeel"] is not None:
            skeel_bits = [f"cond(A)={kappa['skeel']:.3e}"]
            if kappa["skeel_x"] is not None:
                skeel_bits.append(f"cond(A,x)={kappa['skeel_x']:.3e}  "
                                  f"[limiting ferr ~ {kappa['skeel_x'] * u_work_pre:.2e}]")
            print(f"  Skeel: {'  '.join(skeel_bits)}")
        # The classical LU-IR requirement is stated in the infinity norm (see
        # the module docstring): kappa_inf(A) u_f < 1. GMRES-IR is not bound
        # by it, so the check is informative there, not a prediction of
        # failure. u_f is the roundoff of the factorization precision itself,
        # not of the dtype vectors are cast to on the way in, which at
        # complex32 is complex128; see unit_roundoff and cast_dtype.
        u_f = unit_roundoff(low_dtype)
        bound = kappa["inf"] * u_f
        verdict = "within" if bound < 1 else "EXCEEDS"
        print(f"  LU-IR bound: kappa_inf * u_f = {bound:.2e}  "
              f"({verdict} the kappa_inf * u_f < 1 requirement)")
    else:
        cond_path = cli.CONDITION_DIR / f"{h5path.stem}.h5"
        print(f"Condition number: not available (no valid entry for idx={idx} "
              f"in {cond_path})")

    # The reference solver provides x_true and, kept alive afterwards, the
    # reference corrections phi_solve is measured against; see
    # refinement_metrics. Its own backward error is reported, since every
    # "forward error" here is relative to it and is only as meaningful as it is.
    x_true = None
    ref = None
    ref_eta = None
    # kappa_inf(A) * eps_ext, the accuracy the extended-precision reference
    # itself reaches; see _ExtendedReference. A forward error within an order of
    # magnitude of it is measuring the reference, not the method.
    ref_floor = float("nan")
    if reference_solver is not None:
        if reference_solver == EXTENDED_REFERENCE:
            print(f"Reference: x_true = superlu complex128 + "
                  f"{DEFAULT_REF_STEPS} refinement steps with a "
                  f"double-double residual (eps_ext={EPS_EXT:.1e})", flush=True)
        else:
            print(f"Reference: x_true = {reference_solver} complex128", flush=True)
        try:
            ref = build_reference(reference_solver, A, bs, b)
            x_true = ref.solve(b_high).astype(HIGH_DTYPE)
            ref_res = np.linalg.norm(A_high @ x_true - b_high) / np.linalg.norm(b_high)
            _, ref_etas, _ = backward_errors(A_high, x_true, b_high, normA)
            ref_eta = ref_etas[np.inf]
            print(f"           ||A@x_true - b|| / ||b|| for x_true itself: {ref_res:.2e}"
                  f"   nbe_inf: {ref_eta:.2e}")
            if reference_solver == EXTENDED_REFERENCE and kappa["inf"] is not None:
                ref_floor = kappa["inf"] * EPS_EXT
                print(f"           reference floor kappa_inf*eps_ext = "
                      f"{ref_floor:.2e}; a ferr_ref near it measures the "
                      f"reference, not the method")
        except (ImportError, TypeError, RuntimeError) as e:
            raise SystemExit(f"--reference-solver {reference_solver} failed: {e}")

    # fast_solve exists only on _ExtendedReference, where .solve() would
    # otherwise redo the whole extended-precision refinement loop for every
    # retained residual -- see its docstring. Every other reference solver's
    # own .solve() is already a single direct solve, so there is nothing to
    # skip there.
    _ref_solve_one = getattr(ref, "fast_solve", None) or (
        ref.solve if ref is not None else None)

    def ref_solve(r):
        """Reference correction for a retained residual; see refinement_metrics."""
        return _ref_solve_one(np.asarray(r, dtype=HIGH_DTYPE)).astype(HIGH_DTYPE)

    u_work = unit_roundoff(HIGH_DTYPE)
    # The convergence level, resolved here -- not left to RefinementMonitor's
    # own default -- so it can use cond(A,x) when this index's condition file
    # has it. sqrt(n) u is the level the WORKING PRECISION can represent, but
    # Corollary 3.3's actual limiting accuracy is of order cond(A,x) u, which
    # for an ill-conditioned system is orders of magnitude larger: calling a
    # run "not converged" for stopping at that floor, rather than at machine
    # precision it was never going to reach, mistakes the theorem's own limit
    # for a failure. An explicit --ferr-tol always wins; failing that, this
    # falls back to sqrt(n) u only where cond_skeel_x is unavailable (an older
    # condition-est file, or none at all).
    if ferr_tol is not None:
        ferr_level, ferr_level_note = float(ferr_tol), "(--ferr-tol)"
    elif kappa["skeel_x"] is not None:
        ferr_level = kappa["skeel_x"] * u_work
        ferr_level_note = "= cond(A,x) u"
    else:
        ferr_level = math.sqrt(A.shape[0]) * u_work
        ferr_level_note = ("= sqrt(n) u (cond(A,x) unavailable; run "
                           "condition_est.py --only-skeel)")
    if x_true is not None:
        print(f"Stopping: forward error against the reference increases  |  "
              f"or iter >= max_iter={max_iter}")
    else:
        print(f"Stopping: iter >= max_iter={max_iter} "
              f"(no --reference-solver, so there is no forward error to watch)")
    print(f"Converged: best ferr <= {ferr_level:.2e}  {ferr_level_note}")

    if repeats == 1:
        print("Runs    : 1 per variant\n")
    else:
        print(f"Runs    : {repeats} per variant (median reported)\n")

    # Consume the fixed device start-up cost before any variant is timed, so it
    # is not charged to whichever variant happens to run first.
    _warm_up_gpu(solver_name, A, b, bs, low_dtype, inv_dtype)

    # u_s, the effective precision of the correction solve: u_f for LU-IR,
    # whose correction is whatever the low-precision factors return, and u for
    # GMRES-IR, whose inner solve runs at the working precision. It sets the
    # scale of the whole convergence factor; see refinement_metrics.
    if inner == "gmres":
        u_s = unit_roundoff(HIGH_DTYPE)
        ir_fn = lambda: solve_gmres_ir(solver_name, A, b, bs, low_dtype, max_iter,
                                       x_true=x_true, gmres_tol=gmres_tol,
                                       gmres_restart=gmres_restart,
                                       gmres_max_iter=gmres_max_iter,
                                       normA=normA, inv_dtype=inv_dtype,
                                       ferr_tol=ferr_level)
    else:
        u_s = unit_roundoff(low_dtype)
        ir_fn = lambda: solve_mixed_ir(solver_name, A, b, bs, low_dtype, max_iter,
                                       x_true=x_true, normA=normA,
                                       inv_dtype=inv_dtype, ferr_tol=ferr_level)

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
         lambda: solve_direct(solver_name, A, b, bs, HIGH_DTYPE, inv_dtype),
         {"matrix_dtype": HIGH_DTYPE, "krylov": False}),
        (f"{solver_name} {low_name} (no refine)",
         lambda: solve_direct(solver_name, A, b, bs, low_dtype, inv_dtype),
         {"matrix_dtype": low_dtype, "krylov": False}),
    ]
    variant_info = {name: info for name, _fn, info in variants}

    # failures records which variants did not run and why, keyed by the exact
    # name in `variants`. The refined variant's entry, if any, is what turns
    # into a synthetic run row below rather than a silently dropped index.
    all_records, failures = {}, {}
    for name, fn, _info in variants:
        print(f"  Benchmarking '{name}' x{repeats} ...", flush=True)
        try:
            all_records[name] = benchmark_solver(fn, A_high, b_high, repeats,
                                                 x_true=x_true, normA=normA)
        except (ImportError, TypeError, RuntimeError) as e:
            print(f"    skipped: {e}")
            failures[name] = e
        except FACTORIZATION_FAILURES as e:
            # The low-precision factorization itself could not be formed; see
            # FACTORIZATION_FAILURES. Caught per variant, not left to escape
            # this function, specifically so that when it is the REFINED
            # variant that fails, the other two (typically just the complex128
            # baseline; "no refine" shares the same factorization and usually
            # fails identically) can still be benchmarked and reported.
            print(f"    skipped (factorization failed): {e}")
            failures[name] = e
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

    def stage_other(name):
        """
        inner_s minus the solve and residual stages: the working-precision
        vector arithmetic of the update and the stopping monitor, plus, for
        GMRES-IR, the products with A and the orthogonalization.
        """
        total = med_extra(name, "inner_s")
        if total is None:
            return float("nan")
        return float(total - (med_extra(name, "solve_s") or 0.0)
                     - (med_extra(name, "residual_s") or 0.0))

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
    print()

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

    # Analysis (ordering and symbolic factorization) against numerical
    # factorization -- a breakdown of the row directly above. Populated for
    # cuDSS and MUMPS, which time their own phases, and for the two Block
    # Thomas families, whose block detection and extraction is the
    # corresponding structural phase. SuperLU (scipy's splu) and UMFPACK fuse
    # both into one call and expose no split. Only the numerical phase
    # performs floating-point arithmetic, so only it becomes cheaper at a
    # lower u_f; see _factor_breakdown.
    def med_breakdown(name):
        """
        Median over repeats of the factorization phase split, reduced the same
        way factor_s is so that the parts and the whole refer to the same run.
        """
        pairs = [r["extra"]["factor_breakdown"] for r in all_records[name]
                 if r["extra"].get("factor_breakdown") is not None]
        if not pairs:
            return None
        return (float(np.median([p[0] for p in pairs])),
                float(np.median([p[1] for p in pairs])))

    breakdowns = {nm: med_breakdown(nm) for nm in names}
    if any(bd is not None for bd in breakdowns.values()):
        for label, i in [("    analysis (ms)", 0), ("    numeric (ms)", 1)]:
            vals = [f"{breakdowns[nm][i]*1e3:.1f}" if breakdowns[nm] is not None
                    else "n/a (fused)" for nm in names]
            print(f"  {label:<{col-2}}" + "".join(f"  {v:<28}" for v in vals))

    solve_vals = [f"{med_extra(nm, 'inner_s')*1e3:.1f}" if med_extra(nm, "inner_s") is not None
                  else "n/a" for nm in names]
    print(f"  {'Solve / inner-loop time (ms)':<{col-2}}" +
          "".join(f"  {v:<28}" for v in solve_vals))

    # The row above split into its stages. For a direct variant this is one
    # solve and nothing else; for a refinement variant it separates the
    # low-precision solves -- the only stage that becomes cheaper when u_f is
    # lowered -- from the working-precision residuals and from the remaining
    # vector work. The solve count is reported with them because the stage
    # cost is that count times the cost of one solve, and the two variants
    # differ mainly in the count: LU-IR performs one solve per outer step,
    # GMRES-IR one per inner GMRES iteration.
    for label, getter in [
            ("    low-precision solves (ms)",
             lambda nm: med_extra(nm, "solve_s")),
            ("    residual b - Ax (ms)",
             lambda nm: med_extra(nm, "residual_s")),
            ("    other (update, Krylov) (ms)", stage_other)]:
        vals = []
        for nm in names:
            v = getter(nm)
            vals.append(f"{v*1e3:.1f}" if v is not None and np.isfinite(v)
                        else "n/a")
        print(f"  {label:<{col-2}}" + "".join(f"  {v:<28}" for v in vals))
    count_vals = [f"{int(med_extra(nm, 'n_solves'))}"
                  if med_extra(nm, "n_solves") is not None else "n/a"
                  for nm in names]
    print(f"  {'    solve calls':<{col-2}}" +
          "".join(f"  {v:<28}" for v in count_vals))
    print()

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

    # The complex128 direct variant is the factorization refinement is meant to
    # replace, so it is the denominator that makes each saving readable. The
    # ratio is printed beside its own value rather than on a row of its own, so
    # that one column holds everything about one variant.
    mem_ref = next((nm for nm in names if "complex128 (direct)" in nm), None)

    def mem_row(label, size):
        vals = []
        for nm in names:
            # A factorization never occupies zero bytes, so zero means the
            # backend exposed no size rather than a measurement. Printing
            # "not reported" keeps that distinct from a small factorization.
            if not size[nm]:
                vals.append("not reported")
                continue
            cell = f"{size[nm]/MIB:.2f}"
            if mem_ref is not None and size.get(mem_ref):
                cell += f" ({size[nm]/size[mem_ref]:.2f}x)"
            vals.append(cell)
        print(f"  {label:<{col-2}}" + "".join(f"  {v:<28}" for v in vals))

    mem_row("Factor memory (MiB, factor_nbytes)", mem)
    mem_row("Working set (MiB, factor + A + basis)", working)
    if any(not mem[nm] for nm in names):
        print("    [note] the working set of a variant whose factor memory is "
              "not reported\n           counts only A and the Krylov basis "
              "and so understates the total.")

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

    # ---- the refinement variant's convergence, and the metrics behind it ----
    #
    # Everything below is reconstructed from the arrays the loop retained, in
    # the same pass that will write them out; see refinement_metrics.
    # names[0] would silently be wrong here whenever the refined variant is
    # the one that failed: all_records then starts with whichever variant DID
    # succeed (typically the complex128 baseline), and every convergence
    # quantity below would be read off the wrong run instead of raising.
    # ir_name_requested is always the refined variant's own name, by
    # construction of `variants` above, regardless of what ran.
    ir_name_requested = variants[0][0]
    ir_name = ir_name_requested if ir_name_requested in all_records else None
    if ir_name is not None:
        ir_extra = all_records[ir_name][0]["extra"]
        metrics = refinement_metrics(A_high, x_true, ir_extra, u_s, kappa["inf"],
                                     ref_solve=ref_solve if ref is not None else None,
                                     cond_skeel=kappa["skeel"])
        monitor = ir_extra.get("monitor")
        etainf_history = ir_extra.get("etainf_history", [])
        omega_history = ir_extra.get("omega_history", [])
    else:
        ir_extra, metrics, monitor = None, [], None
        etainf_history, omega_history = [], []
        print(f"\n  Refinement variant failed to factorize -- recorded below "
              f"as converged=False, outer_iters=0 rather than dropped "
              f"({failures[ir_name_requested]})")

    if monitor is not None:
        summary = monitor.summary()
        print(f"\n  Outer iterations: {summary['outer_iters']}   "
              f"(stopped: {summary['stop_reason']})")
        if summary["ferr_best"] is not None:
            verdict = "CONVERGED" if summary["converged"] else "did NOT converge"
            print(f"    best ferr = {summary['ferr_best']:.3e}  vs  "
                  f"ferr_tol = {summary['ferr_tol']:.3e}   -> {verdict}")

    if metrics:
        print(f"\n  Convergence history (first run, {ir_name}):")
        for i, m in enumerate(metrics):
            line = f"    iter {i}:"
            if m.get("relres") is not None:
                line += f" ||r||/||b|| = {m['relres']:.3e}"
            if m.get("ferr_ref") is not None:
                line += f"   ferr_ref = {m['ferr_ref']:.3e}"
            if i < len(etainf_history):
                line += f"   nbe_inf = {etainf_history[i]:.3e}"
            if i < len(omega_history):
                line += f"   cbe = {omega_history[i]:.3e}"
            if m.get("gmres_inner_iterations") is not None:
                line += f"   gmres = {m['gmres_inner_iterations']}"
            print(line)

        # The convergence-factor panel: the right-hand plots of Carson and
        # Higham's numerical experiments, as a table.
        print(f"\n  Convergence factor (u_s = {u_s:.2e}, "
              f"{'u (GMRES-IR)' if inner == 'gmres' else 'u_f (LU-IR)'}):")
        head = (f"    {'iter':<6}{'rho (obs)':>12}{'mu_hat':>12}"
                f"{'phi_cond':>12}{'binds':>10}{'phi_solve':>12}{'phi_hat':>12}")
        print(head)
        print("    " + "─" * (len(head) - 4))
        for i, m in enumerate(metrics):
            def cell(key):
                v = m.get(key)
                return f"{v:>12.3e}" if v is not None else f"{'n/a':>12}"
            # Which half of min(cond(A), kappa_inf*mu_hat) the Corollary's
            # min selected this step. 'cond' means mu_hat was large enough that
            # the direction of the error bought nothing; 'kappa_mu' means it
            # did. See refinement_metrics.
            binding = m.get("phi_cond_binding") or "n/a"
            print(f"    {i:<6}" + cell("rho") + cell("mu_hat")
                  + cell("phi_cond_hat") + f"{binding:>10}"
                  + cell("phi_solve_hat") + cell("phi_hat"))
        if kappa["inf"] is None and kappa["skeel"] is None:
            print("    phi_cond is n/a: neither kappa_inf nor cond(A) was "
                  "available for this index; run "
                  "condition-est/condition_est.py first")
        elif kappa["skeel"] is None:
            print("    phi_cond is the kappa_inf*mu_hat half of the "
                  "Corollary's min alone: this condition-est file has no "
                  "cond_skeel column; add it with condition_est.py "
                  "--only-skeel")
        if ref is None:
            print("    phi_solve is n/a: no --reference-solver, so no reference "
                  "correction to compare against")

    probe_idx = 100
    if A.shape[0] > probe_idx:
        print(f"\n  x[{probe_idx}] comparison across variants (first run):")
        for nm in names:
            xv = all_records[nm][0]["x"]
            val = xv[probe_idx] if xv.ndim == 1 else xv[probe_idx, 0]
            print(f"    {nm:<35} x[{probe_idx}] = {val!r}")

    nnz = int(A.nnz)
    n = A.shape[0]
    idx_mib = (nnz + n + 1) * 4 / 1024**2
    ref_bytes = ITEMSIZE["complex128"]
    print(f"\n  Theoretical matrix value storage ({nnz} nonzeros, indices excluded):")
    for name in ("complex128", "complex64", C32):
        mib = nnz * ITEMSIZE[name] / 1024**2
        saving = 1.0 - ITEMSIZE[name] / ref_bytes
        tag = "" if not saving else f"  ({saving:.0%} saving on values)"
        print(f"    {name:<11}: {mib:.3f} MiB{tag}")
    # complex32 matches complex64 above rather than halving it again: the real
    # embedding stores each real component of an entry twice. See ITEMSIZE.
    print(f"    Index arrays (shared): ~{idx_mib:.3f} MiB")

    if ref is not None and hasattr(ref, "free"):
        ref.free()

    # ---- the two result tables --------------------------------------------
    #
    # One run row per variant, and one iteration row per outer step of the
    # refinement variant. Both are keyed by (idx, variant) so that a sweep
    # concatenates cleanly and either table can be read on its own.
    u_f = unit_roundoff(low_dtype)
    u_work = unit_roundoff(HIGH_DTYPE)
    n_blocks = len(block_sizes_from_matrix(A)) \
        if solver_name.startswith("block-thomas") else -1
    common = dict(idx=int(idx), n=int(n), nnz=nnz, n_rhs=int(n_rhs),
                  n_blocks=int(n_blocks),
                  energy=float(energy) if energy is not None else float("nan"),
                  solver=solver_name, factor_dtype=low_name, inner=inner)
    run_rows = []
    for nm in names:
        recs = all_records[nm]
        extra0 = recs[0]["extra"]
        mon = extra0.get("monitor")
        summary = mon.summary() if mon is not None else {}
        gm = extra0.get("gmres_iters_history", [])
        breakdown = med_breakdown(nm)
        run_rows.append(dict(
            common, variant=nm,
            is_refined=int(mon is not None),
            u_f=float(u_f), u=u_work,
            u_s=float(u_s) if mon is not None else float("nan"),
            kappa_2=kappa[2] if kappa[2] is not None else float("nan"),
            kappa_inf=kappa["inf"] if kappa["inf"] is not None else float("nan"),
            # cond(A) enters phi_cond as the left half of the Corollary's min;
            # cond(A, x) is not part of the rate at all, and is carried so that
            # a figure can draw the limiting accuracy cond(A, x) u beside the
            # forward error. See load_condition_numbers and refinement_metrics.
            cond_skeel=kappa["skeel"] if kappa["skeel"] is not None else float("nan"),
            cond_skeel_x=kappa["skeel_x"] if kappa["skeel_x"] is not None
                         else float("nan"),
            lu_ir_bound=(kappa["inf"] * u_f) if kappa["inf"] is not None
                        else float("nan"),
            relres=med(nm, "residual"),
            ferr_ref=med_opt(nm, "true_err") if have_x_true else float("nan"),
            eta1=med_opt(nm, "eta1") if med_opt(nm, "eta1") is not None else float("nan"),
            eta2=med_opt(nm, "eta2") if med_opt(nm, "eta2") is not None else float("nan"),
            etainf=med_opt(nm, "etainf") if med_opt(nm, "etainf") is not None else float("nan"),
            omega=med_opt(nm, "omega") if med_opt(nm, "omega") is not None else float("nan"),
            outer_iters=int(summary.get("outer_iters", 0)),
            converged=int(bool(summary.get("converged", False))),
            # The accuracy the returned solution reached and the level it had
            # to reach to count as converged; see RefinementMonitor.
            ferr_best=float(summary["ferr_best"]) if summary.get("ferr_best") is not None
                      else float("nan"),
            ferr_tol=float(summary.get("ferr_tol", float("nan"))),
            stop_reason=summary.get("stop_reason", "no refinement"),
            gmres_total=int(sum(sum(g) for g in gm)) if gm else -1,
            # The mean inner-GMRES iteration count of a single correction,
            # over every (outer step, rhs column) pair -- not gmres_total
            # divided by outer_iters * n_rhs, which would be wrong whenever
            # the run's last correction was discarded (see solve_gmres_ir):
            # gm holds one entry per correction actually computed, including
            # that discarded one, so this is the exact mean of what was
            # computed rather than an approximation from stored aggregates.
            # This is what the summary figure labels each GMRES-IR point
            # with, since the y position already shows outer_iters and
            # repeating it would say nothing new.
            gmres_avg=(float(np.mean([it for step in gm for it in step]))
                      if gm else float("nan")),
            wall_s=med(nm, "wall_s"),
            factor_s=med_extra(nm, "factor_s") or float("nan"),
            factor_symbolic_s=float(breakdown[0]) if breakdown else float("nan"),
            factor_numeric_s=float(breakdown[1]) if breakdown else float("nan"),
            inner_s=med_extra(nm, "inner_s") or float("nan"),
            solve_s=med_extra(nm, "solve_s") or float("nan"),
            residual_s=med_extra(nm, "residual_s") or 0.0,
            # Both sub-timers run strictly inside the inner_s window, so the
            # remainder is non-negative: the update, the stopping monitor and,
            # for GMRES-IR, the Krylov work.
            other_s=stage_other(nm),
            n_solves=int(med_extra(nm, "n_solves") or 0),
            factor_mb=mem[nm] / MIB,
            # factor_nbytes returns 0 rather than raising where a backend does
            # not expose the size of its factors: MUMPS when INFOG(3) is
            # unreachable through this python-mumps build, cuDSS when the
            # factorization info carries no lu_nnz. Zero bytes of factors is
            # not a possible measurement, so the flag distinguishes "not
            # reported" from a real value and a memory figure can leave the
            # entry out instead of drawing a bar at zero.
            factor_mb_reported=int(mem[nm] > 0),
            working_mb=working[nm] / MIB,
            reference_solver=reference_solver or "",
            reference_nbe=float(ref_eta) if ref_eta is not None else float("nan"),
            reference_floor=ref_floor,
        ))

    if ir_name is None:
        # The refined variant produced no iterate at all, so its accuracy,
        # timing and memory fields are genuinely unmeasured rather than small
        # -- nan/-1/0, matching this table's existing sentinel convention
        # (see factor_mb_reported above) -- while everything knowable before
        # any variant ran (dimensions, conditioning, precisions, the
        # convergence level) is filled in. That is enough for a figure to
        # place this index on its kappa_inf axis and colour it red: dropping
        # the row entirely, as this module did previously, made a
        # kappa_inf-vs-iterations sweep stop early exactly at the hardest
        # indices, which is the one place its trend matters most.
        run_rows.append(dict(
            common, variant=ir_name_requested,
            is_refined=1,
            u_f=float(u_f), u=u_work, u_s=float(u_s),
            kappa_2=kappa[2] if kappa[2] is not None else float("nan"),
            kappa_inf=kappa["inf"] if kappa["inf"] is not None else float("nan"),
            cond_skeel=kappa["skeel"] if kappa["skeel"] is not None else float("nan"),
            cond_skeel_x=kappa["skeel_x"] if kappa["skeel_x"] is not None
                         else float("nan"),
            lu_ir_bound=(kappa["inf"] * u_f) if kappa["inf"] is not None
                        else float("nan"),
            relres=float("nan"), ferr_ref=float("nan"),
            eta1=float("nan"), eta2=float("nan"), etainf=float("nan"),
            omega=float("nan"),
            outer_iters=0, converged=0,
            ferr_best=float("nan"), ferr_tol=float(ferr_level),
            stop_reason=(f"factorization failed: "
                        f"{type(failures[ir_name_requested]).__name__}: "
                        f"{failures[ir_name_requested]}"),
            gmres_total=-1, gmres_avg=float("nan"),
            wall_s=float("nan"), factor_s=float("nan"),
            factor_symbolic_s=float("nan"), factor_numeric_s=float("nan"),
            inner_s=float("nan"), solve_s=float("nan"), residual_s=float("nan"),
            other_s=float("nan"), n_solves=0,
            factor_mb=0.0, factor_mb_reported=0, working_mb=0.0,
            reference_solver=reference_solver or "",
            reference_nbe=float(ref_eta) if ref_eta is not None else float("nan"),
            reference_floor=ref_floor,
        ))

    iter_rows = []
    for m in metrics:
        row = dict(common, variant=ir_name)
        for key in ITERATION_COLUMNS:
            if key in row:
                continue
            value = m.get(key)
            row[key] = (float("nan") if value is None and key not in
                        ("phi_cond_binding", "phi_cond_form", "note",
                         "gmres_inner_iterations", "gmres_inner_max",
                         "outer_iteration")
                        else value)
        row.setdefault("phi_cond_binding", "unavailable")
        row.setdefault("phi_cond_form", "unavailable")
        row.setdefault("note", "")
        for key in ("gmres_inner_iterations", "gmres_inner_max"):
            if row.get(key) is None:
                row[key] = -1
        iter_rows.append(row)

    return all_records, run_rows, iter_rows


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
    cli.add_factor_dtype(
        ap, choices=cli.COMPLEX_DTYPES, default="complex64",
        help="precision of the low-precision factorization, u_f "
             "(default: complex64). complex32 is the embedded-real float16 "
             "factorization and is available for block-thomas and "
             "block-thomas-inv only")
    cli.add_inv_dtype(ap, default=np.dtype(DEFAULT_INV_DTYPE).name)
    ap.add_argument("--inner", choices=["direct", "gmres"], default="direct",
                    help="inner correction solve: 'direct' is classic LU-IR, "
                         "a single low-precision triangular solve; 'gmres' is "
                         "GMRES-IR, GMRES in complex128 preconditioned by the "
                         "low-precision factorization")
    # The stopping rule -- the forward error increased -- takes no option: an
    # increase is an increase, and needs no constant. Only the safety net and
    # the convergence level are settable.
    ap.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                    metavar="N",
                    help="safety net on the outer refinement steps. A run with "
                         "a reference solution normally stops well before it, "
                         "when the forward error increases; without one this "
                         f"is the only thing that ends the loop "
                         f"(default: {DEFAULT_MAX_ITER})")
    ap.add_argument("--ferr-tol", type=float, default=None, metavar="TOL",
                    help="accuracy the returned solution must reach to count "
                         "as converged. Default is cond(A,x) u, Corollary "
                         "3.3's own limiting accuracy, read from the "
                         "condition-est file's cond_skeel_x column; falls "
                         "back to sqrt(n) u, the level the working precision "
                         "can hold in the abstract, where cond(A,x) is "
                         "unavailable. Judging convergence against sqrt(n) u "
                         "on an ill-conditioned system mistakes the "
                         "theorem's own limit -- often orders of magnitude "
                         "larger -- for a failure")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats per variant; the median is reported")
    ap.add_argument("--reference-solver",
                    choices=["superlu", "mumps", "cudss", EXTENDED_REFERENCE],
                    default=EXTENDED_REFERENCE, metavar="NAME",
                    help="compute x_true with this solver, and the reference "
                         "corrections phi_solve is measured against. Defaults "
                         f"to '{EXTENDED_REFERENCE}', which refines a "
                         "complex128 solve with a double-double residual, about "
                         "2e3 times more accurate than plain complex128 -- the "
                         "'true' solution every run should be measured "
                         "against. The other three named solvers run at plain "
                         "complex128 and so carry a forward error of order "
                         "cond(A,x)*u, the same order as the limiting accuracy "
                         "of refinement itself, which makes them unusable as "
                         "a ruler for a convergence study; pass one only to "
                         "check a specific solver's own reported solution, "
                         "not to judge convergence")
    ap.add_argument("--gmres-tol", type=float, default=1e-8,
                    help="relative tolerance of the inner GMRES solve "
                         "(--inner gmres only)")
    ap.add_argument("--gmres-restart", type=int, default=30,
                    help="inner GMRES restart parameter (--inner gmres only)")
    ap.add_argument("--gmres-max-iter", type=int, default=50,
                    help="maximum inner GMRES iterations per outer step "
                         "(--inner gmres only)")
    cli.add_output(ap, outdir_default=str(cli.MIXED_PREC_DIR),
                   outdir_help=f"directory holding the analysis file "
                               f"<material>.h5, to which each run appends one "
                               f"numbered experiment "
                               f"(default: {cli.MIXED_PREC_DIR})")
    ap.add_argument("--no-save", action="store_true",
                    help="print the report but append no experiment")
    ap.add_argument("--list-experiments", action="store_true",
                    help="list the experiments already in the analysis file "
                         "and exit, without running anything")
    args = ap.parse_args()

    if args.list_experiments:
        material = args.material or Path(args.h5path).stem
        out_path = analysis_path(args.outdir, material)
        names = experiment_names(out_path)
        if not names:
            print(f"{out_path} holds no experiments")
            return
        print(f"{out_path}")
        for nm in names:
            _, attrs, runs, iters = load_experiment(out_path, nm)
            idxs = attrs.get("indices", [])
            print(f"  {nm}  {attrs.get('timestamp', '?'):<26} "
                  f"{attrs.get('solver', '?')} {attrs.get('factor_dtype', '?')} "
                  f"{attrs.get('inner_label', '?'):<9} "
                  f"{len(idxs)} idx  {len(runs.get('idx', []))} runs  "
                  f"{len(iters.get('idx', []))} iters")
        return

    h5path = Path(args.h5path)
    if args.energy is not None:
        if args.idx is not None or args.start is not None:
            ap.error("--energy is mutually exclusive with --idx/--start/--end")
        file_indices, file_energies, _, _ = load_energy_metadata(h5path)
        indices = [idx_of_energy(file_indices, file_energies, e)
                  for e in args.energy]
    else:
        indices = cli.resolve_indices(ap, args)

    # Rejected here rather than by whichever library was asked for it, so that
    # an unsupported pairing costs nothing; see solver_dtypes.
    accepted = solver_dtypes(args.solver)
    if args.factor_dtype not in accepted:
        ap.error(f"--solver {args.solver} has no {args.factor_dtype} "
                 f"factorization; it accepts {', '.join(accepted)}")

    # The complex32 label travels through the drivers as itself; every other
    # precision becomes a NumPy dtype here. See the Precisions section of the
    # module docstring.
    factor_dtype = (C32 if args.factor_dtype == C32
                    else np.dtype(args.factor_dtype))
    inv_dtype = np.dtype(args.inv_dtype)

    # skipped holds every index that produced no rows, for the experiment
    # attributes; hard_skipped is the subset whose low-precision factorization
    # could not be formed. The two are reported differently because they mean
    # opposite things: a factorization failure is a property of the matrix at
    # this u_f and belongs in any conclusion about the sweep, whereas an index
    # skipped because a solver library is missing says nothing about the
    # matrix at all.
    all_runs, all_iters, skipped, hard_skipped = [], [], [], []
    for idx in indices:
        if len(indices) > 1:
            print("=" * 78)
        try:
            _records, run_rows, iter_rows = run_benchmarks(
                h5path, idx, args.solver, None,
                factor_dtype, args.max_iter, args.repeats,
                reference_solver=args.reference_solver,
                inner=args.inner, gmres_tol=args.gmres_tol,
                gmres_restart=args.gmres_restart,
                gmres_max_iter=args.gmres_max_iter,
                inv_dtype=inv_dtype, ferr_tol=args.ferr_tol)
            all_runs.extend(run_rows)
            all_iters.extend(iter_rows)
        except SystemExit as e:              # a bad index, from load_system
            skipped.append((idx, str(e)))
            print(f"\nE_{idx}: skipped ({e})")
        except FACTORIZATION_FAILURES as e:
            # The low-precision factorization could not be formed at all. At
            # complex32 this is expected rather than exceptional: half
            # precision overflows at 65504, and the Schur recursion of Block
            # Thomas squares the block norms, so a large enough kappa_inf
            # drives a block to inf and then to nan. Both variants that use
            # that factorization are then impossible, leaving only the
            # complex128 baseline, which answers nothing on its own -- so the
            # whole index is skipped rather than half-recorded.
            #
            # Note that these are the HARDEST indices, so a sweep that drops
            # them silently is biased at exactly the end it is about. They are
            # recorded in the experiment's skipped_idx / skipped_reason
            # attributes and counted in the summary below; a figure that plots
            # iterations against kappa_inf should say how many there were.
            reason = f"{type(e).__name__}: {e}"
            skipped.append((idx, reason))
            hard_skipped.append(idx)
            print(f"\nE_{idx}: skipped -- the {dtype_label(factor_dtype)} "
                  f"factorization could not be formed ({reason})")

    # Said once at the end, because in a long sweep the per-index lines have
    # scrolled away and a silently shorter result set is the thing most likely
    # to be misread.
    if skipped:
        def _names(values):
            head = ", ".join(str(i) for i in values[:12])
            return head + ("" if len(values) <= 12
                           else f", ... (+{len(values) - 12} more)")

        print(f"\n{len(skipped)} of {len(indices)} indices skipped: "
              f"{_names([i for i, _ in skipped])}")
        if hard_skipped:
            print(f"  {len(hard_skipped)} of them because the "
                  f"{dtype_label(factor_dtype)} factorization overflowed: "
                  f"{_names(hard_skipped)}")
            print(f"  Those are the hardest indices in the sweep -- the "
                  f"precision failing outright, not converging slowly -- so a "
                  f"figure drawn from the rest understates the difficulty at "
                  f"large kappa_inf. The full list is in the experiment's "
                  f"skipped_idx attribute.")

    if args.no_save or not all_runs:
        return

    material = args.material or h5path.stem
    out_path = analysis_path(args.outdir, material)
    attrs = dict(
        material=material,
        source=str(h5path),
        timestamp=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        solver=args.solver,
        factor_dtype=args.factor_dtype,
        inv_dtype=args.inv_dtype,
        inner=args.inner,
        inner_label="GMRES-IR" if args.inner == "gmres" else "LU-IR",
        working_dtype=np.dtype(HIGH_DTYPE).name,
        residual_dtype=np.dtype(HIGH_DTYPE).name,
        # u itself, so that a figure can draw the level refinement aims at
        # without having to know what complex128 rounds to.
        working_u=unit_roundoff(HIGH_DTYPE),
        max_iter=int(args.max_iter),
        # -1 means "not set, so sqrt(n) u per index"; the per-index value used
        # is in the runs table's ferr_tol column.
        ferr_tol=float(args.ferr_tol) if args.ferr_tol is not None else -1.0,
        gmres_tol=float(args.gmres_tol),
        gmres_restart=int(args.gmres_restart),
        gmres_max_iter=int(args.gmres_max_iter),
        reference_solver=args.reference_solver,
        repeats=int(args.repeats),
        indices=np.asarray(indices, dtype=np.int64),
        n_requested=len(indices),
        n_skipped=len(skipped),
        criteria="stop when the forward error against the reference increases",
        convergence_factor="Carson and Higham 2018, Corollary 3.3",
        **material_metadata(h5path),
    )
    if skipped:
        attrs["skipped_idx"] = np.asarray([i for i, _ in skipped], dtype=np.int64)
        attrs["skipped_reason"] = [message for _, message in skipped]
    name = save_experiment(out_path, attrs, all_runs, all_iters)
    print(f"\nwrote {out_path}:/{EXPERIMENTS_GROUP}/{name}")
    print(f"  runs        {len(all_runs)} rows (one per index and variant)")
    print(f"  iterations  {len(all_iters)} rows (one per index and outer step)")
    print(f"  plot with: python ../plotting/mixed_prec_ir/plot_mpir.py "
          f"{out_path} --experiment {int(name)}")


if __name__ == "__main__":
    main()
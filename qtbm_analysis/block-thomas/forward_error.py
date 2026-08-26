#!/usr/bin/env python3
"""
Forward error of every stored solution, against the bounds that predict it.

Post-hoc analysis. Reads the solutions run_bench/run_benchmarks.py wrote into a
material HDF5 file and the condition numbers condition-est/condition_est.py
wrote into its own file; solves nothing that the benchmark already solved, and
modifies neither file.

The question
------------
A backward-stable solver returns the exact solution of a nearby problem, which
says nothing on its own about how close x is to the truth. The two classical
bounds close that gap:

    ||xhat - x||_inf / ||x||_inf  <~  kappa_inf(A) * eta_inf     normwise
    ||xhat - x||_inf / ||x||_inf  <~  cond(A, x)   * omega       componentwise

eta_inf and omega are measured per solve by bench_all.backward_errors and are
already in the material file; kappa_inf and cond(A, x) come from the condition
file. What is missing is the left-hand side, and that is what this script
supplies. The stored vs_base column cannot serve: its reference is the SuperLU
complex128 solution, which carries an error of the same order as the solutions
being judged, so at complex128 it compares two errors of equal size.

The reference solution
----------------------
x is produced by iterative refinement of the SuperLU complex128 solution, with
the residual accumulated in np.clongdouble (80-bit on x86-64, eps = 1.1e-19)
and the correction solved with the same double-precision factorization. SciPy
has no sparse type in that precision, so the product A xhat is formed directly
from the CSR arrays: the elementwise products are taken in extended precision
and summed per row with np.add.reduceat.

Refinement in a residual precision u_r converges to a relative error of order
kappa * u_r, so the reference is better than the double-precision solutions it
judges by roughly eps_double / eps_ext = 2e3, not by the u^2 a true
double-double residual would give. That limit is not hidden: `ref_floor`
= kappa_inf * eps_ext is recorded per index, and any measured forward error
within an order of magnitude of it is a measurement of the reference rather
than of the solver. For complex64 and complex32 solutions the separation is
ample; at complex128 it is about three orders of magnitude for a moderately
conditioned index and vanishes once kappa_inf approaches 1/eps_ext.

Output
------
    <outdir>/<material>.h5, group forward_error, one row per
    (index, solver, dtype):

    fwd_inf, fwd_2       ||xhat - x|| / ||x||, worst right-hand side column
    eta_inf, eta_1, omega    backward errors, as recorded by the benchmark
    cond_inf, cond_skeel, cond_skeel_x   from the condition file, NaN if absent
    bound_nw             cond_inf * eta_inf
    bound_cw             cond_skeel_x * omega
    ratio_nw, ratio_cw   fwd_inf / bound, the quantity the chapter plots:
                         at most 1 if the bound holds, and small when the
                         bound is pessimistic
    ref_res, ref_floor, ref_steps        quality of the reference itself

Usage
-----
With no index selection every index the file holds is analysed. Each one costs
one sparse factorization and a few refinement steps, so a full-resolution sweep
of a few thousand indices is an overnight job; --stride thins it.

    python forward_error.py /scratch/yimili/matrices2/hdf5/carbon-nanotube.h5
    python forward_error.py .../carbon-chain.h5 --stride 10
    python forward_error.py .../carbon-nanotube.h5 --idx 25 --no-save
    python ../plotting/block-thomas/plot_forward_error.py \
        /scratch/yimili/error-analysis-block-thomas/carbon-nanotube.h5
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

import cli
from factor_io import save_table, material_metadata

GROUP = "forward_error"
CONDITION_GROUP = "condition"
DEFAULT_OUTDIR = cli.BLOCK_THOMAS_DIR

COLUMNS = ["idx", "solver", "dtype", "n", "nrhs",
           "fwd_inf", "fwd_2", "eta_inf", "eta_1", "omega",
           "cond_inf", "cond_skeel", "cond_skeel_x",
           "bound_nw", "bound_cw", "ratio_nw", "ratio_cw",
           "ref_res", "ref_floor", "ref_steps"]

# Every solver that stores a solution, which is all of them: the forward error
# needs x alone and, unlike growth_factor.py, does not need the factors. MUMPS
# and cuDSS are therefore comparable here even though they expose no L and U.
DEFAULT_SOLVERS = cli.ALL_SOLVERS
EXT = np.clongdouble
EPS_EXT = float(np.finfo(np.longdouble).eps)


# ---------------------------------------------------------------------------
# extended-precision residual
# ---------------------------------------------------------------------------
def ext_matvec(Acsr, x):
    """
    A @ x accumulated in extended precision, for one column x.

    scipy.sparse holds no clongdouble type, so the product is formed from the
    CSR arrays: data[k] * x[indices[k]] elementwise in extended precision, then
    summed per row. np.add.reduceat returns the element at the start offset for
    an empty row instead of zero, so empty rows are zeroed explicitly rather
    than assumed away.
    """
    indptr = Acsr.indptr
    products = Acsr.data.astype(EXT) * x.astype(EXT)[Acsr.indices]
    out = np.zeros(Acsr.shape[0], dtype=EXT)
    if products.size:
        nonempty = np.flatnonzero(np.diff(indptr) > 0)
        out[nonempty] = np.add.reduceat(products, indptr[nonempty])
    return out


def refine(Acsr, lu, b, max_steps=3, tol=None):
    """
    Reference solution of A x = b, by iterative refinement with an
    extended-precision residual.

    The factorization is reused for every correction, so each step costs one
    triangular solve pair and one extended-precision product. Refinement stops
    early once a correction no longer moves the iterate by more than tol
    relative, which is where the residual precision, not the solver, has become
    the limit. The iterate is carried in extended precision from the first
    correction onwards: rounded back to double at every step it could never
    become more accurate than the solutions it is meant to judge.

    Returns (x, steps, backward_residual), the last being
    ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf) in double, a check that
    the reference is itself a backward-stable solution of the system.
    """
    tol = EPS_EXT * 10 if tol is None else tol
    x = lu.solve(np.asarray(b, dtype=np.complex128))
    steps = 0
    for _ in range(max_steps):
        r = b.astype(EXT) - ext_matvec(Acsr, x)
        d = lu.solve(np.asarray(r, dtype=np.complex128))
        x = (x.astype(EXT) + d.astype(EXT))
        steps += 1
        norm_x = float(np.max(np.abs(x)))
        if norm_x and float(np.max(np.abs(d))) <= tol * norm_x:
            break

    r = b.astype(EXT) - ext_matvec(Acsr, x)
    denom = (float(abs(Acsr).sum(axis=1).max()) * float(np.max(np.abs(x)))
             + float(np.max(np.abs(b))))
    ref_res = float(np.max(np.abs(r)) / denom) if denom else np.nan
    return x, steps, ref_res


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
def load_M(f, idx):
    g = f[f"E_{idx}/M"]
    n = len(g["indptr"]) - 1
    return sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                         shape=(n, n))


def load_conditions(path):
    """
    {index: (cond_inf, cond_skeel, cond_skeel_x)} from a condition file.

    Rows not marked valid are dropped, and a missing file or a file predating
    the Skeel columns yields NaN for what it does not have, so the run
    proceeds and the affected bound columns are simply empty.
    """
    out = {}
    if path is None or not Path(path).exists():
        return out
    with h5py.File(path, "r") as f:
        if CONDITION_GROUP not in f:
            return out
        g = f[CONDITION_GROUP]
        indices = g["indices"][:]
        valid = g["valid"][:]
        nan = np.full(len(indices), np.nan)
        columns = [g[name][:] if name in g else nan
                   for name in ("cond_inf", "cond_skeel", "cond_skeel_x")]
        for row, index in enumerate(indices):
            if valid[row]:
                out[int(index)] = tuple(float(c[row]) for c in columns)
    return out


def _scalar(group, name):
    """A scalar dataset of a solver group, NaN if the run did not record it."""
    return float(group[name][()]) if name in group else np.nan


def solver_groups(f, idx, solvers, dtypes):
    """
    (solver, dtype, group) for every combination present at this index.

    The two half-precision Block Thomas variants share the HDF5 group of their
    complex counterparts and are distinguished only by the complex32 level
    beneath it, so each solver is restricted to the precisions cli.SOLVERS
    declares for it. Without that restriction, asking for block-thomas-fp16 at
    complex128 would silently read and relabel the complex128 Block Thomas
    results.
    """
    for solver in solvers:
        root = f.get(f"E_{idx}/{cli.h5_group(solver)}")
        if root is None:
            continue
        valid = cli.SOLVERS[solver].get("dtypes")
        for dt in dtypes:
            if valid is not None and dt not in valid:
                continue
            g = root.get(dt)
            if g is not None and "x" in g:
                yield solver, dt, g


# ---------------------------------------------------------------------------
# one index
# ---------------------------------------------------------------------------
def process_index(f, idx, solvers, dtypes, conditions, records, max_steps):
    """
    Reference solution of one index, then the forward error of every stored
    solution against it. Appends one record per (solver, dtype) to `records`.
    """
    rhs = f[f"E_{idx}/rhs"][:]
    if rhs.size == 0 or rhs.shape[-1] == 0:
        print(f"idx = {idx}: empty right-hand side, skipping")
        return
    B = rhs if rhs.ndim == 2 else rhs[:, None]

    present = list(solver_groups(f, idx, solvers, dtypes))
    if not present:
        print(f"idx = {idx}: no stored solutions, skipping")
        return

    A = load_M(f, idx)
    Acsr = A.tocsr()
    lu = splu(A.tocsc())

    X_ref = np.empty((A.shape[0], B.shape[1]), dtype=EXT)
    ref_res, ref_steps = 0.0, 0
    for col in range(B.shape[1]):
        X_ref[:, col], steps, res = refine(Acsr, lu, B[:, col], max_steps)
        ref_res = max(ref_res, res)
        ref_steps = max(ref_steps, steps)

    cond_inf, cond_skeel, cond_skeel_x = conditions.get(
        idx, (np.nan, np.nan, np.nan))
    ref_floor = cond_inf * EPS_EXT

    norm_inf = np.max(np.abs(X_ref), axis=0)
    norm_2 = np.sqrt(np.sum(np.abs(X_ref) ** 2, axis=0))

    print(f"idx = {idx}   n = {A.shape[0]}  nrhs = {B.shape[1]}  "
          f"reference: {ref_steps} refinement step(s), "
          f"backward residual {ref_res:.2e}, floor {ref_floor:.2e}")

    for solver, dt, g in present:
        xhat = np.asarray(g["x"][:])
        Xhat = xhat if xhat.ndim == 2 else xhat[:, None]
        if Xhat.shape != X_ref.shape:
            print(f"    {solver} / {dt}: shape {Xhat.shape} does not match "
                  f"the reference {X_ref.shape}, skipping")
            continue

        D = Xhat.astype(EXT) - X_ref
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd_inf = float(np.max(np.max(np.abs(D), axis=0) / norm_inf))
            fwd_2 = float(np.max(np.sqrt(np.sum(np.abs(D) ** 2, axis=0))
                                 / norm_2))

        eta_inf = _scalar(g, "nbe_inf")
        eta_1 = _scalar(g, "nbe_1")
        omega = _scalar(g, "cbe")
        bound_nw = cond_inf * eta_inf
        bound_cw = cond_skeel_x * omega

        records.append(dict(
            idx=idx, solver=solver, dtype=dt,
            n=A.shape[0], nrhs=B.shape[1],
            fwd_inf=fwd_inf, fwd_2=fwd_2,
            eta_inf=eta_inf, eta_1=eta_1, omega=omega,
            cond_inf=cond_inf, cond_skeel=cond_skeel,
            cond_skeel_x=cond_skeel_x,
            bound_nw=bound_nw, bound_cw=bound_cw,
            ratio_nw=fwd_inf / bound_nw if bound_nw else np.nan,
            ratio_cw=fwd_inf / bound_cw if bound_cw else np.nan,
            ref_res=ref_res, ref_floor=ref_floor, ref_steps=ref_steps))

        flag = "  (at the reference floor)" if fwd_inf <= 10 * ref_floor else ""
        print(f"    {solver:20s} {dt:10s} fwd_inf={fwd_inf:.3e}  "
              f"eta_inf={eta_inf:.2e}  omega={omega:.2e}  "
              f"fwd/(kappa*eta)={records[-1]['ratio_nw']:.2e}  "
              f"fwd/(cond*omega)={records[-1]['ratio_cw']:.2e}{flag}")
    print()


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap)
    cli.add_index_selection(ap, default_all=True)
    cli.add_solver_selection(
        ap, choices=cli.ALL_SOLVERS, default=DEFAULT_SOLVERS,
        help="solvers whose stored solutions to judge; those absent from the "
             "file are skipped")
    cli.add_dtypes(ap, choices=cli.COMPLEX_DTYPES, default=cli.COMPLEX_DTYPES,
                   help="precisions to judge; those absent are skipped")
    cli.add_output(ap, material=True, outdir_default=str(DEFAULT_OUTDIR),
                   outdir_help=f"directory holding the analysis file "
                               f"<material>.h5 (default: {DEFAULT_OUTDIR})")
    ap.add_argument("--condition-file", type=str, default=None, metavar="PATH",
                    help="condition_est.py output supplying kappa_inf and the "
                         "Skeel condition numbers (default: "
                         "<CONDITION_DIR>/<material>.h5). Without it the "
                         "bound columns are NaN and only the forward errors "
                         "are recorded")
    ap.add_argument("--refine-steps", type=int, default=3, metavar="N",
                    help="maximum refinement steps of the reference solution; "
                         "refinement stops earlier once the correction stops "
                         "moving the iterate (default: 3)")
    ap.add_argument("--no-save", action="store_true",
                    help="print the per-index report only, write no HDF5")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    material = args.material or h5path.stem
    cond_path = (Path(args.condition_file) if args.condition_file
                 else cli.analysis_h5(cli.CONDITION_DIR, material))
    conditions = load_conditions(cond_path)
    if not conditions:
        print(f"no condition data at {cond_path}; the bound columns will be "
              f"NaN")
    else:
        print(f"condition data: {cond_path} ({len(conditions)} valid rows)")

    records = []
    with h5py.File(h5path, "r") as f:
        indices = cli.resolve_indices(ap, args, cli.available_indices(f))
        for idx in indices:
            process_index(f, idx, args.solvers, args.dtypes, conditions,
                          records, args.refine_steps)

    if args.no_save:
        return
    if not records:
        print("no forward errors collected; nothing to write")
        return

    out_path = cli.analysis_h5(args.outdir, material)
    save_table(out_path, GROUP, records, columns=COLUMNS,
               attrs=dict(material=material, source=str(h5path),
                          condition_file=str(cond_path),
                          eps_ext=EPS_EXT, refine_steps=args.refine_steps,
                          solvers=list(args.solvers),
                          dtypes=list(args.dtypes),
                          **material_metadata(h5path)))
    print(f"wrote {out_path}:/{GROUP}  ({len(records)} rows)")
    print("Plot with: python ../plotting/block-thomas/plot_forward_error.py "
          f"{out_path}")


if __name__ == "__main__":
    main()

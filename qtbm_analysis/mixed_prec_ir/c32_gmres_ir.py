#!/usr/bin/env python3
"""
Half-precision Block Thomas as a GMRES-IR preconditioner, over an energy sweep.

Question
--------
A half-precision Block Thomas factorization has a unit roundoff of u = 2^-11,
so a direct solve with it delivers a relative residual near 5e-4 and a forward
error of kappa_2(M) times that. Classical iterative refinement cannot repair
this on the QTBM systems, because its convergence requires roughly
kappa_inf(M) u_f < 1, that is kappa below about 1e3.

This script measures whether GMRES-IR does: whether GMRES at complex128,
preconditioned by the half-precision factorization, recovers double-precision
accuracy at the condition numbers these matrices exhibit. If it does,
a half-precision factorization becomes a usable preconditioner rather than a
usable solver, which is the practically relevant claim.

Input
-----
    h5path              a material HDF5 file, read for E_<idx>/M and
                        E_<idx>/rhs and for global/condition_full_svd
    --idx / --start,--end   the energy indices to process
    --tol, --max-iter   the outer refinement criterion
    --gmres-*           the inner GMRES parameters

The Block Thomas partition is always detected from the sparsity pattern; it is
never uniform, and there is no option to make it so.

Algorithm
---------
The refinement drivers themselves are not reimplemented. This script imports
mpir and registers one additional builder into mpir.SOLVER_BUILDERS at run
time, an in-memory dict insertion and not a modification of any file, so that
mpir's solve_gmres_ir, solve_mixed_ir, solve_direct and benchmark_solver drive
the half-precision solver unchanged. Any correction to the refinement logic
therefore applies here automatically.

Five variants are compared per energy index, on the same matrix and the same
right-hand side:

    fp16 direct       BlockThomasFP16 with no refinement; the lower bound
    fp16 + LU-IR      classical refinement, inner solve one fp16 substitution;
                      establishes whether plain refinement suffices
    fp16 + GMRES-IR   inner solve GMRES at complex128 preconditioned by the
                      fp16 factorization; the variant under test
    c64 + GMRES-IR    the same with a complex64 factorization; a reference
                      point at a precision where refinement is known to work
    c128 direct       BlockThomas at complex128; the accuracy ceiling

The reference solution x_true comes from SuperLU at complex128, the same
convention mpir uses, so forward errors are comparable across both scripts.

Note on the cast precision for the fp16 variants
------------------------------------------------
mpir casts each vector with v.astype(low_dtype) before handing it to the
preconditioner. There is no complex32 in NumPy, and BlockThomasFP16 performs
its own rounding to float16 and its own power-of-two rescaling internally on
every solve. complex128 is therefore passed as the cast dtype: the cast is then
lossless and all precision loss occurs inside the half-precision solver, where
it belongs. Passing complex64 would silently insert an additional rounding step
ahead of the half-precision one and would misattribute its effect.

Output
------
    <outdir>/<material>.h5, group gmres_ir

in long format, one row per (index, variant), carrying the residual, the
forward error, the outer iteration count, the convergence flag, the inner GMRES
iteration counts, the wall time and the factor memory. The run configuration
and the skipped indices are group attributes. The file is opened in append mode
and only that group is rewritten, so results of other analyses of the same
material are preserved.

A verbose per-index log is written to stdout, including the refinement
convergence history and the inner iteration counts. That history, rather than
any single final number, is the evidence for whether half-precision
preconditioning works.

No figures are produced; see plotting/plot_mixed_prec_ir.py.

Usage
-----
    python c32_gmres_ir.py .../carbon-nanotube.h5 --idx 84

    python c32_gmres_ir.py .../carbon-nanotube.h5 --start 0 --end 401

    python ../plotting/plot_mixed_prec_ir.py \
        /scratch/yimili/mixed-precision-IR/carbon-nanotube.h5
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# mpir lives beside this file; the solver library, including cli, is one
# directory up. Both are added explicitly rather than relying on mpir's own
# sys.path side effect, so the import order here does not matter.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str((_HERE / ".." / "solvers").resolve()))

import cli
import mpir
from factor_io import save_table, material_metadata
from solver_classes import (extract_blocks_sparse, BlockThomasFP16,
                            block_sizes_from_matrix, offband_nnz)

# Top-level group of the analysis file this script writes.
GROUP = "gmres_ir"
COLUMNS = ["idx", "kappa", "n", "nnz", "variant", "relres", "true_err",
           "outer_iters", "converged", "inner_gmres_total",
           "inner_gmres_mean", "wall_s", "factor_mb", "note"]

DEFAULT_OUTDIR = cli.MIXED_PREC_DIR


# ---------------------------------------------------------------------------
# Registration of the half-precision solver with mpir's builder registry
# ---------------------------------------------------------------------------
def register_fp16_builder():
    """
    Add a block-thomas-fp16 entry to mpir.SOLVER_BUILDERS, in memory only.

    mpir's builder contract is builder(A, dtype, bs, b) returning an object
    exposing solve(b) and factor_nbytes(). BlockThomasFP16 satisfies both and
    ignores the dtype argument, since its working precision is always float16.
    """
    def _build(A, dtype, bs, b):
        # bs is always None here: the partition is detected from the sparsity
        # pattern, never uniform, as everywhere else in the pipeline. See
        # mpir._build_block_thomas.
        if bs is None:
            bs = block_sizes_from_matrix(A)
        bad = offband_nnz(A, bs)
        if bad:
            raise ValueError(
                f"block partition leaves {bad} nonzeros outside the "
                f"block-tridiagonal band; the solution would be wrong")
        D, L, U = extract_blocks_sparse(A, bs)
        return BlockThomasFP16(D, L, U)

    mpir.SOLVER_BUILDERS["block-thomas-fp16"] = _build


# ---------------------------------------------------------------------------
# One energy index
# ---------------------------------------------------------------------------
def run_index(h5path, idx, bs, args):
    """
    Run every variant on one energy index.

    Returns a list of per-variant result dicts matching the CSV schema. A
    variant that raises is recorded with NaN metrics and the exception in its
    note field, so a single failure does not remove the index from the sweep.
    """
    HIGH = mpir.HIGH_DTYPE

    A, b = mpir.load_system(h5path, idx)
    A_high = A.tocsc().astype(HIGH)
    b_high = np.asarray(b, dtype=HIGH)
    kappa = mpir.load_condition_number(h5path, idx)

    n, nnz = A.shape[0], int(A.nnz)
    print(f"\n{'='*78}")
    print(f"E_{idx}   n={n}  nnz={nnz}  b.shape={b.shape}  "
          f"kappa={'n/a' if kappa is None else f'{kappa:.3e}'}")
    print(f"{'='*78}")

    # Reference solution, by the same convention mpir uses: SuperLU at
    # complex128.
    ref = mpir.SOLVER_BUILDERS["superlu"](A, HIGH, bs, b)
    x_true = ref.solve(b_high).astype(HIGH)
    if hasattr(ref, "free"):
        ref.free()

    # Cast precision for the half-precision variants. complex128 makes the cast
    # lossless, so that all precision loss occurs inside the half-precision
    # solver. See the module docstring.
    FP16_CAST = np.complex128

    variants = []

    variants.append((
        "fp16 direct",
        lambda: mpir.solve_direct("block-thomas-fp16", A, b, bs, FP16_CAST),
    ))

    if not args.skip_lu_ir:
        variants.append((
            "fp16 + LU-IR",
            lambda: mpir.solve_mixed_ir("block-thomas-fp16", A, b, bs, FP16_CAST,
                                        args.tol, args.max_iter, x_true=x_true),
        ))

    variants.append((
        "fp16 + GMRES-IR",
        lambda: mpir.solve_gmres_ir("block-thomas-fp16", A, b, bs, FP16_CAST,
                                    args.tol, args.max_iter, x_true=x_true,
                                    gmres_tol=args.gmres_tol,
                                    gmres_restart=args.gmres_restart,
                                    gmres_max_iter=args.gmres_max_iter),
    ))

    if not args.skip_c64:
        variants.append((
            "c64 + GMRES-IR",
            lambda: mpir.solve_gmres_ir("block-thomas", A, b, bs, np.complex64,
                                        args.tol, args.max_iter, x_true=x_true,
                                        gmres_tol=args.gmres_tol,
                                        gmres_restart=args.gmres_restart,
                                        gmres_max_iter=args.gmres_max_iter),
        ))

    if not args.skip_c128:
        variants.append((
            "c128 direct",
            lambda: mpir.solve_direct("block-thomas", A, b, bs, HIGH),
        ))

    rows = []
    for name, fn in variants:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                t0 = time.perf_counter()
                recs = mpir.benchmark_solver(fn, A_high, b_high,
                                             args.repeats, x_true=x_true)
                wall_total = time.perf_counter() - t0
            n_warn = len(caught)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {name:<18}: FAILED ({type(exc).__name__}: {exc})")
            rows.append(dict(idx=idx, kappa=kappa, n=n, nnz=nnz, variant=name,
                             relres=float("nan"), true_err=float("nan"),
                             outer_iters=-1, converged=0,
                             inner_gmres_total=-1, inner_gmres_mean=float("nan"),
                             wall_s=float("nan"), factor_mb=float("nan"),
                             note=f"{type(exc).__name__}: {exc}"))
            continue

        r0 = recs[0]
        extra = r0["extra"]
        hist = extra.get("history", [])
        gmres_hist = extra.get("gmres_iters_history", [])

        outer = len(hist)
        converged = bool(hist and hist[-1] < args.tol)
        inner_flat = [it for round_ in gmres_hist for it in round_]
        inner_total = int(sum(inner_flat)) if inner_flat else 0
        inner_mean = float(np.mean(inner_flat)) if inner_flat else float("nan")

        relres = float(np.median([r["residual"] for r in recs]))
        true_err = r0["true_err"]
        true_err = float(true_err) if true_err is not None else float("nan")
        wall_s = float(np.median([r["wall_s"] for r in recs]))
        factor_mb = extra.get("mem_bytes", 0) / 1e6

        tag = "converged" if converged else ("no-IR" if not hist else "NOT converged")
        print(f"  {name:<18}: relres {relres:.3e}   true_err {true_err:.3e}   "
              f"{wall_s*1e3:8.1f} ms   factor {factor_mb:6.2f} MB   [{tag}]")
        if hist:
            hist_str = "  ".join(f"{h:.2e}" for h in hist)
            print(f"  {'':<18}  IR residual history: {hist_str}")
            te_hist = extra.get("true_err_history", [])
            if te_hist:
                te_str = "  ".join(f"{h:.2e}" for h in te_hist)
                print(f"  {'':<18}  IR true-err history: {te_str}")
            if gmres_hist:
                gm_str = "  ".join(str(g[0] if len(g) == 1 else g) for g in gmres_hist)
                print(f"  {'':<18}  inner GMRES iters:   {gm_str}")
        if n_warn:
            print(f"  {'':<18}  ({n_warn} warning(s) raised, e.g. inner GMRES "
                  f"not fully converged)")

        rows.append(dict(idx=idx, kappa=kappa, n=n, nnz=nnz, variant=name,
                         relres=relres, true_err=true_err,
                         outer_iters=outer, converged=int(converged),
                         inner_gmres_total=inner_total, inner_gmres_mean=inner_mean,
                         wall_s=wall_s, factor_mb=factor_mb, note=""))

    return rows


# ---------------------------------------------------------------------------
def parse_args():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap)
    cli.add_index_selection(ap, default_all=False)

    ap.add_argument("--tol", type=float, default=1e-14,
                    help="outer convergence tolerance on ||r||/||b||")
    ap.add_argument("--max-iter", type=int, default=10, metavar="N",
                    help="maximum outer refinement iterations")
    ap.add_argument("--gmres-tol", type=float, default=1e-8,
                    help="relative tolerance of the inner GMRES solve")
    ap.add_argument("--gmres-restart", type=int, default=30,
                    help="inner GMRES restart parameter")
    ap.add_argument("--gmres-max-iter", type=int, default=50,
                    help="maximum inner GMRES iterations per outer step")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats per variant; the median is reported")

    ap.add_argument("--skip-lu-ir", action="store_true",
                    help="skip the fp16 + LU-IR variant")
    ap.add_argument("--skip-c64", action="store_true",
                    help="skip the complex64 + GMRES-IR variant")
    ap.add_argument("--skip-c128", action="store_true",
                    help="skip the complex128 direct variant")

    cli.add_output(ap, outdir_default=str(DEFAULT_OUTDIR),
                   outdir_help=f"directory holding the analysis file "
                               f"<material>.h5 (default: {DEFAULT_OUTDIR})")
    return ap, ap.parse_args()


def main():
    ap, args = parse_args()
    register_fp16_builder()

    args.h5path = Path(args.h5path)
    indices = cli.resolve_indices(ap, args)
    material = args.material or args.h5path.stem
    out_path = cli.analysis_h5(args.outdir, material)

    print(f"material   : {material}")
    print(f"h5path     : {args.h5path}")
    print(f"indices    : {len(indices)} ({indices[0]}..{indices[-1]})")
    print(f"outer tol  : {args.tol:.1e}   max outer iters: {args.max_iter}")
    print(f"inner GMRES: rtol {args.gmres_tol:.1e}  restart {args.gmres_restart}  "
          f"maxiter {args.gmres_max_iter}")

    all_rows = []
    skipped = []
    for idx in indices:
        try:
            all_rows.extend(run_index(args.h5path, idx, None, args))
        except SystemExit as e:            # raised by mpir.load_system for a bad index
            skipped.append((idx, str(e)))
            print(f"\nE_{idx}: skipped ({e})")
        except Exception as e:             # noqa: BLE001
            skipped.append((idx, f"{type(e).__name__}: {e}"))
            print(f"\nE_{idx}: skipped ({type(e).__name__}: {e})")

    # ---- table, with the run configuration as group attributes --------------
    attrs = dict(
        material=material,
        source=str(args.h5path),
        partition="auto (block_sizes_from_matrix)",
        idx_requested=[int(indices[0]), int(indices[-1])],
        n_requested=len(indices),
        outer_tol=float(args.tol),
        outer_max_iter=int(args.max_iter),
        gmres_tol=float(args.gmres_tol),
        gmres_restart=int(args.gmres_restart),
        gmres_max_iter=int(args.gmres_max_iter),
        x_true="superlu complex128",
        n_skipped=len(skipped),
        **material_metadata(args.h5path),
    )
    if skipped:
        attrs["skipped_idx"] = np.asarray([i for i, _ in skipped], dtype=np.int64)
        attrs["skipped_reason"] = [message for _, message in skipped]
    save_table(out_path, GROUP, all_rows, columns=COLUMNS, attrs=attrs)

    # ---- summary ------------------------------------------------------------
    print(f"\n{'='*78}")
    gm = [r for r in all_rows if r["variant"] == "fp16 + GMRES-IR"]
    if gm:
        conv = sum(r["converged"] for r in gm)
        print(f"fp16 + GMRES-IR converged on {conv}/{len(gm)} indices "
              f"(tol {args.tol:.1e}, max {args.max_iter} outer iters)")
        it = [r["outer_iters"] for r in gm if r["converged"]]
        if it:
            print(f"  outer iterations when converged: min {min(it)}  "
                  f"median {int(np.median(it))}  max {max(it)}")
        bad = [r["idx"] for r in gm if not r["converged"]]
        if bad:
            head = ", ".join(str(i) for i in bad[:20])
            print(f"  did NOT converge at idx: {head}"
                  f"{' ...' if len(bad) > 20 else ''}")
    if skipped:
        print(f"{len(skipped)} indices skipped entirely.")
    print(f"\nwrote {out_path}:/{GROUP}  ({len(all_rows)} rows)")
    print(f"Plot with: python ../plotting/plot_mixed_prec_ir.py {out_path}")


if __name__ == "__main__":
    main()
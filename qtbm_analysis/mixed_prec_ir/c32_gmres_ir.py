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
    --bs                the Block Thomas block size
    --tol, --max-iter   the outer refinement criterion
    --gmres-*           the inner GMRES parameters

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
    <outdir>/<material>_fp16_gmres_ir.csv

in long format, one row per (index, variant), carrying the residual, the
forward error, the outer iteration count, the convergence flag, the inner GMRES
iteration counts, the wall time and the factor memory. The header lines record
the run configuration.

A verbose per-index log is written to stdout, including the refinement
convergence history and the inner iteration counts. That history, rather than
any single final number, is the evidence for whether half-precision
preconditioning works.

No figures are produced; see plotting/plot_mixed_prec_ir.py.

Usage
-----
    python c32_gmres_ir.py /scratch/yimili/matrices/hdf5/carbon-nanotube.h5 \
        --idx 84 --bs 32

    python c32_gmres_ir.py /scratch/yimili/matrices/hdf5/carbon-nanotube.h5 \
        --start 0 --end 401 --bs 32 --outdir plots

    python ../plotting/plot_mixed_prec_ir.py \
        plots/carbon-nanotube_fp16_gmres_ir.csv
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# mpir lives beside this file and appends ../solvers to sys.path itself when
# imported, so solver_classes becomes importable along with it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpir
from solver_classes import extract_blocks_sparse, BlockThomasFP16


# ---------------------------------------------------------------------------
# Registration of the half-precision solver with mpir's builder registry
# ---------------------------------------------------------------------------
def register_fp16_builder():
    """
    Add a block_thomas_fp16 entry to mpir.SOLVER_BUILDERS, in memory only.

    mpir's builder contract is builder(A, dtype, bs, b) returning an object
    exposing solve(b) and factor_nbytes(). BlockThomasFP16 satisfies both and
    ignores the dtype argument, since its working precision is always float16.
    """
    def _build(A, dtype, bs, b):
        if bs is None:
            raise ValueError("--bs is required for block_thomas_fp16")
        D, L, U = extract_blocks_sparse(A, bs)
        return BlockThomasFP16(D, L, U)

    mpir.SOLVER_BUILDERS["block_thomas_fp16"] = _build


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
        lambda: mpir.solve_direct("block_thomas_fp16", A, b, bs, FP16_CAST),
    ))

    if not args.skip_lu_ir:
        variants.append((
            "fp16 + LU-IR",
            lambda: mpir.solve_mixed_ir("block_thomas_fp16", A, b, bs, FP16_CAST,
                                        args.tol, args.max_iter, x_true=x_true),
        ))

    variants.append((
        "fp16 + GMRES-IR",
        lambda: mpir.solve_gmres_ir("block_thomas_fp16", A, b, bs, FP16_CAST,
                                    args.tol, args.max_iter, x_true=x_true,
                                    gmres_tol=args.gmres_tol,
                                    gmres_restart=args.gmres_restart,
                                    gmres_maxiter=args.gmres_maxiter),
    ))

    if not args.skip_c64:
        variants.append((
            "c64 + GMRES-IR",
            lambda: mpir.solve_gmres_ir("block_thomas", A, b, bs, np.complex64,
                                        args.tol, args.max_iter, x_true=x_true,
                                        gmres_tol=args.gmres_tol,
                                        gmres_restart=args.gmres_restart,
                                        gmres_maxiter=args.gmres_maxiter),
        ))

    if not args.skip_c128:
        variants.append((
            "c128 direct",
            lambda: mpir.solve_direct("block_thomas", A, b, bs, HIGH),
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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5path", type=Path, help="material HDF5 file")
    p.add_argument("--idx", type=int, nargs="*", default=None,
                   help="one or more energy indices (alternative to --start/--end)")
    p.add_argument("--start", type=int, default=None, help="first energy index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="last energy index (inclusive)")
    p.add_argument("--bs", type=int, default=32, help="block size")

    p.add_argument("--tol", type=float, default=1e-14,
                   help="outer IR convergence tolerance on ||r||/||b||")
    p.add_argument("--max-iter", type=int, default=10, help="max outer IR iterations")
    p.add_argument("--gmres-tol", type=float, default=1e-8, help="inner GMRES rtol")
    p.add_argument("--gmres-restart", type=int, default=30, help="inner GMRES restart")
    p.add_argument("--gmres-maxiter", type=int, default=50, help="inner GMRES maxiter")
    p.add_argument("--repeats", type=int, default=1, help="repeats per variant (median reported)")

    p.add_argument("--skip-lu-ir", action="store_true", help="skip the fp16 + LU-IR variant")
    p.add_argument("--skip-c64", action="store_true", help="skip the c64 + GMRES-IR variant")
    p.add_argument("--skip-c128", action="store_true", help="skip the c128 direct variant")

    p.add_argument("--material", type=str, default=None,
                   help="tag for output filename (default: h5 filename stem)")
    p.add_argument("--outdir", type=str, default=None,
                   help="output directory (default: <script_dir>/plots)")
    return p.parse_args()


def main():
    args = parse_args()

    register_fp16_builder()

    # Index selection: --idx and --start/--end are alternatives.
    if args.idx:
        indices = list(args.idx)
    elif args.start is not None and args.end is not None:
        indices = list(range(args.start, args.end + 1))
    else:
        raise SystemExit("give either --idx N [N ...] or --start S --end E")

    material = args.material or args.h5path.stem
    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir) if args.outdir else script_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{material}_fp16_gmres_ir.csv"

    print(f"material   : {material}")
    print(f"h5path     : {args.h5path}")
    print(f"block size : {args.bs}")
    print(f"indices    : {len(indices)} ({indices[0]}..{indices[-1]})")
    print(f"outer tol  : {args.tol:.1e}   max outer iters: {args.max_iter}")
    print(f"inner GMRES: rtol {args.gmres_tol:.1e}  restart {args.gmres_restart}  "
          f"maxiter {args.gmres_maxiter}")

    all_rows = []
    skipped = []
    for idx in indices:
        try:
            all_rows.extend(run_index(args.h5path, idx, args.bs, args))
        except SystemExit as e:            # raised by mpir.load_system for a bad index
            skipped.append((idx, str(e)))
            print(f"\nE_{idx}: skipped ({e})")
        except Exception as e:             # noqa: BLE001
            skipped.append((idx, f"{type(e).__name__}: {e}"))
            print(f"\nE_{idx}: skipped ({type(e).__name__}: {e})")

    # ---- CSV, with the run configuration in the header lines ----------------
    fields = ["idx", "kappa", "n", "nnz", "variant", "relres", "true_err",
              "outer_iters", "converged", "inner_gmres_total", "inner_gmres_mean",
              "wall_s", "factor_mb", "note"]
    with open(csv_path, "w", newline="") as f:
        f.write(f"# material     : {material}\n")
        f.write(f"# h5path       : {args.h5path}\n")
        f.write(f"# block size   : {args.bs}\n")
        f.write(f"# indices      : {indices[0]}..{indices[-1]} ({len(indices)} requested)\n")
        f.write(f"# outer tol    : {args.tol}   max_iter {args.max_iter}\n")
        f.write(f"# inner gmres  : rtol {args.gmres_tol} restart {args.gmres_restart} "
                f"maxiter {args.gmres_maxiter}\n")
        f.write(f"# x_true       : superlu complex128\n")
        f.write(f"# skipped      : {len(skipped)}\n")
        for i, msg in skipped:
            f.write(f"#   idx={i}: {msg}\n")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

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
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
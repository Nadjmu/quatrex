#!/usr/bin/env python3
"""
test_fp16_gmres_ir.py -- isolate ONE question:

    Can BlockThomasFP16 (half-precision block Thomas) serve as a usable
    preconditioner for GMRES-IR, i.e. does GMRES-IR recover accuracy that
    the raw fp16 solve cannot?

Nothing in mpir.py / solver_classes.py / half_blockthomas.py is modified.
This script only IMPORTS from them and registers one extra builder into
mpir.SOLVER_BUILDERS at runtime (an in-memory dict insertion, not an edit
to the file on disk), so that mpir's own solve_gmres_ir / solve_mixed_ir /
solve_direct / benchmark_solver can drive the fp16 solver unchanged.

Variants compared per energy index (all on the SAME matrix + RHS):

    fp16 direct            BlockThomasFP16, no refinement          (lower bound)
    fp16 + LU-IR           classic IR, inner = one fp16 solve      (does plain IR suffice?)
    fp16 + GMRES-IR        inner = GMRES(complex128) preconditioned
                           by the fp16 factorization               <-- THE TEST
    c64  + GMRES-IR        same, but complex64 block Thomas        (reference point)
    c128 direct            BlockThomas complex128, no refinement   (accuracy ceiling)

x_true comes from SuperLU at complex128, so "true error" is measured against
the same reference mpir uses.

Usage:
    python test_fp16_gmres_ir.py /scratch/yimili/matrices/hdf5/carbon-nanotube.h5 \
        --idx 84 --bs 32

    # sweep a range, write CSV for later analysis
    python test_fp16_gmres_ir.py /scratch/yimili/matrices/hdf5/carbon-nanotube.h5 \
        --start 0 --end 401 --bs 32 --outdir plots

Output: one CSV (long format, one row per (idx, variant)) plus a verbose
per-index log on stdout showing the IR convergence history and inner GMRES
iteration counts -- that history is the actual evidence for whether fp16
preconditioning works, more than any single final number.
"""

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# import the existing modules (no edits to any of them)
# ---------------------------------------------------------------------------
def _setup_paths(mpir_dir=None, half_dir=None, solvers_dir=None):
    """Put mpir.py / half_blockthomas.py / solver_classes.py on sys.path.

    Layout is guessed from a few plausible locations relative to this script;
    override with --mpir-dir / --half-dir / --solvers-dir if the guess is wrong.
    Note mpir.py itself appends ../solvers to sys.path when imported, so
    solver_classes usually comes along for free once mpir is importable.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here,
        here / "mixed_prec_ir",
        here / "solvers",
        here.parent,
        here.parent / "mixed_prec_ir",
        here.parent / "solvers",
    ]
    for explicit in (mpir_dir, half_dir, solvers_dir):
        if explicit:
            candidates.insert(0, Path(explicit).resolve())
    for c in candidates:
        c = Path(c)
        if c.is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


def _import_components():
    import mpir
    from solver_classes import extract_blocks_sparse, BlockThomas
    from half_blockthomas import BlockThomasFP16
    return mpir, extract_blocks_sparse, BlockThomas, BlockThomasFP16


# ---------------------------------------------------------------------------
# register BlockThomasFP16 with mpir's builder registry (in-memory only)
# ---------------------------------------------------------------------------
def register_fp16_builder(mpir, extract_blocks_sparse, BlockThomasFP16):
    """Add a 'block_thomas_fp16' entry to mpir.SOLVER_BUILDERS.

    mpir's builder contract is builder(A, dtype, bs, b) -> object with
    .solve(b) and .factor_nbytes(); BlockThomasFP16 satisfies both, and
    ignores the dtype argument (it is always fp16 internally).
    """
    def _build(A, dtype, bs, b):
        if bs is None:
            raise ValueError("--bs is required for block_thomas_fp16")
        D, L, U = extract_blocks_sparse(A, bs)
        return BlockThomasFP16(D, L, U)

    mpir.SOLVER_BUILDERS["block_thomas_fp16"] = _build


# ---------------------------------------------------------------------------
# one energy index
# ---------------------------------------------------------------------------
def run_index(mpir, h5path, idx, bs, args):
    """Run every variant on E_<idx>. Returns list of per-variant result dicts."""
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

    # reference solution (same convention as mpir: superlu @ complex128)
    ref = mpir.SOLVER_BUILDERS["superlu"](A, HIGH, bs, b)
    x_true = ref.solve(b_high).astype(HIGH)
    if hasattr(ref, "free"):
        ref.free()

    # NOTE ON low_dtype FOR THE fp16 VARIANTS:
    # mpir casts vectors with v.astype(low_dtype) before handing them to the
    # preconditioner. There is no numpy complex32, and BlockThomasFP16 does
    # its own fp16 rounding (and its own power-of-2 rescaling) internally on
    # every solve. So we pass complex128 as the "cast" dtype -- the cast is
    # then lossless and ALL precision loss happens inside the fp16 solver,
    # where we want it. Passing complex64 instead would silently insert an
    # extra c64 rounding step ahead of the fp16 one.
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
        description="Test BlockThomasFP16 as a GMRES-IR preconditioner.")
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
    p.add_argument("--mpir-dir", type=str, default=None, help="folder containing mpir.py")
    p.add_argument("--half-dir", type=str, default=None,
                   help="folder containing half_blockthomas.py")
    p.add_argument("--solvers-dir", type=str, default=None,
                   help="folder containing solver_classes.py")
    return p.parse_args()


def main():
    args = parse_args()

    _setup_paths(args.mpir_dir, args.half_dir, args.solvers_dir)
    try:
        mpir, extract_blocks_sparse, BlockThomas, BlockThomasFP16 = _import_components()
    except ImportError as e:
        raise SystemExit(
            f"import failed: {e}\n"
            f"Point the script at the right folders explicitly, e.g.:\n"
            f"  --mpir-dir ../mixed_prec_ir --solvers-dir ../solvers --half-dir .")

    register_fp16_builder(mpir, extract_blocks_sparse, BlockThomasFP16)

    # index selection
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
            all_rows.extend(run_index(mpir, args.h5path, idx, args.bs, args))
        except SystemExit as e:            # mpir.load_system raises this for bad idx
            skipped.append((idx, str(e)))
            print(f"\nE_{idx}: skipped ({e})")
        except Exception as e:             # noqa: BLE001
            skipped.append((idx, f"{type(e).__name__}: {e}"))
            print(f"\nE_{idx}: skipped ({type(e).__name__}: {e})")

    # ------------------------------------------------------------------ CSV
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

    # ------------------------------------------------------------- summary
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
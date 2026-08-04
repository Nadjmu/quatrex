#!/usr/bin/env python3
"""
#5. Growth factor analysis of the Block Thomas factorization

blockthomas_stability.py -- backward-stability analysis of the Block Thomas
factorization, reading directly from a material's RAW HDF5 file.

Block Thomas is an LU-based method, so (per Higham) it is NOT unconditionally
backward stable: the perturbation it introduces is bounded by

        |A - L U|  <=  gamma_n * |L| |U|      (entrywise)

which, taking a monotone norm (1- or inf-norm), gives

        ||A - L U||  <=  gamma_n * || |L| |U| ||.

The factorization is backward stable in practice iff the growth of the factors
relative to A stays modest -- this is governed by the pivot GROWTH FACTOR, a
property of the block-pivoting structure, and is INDEPENDENT of cond(A): an
ill-conditioned A can still factor with tiny growth, and a well-conditioned A
can have large growth if a diagonal (Schur-complement) block becomes badly
scaled. Conditioning governs the accuracy of x, not the backward error.

This script assembles the genuine global block-bidiagonal L, U such that
A == L @ U exactly (same construction as save_lu_npz.py, verified to machine
precision), then for each of the 1-norm and inf-norm reports two ratios:

    loose ratio  =  ||L|| * ||U|| / ||A||          (classical, always an upper bound)
    tight ratio  =  || |L| |U| || / ||A||          (true Wilkinson quantity, sharper)

Both dtypes present (complex128 and/or complex64) are analysed side by side.

Because the assembled L @ U reproduces A directly (no row/column permutation and
no diagonal scaling -- unlike SuperLU/UMFPACK), ||A|| can be used as-is in the
denominator with no permutation bookkeeping.

Over a sweep (--start/--end) the per-index metrics are accumulated and, unless
--no-plot is given, written to PLOTDIR (default /scratch/yimili/matrices/plots)
as:
    <material>_blockthomas_stability.png    tight/loose ratio + residual vs index
    <material>_blockthomas_stability.csv    the raw per-(index,dtype,norm) numbers
where <material> is the HDF5 file's stem (e.g. "graphene").

Usage:
    python blockthomas_stability.py /scratch/yimili/matrices/hdf5/graphene.h5 --idx 25
    python blockthomas_stability.py /scratch/.../graphene.h5 --start 1 --end 400
    python blockthomas_stability.py /scratch/.../graphene.h5 --start 1 --end 400 --dtype complex128
    python blockthomas_stability.py /scratch/.../graphene.h5 --start 1 --end 400 --plotdir /some/other/dir

The HDF5 layout assumed (written by bench_all / factor_io) is:
    E_<idx>/M/{data,indices,indptr}                 original matrix, CSC triplet
    E_<idx>/blockthomas/<dtype>/L                   (N-1, bs, bs) sub-diagonal blocks
    E_<idx>/blockthomas/<dtype>/U                   (N-1, bs, bs) super-diagonal blocks
    E_<idx>/blockthomas/<dtype>/Dmod_lu             (N, bs, bs) packed LU of modified diag blocks
    E_<idx>/blockthomas/<dtype>/Dmod_piv            (N, bs) pivots
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import norm as spnorm

DTYPES = ("complex128", "complex64")
NORMS = ("1-norm", "inf-norm")
DEFAULT_PLOTDIR = Path("/scratch/yimili/block-thomas")


# ---------------------------------------------------------------------------
# original matrix M  (stored as a CSC triplet, same as everywhere else)
# ---------------------------------------------------------------------------
def load_M(f, idx):
    g = f[f"E_{idx}/M"]
    n = len(g["indptr"]) - 1
    return sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                         shape=(n, n))


# ---------------------------------------------------------------------------
# Block Thomas global L, U assembly  (ported verbatim from save_lu_npz.py so
# the two scripts agree bit-for-bit; A == L_global @ U_global to machine eps)
# ---------------------------------------------------------------------------
def _piv_to_perm(piv, m):
    perm = list(range(m))
    for i, p in enumerate(piv):
        perm[i], perm[p] = perm[p], perm[i]
    return perm


def _reconstruct_dmod(lu_k, piv_k, bs):
    Lk = np.tril(lu_k, -1) + np.eye(bs, dtype=lu_k.dtype)
    Uk = np.triu(lu_k)
    Pk = np.eye(bs, dtype=lu_k.dtype)[_piv_to_perm(piv_k, bs)]
    return Pk.T @ Lk @ Uk


def _bmat_bidiag(diag_blocks, offdiag_blocks, position):
    N = len(diag_blocks)
    grid = [[None] * N for _ in range(N)]
    for k in range(N):
        grid[k][k] = sp.csr_matrix(diag_blocks[k])
    for k in range(N - 1):
        if position == "sub":
            grid[k + 1][k] = sp.csr_matrix(offdiag_blocks[k])
        else:
            grid[k][k + 1] = sp.csr_matrix(offdiag_blocks[k])
    return sp.bmat(grid, format="csr")


def assemble_blockthomas_lu(group):
    """Return global (L, U) as CSC such that A == L @ U (see save_lu_npz.py)."""
    L_off = group["L"][:]
    U_off = group["U"][:]
    Dlu = group["Dmod_lu"][:]
    Dpiv = group["Dmod_piv"][:]
    N, bs = Dlu.shape[0], Dlu.shape[1]
    dtype = Dlu.dtype

    D_mod = [_reconstruct_dmod(Dlu[k], Dpiv[k], bs) for k in range(N)]

    E = []
    for k in range(1, N):
        Dinv = sla.lu_solve((Dlu[k - 1], Dpiv[k - 1]), np.eye(bs, dtype=dtype))
        E.append(L_off[k - 1] @ Dinv)

    I_blocks = [np.eye(bs, dtype=dtype)] * N
    L_global = _bmat_bidiag(I_blocks, E, "sub").tocsc()
    U_global = _bmat_bidiag(D_mod, list(U_off), "super").tocsc()
    return L_global, U_global


# ---------------------------------------------------------------------------
# the actual stability metrics
# ---------------------------------------------------------------------------
def analyse(A, L, U):
    """
    Compute, for both the 1-norm and inf-norm:
        ||A||, ||L||, ||U||, ||L||*||U||, || |L||U| ||,
        loose ratio  = ||L|| ||U|| / ||A||
        tight ratio  = || |L||U| || / ||A||
    Also the assembly residual ||A - L U|| / ||A|| as a correctness guard.
    Returns a dict keyed by norm label.
    """
    absLU = (abs(L) @ abs(U))          # |L||U|, stays sparse

    R = (A - (L @ U)).tocsr()          # correctness guard
    R.eliminate_zeros()

    out = {}
    for p, label in ((1, "1-norm"), (np.inf, "inf-norm")):
        nA = spnorm(A, p)
        nL = spnorm(L, p)
        nU = spnorm(U, p)
        nLU_abs = spnorm(absLU, p)
        n_resid = spnorm(R, p) if R.nnz else 0.0
        out[label] = dict(
            nA=nA, nL=nL, nU=nU,
            prod=nL * nU,
            LU_abs=nLU_abs,
            loose=(nL * nU) / nA if nA else np.inf,
            tight=nLU_abs / nA if nA else np.inf,
            resid_rel=(n_resid / nA) if nA else np.inf,
        )
    return out


def _fmt_block(dtype_name, res):
    lines = [f"    dtype = {dtype_name}"]
    for label in NORMS:
        r = res[label]
        lines.append(
            f"      [{label:8s}]  ||A||={r['nA']:.3e}  ||L||={r['nL']:.3e}  "
            f"||U||={r['nU']:.3e}"
        )
        lines.append(
            f"                  || |L||U| || = {r['LU_abs']:.3e}    "
            f"||L||*||U|| = {r['prod']:.3e}"
        )
        lines.append(
            f"                  loose ratio ||L||||U||/||A|| = {r['loose']:.3e}"
        )
        lines.append(
            f"                  tight ratio || |L||U| ||/||A|| = {r['tight']:.3e}"
        )
        lines.append(
            f"                  (assembly check ||A-LU||/||A|| = {r['resid_rel']:.2e})"
        )
    return "\n".join(lines)


def process_index(f, idx, dtypes, records):
    """Analyse one index; print a report and append flat rows to `records`."""
    print(f"idx = {idx}")
    bt_root = f.get(f"E_{idx}/blockthomas")
    if bt_root is None:
        print("    no blockthomas group at this index -- skipping")
        return

    A = load_M(f, idx)

    per_dtype = {}
    for dt in dtypes:
        g = bt_root.get(dt)
        if g is None:
            print(f"    dtype = {dt}: not present -- skipping")
            continue
        # measure ||A|| in the same precision the factors were computed at
        A_dt = A.astype(np.dtype(dt))
        L, U = assemble_blockthomas_lu(g)
        res = analyse(A_dt, L, U)
        per_dtype[dt] = res
        print(_fmt_block(dt, res))

        for label in NORMS:
            r = res[label]
            records.append(dict(idx=idx, dtype=dt, norm=label,
                                nA=r["nA"], nL=r["nL"], nU=r["nU"],
                                prod=r["prod"], LU_abs=r["LU_abs"],
                                loose=r["loose"], tight=r["tight"],
                                resid_rel=r["resid_rel"]))

    if "complex128" in per_dtype and "complex64" in per_dtype:
        for label in NORMS:
            t128 = per_dtype["complex128"][label]["tight"]
            t64 = per_dtype["complex64"][label]["tight"]
            factor = t64 / t128 if t128 else np.inf
            print(f"    [{label:8s}] tight-ratio change c64/c128 = {factor:.10f}x")
    print()


# ---------------------------------------------------------------------------
# output: CSV + plots
# ---------------------------------------------------------------------------
def write_csv(records, csv_path):
    fields = ["idx", "dtype", "norm", "nA", "nL", "nU", "prod",
              "LU_abs", "loose", "tight", "resid_rel"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in records:
            w.writerow(row)
    print(f"wrote {csv_path}  ({len(records)} rows)")


def make_plot(records, material, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtypes_present = [d for d in DTYPES if any(r["dtype"] == d for r in records)]
    if not records:
        print("no records to plot")
        return

    # one row of panels per norm: (tight & loose ratio) | (assembly residual)
    fig, axes = plt.subplots(len(NORMS), 2, figsize=(13, 4.2 * len(NORMS)),
                             squeeze=False)

    color = {"complex128": "#1f4e79", "complex64": "#c0504d"}

    for row_i, label in enumerate(NORMS):
        ax_ratio = axes[row_i][0]
        ax_resid = axes[row_i][1]

        for dt in dtypes_present:
            rows = sorted((r for r in records
                           if r["norm"] == label and r["dtype"] == dt),
                          key=lambda r: r["idx"])
            if not rows:
                continue
            idxs = [r["idx"] for r in rows]
            tight = [r["tight"] for r in rows]
            loose = [r["loose"] for r in rows]
            resid = [r["resid_rel"] for r in rows]
            c = color.get(dt, None)

            ax_ratio.semilogy(idxs, tight, "-o", ms=3, lw=1.1, color=c,
                              label=f"tight  ({dt})")
            ax_ratio.semilogy(idxs, loose, "--", lw=0.9, color=c, alpha=0.6,
                              label=f"loose  ({dt})")
            ax_resid.semilogy(idxs, resid, "-o", ms=3, lw=1.1, color=c,
                              label=dt)

        ax_ratio.set_title(f"factor-growth ratios vs A  [{label}]")
        ax_ratio.set_xlabel("energy index")
        ax_ratio.set_ylabel("ratio to ||A||")
        ax_ratio.grid(True, which="both", ls=":", alpha=0.4)
        ax_ratio.legend(fontsize=8, ncol=2)

        ax_resid.set_title(f"assembly residual ||A-LU||/||A||  [{label}]")
        ax_resid.set_xlabel("energy index")
        ax_resid.set_ylabel("relative residual")
        ax_resid.grid(True, which="both", ls=":", alpha=0.4)
        ax_resid.legend(fontsize=8)

    fig.suptitle(f"Block Thomas backward-stability -- {material}",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5path", type=Path, help="raw material HDF5 file")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--idx", type=int, help="a single energy index")
    grp.add_argument("--start", type=int, help="first index (inclusive, with --end)")
    ap.add_argument("--end", type=int, help="last index (inclusive, with --start)")
    ap.add_argument("--dtype", choices=list(DTYPES), default=None,
                    help="restrict to one precision (default: analyse both if present)")
    ap.add_argument("--plotdir", type=Path, default=DEFAULT_PLOTDIR,
                    help=f"where to write the plot + csv (default: {DEFAULT_PLOTDIR})")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip plot/CSV output (print only)")
    args = ap.parse_args()

    if args.start is not None and args.end is None:
        ap.error("--start requires --end")
    indices = [args.idx] if args.idx is not None else range(args.start, args.end + 1)
    dtypes = (args.dtype,) if args.dtype else DTYPES

    material = args.h5path.stem            # e.g. "graphene"
    records = []

    with h5py.File(args.h5path, "r") as f:
        for idx in indices:
            process_index(f, idx, dtypes, records)

    if args.no_plot:
        return
    if not records:
        print("no metrics collected -- nothing to plot or save")
        return

    args.plotdir.mkdir(parents=True, exist_ok=True)
    stem = f"{material}_blockthomas_stability"
    #write_csv(records, args.plotdir / f"{stem}.csv")
    make_plot(records, material, args.plotdir / f"{stem}.png")


if __name__ == "__main__":
    main()
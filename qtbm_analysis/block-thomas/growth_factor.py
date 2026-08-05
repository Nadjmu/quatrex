#!/usr/bin/env python3
"""
#5. Growth factor / backward-stability analysis of the LU-based solvers

growth_factor.py -- reads a material's HDF5 file and, for every solver whose
explicit factors are stored there, reports how much the factorization grew
relative to the matrix it factored.

(Was blockthomas_growth_factor.py, which only handled Block Thomas. SuperLU
and UMFPACK expose explicit L, U too, so they are analysed by the same code
path here -- the only thing that differs per solver is which matrix the
factors actually reconstruct.)

WHY. All of these are LU-based, so (per Higham) none is unconditionally
backward stable: the perturbation introduced is bounded entrywise by

        |A - L U|  <=  gamma_n * |L| |U|

which, in a monotone norm (1- or inf-norm), gives

        ||A - L U||  <=  gamma_n * || |L| |U| ||.

A factorization is backward stable in practice iff the growth of the factors
relative to A stays modest. That growth is a property of the PIVOTING, and is
INDEPENDENT of cond(A): an ill-conditioned A can factor with tiny growth, and
a well-conditioned A can grow badly if a Schur-complement block becomes badly
scaled. Conditioning governs the accuracy of x; growth governs the backward
error. This is exactly the comparison of interest for Block Thomas, whose
block pivoting is weaker than the global partial pivoting with column
ordering that SuperLU and UMFPACK do -- those two are the reference points
that say whether exploiting the block structure costs stability.

WHAT IS REPORTED, per (index, solver, dtype, norm):

    loose ratio  =  ||L|| * ||U|| / ||A_eff||     (classical, always an upper bound)
    tight ratio  =  || |L| |U| || / ||A_eff||     (true Wilkinson quantity, sharper)
    rho          =  max|U_ij| / max|A_eff_ij|     (pivot growth factor, norm-free)
    resid_rel    =  ||A_eff - L U|| / ||A_eff||   (correctness guard)

A_eff IS NOT ALWAYS A. Each solver's factors reconstruct a different matrix,
and using ||A|| as the denominator regardless would silently misreport the
ratios for UMFPACK. The per-solver conventions:

    blockthomas      A            == L @ U     (no permutation, no scaling)
    blockthomas_inv  A            == L @ U     (same, assembled from D_mod)
    superlu          Pr @ A @ Pc  == L @ U
    umfpack          Pr @ diag(1/R) @ A @ Pc == L @ U

Permutations alone do not change the 1- or inf-norm (a row permutation
reorders rows; a column permutation permutes entries within each row, leaving
every row sum intact), so SuperLU could reuse ||A||. UMFPACK's ROW SCALING
does change it, so A_eff is formed explicitly in all cases rather than
special-cased -- that also keeps the residual guard honest.

Over a sweep (--start/--end) the per-index metrics are accumulated and, unless
--no-plot is given, written to PLOTDIR as:
    <material>_growth_factor.png     ratios + rho + residual vs index
    <material>_growth_factor.csv     the raw per-(index,solver,dtype,norm) numbers

Usage:
    python growth_factor.py /scratch/yimili/matrices/hdf5/graphene.h5 --idx 25
    python growth_factor.py /scratch/.../graphene.h5 --start 1 --end 400
    python growth_factor.py /scratch/.../graphene.h5 --start 1 --end 400 \
        --solvers blockthomas superlu umfpack --dtype complex128

The HDF5 layout assumed is the one factor_io.py writes; see its docstring.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import norm as spnorm

from factor_io import load_sparse_factor, load_blocks
from solver_classes import extract_blocks_sparse, embed_block

DTYPES = ("complex128", "complex64", "complex32")
SOLVERS = ("blockthomas", "blockthomas_inv", "superlu", "umfpack")
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


def _permutation_matrix(perm):
    """perm[i] = j means row/col i maps to position j."""
    n = len(perm)
    return sp.csr_matrix((np.ones(n), (np.arange(n), perm)), shape=(n, n))


# ---------------------------------------------------------------------------
# Block Thomas: assemble the genuine global block-bidiagonal L, U with
# A == L_global @ U_global exactly (verified to machine precision).
# ---------------------------------------------------------------------------
def _piv_to_perm(piv, m):
    perm = list(range(m))
    for i, p in enumerate(piv):
        perm[i], perm[p] = perm[p], perm[i]
    return perm


def _reconstruct_dmod(lu_k, piv_k):
    bs = lu_k.shape[0]
    Lk = np.tril(lu_k, -1) + np.eye(bs, dtype=lu_k.dtype)
    Uk = np.triu(lu_k)
    Pk = np.eye(bs, dtype=lu_k.dtype)[_piv_to_perm(piv_k, bs)]
    return Pk.T @ Lk @ Uk


def _sparse_ok(a):
    """scipy.sparse has no float16, so fp16 factors are promoted to float32.
    The promotion is exact and every quantity reported here is a ratio, so it
    changes nothing -- it only makes the blocks storable as sparse."""
    a = np.asarray(a)
    return a.astype(np.float32) if a.dtype == np.float16 else a


def _bmat_bidiag(diag_blocks, offdiag_blocks, position):
    N = len(diag_blocks)
    grid = [[None] * N for _ in range(N)]
    for k in range(N):
        grid[k][k] = sp.csr_matrix(_sparse_ok(diag_blocks[k]))
    for k in range(N - 1):
        if position == "sub":
            grid[k + 1][k] = sp.csr_matrix(_sparse_ok(offdiag_blocks[k]))
        else:
            grid[k][k + 1] = sp.csr_matrix(_sparse_ok(offdiag_blocks[k]))
    return sp.bmat(grid, format="csr")


def effective_A(group, A):
    """
    The matrix a stored Block Thomas factorization actually factored.

    For the complex variants that is A itself. The fp16 variants never see A:
    they factor the exact real embedding  z = a+bi -> [[a,-b],[b,a]]  applied
    BLOCK BY BLOCK, scaled by a global power of two s. Block-local embedding
    is not the same matrix as embedding A globally -- the two differ by a
    permutation -- so A_eff is rebuilt here the same way the solver built it,
    from the recorded block partition, rather than derived from A wholesale.
    """
    if not group.attrs.get("embedded_real", False):
        return A
    sizes = [int(v) for v in group["block_sizes"][:]]
    s = float(group.attrs["scale_s"])
    D, L, U = extract_blocks_sparse(A, sizes)
    return _bmat_bidiag(
        [embed_block(d) * s for d in D],
        [embed_block(u) * s for u in U], "super"
    ).tolil().tocsc() + _bmat_bidiag(
        [np.zeros((2 * m, 2 * m)) for m in sizes],
        [embed_block(l) * s for l in L], "sub"
    ).tocsc()


def assemble_blockthomas_lu(group, A):
    """
    Implementation 1: rebuild each modified diagonal block from its packed LU
    and pivots, then form the global factors.

        L_global = block-bidiagonal(I;  E_k = L_off[k-1] @ D_mod[k-1]^-1)
        U_global = block-bidiagonal(D_mod;  U_off)

    Works for uniform and ragged partitions alike -- load_blocks() hands back
    a stacked array or a list of per-block arrays, and both index the same way.
    """
    L_off = load_blocks(group, "L")
    U_off = load_blocks(group, "U")
    Dlu = load_blocks(group, "Dmod_lu")
    Dpiv = load_blocks(group, "Dmod_piv")
    N = len(Dlu)
    dtype = np.asarray(Dlu[0]).dtype

    D_mod = [_reconstruct_dmod(np.asarray(Dlu[k]), np.asarray(Dpiv[k]))
             for k in range(N)]

    E = []
    for k in range(1, N):
        bs = np.asarray(Dlu[k - 1]).shape[0]
        Dinv = sla.lu_solve((np.asarray(Dlu[k - 1]), np.asarray(Dpiv[k - 1])),
                            np.eye(bs, dtype=dtype))
        E.append(np.asarray(L_off[k - 1]) @ Dinv)

    I_blocks = [np.eye(np.asarray(D_mod[k]).shape[0], dtype=dtype)
                for k in range(N)]
    L_global = _bmat_bidiag(I_blocks, E, "sub").tocsc()
    U_global = _bmat_bidiag(D_mod, [np.asarray(u) for u in U_off],
                            "super").tocsc()
    return effective_A(group, A), L_global, U_global


def assemble_blockthomas_inv_lu(group, A):
    """
    Implementation 2: the same global L, U exist mathematically, but this
    variant never forms LU factors -- it stores D_mod and its explicit
    inverse, which is exactly what the assembly needs, and more directly than
    Implementation 1 (no triangular solve against packed factors required).
    """
    L_off = load_blocks(group, "L")
    U_off = load_blocks(group, "U")
    D_mod = load_blocks(group, "Dmod")
    D_inv = load_blocks(group, "Dmod_inv")
    N = len(D_mod)
    dtype = np.dtype(np.asarray(D_mod[0]).dtype)

    # fp16 factors are stored scaled: G[k] / t[k] is the inverse of the
    # s-scaled embedded block. The per-block scales t are exact powers of two
    # and cancel out of every ratio reported here, but the assembly still
    # needs them to reproduce A_eff. Undo them in fp32 -- 1/t is routinely
    # subnormal in fp16.
    t = group["inv_scale_t"][:] if "inv_scale_t" in group else np.ones(N)
    if dtype == np.dtype(np.float16):
        dtype = np.dtype(np.float32)

    E = [(_sparse_ok(L_off[k - 1]).astype(dtype)
          @ _sparse_ok(D_inv[k - 1]).astype(dtype)) / t[k - 1]
         for k in range(1, N)]
    I_blocks = [np.eye(np.asarray(D_mod[k]).shape[0], dtype=dtype)
                for k in range(N)]
    L_global = _bmat_bidiag(I_blocks, E, "sub").tocsc()
    U_global = _bmat_bidiag([np.asarray(d) for d in D_mod],
                            [np.asarray(u) for u in U_off], "super").tocsc()
    return effective_A(group, A), L_global, U_global


# ---------------------------------------------------------------------------
# SuperLU / UMFPACK: factors are stored sparse and are already global. Only
# the matrix they reconstruct differs.
# ---------------------------------------------------------------------------
def assemble_superlu(group, A):
    """Pr @ A @ Pc == L @ U, with Pr built from argsort(perm_r)."""
    L = load_sparse_factor(group["L"])
    U = load_sparse_factor(group["U"])
    Pr = _permutation_matrix(np.argsort(group["perm_r"][:]))
    Pc = _permutation_matrix(group["perm_c"][:])
    return (Pr @ A @ Pc).tocsc(), L.tocsc(), U.tocsc()


def assemble_umfpack(group, A):
    """
    Pr @ diag(1/R) @ A @ Pc == L @ U -- UMFPACK additionally row-scales, and
    that scaling DOES change the norms, so it belongs in A_eff. (The
    convention was pinned down empirically; the resid_rel column is the guard
    that it still holds for your UMFPACK build.)
    """
    L = load_sparse_factor(group["L"])
    U = load_sparse_factor(group["U"])
    Pr = _permutation_matrix(np.argsort(group["perm_r"][:]))
    Pc = _permutation_matrix(group["perm_c"][:])
    A_scaled = sp.diags(1.0 / group["R"][:]) @ A if "R" in group else A
    return (Pr @ A_scaled @ Pc).tocsc(), L.tocsc(), U.tocsc()


ASSEMBLERS = {
    "blockthomas":     assemble_blockthomas_lu,
    "blockthomas_inv": assemble_blockthomas_inv_lu,
    "superlu":         assemble_superlu,
    "umfpack":         assemble_umfpack,
}


# ---------------------------------------------------------------------------
# the actual stability metrics
# ---------------------------------------------------------------------------
def _absmax(M):
    M = M.tocoo()
    return float(np.abs(M.data).max()) if M.nnz else 0.0


def analyse(A, L, U):
    """
    For both the 1-norm and inf-norm: ||A||, ||L||, ||U||, ||L||*||U||,
    || |L||U| ||, the loose and tight ratios, and the assembly residual.
    The pivot growth factor rho = max|U| / max|A| is norm-free, so it is
    computed once and repeated in each row for convenient plotting.
    """
    absLU = (abs(L) @ abs(U))          # |L||U|, stays sparse
    R = (A - (L @ U)).tocsr()          # correctness guard
    R.eliminate_zeros()

    a_absmax = _absmax(A)
    rho = (_absmax(U) / a_absmax) if a_absmax else np.inf

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
            rho=rho,
            loose=(nL * nU) / nA if nA else np.inf,
            tight=nLU_abs / nA if nA else np.inf,
            resid_rel=(n_resid / nA) if nA else np.inf,
        )
    return out


def _fmt_block(solver, dtype_name, res):
    lines = [f"    {solver} / {dtype_name}"]
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
            f"                  growth factor rho = max|U|/max|A| = {r['rho']:.3e}"
        )
        lines.append(
            f"                  (assembly check ||A-LU||/||A|| = {r['resid_rel']:.2e})"
        )
    return "\n".join(lines)


def process_index(f, idx, solvers, dtypes, records):
    """Analyse one index across solvers/dtypes; print a report, append rows."""
    print(f"idx = {idx}")
    A = None
    per_key = {}

    for solver in solvers:
        root = f.get(f"E_{idx}/{solver}")
        if root is None:
            print(f"    {solver}: no group at this index -- skipping")
            continue
        for dt in dtypes:
            g = root.get(dt)
            if g is None:
                continue
            if A is None:
                A = load_M(f, idx)
            try:
                # measure in the precision the factors were computed at;
                # fp16 factors are embedded-real, so keep A at full precision
                A_dt = A if dt == "complex32" else A.astype(np.dtype(dt))
                A_eff, L, U = ASSEMBLERS[solver](g, A_dt)
                res = analyse(A_eff, L, U)
            except Exception as exc:            # noqa: BLE001
                print(f"    {solver} / {dt}: FAILED ({type(exc).__name__}: {exc})")
                continue

            per_key[(solver, dt)] = res
            print(_fmt_block(solver, dt, res))
            for label in NORMS:
                r = res[label]
                records.append(dict(idx=idx, solver=solver, dtype=dt, norm=label,
                                    nA=r["nA"], nL=r["nL"], nU=r["nU"],
                                    prod=r["prod"], LU_abs=r["LU_abs"],
                                    rho=r["rho"], loose=r["loose"],
                                    tight=r["tight"], resid_rel=r["resid_rel"]))

    # precision comparison within each solver, as before
    for solver in solvers:
        a = per_key.get((solver, "complex128"))
        b = per_key.get((solver, "complex64"))
        if a and b:
            for label in NORMS:
                t128, t64 = a[label]["tight"], b[label]["tight"]
                factor = t64 / t128 if t128 else np.inf
                print(f"    {solver} [{label:8s}] tight-ratio change "
                      f"c64/c128 = {factor:.10f}x")
    print()


# ---------------------------------------------------------------------------
# output: CSV + plots
# ---------------------------------------------------------------------------
def write_csv(records, csv_path):
    fields = ["idx", "solver", "dtype", "norm", "nA", "nL", "nU", "prod",
              "LU_abs", "rho", "loose", "tight", "resid_rel"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in records:
            w.writerow(row)
    print(f"wrote {csv_path}  ({len(records)} rows)")


SOLVER_COLOR = {
    "blockthomas":     "#2E86AB",
    "blockthomas_inv": "#8E44AD",
    "superlu":         "#555555",
    "umfpack":         "#E67E22",
}
DTYPE_LS = {"complex128": "-", "complex64": "--", "complex32": ":"}


def make_plot(records, material, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        print("no records to plot")
        return

    combos = sorted({(r["solver"], r["dtype"]) for r in records})

    # one row per norm: tight/loose ratio | growth factor rho | residual
    fig, axes = plt.subplots(len(NORMS), 3, figsize=(18, 4.2 * len(NORMS)),
                             squeeze=False)

    for row_i, label in enumerate(NORMS):
        ax_ratio, ax_rho, ax_resid = axes[row_i]

        for solver, dt in combos:
            rows = sorted((r for r in records if r["norm"] == label
                           and r["solver"] == solver and r["dtype"] == dt),
                          key=lambda r: r["idx"])
            if not rows:
                continue
            idxs = [r["idx"] for r in rows]
            c = SOLVER_COLOR.get(solver, None)
            ls = DTYPE_LS.get(dt, "-")
            tag = f"{solver} ({dt})"

            ax_ratio.semilogy(idxs, [r["tight"] for r in rows], ls, marker=".",
                              ms=3, lw=1.1, color=c, label=f"tight  {tag}")
            ax_ratio.semilogy(idxs, [r["loose"] for r in rows], ls, lw=0.9,
                              color=c, alpha=0.45, label=f"loose  {tag}")
            ax_rho.semilogy(idxs, [r["rho"] for r in rows], ls, marker=".",
                            ms=3, lw=1.1, color=c, label=tag)
            ax_resid.semilogy(idxs, [r["resid_rel"] for r in rows], ls,
                              marker=".", ms=3, lw=1.1, color=c, label=tag)

        ax_ratio.set_title(f"factor-growth ratios vs A  [{label}]")
        ax_ratio.set_ylabel("ratio to ||A||")
        ax_rho.set_title(r"pivot growth factor  $\rho = \max|U| / \max|A|$")
        ax_rho.set_ylabel(r"$\rho$")
        ax_resid.set_title(f"assembly residual ||A-LU||/||A||  [{label}]")
        ax_resid.set_ylabel("relative residual")

        for ax in (ax_ratio, ax_rho, ax_resid):
            ax.set_xlabel("energy index")
            ax.grid(True, which="both", ls=":", alpha=0.4)
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"LU backward-stability / growth factor -- {material}",
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
    ap.add_argument("--solvers", nargs="+", choices=list(SOLVERS),
                    default=list(SOLVERS),
                    help="which solvers' stored factors to analyse "
                         "(default: all that are present)")
    ap.add_argument("--dtype", choices=list(DTYPES), default=None,
                    help="restrict to one precision (default: all present)")
    ap.add_argument("--plotdir", type=Path, default=DEFAULT_PLOTDIR,
                    help=f"where to write the plot + csv (default: {DEFAULT_PLOTDIR})")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip plot/CSV output (print only)")
    ap.add_argument("--no-csv", action="store_true", help="skip the CSV")
    args = ap.parse_args()

    if args.start is not None and args.end is None:
        ap.error("--start requires --end")
    indices = [args.idx] if args.idx is not None else range(args.start, args.end + 1)
    dtypes = (args.dtype,) if args.dtype else DTYPES

    material = args.h5path.stem            # e.g. "graphene"
    records = []

    with h5py.File(args.h5path, "r") as f:
        for idx in indices:
            process_index(f, idx, args.solvers, dtypes, records)

    if args.no_plot:
        return
    if not records:
        print("no metrics collected -- nothing to plot or save")
        return

    args.plotdir.mkdir(parents=True, exist_ok=True)
    stem = f"{material}_growth_factor"
    if not args.no_csv:
        write_csv(records, args.plotdir / f"{stem}.csv")
    make_plot(records, material, args.plotdir / f"{stem}.png")


if __name__ == "__main__":
    main()

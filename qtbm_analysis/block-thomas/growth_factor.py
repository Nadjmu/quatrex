#!/usr/bin/env python3
"""
Factor growth and backward stability of the stored LU factorizations.

Input
-----
A material HDF5 file whose solver groups were written by
``solvers/factor_io.py``. For every energy index requested and every solver
that stores explicit factors, the following are read:

    E_<idx>/M                             the system matrix A
    E_<idx>/blockthomas/<dtype>/          L, U, Dmod_lu, Dmod_piv, block_sizes
    E_<idx>/blockthomas_inv/<dtype>/      L, U, Dmod, Dmod_inv, inv_scale_t
    E_<idx>/superlu/<dtype>/              L, U, perm_r, perm_c
    E_<idx>/umfpack/<dtype>/              L, U, perm_r, perm_c, R

The file is opened read-only. Solvers or precisions absent from the file are
reported and skipped.

Motivation
----------
All four solvers are LU based, so by Higham none is unconditionally backward
stable. The computed factors satisfy the entrywise bound

    |A - L U|  <=  gamma_n |L| |U|,        gamma_n = n u / (1 - n u),

which in any monotone norm gives ||A - L U|| <= gamma_n || |L| |U| ||. A
factorization is therefore backward stable in practice precisely when the
factors do not grow relative to A.

That growth is a property of the pivoting and is independent of kappa(A): an
ill-conditioned A may factor with negligible growth, and a well-conditioned A
may grow badly if a Schur complement becomes badly scaled. Conditioning bounds
the forward error of x; growth bounds the backward error. The two must not be
conflated.

Block Thomas pivots only within each diagonal block, which is weaker than the
global partial pivoting with column ordering performed by SuperLU and UMFPACK.
Those two are included as reference points that quantify what, if anything,
exploiting the block structure costs in stability.

Algorithm
---------
Per (index, solver, dtype):

1. Assemble the global factors L and U from what that solver stored, and form
   A_eff, the matrix those factors reconstruct (see below).
2. Report, for both the 1-norm and the infinity norm,

       loose ratio  ||L|| ||U|| / ||A_eff||        classical, an upper bound
       tight ratio  || |L| |U| || / ||A_eff||      the quantity in the bound
       rho          max|U_ij| / max|A_eff_ij|      pivot growth factor
       resid_rel    ||A_eff - L U|| / ||A_eff||    reconstruction guard

resid_rel is not a stability metric. It verifies that the assumed factor
convention holds for the build that produced the file; if it is not near the
unit roundoff of the stored precision, the other three columns are meaningless.

A_eff is solver dependent
-------------------------
Each solver's factors reconstruct a different matrix, and using ||A|| as the
denominator throughout would misreport UMFPACK:

    block-thomas       A                        == L U
    block-thomas-inv   A                        == L U
    superlu            Pr A Pc                  == L U
    umfpack            Pr diag(1/R) A Pc        == L U

A permutation alone leaves the 1-norm and the infinity norm unchanged: a row
permutation reorders rows, and a column permutation permutes entries within
each row, so every row sum is preserved. UMFPACK's row scaling does change
them. A_eff is therefore built explicitly in all cases rather than
special-cased, which also keeps resid_rel meaningful.

For the half-precision groups A_eff is neither A nor a permutation of it: those
factors were computed from the real embedding of A applied block by block and
scaled by a global power of two s. Block-local embedding differs from embedding
A globally by a permutation, so effective_A() rebuilds the matrix exactly as
the solver did, from the recorded partition.

Output
------
    <outdir>/<material>.h5, group growth_factor

one row per (index, solver, dtype, norm), plus a per-index report on stdout.
The file is opened in append mode and only that group is rewritten, so results
of other analyses of the same material are preserved. No figures are produced;
see plotting/plot_growth_factor.py, which consumes the group.

Usage
-----
    python growth_factor.py /scratch/yimili/matrices2/hdf5/graphene.h5 --idx 25
    python growth_factor.py .../graphene.h5 --start 1 --end 400
    python growth_factor.py .../graphene.h5 --start 1 --end 400 \
        --solvers block-thomas superlu umfpack --dtypes complex128
    python ../plotting/plot_growth_factor.py \
        /scratch/yimili/error-analysis-block-thomas/graphene.h5
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import norm as spnorm

import cli
from cli import COMPLEX_DTYPES as DTYPES, FACTOR_SOLVERS as SOLVERS
from factor_io import (load_sparse_factor, load_blocks, save_table,
                       material_metadata)
from solver_classes import extract_blocks_sparse, embed_block

NORMS = ("1-norm", "inf-norm")
DEFAULT_OUTDIR = cli.BLOCK_THOMAS_DIR

# Top-level group of the analysis file this script writes.
GROUP = "growth_factor"
COLUMNS = ["idx", "solver", "dtype", "norm", "nA", "nL", "nU", "prod",
           "LU_abs", "rho", "loose", "tight", "resid_rel"]


# ---------------------------------------------------------------------------
# the original matrix, stored as a CSC triplet
# ---------------------------------------------------------------------------
def load_M(f, idx):
    g = f[f"E_{idx}/M"]
    n = len(g["indptr"]) - 1
    return sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                         shape=(n, n))


def _permutation_matrix(perm):
    """Permutation matrix P with perm[i] = j meaning row or column i maps to j."""
    n = len(perm)
    return sp.csr_matrix((np.ones(n), (np.arange(n), perm)), shape=(n, n))


# ---------------------------------------------------------------------------
# Block Thomas: assembly of the global block-bidiagonal factors satisfying
# A == L_global @ U_global exactly.
# ---------------------------------------------------------------------------
def _piv_to_perm(piv, m):
    """LAPACK sequential row-interchange vector to an explicit permutation."""
    perm = list(range(m))
    for i, p in enumerate(piv):
        perm[i], perm[p] = perm[p], perm[i]
    return perm


def _reconstruct_dmod(lu_k, piv_k):
    """Modified diagonal block D_mod[k] = P^T L U from its packed LU factors."""
    bs = lu_k.shape[0]
    Lk = np.tril(lu_k, -1) + np.eye(bs, dtype=lu_k.dtype)
    Uk = np.triu(lu_k)
    Pk = np.eye(bs, dtype=lu_k.dtype)[_piv_to_perm(piv_k, bs)]
    return Pk.T @ Lk @ Uk


def _sparse_ok(a):
    """
    Promote float16 blocks to float32, which scipy.sparse supports.

    The promotion is exact, and every quantity reported by this module is a
    ratio, so it does not affect any result. It only makes the blocks storable
    in a sparse container.
    """
    a = np.asarray(a)
    return a.astype(np.float32) if a.dtype == np.float16 else a


def _bmat_bidiag(diag_blocks, offdiag_blocks, position):
    """
    Block-bidiagonal sparse matrix from a diagonal and one off-diagonal.

    position is "sub" for the first subdiagonal and "super" for the first
    superdiagonal.
    """
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
    The matrix that a stored Block Thomas factorization was computed from.

    For the complex variants this is A itself. The half-precision variants
    never see A: they factor the exact real embedding z = a + bi -> [[a, -b],
    [b, a]] applied block by block and scaled by a global power of two s.
    Embedding block by block and embedding A globally differ by a permutation,
    so A_eff is rebuilt here exactly as the solver built it, from the recorded
    partition, rather than derived from A wholesale.
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
    Global factors of an implementation 1 factorization.

    Each modified diagonal block is rebuilt from its packed LU and pivots, then
    the global block-bidiagonal factors are formed as

        L_global = block-bidiagonal(I;      E_k = L_off[k-1] D_mod[k-1]^-1)
        U_global = block-bidiagonal(D_mod;  U_off)

    Uniform and ragged partitions are handled identically: load_blocks returns
    a stacked array in the first case and a list of per-block arrays in the
    second, and both are indexed the same way.

    Returns (A_eff, L_global, U_global).
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
    Global factors of an implementation 2 factorization.

    The same global L and U exist mathematically, but this variant never forms
    LU factors. It stores D_mod and its explicit inverse, which is precisely
    what the assembly requires, and more directly than implementation 1: no
    triangular solve against packed factors is needed.

    Returns (A_eff, L_global, U_global).
    """
    L_off = load_blocks(group, "L")
    U_off = load_blocks(group, "U")
    D_mod = load_blocks(group, "Dmod")
    D_inv = load_blocks(group, "Dmod_inv")
    N = len(D_mod)
    dtype = np.dtype(np.asarray(D_mod[0]).dtype)

    # Half-precision inverses are stored scaled: G[k] / t[k] is the inverse of
    # the s-scaled embedded block. The per-block scales t are exact powers of
    # two and cancel out of every ratio reported here, but the assembly needs
    # them to reproduce A_eff. They are undone in fp32 because 1/t is
    # frequently subnormal in fp16.
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
    """
    Pr A Pc == L U, with Pr built from argsort(perm_r).

    Returns (A_eff, L, U).
    """
    L = load_sparse_factor(group["L"])
    U = load_sparse_factor(group["U"])
    Pr = _permutation_matrix(np.argsort(group["perm_r"][:]))
    Pc = _permutation_matrix(group["perm_c"][:])
    return (Pr @ A @ Pc).tocsc(), L.tocsc(), U.tocsc()


def assemble_umfpack(group, A):
    """
    Pr diag(1/R) A Pc == L U, with Pr built from argsort(perm_r).

    UMFPACK additionally scales the rows, and that scaling does change the
    norms, so it belongs in A_eff. The convention was determined empirically;
    the resid_rel column is the guard that it still holds for the UMFPACK build
    that produced the file.

    Returns (A_eff, L, U).
    """
    L = load_sparse_factor(group["L"])
    U = load_sparse_factor(group["U"])
    Pr = _permutation_matrix(np.argsort(group["perm_r"][:]))
    Pc = _permutation_matrix(group["perm_c"][:])
    A_scaled = sp.diags(1.0 / group["R"][:]) @ A if "R" in group else A
    return (Pr @ A_scaled @ Pc).tocsc(), L.tocsc(), U.tocsc()


# Keyed by canonical solver name; see solvers/cli.py.
ASSEMBLERS = {
    "block-thomas":     assemble_blockthomas_lu,
    "block-thomas-inv": assemble_blockthomas_inv_lu,
    "superlu":          assemble_superlu,
    "umfpack":          assemble_umfpack,
}


# ---------------------------------------------------------------------------
# stability metrics
# ---------------------------------------------------------------------------
def _absmax(M):
    """Largest absolute stored entry of a sparse matrix, 0 if it has none."""
    M = M.tocoo()
    return float(np.abs(M.data).max()) if M.nnz else 0.0


def analyse(A, L, U):
    """
    Growth and residual metrics of one factorization.

    Computed for both the 1-norm and the infinity norm: ||A||, ||L||, ||U||,
    the product ||L|| ||U||, the norm of |L| |U|, the loose and tight ratios,
    and the reconstruction residual. The pivot growth factor
    rho = max|U| / max|A| is norm-free; it is computed once and repeated in
    both entries so that the caller can emit uniform rows.

    Returns dict keyed by norm label.
    """
    absLU = (abs(L) @ abs(U))          # |L| |U|, stays sparse
    R = (A - (L @ U)).tocsr()          # reconstruction guard
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
    """Human-readable report of one analyse() result."""
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
    """
    Analyse one energy index across the requested solvers and precisions.

    Prints a per-combination report and appends one record per (solver, dtype,
    norm) to `records` in place. Combinations absent from the file are skipped;
    a failure to assemble or analyse one combination is reported and does not
    abort the sweep.
    """
    print(f"idx = {idx}")
    A = None
    per_key = {}

    for solver in solvers:
        root = f.get(f"E_{idx}/{cli.h5_group(solver)}")
        if root is None:
            print(f"    {solver}: no group at this index, skipping")
            continue
        for dt in dtypes:
            g = root.get(dt)
            if g is None:
                continue
            if A is None:
                A = load_M(f, idx)
            try:
                # Measure in the precision the factors were computed at. The
                # half-precision factors are embedded-real, so A is kept at
                # full precision and effective_A performs the embedding.
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

    # Ratio of the tight growth ratio at single against double precision. The
    # ratio is precision-independent in exact arithmetic, so a value far from
    # unity indicates that rounding, not the pivoting strategy, dominates.
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
# output
# ---------------------------------------------------------------------------
def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap)
    cli.add_index_selection(ap, default_all=False)
    cli.add_solver_selection(
        ap, choices=SOLVERS, default=SOLVERS,
        help="solvers whose stored factors to analyse; those absent from the "
             "file are skipped")
    cli.add_dtypes(ap, choices=DTYPES, default=DTYPES,
                   help="precisions to analyse; those absent are skipped")
    cli.add_output(ap, material=True, outdir_default=str(DEFAULT_OUTDIR),
                   outdir_help=f"directory holding the analysis file "
                               f"<material>.h5 (default: {DEFAULT_OUTDIR})")
    ap.add_argument("--no-save", action="store_true",
                    help="print the per-index report only, write no HDF5")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    indices = cli.resolve_indices(ap, args)
    material = args.material or h5path.stem
    records = []

    with h5py.File(h5path, "r") as f:
        for idx in indices:
            process_index(f, idx, args.solvers, args.dtypes, records)

    if args.no_save:
        return
    if not records:
        print("no metrics collected; nothing to write")
        return

    out_path = cli.analysis_h5(args.outdir, material)
    save_table(out_path, GROUP, records, columns=COLUMNS,
               attrs=dict(material=material, source=str(h5path),
                          solvers=list(args.solvers), dtypes=list(args.dtypes),
                          norms=list(NORMS),
                          **material_metadata(h5path)))
    print(f"wrote {out_path}:/{GROUP}  ({len(records)} rows)")
    print(f"Plot with: python ../plotting/plot_growth_factor.py {out_path}")


if __name__ == "__main__":
    main()

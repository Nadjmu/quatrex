#!/usr/bin/env python3
"""
Extract the global L and U factors of a stored factorization.

Input
-----
A material HDF5 file whose solver groups were written by
``solvers/factor_io.py``, the same input ``growth_factor.py`` takes. For each
energy index, solver and precision requested, the stored factors are read from

    E_<idx>/M                             the system matrix A
    E_<idx>/blockthomas/<dtype>/          L, U, Dmod_lu, Dmod_piv, block_sizes
    E_<idx>/blockthomas_inv/<dtype>/      L, U, Dmod, Dmod_inv, inv_scale_t
    E_<idx>/superlu/<dtype>/              L, U, perm_r, perm_c
    E_<idx>/umfpack/<dtype>/              L, U, perm_r, perm_c, R

and the global factors are assembled by ``growth_factor.ASSEMBLERS``, the same
assembly the growth analysis measures. Nothing is recomputed and no
factorization is performed; the file is opened read-only.

What is written
---------------
Per combination, three sparse matrices: the assembled global L and U, and
A_eff, the matrix those factors reconstruct. A_eff is solver dependent --
A for the two Block Thomas variants, Pr A Pc for SuperLU, Pr diag(1/R) A Pc for
UMFPACK -- so it is stored alongside rather than left to be rebuilt from A; see
the header of ``growth_factor.py``. resid_rel = ||A_eff - L U|| / ||A_eff|| is
recorded as an attribute of each combination and is the guard that the assembly
convention held: at a value far above the unit roundoff of the stored
precision, the factors written are not the ones the solver computed.

Both Block Thomas implementations reconstruct genuine global block-bidiagonal
factors with A == L_global U_global exactly,

    L_global = block-bidiagonal(I;      E_k = L_off[k-1] D_mod[k-1]^-1)
    U_global = block-bidiagonal(D_mod;  U_off)

so the extracted L and U are directly comparable with SuperLU's and UMFPACK's,
which are global to begin with.

Output
------
    <outdir>/<material>.h5, group lu_factors

        lu_factors/E_<idx>/<solver>/<dtype>/{A_eff,L,U}   CSC triplets

with <solver> the stored group name of the material file. The file is opened in
append mode and only the combinations extracted in this run are replaced, so
neither the other groups of an analysis file nor previously extracted indices
are disturbed. That differs from every other script in this directory, which
rewrites its group wholesale: an extraction is per index and a run is normally
one index, so wholesale replacement would discard the extractions before it.

Figures are produced by plotting/block-thomas/plot_lu_factors.py.

Usage
-----
    python extract_lu.py /scratch/yimili/matrices2/hdf5/carbon-nanotube.h5 --idx 5
    python extract_lu.py .../graphene.h5 --idx 25 \
        --solvers block-thomas superlu umfpack --dtypes complex128
    python extract_lu.py .../graphene.h5 --start 10 --end 20 --stride 5

The factors of a large material are far denser than A and every combination is
stored in full, so a whole sweep is not a sensible selection: the intended use
is a handful of indices. With no index selection every index in the file is
extracted, and the run is reported index by index.

    python ../plotting/block-thomas/plot_lu_factors.py \
        /scratch/yimili/error-analysis-block-thomas/graphene.h5 --idx 5
"""

import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm as spnorm

import cli
from cli import COMPLEX_DTYPES as DTYPES, FACTOR_SOLVERS as SOLVERS
# _save_sparse_factor is the writer every stored factor in the project already
# uses; the extracted factors are written with it so that they load back with
# the public load_sparse_factor, exactly like the material file's own.
from factor_io import _save_sparse_factor as save_sparse_factor
from factor_io import material_metadata
from growth_factor import ASSEMBLERS, load_M

DEFAULT_OUTDIR = cli.BLOCK_THOMAS_DIR

# Top-level group of the analysis file this script writes.
GROUP = "lu_factors"


def extract_one(f, idx, solver, dtype_name, out):
    """
    Assemble and store one (index, solver, precision) combination.

    Returns True if the combination was present in the material file and was
    written. `out` is the open output file.
    """
    root = f.get(f"E_{idx}/{cli.h5_group(solver)}")
    group = None if root is None else root.get(dtype_name)
    if group is None:
        print(f"    {solver} / {dtype_name}: absent, skipping")
        return False

    A = load_M(f, idx)
    # Measure in the precision the factors were computed at. The
    # half-precision factors are embedded-real, so A is kept at full precision
    # and the assembler performs the embedding itself.
    A_dt = A if dtype_name == "complex32" else A.astype(np.dtype(dtype_name))
    try:
        A_eff, L, U = ASSEMBLERS[solver](group, A_dt)
    except Exception as exc:                    # noqa: BLE001
        print(f"    {solver} / {dtype_name}: FAILED "
              f"({type(exc).__name__}: {exc})")
        return False

    nA = spnorm(A_eff, np.inf)
    residual = (A_eff - (L @ U)).tocsr()
    residual.eliminate_zeros()
    resid_rel = (spnorm(residual, np.inf) / nA) if nA else np.inf

    path = f"{GROUP}/E_{idx}/{cli.h5_group(solver)}/{dtype_name}"
    if path in out:
        del out[path]
    g = out.require_group(path)
    save_sparse_factor(g, "A_eff", A_eff)
    save_sparse_factor(g, "L", L)
    save_sparse_factor(g, "U", U)
    g.attrs["idx"] = int(idx)
    g.attrs["solver"] = solver
    g.attrs["dtype"] = dtype_name
    g.attrs["n"] = int(A_eff.shape[0])
    g.attrs["resid_rel"] = float(resid_rel)
    # The partition the solver used, where it recorded one. It is what a figure
    # needs to draw the block boundaries the factors follow.
    if "block_sizes" in group:
        g.attrs["block_sizes"] = np.asarray(group["block_sizes"][:],
                                            dtype=np.int64)

    print(f"    {solver} / {dtype_name}: n = {A_eff.shape[0]}, "
          f"nnz(L) = {L.nnz}, nnz(U) = {U.nnz}, "
          f"nnz(A_eff) = {A_eff.nnz}, "
          f"||A_eff - LU||/||A_eff|| = {resid_rel:.2e}")
    return True


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap)
    cli.add_index_selection(ap, default_all=True)
    cli.add_solver_selection(
        ap, choices=SOLVERS, default=SOLVERS,
        help="solvers whose stored factors to extract; those absent from the "
             "file are skipped")
    cli.add_dtypes(ap, choices=DTYPES, default=DTYPES,
                   help="precisions to extract; those absent are skipped")
    cli.add_output(ap, material=True, outdir_default=str(DEFAULT_OUTDIR),
                   outdir_help=f"directory holding the analysis file "
                               f"<material>.h5 (default: {DEFAULT_OUTDIR})")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    material = args.material or h5path.stem
    out_path = cli.analysis_h5(args.outdir, material)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with h5py.File(h5path, "r") as f, h5py.File(out_path, "a") as out:
        indices = cli.resolve_indices(ap, args, cli.available_indices(f))
        root = out.require_group(GROUP)
        for key, value in dict(material=material, source=str(h5path),
                               **material_metadata(h5path)).items():
            root.attrs[key] = value
        for idx in indices:
            print(f"idx = {idx}")
            for solver in args.solvers:
                for dtype_name in args.dtypes:
                    written += extract_one(f, idx, solver, dtype_name, out)

    if not written:
        print("nothing extracted; no combination requested is in the file")
        return
    print(f"\nwrote {out_path}:/{GROUP}  ({written} factorizations)")
    print(f"Plot with: python ../plotting/block-thomas/plot_lu_factors.py "
          f"{out_path}")


if __name__ == "__main__":
    main()

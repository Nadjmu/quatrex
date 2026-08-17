#!/usr/bin/env python3
"""
Condition number of the QTBM system matrices in three norms, by norm estimation
and by extreme singular values, over an energy sweep.

Input
-----
A material HDF5 file providing E_<idx>/M as a CSC triplet, or, with no path
given, every material in --materials taken from the standard scratch layout.
--idx or --start/--end select the energy indices; the default is every index
present, thinned by --stride.

Algorithm
---------
One sparse LU factorization of M is computed per energy index and reused for
all three estimates, since each of them needs nothing from M^-1 but the ability
to apply it.

    kappa_1(M)    = ||M||_1    * ||M^-1||_1
    kappa_inf(M)  = ||M||_inf  * ||M^-1||_inf
    kappa_2(M)    = sigma_max / sigma_min

The two matrix norms are exact and taken from the sparse M directly. The two
inverse norms are estimated by the Higham-Tisseur algorithm (onenormest), which
needs only products with M^-1 and M^-H and returns a lower bound, in practice
within a small factor of the true norm. The infinity norm is reached through

    ||M^-1||_inf = ||M^-H||_1,

so that both estimates are one-norm estimates of an operator the same
factorization supplies; forming M^-T explicitly is never necessary.

kappa_2 needs the two ends of the singular spectrum. sigma_max is the
well-conditioned end and is taken by svds on M itself, without a
factorization. sigma_min is reached as the LARGEST singular value of M^-1,
reciprocated: M = U S V^H implies M^-1 = V S^-1 U^H, so sigma_min(M) =
1 / sigma_max(M^-1). This is the same shift-invert argument as in
block-thomas/arnoldi_shift_invert_cpu.py, which does it for a single matrix
with a choice of backends; here the backend is fixed to SuperLU because the
same factorization is already required by the two norm estimates.

Note that kappa_1 and kappa_inf are estimates from below of an exactly defined
quantity, while kappa_2 is a converged computation of both singular values;
the three differ by more than the choice of norm and are reported separately
rather than reduced to one number.

Each row is written as soon as it is produced, so a run may be interrupted and
continued with --resume; rows already marked valid are not recomputed.

Output
------
    <outdir>/<material>.h5, group condition

    indices          (P,)  energy index of each row
    valid            (P,)  bool, row fully computed
    nnz              (P,)  nnz of M, -1 if unknown
    norm1            (P,)  ||M||_1, exact
    norminf          (P,)  ||M||_inf, exact
    norm1_inv        (P,)  ||M^-1||_1, estimated
    norminf_inv      (P,)  ||M^-1||_inf, estimated
    sigma_max        (P,)  largest singular value
    sigma_min        (P,)  smallest singular value
    cond_1           (P,)  norm1 * norm1_inv
    cond_inf         (P,)  norminf * norminf_inv
    cond_2           (P,)  sigma_max / sigma_min
    seconds          (P,)  wall time of the row

The row dimension P is fixed by the index selection when the group is created;
a later run over a different selection requires --overwrite. The file is opened
in append mode and only this group is written, so results of other analyses of
the same material are preserved.

Usage
-----
    python condition_est.py
    python condition_est.py --stride 10
    python condition_est.py /scratch/yimili/matrices2/hdf5/graphene.h5 \\
        --start 0 --end 400 --resume
    python ../plotting/plot_condition.py
"""

import gc
import os
import re
import sys
import time
import warnings
from pathlib import Path


# ============================================================
# Parse threads early, before importing NumPy/SciPy
# ============================================================

def parse_threads_from_argv(default=8):
    if "--threads" in sys.argv:
        pos = sys.argv.index("--threads")
        if pos + 1 < len(sys.argv):
            return int(sys.argv[pos + 1])
    return default


THREADS = parse_threads_from_argv(default=8)

os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
os.environ["MKL_NUM_THREADS"] = str(THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)


# ============================================================
# Imports after thread environment variables
# ============================================================

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, norm, onenormest, splu, svds

sys.path.insert(0, str((Path(__file__).resolve().parent / ".."
                        / "solvers").resolve()))
import cli
from factor_io import material_metadata


GROUP = "condition"

# All (P,) float64, written one row at a time.
SCALAR_DATASETS = (
    "norm1",
    "norminf",
    "norm1_inv",
    "norminf_inv",
    "sigma_max",
    "sigma_min",
    "cond_1",
    "cond_inf",
    "cond_2",
    "seconds",
)

DEFAULT_MATERIALS = ("carbon-nanotube", "carbon-chain", "si-bulk", "graphene")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = cli.new_parser(__doc__)

    cli.add_h5_input(
        parser, required=False,
        help="material HDF5 file. With no path given, --materials selects the "
             "files from the standard scratch layout")
    cli.add_index_selection(parser)
    cli.add_output(
        parser,
        outdir_default=str(cli.CONDITION_DIR),
        outdir_help=f"directory holding the analysis file <material>.h5 "
                    f"(default: {cli.CONDITION_DIR})")

    parser.add_argument(
        "--materials",
        nargs="+",
        default=list(DEFAULT_MATERIALS),
        metavar="NAME",
        help=f"materials to process when no path is given "
             f"(default: {' '.join(DEFAULT_MATERIALS)})",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="keep every Nth selected index. The sweep is dense and one row "
             "costs a factorization and two singular-value computations, so a "
             "stride is the cheap way to a first pass. Default: 1.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="BLAS/LAPACK threads. Default: 8.",
    )

    parser.add_argument(
        "--onenorm-t",
        type=int,
        default=2,
        metavar="T",
        help="column count of the norm estimator; larger values cost more "
             "solves and tighten the lower bound. Default: 2, the value "
             "Higham and Tisseur recommend.",
    )

    parser.add_argument(
        "--svd-tol",
        type=float,
        default=1e-10,
        help="convergence tolerance of svds; 0 means machine precision. "
             "Default: 1e-10.",
    )

    parser.add_argument(
        "--svd-ncv",
        type=int,
        default=None,
        metavar="N",
        help="ARPACK subspace size for svds (default: scipy's own). Raise it "
             "if a clustered spectrum stalls convergence.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep an existing group and skip rows already marked valid.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing group and recompute every row.",
    )

    return parser, parser.parse_args()


# ============================================================
# Input
# ============================================================

def discover_indices(h5_file):
    pattern = re.compile(r"^E_(\d+)$")
    found = []

    for name in h5_file.keys():
        match = pattern.fullmatch(name)

        if match is None:
            continue

        if not isinstance(h5_file[name], h5py.Group):
            continue

        if "M" in h5_file[name]:
            found.append(int(match.group(1)))

    return sorted(found)


def load_csc_matrix(h5_file, group_name):
    """M of one energy index, stored as a CSC triplet."""
    grp = h5_file[f"{group_name}/M"]

    data = grp["data"][...]
    row_indices = grp["indices"][...]
    column_pointer = grp["indptr"][...]

    shape = None

    if "shape" in grp.attrs:
        shape = tuple(
            int(value)
            for value in np.asarray(grp.attrs["shape"]).ravel()
        )

    elif "shape" in grp:
        shape = tuple(
            int(value)
            for value in np.asarray(grp["shape"][...]).ravel()
        )

    if shape is None:
        n_cols = len(column_pointer) - 1
        shape = (n_cols, n_cols)
        print(f"[warning] {group_name}: no shape stored; assuming {shape}")

    if len(shape) != 2:
        raise ValueError(f"{group_name}: invalid matrix shape: {shape}")

    return sp.csc_matrix((data, row_indices, column_pointer), shape=shape)


# ============================================================
# Estimates
# ============================================================

def inverse_operators(lu, n, dtype):
    """
    M^-1 and M^-H as linear operators over one LU factorization.

    SciPy takes rmatvec to be the ADJOINT, not the transpose, and onenormest
    forms A.H from it; a transpose solve in that slot silently estimates the
    norm of the wrong operator whenever M is complex. Both operators are
    therefore built from the trans="N" and trans="H" solves of the same
    factorization, which are exact adjoints of one another.

    M^-H is what carries the infinity norm: ||M^-1||_inf = ||M^-H||_1, so both
    inverse norms are one-norm estimates and neither requires M^-T.
    """
    apply_inv = lambda x: lu.solve(np.asarray(x, dtype=dtype), trans="N")
    apply_inv_H = lambda x: lu.solve(np.asarray(x, dtype=dtype), trans="H")

    inv = LinearOperator((n, n), matvec=apply_inv, rmatvec=apply_inv_H,
                         dtype=dtype)
    inv_H = LinearOperator((n, n), matvec=apply_inv_H, rmatvec=apply_inv,
                           dtype=dtype)
    return inv, inv_H


def singular_extremes(M, inv, tol, ncv):
    """
    (sigma_max, sigma_min) of M.

    sigma_max is the dominant singular value and comes from M itself. sigma_min
    is the dominant singular value of M^-1, reciprocated: the large end is the
    well-conditioned end for a Krylov method, whereas which="SM" on M reaches
    the small end through M^H M and therefore through kappa(M)^2.
    """
    kwargs = dict(k=1, which="LM", tol=tol, return_singular_vectors=False)
    if ncv is not None:
        kwargs["ncv"] = ncv

    sigma_max = float(np.max(svds(M, **kwargs)))
    sigma_max_inv = float(np.max(svds(inv, **kwargs)))

    if sigma_max_inv <= 0:
        raise ValueError("M^-1 has a zero singular value; M is numerically "
                         "singular and sigma_min is not recoverable this way")

    return sigma_max, 1.0 / sigma_max_inv


def condition_estimates(M, onenorm_t, svd_tol, svd_ncv):
    """
    The three condition numbers of one matrix, as a dict of the datasets in
    SCALAR_DATASETS other than `seconds`.
    """
    M = M.tocsc().astype(np.complex128)
    n = M.shape[0]

    lu = splu(M)
    inv, inv_H = inverse_operators(lu, n, np.complex128)

    norm1 = float(norm(M, 1))
    norminf = float(norm(M, np.inf))

    norm1_inv = float(onenormest(inv, t=onenorm_t))
    norminf_inv = float(onenormest(inv_H, t=onenorm_t))

    sigma_max, sigma_min = singular_extremes(M, inv, svd_tol, svd_ncv)

    return dict(
        norm1=norm1,
        norminf=norminf,
        norm1_inv=norm1_inv,
        norminf_inv=norminf_inv,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        cond_1=norm1 * norm1_inv,
        cond_inf=norminf * norminf_inv,
        cond_2=sigma_max / sigma_min,
    )


# ============================================================
# Output
# ============================================================

def open_group(h5_file, indices, resume, overwrite, attrs):
    """
    Create or reopen the output group.

    The row dimension is fixed by the index selection, so an existing group is
    only reusable for the same selection; a mismatch is an error rather than a
    silent partial overwrite.
    """
    if GROUP in h5_file and overwrite:
        del h5_file[GROUP]

    if GROUP in h5_file:
        if not resume:
            raise FileExistsError(
                f"group '{GROUP}' already exists in {h5_file.filename}. "
                "Use --resume or --overwrite."
            )
        group = h5_file[GROUP]
        existing = group["indices"][:]
        if not np.array_equal(existing, np.asarray(indices, dtype=np.int64)):
            raise ValueError(
                "The existing group was written for a different index "
                "selection. Use --overwrite or a different --outdir."
            )
        return group

    group = h5_file.create_group(GROUP)
    for key, value in attrs.items():
        group.attrs[key] = value
    group.create_dataset("indices", data=np.asarray(indices, dtype=np.int64))
    group.create_dataset("valid", shape=(len(indices),), dtype=bool,
                         fillvalue=False)
    group.create_dataset("nnz", shape=(len(indices),), dtype=np.int64,
                         fillvalue=-1)
    for name in SCALAR_DATASETS:
        group.create_dataset(name, shape=(len(indices),), dtype=np.float64,
                             fillvalue=np.nan)
    return group


# ============================================================
# One material
# ============================================================

def process_material(parser, args, h5_path, material_name):
    h5_path = Path(h5_path).expanduser().resolve()

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    out_path = cli.analysis_h5(args.outdir, material_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Condition number estimates: {material_name}")
    print("=" * 72)
    print(f"HDF5 path:       {h5_path}")
    print(f"Output file:     {out_path}:/{GROUP}")
    print(f"Threads:         {args.threads}")
    print(f"Resume:          {args.resume}")
    print(f"Overwrite:       {args.overwrite}")

    with h5py.File(h5_path, "r") as h5_file:
        available_indices = discover_indices(h5_file)

        if not available_indices:
            raise RuntimeError(f"{h5_path}: no E_<index> groups with M found.")

        selected = sorted(cli.resolve_indices(parser, args, available_indices))
        selected = selected[::args.stride]

        if not selected:
            raise RuntimeError(f"{h5_path}: no indices selected.")

    num_indices = len(selected)
    print(f"Selected matrices: {num_indices} "
          f"(indices {selected[0]}..{selected[-1]}, stride {args.stride})")
    print("=" * 72, flush=True)

    group_attrs = dict(
        material=material_name,
        source=str(h5_path),
        n_indices=num_indices,
        stride=args.stride,
        onenorm_t=args.onenorm_t,
        svd_tol=args.svd_tol,
        **material_metadata(h5_path),
    )

    start_time = time.time()

    with h5py.File(h5_path, "r") as h5_file, h5py.File(out_path, "a") as out_file:
        group = open_group(
            out_file,
            indices=selected,
            resume=args.resume,
            overwrite=args.overwrite,
            attrs=group_attrs,
        )

        valid_dataset = group["valid"]
        nnz_dataset = group["nnz"]

        for row, index in enumerate(selected):
            group_name = f"E_{index}"

            if args.resume and valid_dataset[row]:
                print(f"[skip] {group_name}: already valid")
                continue

            M = None
            results = None
            row_start = time.time()

            try:
                M = load_csc_matrix(h5_file, group_name)

                if M.shape[0] != M.shape[1]:
                    raise ValueError(f"matrix is not square: {M.shape}")

                nnz_dataset[row] = M.nnz

                results = condition_estimates(
                    M,
                    onenorm_t=args.onenorm_t,
                    svd_tol=args.svd_tol,
                    svd_ncv=args.svd_ncv,
                )
                results["seconds"] = time.time() - row_start

                for name in SCALAR_DATASETS:
                    group[name][row] = results[name]
                valid_dataset[row] = True
                out_file.flush()

                print(
                    f"[{row + 1:>4}/{num_indices}] {group_name}: "
                    f"n={M.shape[0]}, nnz={M.nnz}, "
                    f"kappa_1={results['cond_1']:.3e}, "
                    f"kappa_inf={results['cond_inf']:.3e}, "
                    f"kappa_2={results['cond_2']:.3e}, "
                    f"{results['seconds']:.1f}s",
                    flush=True,
                )

            except Exception as error:
                warnings.warn(f"{group_name}: failed: {error}")

                for name in SCALAR_DATASETS:
                    group[name][row] = np.nan
                valid_dataset[row] = False
                out_file.flush()

            finally:
                del results
                del M
                gc.collect()

        num_valid = int(np.count_nonzero(valid_dataset[:]))

    elapsed = time.time() - start_time

    print("=" * 72)
    print(f"{material_name}: {num_valid} / {num_indices} valid rows in "
          f"{elapsed / 60:.1f} min")
    print(f"wrote {out_path}:/{GROUP}")
    print("=" * 72, flush=True)

    return num_valid


# ============================================================
# Main
# ============================================================

def main():
    parser, args = parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")

    if args.h5path:
        targets = [(Path(args.h5path), args.material or Path(args.h5path).stem)]
    else:
        if args.material:
            parser.error("--material names the output of one file; it cannot "
                         "be combined with a list of --materials")
        targets = [(cli.material_h5(name), name) for name in args.materials]

    failed = []
    for h5_path, material_name in targets:
        try:
            process_material(parser, args, h5_path, material_name)
        except Exception as error:
            # One missing or broken material file must not discard the sweeps
            # of the others, which each cost hours.
            warnings.warn(f"{material_name}: skipped: {error}")
            failed.append(material_name)

    if failed:
        print(f"[warning] materials skipped: {', '.join(failed)}")
    if len(failed) == len(targets):
        raise RuntimeError("every material failed")

    print()
    print("Plot the result with:")
    print(f"  python ../plotting/plot_condition.py --outdir {args.outdir}")


if __name__ == "__main__":
    main()

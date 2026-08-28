#!/usr/bin/env python3
"""
Exact condition numbers of M(E) at selected energy indices, by dense inversion
and full SVD, as the reference the estimates of condition_est.py are read
against.

Input
-----
A material HDF5 file providing E_<idx>/M as a CSC triplet, and E_<idx>/rhs
where the componentwise reference is wanted. The index selection is REQUIRED:
one row costs a dense inversion and a full SVD, both O(n^3), so there is no
default-all sweep here as there is in condition_est.py.

Algorithm
---------
M is densified once per index and both O(n^3) factorizations are taken from
that one array:

    Minv = inv(M)                     LAPACK getrf + getri
    s    = svdvals(M)                 LAPACK gesdd, singular values only

Everything else is O(n^2) and is read off those two results. The four norms are
the maximum absolute column and row sums of M and of Minv. The two Skeel
numbers do not need the product |Minv| |M| to be formed: it is a nonnegative
matrix, so its infinity norm is the largest entry of its action on a vector,

    cond_skeel   = max( |Minv| ( |M| e ) )
    cond_skeel_x = max( |Minv| ( |M| |x| ) ) / ||x||_inf

which is two matrix-vector products rather than a matrix-matrix product. This
is the same identity abs_inverse_apply_norm() estimates in condition_est.py,
evaluated exactly instead.

kappa_2 comes from the two ends of the full singular value spectrum, with no
Krylov method and no shift-invert, so it is the reference for the svds path of
condition_est.py as well as for the norm estimates.

Every column here is exact to working precision. Nothing is a lower bound. The
point of the script is that the columns of condition_est.py are lower bounds,
and the ratio between the two files is the only measurement of how tight they
are on these matrices.

Cost and the size guard
-----------------------
Dense complex128 needs 16 n^2 bytes per array, and inv() and gesdd each hold
several. --max-n refuses an index whose n exceeds it, rather than letting the
node swap or the job be killed part-way through a sweep; raise it deliberately.
--no-svd and --no-inverse each skip one half, for a run that only needs the
other.

Output
------
    <outdir>/<material>.h5, group condition_exact

The same file condition_est.py writes, in a separate group, so that a plotting
script can read the estimate and the reference of one material together. Only
this group is written; the file is opened in append mode.

    indices              (P,)  energy index of each row
    valid                (P,)  bool, row fully computed
    n, nnz               (P,)  size and nnz of M
    norm1, norminf       (P,)  ||M||_1, ||M||_inf
    norm1_inv            (P,)  ||M^-1||_1
    norminf_inv          (P,)  ||M^-1||_inf
    sigma_max, sigma_min (P,)  extreme singular values, from the full spectrum
    cond_1_exact         (P,)  norm1 * norm1_inv
    cond_inf_exact       (P,)  norminf * norminf_inv
    cond_2_exact         (P,)  sigma_max / sigma_min
    cond_skeel_exact     (P,)  || |M^-1| |M| ||_inf
    cond_skeel_x_exact   (P,)  worst rhs column, NaN without a rhs
    seconds              (P,)  wall time of the row

Usage
-----
    python exact_condition.py --material carbon-chain --idx 84
    python exact_condition.py --material carbon-chain --start 800 --end 900 \\
        --stride 10
    python exact_condition.py /scratch/yimili/matrices2/hdf5/graphene.h5 \\
        --idx 120 250 400 --resume
    python exact_condition.py --material si-bulk --idx 254 --no-svd
    python ../plotting/condition-est/plot_condition.py --materials carbon-chain
"""

import gc
import os
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str((Path(__file__).resolve().parent / ".."
                        / "solvers").resolve()))
import cli
from factor_io import material_metadata

# The two loaders are shared with the estimator rather than repeated, so that
# both scripts read a material file the same way. condition_est.py sets the
# same thread variables at import; this module has already set them above, so
# the import is a no-op in that respect.
from condition_est import load_csc_matrix, load_rhs


GROUP = "condition_exact"

SCALAR_DATASETS = (
    "norm1",
    "norminf",
    "norm1_inv",
    "norminf_inv",
    "sigma_max",
    "sigma_min",
    "cond_1_exact",
    "cond_inf_exact",
    "cond_2_exact",
    "cond_skeel_exact",
    "cond_skeel_x_exact",
    "seconds",
)

# Bytes per entry of a dense complex128 array.
BYTES_PER_ENTRY = 16


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = cli.new_parser(__doc__)

    cli.add_h5_input(
        parser, required=False,
        help="material HDF5 file. With no path given, --material selects it "
             "from the standard scratch layout")

    # default_all=False: an index selection is mandatory. A dense inversion and
    # a full SVD per index is not something to start by accident over a sweep
    # of thousands.
    cli.add_index_selection(parser, default_all=False)

    cli.add_output(
        parser,
        outdir_default=str(cli.CONDITION_DIR),
        outdir_help=f"directory holding the analysis file <material>.h5, the "
                    f"same file condition_est.py writes "
                    f"(default: {cli.CONDITION_DIR})")

    parser.add_argument(
        "--threads", type=int, default=8,
        help="BLAS/LAPACK threads. Default: 8.")

    parser.add_argument(
        "--max-n", type=int, default=8000, metavar="N",
        help="refuse an index whose matrix is larger than this, since the "
             "dense arrays are 16 n^2 bytes each and several are live at "
             "once. Default: 8000, about 1 GB per array.")

    parser.add_argument(
        "--no-svd", action="store_true",
        help="skip the full SVD; sigma_max, sigma_min and cond_2_exact stay "
             "NaN.")

    parser.add_argument(
        "--no-inverse", action="store_true",
        help="skip the dense inversion; every column derived from M^-1 stays "
             "NaN. Leaves the SVD, for a reference on cond_2 alone.")

    parser.add_argument(
        "--resume", action="store_true",
        help="keep an existing group and skip rows already marked valid.")

    parser.add_argument(
        "--overwrite", action="store_true",
        help="delete an existing group and recompute every row.")

    return parser, parser.parse_args()


# ============================================================
# The exact quantities
# ============================================================

def norms_of(dense):
    """(||A||_1, ||A||_inf) of a dense array: max column sum, max row sum."""
    absolute = np.abs(dense)
    return (float(absolute.sum(axis=0).max()),
            float(absolute.sum(axis=1).max()))


def skeel_exact(abs_M, abs_Minv, rhs, Minv):
    """
    (cond_skeel, cond_skeel_x), exactly, in O(n^2).

    |Minv| |M| is nonnegative, so its infinity norm is the largest entry of its
    action on the vector of ones, and no matrix-matrix product is needed. The
    same holds for the action on |x|, which is what makes the two quantities
    one expression with a different vector.

    cond_skeel_x is the worst over the columns of rhs, and NaN when no
    right-hand side is available -- the same convention as condition_est.py.
    """
    n = abs_M.shape[0]
    cond_skeel = float(np.max(abs_Minv @ (abs_M @ np.ones(n))))

    cond_skeel_x = np.nan
    if rhs is not None and np.size(rhs):
        worst = 0.0
        for col in range(rhs.shape[1]):
            x = Minv @ rhs[:, col]
            norm_x = float(np.max(np.abs(x)))
            if norm_x == 0:
                continue
            value = float(np.max(abs_Minv @ (abs_M @ np.abs(x)))) / norm_x
            worst = max(worst, value)
        cond_skeel_x = worst if worst > 0 else np.nan

    return cond_skeel, cond_skeel_x


def exact_condition(M, rhs=None, do_inverse=True, do_svd=True):
    """
    Every column of SCALAR_DATASETS other than `seconds`, computed exactly.

    M is densified once; the dense array is the only large allocation the
    caller cannot avoid, and both O(n^3) routines read it.
    """
    dense = np.asarray(M.todense(), dtype=np.complex128)
    results = {name: np.nan for name in SCALAR_DATASETS}

    norm1, norminf = norms_of(dense)
    results["norm1"] = norm1
    results["norminf"] = norminf

    if do_inverse:
        Minv = np.linalg.inv(dense)
        norm1_inv, norminf_inv = norms_of(Minv)
        results["norm1_inv"] = norm1_inv
        results["norminf_inv"] = norminf_inv
        results["cond_1_exact"] = norm1 * norm1_inv
        results["cond_inf_exact"] = norminf * norminf_inv

        cond_skeel, cond_skeel_x = skeel_exact(
            np.abs(dense), np.abs(Minv), rhs, Minv)
        results["cond_skeel_exact"] = cond_skeel
        results["cond_skeel_x_exact"] = cond_skeel_x
        del Minv

    if do_svd:
        singular_values = np.linalg.svd(dense, compute_uv=False)
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        results["sigma_max"] = sigma_max
        results["sigma_min"] = sigma_min
        results["cond_2_exact"] = (sigma_max / sigma_min
                                   if sigma_min > 0 else np.inf)
        del singular_values

    del dense
    return results


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
                "Use --resume or --overwrite.")
        group = h5_file[GROUP]
        existing = group["indices"][:]
        if not np.array_equal(existing, np.asarray(indices, dtype=np.int64)):
            raise ValueError(
                "The existing group was written for a different index "
                "selection. Use --overwrite or a different --outdir.")
        for name in SCALAR_DATASETS:
            if name not in group:
                group.create_dataset(name, shape=(len(indices),),
                                     dtype=np.float64, fillvalue=np.nan)
                print(f"added missing column '{name}' to the existing group")
        return group

    group = h5_file.create_group(GROUP)
    for key, value in attrs.items():
        group.attrs[key] = value
    group.create_dataset("indices", data=np.asarray(indices, dtype=np.int64))
    group.create_dataset("valid", shape=(len(indices),), dtype=bool,
                         fillvalue=False)
    for name in ("n", "nnz"):
        group.create_dataset(name, shape=(len(indices),), dtype=np.int64,
                             fillvalue=-1)
    for name in SCALAR_DATASETS:
        group.create_dataset(name, shape=(len(indices),), dtype=np.float64,
                             fillvalue=np.nan)
    return group


# ============================================================
# Main
# ============================================================

def main():
    parser, args = parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.no_svd and args.no_inverse:
        parser.error("--no-svd and --no-inverse together leave nothing to do")

    if args.h5path:
        h5_path = Path(args.h5path).expanduser().resolve()
        material_name = args.material or h5_path.stem
    else:
        if not args.material:
            parser.error("give a path to a material file, or --material NAME")
        material_name = args.material
        h5_path = cli.material_h5(material_name)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    out_path = cli.analysis_h5(args.outdir, material_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5_file:
        available = cli.available_indices(h5_file, require="M")
        if not available:
            raise RuntimeError(f"{h5_path}: no E_<index> groups with M found.")
        selected = sorted(cli.resolve_indices(parser, args, available))

    if not selected:
        raise RuntimeError(f"{h5_path}: no indices selected.")

    num_indices = len(selected)

    print("=" * 72)
    print(f"Exact condition numbers: {material_name}")
    print("=" * 72)
    print(f"HDF5 path:       {h5_path}")
    print(f"Output file:     {out_path}:/{GROUP}")
    print(f"Threads:         {args.threads}")
    print(f"Dense inverse:   {not args.no_inverse}")
    print(f"Full SVD:        {not args.no_svd}")
    print(f"Indices:         {num_indices} "
          f"({selected[0]}..{selected[-1]}, stride {args.stride})")
    print("=" * 72, flush=True)

    group_attrs = dict(
        material=material_name,
        source=str(h5_path),
        n_indices=num_indices,
        stride=args.stride,
        max_n=args.max_n,
        did_inverse=not args.no_inverse,
        did_svd=not args.no_svd,
        **material_metadata(h5_path),
    )

    start_time = time.time()

    with h5py.File(h5_path, "r") as h5_file, \
            h5py.File(out_path, "a") as out_file:

        group = open_group(out_file, indices=selected, resume=args.resume,
                           overwrite=args.overwrite, attrs=group_attrs)
        valid_dataset = group["valid"]

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

                n = M.shape[0]
                if n > args.max_n:
                    raise ValueError(
                        f"n = {n} exceeds --max-n {args.max_n}; a dense array "
                        f"would be {BYTES_PER_ENTRY * n * n / 2**30:.1f} GiB "
                        f"and several are live at once")

                group["n"][row] = n
                group["nnz"][row] = M.nnz

                results = exact_condition(
                    M,
                    rhs=load_rhs(h5_file, group_name),
                    do_inverse=not args.no_inverse,
                    do_svd=not args.no_svd,
                )
                results["seconds"] = time.time() - row_start

                for name in SCALAR_DATASETS:
                    group[name][row] = results[name]
                valid_dataset[row] = True
                out_file.flush()

                print(
                    f"[{row + 1:>4}/{num_indices}] {group_name}: n={n}, "
                    f"kappa_inf={results['cond_inf_exact']:.6e}, "
                    f"cond={results['cond_skeel_exact']:.6e}, "
                    f"kappa_2={results['cond_2_exact']:.6e}, "
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
    print(f"{material_name}: {num_valid} / {num_indices} exact rows in "
          f"{elapsed / 60:.1f} min")
    print(f"wrote {out_path}:/{GROUP}")
    print("=" * 72, flush=True)

    print()
    print("Plot the estimates against these points with:")
    print(f"  python ../plotting/condition-est/plot_condition.py "
          f"--materials {material_name} --outdir {args.outdir}")


if __name__ == "__main__":
    main()

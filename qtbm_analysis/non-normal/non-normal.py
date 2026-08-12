#!/usr/bin/env python3
"""
Non-normality of the QTBM system matrices, by full SVD and full eigenvalue
decomposition of every M(E) in an energy sweep.

Input
-----
A material HDF5 file providing E_<idx>/M as a CSC triplet. --idx or
--start/--end select the subset of energy indices to process; the default is
every index present.

Algorithm
---------
For each selected index the matrix is densified and both its singular values
and its eigenvalues are computed exactly, with no truncation or iterative
approximation. Both sequences are then sorted in descending order and paired
by rank:

    sigma_1 >= ... >= sigma_n        singular values
    |lambda_1| >= ... >= |lambda_n|  eigenvalue magnitudes

Two derived quantities are recorded per rank:

    ratio_i     = sigma_i / |lambda_i|
    logcum_k    = log( prod_{i<=k} sigma_i / prod_{i<=k} |lambda_i| )

For a normal matrix the two sequences coincide, so ratio_i = 1 for every i and
logcum_k = 0 for every k. Weyl's majorant theorem makes logcum non-negative and
non-decreasing in k, with logcum_n = 0 exactly, because both products equal
|det M|. Its interior maximum is therefore a scalar measure of non-normality.

Each row is written to disk as soon as it is produced, so a run may be
interrupted and continued with --resume; rows already marked valid are not
recomputed. A dense SVD is O(n^3) in time and O(n^2) in memory, so the
estimated requirement is checked against --max-estimated-svd-gb before each
factorization unless --no-memory-check is given.

Output
------
    <outdir>/<material>.h5, group non_normality

    indices                (P,)    energy index of each row
    valid                  (P,)    bool, row fully computed
    nnz                    (P,)    nnz of M, -1 if unknown
    ratio                  (P, n)  ratio_i
    log_cumulative_ratio   (P, n)  logcum_k
    singular_values        (P, n)  sigma_i
    eigenvalue_magnitudes  (P, n)  |lambda_i|

The row dimension P is the number of selected indices and is fixed when the
group is created; a later run over a different index selection requires
--overwrite. The file is opened in append mode and only this group is written,
so results of other analyses of the same material are preserved.

No figures are produced; see plotting/plot_non_normal.py, which reads this group
and renders the per-index frames and the animation.

Usage
-----
    python non-normal.py /scratch/yimili/matrices2/hdf5/carbon-chain.h5 \
        --start 0 --end 401
    python ../plotting/plot_non_normal.py \
        /scratch/yimili/non-normal/carbon-chain.h5
"""

import argparse
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
from scipy.linalg import svd

sys.path.insert(0, str((Path(__file__).resolve().parent / ".."
                        / "solvers").resolve()))
import cli
from factor_io import material_metadata


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = cli.new_parser(__doc__)

    cli.add_h5_input(parser)
    cli.add_index_selection(parser)
    cli.add_output(
        parser,
        outdir_default=str(cli.NON_NORMAL_DIR),
        outdir_help=f"directory holding the analysis file <material>.h5 "
                    f"(default: {cli.NON_NORMAL_DIR})")

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="BLAS/LAPACK threads per SVD. Default: 8.",
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

    parser.add_argument(
        "--max-estimated-svd-gb",
        type=float,
        default=20.0,
        help=(
            "Safety check. Estimated dense+workspace memory above this value "
            "will stop the script before SVD. Default: 20 GB."
        ),
    )

    parser.add_argument(
        "--no-memory-check",
        action="store_true",
        help="Disable estimated memory safety check.",
    )

    return parser, parser.parse_args()


# ============================================================
# Helpers
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

        if "M" in h5_file[name] and "spectrum" in h5_file[name]:
            found.append(int(match.group(1)))

    return sorted(found)


def load_csc_matrix(h5_file, group_name):
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

    return sp.csc_matrix(
        (data, row_indices, column_pointer),
        shape=shape,
    )


def estimate_svd_memory_gb(shape, is_complex):
    """
    Conservative estimate for dense SVD memory.

    Dense matrix:
      real float64    = 8 bytes per entry
      complex128      = 16 bytes per entry

    LAPACK SVD workspace varies by backend and matrix shape.
    We use a conservative multiplier.
    """
    m, n = shape
    bytes_per_entry = 16 if is_complex else 8
    dense_gb = m * n * bytes_per_entry / 1024**3

    # Conservative for gesdd singular-values-only, including copies/workspace.
    estimated_total_gb = dense_gb * 10.0

    return dense_gb, estimated_total_gb


def sparse_to_dense_for_svd(M):
    """
    Convert sparse matrix to dense Fortran-contiguous array.

    Full SVD of all singular values needs a dense LAPACK call.
    Fortran order reduces chances of extra copies inside LAPACK.
    """
    A = M.toarray()

    if np.isrealobj(A):
        A = np.asarray(A, dtype=np.float64, order="F")
    else:
        A = np.asarray(A, dtype=np.complex128, order="F")

    return A


GROUP = "non_normality"

# Per-rank matrices, all (P, n) float64. The keys are the dataset names.
MATRIX_DATASETS = (
    "ratio",
    "log_cumulative_ratio",
    "singular_values",
    "eigenvalue_magnitudes",
)


def open_group(h5_file, indices, n, resume, overwrite, attrs):
    """
    Create or reopen the output group.

    The row dimension is fixed by the index selection, so an existing group is
    only reusable for the same selection; a mismatch is an error rather than a
    silent partial overwrite. Rows are chunked individually, since they are
    written and read one at a time.

    Returns the group.
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
        if group["ratio"].shape != (len(indices), n):
            raise ValueError(
                f"Existing datasets have shape {group['ratio'].shape}, "
                f"expected {(len(indices), n)}. Use --overwrite."
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
    for name in MATRIX_DATASETS:
        group.create_dataset(name, shape=(len(indices), n), dtype=np.float64,
                             chunks=(1, n), compression="gzip",
                             fillvalue=np.nan)
    return group


def finite_row(dataset, row):
    return bool(np.all(np.isfinite(dataset[row, :])))


# ============================================================
# Main
# ============================================================

def main():
    parser, args = parse_args()

    h5_path = Path(args.h5path).expanduser().resolve()

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    material_name = args.material or h5_path.stem
    out_path = cli.analysis_h5(args.outdir, material_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Non-normal SVD pipeline")
    print("=" * 72)
    print(f"HDF5 path:       {h5_path}")
    print(f"Material name:   {material_name}")
    print(f"Output file:     {out_path}:/{GROUP}")
    print(f"Threads:         {args.threads}")

    print(f"Resume:          {args.resume}")
    print(f"Overwrite:       {args.overwrite}")
    print("=" * 72)

    # --------------------------------------------------------
    # Discover indices and size
    # --------------------------------------------------------

    with h5py.File(h5_path, "r") as h5_file:
        available_indices = discover_indices(h5_file)

        if not available_indices:
            raise RuntimeError("No E_<index> groups with M and spectrum found.")

        selected_indices = cli.resolve_indices(parser, args, available_indices)
        selected_indices = sorted(selected_indices)

        if not selected_indices:
            raise RuntimeError("No indices selected.")

        first_group = f"E_{selected_indices[0]}"
        first_spectrum = np.asarray(
            h5_file[f"{first_group}/spectrum"][...]
        ).squeeze()

        n = first_spectrum.size

        for index in selected_indices:
            group_name = f"E_{index}"
            spectrum_size = np.asarray(
                h5_file[f"{group_name}/spectrum"][...]
            ).squeeze().size

            if spectrum_size != n:
                raise ValueError(
                    f"{group_name} has spectrum size {spectrum_size}, "
                    f"but first selected group has size {n}. "
                    "This script assumes equal matrix sizes for one run."
                )

    num_indices = len(selected_indices)

    print(f"Selected matrices: {num_indices}")
    print(f"First index:       {selected_indices[0]}")
    print(f"Last index:        {selected_indices[-1]}")
    print(f"Matrix size:       {n}")

    group_attrs = dict(
        material=material_name,
        source=str(h5_path),
        n_indices=num_indices,
        matrix_size=n,
        threads=args.threads,
        **material_metadata(h5_path),
    )

    # --------------------------------------------------------
    # Compute SVDs
    # --------------------------------------------------------

    start_time = time.time()

    with h5py.File(h5_path, "r") as h5_file, h5py.File(out_path, "a") as out_file:
        group = open_group(
            out_file,
            indices=selected_indices,
            n=n,
            resume=args.resume,
            overwrite=args.overwrite,
            attrs=group_attrs,
        )

        ratio_matrix = group["ratio"]
        logcum_matrix = group["log_cumulative_ratio"]
        singular_matrix = group["singular_values"]
        eigmag_matrix = group["eigenvalue_magnitudes"]
        valid_dataset = group["valid"]
        nnz_dataset = group["nnz"]

        for row, index in enumerate(selected_indices):
            group_name = f"E_{index}"

            if args.resume and valid_dataset[row]:
                if finite_row(ratio_matrix, row):
                    print(f"[skip] {group_name}: already valid")
                    continue
                else:
                    print(f"[resume warning] {group_name}: marked valid but row not finite; recomputing")
                    valid_dataset[row] = False

            M = None
            A = None
            spectrum = None
            eigmag = None
            singular_values = None
            ratio = None
            logcum = None

            try:
                spectrum = np.asarray(
                    h5_file[f"{group_name}/spectrum"][...]
                ).squeeze()

                if spectrum.ndim != 1:
                    raise ValueError(
                        f"spectrum shape {spectrum.shape}; expected 1D"
                    )

                eigmag = np.sort(
                    np.asarray(np.abs(spectrum), dtype=np.float64)
                )[::-1]

                if not np.all(np.isfinite(eigmag)):
                    raise ValueError("spectrum contains NaN or Inf")

                if np.any(eigmag <= 0):
                    raise ValueError("one or more eigenvalue magnitudes are zero")

                M = load_csc_matrix(h5_file, group_name)

                if M.shape[0] != M.shape[1]:
                    raise ValueError(f"matrix is not square: {M.shape}")

                if M.shape[0] != n:
                    raise ValueError(
                        f"matrix shape {M.shape} does not match spectrum size {n}"
                    )

                is_complex = np.iscomplexobj(M.data)
                dense_gb, estimated_svd_gb = estimate_svd_memory_gb(
                    M.shape,
                    is_complex=is_complex,
                )

                if not args.no_memory_check:
                    if estimated_svd_gb > args.max_estimated_svd_gb:
                        raise MemoryError(
                            f"{group_name}: estimated SVD memory "
                            f"{estimated_svd_gb:.2f} GB exceeds limit "
                            f"{args.max_estimated_svd_gb:.2f} GB. "
                            "Increase --max-estimated-svd-gb if your node has enough RAM."
                        )

                nnz_dataset[row] = M.nnz

                print(
                    f"[{row + 1:>4}/{num_indices}] {group_name}: "
                    f"shape={M.shape}, nnz={M.nnz}, "
                    f"dense≈{dense_gb:.2f}GB, est_svd≈{estimated_svd_gb:.2f}GB",
                    flush=True,
                )

                A = sparse_to_dense_for_svd(M)

                if not np.all(np.isfinite(A)):
                    raise ValueError("dense matrix contains NaN or Inf")

                singular_values = svd(
                    A,
                    compute_uv=False,
                    full_matrices=False,
                    overwrite_a=True,
                    check_finite=False,
                    lapack_driver="gesdd",
                )

                singular_values = np.asarray(singular_values, dtype=np.float64)

                if singular_values.size != n:
                    raise ValueError(
                        f"found {singular_values.size} singular values; expected {n}"
                    )

                if not np.all(np.isfinite(singular_values)):
                    raise ValueError("singular values contain NaN or Inf")

                if np.any(singular_values <= 0):
                    raise ValueError("one or more singular values are zero")

                ratio = singular_values / eigmag

                logcum = np.cumsum(
                    np.log(singular_values) - np.log(eigmag)
                )

                if not np.all(np.isfinite(ratio)):
                    raise ValueError("ratio contains NaN or Inf")

                if not np.all(np.isfinite(logcum)):
                    raise ValueError("log cumulative ratio contains NaN or Inf")

                ratio_matrix[row, :] = ratio
                logcum_matrix[row, :] = logcum
                singular_matrix[row, :] = singular_values
                eigmag_matrix[row, :] = eigmag

                valid_dataset[row] = True
                out_file.flush()

                print(
                    f"      sigma_max={singular_values[0]:.6e}, "
                    f"sigma_min={singular_values[-1]:.6e}, "
                    f"endpoint={logcum[-1]:+.3e}",
                    flush=True,
                )

            except Exception as error:
                warnings.warn(f"{group_name}: failed: {error}")

                ratio_matrix[row, :] = np.nan
                logcum_matrix[row, :] = np.nan
                singular_matrix[row, :] = np.nan
                eigmag_matrix[row, :] = np.nan
                valid_dataset[row] = False
                out_file.flush()

            finally:
                # Critical cleanup after each matrix.
                del logcum
                del ratio
                del singular_values
                del eigmag
                del spectrum
                del A
                del M
                gc.collect()

        num_valid = int(np.count_nonzero(valid_dataset[:]))

    elapsed = time.time() - start_time

    print("=" * 72)
    print(f"SVD pass complete in {elapsed / 3600:.2f} hours")
    print(f"Valid rows: {num_valid} / {num_indices}")
    print("=" * 72)

    if num_valid == 0:
        raise RuntimeError("No valid rows were computed.")

    print()
    print(f"wrote {out_path}:/{GROUP}")
    print("datasets: indices, valid, nnz, "
          + ", ".join(MATRIX_DATASETS))
    print()
    print("Render the frames and the animation with:")
    print(f"  python ../plotting/plot_non_normal.py {out_path}")


if __name__ == "__main__":
    main()

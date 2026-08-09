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

Results are written into memory-mapped arrays as they are produced, so a run
may be interrupted and continued with --resume; rows already marked valid are
not recomputed. A dense SVD is O(n^3) in time and O(n^2) in memory, so the
estimated requirement is checked against --max-estimated-svd-gb before each
factorization unless --no-memory-check is given.

Output
------
Written to <outdir>/<material>/:

    ratio_matrix.npy                 (num_indices, n)  ratio_i per index
    log_cumulative_ratio_matrix.npy  (num_indices, n)  logcum_k per index
    singular_value_matrix.npy        (num_indices, n)  sigma_i
    eigenvalue_magnitude_matrix.npy  (num_indices, n)  |lambda_i|
    ratio_matrix_indices.npy         (num_indices,)    energy index per row
    nnz_by_index.npy                 (num_indices,)    nnz of M, -1 if unknown
    valid_rows.npy                   (num_indices,)    bool, row fully computed
    singular_eigenvalue_ratios.csv   long-format table, only with --save-csv

No figures are produced; see plotting/plot_non_normal.py, which reads these
arrays and renders the per-index frames and the animation.

Usage
-----
    python non-normal.py /scratch/yimili/matrices/hdf5/carbon-chain.h5 \
        --start 0 --end 401
    python ../plotting/plot_non_normal.py /scratch/yimili/non-normal/carbon-chain
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


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = cli.new_parser(__doc__)

    cli.add_h5_input(parser)
    cli.add_index_selection(parser)
    cli.add_output(
        parser,
        outdir_default="/scratch/yimili/non-normal",
        outdir_help="root output directory; a per-material subdirectory is "
                    "created inside it (default: /scratch/yimili/non-normal)")

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="BLAS/LAPACK threads per SVD. Default: 8.",
    )

    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save a CSV. This can be large and slow.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing output folder and skip rows already valid.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output arrays and frames.",
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


def open_or_create_memmap(path, shape, dtype, resume, overwrite):
    path = Path(path)

    if path.exists() and resume and not overwrite:
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape:
            raise ValueError(
                f"Existing {path} has shape {arr.shape}, expected {shape}. "
                "Use --overwrite or a different output folder."
            )
        return arr

    if path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"{path} already exists. Use --resume or --overwrite."
        )

    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def finite_row(arr, row):
    return bool(np.all(np.isfinite(arr[row, :])))


def save_csv(
    csv_path,
    indices,
    ratio_matrix,
    logcum_matrix,
    singular_matrix,
    eigmag_matrix,
    valid_rows,
):
    print("[csv] writing CSV; this can be slow and large")

    with open(csv_path, "w", newline="") as csv_file:
        import csv

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "matrix_index",
                "group_name",
                "rank",
                "singular_value",
                "eigenvalue_magnitude",
                "ratio_sigma_over_abs_lambda",
                "log_cumulative_ratio",
            ]
        )

        n = ratio_matrix.shape[1]

        for row in valid_rows:
            index = int(indices[row])

            for rank in range(n):
                writer.writerow(
                    [
                        index,
                        f"E_{index}",
                        rank + 1,
                        singular_matrix[row, rank],
                        eigmag_matrix[row, rank],
                        ratio_matrix[row, rank],
                        logcum_matrix[row, rank],
                    ]
                )


# ============================================================
# Main
# ============================================================

def main():
    parser, args = parse_args()

    h5_path = Path(args.h5path).expanduser().resolve()

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    material_name = args.material or h5_path.stem

    out_root = Path(args.outdir)
    out_dir = out_root / material_name

    ratio_path = out_dir / "ratio_matrix.npy"
    logcum_path = out_dir / "log_cumulative_ratio_matrix.npy"
    singular_path = out_dir / "singular_value_matrix.npy"
    eigmag_path = out_dir / "eigenvalue_magnitude_matrix.npy"
    index_path = out_dir / "ratio_matrix_indices.npy"
    nnz_path = out_dir / "nnz_by_index.npy"
    valid_path = out_dir / "valid_rows.npy"
    csv_path = out_dir / "singular_eigenvalue_ratios.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in [
            ratio_path,
            logcum_path,
            singular_path,
            eigmag_path,
            index_path,
            nnz_path,
            valid_path,
            csv_path,
        ]:
            if path.exists():
                path.unlink()

    print("=" * 72)
    print("Non-normal SVD pipeline")
    print("=" * 72)
    print(f"HDF5 path:       {h5_path}")
    print(f"Material name:   {material_name}")
    print(f"Output folder:   {out_dir}")
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

    # --------------------------------------------------------
    # Create/load disk arrays
    # --------------------------------------------------------

    shape = (num_indices, n)

    ratio_matrix = open_or_create_memmap(
        ratio_path,
        shape=shape,
        dtype=np.float64,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    logcum_matrix = open_or_create_memmap(
        logcum_path,
        shape=shape,
        dtype=np.float64,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    singular_matrix = open_or_create_memmap(
        singular_path,
        shape=shape,
        dtype=np.float64,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    eigmag_matrix = open_or_create_memmap(
        eigmag_path,
        shape=shape,
        dtype=np.float64,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    indices_array = np.asarray(selected_indices, dtype=np.int64)

    if index_path.exists() and args.resume and not args.overwrite:
        existing_indices = np.load(index_path)
        if not np.array_equal(existing_indices, indices_array):
            raise ValueError(
                "Existing index file does not match selected indices. "
                "Use --overwrite or a different output folder."
            )
    else:
        np.save(index_path, indices_array)

    if nnz_path.exists() and args.resume and not args.overwrite:
        nnz_array = np.load(nnz_path)
        if nnz_array.shape != (num_indices,):
            raise ValueError(
                f"Existing nnz array shape {nnz_array.shape}, "
                f"expected {(num_indices,)}."
            )
    else:
        nnz_array = np.full(num_indices, -1, dtype=np.int64)

    if valid_path.exists() and args.resume and not args.overwrite:
        valid_rows_bool = np.load(valid_path)
        if valid_rows_bool.shape != (num_indices,):
            raise ValueError(
                f"Existing valid rows shape {valid_rows_bool.shape}, "
                f"expected {(num_indices,)}."
            )
    else:
        valid_rows_bool = np.zeros(num_indices, dtype=bool)

    # --------------------------------------------------------
    # Compute SVDs
    # --------------------------------------------------------

    start_time = time.time()

    with h5py.File(h5_path, "r") as h5_file:
        for row, index in enumerate(selected_indices):
            group_name = f"E_{index}"

            if args.resume and valid_rows_bool[row]:
                if finite_row(ratio_matrix, row):
                    print(f"[skip] {group_name}: already valid")
                    continue
                else:
                    print(f"[resume warning] {group_name}: marked valid but row not finite; recomputing")
                    valid_rows_bool[row] = False

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

                nnz_array[row] = M.nnz

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

                valid_rows_bool[row] = True

                ratio_matrix.flush()
                logcum_matrix.flush()
                singular_matrix.flush()
                eigmag_matrix.flush()

                np.save(nnz_path, nnz_array)
                np.save(valid_path, valid_rows_bool)

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
                valid_rows_bool[row] = False

                ratio_matrix.flush()
                logcum_matrix.flush()
                singular_matrix.flush()
                eigmag_matrix.flush()

                np.save(nnz_path, nnz_array)
                np.save(valid_path, valid_rows_bool)

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

    elapsed = time.time() - start_time

    valid_rows = np.where(valid_rows_bool)[0]

    print("=" * 72)
    print(f"SVD pass complete in {elapsed / 3600:.2f} hours")
    print(f"Valid rows: {len(valid_rows)} / {num_indices}")
    print("=" * 72)

    if len(valid_rows) == 0:
        raise RuntimeError("No valid rows were computed.")

    # --------------------------------------------------------
    # Optional CSV
    # --------------------------------------------------------

    if args.save_csv:
        save_csv(
            csv_path=csv_path,
            indices=indices_array,
            ratio_matrix=ratio_matrix,
            logcum_matrix=logcum_matrix,
            singular_matrix=singular_matrix,
            eigmag_matrix=eigmag_matrix,
            valid_rows=valid_rows,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("Done.")
    print()
    print("Output folder:")
    print(out_dir)
    print()
    print("Saved arrays:")
    print(ratio_path)
    print(logcum_path)
    print(singular_path)
    print(eigmag_path)
    print(index_path)
    print(nnz_path)
    print(valid_path)

    print()
    print("Render the frames and the animation with:")
    print(f"  python ../plotting/plot_non_normal.py {out_dir}")

    if args.save_csv:
        print()
        print("Saved CSV:")
        print(csv_path)


if __name__ == "__main__":
    main()
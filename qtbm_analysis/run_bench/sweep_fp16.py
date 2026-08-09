#!/usr/bin/env python3
"""
Accuracy sweep of the two half-precision Block Thomas implementations.

Input
-----
A material HDF5 file that ``run_benchmarks.py`` has already processed, since
the reference solutions are read from it:

    E_<idx>/M                            system matrix, CSC triplet
    E_<idx>/rhs                          right-hand side
    E_<idx>/blockthomas/complex128/x     reference solution x_128
    E_<idx>/blockthomas/complex64/x      single-precision solution x_64
    global/condition_full_svd            kappa_2(M) per index, optional

The file is opened read-only. Unlike the other drivers, this one writes nothing
back into the HDF5 file, so a sweep may be repeated freely.

Algorithm
---------
For every selected energy index:

1. Extract the block-tridiagonal blocks of M under the requested partition,
   either the uniform --block-size or, with --auto-blocks, the partition
   detected from the sparsity pattern once on the first matrix.
2. Solve M x = b with both half-precision variants,
       implementation 1  BlockThomasFP16             LU with substitution,
       implementation 2  BlockThomasExplicitInvFP16  explicit block inverses.
3. Record, against the stored complex128 solution x_128,
       relative residual  ||M x - b|| / ||b||          for both variants,
       forward error      ||x - x_128|| / ||x_128||    for both variants and
                                                       for the stored x_64.

The residual measures backward error and is expected near the half-precision
unit roundoff u = 2^-11; the forward error additionally carries kappa_2(M), so
the two must be reported together for the numbers to be interpretable.

For implementation 2, --inv-dtype sets the precision in which the explicit
inverse is formed before being rounded to fp16 for storage and application. It
defaults to float32. Passing float16 makes the factorization genuinely
all-half-precision and measures what the higher-precision inversion buys; see
the Block Thomas section of the top-level README.

Failures at a single index (a missing group, a singular modified diagonal
block, an fp16 overflow) are recorded and the sweep continues.

Output
------
Written to --outdir, default <script_dir>/plots:

    <material>_metrics.csv   idx, relres_fp16, relres_fp16_inv,
                             fwd_err_fp16_vs_c128, fwd_err_fp16_inv_vs_c128,
                             fwd_err_c64_vs_c128, cond_full_svd
    <material>_metrics.txt   the same table plus run metadata and the list of
                             failed indices

No figures are produced; see plotting/plot_fp16_accuracy.py, which consumes the
CSV.

Usage
-----
    python sweep_fp16.py .../carbon-nanotube.h5 --start 0 --end 401 --block-size 32
    python sweep_fp16.py .../graphene.h5 --start 0 --end 401 --auto-blocks
    python sweep_fp16.py .../graphene.h5 --idx 0 25 50 --inv-dtype float16
    python ../plotting/plot_fp16_accuracy.py plots/carbon-nanotube_metrics.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import h5py
import numpy as np
import scipy.sparse as sp

import cli
from solver_classes import (
    extract_blocks_sparse, block_sizes_from_matrix, offband_nnz,
    BlockThomasFP16, BlockThomasExplicitInvFP16,
)

CSV_COLUMNS = ["idx", "relres_fp16", "relres_fp16_inv",
               "fwd_err_fp16_vs_c128", "fwd_err_fp16_inv_vs_c128",
               "fwd_err_c64_vs_c128", "cond_full_svd"]


def parse_args():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, required=False,
                     default="/scratch/yimili/matrices/hdf5/carbon-nanotube.h5")
    cli.add_index_selection(ap)
    cli.add_block_partition(ap, default_block_size=32)
    cli.add_inv_dtype(ap)
    ap.add_argument("--cond-path", type=str,
                    default="global/condition_full_svd", metavar="PATH",
                    help="dataset holding one condition number per index")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: <script_dir>/plots)")
    return ap, ap.parse_args()


def load_matrix(g):
    """CSC matrix from an HDF5 group holding data/indices/indptr and a shape."""
    shape = tuple(g.attrs["shape"]) if "shape" in g.attrs else None
    return sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                         shape=shape)


def resolve_partition(M, partition, auto_blocks):
    """
    Partition to use for this matrix.

    Returns `partition` unchanged unless --auto-blocks was given and no
    partition has been detected yet, in which case it is derived from the
    sparsity pattern and validated. `partition` is None on the first call under
    --auto-blocks. Detection happens once because the pattern does not depend
    on the energy index.

    Raises ValueError if the detected partition is not block tridiagonal:
    extract_blocks_sparse would then silently discard the out-of-band entries
    and both variants would return a plausible but wrong solution.
    """
    if not auto_blocks or isinstance(partition, (list, tuple)):
        return partition
    partition = block_sizes_from_matrix(M)
    bad = offband_nnz(M, partition)
    print(f"auto-detected partition: {len(partition)} blocks, "
          f"sizes {min(partition)}..{max(partition)}, off-band nnz = {bad}")
    if bad:
        raise ValueError(f"detected partition leaves {bad} nonzeros outside "
                         f"the block-tridiagonal band; refusing to solve")
    return partition


def sweep(args, indices, inv_dtype):
    """
    Run both half-precision variants over the requested index range.

    Returns (rows, partition, failed) where rows is a list of dicts keyed by
    CSV_COLUMNS, partition is the block partition used, and failed is
    a list of (index, exception repr).
    """
    rows, failed = [], []
    partition = cli.resolve_partition(args)

    with h5py.File(args.h5path, "r") as f:
        cond_arr = f[args.cond_path][:] if args.cond_path in f else None
        if cond_arr is None:
            print(f"warning: '{args.cond_path}' not found in {args.h5path}; "
                  f"the condition-number column will be NaN.")

        for idx in indices:
            key = f"E_{idx}"
            try:
                if key not in f:
                    raise KeyError(f"{key} not in file")
                M = load_matrix(f[f"{key}/M"])
                b = f[f"{key}/rhs"][:]
                x128 = f[f"{key}/blockthomas/complex128/x"][:]
                x64 = f[f"{key}/blockthomas/complex64/x"][:]

                partition = resolve_partition(M, partition, args.auto_blocks)
                D, L, U = extract_blocks_sparse(M, partition)

                x16 = BlockThomasFP16(D, L, U).solve(b)
                x16i = BlockThomasExplicitInvFP16(
                    D, L, U, inv_dtype=inv_dtype).solve(b)

                norm_b = np.linalg.norm(b)
                norm_x128 = np.linalg.norm(x128)
                row = {
                    "idx": idx,
                    "relres_fp16": np.linalg.norm(M @ x16 - b) / norm_b,
                    "relres_fp16_inv": np.linalg.norm(M @ x16i - b) / norm_b,
                    "fwd_err_fp16_vs_c128":
                        np.linalg.norm(x16 - x128) / norm_x128,
                    "fwd_err_fp16_inv_vs_c128":
                        np.linalg.norm(x16i - x128) / norm_x128,
                    "fwd_err_c64_vs_c128":
                        np.linalg.norm(x64 - x128) / norm_x128,
                    "cond_full_svd":
                        float(cond_arr[idx])
                        if cond_arr is not None and idx < len(cond_arr)
                        else float("nan"),
                }
            except Exception as exc:                      # noqa: BLE001
                failed.append((idx, repr(exc)))
                continue

            rows.append(row)
            if idx % 25 == 0:
                print(f"idx={idx:4d}  "
                      f"relres16={row['relres_fp16']:.3e}  "
                      f"relres16_inv={row['relres_fp16_inv']:.3e}  "
                      f"fwd16={row['fwd_err_fp16_vs_c128']:.3e}  "
                      f"fwd16_inv={row['fwd_err_fp16_inv_vs_c128']:.3e}  "
                      f"fwd64={row['fwd_err_c64_vs_c128']:.3e}  "
                      f"cond={row['cond_full_svd']:.3e}")

    return rows, partition, failed


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(rows, txt_path, args, partition, part_desc, failed, indices):
    with open(txt_path, "w") as fh:
        fh.write(f"material   : {args.material}\n")
        fh.write(f"h5path     : {args.h5path}\n")
        fh.write(f"partition  : {part_desc}\n")
        if isinstance(partition, (list, tuple)):
            fh.write(f"block sizes: {list(partition)}\n")
        fh.write(f"inv dtype  : {args.inv_dtype} (implementation 2)\n")
        fh.write(f"requested  : idx {indices[0]}..{indices[-1]} "
                 f"({len(indices)} indices)\n")
        fh.write(f"completed  : {len(rows)}\n")
        fh.write(f"failed     : {len(failed)}\n")
        if failed:
            fh.write("failed indices:\n")
            for idx, message in failed:
                fh.write(f"  idx={idx}: {message}\n")
        fh.write("\nidx  relres_fp16   relres_fp16_inv  fwd_err_fp16  "
                 "fwd_err_fp16_inv  fwd_err_c64   cond_full_svd\n")
        for row in rows:
            fh.write(f"{row['idx']:4d}  {row['relres_fp16']:.6e}  "
                     f"{row['relres_fp16_inv']:.6e}  "
                     f"{row['fwd_err_fp16_vs_c128']:.6e}  "
                     f"{row['fwd_err_fp16_inv_vs_c128']:.6e}  "
                     f"{row['fwd_err_c64_vs_c128']:.6e}  "
                     f"{row['cond_full_svd']:.6e}\n")


def main():
    ap, args = parse_args()
    args.material = args.material or Path(args.h5path).stem

    with h5py.File(args.h5path, "r") as f:
        available = ([int(i) for i in f["metadata/indices"][:]]
                     if "metadata/indices" in f
                     else sorted(int(k[2:]) for k in f if k.startswith("E_")))
    indices = cli.resolve_indices(ap, args, available)
    if not indices:
        raise SystemExit("no requested index is present in the file")

    outdir = Path(args.outdir or Path(__file__).resolve().parent / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    inv_dtype = getattr(np, args.inv_dtype)
    rows, partition, failed = sweep(args, indices, inv_dtype)

    part_desc = (f"{len(partition)} custom blocks"
                 if isinstance(partition, (list, tuple))
                 else f"block size {partition}")
    print(f"\nCompleted {len(rows)}/{len(indices)} indices "
          f"({args.material}, {part_desc}).")
    if failed:
        print(f"{len(failed)} indices failed or were skipped, e.g.:")
        for idx, message in failed[:10]:
            print(f"  idx={idx}: {message}")

    csv_path = outdir / f"{args.material}_metrics.csv"
    txt_path = outdir / f"{args.material}_metrics.txt"
    write_csv(rows, csv_path)
    write_txt(rows, txt_path, args, partition, part_desc, failed, indices)

    print(f"\nwrote {csv_path}\nwrote {txt_path}")
    print(f"Plot with: python ../plotting/plot_fp16_accuracy.py {csv_path}")


if __name__ == "__main__":
    main()

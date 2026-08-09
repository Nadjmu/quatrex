#!/usr/bin/env python3
"""
Derivation and verification of a non-uniform block partition.

Input
-----
Either a material HDF5 file, in which case one energy index is read and the
choice of index is immaterial because the sparsity pattern does not depend on
it, or a bare CSR .npz triplet of data, indices, indptr and shape.

Algorithm
---------
The detection itself is solver_classes.find_block_slices, which grows a reach
frontier row by row and declares a block boundary wherever the frontier stops
advancing. It resides in the solver module rather than here so that the
detector and the solvers consuming its output cannot diverge; this file is the
command-line front end and the verification wrapper.

Verification
------------
The detector looks forward only, using the largest column index of each row.
That produces a correct block-tridiagonal partition when the matrix is
structurally symmetric, which the QTBM matrices are, but nothing in the
algorithm enforces the property. A partition that cuts through a real coupling
does not fail loudly: extract_blocks_sparse would discard the out-of-band
entries and Block Thomas would return a plausible but wrong solution.
offband_nnz counts exactly those entries, so it is checked here and again
inside bench().

The detector also merges its first two slices by construction, the seed row
being absorbed into the first real block, so the leading block is coarser than
the true structure. This is harmless: a coarser partition remains valid, at the
cost of slightly more arithmetic in the first block.

Output
------
The block count, the size range, sum(bs^2), which is the dense block-storage
cost that determines whether a custom partition is worthwhile, and the
off-band nonzero count. Exit status 1 if the off-band count is nonzero.

--compare-block-size reports the same figures for a uniform partition and
their storage ratio. --emit-python prints a line to paste into
run_benchmarks.MATERIAL_BLOCKS.

Usage
-----
    python determine_custom_block_size.py /scratch/yimili/matrices/hdf5/graphene.h5
    python determine_custom_block_size.py .../graphene.h5 --idx 25
    python determine_custom_block_size.py .../M_E_0.npz
    python determine_custom_block_size.py .../graphene.h5 --compare-block-size 416
    python determine_custom_block_size.py .../graphene.h5 --emit-python
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp

import cli
from solver_classes import block_sizes_from_matrix, offband_nnz


def load_matrix(path, idx=0):
    """
    Load a matrix from a material HDF5 file or a bare CSR .npz triplet.

    HDF5 groups store CSC; the .npz files written by
    export_qtbm_systems._save_csr_npz store CSR.

    find_block_slices walks indptr row by row and therefore requires CSR, hence
    the conversion on the HDF5 branch. Constructing a csr_matrix directly from
    the CSC triplet would silently yield the transpose.
    """
    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            g = f[f"E_{idx}/M"]
            n = len(g["indptr"]) - 1
            A = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                              shape=(n, n))
        return A.tocsr()
    d = np.load(path)
    return sp.csr_matrix((d["data"], d["indices"], d["indptr"]),
                         shape=tuple(d["shape"]))


def describe(sizes, A):
    """
    Report the properties of one partition. Returns True if it is a valid
    block-tridiagonal partition of A.
    """
    sizes = list(sizes)
    n = A.shape[0]
    print(f"matrix shape      = {A.shape}, nnz = {A.nnz} "
          f"({A.nnz / (n * n):.3%} of dense)")
    print(f"blocks found      = {len(sizes)}")
    print(f"block sizes       = {sizes}")
    print(f"  min / max / sum = {min(sizes)} / {max(sizes)} / {sum(sizes)}")
    print(f"  mean            = {np.mean(sizes):.1f}")

    bad = offband_nnz(A, sizes)
    print(f"off-band nnz      = {bad}"
          f"{'   <-- NOT block tridiagonal, do not use' if bad else '   (clean)'}")

    # Dense storage cost of the block factors. This is the figure that decides
    # whether a non-uniform partition is worth adopting: the diagonal blocks
    # are stored densely, so the cost scales as the sum of their squared sizes.
    cost = sum(s * s for s in sizes)
    print(f"sum of bs^2       = {cost:,}  (dense diagonal-block storage units)")
    return bad == 0


def main():
    ap = cli.new_parser(__doc__)
    ap.add_argument("matrix", type=Path,
                    help="material .h5 file, or a CSR .npz triplet")
    ap.add_argument("--idx", type=int, nargs=1, default=[0], metavar="N",
                    help="energy index to read from an .h5 file (default: 0). "
                         "The sparsity pattern is the same at every index, so "
                         "the choice does not affect the partition")
    ap.add_argument("--compare-block-size", type=int, default=None,
                    metavar="M",
                    help="also report the same figures for this uniform "
                         "block size, and the storage ratio between the two")
    ap.add_argument("--emit-python", action="store_true",
                    help="print the partition as a python list literal only")
    args = ap.parse_args()

    A = load_matrix(args.matrix, args.idx[0])
    A.sort_indices()
    sizes = block_sizes_from_matrix(A)

    if args.emit_python:
        print(f'"{args.matrix.stem}": {list(sizes)},')
        return 0

    print(f"=== custom partition ({args.matrix.name}) ===")
    ok = describe(sizes, A)

    if args.compare_block_size is not None:
        print(f"\n=== uniform partition, block size "
              f"{args.compare_block_size} ===")
        n = A.shape[0]
        if n % args.compare_block_size:
            print(f"n = {n} is not divisible by {args.compare_block_size}; "
                  f"a uniform partition is not applicable")
        else:
            uniform = [args.compare_block_size] * (n // args.compare_block_size)
            describe(uniform, A)
            print(f"\ncustom / uniform dense block storage = "
                  f"{sum(s*s for s in sizes) / sum(s*s for s in uniform):.3f}x")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

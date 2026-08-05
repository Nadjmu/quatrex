#!/usr/bin/env python3
"""
determine_custom_block_size.py -- derive a custom (non-uniform) block
partition from a matrix's sparsity pattern and check it is safe to solve with.

The detection algorithm itself now lives in solver_classes (find_block_slices
/ block_sizes_from_matrix), so the solvers and the detector cannot drift apart;
this file is the command-line front end and the verification wrapper.

WHY VERIFY. The detector grows a reach frontier row by row and only ever looks
FORWARD (the largest column index in each row). That yields a correct
block-tridiagonal partition when the matrix is structurally symmetric, which
the QTBM matrices are -- but nothing in the algorithm enforces it. A partition
that cuts through a real coupling does not fail loudly: extract_blocks_sparse
would silently discard the out-of-band entries and Block Thomas would return a
plausible, wrong x. offband_nnz() counts exactly those entries, so it is
checked here and again inside bench().

The detector also merges its first two slices by construction (the seed row is
absorbed into the first real block), so the leading block comes out coarser
than the true structure. That is harmless -- a coarser partition is still a
valid one, just with slightly more arithmetic in the first block.

Usage:
    # from a material HDF5 file (uses one energy index; the pattern is the
    # same at every index, so which one does not matter)
    python determine_custom_block_size.py /scratch/yimili/matrices/hdf5/graphene.h5
    python determine_custom_block_size.py .../graphene.h5 --idx 25

    # from a bare CSR .npz (data/indices/indptr/shape)
    python determine_custom_block_size.py /scratch/yimili/matrices/.../M_E_0.npz

    # compare against the uniform block size currently used for that material
    python determine_custom_block_size.py .../graphene.h5 --compare-bs 416

    # emit a python literal to paste into run_benchmarks.MATERIAL_BLOCKS
    python determine_custom_block_size.py .../graphene.h5 --emit-python
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp

from solver_classes import block_sizes_from_matrix, offband_nnz


def load_matrix(path, idx=0):
    """
    Accept either a material HDF5 file (groups are CSC) or a bare CSR .npz
    triplet as written by export_qtbm_systems._save_csr_npz.

    find_block_slices walks indptr row by row, so it needs CSR -- hence the
    .tocsr() on the h5 branch. Building csr_matrix straight from the CSC
    triplet would hand back the transpose instead.
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

    # dense storage cost of the block factors, the figure that decides whether
    # a non-uniform partition is worth it at all
    cost = sum(s * s for s in sizes)
    print(f"sum of bs^2       = {cost:,}  (dense diagonal-block storage units)")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", type=Path,
                    help="material .h5 file, or a CSR .npz triplet")
    ap.add_argument("--idx", type=int, default=0,
                    help="energy index to read from an .h5 file (default 0)")
    ap.add_argument("--compare-bs", type=int, default=None,
                    help="also report the same figures for this uniform block size")
    ap.add_argument("--emit-python", action="store_true",
                    help="print the partition as a python list literal only")
    args = ap.parse_args()

    A = load_matrix(args.matrix, args.idx)
    A.sort_indices()
    sizes = block_sizes_from_matrix(A)

    if args.emit_python:
        print(f'"{args.matrix.stem}": {list(sizes)},')
        return 0

    print(f"=== custom partition ({args.matrix.name}) ===")
    ok = describe(sizes, A)

    if args.compare_bs is not None:
        print(f"\n=== uniform partition, bs = {args.compare_bs} ===")
        n = A.shape[0]
        if n % args.compare_bs:
            print(f"n={n} is not divisible by {args.compare_bs} -- "
                  f"uniform partition not applicable")
        else:
            uniform = [args.compare_bs] * (n // args.compare_bs)
            describe(uniform, A)
            print(f"\ncustom / uniform dense block storage = "
                  f"{sum(s*s for s in sizes) / sum(s*s for s in uniform):.3f}x")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Batch benchmark driver: every material, the direct solvers, both working
precisions.

Input
-----
One HDF5 file per material in HDF5_DIR, named <material>.h5, each containing

    metadata/indices        the energy indices present
    metadata/energies       the corresponding energies, eV
    E_<idx>/M               system matrix M(E), CSC triplet
    E_<idx>/rhs             right-hand side, (n, nrhs)

The materials processed are listed in MATERIAL_BS. The Block Thomas partition
is always detected from the sparsity pattern; it is never uniform.

Algorithm
---------
For each material, for each energy index with a non-empty right-hand side, the
matrix and right-hand side are passed to ``solvers.bench_all.bench``, which
runs every requested solver once per dtype and records factorization time,
solve time, relative residual and factor memory. This module contains no
benchmarking logic of its own: there is exactly one bench() implementation, so
a solver added or a bug fixed there is picked up by every driver.

Indices whose right-hand side has zero columns are skipped; they appear as gaps
in the figures produced downstream.

Output
------
Solver results are appended into each material's own HDF5 file as siblings of
E_<idx>/M:

    E_<idx>/superlu/<dtype>/, E_<idx>/blockthomas/<dtype>/, ...

This mutates the source file in place. Copy it first if the original must be
preserved. No figures are produced here; see plotting/plot_speedup.py.

Usage
-----
    python run_benchmarks.py                  # all materials, one process
    python run_benchmarks.py --material si-bulk  # one material only
    python run_benchmarks.py --material si-bulk --exclude-solvers superlu

Run each material as its own process (e.g. one `nohup ... &` per material) to
keep them independent: a crash or OOM kill in one no longer takes the others
down with it, and does not corrupt their HDF5 files (a killed process can
corrupt whichever file it was writing to when it died).

    python ../plotting/plot_speedup.py /scratch/yimili/matrices2/hdf5/graphene.h5
"""

import argparse
import sys
from pathlib import Path

# Resolved relative to this file rather than the working directory, so the
# script runs correctly from anywhere.
sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp
import h5py

import cli
from bench_all import bench, DEFAULT_SOLVERS
from solver_classes import block_sizes_from_matrix, offband_nnz

HDF5_DIR = cli.HDF5_DIR

# Materials to benchmark. Taken from cli.MATERIALS, which is the one place a
# per-material property is edited; only the materials that declare a block size
# can be partitioned, so only those are listed. The block size itself is not
# used: the partition is always detected from the sparsity pattern, never
# uniform. See resolve_partition.
MATERIAL_BS = {name: mat.block_size for name, mat in cli.MATERIALS.items()
               if mat.block_size is not None}

# The first entry defines the baseline that speedups and "vs base" errors in
# bench() refer to.
DTYPES = (np.complex128, np.complex64)

# GMRES (both backends) and cuDSS are dropped: this sweep only covers the
# direct solvers, at complex128 and complex64. The fp16 Block Thomas variants
# are excluded, same as DEFAULT_SOLVERS: their NumPy kernels made a full sweep
# impractically slow, confirmed with single_solve.py.
SOLVERS = tuple(s for s in DEFAULT_SOLVERS
                if s not in {"gmres", "gmres-cupy", "cudss"})

# Dropped by default: nothing. --exclude-solvers adds to this per invocation,
# so one slow material can drop a solver (e.g. SuperLU) without affecting the
# others; see main(). Dropping superlu entirely is safe with plot_speedup.py's
# baseline made a plot-time choice (--baseline-solver/--baseline-dtype) rather
# than a hard-coded requirement.
EXCLUDE = {}


def resolve_partition(material, M_first):
    """
    Block partition for one material, detected from its sparsity pattern.

    The partition is never uniform: the exported matrices have a non-uniform
    block structure, and a uniform partition of one of them either cuts a real
    coupling or pads the blocks. There is deliberately no option to select one.

    Parameters
    ----------
    material : material name, used only in the error message.
    M_first  : the matrix at the first energy index. The sparsity pattern is
               identical at every index, so the partition is detected once per
               material rather than once per index.

    Returns
    -------
    list of per-block sizes summing to n.

    Raises
    ------
    ValueError the detected partition is not block tridiagonal, which would
               make every Block Thomas result silently wrong.
    """
    sizes = block_sizes_from_matrix(M_first)
    bad = offband_nnz(M_first, sizes)
    print(f"  auto partition: {len(sizes)} blocks, "
          f"sizes {min(sizes)}..{max(sizes)}, off-band nnz = {bad}")
    if bad:
        raise ValueError(f"{material}: detected partition leaves {bad} "
                         f"nonzeros outside the block-tridiagonal band")
    return list(sizes)


def load_sparse(g):
    """CSC matrix from an HDF5 group holding data/indices/indptr and a shape."""
    shape = tuple(g.attrs["shape"]) if "shape" in g.attrs else None
    return sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                         shape=shape)


def run_material(material, h5path, solvers=SOLVERS):
    """
    Benchmark every energy index of one material and append the results into
    its HDF5 file. Returns the number of indices benchmarked.

    M and rhs are loaded one index at a time, inside the loop, rather than for
    every index up front: a material's matrices held all at once can run into
    the tens of GB (e.g. si-bulk, n=3840 at ~9% density, over ~2800 indices),
    which is what was silently killing the process partway through a sweep
    before this was per-index.
    """
    with h5py.File(h5path, "r") as f:
        indices = f["metadata/indices"][:].tolist()
        energies = f["metadata/energies"][:]
        first_M = load_sparse(f[f"E_{indices[0]}/M"])

    idx_arr = np.array(indices)
    print(f"{material} | n_energies = {len(indices)} | E[-1] = {energies[-1]}")
    if solvers != SOLVERS:
        print(f"Solvers: {', '.join(solvers)} (--exclude-solvers applied)")

    bs = resolve_partition(material, first_M)
    print(f"Partition: {len(bs)} detected blocks, "
          f"sizes {min(bs)}..{max(bs)}")
    print(f"Indices: {idx_arr[:10]}... (total {len(idx_arr)})")

    benchmarked = 0
    with h5py.File(h5path, "a") as f:
        for idx in idx_arr:
            rhs_idx = f[f"E_{idx}/rhs"][:]
            if rhs_idx.shape[-1] == 0:
                continue
            M_idx = load_sparse(f[f"E_{idx}/M"])
            bench(M_idx, rhs_idx, idx, bs, dtypes=DTYPES, h5file=f,
                  save=True, solvers=solvers, exclude=EXCLUDE)
            benchmarked += 1
    return benchmarked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=sorted(MATERIAL_BS),
                        help="benchmark only this material, as its own "
                             "process (default: all, in one process)")
    parser.add_argument("--exclude-solvers", nargs="+", choices=SOLVERS,
                        default=(), metavar="NAME",
                        help="drop these solvers for this invocation only, "
                             "e.g. --exclude-solvers superlu for a material "
                             "where it is impractically slow. Materials "
                             "already benchmarked with the full solver set "
                             "are unaffected; plot_speedup.py's baseline is "
                             "chosen at plot time and need not be superlu.")
    args = parser.parse_args()
    materials = [args.material] if args.material else MATERIAL_BS
    solvers = tuple(s for s in SOLVERS if s not in args.exclude_solvers)

    for material in materials:
        print("=" * 80)
        print(f"Processing material: {material}")
        print("=" * 80)

        h5path = HDF5_DIR / f"{material}.h5"
        if not h5path.exists():
            print(f"Warning: {h5path} not found, skipping.")
            continue

        count = run_material(material, h5path, solvers=solvers)
        print(f"Finished {material}: appended solver results for {count} "
              f"indices into {h5path}")
        print(f"Plot with: python ../plotting/plot_speedup.py {h5path}\n")

    print("All materials processed.")


if __name__ == "__main__":
    main()

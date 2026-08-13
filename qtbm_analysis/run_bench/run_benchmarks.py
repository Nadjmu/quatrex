#!/usr/bin/env python3
"""
Batch benchmark driver: every material, the direct solvers plus the fp16
Block Thomas variants, both working precisions.

Input
-----
One HDF5 file per material in HDF5_DIR, named <material>.h5, each containing

    metadata/indices        the energy indices present
    metadata/energies       the corresponding energies, eV
    E_<idx>/M               system matrix M(E), CSC triplet
    E_<idx>/rhs             right-hand side, (n, nrhs)

The materials processed and their uniform block sizes are listed in
MATERIAL_BS; BLOCK_MODE selects where the Block Thomas partition comes from.

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
    python run_benchmarks.py
    python ../plotting/plot_speedup.py /scratch/yimili/matrices2/hdf5/graphene.h5
"""

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
from cli import FP16_SOLVERS
from solver_classes import block_sizes_from_matrix, offband_nnz

HDF5_DIR = cli.HDF5_DIR

# Materials to benchmark, and their block sizes. Both come from cli.MATERIALS,
# which is the one place a per-material property is edited; only the materials
# that declare a block size can be partitioned, so only those are listed.
MATERIAL_BS = {name: mat.block_size for name, mat in cli.MATERIALS.items()
               if mat.block_size is not None}

# Custom non-uniform partitions per material, used when BLOCK_MODE == "custom",
# taken from the `blocks` field of cli.MATERIALS. Generate an entry with
#     python ../block-thomas/determine_custom_block_size.py <material>.h5 --emit-python
# and set it there.
MATERIAL_BLOCKS = {name: mat.blocks for name, mat in cli.MATERIALS.items()
                   if mat.blocks is not None}

# Source of the Block Thomas partition:
#   "uniform"  MATERIAL_BS
#   "custom"   MATERIAL_BLOCKS, an error if the material has no entry
#   "auto"     detected from the sparsity pattern of the first matrix
BLOCK_MODE = "auto"

# The first entry defines the baseline that speedups and "vs base" errors in
# bench() refer to.
DTYPES = (np.complex128, np.complex64)

# GMRES (both backends) and cuDSS are dropped: this sweep only covers the
# direct solvers. The two fp16 Block Thomas variants are added on top of
# DEFAULT_SOLVERS, since bench() otherwise only runs them when named
# explicitly.
SOLVERS = tuple(s for s in DEFAULT_SOLVERS
                if s not in {"gmres", "gmres-cupy", "cudss"}) + FP16_SOLVERS

EXCLUDE = {}


def resolve_partition(material, M_first):
    """
    Block partition for one material, per BLOCK_MODE.

    Parameters
    ----------
    material : material name, used to index MATERIAL_BS / MATERIAL_BLOCKS.
    M_first  : the matrix at the first energy index. Only used by "auto"; the
               sparsity pattern is identical at every index, so the partition
               is detected once per material rather than once per index.

    Returns
    -------
    int for a uniform partition, or a list of per-block sizes summing to n.

    Raises
    ------
    KeyError   BLOCK_MODE == "custom" and the material has no entry.
    ValueError the detected partition is not block tridiagonal, which would
               make every Block Thomas result silently wrong.
    """
    if BLOCK_MODE == "uniform":
        return MATERIAL_BS[material]
    if BLOCK_MODE == "custom":
        sizes = MATERIAL_BLOCKS.get(material)
        if sizes is None:
            raise KeyError(
                f"BLOCK_MODE='custom' but MATERIAL_BLOCKS has no entry for "
                f"'{material}'. Generate one with determine_custom_block_size.py "
                f"--emit-python, or switch BLOCK_MODE to 'auto'.")
        return list(sizes)
    if BLOCK_MODE == "auto":
        sizes = block_sizes_from_matrix(M_first)
        bad = offband_nnz(M_first, sizes)
        print(f"  auto partition: {len(sizes)} blocks, "
              f"sizes {min(sizes)}..{max(sizes)}, off-band nnz = {bad}")
        if bad:
            raise ValueError(f"{material}: detected partition leaves {bad} "
                             f"nonzeros outside the block-tridiagonal band")
        return list(sizes)
    raise ValueError(f"unknown BLOCK_MODE {BLOCK_MODE!r}")


def load_sparse(g):
    """CSC matrix from an HDF5 group holding data/indices/indptr and a shape."""
    shape = tuple(g.attrs["shape"]) if "shape" in g.attrs else None
    return sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                         shape=shape)


def run_material(material, h5path):
    """
    Benchmark every energy index of one material and append the results into
    its HDF5 file. Returns the number of indices benchmarked.
    """
    with h5py.File(h5path, "r") as f:
        indices = f["metadata/indices"][:].tolist()
        energies = f["metadata/energies"][:]
        M = {idx: load_sparse(f[f"E_{idx}/M"]) for idx in indices}
        rhs = {idx: f[f"E_{idx}/rhs"][:] for idx in indices}

    idx_arr = np.array(indices)
    print(f"{material} | n_energies = {len(indices)} | E[-1] = {energies[-1]}")

    bs = resolve_partition(material, M[indices[0]])
    if isinstance(bs, (list, tuple)):
        print(f"Partition: {len(bs)} custom blocks, "
              f"sizes {min(bs)}..{max(bs)} (BLOCK_MODE={BLOCK_MODE})")
    else:
        print(f"Partition: uniform, block size = {bs}")
    print(f"Indices: {idx_arr[:10]}... (total {len(idx_arr)})")

    benchmarked = 0
    with h5py.File(h5path, "a") as f:
        for idx in idx_arr:
            if rhs[idx].shape[-1] == 0:
                continue
            bench(M[idx], rhs[idx], idx, bs, dtypes=DTYPES, h5file=f,
                  save=True, solvers=SOLVERS, exclude=EXCLUDE)
            benchmarked += 1
    return benchmarked


def main():
    for material in MATERIAL_BS:
        print("=" * 80)
        print(f"Processing material: {material}")
        print("=" * 80)

        h5path = HDF5_DIR / f"{material}.h5"
        if not h5path.exists():
            print(f"Warning: {h5path} not found, skipping.")
            continue

        count = run_material(material, h5path)
        print(f"Finished {material}: appended solver results for {count} "
              f"indices into {h5path}")
        print(f"Plot with: python ../plotting/plot_speedup.py {h5path}\n")

    print("All materials processed.")


if __name__ == "__main__":
    main()

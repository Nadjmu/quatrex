#!/usr/bin/env python3
"""
Batch benchmark driver restricted to the GPU solvers.

Input
-----
Identical to ``run_benchmarks.py``: one HDF5 file per material in HDF5_DIR,
providing E_<idx>/M and E_<idx>/rhs. The CPU driver is normally run first, but
that is not a precondition here; it only determines whether the figure produced
downstream has a baseline to divide by.

Algorithm
---------
For each material and each energy index with a non-empty right-hand side, the
solvers in GPU_SOLVERS are run through ``solvers.bench_all.bench`` at the
precisions in GPU_DTYPES. No CPU solver is rerun, so this driver may be
executed on a GPU node without repeating the CPU sweep.

Every GPU solver checks for a visible CUDA device at construction, so on a
machine without one each combination prints a skip line and the driver
completes without error.

Output
------
Results are appended into each material's own HDF5 file under
E_<idx>/<solver>/<dtype>/, exactly as in the CPU driver. No figures are
produced here; the stored solutions and backward errors are read by
block-thomas/forward_error.py.

Usage
-----
    python gpu_run_benchmarks.py                  # all materials, one process
    python gpu_run_benchmarks.py --material si-bulk  # one material only

Run each material as its own process, as with run_benchmarks.py: a crash or
OOM kill in one no longer takes the others down with it, and does not corrupt
their HDF5 files.

    python ../block-thomas/forward_error.py \
        /scratch/yimili/matrices2/hdf5/graphene.h5
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import h5py

from bench_all import bench
from run_benchmarks import HDF5_DIR, MATERIAL_BS, load_sparse, resolve_partition

# Canonical solver names, see solvers/cli.py. A one-element tuple needs its
# trailing comma: iterating a bare string would make `solver in solvers` a
# substring test inside bench().
GPU_SOLVERS = ("cudss",)

# cuDSS supports both precisions; GMRES on CuPy is excluded at single precision
# for the same reason as in the CPU driver.
GPU_DTYPES = (np.complex128, np.complex64)
GPU_EXCLUDE = {}


def run_material(material, h5path):
    """
    Benchmark the GPU solvers over one material. Returns the index count.

    M and rhs are loaded one index at a time inside the loop rather than for
    every index up front; see run_benchmarks.run_material for why (a whole
    material's matrices held at once can run into tens of GB).
    """
    with h5py.File(h5path, "r") as f:
        indices = f["metadata/indices"][:].tolist()
        first_M = load_sparse(f[f"E_{indices[0]}/M"])

    bs = resolve_partition(material, first_M)

    benchmarked = 0
    with h5py.File(h5path, "a") as f:
        for idx in indices:
            rhs_idx = f[f"E_{idx}/rhs"][:]
            if rhs_idx.shape[-1] == 0:
                continue
            M_idx = load_sparse(f[f"E_{idx}/M"])
            bench(M_idx, rhs_idx, idx, bs, dtypes=GPU_DTYPES, h5file=f,
                  save=True, solvers=GPU_SOLVERS, exclude=GPU_EXCLUDE)
            benchmarked += 1
    return benchmarked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=sorted(MATERIAL_BS),
                        help="benchmark only this material, as its own "
                             "process (default: all, in one process)")
    args = parser.parse_args()
    materials = [args.material] if args.material else MATERIAL_BS

    for material in materials:
        h5path = HDF5_DIR / f"{material}.h5"
        if not h5path.exists():
            print(f"Warning: {h5path} not found, skipping.")
            continue

        count = run_material(material, h5path)
        print(f"Finished {material}: appended {', '.join(GPU_SOLVERS)} results "
              f"for {count} indices into {h5path}")
        print(f"Analyse with: python ../block-thomas/forward_error.py "
              f"{h5path}\n")


if __name__ == "__main__":
    main()

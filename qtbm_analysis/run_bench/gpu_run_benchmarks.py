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
produced; see plotting/plot_speedup.py, which reads the stored timings and
divides by the SuperLU complex128 baseline already in the file.

Usage
-----
    python gpu_run_benchmarks.py
    python ../plotting/plot_speedup.py /scratch/yimili/matrices2/hdf5/graphene.h5 \
        --solvers cudss gmres_cupy --suffix _gpu
"""

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
    """Benchmark the GPU solvers over one material. Returns the index count."""
    with h5py.File(h5path, "r") as f:
        indices = f["metadata/indices"][:].tolist()
        M = {idx: load_sparse(f[f"E_{idx}/M"]) for idx in indices}
        rhs = {idx: f[f"E_{idx}/rhs"][:] for idx in indices}

    bs = resolve_partition(material, M[indices[0]])

    benchmarked = 0
    with h5py.File(h5path, "a") as f:
        for idx in indices:
            if rhs[idx].shape[-1] == 0:
                continue
            bench(M[idx], rhs[idx], idx, bs, dtypes=GPU_DTYPES, h5file=f,
                  save=True, solvers=GPU_SOLVERS, exclude=GPU_EXCLUDE)
            benchmarked += 1
    return benchmarked


def main():
    for material in MATERIAL_BS:
        h5path = HDF5_DIR / f"{material}.h5"
        if not h5path.exists():
            print(f"Warning: {h5path} not found, skipping.")
            continue

        count = run_material(material, h5path)
        print(f"Finished {material}: appended {', '.join(GPU_SOLVERS)} results "
              f"for {count} indices into {h5path}")
        print(f"Plot with: python ../plotting/plot_speedup.py {h5path} "
              f"--solvers {' '.join(GPU_SOLVERS)} --suffix _gpu\n")


if __name__ == "__main__":
    main()

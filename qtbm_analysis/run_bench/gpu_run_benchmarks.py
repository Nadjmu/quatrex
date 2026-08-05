#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bench_all import bench
from run_benchmarks import (  # reuse the plotting code as-is
    _load_sparse, plot_speedup, MATERIAL_BS,
)

GPU_SOLVERS = ("cudss",)       # note the comma -- without it this is a string,
                               # and `solver in solvers` becomes substring matching
GPU_DTYPES = (np.complex128, np.complex64)   # cudss runs both; gmres_cupy stays c128-only
GPU_EXCLUDE = {}


def _load_stored_metric(f, idx, solver, dname, M_i, rhs_i):
    """Pull a previously-saved (solver, dtype) result back out of the h5
    file so the plot's speedup-vs-SuperLU baseline still works, without
    rerunning anything on CPU."""
    path = f"E_{idx}/{solver}/{dname}"
    if path not in f:
        return None
    g = f[path]
    x = g["x"][:]
    t_f = g["time_fact"][()]
    t_s = g["time_solve"][()] if "time_solve" in g else None
    res = np.linalg.norm(M_i @ x - rhs_i) / np.linalg.norm(rhs_i)
    mem = g.attrs.get("factor_nbytes", 0)
    return {"factor": t_f, "solve": t_s, "res": res, "mem": mem}


def main():
    hdf5_dir = Path("/scratch/yimili/matrices/hdf5")
    plot_dir = hdf5_dir.parent / "plots"

    for material in MATERIAL_BS:
        h5path = hdf5_dir / f"{material}.h5"
        if not h5path.exists():
            print(f"Warning: {h5path} not found, skipping.")
            continue

        with h5py.File(h5path, "r") as f:
            indices = f["metadata/indices"][:].tolist()
            M = {idx: _load_sparse(f[f"E_{idx}/M"]) for idx in indices}
            rhs = {idx: f[f"E_{idx}/rhs"][:] for idx in indices}

        idx_arr = np.array(indices)
        bs = MATERIAL_BS[material]

        metrics = []
        with h5py.File(h5path, "a") as f:      # append mode -- same file
            for idx in idx_arr:
                if rhs[idx].shape[-1] == 0:
                    continue
                m = bench(M[idx], rhs[idx], idx, bs,
                          dtypes=GPU_DTYPES, h5file=f, save=True,
                          solvers=GPU_SOLVERS, exclude=GPU_EXCLUDE)
                base = _load_stored_metric(f, idx, "superlu", "complex128",
                                            M[idx], rhs[idx])
                if base is not None:
                    m["superlu_c128"] = base            # restore baseline for plotting
                metrics.append(m)

        print(f"Finished {material}: appended gmres_cupy (c128) and cudss (c128/c64) into {h5path}")
        plot_speedup(material, metrics, idx_arr.tolist(),
                    plot_dir / f"{material}_speedup_gpu.png")


if __name__ == "__main__":
    main()
#Compares the forward accuracy of different solvers for a given energy in a HDF5 file.

import h5py
import numpy as np

# ---- config: edit these ----
FILE = "/scratch/yimili/matrices/hdf5/graphene.h5"
ENERGY = "E_25"

# list of (solver, dtype) to load
SOLVERS = [
    ("blockthomas", "complex128"),
    ("superlu", "complex128"),
    ("mumps", "complex128"),
    ("cudss", "complex64"),
]

# reference (solver, dtype) used as base
BASE = ("mumps", "complex128")
# -----------------------------

def load_x(f, solver, dtype):
    return np.array(f[f"/{ENERGY}/{solver}/{dtype}/x"])

with h5py.File(FILE, "r") as f:
    data = {(s, d): load_x(f, s, d) for s, d in SOLVERS}

for (s, d), x in data.items():
    first = x[0] if x.ndim == 1 else x[0, :]
    print(f"--- {s} [{d}] (shape={x.shape}) first entry: {first}")

x_base = data[BASE]
ncols = x_base.shape[1] if x_base.ndim > 1 else 1

print(f"\nRelative accuracy (||x - x_base|| / ||x_base||) per column, base = {BASE[0]} [{BASE[1]}]")
for (s, d), x in data.items():
    if (s, d) == BASE:
        continue
    print(f"\n{s} [{d}]:")
    for j in range(ncols):
        col_base = x_base[:, j] if x_base.ndim > 1 else x_base
        col = x[:, j] if x.ndim > 1 else x
        rel = np.linalg.norm(col - col_base) / np.linalg.norm(col_base)
        print(f"  x{j+1}: {rel:.3e}")
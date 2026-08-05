#!/usr/bin/env python3
"""
Run benchmarks AND produce the speedup-comparison plot for all materials.
Block sizes: carbon-nanotube=32, carbon-chain=104, graphene=416, si-bulk=256.

Solver results are appended directly into each material's own HDF5 file
(e.g. /scratch/yimili/matrices/hdf5/carbon-nanotube.h5), as a DIRECT sibling
of each energy index's own M/Sigma/rhs/spectrum:

    E_<idx>/superlu/<dtype>/, E_<idx>/umfpack/<dtype>/, E_<idx>/gmres_scipy/<dtype>/, ...

This mutates the source file in place; back it up first if you'd rather
keep the original untouched.

For each material, after the sweep finishes, a two-panel speedup-vs-SuperLU
plot (factorization time / solve time, across all solvers and both dtypes)
is saved as PNG next to that material's HDF5 file:

    <hdf5_dir>/../plots/<material>_speedup.png

This script contains NO benchmarking logic of its own -- it's a thin driver
around bench_all.bench(), so a fix or a new solver added there is
automatically picked up here. There is exactly one bench() implementation
in the whole project.

Usage: python run_benchmarks.py
"""

import sys
from pathlib import Path

# Add solvers directory to path (resolved relative to THIS file, not cwd,
# so the script works regardless of where it's invoked from)
sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import scipy.sparse as sp
import h5py

import matplotlib
matplotlib.use("Agg")                  # headless -- no display in a script
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bench_all import bench, DEFAULT_SOLVERS
from solver_classes import block_sizes_from_matrix, offband_nnz

# GMRES (SciPy) is skipped at single precision for this batch run -- see
# bench_all.bench's `exclude` docstring for the mechanism.
EXCLUDE = {"gmres": {"complex64"},
            "gmres_cupy": {"complex64"}}

MATERIAL_BS = {
    "carbon-chain": 104,
    "carbon-nanotube": 32,
    "graphene": 416,
    "si-bulk": 256,
}

# Custom non-uniform partitions, keyed by material. Leave a material out (or
# set it to None) to use the uniform MATERIAL_BS entry. Generate an entry with
#     python ../block-thomas/determine_custom_block_size.py <material>.h5 --emit-python
# and paste the printed line here. BLOCK_MODE below selects which is used.
MATERIAL_BLOCKS = {}

# "uniform" -- MATERIAL_BS, the historical behaviour
# "custom"  -- MATERIAL_BLOCKS, erroring if the material has no entry
# "auto"    -- detect the partition from the first matrix of each material
BLOCK_MODE = "uniform"

DTYPES = (np.complex128, np.complex64)     # first entry = baseline dtype

# ---- plot styling (same as the notebook's comparison cell) -----------------
PRIMARY = "c128"
DTYPE_LS = {"c128": "-", "c64": "--"}
STYLE = {
    "superlu":          ("SuperLU",              "#555555", "x"),
    "umfpack":          ("UMFPACK",              "#E67E22", "s"),
    "mumps":            ("MUMPS",                "#27AE60", "^"),
    "gmres":            ("GMRES (SciPy)",        "#8E44AD", "D"),
    "gmres_cupy":       ("GMRES (CuPy)",         "#C0392B", "v"),
    "cudss":            ("cuDSS",                "#16A085", "P"),
    "block_thomas":     ("Block Thomas (LU)",    "#2E86AB", "o"),
    "block_thomas_inv": ("Block Thomas (inv)",   "#9B59B6", "*"),
}


def resolve_partition(material, M_first):
    """The block partition to bench this material with -- see BLOCK_MODE."""
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
        # The sparsity pattern is identical at every energy index, so the
        # partition is detected once per material rather than per index.
        sizes = block_sizes_from_matrix(M_first)
        bad = offband_nnz(M_first, sizes)
        print(f"  auto partition: {len(sizes)} blocks, "
              f"sizes {min(sizes)}..{max(sizes)}, off-band nnz = {bad}")
        if bad:
            raise ValueError(f"{material}: detected partition leaves {bad} "
                             f"nonzeros outside the block-tridiagonal band")
        return list(sizes)
    raise ValueError(f"unknown BLOCK_MODE {BLOCK_MODE!r}")


def _load_sparse(g):
    shape = tuple(g.attrs["shape"]) if "shape" in g.attrs else None
    return sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)


# -----------------------------------------------------------------------------
# Plotting (ported from the notebook's comparison cell)
# -----------------------------------------------------------------------------
def _speedup_series(metrics, solver, sfx, field):
    """(indices, baseline_time / solver_time) where this solver+dtype ran."""
    key, base_key = f"{solver}_{sfx}", f"superlu_{PRIMARY}"
    pts = [(m["idx"], m[base_key][field] / m[key][field])
           for m in metrics
           if key in m and base_key in m and m[key][field] > 0]
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def plot_speedup(material, metrics, sweep, out_path):
    """Two-panel (factor time / solve time) speedup-vs-SuperLU-c128 plot,
    saved to out_path. No-op (prints a note) if metrics is empty."""
    if not metrics:
        print(f"  (no successful indices for {material} -- skipping plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True, sharey=True)
    solved_indices = [m["idx"] for m in metrics]
    skipped = np.array(sorted(set(sweep) - set(solved_indices)))

    all_speedups = [y for field in ("factor", "solve") for solver in STYLE
                    for sfx in DTYPE_LS
                    for y in _speedup_series(metrics, solver, sfx, field)[1]]
    ymin = min(all_speedups + [1.0]) / 2.0
    ymax = max(all_speedups + [1.0]) * 2.0

    plotted_solvers, plotted_dtypes = [], []
    for ax, field, panel in zip(axes, ["factor", "solve"],
                                ["Factorization time", "Solve time"]):
        for solver, (label, color, marker) in STYLE.items():
            for sfx, ls in DTYPE_LS.items():
                if solver == "superlu" and sfx == PRIMARY:
                    continue
                xs, ys = _speedup_series(metrics, solver, sfx, field)
                if not xs:
                    continue
                ax.plot(xs, ys, color=color, marker=marker, ls=ls,
                        markersize=5, markeredgecolor="white",
                        markeredgewidth=0.6, lw=1.6, alpha=0.95, zorder=3)
                if solver not in plotted_solvers:
                    plotted_solvers.append(solver)
                if sfx not in plotted_dtypes:
                    plotted_dtypes.append(sfx)

        ax.axhline(1.0, color="0.2", lw=1.1, ls=":", zorder=2)
        ax.axhspan(1.0, ymax, color="#27AE60", alpha=0.05, zorder=0)
        ax.axhspan(ymin, 1.0, color="#C0392B", alpha=0.05, zorder=0)

        if len(skipped):
            breaks = np.where(np.diff(skipped) > 1)[0] + 1
            for run in np.split(skipped, breaks):
                ax.axvspan(run[0] - 0.5, run[-1] + 0.5, color="0.55",
                          alpha=0.25, zorder=1)

        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Energy index")
        ax.set_title(panel, fontsize=11)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.12)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(f"speedup vs SuperLU {PRIMARY}")
    axes[0].text(0.02, 0.97, f"faster than SuperLU {PRIMARY}",
                 transform=axes[0].transAxes, fontsize=8, color="#1E8449", va="top")
    axes[0].text(0.02, 0.03, f"slower than SuperLU {PRIMARY}",
                 transform=axes[0].transAxes, fontsize=8, color="#922B21", va="bottom")

    handles = [Line2D([], [], color=STYLE[s][1], marker=STYLE[s][2], ls="-",
                      markersize=5, markeredgecolor="white", markeredgewidth=0.6)
               for s in plotted_solvers]
    labels = [STYLE[s][0] for s in plotted_solvers]
    for sfx in plotted_dtypes:
        handles.append(Line2D([], [], color="0.3", ls=DTYPE_LS[sfx]))
        labels.append({"c128": "complex128", "c64": "complex64"}.get(sfx, sfx))
    handles.append(Line2D([], [], color="0.2", lw=1.1, ls=":"))
    labels.append(f"SuperLU {PRIMARY} baseline")
    if len(skipped):
        handles.append(plt.Rectangle((0, 0), 1, 1, color="0.55", alpha=0.25))
        labels.append("no RHS")
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(f"{material} — runtime vs SuperLU {PRIMARY} baseline",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {out_path}")


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main():
    hdf5_dir = Path("/scratch/yimili/matrices/hdf5")
    plot_dir = hdf5_dir.parent / "plots"

    materials = list(MATERIAL_BS.keys())

    for material in materials:
        print("=" * 80)
        print(f"Processing material: {material}")
        print("=" * 80)

        h5path = hdf5_dir / f"{material}.h5"
        if not h5path.exists():
            print(f"Warning: {h5path} not found, skipping.")
            continue

        with h5py.File(h5path, "r") as f:
            indices = f["metadata/indices"][:].tolist()
            energies = f["metadata/energies"][:]
            M = {idx: _load_sparse(f[f"E_{idx}/M"]) for idx in indices}
            rhs = {idx: f[f"E_{idx}/rhs"][:] for idx in indices}

        idx_arr = np.array(indices)
        print(f"{material} | n_energies = {len(indices)} | E[-1] = {energies[-1]}")
        bs = resolve_partition(material, M[indices[0]])
        if isinstance(bs, (list, tuple)):
            print(f"Partition: {len(bs)} custom blocks, "
                  f"sizes {min(bs)}..{max(bs)} (BLOCK_MODE={BLOCK_MODE})")
        else:
            print(f"Partition: uniform, block size (bs) = {bs}")
        print(f"Indices: {idx_arr[:10]}... (total {len(idx_arr)})")

        # Append solver results directly into the material's own file,
        # as a direct sibling of E_<idx>/M, /rhs, /Sigma -- no separate
        # *_LU.h5 and no extra "solvers" nesting level.
        metrics = []
        with h5py.File(h5path, "a") as f:
            for idx in idx_arr:
                if rhs[idx].shape[-1] == 0:
                    continue
                m = bench(M[idx], rhs[idx], idx, bs,
                          dtypes=DTYPES, h5file=f, save=True,
                          solvers=DEFAULT_SOLVERS, exclude=EXCLUDE)
                metrics.append(m)

        print(f"Finished {material}, appended solver results into {h5path}")

        # ---- comparison plot for this material -----------------------------
        plot_speedup(material, metrics, idx_arr.tolist(),
                    plot_dir / f"{material}_speedup.png")
        print()

    print("All materials processed.")


if __name__ == "__main__":
    main()
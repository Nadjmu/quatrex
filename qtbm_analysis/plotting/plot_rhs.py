#!/usr/bin/env python3
"""
Right-hand side magnitude of the QTBM system along the energy sweep.

Input
-----
One material HDF5 file per material, as written by make_hdf5.py:

    metadata/            attrs: valence_band_edge, conduction_band_edge,
                          grid_energy_min, resolution
    E_<idx>/rhs           right-hand side of M(E) x = b, (n, nmodes)

The right-hand side is present only at indices where the number of injected
modes is nonzero; main3.py already skips writing it otherwise, and those
indices are simply absent from the sweep drawn here.

Algorithm
---------
The right-hand side is matrix-valued at every energy, (n, nmodes) with nmodes
the number of open-channel modes injected by the contacts, so it cannot be
overlaid directly across a sweep. Its Frobenius norm reduces it to one number
per energy, the same reduction plot_qtbm_spectra.py applies to the extreme
singular values of M(E).

Output
------
One figure per material, ||b(E)||_F on a logarithmic axis against energy, with
the valence and conduction band edges marked.

    <outdir>/<material>_rhs.png

Materials default to those main3.py exports; pass --material to restrict.

Usage
-----
    python plot_rhs.py
    python plot_rhs.py --material graphene carbon-chain
    python plot_rhs.py --outdir figures
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
sys.path.append(str((_HERE / ".." / "solvers").resolve()))

import h5py
import numpy as np
import matplotlib.pyplot as plt

import cli
from factor_io import material_metadata
from style import axis_label, energies_of, mark_band_edges, save_figure

# Mirrors main3.EXAMPLES: the materials stage 1 exports by default.
DEFAULT_MATERIALS = ["carbon-nanotube", "si-bulk", "carbon-chain", "graphene"]


def load_rhs_sweep(h5path):
    """
    Frobenius norm of the right-hand side at every energy index that has one.

    Returns (indices, norms) as parallel arrays, sorted by index.
    """
    indices, norms = [], []
    with h5py.File(h5path, "r") as f:
        for key in f:
            if not key.startswith("E_"):
                continue
            g = f[key]
            if "rhs" not in g:
                continue
            rhs = g["rhs"][:]
            if rhs.size == 0:
                continue
            indices.append(int(key[2:]))
            norms.append(float(np.linalg.norm(rhs)))
    order = np.argsort(indices)
    return np.array(indices)[order], np.array(norms)[order]


def plot_rhs(indices, norms, attrs, material, out_path):
    """||b(E)||_F along the sweep, with the band edges marked."""
    have_energy = energies_of(attrs, [0]) is not None
    xs = energies_of(attrs, indices) if have_energy else indices

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(xs, norms, marker="o", ms=3, lw=0.9, color="tab:purple",
               label=r"$\|b(E)\|_F$")
    mark_band_edges(ax, attrs)
    ax.set_xlabel(axis_label(have_energy))
    ax.set_ylabel(r"$\|b(E)\|_F$")
    ax.set_title(f"Right-hand side norm along the energy sweep — {material}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=300)


def main():
    ap = cli.new_parser(__doc__)
    ap.add_argument("--material", type=str, nargs="+", default=None,
                    metavar="NAME",
                    help="materials to plot (default: "
                         f"{', '.join(DEFAULT_MATERIALS)})")
    cli.add_output(ap, material=False, outdir_default=str(cli.CONDITION_DIR),
                   outdir_help=f"output directory "
                               f"(default: {cli.CONDITION_DIR})")
    args = ap.parse_args()

    materials = args.material or DEFAULT_MATERIALS
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    produced = 0
    for material in materials:
        h5path = cli.material_h5(material)
        if not h5path.exists():
            print(f"[skip] {material}: {h5path} not found")
            continue

        indices, norms = load_rhs_sweep(h5path)
        if len(indices) == 0:
            print(f"[skip] {material}: no right-hand side found in {h5path}")
            continue

        attrs = material_metadata(h5path)
        out_path = outdir / f"{material}_rhs.png"
        plot_rhs(indices, norms, attrs, material, out_path)
        produced += 1

    if produced == 0:
        raise SystemExit("no material had right-hand side data to plot")


if __name__ == "__main__":
    main()

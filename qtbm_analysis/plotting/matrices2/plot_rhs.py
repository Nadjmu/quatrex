#!/usr/bin/env python3
"""
Number of injected modes of the QTBM right-hand side along the energy sweep.

Input
-----
One material HDF5 file per material, as written by make_hdf5.py:

    metadata/            attrs: valence_band_edge, conduction_band_edge,
                          grid_energy_min, resolution
    E_<idx>/rhs           right-hand side of M(E) x = b, (n, nmodes)

main3.py writes rhs at every exported index, including those with nmodes = 0
where no contact mode is open, so the band gap shows up as a run of zeros
rather than a gap in the sweep.

Algorithm
---------
nmodes = rhs.shape[-1] at every exported index. No other reduction of the
right-hand side is taken: nmodes is the quantity of interest, since it counts
the open-channel modes injected by the contacts and is expected to vanish
inside the band gap and step up outside it.

Output
------
One figure per material, nmodes on a linear axis against energy, with the
valence and conduction band edges marked.

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
sys.path.append(str((_HERE / "..").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "solvers").resolve()))

import h5py
import numpy as np
import matplotlib.pyplot as plt

import cli
from factor_io import material_metadata
from style import (axis_label, energies_of, mark_band_edges, save_figure,
                   write_data_report)

# Mirrors main3.EXAMPLES: the materials stage 1 exports by default.
DEFAULT_MATERIALS = ["carbon-nanotube", "si-bulk", "carbon-chain", "graphene"]


def load_rhs_sweep(h5path):
    """
    Number of columns of the right-hand side at every exported energy index.

    Returns (indices, nmodes) as parallel arrays, sorted by index. Indices
    with zero open modes are kept, not dropped, since nmodes = 0 inside the
    band gap is the point of the figure.
    """
    indices, nmodes = [], []
    with h5py.File(h5path, "r") as f:
        for key in f:
            if not key.startswith("E_"):
                continue
            g = f[key]
            if "rhs" not in g:
                continue
            indices.append(int(key[2:]))
            nmodes.append(g["rhs"].shape[-1])
    order = np.argsort(indices)
    return np.array(indices)[order], np.array(nmodes)[order]


def plot_rhs(indices, nmodes, attrs, material, out_path):
    """nmodes along the sweep, with the band edges marked."""
    have_energy = energies_of(attrs, [0]) is not None
    xs = energies_of(attrs, indices) if have_energy else indices

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(xs, nmodes, where="mid", lw=1.2, color="tab:purple",
           label="open modes")
    mark_band_edges(ax, attrs)
    ax.set_xlabel(axis_label(have_energy))
    ax.set_ylabel("number of injected modes")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Right-hand side width along the energy sweep — {material}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
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

        indices, nmodes = load_rhs_sweep(h5path)
        if len(indices) == 0:
            print(f"[skip] {material}: no right-hand side found in {h5path}")
            continue

        attrs = material_metadata(h5path)
        out_path = outdir / f"{material}_rhs.png"
        plot_rhs(indices, nmodes, attrs, material, out_path)

        colmap = {"idx": np.asarray(indices)}
        energies = energies_of(attrs, indices)
        if energies is not None:
            colmap["energy_eV"] = energies
        colmap["n_modes"] = np.asarray(nmodes)
        write_data_report(
            outdir / f"{material}_rhs_data.txt",
            title=f"right-hand side width along the energy sweep  —  {material}",
            source=str(h5path),
            source_attrs=attrs,
            config={"figures": f"{material}_rhs.png",
                    "exported indices": str(len(indices)),
                    "quantity": "n_modes = rhs.shape[-1] at each exported "
                                "E_<idx>/rhs"},
            series={"open contact modes per exported energy index": colmap},
            notes=["n_modes = 0 inside the band gap is kept, not dropped: "
                   "main3.py writes rhs at every exported index."],
        )
        produced += 1

    if produced == 0:
        raise SystemExit("no material had right-hand side data to plot")


if __name__ == "__main__":
    main()

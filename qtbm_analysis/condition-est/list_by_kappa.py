#!/usr/bin/env python3
"""
List energy indices by kappa_2 bucket, from a condition_est.py output file.

Input
-----
A condition-est analysis file (condition_est.py's output), holding the
condition group written there: indices, cond_2, valid, and the grid_energy_min
/ resolution attributes copied from the material file.

Only rows with valid == True are considered; energy(i) = grid_energy_min +
resolution * i.

Output
------
For each bucket [k, k + --bin-width) covering the observed kappa_2 range, the
index and energy of up to --max-per-bin rows falling in it, in index order.

Usage
-----
    python list_by_kappa.py /scratch/yimili/condition-est/carbon-nanotube.h5
    python list_by_kappa.py .../si-bulk.h5 --bin-width 1000 --max-per-bin 20
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))
import cli

GROUP = "condition"


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help="condition-est analysis file, e.g. "
                             "condition-est/carbon-nanotube.h5")
    ap.add_argument("--bin-width", type=float, default=500.0, metavar="K",
                    help="width of each kappa_2 bucket (default: 500)")
    ap.add_argument("--max-per-bin", type=int, default=50, metavar="N",
                    help="rows listed per bucket (default: 50)")
    args = ap.parse_args()

    with h5py.File(args.h5path, "r") as f:
        if GROUP not in f:
            ap.error(f"{args.h5path} has no /{GROUP} group; "
                     f"run condition_est.py first")
        g = f[GROUP]
        indices = g["indices"][:]
        cond_2 = g["cond_2"][:]
        valid = g["valid"][:]
        grid_energy_min = g.attrs.get("grid_energy_min")
        resolution = g.attrs.get("resolution")
        valence = g.attrs.get("valence_band_edge")
        conduction = g.attrs.get("conduction_band_edge")

    def energy_of(idx):
        if grid_energy_min is None or resolution is None:
            return None
        return grid_energy_min + resolution * idx

    indices, cond_2 = indices[valid], cond_2[valid]
    if indices.size == 0:
        print("no valid rows")
        return

    if valence is not None and conduction is not None:
        energies = np.array([energy_of(int(i)) for i in indices])
        in_gap = (energies >= valence) & (energies <= conduction)
        indices, cond_2 = indices[~in_gap], cond_2[~in_gap]
        if indices.size == 0:
            print("no valid rows outside the band gap "
                  f"[{valence:.4f}, {conduction:.4f}] eV")
            return
    else:
        print("[warning] no band edges recorded; showing all indices")

    order = np.argsort(indices)
    indices, cond_2 = indices[order], cond_2[order]

    def energy_of(idx):
        if grid_energy_min is None or resolution is None:
            return None
        return grid_energy_min + resolution * idx

    n_bins = int(np.floor(cond_2.max() / args.bin_width)) + 1
    for b in range(n_bins):
        lo, hi = b * args.bin_width, (b + 1) * args.bin_width
        in_bin = np.flatnonzero((cond_2 >= lo) & (cond_2 < hi))
        if in_bin.size == 0:
            continue
        shown = in_bin[:args.max_per_bin]
        print(f"kappa_2 in [{lo:.2e}, {hi:.2e}): {in_bin.size} rows"
              f"{f', showing {shown.size}' if shown.size < in_bin.size else ''}")
        for i in shown:
            idx = int(indices[i])
            e = energy_of(idx)
            e_str = f"{e:.4f} eV" if e is not None else "unknown"
            print(f"  idx={idx:<6} E={e_str:<14} kappa_2={cond_2[i]:.3e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
List energy indices by kappa_inf bucket, from a condition_est.py output file.

Input
-----
A condition-est analysis file (condition_est.py's output), holding the
condition group written there: indices, cond_2, cond_inf, valid, and the
grid_energy_min / resolution attributes copied from the material file.

Only rows with valid == True are considered; energy(i) = grid_energy_min +
resolution * i.

Output
------
Buckets widen by a factor of 10 above --first-edge, e.g. 0-100, 100-1e3,
1e3-1e4, 1e4-1e5, ..., covering the observed kappa_inf range -- kappa_inf is
the bucketing variable, matching the LU-IR bound, which is stated in the
infinity norm (see mixed_prec_ir/README.md); kappa_2 is listed alongside it.
For each bucket, the index, energy and both
condition numbers of every row falling in it, in index order, unless
--max-per-bin caps it.

The listing is written to --out as well as printed, so a full (uncapped)
sweep over 2000+ indices stays usable without scrolling back through the
terminal.

Usage
-----
    python list_by_kappa.py /scratch/yimili/condition-est/carbon-nanotube.h5
    python list_by_kappa.py .../si-bulk.h5 --first-edge 1000 --max-per-bin 20
    python list_by_kappa.py .../si-bulk.h5 --out si-bulk_by_kappa.txt
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))
import cli

GROUP = "condition"


def bin_edges(first_edge, upper):
    """
    [0, first_edge, ...] widening by a factor of 10, up to and including the
    first edge at or above `upper`.

    e.g. first_edge=100: 0, 100, 1e3, 1e4, 1e5, 1e6, ...
    """
    edges = [0.0, float(first_edge)]
    while edges[-1] < upper:
        edges.append(edges[-1] * 10.0)
    return edges


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help="condition-est analysis file, e.g. "
                             "condition-est/carbon-nanotube.h5")
    ap.add_argument("--first-edge", type=float, default=100.0, metavar="K",
                    help="first bucket boundary above 0; later boundaries "
                         "widen by a factor of 10 (default: 100)")
    ap.add_argument("--max-per-bin", type=int, default=None, metavar="N",
                    help="cap on rows listed per bucket (default: no cap, "
                         "show every row)")
    ap.add_argument("--out", type=str, default=None, metavar="PATH",
                    help="text file the listing is written to, besides being "
                         "printed (default: <material>_by_kappa.txt beside "
                         "the input file)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else \
        Path(args.h5path).with_name(f"{Path(args.h5path).stem}_by_kappa.txt")
    print(f"[output] writing listing to {out_path}")
    out_fh = out_path.open("w")

    def emit(line=""):
        print(line)
        print(line, file=out_fh)

    with h5py.File(args.h5path, "r") as f:
        if GROUP not in f:
            ap.error(f"{args.h5path} has no /{GROUP} group; "
                     f"run condition_est.py first")
        g = f[GROUP]
        indices = g["indices"][:]
        cond_2 = g["cond_2"][:]
        cond_inf = g["cond_inf"][:]
        valid = g["valid"][:]
        grid_energy_min = g.attrs.get("grid_energy_min")
        resolution = g.attrs.get("resolution")
        valence = g.attrs.get("valence_band_edge")
        conduction = g.attrs.get("conduction_band_edge")

    def energy_of(idx):
        if grid_energy_min is None or resolution is None:
            return None
        return grid_energy_min + resolution * idx

    try:
        indices, cond_2, cond_inf = indices[valid], cond_2[valid], cond_inf[valid]
        if indices.size == 0:
            emit("no valid rows")
            return

        if valence is not None and conduction is not None:
            energies = np.array([energy_of(int(i)) for i in indices])
            in_gap = (energies >= valence) & (energies <= conduction)
            indices, cond_2, cond_inf = \
                indices[~in_gap], cond_2[~in_gap], cond_inf[~in_gap]
            if indices.size == 0:
                emit("no valid rows outside the band gap "
                     f"[{valence:.4f}, {conduction:.4f}] eV")
                return
        else:
            emit("[warning] no band edges recorded; showing all indices")

        order = np.argsort(indices)
        indices, cond_2, cond_inf = indices[order], cond_2[order], cond_inf[order]

        edges = bin_edges(args.first_edge, cond_inf.max())
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = np.flatnonzero((cond_inf >= lo) & (cond_inf < hi))
            if in_bin.size == 0:
                continue
            shown = in_bin[:args.max_per_bin]
            emit(f"kappa_inf in [{lo:.2e}, {hi:.2e}): {in_bin.size} rows"
                 f"{f', showing {shown.size}' if shown.size < in_bin.size else ''}")
            for i in shown:
                idx = int(indices[i])
                e = energy_of(idx)
                e_str = f"{e:.4f} eV" if e is not None else "unknown"
                emit(f"  idx={idx:<6} E={e_str:<14} kappa_inf={cond_inf[i]:.3e}  "
                     f"kappa_2={cond_2[i]:.3e}")
    finally:
        out_fh.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
List energy indices by kappa_inf bucket, from a condition_est.py output file.

Input
-----
A condition-est analysis file (condition_est.py's output), holding the
condition group written there: indices, cond_2, cond_inf, cond_skeel_x, valid,
and the grid_energy_min / resolution attributes copied from the material
file. cond_skeel_x is cond(M, x), the componentwise condition number for the
actual stored right-hand side; it is NaN at an index whose right-hand side has
no columns, and it is absent from a file made before condition_est.py wrote
that column, in which case every row prints "n/a" for it instead of the
listing failing.

Only rows with valid == True are considered; energy(i) = grid_energy_min +
resolution * i.

The number of right-hand sides at each index is read from the material file
itself, at E_<idx>/rhs, since condition_est.py does not store it. The material
file's path is taken from the condition group's own `source` attribute, so no
second path needs to be given on the command line. Where the material file
cannot be opened -- moved, or the attribute absent from an older analysis file
-- nrhs is printed as "n/a" for every row rather than the listing failing.

Output
------
Buckets widen by a factor of 10 above --first-edge, e.g. 0-100, 100-1e3,
1e3-1e4, 1e4-1e5, ..., covering the observed kappa_inf range -- kappa_inf is
the bucketing variable, matching the LU-IR bound, which is stated in the
infinity norm (see mixed_prec_ir/README.md); kappa_2 is listed alongside it.
For each bucket, the index, energy, kappa_inf, kappa_2, cond(M, x) and the
number of right-hand sides of every row falling in it, in index order, unless
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


def rhs_counts(source_path, indices):
    """
    {index: number of right-hand side columns at E_<index>/rhs}, read from the
    material file at `source_path`.

    Only each dataset's shape is read, not its data, so this stays cheap even
    over a sweep of thousands of indices. An index with no rhs dataset, or an
    empty one, counts as 0; a 1-D dataset counts as 1, on the same convention
    condition_est.py's load_rhs() uses. Returns an empty dict, rather than
    raising, when `source_path` cannot be opened -- the caller then reports
    "n/a" for every row instead of failing the whole listing over one missing
    or moved file.
    """
    if not source_path:
        return {}
    path = Path(source_path)
    if not path.exists():
        print(f"[warning] material file not found, nrhs will show as n/a: "
              f"{path}")
        return {}

    counts = {}
    with h5py.File(path, "r") as f:
        for idx in indices:
            dataset = f.get(f"E_{int(idx)}/rhs")
            if dataset is None:
                counts[int(idx)] = 0
            elif dataset.ndim == 2:
                counts[int(idx)] = int(dataset.shape[1])
            else:
                counts[int(idx)] = 1 if dataset.shape[-1] > 0 else 0
    return counts


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
        cond_skeel_x = (g["cond_skeel_x"][:] if "cond_skeel_x" in g
                        else np.full(indices.shape, np.nan))
        have_skeel_x = "cond_skeel_x" in g
        valid = g["valid"][:]
        grid_energy_min = g.attrs.get("grid_energy_min")
        resolution = g.attrs.get("resolution")
        valence = g.attrs.get("valence_band_edge")
        conduction = g.attrs.get("conduction_band_edge")
        source = g.attrs.get("source")

    def energy_of(idx):
        if grid_energy_min is None or resolution is None:
            return None
        return grid_energy_min + resolution * idx

    try:
        if not have_skeel_x:
            emit("[warning] no cond_skeel_x column in this file; run "
                 "condition_est.py --only-skeel to add it. Showing n/a for "
                 "cond(M, x) on every row.")

        indices, cond_2, cond_inf, cond_skeel_x = \
            indices[valid], cond_2[valid], cond_inf[valid], cond_skeel_x[valid]
        if indices.size == 0:
            emit("no valid rows")
            return

        if valence is not None and conduction is not None:
            energies = np.array([energy_of(int(i)) for i in indices])
            in_gap = (energies >= valence) & (energies <= conduction)
            indices, cond_2, cond_inf, cond_skeel_x = (
                indices[~in_gap], cond_2[~in_gap], cond_inf[~in_gap],
                cond_skeel_x[~in_gap])
            if indices.size == 0:
                emit("no valid rows outside the band gap "
                     f"[{valence:.4f}, {conduction:.4f}] eV")
                return
        else:
            emit("[warning] no band edges recorded; showing all indices")

        order = np.argsort(indices)
        indices, cond_2, cond_inf, cond_skeel_x = (
            indices[order], cond_2[order], cond_inf[order], cond_skeel_x[order])

        nrhs = rhs_counts(source, indices)

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
                n = nrhs.get(idx, "n/a")
                cx = (f"{cond_skeel_x[i]:.3e}" if have_skeel_x
                     and np.isfinite(cond_skeel_x[i]) else "n/a")
                emit(f"  idx={idx:<6} E={e_str:<14} kappa_inf={cond_inf[i]:.3e}  "
                     f"kappa_2={cond_2[i]:.3e}  cond_skeel_x={cx:<10}  nrhs={n}")
    finally:
        out_fh.close()


if __name__ == "__main__":
    main()

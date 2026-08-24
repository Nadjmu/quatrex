#!/usr/bin/env python3
"""
Solver runtime relative to the SuperLU complex128 baseline.

Input
-----
A material HDF5 file already populated by ``run_bench/run_benchmarks.py`` or
``run_bench/gpu_run_benchmarks.py``. Only timing datasets are read:

    metadata/indices                    the full energy sweep
    E_<idx>/<solver>/<dtype>/time_fact  factorization wall time, seconds
    E_<idx>/<solver>/<dtype>/time_solve triangular-solve wall time, seconds

The file is opened read-only. This script performs no solves and writes
nothing back into the HDF5 file.

Algorithm
---------
1. Read the requested (solver, dtype) timings for every energy index present.
2. For each index, form the speedup of every (solver, dtype) combination
   against the baseline combination at the same index,

       speedup = t_baseline(idx) / t_solver(idx),

   separately for the factorization and the solve stage. Ratios are taken per
   index rather than on sweep aggregates, so a missing index in one series
   cannot bias another.
3. Indices listed in ``metadata/indices`` for which the baseline is absent are
   reported as gaps; these are the energies whose right-hand side has zero
   columns and which the benchmark driver therefore skipped.

Output
------
One figure, factorization time and solve time as columns and each requested
dtype as a row, on a logarithmic speedup axis with the unit line marked.
Values above unity are faster than the baseline. It is written to the Block
Thomas analysis directory, beside the stability and accuracy figures drawn
from the same solver runs.

Usage
-----
    python plot_speedup.py /scratch/yimili/matrices2/hdf5/graphene.h5
    python plot_speedup.py .../graphene.h5 --outdir figures
    python plot_speedup.py .../graphene.h5 --solvers cudss gmres-cupy \
        --suffix _gpu
"""

import argparse
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
from style import (BAND_EDGE_STYLE, SOLVER_STYLE, axis_label, dtype_label,
                   energies_of, legend_handles, mark_band_edges,
                   save_figure)

# Canonical (solver, precision) of the baseline every ratio refers to.
BASELINE = ("superlu", "complex128")
FIELDS = (("time_fact", "Factorization time"), ("time_solve", "Solve time"))


def read_timings(h5path, solvers, dtypes):
    """
    Collect per-index wall times from a material HDF5 file.

    Returns
    -------
    times   : dict[(solver, dtype)][idx] -> {"time_fact": float,
                                             "time_solve": float}
    indices : sorted list of every index in metadata/indices.
    """
    times = {}
    with h5py.File(h5path, "r") as f:
        if "metadata/indices" in f:
            indices = [int(i) for i in f["metadata/indices"][:]]
        else:
            indices = sorted(int(k[2:]) for k in f if k.startswith("E_"))

        for idx in indices:
            for solver in solvers:
                for dtype in dtypes:
                    g = f.get(f"E_{idx}/{cli.h5_group(solver)}/{dtype}")
                    if g is None:
                        continue
                    entry = {}
                    for field, _ in FIELDS:
                        if field in g:
                            entry[field] = float(g[field][()])
                    if entry:
                        times.setdefault((solver, dtype), {})[idx] = entry
    return times, indices


def speedup_series(times, key, field):
    """
    Speedup of `key` over the baseline, at every index where both ran and the
    denominator is strictly positive.

    Returns (indices, speedups) as parallel lists, sorted by index.
    """
    base = times.get(BASELINE, {})
    series = times.get(key, {})
    points = [(idx, base[idx][field] / series[idx][field])
              for idx in sorted(series)
              if idx in base
              and field in base[idx] and field in series[idx]
              and series[idx][field] > 0]
    if not points:
        return [], []
    xs, ys = zip(*points)
    return list(xs), list(ys)


def plot(times, indices, dtypes, attrs, material, out_path):
    """
    Speedup figure with one row per dtype and one column per FIELDS entry;
    returns False if no series could be drawn.

    Splitting dtypes across rows means every panel already carries a single
    precision, so the line style no longer needs to encode it: all lines are
    solid, and the legend lists solvers only.
    """
    keys = [k for k in times if k != BASELINE]
    if not keys:
        print("no (solver, dtype) combination besides the baseline was found")
        return False

    all_values = [y for _, field in ((None, "time_fact"), (None, "time_solve"))
                  for key in keys
                  for y in speedup_series(times, key, field)[1]]
    if not all_values:
        print("no index has both a baseline and a comparison timing")
        return False

    ymin = min(all_values + [1.0]) / 2.0
    ymax = max(all_values + [1.0]) * 2.0

    solved = set(times.get(BASELINE, {}))
    missing = np.array(sorted(set(indices) - solved))

    n_rows = len(dtypes)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12.5, 4.4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    drawn_solvers, drawn_edges = [], []

    have_energy = energies_of(attrs, [0]) is not None
    # Half a step, used to give a skipped index a visible width on either axis.
    half_step = 0.5 * float(attrs["resolution"]) if have_energy else 0.5

    def to_x(values):
        converted = energies_of(attrs, values) if have_energy else None
        return values if converted is None else converted

    base_text = f"SuperLU {BASELINE[1]}"

    for row, dtype in enumerate(dtypes):
        row_keys = [k for k in keys if k[1] == dtype]
        for col, (field, panel_title) in enumerate(FIELDS):
            ax = axes[row, col]
            for solver, _ in row_keys:
                xs, ys = speedup_series(times, (solver, dtype), field)
                if not xs:
                    continue
                _, colour, marker = SOLVER_STYLE.get(solver, (solver, None, None))
                # Marker every point overlaps into a solid smear on a dense
                # sweep; cap the number drawn regardless of how many there are.
                stride = max(1, len(xs) // 25)
                ax.plot(to_x(xs), ys, color=colour, marker=marker,
                        markersize=5, markeredgecolor="white",
                        markeredgewidth=0.6, markevery=stride, lw=1.6,
                        alpha=0.95, zorder=3)
                if solver not in drawn_solvers:
                    drawn_solvers.append(solver)

            ax.axhline(1.0, color="0.2", lw=1.1, ls=":", zorder=2)
            ax.axhspan(1.0, ymax, color="#27AE60", alpha=0.05, zorder=0)
            ax.axhspan(ymin, 1.0, color="#C0392B", alpha=0.05, zorder=0)

            # Contiguous runs of skipped indices are shaded as single spans.
            if len(missing):
                breaks = np.where(np.diff(missing) > 1)[0] + 1
                for run in np.split(missing, breaks):
                    left, right = to_x([run[0], run[-1]])
                    ax.axvspan(left - half_step, right + half_step,
                               color="0.55", alpha=0.25, zorder=1)

            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
            ax.grid(True, which="major", alpha=0.3)
            ax.grid(True, which="minor", alpha=0.12)
            if have_energy:
                drawn_edges = mark_band_edges(ax, attrs, label=False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if row == n_rows - 1:
                ax.set_xlabel(axis_label(have_energy))
            if row == 0:
                ax.set_title(panel_title, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"speedup vs {base_text}\n({dtype_label(dtype)})")

        axes[row, 0].text(0.02, 0.97, f"faster than {base_text}", va="top",
                          fontsize=8, color="#1E8449",
                          transform=axes[row, 0].transAxes)
        axes[row, 0].text(0.02, 0.03, f"slower than {base_text}", va="bottom",
                          fontsize=8, color="#922B21",
                          transform=axes[row, 0].transAxes)

    extra = [(plt.Line2D([], [], color="0.2", lw=1.1, ls=":"),
              f"{base_text} baseline")]
    if len(missing):
        extra.append((plt.Rectangle((0, 0), 1, 1, color="0.55", alpha=0.25),
                      "no right-hand side"))
    # The band edge lines are drawn without labels, since every panel carries
    # them; they are named once here instead. Only the edges that fell inside
    # the swept range were drawn, so only those are named.
    for key in drawn_edges:
        colour, text = BAND_EDGE_STYLE[key]
        extra.append((plt.Line2D([], [], color=colour, ls="--", lw=1.0), text))
    handles, labels = legend_handles(drawn_solvers, [], extra)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04 / n_rows))

    fig.suptitle(f"{material} — runtime relative to {base_text}",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    save_figure(fig, out_path)
    return True


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap)
    cli.add_solver_selection(ap, choices=cli.ALL_SOLVERS,
                             default=cli.ALL_SOLVERS,
                             help="solvers to include; those absent from the "
                                  "file are skipped")
    cli.add_dtypes(ap, choices=cli.COMPLEX_DTYPES,
                   default=("complex128", "complex64"))
    cli.add_output(ap, outdir_default=str(cli.BLOCK_THOMAS_DIR),
                   outdir_help=f"output directory "
                               f"(default: {cli.BLOCK_THOMAS_DIR})")
    ap.add_argument("--suffix", type=str, default="", metavar="TEXT",
                    help="appended to the output filename stem, e.g. '_gpu'")
    args = ap.parse_args()

    args.h5path = Path(args.h5path)
    material = args.material or args.h5path.stem
    out_path = Path(args.outdir) / f"{material}_speedup{args.suffix}.png"

    solvers = list(dict.fromkeys([BASELINE[0]] + list(args.solvers)))
    dtypes = list(dict.fromkeys([BASELINE[1]] + list(args.dtypes)))

    times, indices = read_timings(args.h5path, solvers, dtypes)
    if BASELINE not in times:
        raise SystemExit(f"{args.h5path} has no {BASELINE[0]}/{BASELINE[1]} "
                         f"result to use as a baseline; run "
                         f"run_bench/run_benchmarks.py first")
    attrs = material_metadata(args.h5path)
    if not plot(times, indices, dtypes, attrs, material, out_path):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

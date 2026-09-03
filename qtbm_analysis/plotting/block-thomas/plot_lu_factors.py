#!/usr/bin/env python3
"""
Sparsity and magnitude of the extracted L and U factors, Block Thomas against
SuperLU and UMFPACK.

Input
-----
The ``lu_factors`` group written by ``block-thomas/extract_lu.py`` into its
analysis file:

    lu_factors/E_<idx>/<solver>/<dtype>/{A_eff,L,U}   CSC triplets

A_eff is the matrix the stored factors reconstruct, which is solver dependent;
see the header of ``block-thomas/growth_factor.py``.

Algorithm
---------
No computation is performed. One figure is drawn per (index, dtype): a 3x3
grid, one row per solver (block-thomas, superlu, umfpack, in that order) and
one column per matrix (A_eff, L, U), each panel showing log10 of the entry
magnitude over the full matrix.

Every panel shares one colour scale, so fill-in and factor growth are
comparable both across the row (a solver's L or U brighter than its own
A_eff is growth) and across solvers (more or less fill-in at the same scale).
The scale spans `--dynamic-range` decades below the largest entry drawn, which
keeps entries at the level of the rounding error of the factorization from
occupying half the colour map.

The full matrix is drawn, never a windowed subset. Block boundaries of the
Block Thomas partition are marked on its row only, since that is the structure
the block-bidiagonal factors follow; SuperLU and UMFPACK carry no partition.

Output
------
    <outdir>/<material>_E<idx>_<dtype>_lu.png

one file per (index, dtype) drawn, with whichever of the three solvers are
present in the file. The default output directory is the analysis file's own
directory, so the figures are written beside the data.

Usage
-----
    python plot_lu_factors.py /scratch/yimili/error-analysis-block-thomas/carbon-nanotube.h5
    python plot_lu_factors.py .../carbon-nanotube.h5 --idx 5 --dtypes complex128
    python plot_lu_factors.py .../carbon-nanotube.h5 --energy -3.57
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str((_HERE / "..").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "solvers").resolve()))

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import cli
from factor_io import load_sparse_factor
from style import columns_from_rows, energies_of, save_figure, write_data_report

GROUP = "lu_factors"

# Decades of magnitude below the largest entry drawn that the colour scale
# spans. Twelve covers a double-precision factorization down to the level of
# its own rounding error and no further.
DEFAULT_DYNAMIC_RANGE = 12

ROWS = ("block-thomas", "superlu", "umfpack")
COLUMNS = ("A_eff", "L", "U")
COLUMN_TITLE = {"A_eff": r"$A_{\mathrm{eff}}$", "L": r"$L$", "U": r"$U$"}


def figures(f, indices, dtypes):
    """
    (idx, dtype) pairs to draw, and for each the {solver: group} dict of
    whatever ROWS entries are actually present in the file.
    """
    root = f.get(GROUP)
    if root is None:
        raise SystemExit(f"no /{GROUP} group; "
                         f"run block-thomas/extract_lu.py first")
    per_key = {}
    for key in root:
        if not key.startswith("E_"):
            continue
        idx = int(key[2:])
        if idx not in indices:
            continue
        for stored in root[key]:
            for dtype_name in root[key][stored]:
                solver = cli.from_h5_group(stored, dtype_name)
                if solver not in ROWS:
                    continue
                if dtypes is not None and dtype_name not in dtypes:
                    continue
                per_key.setdefault((idx, dtype_name), {})[solver] = \
                    root[key][stored][dtype_name]
    return sorted(per_key.items())


def block_boundaries(group, n):
    """Block boundaries of the recorded partition, if any."""
    sizes = group.attrs.get("block_sizes")
    if sizes is None:
        return []
    edges = np.cumsum(np.asarray(sizes, dtype=np.int64))
    return [int(e) for e in edges if 0 < e < n]


def plot(groups, idx, dtype_name, attrs, dynamic_range, out_path):
    """One 3x3 figure: rows are solvers present in `groups`, columns A_eff/L/U."""
    rows = [s for s in ROWS if s in groups]

    panels = {}
    for solver in rows:
        group = groups[solver]
        for name in COLUMNS:
            mat = load_sparse_factor(group[name]).toarray()
            # A structural zero is not a small value and must not be drawn at
            # the bottom of the colour scale; it is masked and rendered as
            # background instead. The previous form, log10(|x| + 1e-300), made
            # that choice depend on the storage width: 1e-300 underflows to
            # zero in the float32 that np.abs returns for a complex64 factor,
            # so those panels produced -inf and came out on a white ground,
            # while complex128 panels produced a finite -300 and came out on
            # the dark end of viridis. Same figure, two backgrounds.
            with np.errstate(divide="ignore"):
                logmag = np.log10(np.abs(mat).astype(np.float64))
            panels[(solver, name)] = np.ma.masked_invalid(logmag)

    populated = [float(p.max()) for p in panels.values() if p.count()]
    if not populated:
        raise SystemExit(f"idx {idx} {dtype_name}: every panel is empty")
    vmax = np.ceil(max(populated))
    vmin = vmax - dynamic_range

    # Masked entries (the structural zeros) take the background colour. Values
    # that are merely below vmin keep clamping to the bottom of the colormap,
    # so a small nonzero entry stays distinguishable from no entry at all.
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="white")

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5.2 * len(rows)),
                             squeeze=False)
    for r, solver in enumerate(rows):
        group = groups[solver]
        n = int(group.attrs["n"])
        blocks = block_boundaries(group, n) if solver == "block-thomas" else []
        for c, name in enumerate(COLUMNS):
            ax = axes[r][c]
            ax.set_facecolor("white")
            image = ax.matshow(panels[(solver, name)], cmap=cmap,
                               vmin=vmin, vmax=vmax)
            ax.set_title(COLUMN_TITLE[name], fontsize=12, pad=10)
            for edge in blocks:
                # Grey, not white: the background is white now.
                ax.axhline(edge - 0.5, color="0.45", lw=0.3, alpha=0.6)
                ax.axvline(edge - 0.5, color="0.45", lw=0.3, alpha=0.6)
        # The solver name alone. The material, the energy and the precision are
        # stated in the caption of the figure in the text, and the
        # reconstruction residual is a guard reported in the data file rather
        # than a quantity this figure is about.
        axes[r][0].set_ylabel(cli.label(solver), fontsize=11)

    fig.colorbar(image, ax=axes, label=r"$\log_{10}|\cdot|$",
                 fraction=0.02, pad=0.02)
    save_figure(fig, out_path, dpi=160)

    return _panel_rows(panels, groups, idx, dtype_name, attrs)


def _panel_rows(panels, groups, idx, dtype_name, attrs):
    """
    One summary row per (solver, matrix) drawn: shape, nnz, fill and the
    extreme and median entry magnitude -- the numbers behind a heat-map panel,
    read straight off the arrays plot() already built. The full CSC triplets
    stay in the source file at lu_factors/E_<idx>/<solver>/<dtype>.
    """
    energy = energies_of(attrs, [idx])
    out = []
    for solver in (s for s in ROWS if s in groups):
        resid = float(groups[solver].attrs.get("resid_rel", np.nan))
        for name in COLUMNS:
            p = panels[(solver, name)]                 # log10(|entry|), zeros masked
            logs = p.compressed()
            out.append({
                "idx": idx,
                "energy_eV": np.nan if energy is None else float(energy[0]),
                "dtype": dtype_name,
                "solver": solver,
                "matrix": name,
                "rows": p.shape[0],
                "cols": p.shape[1],
                "nnz": int(logs.size),
                "fill_frac": logs.size / (p.shape[0] * p.shape[1]),
                "abs_min": float(10.0 ** logs.min()) if logs.size else np.nan,
                "abs_median": float(10.0 ** np.median(logs)) if logs.size else np.nan,
                "abs_max": float(10.0 ** logs.max()) if logs.size else np.nan,
                "resid_rel": resid,
            })
    return out


def write_report(h5path, material, out_path, group_attrs, rows, args):
    """The per-panel summary over every (index, dtype) figure, as text."""
    cols = ("idx", "energy_eV", "dtype", "solver", "matrix", "rows", "cols",
            "nnz", "fill_frac", "abs_min", "abs_median", "abs_max", "resid_rel")
    write_data_report(
        out_path,
        title=f"extracted L and U factors  —  {material}",
        source=str(h5path),
        source_attrs=group_attrs,
        config={
            "analysis group": GROUP,
            "figures": f"{material}_E<idx>_<dtype>_lu.png, one per (index, dtype)",
            "figures written": str(len({(r["idx"], r["dtype"]) for r in rows})),
            "solver rows": ", ".join(ROWS),
            "matrices per row": ", ".join(COLUMNS),
            "dynamic range (decades)": f"{args.dynamic_range:g}",
            "precision selection": (" ".join(args.dtypes) if args.dtypes
                                    else "all present"),
        },
        series={"one row per heat-map panel": columns_from_rows(rows, cols)},
        notes=["A_eff is the matrix the stored factors reconstruct, which is "
               "solver dependent (see block-thomas/growth_factor.py).  "
               "resid_rel = ||A_eff - LU|| / ||A_eff||."],
    )


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"block-thomas/extract_lu.py, group {GROUP}")
    cli.add_index_selection(ap, default_all=True)
    ap.add_argument("--energy", type=float, nargs="+", default=None,
                    metavar="EV",
                    help="select energy indices by nearest energy in eV, "
                         "instead of --idx/--start/--end (overrides them if "
                         "both are given); requires the analysis file to "
                         "carry grid_energy_min/resolution, which "
                         "extract_lu.py copies in when the material file has "
                         "it")
    cli.add_dtypes(ap, choices=cli.COMPLEX_DTYPES, default=None,
                   help="restrict to these precisions "
                        "(default: all present in the file)")
    ap.add_argument("--dynamic-range", type=float,
                    default=DEFAULT_DYNAMIC_RANGE, metavar="D",
                    help=f"decades below the largest entry drawn that the "
                         f"colour scale spans "
                         f"(default: {DEFAULT_DYNAMIC_RANGE})")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: the analysis file's directory)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    material = args.material or h5path.stem
    outdir = Path(args.outdir) if args.outdir else h5path.parent

    with h5py.File(h5path, "r") as f:
        group_attrs = dict(f[GROUP].attrs) if GROUP in f else {}
        if args.energy is not None:
            args.idx = cli.index_of_energy(group_attrs, args.energy)
            for e, idx in zip(args.energy, args.idx):
                print(f"energy {e:.4f} eV -> idx {idx}")

        # The indices available are those the extraction wrote, not those a
        # material file holds, so the selection is resolved against this
        # file's own groups.
        stored = sorted(int(k[2:]) for k in f.get(GROUP, {})
                        if k.startswith("E_"))
        indices = cli.resolve_indices(ap, args, stored)
        found = figures(f, set(indices),
                        set(args.dtypes) if args.dtypes else None)
        if not found:
            raise SystemExit("no combination remains after filtering")
        report_rows = []
        for (idx, dtype_name), groups in found:
            report_rows.extend(plot(
                groups, idx, dtype_name, group_attrs, args.dynamic_range,
                outdir / f"{material}_E{idx}_{dtype_name}_lu.png"))

        write_report(h5path, material,
                     outdir / f"{material}_lu_factors_data.txt",
                     group_attrs, report_rows, args)


if __name__ == "__main__":
    main()

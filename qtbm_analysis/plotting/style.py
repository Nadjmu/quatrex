"""
Shared figure conventions for the plotting scripts in this directory.

Purpose
-------
Every figure produced by this project identifies a solver by colour and a
working precision by line style. Centralising both maps here guarantees that
the same solver is drawn with the same colour in the timing figure, the
stability figure and the accuracy figure, so figures may be compared directly.

Contents
--------
SOLVER_STYLE   dict: canonical solver name -> (legend label, colour, marker).
               The keys are the canonical kebab-case names defined in
               ``solvers/cli.py`` (``block-thomas``, ``gmres``, ...), the same
               spelling every command line uses. A script reading an HDF5 file
               converts the stored group name with ``cli.from_h5_group`` first.
DTYPE_STYLE    dict: precision label -> (legend label, line style).
               ``complex32`` is not a NumPy dtype; it is the storage label used
               for the half-precision embedded-real factorizations.
FP16_UNIT_ROUNDOFF
               u = 2^-11 for IEEE binary16, the reference level against which
               half-precision residuals are read.

The energy axis
---------------
Every sweep figure is drawn against energy in eV, not against the energy index.
The index is a position in a grid whose resolution and range are a per-material
choice, so a figure labelled by index cannot be compared across materials or
across two runs of the same material at different resolutions, and the band
edges cannot be marked on it.

The conversion is carried in the metadata of every material and analysis file,
as `grid_energy_min` and `resolution`, and applied by energies_of(). Files
written before those attributes existed carry neither; the axis then falls back
to the index, which axis_label() reports.

Functions
---------
solver_label(key), dtype_label(key)  Legend text, falling back to the raw key.
named_for_legend(solvers)            Solver list minus the -fp16 variants.
legend_handles(solvers, dtypes, ...) Proxy artists for a combined legend.
energies_of(attrs, indices)          Energies in eV, or None if unavailable.
axis_label(have_energy)              Axis label matching what was plotted.
mark_band_edges(ax, attrs, ...)      Vertical lines at the two band edges.
split_gaps(indices, x, *series)      NaN-break series where the sweep skips indices.
sweep_line(n_points, weight)         Line kwargs scaled to the sweep density.
save_figure(fig, path, dpi)          Create the parent directory, write, close.

Data reports
------------
Every plotting script writes a plain-text companion beside its figures, holding
the command that produced them, the provenance the source file records, the
options the figures were drawn under, and every numeric series behind them. The
figures are then reproducible and readable without opening the HDF5 file: the
numbers can be replotted, diffed between runs, or handed to a reader to
interpret directly.

plot_mpir.py and plot_mpperf.py predate this and keep their own hand-written
reports; every other script goes through write_data_report().

plot_provenance()                   The plot command, time, host and versions.
fmt_value(v)                         One scalar / attribute / cell as one line.
kv_lines(mapping, keys)              `name : value` lines, aligned.
columns_from_rows(rows, cols)        List of per-row dicts to {name: array}.
aligned_table(colmap) / tsv_table    A dict of 1-D arrays as text.
write_data_report(path, ...)         Assemble and write the companion file.
"""

import datetime
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no interactive display in batch use
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Solver identity: colour and marker are fixed per solver across all figures.
# Keys are the canonical names of solvers/cli.py.
SOLVER_STYLE = {
    "superlu":               ("SuperLU",              "#555555", "x"),
    "umfpack":               ("UMFPACK",              "#E67E22", "s"),
    "mumps":                 ("MUMPS",                "#27AE60", "^"),
    # cuDSS was previously #16A085, a teal indistinguishable from MUMPS'
    # green at a glance on a line plot; #D81B60 (magenta) is the only unused
    # hue family left once gray/orange/green/purple/dark-red/blue are spoken
    # for by the rest of this table.
    "gmres":                 ("GMRES (SciPy)",        "#8E44AD", "D"),
    "gmres-cupy":            ("GMRES (CuPy)",         "#C0392B", "v"),
    "cudss":                 ("cuDSS",                "#D81B60", "P"),
    "block-thomas":          ("Block Thomas (LU)",    "#2E86AB", "o"),
    "block-thomas-inv":      ("Block Thomas (inv)",   "#9B59B6", "*"),
    "block-thomas-fp16":     ("Block Thomas fp16",    "#2E86AB", "o"),
    "block-thomas-inv-fp16": ("Block Thomas fp16 (inv)", "#9B59B6", "*"),
}

# Working precision: line style only, so precision and solver are separable.
# complex32 is the storage label for the half-precision embedded-real
# factorizations; it is shown verbatim, the same string every figure uses.
DTYPE_STYLE = {
    "complex128": ("complex128", "-"),
    "complex64":  ("complex64",  "--"),
    "complex32":  ("complex32",  ":"),
}

FP16_UNIT_ROUNDOFF = 2.0 ** -11           # 4.883e-4

# Band edge marks. The valence and conduction edge are distinguished by colour,
# matching plotting/materials/bandstructure.py, where the same two levels are drawn.
BAND_EDGE_STYLE = {
    "valence_band_edge": ("tab:blue", "valence band edge"),
    "conduction_band_edge": ("tab:red", "conduction band edge"),
}


def energies_of(attrs, indices):
    """
    Energies in eV of the given indices, or None if the file does not record
    the mapping.

    `attrs` is the attribute dict of a material file's metadata group or of an
    analysis group, both of which carry grid_energy_min and resolution as
    written by make_hdf5.py. The grid is uniform, so

        energy(i) = grid_energy_min + resolution * i.
    """
    if attrs is None:
        return None
    if "grid_energy_min" not in attrs or "resolution" not in attrs:
        return None
    start = float(attrs["grid_energy_min"])
    step = float(attrs["resolution"])
    return start + step * np.asarray(indices, dtype=float)


def axis_label(have_energy):
    """Label of the sweep axis, naming what was actually plotted."""
    return "Energy (eV)" if have_energy else "energy index"


def mark_band_edges(ax, attrs, orientation="vertical", label=True):
    """
    Draw the valence and conduction band edges on an energy axis.

    Only edges recorded in `attrs` are drawn, and only those inside the current
    axis limits: a band edge outside the swept range would otherwise widen the
    axis to accommodate a line carrying no data. A sweep positioned on the
    conduction edge commonly excludes the valence edge for that reason.

    Returns the list of keys actually drawn, which a caller building its own
    legend must use rather than the keys present in `attrs`, so that the legend
    names only lines the figure contains.

    `orientation` is "vertical" for a figure with energy on the x-axis and
    "horizontal" for one with energy on the y-axis. With `label` false the
    lines are drawn without legend entries, for panels that share a legend.
    """
    if not attrs:
        return []
    lo, hi = ax.get_xlim() if orientation == "vertical" else ax.get_ylim()
    drawn = []
    for key, (colour, text) in BAND_EDGE_STYLE.items():
        if key not in attrs:
            continue
        edge = float(attrs[key])
        if not (lo <= edge <= hi):
            continue
        line = ax.axvline if orientation == "vertical" else ax.axhline
        line(edge, color=colour, ls="--", lw=1.0, alpha=0.8, zorder=1.5,
             label=text if label else None)
        drawn.append(key)
    return drawn


def solver_label(key):
    """Legend text for a canonical solver name."""
    return SOLVER_STYLE.get(key, (key, None, None))[0]


def dtype_label(key):
    """Legend text for a precision label."""
    return DTYPE_STYLE.get(key, (key, "-"))[0]


def named_for_legend(solvers):
    """
    Solver names to list in a legend, with the half-precision variants removed.

    ``block-thomas-fp16`` and ``block-thomas-inv-fp16`` are drawn in the same
    colour as their full-precision counterparts, and every figure that shows
    them names the precision on the panel title or the line style, so a
    separate legend entry for them is redundant and, sharing a colour, actively
    misleading. Falls back to the full list if that would leave the legend
    empty.
    """
    kept = [s for s in solvers if not str(s).endswith("-fp16")]
    return kept or list(solvers)


def legend_handles(solvers, dtypes, extra=()):
    """
    Proxy artists for a two-part legend: one entry per solver carrying its
    colour and marker, one entry per precision carrying its line style.

    Parameters
    ----------
    solvers : iterable of canonical solver names, in the order to be listed.
    dtypes  : iterable of precision labels, in the order to be listed.
    extra   : iterable of (artist, label) pairs appended verbatim.

    Returns
    -------
    (handles, labels) : lists suitable for ``Figure.legend``.
    """
    handles, labels = [], []
    for key in solvers:
        label, colour, marker = SOLVER_STYLE.get(key, (key, None, None))
        handles.append(Line2D([], [], color=colour, marker=marker, ls="-",
                              markersize=5, markeredgecolor="white",
                              markeredgewidth=0.6))
        labels.append(label)
    for key in dtypes:
        label, ls = DTYPE_STYLE.get(key, (key, "-"))
        handles.append(Line2D([], [], color="0.3", ls=ls))
        labels.append(label)
    for artist, label in extra:
        handles.append(artist)
        labels.append(label)
    return handles, labels


def split_gaps(indices, x, *series, factor=3.0):
    """
    Insert NaN breaks where the swept index sequence skips a block of energies.

    A QTBM sweep has no solution inside the band gap -- those indices carry no
    right-hand side and are absent from the analysis file -- so the kept
    indices arrive in contiguous runs with wide holes between them. Plotted
    directly, matplotlib bridges each hole with a straight segment that reads
    as data. This returns `x` and every array in `series` with a NaN inserted
    at each hole, so the line breaks there.

    A hole is a step in `indices` larger than `factor` times the median step.
    """
    indices = np.asarray(indices, dtype=float)
    x = np.asarray(x, dtype=float)
    series = [np.asarray(s, dtype=float) for s in series]
    if indices.size < 3:
        return (x, *series)
    step = np.diff(indices)
    at = np.flatnonzero(step > factor * max(float(np.median(step)), 1.0)) + 1
    if at.size == 0:
        return (x, *series)
    return (np.insert(x, at, np.nan),
            *[np.insert(s, at, np.nan) for s in series])


def sweep_line(n_points, weight="primary", marker="."):
    """
    Line kwargs for one series of a sweep figure, scaled to how many points it
    has.

    A full-resolution sweep is thousands of energy indices. Drawn with the
    markers and line weight that suit a strided sweep of a few dozen, every
    series fills in to a solid band and the figure carries no information. Past
    a few hundred points the markers are dropped and the line is made thin and
    slightly translucent so that overlapping series stay individually visible;
    below that the marked style is kept, using `marker` (the caller passes the
    solver's own marker where identity matters).

    `weight` is "primary" for the quantity a panel is about and "secondary" for
    one drawn only for context. Returns a dict to pass to ``Axes.plot`` /
    ``semilogy`` alongside colour and line style.
    """
    dense = n_points > 200
    m = "" if dense else marker
    if weight == "secondary":
        return dict(marker=m, ms=2, lw=0.5 if dense else 1.0,
                    alpha=0.7 if dense else 0.8)
    return dict(marker=m, ms=3, lw=0.6 if dense else 1.3,
                alpha=0.85 if dense else 1.0)


def save_figure(fig, path, dpi=150):
    """Write `fig` to `path`, creating the parent directory, and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Data reports
# ---------------------------------------------------------------------------
_REPORT_RULE = "=" * 78
_REPORT_SUBRULE = "-" * 78

# Attributes that describe the energy grid and the band edges rather than the
# run. Pulled to the top of the metadata section, since they are what a reader
# needs to turn an energy index into eV and place the band edges.
_GRID_KEYS = ("valence_band_edge", "conduction_band_edge", "band_gap",
              "grid_energy_min", "resolution", "grid_points")


def _git_commit():
    """Short HEAD of the repository this file lives in, with ``(dirty)`` when
    the working tree has uncommitted changes, or None outside a checkout."""
    here = str(Path(__file__).resolve().parent)
    try:
        head = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        return head.stdout.strip() + (" (dirty)" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def plot_provenance(argv=None):
    """
    What produced a figure now: the plot command, the time, the machine and the
    versions a figure depends on.

    Distinct from the provenance the source file already carries, which records
    the compute run that wrote the data rather than the plotting of it.
    """
    out = {
        "plot command": " ".join(str(a) for a in (argv or sys.argv)),
        "plotted at": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "plot host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    commit = _git_commit()
    if commit:
        out["git commit"] = commit
    return out


def fmt_value(value):
    """
    One scalar, attribute or table cell as a single line, without NumPy's array
    decoration and with a fixed float format so two reports diff cleanly.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [fmt_value(v) for v in np.asarray(value).ravel().tolist()]
        if len(items) > 32:
            return " ".join(items[:32]) + f" ... (+{len(items) - 32} more)"
        return " ".join(items)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not np.isfinite(f):
            return str(f)
        if f == 0.0 or 1e-4 <= abs(f) < 1e6:
            return f"{f:.6g}"
        return f"{f:.6e}"
    if value is None:
        return ""
    return str(value)


def kv_lines(mapping, keys=None, indent="  "):
    """
    ``name : value`` lines with the names left-aligned to a common width.

    `keys` selects and orders the entries; keys absent from `mapping` are
    skipped. With `keys` None every key is printed in the mapping's own order.
    """
    items = ([(k, mapping[k]) for k in keys if k in mapping] if keys is not None
             else list(mapping.items()))
    if not items:
        return [f"{indent}(none recorded)"]
    width = max(len(str(k)) for k, _ in items)
    return [f"{indent}{str(k):<{width}} : {fmt_value(v)}" for k, v in items]


def columns_from_rows(rows, cols):
    """
    The named columns of a list of per-row dicts as an ordered
    ``{name: 1-D array}`` dict.

    A column whose present values are all strings stays as an object array;
    every other column becomes float64 with a missing entry mapped to NaN. A
    column absent from every row is dropped.
    """
    out = {}
    for name in cols:
        if not any(name in r for r in rows):
            continue
        values = [r.get(name) for r in rows]
        present = [v for v in values if v is not None]
        if present and all(isinstance(v, str) for v in present):
            out[name] = np.array(["" if v is None else str(v) for v in values],
                                 dtype=object)
        else:
            out[name] = np.array(
                [np.nan if v is None else float(v) for v in values],
                dtype=float)
    return out


def _table_cells(colmap):
    names = list(colmap)
    arrays = [np.asarray(colmap[n]).ravel() for n in names]
    length = max((a.size for a in arrays), default=0)
    rows = [[fmt_value(a[i]) if i < a.size else "" for a in arrays]
            for i in range(length)]
    return names, rows


def aligned_table(colmap, indent="  "):
    """A dict of equal-length 1-D arrays as a fixed-width table, a rule under
    the header."""
    names, rows = _table_cells(colmap)
    if not names:
        return [f"{indent}(no columns)"]
    width = [max([len(names[j])] + [len(r[j]) for r in rows])
             for j in range(len(names))]
    out = [indent + "  ".join(names[j].ljust(width[j])
                              for j in range(len(names))),
           indent + "  ".join("-" * width[j] for j in range(len(names)))]
    out += [indent + "  ".join(r[j].ljust(width[j]) for j in range(len(names)))
            for r in rows]
    return out


def tsv_table(colmap):
    """The same columns tab-separated, header first, so nothing is lost to the
    fixed-width rounding of the widths."""
    names, rows = _table_cells(colmap)
    return ["\t".join(names)] + ["\t".join(r) for r in rows]


def write_data_report(path, *, title, source, series, source_attrs=None,
                      config=None, notes=None, argv=None):
    """
    Write the plain-text companion to a script's figures, to `path`.

    Parameters
    ----------
    path         : the report file. Its parent is created; an existing file is
                   overwritten, since it describes one plotting run.
    title        : short heading, e.g. ``"growth_factor  —  graphene"``.
    source       : the input file(s), a string or a list of strings.
    series       : ``{name: colmap}``, each colmap a dict of equal-length 1-D
                   arrays. One aligned table per entry in section 4, and the
                   same columns tab-separated in section 5.
    source_attrs : the HDF5 group / metadata attributes of the source, dumped
                   verbatim in section 3 so the compute run's provenance, the
                   band edges and the energy grid travel with the numbers.
    config       : dict of the choices this figure set was drawn under --
                   solvers and precisions kept, defaults excluded, thresholds,
                   the figures written.
    notes        : extra lines appended as section 6.

    Returns the path written.
    """
    path = Path(path)
    sources = [source] if isinstance(source, str) else list(source)

    lines = [_REPORT_RULE, title, _REPORT_RULE,
             f"Data behind the figures in this directory, written by "
             f"{Path(sys.argv[0]).name}.",
             "1 how it was plotted, 2 the options, 3 the source attributes, "
             "4 the series,",
             "5 the same tab-separated, and any notes.", ""]

    def section(number, name):
        lines.extend([_REPORT_SUBRULE, f"{number}. {name}", _REPORT_SUBRULE])

    section(1, "HOW THIS FIGURE SET WAS PRODUCED")
    lines.extend(kv_lines({"source": "  ".join(sources), **plot_provenance(argv)}))
    lines.append("")

    section(2, "CONFIGURATION")
    lines.extend(kv_lines(config) if config else ["  (no options recorded)"])
    lines.append("")

    section(3, "SOURCE FILE METADATA")
    if source_attrs:
        grid = {k: source_attrs[k] for k in _GRID_KEYS if k in source_attrs}
        rest = {k: v for k, v in source_attrs.items() if k not in grid}
        if grid:
            lines.append("  energy grid and band edges  "
                         "(energy(i) = grid_energy_min + resolution * i):")
            lines.extend(kv_lines(grid, indent="    "))
            if rest:
                lines.append("")
        if rest:
            lines.append("  every other recorded attribute:")
            lines.extend(kv_lines(rest, indent="    "))
    else:
        lines.append("  (the source records no attributes)")
    lines.append("")

    section(4, "SERIES  (the numbers behind each figure input)")
    for name, colmap in series.items():
        n_rows = max((np.asarray(v).size for v in colmap.values()), default=0)
        lines.append(f"  [{name}]  {n_rows} rows, {len(colmap)} columns")
        lines.extend(aligned_table(colmap))
        lines.append("")

    section(5, "SERIES AS TSV")
    for name, colmap in series.items():
        lines.append(f"## {name}")
        lines.extend(tsv_table(colmap))
        lines.append("")

    if notes:
        section(6, "NOTES")
        lines.extend([notes] if isinstance(notes, str) else list(notes))
        lines.append("")

    lines.append(_REPORT_RULE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return path

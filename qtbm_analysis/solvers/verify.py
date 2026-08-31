#!/usr/bin/env python3
"""
Integrity and completeness checks for the files the pipeline writes.

Every stage of the pipeline mutates or creates an HDF5 file, and several of the
ways a stage can go wrong are silent: a killed writer truncates a file, an
excluded solver leaves a hole that only shows up as a short curve, and a
registry edit after stage 2 leaves the packed band edges stale. These are the
checks that catch those, collected here so a run can be validated without
writing a throwaway script each time.

Actions
-------
material    Walk a material file and report, per (solver, dtype), how many of
            the indices with a non-empty right-hand side carry a stored
            result. The walk itself is the corruption check: a file truncated
            by a SIGKILL raises here rather than several stages later.

analysis    Report the groups of a stage-4 analysis file: row counts, the
            solver/dtype/norm coverage, the fraction of non-finite entries per
            column, and the band edges actually stored. Compare the non-finite
            fractions against a material already known good rather than
            reading them as absolute; several columns are legitimately NaN for
            solver rows that do not define them.

band-edges  Rewrite metadata/{valence,conduction,band_gap} in the material
            file and any analysis file, from cli.MATERIALS. make_hdf5.py
            prefers the values the export carries and writes them once at pack
            time, so editing the registry afterwards does not reach a file that
            is already packed, and every analysis written from it inherits the
            stale pair. There is no metadata-only mode in make_hdf5.py; this is
            it.

Usage
-----
    python verify.py material graphene
    python verify.py material /scratch/yimili/matrices2/hdf5/graphene.h5
    python verify.py analysis /scratch/yimili/condition-est/graphene.h5
    python verify.py analysis /scratch/yimili/error-analysis-block-thomas/*.h5
    python verify.py band-edges graphene

A material name is resolved through cli.material_h5; anything containing a
path separator is taken as a path.
"""

import collections
import sys
from pathlib import Path

import h5py
import numpy as np

import cli

RAW = ("M", "Sigma", "rhs")

# Copied into every stage-4 group by factor_io.material_metadata.
EDGE_KEYS = ("valence_band_edge", "conduction_band_edge", "band_gap")


def resolve(name):
    """Material name or explicit path -> Path."""
    return Path(name) if ("/" in name or name.endswith(".h5")) \
        else cli.material_h5(name)


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------
def check_material(path):
    """Corruption walk plus per-(solver, dtype) coverage. True if complete."""
    print("=" * 72)
    print(path)
    with h5py.File(path, "r") as f:
        # Raises on a truncated file, which is the point of doing it first.
        f.visititems(lambda name, obj: None)
        print("structure OK")

        indices = cli.available_indices(f, require="M")
        nonzero = [i for i in indices if f[f"E_{i}/rhs"].shape[-1] > 0]
        print(f"indices {len(indices)}, non-empty rhs {len(nonzero)}")
        if "metadata" in f:
            attrs = {k: f["metadata"].attrs[k] for k in EDGE_KEYS
                     if k in f["metadata"].attrs}
            print(f"band edges {attrs}")

        counts = collections.Counter()
        for idx in nonzero:
            group = f[f"E_{idx}"]
            for solver in group:
                if solver in RAW:
                    continue
                for dtype in group[solver]:
                    counts[(solver, dtype)] += 1

    complete = True
    for key in sorted(counts):
        missing = len(nonzero) - counts[key]
        flag = "" if not missing else f"   <-- {missing} MISSING"
        complete &= not missing
        print(f"   {key[0]:<18} {key[1]:<11} {counts[key]:5d} / "
              f"{len(nonzero)}{flag}")
    if not counts:
        print("   no solver results stored")
        complete = False
    return complete


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _decoded(values):
    return [v.decode() if isinstance(v, bytes) else v for v in values]


def check_analysis(path):
    """Row counts, coverage and non-finite fractions of every group."""
    print("=" * 72)
    print(path)
    with h5py.File(path, "r") as f:
        for name in f:
            group = f[name]
            if not isinstance(group, h5py.Group):
                continue
            columns = [c for c in group if isinstance(group[c], h5py.Dataset)]
            if not columns:
                # lu_factors holds per-index subgroups, not columns.
                print(f"  {name:<18} {len(group)} subgroups")
                continue
            print(f"  {name:<18} rows={group[columns[0]].shape[0]}")

            for key in ("solver", "dtype", "norm"):
                if key in columns:
                    print(f"      {key:<7} {sorted(set(_decoded(group[key][:])))}")
            for key in ("idx", "indices"):
                if key in columns:
                    values = group[key][:]
                    print(f"      {key:<7} {len(set(values.tolist()))} unique "
                          f"{values.min()}..{values.max()}")
            if "valid" in columns:
                valid = group["valid"][:]
                print(f"      valid   {int(np.sum(valid))} / {len(valid)}")

            nonfinite = []
            for column in columns:
                data = group[column]
                if data.dtype.kind != "f" or data.ndim != 1:
                    continue
                fraction = float(np.mean(~np.isfinite(data[:])))
                if fraction:
                    nonfinite.append(f"{column} {100 * fraction:.0f}%")
            print("      non-finite: "
                  + (", ".join(nonfinite) if nonfinite else "none"))
            edges = {k: float(group.attrs[k]) for k in EDGE_KEYS
                     if k in group.attrs}
            if edges:
                print(f"      edges: {edges}")
    return True


# ---------------------------------------------------------------------------
# band-edges
# ---------------------------------------------------------------------------
def restamp_band_edges(material):
    """Rewrite the band-edge attrs of every file holding this material."""
    wanted = cli.band_edge_attrs(cli.material(material))
    if not wanted:
        raise SystemExit(f"{material}: cli.MATERIALS records no band edges")
    print(f"registry edges: {wanted}")

    paths = [cli.material_h5(material)] + [
        cli.analysis_h5(directory, material)
        for directory in (cli.CONDITION_DIR, cli.BLOCK_THOMAS_DIR,
                          cli.NON_NORMAL_DIR, cli.MIXED_PREC_DIR)]

    for path in paths:
        if not Path(path).exists():
            print(f"  skip (absent) {path}")
            continue
        with h5py.File(path, "a") as f:
            targets = [g for g in f if isinstance(f[g], h5py.Group)
                       and (g == "metadata" or "valence_band_edge" in f[g].attrs)]
            for name in targets:
                old = {k: float(f[name].attrs[k]) for k in wanted
                       if k in f[name].attrs}
                for key, value in wanted.items():
                    f[name].attrs[key] = value
                print(f"  {path}::{name}  {old} -> {wanted}")
    return True


def main():
    ap = cli.new_parser(__doc__)
    ap.add_argument("action", choices=("material", "analysis", "band-edges"))
    ap.add_argument("targets", nargs="+", metavar="NAME",
                    help="material name or HDF5 path, one or more")
    args = ap.parse_args()

    ok = True
    for target in args.targets:
        if args.action == "material":
            ok &= check_material(resolve(target))
        elif args.action == "analysis":
            ok &= check_analysis(Path(target))
        else:
            ok &= restamp_band_edges(target)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

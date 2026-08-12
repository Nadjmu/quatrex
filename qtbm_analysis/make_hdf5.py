"""
Consolidation of a material's exported matrices into one HDF5 file.

Purpose
-------
The export scripts write one file per operator per energy index, which is
convenient to produce and inconvenient to analyse. This packs a whole material
into a single HDF5 file, which becomes the sole input of every benchmark and
analysis script in this project and, because the solver results are appended
back into it, also their output.

Input
-----
A directory of per-energy .npz and .npy files, as written by
export_qtbm_systems.py or main3.py, plus the global arrays where they exist.
It defaults to cli.EXPORT_DIR, the stage 1 output directory.

Output
------
    <outdir>/<material>.h5, cli.HDF5_DIR by default
    ├── global/          H, S, singular values, condition numbers,
    │                    spectrum_bare
    ├── metadata/        indices, energies
    │                    attrs: valence_band_edge, conduction_band_edge,
    │                           band_gap, grid resolution and window
    └── E_<idx>/         one group per energy point
        ├── M/           system matrix, CSC triplet
        ├── Sigma/       contact self-energy, CSC triplet
        ├── rhs          right-hand side, dense
        └── spectrum     dense

Solver results are later appended by solvers/factor_io.py as siblings of M
inside each E_<idx> group; see its module docstring for that layout.

Energies and band edges
-----------------------
The energy grid is not recomputed here. Stage 1 saves the grid it assembled the
matrices from, and this reads that array back, so the energies recorded against
an index cannot disagree with the matrix stored under it. The grid in
cli.MATERIALS is used only when a run predates that file, and the resulting
energies are then correct only if it has not been edited since.

The band edges are read from the same directory and fall back to cli.MATERIALS.
Where both exist and disagree, the exported value is kept, since it is the one
the matrices were assembled with, and the difference is reported.

Neither the index count nor its range is assumed: the indices present are
discovered from the exported files.

Usage
-----
    python make_hdf5.py --all
    python make_hdf5.py --material carbon-nanotube
    python make_hdf5.py --material graphene --start 0 --end 200
    python make_hdf5.py /scratch/yimili/matrices --all     # the older export
"""

import argparse
import re
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import h5py

sys.path.insert(0, str((Path(__file__).resolve().parent / "solvers").resolve()))

import cli


def discover_indices(src):
    """
    Energy indices exported into `src`, from the M_E_<idx>.npz filenames.

    The index set is whatever stage 1 produced; no count and no contiguous
    range is assumed.
    """
    found = []
    for path in src.glob("M_E_*.npz"):
        match = re.fullmatch(r"M_E_(\d+)\.npz", path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def load_scalar(src, *names):
    """First of `names` present in `src` as a float, or None."""
    for name in names:
        path = src / f"{name}.npy"
        if path.exists():
            return float(np.load(path))
    return None


def resolve_band_edges(src, material, mat):
    """
    Valence and conduction band edge for one material, either may be None.

    The values stage 1 exported take precedence over cli.MATERIALS, since they
    are the ones the matrices were assembled with. A disagreement is reported
    rather than resolved silently: it means the registry was edited after the
    export, and the exported data belongs to the older value.

    Returns (valence, conduction).
    """
    exported = {
        "valence_band_edge": load_scalar(src, "valence_band_edge"),
        "conduction_band_edge": load_scalar(src, "conduction_band_edge",
                                            "band_edge"),
    }
    registry = cli.band_edge_attrs(mat)

    resolved = {}
    for key, value in exported.items():
        known = registry.get(key)
        if value is None:
            resolved[key] = known
            if known is None:
                print(f"     [warn] {material}: {key} is set neither in the "
                      f"export directory nor in cli.MATERIALS")
        else:
            resolved[key] = value
            if known is not None and not np.isclose(known, value):
                print(f"     [warn] {material}: exported {key} = {value} but "
                      f"cli.MATERIALS has {known}; keeping the exported value, "
                      f"which is the one the matrices were assembled with")
    return resolved["valence_band_edge"], resolved["conduction_band_edge"]


def resolve_energies(src, material, indices, mat):
    """
    Energy in eV of each index in `indices`.

    Read from the grid stage 1 saved. The grid in cli.MATERIALS is the fallback
    for exports that predate that file; it reproduces the sweep only if it has
    not been edited since, which is why the fallback is reported.
    """
    path = src / "energies.npy"
    if path.exists():
        grid = np.asarray(np.load(path), dtype=float)
    else:
        if mat is None or mat.grid is None:
            raise ValueError(
                f"{material}: neither {path.name} nor a grid in cli.MATERIALS "
                f"is available, so the energies cannot be determined. Set "
                f"grid=EnergyGrid(start=..., end=..., resolution=...) or "
                f"re-export.")
        print(f"     [warn] {material}: no energies.npy; reconstructing the "
              f"grid from cli.MATERIALS. This is correct only if it is "
              f"unchanged since the export.")
        grid = mat.grid.energies()

    if indices and max(indices) >= len(grid):
        raise ValueError(
            f"{material}: index {max(indices)} was exported but the grid holds "
            f"only {len(grid)} energies. The export directory mixes runs made "
            f"with different grids, or the grid parameters were changed after "
            f"the export.")

    # The grid is uniform by construction, so its first energy and its step
    # fix energy(i) for every index without storing the array downstream.
    resolution = float(grid[1] - grid[0]) if len(grid) > 1 else float("nan")
    return grid[list(indices)], len(grid), float(grid[0]), resolution


def store_sparse(group, name, mat):
    """Store a scipy sparse matrix as CSC components in an HDF5 subgroup."""
    mat = mat.tocsc()
    g = group.create_group(name)
    g.create_dataset("data", data=mat.data, compression="gzip")
    g.create_dataset("indices", data=mat.indices, compression="gzip")
    g.create_dataset("indptr", data=mat.indptr, compression="gzip")
    g.attrs["shape"] = mat.shape
    g.attrs["format"] = "csc"


def build_material(folder: Path, material: str, out_dir: Path,
                   indices=None):
    """
    Pack one material's exported files into a single HDF5 file.

    `indices` restricts the energy indices packed; the default is every index
    found in the export directory.
    """
    src = folder / material
    if not src.exists():
        print(f"  [SKIP] {material}: folder not found")
        return

    available = discover_indices(src)
    if not available:
        print(f"  [SKIP] {material}: no M_E_<idx>.npz in {src}")
        return
    if indices is None:
        indices = available
    else:
        missing = sorted(set(indices) - set(available))
        indices = [i for i in indices if i in set(available)]
        if missing:
            print(f"     [warn] {material}: {len(missing)} requested indices "
                  f"are not exported, e.g. {missing[:10]}")
        if not indices:
            print(f"  [SKIP] {material}: no requested index is exported")
            return

    idx_arr = np.array(indices)
    mat = cli.MATERIALS.get(material)
    valence, conduction = resolve_band_edges(src, material, mat)
    energies, grid_points, grid_energy_min, resolution = resolve_energies(
        src, material, indices, mat)

    out_path = out_dir / f"{material}.h5"
    print(f"--- {material} -> {out_path}")
    print(f"     {len(indices)} indices, {indices[0]}..{indices[-1]} "
          f"of a {grid_points}-point grid, "
          f"E = {energies[0]:.4f}..{energies[-1]:.4f} eV")
    print(f"     valence band edge = {valence}, "
          f"conduction band edge = {conduction}")

    with h5py.File(out_path, "w") as f:
        f.attrs["material"] = material
        f.attrs["n_energies"] = len(indices)
        f.attrs["idx_start"] = int(idx_arr[0])
        f.attrs["idx_end"] = int(idx_arr[-1])

        # ---- metadata ----
        meta = f.create_group("metadata")
        meta.create_dataset("indices", data=idx_arr)
        meta.create_dataset("energies", data=energies)

        # Band edges, in eV. Recorded here rather than derived downstream, so
        # that a figure can mark them without consulting the registry, and so
        # that a file stays interpretable if the registry is later edited.
        if valence is not None:
            meta.attrs["valence_band_edge"] = float(valence)
        if conduction is not None:
            meta.attrs["conduction_band_edge"] = float(conduction)
        if valence is not None and conduction is not None:
            meta.attrs["band_gap"] = float(conduction - valence)
        # The grid, as the two numbers that define it. energy(i) is
        # grid_energy_min + resolution * i for any index i of the full grid,
        # which is what lets a figure with an index axis be labelled in eV.
        meta.attrs["grid_points"] = int(grid_points)
        meta.attrs["grid_energy_min"] = float(grid_energy_min)
        meta.attrs["resolution"] = float(resolution)

        # ---- global arrays ----
        glob = f.create_group("global")
        global_files = {
            "H": "H.npy",
            "S": "S.npy",
            "max_singular_values": "max_singular_values.npy",
            "min_singular_values": "min_singular_values.npy",
            "condition_full_svd": "condition_full_svd.npy",
            "condition_bare": "condition_bare.npy",
            "spectrum_bare": "spectrum_bare.npy",
        }
        for key, fname in global_files.items():
            p = src / fname
            if p.exists():
                glob.create_dataset(key, data=np.load(p), compression="gzip")
            else:
                print(f"     [warn] missing global file {fname}")

        # ---- per-energy groups ----
        missing = 0
        for idx in indices:
            pM = src / f"M_E_{idx}.npz"
            pS = src / f"Sigma_E_{idx}.npz"
            pr = src / f"rhs_E_{idx}.npy"
            pw = src / f"spectrum_M_E_{idx}.npy"

            gE = f.create_group(f"E_{idx}")

            if pM.exists():
                store_sparse(gE, "M", sp.load_npz(pM))
            else:
                missing += 1
            if pS.exists():
                store_sparse(gE, "Sigma", sp.load_npz(pS))
            else:
                missing += 1
            if pr.exists():
                gE.create_dataset("rhs", data=np.load(pr), compression="gzip")
            if pw.exists():
                gE.create_dataset("spectrum", data=np.load(pw), compression="gzip")

        print(f"     done ({len(indices)} indices, missing sparse files: {missing})")


def exported_materials(folder):
    """Subdirectories of `folder` holding at least one exported matrix."""
    return sorted(p.name for p in folder.iterdir()
                  if p.is_dir() and discover_indices(p))


def main():
    parser = cli.new_parser(__doc__)
    parser.add_argument("folder", type=str, nargs="?",
                        default=str(cli.EXPORT_DIR),
                        help=f"root directory containing the per-material "
                             f"subdirectories (default: {cli.EXPORT_DIR})")
    parser.add_argument("--material", type=str, metavar="NAME",
                        help="pack this material only")
    parser.add_argument("--all", action="store_true",
                        help="pack every material found in the folder")
    parser.add_argument("--outdir", type=str, default=str(cli.HDF5_DIR),
                        metavar="DIR",
                        help=f"output directory for the .h5 files "
                             f"(default: {cli.HDF5_DIR})")
    cli.add_index_selection(parser)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: folder '{folder}' does not exist.")
        sys.exit(1)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = exported_materials(folder)
        if not targets:
            print(f"Error: no exported material found in {folder}.")
            sys.exit(1)
    elif args.material:
        targets = [args.material]
    else:
        found = ", ".join(exported_materials(folder)) or "none"
        print(f"Specify --material <name> or --all. "
              f"Materials exported in {folder}: {found}")
        sys.exit(1)

    # With neither --idx nor --start/--end, every exported index is packed;
    # build_material discovers them per material.
    requested = None
    if args.idx is not None or args.start is not None:
        requested = cli.resolve_indices(parser, args)

    for material in targets:
        build_material(folder, material, out_dir, requested)

    print(f"\nHDF5 files written to: {out_dir}")


if __name__ == "__main__":
    main()
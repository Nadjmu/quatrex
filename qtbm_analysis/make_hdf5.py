"""
Pack a material's matrix dataset into a single HDF5 file.

Structure:
    material.h5
    ├── global/          H, S, singular values, condition numbers, spectrum_bare
    ├── metadata/        indices, energies
    └── E_{idx}/         one group per energy point
        ├── M/           sparse (csc)
        ├── Sigma/       sparse (csc)
        ├── rhs          dense
        └── spectrum     dense

Usage:
    python build_hdf5.py /scratch/yimili/matrices2 --material carbon-nanotube
    python build_hdf5.py /scratch/yimili/matrices2 --all
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import h5py


MATERIALS = {
    "carbon-nanotube": -3.6,
    "carbon-chain": -2.4,
    "si-bulk": -2.4,
    "graphene": -2.4,
}
OFFSET2 = 0.005
RES = 0.005


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
                    start: int = 0, end: int = 401):
    src = folder / material
    if not src.exists():
        print(f"  [SKIP] {material}: folder not found")
        return

    indices = list(range(start, end + 1))
    idx_arr = np.array(indices)
    band_edge = MATERIALS[material]
    energies = band_edge - 1 - OFFSET2 + RES * idx_arr.astype(float)

    out_path = out_dir / f"{material}.h5"
    print(f"--- {material} -> {out_path}")

    with h5py.File(out_path, "w") as f:
        f.attrs["material"] = material
        f.attrs["n_energies"] = len(indices)
        f.attrs["idx_start"] = start
        f.attrs["idx_end"] = end

        # ---- metadata ----
        meta = f.create_group("metadata")
        meta.create_dataset("indices", data=idx_arr)
        meta.create_dataset("energies", data=energies)

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


def main():
    parser = argparse.ArgumentParser(description="Pack matrix datasets into HDF5")
    parser.add_argument("folder", type=str, help="Root folder containing material subfolders")
    parser.add_argument("--material", type=str, help="Single material to pack")
    parser.add_argument("--all", action="store_true", help="Pack all known materials")
    parser.add_argument("--out", type=str, default=None,
                         help="Output dir for .h5 files (default: <folder>/hdf5)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=401)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: folder '{folder}' does not exist.")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else folder / "hdf5"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = list(MATERIALS)
    elif args.material:
        targets = [args.material]
    else:
        print("Specify --material <name> or --all")
        sys.exit(1)

    for material in targets:
        build_material(folder, material, out_dir, args.start, args.end)

    print(f"\nHDF5 files written to: {out_dir}")


if __name__ == "__main__":
    main()
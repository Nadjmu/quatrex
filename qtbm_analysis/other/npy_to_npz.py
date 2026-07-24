"""
Convert .npy matrix files to sparse .npz format.

Usage:
    python npy_to_npz.py /path/to/folder
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def convert_to_sparse_npz(npy_path: Path, out_dir: Path) -> None:
    """Load a .npy file, convert to sparse CSC, save as .npz."""
    matrix = np.load(npy_path)

    if matrix.ndim != 2:
        print(f"  [SKIP] {npy_path.name} — not a 2D matrix (shape: {matrix.shape})")
        return

    sparse_matrix = sp.csc_matrix(matrix)
    out_path = out_dir / (npy_path.stem + ".npz")
    sp.save_npz(out_path, sparse_matrix)

    nnz = sparse_matrix.nnz
    total = matrix.size
    print(f"  [OK] {npy_path.name} -> {out_path.name}  "
          f"(shape: {matrix.shape}, nnz: {nnz}, sparsity: {nnz/total*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert M_E_*.npy and Sigma_E_*.npy files to sparse .npz"
    )
    parser.add_argument("folder", type=str, help="Path to folder containing .npy files")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--end", type=int, default=401, help="End index inclusive (default: 401)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: folder '{folder}' does not exist.")
        sys.exit(1)

    # Create output folder
    out_dir = folder / "sparse"
    out_dir.mkdir(exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    indices = range(args.start, args.end + 1)
    prefixes = ["M_E", "Sigma_E"]

    missing = []
    converted = 0

    for prefix in prefixes:
        print(f"--- Processing {prefix}_*.npy ---")
        for i in indices:
            npy_path = folder / f"{prefix}_{i}.npy"
            if not npy_path.exists():
                missing.append(npy_path.name)
                continue
            convert_to_sparse_npz(npy_path, out_dir)
            converted += 1
        print()

    print(f"Done. Converted: {converted} files.")
    if missing:
        print(f"Missing ({len(missing)} files): {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))


if __name__ == "__main__":
    main()
"""
Delete M_E_{idx}.npy and Sigma_E_{idx}.npy files for idx in [start, end].

Usage:
    python delete_npy.py /path/to/folder
    python delete_npy.py /path/to/folder --start 0 --end 401
    python delete_npy.py /path/to/folder --dry-run
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Delete M_E and Sigma_E .npy files")
    parser.add_argument("folder", type=str, help="Path to folder containing .npy files")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--end", type=int, default=401, help="End index inclusive (default: 401)")
    parser.add_argument("--dry-run", action="store_true", help="Print files that would be deleted without deleting")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: folder '{folder}' does not exist.")
        sys.exit(1)

    prefixes = ["M_E", "Sigma_E"]
    indices = range(args.start, args.end + 1)

    deleted = 0
    missing = 0

    if args.dry_run:
        print("DRY RUN — no files will be deleted\n")

    for prefix in prefixes:
        for idx in indices:
            path = folder / f"{prefix}_{idx}.npy"
            if path.exists():
                if args.dry_run:
                    print(f"  would delete: {path}")
                else:
                    path.unlink()
                deleted += 1
            else:
                missing += 1

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{action}: {deleted}   Not found: {missing}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
One-shot check of bench() on a single (material, index).

Purpose
-------
single_solve.py calls the solver classes directly, so it never exercises
bench_all.bench() itself and therefore never prints the residual/backward-error
fields bench() computes (res, nbe_1, nbe_2, nbe_inf, cbe; see bench_all
.backward_errors). This is the diagnostic for that path: it calls bench()
exactly as run_benchmarks.py does, with save=False, so nothing is written to
the material file and it may be run freely.

Input
-----
    material   key of cli.MATERIALS; its .h5 file must already exist
               (cli.material_h5(material), written by make_hdf5.py)
    --idx      energy index to read from that file (default: 0); must have a
               non-empty right-hand side

Output
------
The same per-solver report bench() always prints, nothing else. Nothing is
written to disk.

Usage
-----
    python check_bench.py carbon-nanotube
    python check_bench.py graphene --idx 500 --solvers superlu block-thomas
"""

import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / ".." / "solvers").resolve()))

import numpy as np
import h5py

import cli
from bench_all import bench
from run_benchmarks import load_sparse, SOLVERS
from solver_classes import block_sizes_from_matrix


def main():
    ap = cli.new_parser(__doc__)
    ap.add_argument("material", type=str, help="key of cli.MATERIALS")
    ap.add_argument("--idx", type=int, default=0,
                    help="energy index to read (default: 0)")
    cli.add_solver_selection(ap, choices=cli.ALL_SOLVERS, default=list(SOLVERS))
    cli.add_dtypes(ap)
    args = ap.parse_args()

    h5path = cli.material_h5(args.material)
    with h5py.File(h5path, "r") as f:
        key = f"E_{args.idx}"
        if key not in f:
            raise SystemExit(f"{key} not in {h5path}")
        M = load_sparse(f[f"{key}/M"])
        rhs = f[f"{key}/rhs"][:]

    if rhs.shape[-1] == 0:
        raise SystemExit(f"{key}/rhs has zero columns; pick another --idx")

    bs = block_sizes_from_matrix(M)
    print(f"{args.material} idx={args.idx}: partition {len(bs)} blocks, "
          f"sizes {min(bs)}..{max(bs)}")

    dtypes = tuple(getattr(np, d) for d in args.dtypes)
    results = bench(M, rhs, args.idx, bs, dtypes=dtypes, h5file=None,
                    save=False, solvers=args.solvers)

    print("\nkeys recorded:", [k for k in results if k not in ("idx", "dtypes")])


if __name__ == "__main__":
    main()

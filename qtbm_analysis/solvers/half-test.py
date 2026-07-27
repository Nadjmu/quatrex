"""
sweep_half_blockthomas.py -- run BlockThomasFP16 over a range of energy
indices in a materials HDF5 file and record/plot accuracy metrics.

Usage:
    python sweep_half_blockthomas.py --h5path /scratch/yimili/matrices/hdf5/carbon-nanotube.h5 \
        --start 0 --end 401 --bs 32

Outputs (all under <script_dir>/plots/, filenames tagged with material name):
    plots/<material>_relres_fwderr.png
    plots/<material>_forward_accuracy.png
    plots/<material>_metrics.csv        <- feed this back for analysis

The CSV starts with '#'-prefixed metadata lines (material, h5path, block size,
requested range, completed count, failed indices) followed by the header row
and one data row per successful index. pandas.read_csv(path, comment='#')
skips the metadata automatically; if you paste the file's text directly into
chat I can read the metadata too.

Columns: idx, relres_fp16, fwd_err_fp16_vs_c128, fwd_err_c64_vs_c128, cond_full_svd
"""

import argparse
import csv
import os

import h5py
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")          # headless-safe; remove if running with a display
import matplotlib.pyplot as plt

from solver_classes import extract_blocks_sparse   # or paste from blockthomas_fp16
from half_blockthomas import BlockThomasFP16


def parse_args():
    p = argparse.ArgumentParser(description="Sweep BlockThomasFP16 accuracy over energy indices.")
    p.add_argument("--h5path", type=str,
                    default="/scratch/yimili/matrices/hdf5/carbon-nanotube.h5",
                    help="Path to the materials HDF5 file.")
    p.add_argument("--material", type=str, default=None,
                    help="Material tag for output filenames. Defaults to the h5 filename stem.")
    p.add_argument("--bs", type=int, default=32, help="Block size.")
    p.add_argument("--start", type=int, default=0, help="First energy index (inclusive).")
    p.add_argument("--end", type=int, default=401, help="Last energy index (inclusive).")
    p.add_argument("--cond-path", type=str, default="global/condition_full_svd",
                    help="Dataset path (within the h5 file) for the per-index condition number.")
    p.add_argument("--outdir", type=str, default=None,
                    help="Output directory. Defaults to <script_dir>/plots.")
    return p.parse_args()


def main():
    args = parse_args()

    material = args.material or os.path.splitext(os.path.basename(args.h5path))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(script_dir, "plots")
    os.makedirs(outdir, exist_ok=True)

    def outpath(suffix):
        return os.path.join(outdir, f"{material}_{suffix}")

    idx_range = range(args.start, args.end + 1)

    idx_ok = []
    relres32 = []
    fwd_err32 = []
    fwd_err64 = []
    cond_vals = []
    failed = []

    with h5py.File(args.h5path, "r") as f:
        cond_arr = None
        if args.cond_path in f:
            cond_arr = f[args.cond_path][:]
        else:
            print(f"warning: cond dataset '{args.cond_path}' not found in {args.h5path}; "
                  "condition number column will be NaN.")

        for idx in idx_range:
            key = f"E_{idx}"
            try:
                if key not in f:
                    raise KeyError(f"{key} not in file")
                g = f[f"{key}/M"]
                shape = tuple(g.attrs["shape"]) if "shape" in g.attrs else None
                M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
                b = f[f"{key}/rhs"][:]
                x128 = f[f"{key}/blockthomas/complex128/x"][:]
                x64 = f[f"{key}/blockthomas/complex64/x"][:]

                D, L, U = extract_blocks_sparse(M, args.bs)
                bt = BlockThomasFP16(D, L, U)
                x32 = bt.solve(b)

                nb = np.linalg.norm(b)
                n128 = np.linalg.norm(x128)

                r32 = np.linalg.norm(M @ x32 - b) / nb
                e32 = np.linalg.norm(x32 - x128) / n128
                e64 = np.linalg.norm(x64 - x128) / n128
                cond = float(cond_arr[idx]) if cond_arr is not None and idx < len(cond_arr) else float("nan")

            except Exception as exc:                      # noqa: BLE001
                failed.append((idx, repr(exc)))
                continue

            idx_ok.append(idx)
            relres32.append(r32)
            fwd_err32.append(e32)
            fwd_err64.append(e64)
            cond_vals.append(cond)

            if idx % 25 == 0:
                print(f"idx={idx:4d}  relres32={r32:.3e}  fwd_err32={e32:.3e}  "
                      f"fwd_err64={e64:.3e}  cond={cond:.3e}")

    idx_ok = np.array(idx_ok)
    relres32 = np.array(relres32)
    fwd_err32 = np.array(fwd_err32)
    fwd_err64 = np.array(fwd_err64)
    cond_vals = np.array(cond_vals)

    n_requested = args.end - args.start + 1
    print(f"\nCompleted {len(idx_ok)}/{n_requested} indices ({material}, bs={args.bs}).")
    if failed:
        print(f"{len(failed)} indices failed/skipped, e.g.:")
        for idx, msg in failed[:10]:
            print(f"  idx={idx}: {msg}")

    # ------------------------------------------------------------- CSV -----
    csv_path = outpath("metrics.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["idx", "relres_fp16", "fwd_err_fp16_vs_c128",
                          "fwd_err_c64_vs_c128", "cond_full_svd"])
        for i in range(len(idx_ok)):
            writer.writerow([idx_ok[i], relres32[i], fwd_err32[i], fwd_err64[i], cond_vals[i]])

    # ------------------------------------------------------------- TXT -----
    txt_path = outpath("metrics.txt")
    with open(txt_path, "w") as ftxt:
        ftxt.write(f"material   : {material}\n")
        ftxt.write(f"h5path     : {args.h5path}\n")
        ftxt.write(f"block size : {args.bs}\n")
        ftxt.write(f"requested  : idx {args.start}..{args.end} ({n_requested} indices)\n")
        ftxt.write(f"completed  : {len(idx_ok)}\n")
        ftxt.write(f"failed     : {len(failed)}\n")
        if failed:
            ftxt.write("failed indices:\n")
            for idx, msg in failed:
                ftxt.write(f"  idx={idx}: {msg}\n")
        ftxt.write("\nidx  relres_fp16   fwd_err_fp16   fwd_err_c64   cond_full_svd\n")
        for i in range(len(idx_ok)):
            ftxt.write(f"{idx_ok[i]:4d}  {relres32[i]:.6e}  {fwd_err32[i]:.6e}  "
                        f"{fwd_err64[i]:.6e}  {cond_vals[i]:.6e}\n")

    # ------------------------------------------------------------ plot 1 ---
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.semilogy(idx_ok, relres32, marker=".", ms=3, lw=0.8,
                 label=r"relres  $\|Mx_{32}-b\|/\|b\|$")
    ax1.semilogy(idx_ok, fwd_err32, marker=".", ms=3, lw=0.8,
                 label=r"fwd err vs $x_{128}$  $\|x_{32}-x_{128}\|/\|x_{128}\|$")
    ax1.axhline(4.9e-4, color="gray", ls="--", lw=0.8, label="fp16 unit roundoff (4.9e-4)")
    ax1.set_xlabel("energy index")
    ax1.set_ylabel("relative error")
    ax1.set_title(f"BlockThomasFP16: residual vs forward error -- {material}")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(outpath("relres_fwderr.png"), dpi=150)

    # ------------------------------------------------------------ plot 2 ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.semilogy(idx_ok, fwd_err32, marker=".", ms=3, lw=0.8,
                 label=r"$\|x_{32}-x_{128}\|/\|x_{128}\|$")
    ax2.semilogy(idx_ok, fwd_err64, marker=".", ms=3, lw=0.8,
                 label=r"$\|x_{64}-x_{128}\|/\|x_{128}\|$")
    ax2.set_xlabel("energy index")
    ax2.set_ylabel("relative forward error vs $x_{128}$")
    ax2.set_title(f"Forward accuracy: fp16 and c64 vs c128 reference -- {material}")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(outpath("forward_accuracy.png"), dpi=150)

    print(f"\nSaved:\n  {outpath('relres_fwderr.png')}\n  {outpath('forward_accuracy.png')}\n"
          f"  {csv_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
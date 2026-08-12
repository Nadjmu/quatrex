#!/usr/bin/env python3
"""
Non-normality of M(E): singular values against eigenvalue magnitudes.

Input
-----
The ``non_normality`` group written by ``non-normal/non-normal.py`` into its
analysis file:

    ratio                  (P, n)  sigma_i / |lambda_i|
    log_cumulative_ratio   (P, n)  log(prod sigma / prod |lambda|)
    indices                (P,)    energy index of each row
    valid                  (P,)    bool, row fully computed
    nnz                    (P,)    nnz of M, -1 if unknown

The two matrices are read one row at a time, so the sweep may be plotted
without holding it in memory.

Algorithm
---------
Both quantities are defined from singular values sigma_1 >= ... >= sigma_n and
eigenvalue magnitudes |lambda_1| >= ... >= |lambda_n| of the same matrix,
sorted independently in descending order.

For a normal matrix the two sequences coincide, so sigma_i / |lambda_i| = 1 for
every i and the cumulative log-ratio is identically zero. Deviation from unity
therefore measures departure from normality rank by rank. The cumulative form

    log( prod_{i<=k} sigma_i  /  prod_{i<=k} |lambda_i| )

is non-negative and non-decreasing in k by Weyl's majorant theorem, and reaches
exactly zero at k = n because both products equal |det M|. Its interior maximum
is a scalar summary of non-normality that no single rank exposes.

Axis limits are computed once over every valid row so that frames of the same
sweep are directly comparable; the ratio panel is drawn on a linear axis by
default and on a logarithmic axis under --ratio-y-scale log.

Output
------
<outdir>/<material>_frames/E_<index>.png   one two-panel frame per valid index
<outdir>/<material>_non_normal.gif         the frames animated in index order

The default output directory is the analysis file's own directory, so the
figures are written beside the data they were drawn from.

Usage
-----
    python plot_non_normal.py /scratch/yimili/non-normal/carbon-chain.h5
    python plot_non_normal.py /scratch/yimili/non-normal/carbon-chain.h5 \
        --ratio-y-scale log --gif-fps 6 --ping-pong
    python plot_non_normal.py .../carbon-chain.h5 --skip-gif --clean-frames
"""

import argparse
import gc
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
sys.path.append(str((_HERE / ".." / "solvers").resolve()))

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cli
from style import energies_of

GROUP = "non_normality"
REQUIRED = ("ratio", "log_cumulative_ratio", "indices", "valid")


def load_arrays(h5file):
    """
    Bind the two per-rank matrices and read the small index arrays.

    The matrices are returned as HDF5 datasets, not arrays: they are indexed
    one row at a time by the caller, so the sweep never becomes resident. The
    file must stay open for as long as they are used.

    Returns (ratio_matrix, logcum_matrix, indices, valid_rows, nnz_array,
    attrs) where valid_rows is an integer array of row positions, not a boolean
    mask, and attrs carries the band edges and the grid.
    """
    if GROUP not in h5file:
        raise SystemExit(f"{h5file.filename} has no '{GROUP}' group; "
                         f"run non-normal/non-normal.py for this material first")
    group = h5file[GROUP]
    missing = [name for name in REQUIRED if name not in group]
    if missing:
        raise SystemExit(f"{h5file.filename}:/{GROUP} is missing "
                         f"dataset(s): {', '.join(missing)}")

    ratio_matrix = group["ratio"]
    logcum_matrix = group["log_cumulative_ratio"]
    indices = group["indices"][:]
    valid_rows = np.where(group["valid"][:])[0]
    nnz_array = group["nnz"][:] if "nnz" in group else None
    attrs = dict(group.attrs)

    if len(valid_rows) == 0:
        raise SystemExit(f"{h5file.filename}:/{GROUP} marks no row as valid; "
                         f"the SVD pass produced no usable data")
    return ratio_matrix, logcum_matrix, indices, valid_rows, nnz_array, attrs


def make_plot_limits(ratio_matrix, logcum_matrix, valid_rows, diff,
                     ratio_y_scale):
    """
    Common y-limits over every valid row, so frames are comparable.

    The ratio axis always includes the band [1 - diff, 1 + diff] and unity
    itself; the cumulative axis always includes zero, since both are the
    reference levels the frames are read against. Padding is 5 percent of the
    span, taken in log space when the ratio axis is logarithmic.
    """
    global_ratio_min = np.inf
    global_ratio_max = -np.inf
    global_logcum_min = np.inf
    global_logcum_max = -np.inf

    for row in valid_rows:
        ratio = np.asarray(ratio_matrix[row, :])
        logcum = np.asarray(logcum_matrix[row, :])
        global_ratio_min = min(global_ratio_min, float(np.min(ratio)))
        global_ratio_max = max(global_ratio_max, float(np.max(ratio)))
        global_logcum_min = min(global_logcum_min, float(np.min(logcum)))
        global_logcum_max = max(global_logcum_max, float(np.max(logcum)))

    ratio_low = min(global_ratio_min, 1.0 - diff, 1.0)
    ratio_high = max(global_ratio_max, 1.0 + diff, 1.0)

    if ratio_y_scale == "log":
        if ratio_low <= 0:
            raise ValueError("a logarithmic ratio axis requires positive "
                             "ratios; the sweep contains a ratio <= 0")
        log_low = np.log10(ratio_low)
        log_high = np.log10(ratio_high)
        log_padding = max(0.05 * (log_high - log_low), 0.02)
        ratio_ylim = (10.0 ** max(log_low - log_padding, -300.0),
                      10.0 ** min(log_high + log_padding, 300.0))
    else:
        ratio_padding = max(0.05 * (ratio_high - ratio_low), 0.02)
        ratio_ylim = (max(0.0, ratio_low - ratio_padding),
                      ratio_high + ratio_padding)

    cumulative_low = min(global_logcum_min, 0.0)
    cumulative_high = max(global_logcum_max, 0.0)
    cumulative_padding = max(0.05 * (cumulative_high - cumulative_low), 0.05)
    cumulative_ylim = (cumulative_low - cumulative_padding,
                       cumulative_high + cumulative_padding)

    return ratio_ylim, cumulative_ylim


def create_frames(frame_dir, indices, ratio_matrix, logcum_matrix, nnz_array,
                  valid_rows, diff, ratio_y_scale, dpi, attrs=None):
    """
    Render one two-panel frame per valid row. Returns the frame paths in index
    order. Rows are read and released one at a time to bound resident memory.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)

    n = ratio_matrix.shape[1]
    x_right = max(n, 2)
    k = np.arange(1, n + 1)

    ratio_ylim, cumulative_ylim = make_plot_limits(
        ratio_matrix=ratio_matrix, logcum_matrix=logcum_matrix,
        valid_rows=valid_rows, diff=diff, ratio_y_scale=ratio_y_scale)

    frame_paths = []

    for frame_number, row in enumerate(valid_rows, start=1):
        index = int(indices[row])
        ratio = np.asarray(ratio_matrix[row, :])
        logcum = np.asarray(logcum_matrix[row, :])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                       constrained_layout=True)

        ax1.plot(k, ratio, marker="o", markersize=1.2, linewidth=0.7)
        ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1.0,
                    label="ratio = 1 (normal matrix)")
        ax1.axhline(1.0 + diff, color="gray", linestyle=":", linewidth=0.9)
        ax1.axhline(1.0 - diff, color="gray", linestyle=":", linewidth=0.9)
        ax1.set_yscale(ratio_y_scale)
        ax1.set_xlim(1, x_right)
        ax1.set_ylim(*ratio_ylim)
        ax1.set_xlabel("rank i, descending order")
        ax1.set_ylabel(r"$\sigma_i / |\lambda_i|$")
        ax1.set_title("Pointwise singular-value to eigenvalue-magnitude ratio")
        ax1.grid(alpha=0.3, which="both")
        ax1.legend(loc="best")

        ax2.plot(k, logcum, linewidth=0.9)
        ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax2.set_xlim(1, x_right)
        ax2.set_ylim(*cumulative_ylim)
        ax2.set_xlabel("k")
        ax2.set_ylabel(r"$\log\left(\prod_{i\leq k}\sigma_i"
                       r"\,/\,\prod_{i\leq k}|\lambda_i|\right)$")
        ax2.set_title("Cumulative product ratio, logarithmic form")
        ax2.grid(alpha=0.3)

        number_above = int(np.count_nonzero(ratio > 1.0 + diff))
        number_below = int(np.count_nonzero(ratio < 1.0 - diff))
        nnz_text = ""
        if nnz_array is not None and row < len(nnz_array) and nnz_array[row] >= 0:
            nnz_text = f"nnz={int(nnz_array[row])}   "

        # The energy of the index, where the file records the grid. The panels
        # are per rank, so this is the only place the sweep coordinate appears.
        energy = energies_of(attrs, [index])
        energy_text = "" if energy is None else f"E = {energy[0]:.4f} eV   "

        fig.suptitle(f"E_{index}   {energy_text}"
                     f"frame {frame_number}/{len(valid_rows)}   "
                     f"n={n}   {nnz_text}"
                     f"> {1.0 + diff:.2f}: {number_above}   "
                     f"< {1.0 - diff:.2f}: {number_below}   "
                     f"endpoint={logcum[-1]:+.2e}", fontsize=12)

        frame_path = frame_dir / f"E_{index:06d}.png"
        fig.savefig(frame_path, dpi=dpi, facecolor="white")
        plt.close(fig)
        frame_paths.append(frame_path)

        del ratio, logcum
        gc.collect()

        if frame_number % 25 == 0:
            print(f"[frames] created {frame_number}/{len(valid_rows)}")

    return frame_paths


def create_gif(frame_paths, gif_path, gif_fps, ping_pong):
    """
    Assemble the frames into an animated GIF. Frames are converted to an
    adaptive palette individually; with --ping-pong the sequence is followed by
    its reverse, excluding the two endpoints, so the loop has no visible seam.
    """
    if not frame_paths:
        raise RuntimeError("no frames available for GIF creation")

    gif_sequence = list(frame_paths)
    if ping_pong and len(frame_paths) > 2:
        gif_sequence += frame_paths[-2:0:-1]

    duration_ms = int(1000 / gif_fps)
    first_frame = Image.open(gif_sequence[0]).convert("P", palette=Image.ADAPTIVE)
    append_frames = []
    try:
        for i, frame_path in enumerate(gif_sequence[1:], start=2):
            append_frames.append(
                Image.open(frame_path).convert("P", palette=Image.ADAPTIVE))
            if i % 50 == 0:
                print(f"[gif] loaded {i}/{len(gif_sequence)} frames")
        first_frame.save(gif_path, save_all=True, append_images=append_frames,
                         duration=duration_ms, loop=0)
    finally:
        first_frame.close()
        for frame in append_frames:
            frame.close()


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"non-normal/non-normal.py, group {GROUP}")
    ap.add_argument("--ratio-y-scale", choices=["linear", "log"],
                    default="linear",
                    help="y-axis scale of the pointwise ratio panel")
    ap.add_argument("--diff", type=float, default=0.01,
                    help="half-width of the band around 1.0 used for the "
                         "above/below counts in the frame title")
    ap.add_argument("--dpi", type=int, default=100,
                    help="frame resolution; lower values reduce GIF size")
    ap.add_argument("--gif-fps", type=int, default=4, help="GIF frames per second")
    ap.add_argument("--ping-pong", action="store_true",
                    help="append the reversed sequence to the GIF")
    ap.add_argument("--skip-gif", action="store_true",
                    help="render frames only")
    ap.add_argument("--clean-frames", action="store_true",
                    help="delete existing frames before rendering")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: the analysis file's directory)")
    args = ap.parse_args()

    h5path = Path(args.h5path).expanduser().resolve()
    material = args.material or h5path.stem
    out_dir = Path(args.outdir) if args.outdir else h5path.parent
    frame_dir = out_dir / f"{material}_frames"
    gif_path = out_dir / f"{material}_non_normal.gif"

    # The datasets are read row by row inside create_frames, so the file stays
    # open for the whole render.
    with h5py.File(h5path, "r") as h5file:
        ratio_matrix, logcum_matrix, indices, valid_rows, nnz_array, attrs = \
            load_arrays(h5file)
        print(f"[input] {len(valid_rows)} valid rows of "
              f"{ratio_matrix.shape[0]}, n = {ratio_matrix.shape[1]}")

        if args.clean_frames and frame_dir.exists():
            for old_frame in frame_dir.glob("E_*.png"):
                old_frame.unlink()

        frame_paths = create_frames(frame_dir=frame_dir, indices=indices,
                                    ratio_matrix=ratio_matrix,
                                    logcum_matrix=logcum_matrix,
                                    nnz_array=nnz_array,
                                    valid_rows=valid_rows, diff=args.diff,
                                    ratio_y_scale=args.ratio_y_scale,
                                    dpi=args.dpi, attrs=attrs)
    print(f"[frames] created {len(frame_paths)} frames in {frame_dir}")

    if not args.skip_gif:
        create_gif(frame_paths=frame_paths, gif_path=gif_path,
                   gif_fps=args.gif_fps, ping_pong=args.ping_pong)
        print(f"[gif] wrote {gif_path}")


if __name__ == "__main__":
    main()

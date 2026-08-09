#!/usr/bin/env python3
"""
Spectra and conditioning of the QTBM system matrices over an energy sweep.

Input
-----
The per-material directory written by ``main3.py`` (or ``main3_gpu.py``),
containing whichever of these arrays the run produced:

    energies.npy               (P,)   energy grid of the sweep, eV
    band_edge.npy              scalar conduction band edge, eV
    spectrum_bare.npy          (n,)   eigenvalues of H v = lambda S v
    condition_bare.npy         (P,)   kappa_2(H - E S) along the sweep
    condition_full_svd.npy     (P,)   kappa_2(M(E)) from the full SVD
    max_singular_values.npy    (P,)   sigma_max(M(E))
    min_singular_values.npy    (P,)   sigma_min(M(E))

Every array is optional; the figures whose inputs are absent are skipped and
reported. Nothing is recomputed here.

Algorithm
---------
No computation is performed beyond selecting axis ranges.

Figure 1, generalized eigenvalue spectrum. The eigenvalues of the pencil
(H, S) are plotted in the complex plane, with the conduction band edge marked.
A full view and a view restricted to a window around the band edge are drawn as
separate panels, because the eigenvalue nearest the band edge governs the
conditioning of the open-boundary problem there and is invisible at full scale.

Figure 2, conditioning along the sweep. kappa_2(H - E S) and kappa_2(M(E)) on a
shared logarithmic axis. The difference between the two isolates the effect of
the contact self-energies: M(E) = E S - H - Sigma(E) differs from the bare
pencil only by Sigma, so the two curves separate exactly where the open
boundary conditions dominate the conditioning.

Figure 3, extreme singular values. sigma_max and sigma_min of M(E) plotted
separately, since a peak in kappa_2 may arise either from sigma_min
approaching zero (near-singularity, the case of interest here) or from
sigma_max growing, and the ratio alone does not distinguish them.

Output
------
<outdir>/<material>_spectrum.png
<outdir>/<material>_condition.png
<outdir>/<material>_singular_values.png

Usage
-----
    python plot_qtbm_spectra.py /scratch/yimili/matrices/dev_12_sorted_BENCH
    python plot_qtbm_spectra.py .../dev_12_sorted_BENCH --zoom 0.05 \
        --outdir /scratch/yimili/plots/dev_12_sorted_BENCH
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
sys.path.append(str((_HERE / ".." / "solvers").resolve()))

import numpy as np
import matplotlib.pyplot as plt

import cli
from style import save_figure


def load_optional(directory, name):
    """Return the named .npy array, or None when it was not produced."""
    path = directory / name
    return np.load(path) if path.exists() else None


def plot_spectrum(eigenvalues, band_edge, zoom, material, out_path):
    """Pencil eigenvalues in the complex plane, full view and band-edge view."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, title in zip(axes, ("full spectrum",
                                f"window of $\\pm${zoom:g} eV about the "
                                f"band edge")):
        ax.scatter(eigenvalues.real, eigenvalues.imag, s=8, alpha=0.7)
        if band_edge is not None:
            ax.axvline(band_edge, color="red", ls="--", lw=1.0,
                       label="conduction band edge")
            ax.legend(fontsize=8)
        ax.set_xlabel("Re $\\lambda$ (eV)")
        ax.set_ylabel("Im $\\lambda$ (eV)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)

    if band_edge is not None:
        axes[1].set_xlim(band_edge - zoom, band_edge + zoom)
        axes[1].set_ylim(-0.1, 0.1)

    fig.suptitle(f"Generalized eigenvalues of $Hv = \\lambda Sv$ — {material}",
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=300)


def plot_condition(energies, condition_bare, condition_full, band_edge,
                   material, out_path):
    """kappa_2 of the bare pencil and of the full system matrix, one axis."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if condition_bare is not None:
        ax.semilogy(energies, condition_bare, marker="o", ms=3, lw=0.9,
                    label=r"$\kappa_2(H - E S)$")
    if condition_full is not None:
        ax.semilogy(energies, condition_full, marker="s", ms=3, lw=0.9,
                    label=r"$\kappa_2(M(E))$, full SVD")
    if band_edge is not None:
        ax.axvline(band_edge, color="red", ls="--", lw=1.0,
                   label="conduction band edge")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\kappa_2$")
    ax.set_title(f"Conditioning along the energy sweep — {material}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=300)


def plot_singular_values(energies, sigma_max, sigma_min, band_edge, material,
                         out_path):
    """Extreme singular values of M(E), which kappa_2 alone does not separate."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(energies, sigma_max, marker="o", ms=3, lw=0.9,
                label=r"$\sigma_{\max}(M(E))$")
    ax.semilogy(energies, sigma_min, marker="s", ms=3, lw=0.9,
                label=r"$\sigma_{\min}(M(E))$")
    if band_edge is not None:
        ax.axvline(band_edge, color="red", ls="--", lw=1.0,
                   label="conduction band edge")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("singular value")
    ax.set_title(f"Extreme singular values of $M(E)$ — {material}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=300)


def main():
    ap = cli.new_parser(__doc__)
    ap.add_argument("data_dir", type=Path,
                    help="per-material directory written by main3.py")
    ap.add_argument("--zoom", type=float, default=0.1, metavar="EV",
                    help="half-width in eV of the band-edge window in the "
                         "spectrum figure")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: the input directory)")
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    material = args.material or data_dir.name
    outdir = Path(args.outdir) if args.outdir else data_dir

    energies = load_optional(data_dir, "energies.npy")
    band_edge_arr = load_optional(data_dir, "band_edge.npy")
    band_edge = float(band_edge_arr) if band_edge_arr is not None else None
    eigenvalues = load_optional(data_dir, "spectrum_bare.npy")
    condition_bare = load_optional(data_dir, "condition_bare.npy")
    condition_full = load_optional(data_dir, "condition_full_svd.npy")
    sigma_max = load_optional(data_dir, "max_singular_values.npy")
    sigma_min = load_optional(data_dir, "min_singular_values.npy")

    produced = 0

    if eigenvalues is not None:
        plot_spectrum(eigenvalues, band_edge, args.zoom, material,
                      outdir / f"{material}_spectrum.png")
        produced += 1
    else:
        print("spectrum_bare.npy absent; the spectrum figure was skipped")

    if energies is not None and (condition_bare is not None
                                 or condition_full is not None):
        plot_condition(energies, condition_bare, condition_full, band_edge,
                       material, outdir / f"{material}_condition.png")
        produced += 1
    else:
        print("energies.npy and a condition array are required; "
              "the conditioning figure was skipped")

    if energies is not None and sigma_max is not None and sigma_min is not None:
        plot_singular_values(energies, sigma_max, sigma_min, band_edge,
                             material, outdir / f"{material}_singular_values.png")
        produced += 1
    else:
        print("max/min_singular_values.npy absent; "
              "the singular-value figure was skipped")

    if produced == 0:
        raise SystemExit(f"no input arrays found in {data_dir}")


if __name__ == "__main__":
    main()

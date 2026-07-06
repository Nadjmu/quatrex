"""Validate RPA dielectric sign convention on a two-level model.

For one occupied valence state and one empty conduction state separated by
Delta, the retarded RPA response has negative imaginary part near omega=Delta.
With a positive Coulomb interaction V, the passive dielectric convention

    epsilon = 1 - V P
    loss = -Im epsilon^{-1}

has positive loss at the resonance. The opposite sign convention is included as
a visual diagnostic.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quatrex-validation")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/quatrex-validation-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "src" / "quatrex").is_dir():
            return parent
    raise RuntimeError("Could not find repository root containing src/quatrex.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def _find_example_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "quatrex_config.toml").is_file() and (parent / "inputs").is_dir():
            return parent
    raise RuntimeError("Could not find gw-unit-cell example root.")


W90_ROOT = REPO_ROOT / "examples" / "w90"
DATA_ANALYSIS_ROOT = W90_ROOT / "data_analysis"
EXAMPLE_ROOT = W90_ROOT / "carbon-nanotube" / "gw-unit-cell"
RPA_COMPUTE = (
    REPO_ROOT
    / "src"
    / "quatrex"
    / "coulomb_screening"
    / "dielectric_screening"
    / "rpa_compute.py"
)

spec = importlib.util.spec_from_file_location("quatrex_rpa_compute_validation", RPA_COMPUTE)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load RPA helpers from {RPA_COMPUTE}.")
rpa_compute = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rpa_compute
spec.loader.exec_module(rpa_compute)

compute_rpa_polarization = rpa_compute.compute_rpa_polarization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dielectric sign convention with a two-level RPA model."
    )
    parser.add_argument("--gap", type=float, default=2.0, help="Transition energy in eV.")
    parser.add_argument("--broadening", type=float, default=0.05)
    parser.add_argument("--coulomb", type=float, default=1.0)
    parser.add_argument("--max-frequency", type=float, default=4.0)
    parser.add_argument("--num-frequency", type=int, default=801)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_ANALYSIS_ROOT
        / "validation_outputs"
        / "cnt_hbn_general"
        / "general"
        / "dielectric_sign"
        / "validation_rpa_dielectric_sign",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frequencies = np.linspace(0.0, args.max_frequency, args.num_frequency)

    energies_k = np.array([[0.0, args.gap]])
    energies_kq = np.array([[[0.0, args.gap]]])
    occupations_k = np.array([[1.0, 0.0]])
    occupations_kq = np.array([[[1.0, 0.0]]])
    form_factors = np.ones((1, 1, 2, 2), dtype=np.complex128)

    polarization = compute_rpa_polarization(
        energies_k,
        energies_kq,
        occupations_k,
        occupations_kq,
        form_factors,
        frequencies,
        broadening=args.broadening,
        frequency_axis="real",
        normalize_k_sum=False,
    )[0]

    epsilon_minus = 1.0 - args.coulomb * polarization
    epsilon_plus = 1.0 + args.coulomb * polarization
    loss_minus = -np.imag(1.0 / epsilon_minus)
    loss_plus = -np.imag(1.0 / epsilon_plus)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].plot(frequencies, np.real(polarization), label="Re P")
    axes[0].plot(frequencies, np.imag(polarization), label="Im P")
    axes[0].axvline(args.gap, color="black", alpha=0.35, linewidth=0.8)
    axes[0].set_title("Two-level RPA response")

    axes[1].plot(frequencies, np.real(epsilon_minus), label="Re epsilon")
    axes[1].plot(frequencies, np.imag(epsilon_minus), label="Im epsilon")
    axes[1].axvline(args.gap, color="black", alpha=0.35, linewidth=0.8)
    axes[1].set_title(r"$\epsilon = 1 - VP$")

    axes[2].plot(frequencies, loss_minus, label=r"$1 - VP$")
    axes[2].plot(frequencies, loss_plus, label=r"$1 + VP$", linestyle="--")
    axes[2].axhline(0.0, color="black", alpha=0.35, linewidth=0.8)
    axes[2].axvline(args.gap, color="black", alpha=0.35, linewidth=0.8)
    axes[2].set_title(r"Loss $-\mathrm{Im}\,\epsilon^{-1}$")

    for ax in axes:
        ax.set_xlabel(r"$\omega$ (eV)")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False)

    plot_path = output_prefix.with_suffix(".png")
    data_path = output_prefix.with_suffix(".npz")
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    np.savez(
        data_path,
        frequencies=frequencies,
        polarization=polarization,
        epsilon_minus=epsilon_minus,
        epsilon_plus=epsilon_plus,
        loss_minus=loss_minus,
        loss_plus=loss_plus,
        gap=np.array(args.gap),
        broadening=np.array(args.broadening),
        coulomb=np.array(args.coulomb),
    )

    resonance_index = int(np.argmin(np.abs(frequencies - args.gap)))
    print(f"Transition energy: {args.gap:.6g} eV")
    print(f"P(omega=Delta): {polarization[resonance_index]}")
    print(f"Im P at resonance: {np.imag(polarization[resonance_index]):.6g}")
    print(f"Loss with epsilon=1-VP at resonance: {loss_minus[resonance_index]:.6g}")
    print(f"Loss with epsilon=1+VP at resonance: {loss_plus[resonance_index]:.6g}")
    print(f"Max loss with epsilon=1-VP: {np.max(loss_minus):.6g}")
    print(f"Wrote plot: {plot_path.resolve()}")
    print(f"Wrote data: {data_path.resolve()}")


if __name__ == "__main__":
    main()

"""Validate the Fermi-Dirac occupations used by the RPA pipeline."""

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

spec = importlib.util.spec_from_file_location(
    "quatrex_rpa_compute_validation", RPA_COMPUTE
)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load RPA helpers from {RPA_COMPUTE}.")
rpa_compute = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rpa_compute
spec.loader.exec_module(rpa_compute)

fermi_dirac_distribution = rpa_compute.fermi_dirac_distribution


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Fermi-Dirac occupations used by RPA."
    )
    parser.add_argument("--fermi-level", type=float, default=0.0)
    parser.add_argument("--emin", type=float, default=-0.5)
    parser.add_argument("--emax", type=float, default=0.5)
    parser.add_argument("--num-energy", type=int, default=1001)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.0, 50.0, 300.0, 1000.0],
        help="Temperatures in K.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ANALYSIS_ROOT
        / "validation_outputs"
        / "cnt_hbn_general"
        / "general"
        / "occupations"
        / "validation_rpa_occupations.png",
    )
    parser.add_argument(
        "--save-data",
        type=Path,
        default=DATA_ANALYSIS_ROOT
        / "validation_outputs"
        / "cnt_hbn_general"
        / "general"
        / "occupations"
        / "validation_rpa_occupations.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    energies = np.linspace(args.emin, args.emax, args.num_energy)

    occupations_by_temperature = {}
    for temperature in args.temperatures:
        occupations_by_temperature[temperature] = fermi_dirac_distribution(
            energies,
            chemical_potential=args.fermi_level,
            temperature=temperature,
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    for temperature, occupations in occupations_by_temperature.items():
        label = f"{temperature:g} K"
        ax.plot(energies - args.fermi_level, occupations, linewidth=1.5, label=label)

    ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.55)
    ax.axhline(0.5, color="black", linewidth=0.7, alpha=0.35, linestyle="--")
    ax.set_xlabel(r"$E - E_F$ (eV)")
    ax.set_ylabel("Occupation")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("RPA occupation factor validation")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    if args.save_data is not None:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_data,
            energies=energies,
            fermi_level=np.array(args.fermi_level),
            temperatures=np.asarray(args.temperatures, dtype=float),
            occupations=np.stack(
                [occupations_by_temperature[t] for t in args.temperatures],
                axis=0,
            ),
        )

    below_index = int(np.searchsorted(energies, args.fermi_level - 0.1))
    above_index = int(np.searchsorted(energies, args.fermi_level + 0.1))
    fermi_index = int(np.argmin(np.abs(energies - args.fermi_level)))

    print(f"Fermi level: {args.fermi_level:.6g} eV")
    for temperature, occupations in occupations_by_temperature.items():
        below = float(occupations[below_index])
        at_fermi = float(occupations[fermi_index])
        above = float(occupations[above_index])
        monotone = bool(np.all(np.diff(occupations) <= 1e-14))
        print(
            f"T={temperature:g} K: "
            f"f(EF-0.1eV)={below:.6g}, "
            f"f(EF)={at_fermi:.6g}, "
            f"f(EF+0.1eV)={above:.6g}, "
            f"monotone_decreasing={monotone}"
        )

    print(f"Wrote plot: {args.output.resolve()}")
    if args.save_data is not None:
        print(f"Wrote data: {args.save_data.resolve()}")


if __name__ == "__main__":
    main()

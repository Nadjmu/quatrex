"""Export raw matrix-valued RPA polarization without importing the full package.

This is a lightweight companion to ``export_raw_rpa_polarization.py`` for
validation notebooks on systems where the installed Python cannot import the
current package source. It uses the same ``rpa_compute.py`` helper module that
the validation scripts use.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/quatrex-validation-cache")

import numpy as np


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "src" / "quatrex").is_dir():
            return parent
    raise RuntimeError("Could not find repository root containing src/quatrex.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
RPA_COMPUTE = (
    REPO_ROOT
    / "src"
    / "quatrex"
    / "coulomb_screening"
    / "dielectric_screening"
    / "rpa_compute.py"
)

spec = importlib.util.spec_from_file_location("quatrex_rpa_compute_export", RPA_COMPUTE)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load RPA helpers from {RPA_COMPUTE}.")
rpa_compute = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rpa_compute
spec.loader.exec_module(rpa_compute)

RPAPolarization = rpa_compute.RPAPolarization
ScreeningChannels = rpa_compute.ScreeningChannels
build_uniform_brillouin_zone_mesh = rpa_compute.build_uniform_brillouin_zone_mesh
load_translation_blocks = rpa_compute.load_translation_blocks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export matrix-valued retarded RPA polarization arrays."
    )
    parser.add_argument("--hamiltonian", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--lattice-constant", type=float, default=1.0)
    parser.add_argument("--num-k", type=int, default=100)
    parser.add_argument("--num-q", type=int, default=60)
    parser.add_argument("--num-frequency", type=int, required=True)
    parser.add_argument("--max-frequency", type=float, required=True)
    parser.add_argument("--fermi-level", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--broadening", type=float, default=0.05)
    parser.add_argument("--spin-degeneracy", type=float, default=2.0)
    parser.add_argument("--valley-degeneracy", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mesh = build_uniform_brillouin_zone_mesh(
        num_k_points=args.num_k,
        num_q_points=args.num_q,
        num_frequencies=args.num_frequency,
        max_frequency=args.max_frequency,
        lattice_constant=args.lattice_constant,
        include_zero_q=True,
    )
    blocks = load_translation_blocks(args.hamiltonian)
    solver = RPAPolarization(
        channels=ScreeningChannels(
            spin_degeneracy=args.spin_degeneracy,
            valley_degeneracy=args.valley_degeneracy,
        ),
        frequency_axis="real",
    )
    result = solver.solve_matrix_from_translation_blocks(
        translation_blocks=blocks,
        mesh=mesh,
        chemical_potential=args.fermi_level,
        temperature=args.temperature,
        periodic_axis=args.axis,
        lattice_constant=args.lattice_constant,
        broadening=args.broadening,
    )

    output_dir = args.output_dir / "raw_rpa_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "polarization_retarded_qw.npy",
        np.asarray(result.polarization, dtype=np.complex128),
    )
    np.save(output_dir / "frequencies_eV.npy", np.asarray(mesh.frequencies))
    np.save(output_dir / "q_points.npy", np.asarray(mesh.q_points))
    np.save(output_dir / "k_points.npy", np.asarray(mesh.k_points))
    np.save(
        output_dir / "band_eigenvalues.npy",
        np.asarray(result.band_structure.eigenvalues),
    )
    with (output_dir / "README.txt").open("w", encoding="ascii") as handle:
        handle.write("Raw equilibrium RPA debug export\n")
        handle.write(f"Hamiltonian: {args.hamiltonian.resolve()}\n")
        handle.write(
            "polarization_retarded_qw.npy shape: "
            f"{np.asarray(result.polarization).shape}\n"
        )
        handle.write("Frequencies are stored in eV.\n")
        handle.write("q_points and k_points use the 1D solver reciprocal units.\n")
        handle.write(f"chemical_potential_eV: {args.fermi_level}\n")
        handle.write(f"temperature_K: {args.temperature}\n")
        handle.write(f"broadening_eV: {args.broadening}\n")
        handle.write(f"spin_degeneracy: {args.spin_degeneracy}\n")
        handle.write(f"valley_degeneracy: {args.valley_degeneracy}\n")

    print(f"Wrote raw RPA debug export to {output_dir.resolve()}")
    print(
        f"polarization_retarded_qw.npy shape: {np.asarray(result.polarization).shape}"
    )


if __name__ == "__main__":
    main()

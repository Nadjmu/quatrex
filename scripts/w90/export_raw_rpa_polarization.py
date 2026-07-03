from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from mpi4py.MPI import COMM_WORLD as global_comm
except ModuleNotFoundError:
    class _SerialComm:
        rank = 0

    global_comm = _SerialComm()

try:
    from qttools.utils.mpi_utils import distributed_load
except ModuleNotFoundError:
    def distributed_load(path: Path) -> np.ndarray:
        return np.load(path)

from quatrex.core.config import parse_config
from quatrex.coulomb_screening.dielectric_screening.equilibrium_screening import (
    EquilibriumScreening,
)
from quatrex.coulomb_screening.dielectric_screening.negf_bridge import (
    EquilibriumRPAScreeningBridge,
)
from quatrex.coulomb_screening.dielectric_screening.rpa_compute import (
    BrillouinZoneMesh,
    ScreeningChannels,
    build_uniform_brillouin_zone_mesh,
)
from quatrex.grid import get_electron_energies


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


def _get_screening_energies(config) -> np.ndarray:
    energies_path = config.input_dir / "coulomb_screening_energies.npy"
    if os.path.isfile(energies_path):
        return np.asarray(distributed_load(energies_path), dtype=np.float64)

    electron_energies = np.asarray(get_electron_energies(config), dtype=np.float64)
    screening_energies = electron_energies - electron_energies[0]
    screening_energies += 1e-6
    return screening_energies


def _build_mesh(config, screening_energies: np.ndarray) -> BrillouinZoneMesh:
    base_mesh = build_uniform_brillouin_zone_mesh(
        num_k_points=config.coulomb_screening.num_k_points,
        num_q_points=config.coulomb_screening.num_q_points,
        num_frequencies=1,
        max_frequency=0.0,
        lattice_constant=config.coulomb_screening.lattice_constant,
        include_zero_q=config.coulomb_screening.include_zero_q,
    )
    return BrillouinZoneMesh(
        k_points=base_mesh.k_points,
        q_points=base_mesh.q_points,
        frequencies=screening_energies,
    )


def export_raw_rpa(config_path: Path) -> Path:
    config = parse_config(config_path)
    screening_energies = _get_screening_energies(config)
    mesh = _build_mesh(config, screening_energies)

    bridge = EquilibriumRPAScreeningBridge(
        config,
        screening_energies,
        template=None,  # Only the bridge setup helpers are used here.
    )
    bridge._validate_supported_configuration()
    chemical_potential = bridge._resolve_chemical_potential()

    channels = ScreeningChannels(
        spin_degeneracy=config.coulomb_screening.spin_degeneracy,
        valley_degeneracy=config.coulomb_screening.valley_degeneracy,
    )
    solver = EquilibriumScreening(
        channels=channels,
        matrix_polarization=config.coulomb_screening.matrix_valued_polarization,
        frequency_axis="real",
    )
    inputs = solver.load_inputs_from_config(
        config,
        hamiltonian_matrix_name=config.coulomb_screening.hamiltonian_matrix_name,
        coulomb_matrix_name=config.coulomb_screening.coulomb_matrix_name,
    )
    scaled_inputs = type(inputs)(
        hamiltonian_blocks=inputs.hamiltonian_blocks,
        coulomb_blocks={
            translation: block / config.coulomb_screening.epsilon_r
            for translation, block in inputs.coulomb_blocks.items()
        },
    )
    grid_result = solver.solve_grid_from_inputs(
        inputs=scaled_inputs,
        mesh=mesh,
        chemical_potential=chemical_potential,
        temperature=config.coulomb_screening.temperature,
        periodic_axis=config.coulomb_screening.periodic_axis,
        lattice_constant=config.coulomb_screening.lattice_constant,
        broadening=config.coulomb_screening.broadening,
    )

    output_dir = config.output_dir / "raw_rpa_debug"
    if global_comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            output_dir / "polarization_retarded_qw.npy",
            np.asarray(grid_result.polarization_result.polarization, dtype=np.complex128),
        )
        np.save(output_dir / "frequencies_eV.npy", np.asarray(mesh.frequencies))
        np.save(output_dir / "q_points.npy", np.asarray(mesh.q_points))
        np.save(output_dir / "k_points.npy", np.asarray(mesh.k_points))
        np.save(
            output_dir / "band_eigenvalues.npy",
            np.asarray(grid_result.polarization_result.band_structure.eigenvalues),
        )
        with open(output_dir / "README.txt", "w", encoding="ascii") as f:
            f.write("Raw equilibrium RPA debug export\n")
            f.write(f"Config: {config_path.resolve()}\n")
            f.write(
                "polarization_retarded_qw.npy shape: "
                f"{np.asarray(grid_result.polarization_result.polarization).shape}\n"
            )
            f.write("Frequencies are stored in eV.\n")
            f.write("q_points and k_points are stored in reciprocal-lattice units used by the solver.\n")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export raw equilibrium RPA polarization arrays for debugging."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(EXAMPLE_ROOT / "quatrex_config_rpa.toml"),
        help="Path to the Quatrex config file.",
    )
    args = parser.parse_args()

    output_dir = export_raw_rpa(Path(args.config))
    if global_comm.rank == 0:
        print(f"Wrote raw RPA debug export to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

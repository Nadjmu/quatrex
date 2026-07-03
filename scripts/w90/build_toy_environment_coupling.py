"""Build toy CNT-hBN Coulomb coupling matrices for workflow checks.

This script creates deliberately simple point-charge cross interactions for
testing the dielectric-environment code path before physically generated
``v_ce`` and ``v_ec`` matrices are available.

The generated matrices are not intended as a production CNT-hBN model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import scipy.io


EV_ANGSTROM = 14.3996


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "src" / "quatrex").is_dir():
            return parent
    raise RuntimeError("Could not find repository root containing src/quatrex.")


def _find_example_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "quatrex_config.toml").is_file() and (parent / "inputs").is_dir():
            return parent
    raise RuntimeError("Could not find gw-unit-cell example root.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
W90_ROOT = REPO_ROOT / "examples" / "w90"
DATA_ANALYSIS_ROOT = W90_ROOT / "data_analysis"
EXAMPLE_ROOT = W90_ROOT / "carbon-nanotube" / "gw-unit-cell"


def read_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ file '{path}'.")

    lattice_match = re.search(r'Lattice="([^"]+)"', lines[1])
    if lattice_match is None:
        raise ValueError(f"XYZ file '{path}' does not contain a Lattice field.")
    lattice = np.fromstring(lattice_match.group(1), sep=" ", dtype=float).reshape(3, 3)

    num_atoms = int(lines[0].strip())
    positions = []
    for line in lines[2 : 2 + num_atoms]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom line in '{path}': {line!r}")
        positions.append([float(fields[1]), float(fields[2]), float(fields[3])])

    return lattice, np.asarray(positions, dtype=float)


def first_matrix_size(path: Path) -> int:
    matrices = scipy.io.loadmat(path)
    for key, value in matrices.items():
        if not key.startswith("__"):
            if value.ndim != 2 or value.shape[0] != value.shape[1]:
                raise ValueError(f"Matrix '{key}' in '{path}' is not square.")
            return int(value.shape[0])
    raise ValueError(f"No matrix blocks found in '{path}'.")


def infer_central_size(
    *,
    output_dir: Path,
    hamiltonian_path: Path,
    num_cells: int,
) -> int:
    """Infer the central-device matrix size used by the environment bridge."""

    for name in ("p_retarded_diagonal_0.npy", "p_retarded_density_0.npy"):
        path = output_dir / name
        if path.is_file():
            return int(np.load(path, mmap_mode="r").shape[1])
    return first_matrix_size(hamiltonian_path) * num_cells


def expand_positions(
    unit_positions: np.ndarray,
    lattice: np.ndarray,
    *,
    target_size: int,
    transport_direction: str,
) -> np.ndarray:
    axis = "xyz".index(transport_direction)
    step = lattice[axis]
    num_cells = int(np.ceil(target_size / unit_positions.shape[0]))

    positions = []
    for cell in range(num_cells):
        positions.append(unit_positions + cell * step)
    expanded = np.vstack(positions)
    return expanded[:target_size]


def build_cross_coulomb(
    central_positions: np.ndarray,
    environment_positions: np.ndarray,
    *,
    epsilon_r: float,
    softening: float,
    cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if epsilon_r <= 0.0:
        raise ValueError("epsilon_r must be positive.")
    if softening <= 0.0:
        raise ValueError("softening must be positive.")
    if cutoff is not None and cutoff <= 0.0:
        raise ValueError("cutoff must be positive when provided.")

    diff = central_positions[:, np.newaxis, :] - environment_positions[np.newaxis, :, :]
    real_dist = np.linalg.norm(diff, axis=2)
    softened_dist = np.sqrt(real_dist**2 + softening**2)

    v_ce = (EV_ANGSTROM / epsilon_r) / softened_dist
    if cutoff is not None:
        v_ce[real_dist > cutoff] = 0.0

    v_ce = np.asarray(v_ce, dtype=np.complex128)
    return v_ce, v_ce.T.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate toy CNT-hBN v_ce/v_ec coupling matrices."
    )
    parser.add_argument(
        "--cnt-structure",
        type=Path,
        default=EXAMPLE_ROOT / "inputs" / "structure.xyz",
        help="CNT/device structure XYZ file.",
    )
    parser.add_argument(
        "--cnt-hamiltonian",
        type=Path,
        default=EXAMPLE_ROOT / "inputs" / "hamiltonian.mat",
        help="CNT/device unit-cell Hamiltonian used to infer orbitals per cell.",
    )
    parser.add_argument(
        "--cnt-cells",
        type=int,
        default=12,
        help="Number of CNT transport cells.",
    )
    parser.add_argument(
        "--cnt-output-dir",
        type=Path,
        default=EXAMPLE_ROOT / "outputs_negf_environment",
        help="CNT/device output directory used to infer the actual central matrix size.",
    )
    parser.add_argument(
        "--cnt-transport-direction",
        choices=("x", "y", "z"),
        default="z",
        help="CNT/device transport direction.",
    )
    parser.add_argument(
        "--environment-structure",
        type=Path,
        default=Path("/usr/scratch/mont-fort11/awinka/for_cal/hBN/inputs/lattice.xyz"),
        help="Environment/hBN structure XYZ file.",
    )
    parser.add_argument(
        "--environment-v-ee",
        type=Path,
        default=Path("/home/sem26f25/quatrex/examples/w90/hBN/outputs_hBN/environment/v_ee.npy"),
        help="Exported environment bare Coulomb matrix used to infer environment size.",
    )
    parser.add_argument(
        "--environment-transport-direction",
        choices=("x", "y", "z"),
        default="x",
        help="Environment/hBN transport direction.",
    )
    parser.add_argument(
        "--epsilon-r",
        type=float,
        default=4.0,
        help="Effective dielectric constant in the toy point-charge coupling.",
    )
    parser.add_argument(
        "--softening",
        type=float,
        default=1.0,
        help="Softening length in Angstrom.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Distance cutoff in Angstrom. Use 0 to disable cutoff.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_ROOT / "inputs",
        help="Directory where v_ce_toy.npy and v_ec_toy.npy are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cutoff = None if args.cutoff == 0.0 else args.cutoff

    cnt_lattice, cnt_unit_positions = read_xyz(args.cnt_structure)
    environment_lattice, environment_unit_positions = read_xyz(args.environment_structure)

    central_size = infer_central_size(
        output_dir=args.cnt_output_dir,
        hamiltonian_path=args.cnt_hamiltonian,
        num_cells=args.cnt_cells,
    )
    environment_size = int(np.load(args.environment_v_ee, mmap_mode="r").shape[0])

    central_positions = expand_positions(
        cnt_unit_positions,
        cnt_lattice,
        target_size=central_size,
        transport_direction=args.cnt_transport_direction,
    )
    environment_positions = expand_positions(
        environment_unit_positions,
        environment_lattice,
        target_size=environment_size,
        transport_direction=args.environment_transport_direction,
    )

    v_ce, v_ec = build_cross_coulomb(
        central_positions,
        environment_positions,
        epsilon_r=args.epsilon_r,
        softening=args.softening,
        cutoff=cutoff,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v_ce_path = args.output_dir / "v_ce_toy.npy"
    v_ec_path = args.output_dir / "v_ec_toy.npy"
    np.save(v_ce_path, v_ce)
    np.save(v_ec_path, v_ec)

    nonzero = np.count_nonzero(v_ce)
    density = nonzero / v_ce.size
    print(f"Wrote {v_ce_path} with shape {v_ce.shape}")
    print(f"Wrote {v_ec_path} with shape {v_ec.shape}")
    print(
        "Toy coupling stats: "
        f"nonzero={nonzero}/{v_ce.size} ({density:.2%}), "
        f"min={v_ce.real[v_ce.real != 0].min() if nonzero else 0:.6g}, "
        f"max={v_ce.real.max():.6g}"
    )


if __name__ == "__main__":
    main()

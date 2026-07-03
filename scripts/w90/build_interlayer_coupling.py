"""Build MoS2-hBN interlayer Coulomb coupling matrices for Quatrex.

The source file stores rectangular environment-device blocks keyed by lattice
translation, with each block shaped (N_env_orb, N_device_orb). This script
assembles those primitive-cell couplings into the same transport-supercell
layout that Quatrex uses for matrices built from unit cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from quatrex.device.inputs import trim_tight_binding_matrix


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "src" / "quatrex").is_dir():
            return parent
    raise RuntimeError("Could not find repository root containing src/quatrex.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())

SOURCE = Path(
    "/usr/scratch/mont-fort11/awinka/for_cal/hBN_MoS2_heterostructure/inputs_dielectric/interlayer_coulomb_matrix.mat"
)
OUT_DIR = (
    REPO_ROOT / "examples" / "w90" / "mos2" / "dielectric_environment" / "inputs_coupling"
)

TRANSPORT_DIRECTION = "x"
NUM_TRANSPORT_CELLS = 2
NEIGHBOR_CELL_CUTOFF = (2, 2, 0)
EPSILON_R_VALUES = (10.0, 20.0, 100.0)


def epsilon_suffix(epsilon_r: float) -> str:
    return f"epsilon_{epsilon_r:g}".replace(".", "p")


def parse_key(key: str) -> tuple[int, int, int]:
    return tuple(int(part.strip()) for part in key.strip("[]").split(","))


def load_unit_cells(path: Path) -> np.ndarray:
    data = sio.loadmat(path)
    blocks = {parse_key(key): value for key, value in data.items() if not key.startswith("__")}
    coords = np.array(list(blocks))
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    shape = tuple(max_coords - min_coords + 1)
    block_shape = next(iter(blocks.values())).shape
    unit_cells = np.zeros(shape + block_shape, dtype=np.complex128)

    # Keep the same centered/negative-index convention used by the Quatrex
    # unit-cell matrix loader: e.g. key (-1, 0, 0) is stored at index -1.
    for coord, block in blocks.items():
        unit_cells[coord] = np.asarray(block, dtype=np.complex128)
    return unit_cells


def get_cross_transport_block(
    unit_cells: np.ndarray,
    supercell_size: tuple[int, int, int],
    global_shift: tuple[int, int, int],
) -> np.ndarray:
    local_shifts = np.asarray(list(np.ndindex(supercell_size)))
    supercell_size_array = np.asarray(supercell_size)
    global_shift_array = np.asarray(global_shift) * supercell_size_array
    env_orbitals, device_orbitals = unit_cells.shape[-2:]

    rows = []
    for env_cell in local_shifts:
        row = []
        for device_cell in local_shifts:
            index = tuple(device_cell - env_cell + global_shift_array)
            if any(abs(value) > unit_cells.shape[axis] // 2 for axis, value in enumerate(index)):
                block = np.zeros((env_orbitals, device_orbitals), dtype=unit_cells.dtype)
            else:
                block = unit_cells[index]
            row.append(block)
        rows.append(np.hstack(row))
    return np.vstack(rows)


def assemble_cross_matrix(unit_cells: np.ndarray) -> np.ndarray:
    transport_axis = "xyz".index(TRANSPORT_DIRECTION)
    supercell_size = tuple(
        size // 2 if axis == transport_axis else 1
        for axis, size in enumerate(unit_cells.shape[:3])
    )
    block_inds = [
        tuple(
            list((0, 0))[:transport_axis]
            + [transport_shift]
            + list((0, 0))[transport_axis:]
        )
        for transport_shift in (-1, 0, 1)
    ]
    lower_block, onsite_block, upper_block = [
        get_cross_transport_block(unit_cells, supercell_size, shift)
        for shift in block_inds
    ]

    env_block_size, device_block_size = onsite_block.shape
    out = np.zeros(
        (
            NUM_TRANSPORT_CELLS * env_block_size,
            NUM_TRANSPORT_CELLS * device_block_size,
        ),
        dtype=np.complex128,
    )

    for block_index in range(NUM_TRANSPORT_CELLS):
        env0 = block_index * env_block_size
        dev0 = block_index * device_block_size
        out[env0 : env0 + env_block_size, dev0 : dev0 + device_block_size] += onsite_block

        if block_index + 1 < NUM_TRANSPORT_CELLS:
            out[
                (block_index + 1) * env_block_size : (block_index + 2) * env_block_size,
                dev0 : dev0 + device_block_size,
            ] += lower_block
            out[
                env0 : env0 + env_block_size,
                (block_index + 1) * device_block_size : (block_index + 2) * device_block_size,
            ] += upper_block

    return out


def main() -> None:
    unit_cells = load_unit_cells(SOURCE)
    trimmed = trim_tight_binding_matrix(
        unit_cells,
        neighbor_cell_cutoff=NEIGHBOR_CELL_CUTOFF,
    )
    v_ec = assemble_cross_matrix(np.asarray(trimmed))
    v_ce = v_ec.T.conj().copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "v_ec.npy", v_ec)
    np.save(OUT_DIR / "v_ce.npy", v_ce)

    print(f"source: {SOURCE}")
    print(f"trimmed interlayer grid: {trimmed.shape[:3]}")
    print(f"saved v_ec: {v_ec.shape}, nnz={np.count_nonzero(v_ec)}")
    print(f"saved v_ce: {v_ce.shape}, nnz={np.count_nonzero(v_ce)}")

    for epsilon_r in EPSILON_R_VALUES:
        suffix = epsilon_suffix(epsilon_r)
        v_ec_scaled = v_ec / epsilon_r
        v_ce_scaled = v_ce / epsilon_r
        np.save(OUT_DIR / f"v_ec_{suffix}.npy", v_ec_scaled)
        np.save(OUT_DIR / f"v_ce_{suffix}.npy", v_ce_scaled)
        print(f"epsilon_r scaling: {epsilon_r:g}")
        print(f"saved v_ec_{suffix}: {v_ec_scaled.shape}, maxabs={np.max(np.abs(v_ec_scaled))}")
        print(f"saved v_ce_{suffix}: {v_ce_scaled.shape}, maxabs={np.max(np.abs(v_ce_scaled))}")


if __name__ == "__main__":
    main()

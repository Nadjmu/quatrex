#!/usr/bin/env python3
# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Export QTBM linear systems for one energy and one k-point.

This utility reproduces the matrix assembly used in QTBM.run and writes
assembled matrices to disk for inspection/debugging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from qttools import sparse, xp
from qttools.kernels import inplace
from qttools.utils.inplace_utils import compute_update_indices_dense, compute_update_indices_sparse
from quatrex.core.config import parse_config, setup_context
from quatrex.core.qtbm import QTBM, allocate_system_matrix
from quatrex.device import Device
import cupy as cp

MatrixMode = Literal[
    "overlap",
    "hamiltonian",
    "hamiltonian-overlap",
    "system-with-contacts",
    "full",
]


def _resolve_config_path(example: Path) -> Path:
    if example.is_file():
        return example.resolve()

    candidate = example / "quatrex_config.toml"
    if candidate.is_file():
        return candidate.resolve()

    matches = sorted(example.rglob("quatrex_config.toml"))
    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        raise ValueError(
            "Multiple quatrex_config.toml files found below the example path. "
            "Please pass the desired config file path explicitly."
        )

    raise FileNotFoundError(
        f"No quatrex_config.toml found at or below: {example.resolve()}"
    )


def _host_array(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _host_sparse(matrix):
    return matrix.get() if hasattr(matrix, "get") else matrix


def _save_csr_npz(path: Path, matrix) -> None:
    host_matrix = _host_sparse(matrix)
    np.savez(
        path,
        data=np.asarray(host_matrix.data),
        indices=np.asarray(host_matrix.indices),
        indptr=np.asarray(host_matrix.indptr),
        shape=np.asarray(host_matrix.shape, dtype=np.int64),
    )


def _apply_kpoint_phases(device: Device, k: np.ndarray) -> None:
    for r, h_r in device.hamiltonians.items():
        if r == (0, 0, 0):
            continue
        h_r.data *= xp.exp(2j * np.pi * np.dot(k, r))

    for r, s_r in device.overlap_matrices.items():
        if r == (0, 0, 0):
            continue
        s_r.data *= xp.exp(2j * np.pi * np.dot(k, r))


def _select_energy(energies: np.ndarray, energy_index: int | None, energy: float | None) -> tuple[int, float]:
    if energy_index is not None and energy is not None:
        raise ValueError("Use either --energy-index or --energy, not both.")

    if energy_index is not None:
        if energy_index < 0 or energy_index >= len(energies):
            raise IndexError(
                f"Energy index {energy_index} out of range [0, {len(energies) - 1}]."
            )
        return energy_index, float(energies[energy_index])

    if energy is not None:
        return 0, energy

    return 0, float(energies[0])


def _select_kpoint(qtbm: QTBM, k_index: int | None, k_point: tuple[float, float, float] | None) -> tuple[int, np.ndarray]:
    if k_index is not None and k_point is not None:
        raise ValueError("Use either --k-index or --k-point, not both.")

    if k_index is not None:
        if k_index < 0 or k_index >= qtbm.num_kpoints:
            raise IndexError(f"k index {k_index} out of range [0, {qtbm.num_kpoints - 1}].")
        return k_index, qtbm.kpoints[k_index, :]

    if k_point is not None:
        k_arr = np.asarray(k_point, dtype=float)
        distances = np.linalg.norm(qtbm.kpoints - k_arr[None, :], axis=1)
        k_idx = int(np.argmin(distances))
        return k_idx, qtbm.kpoints[k_idx, :]

    return 0, qtbm.kpoints[0, :]


def _assemble_base_matrix(
    system_matrix: sparse.csr_matrix,
    device: Device,
    energy: float,
) -> None:
    hamiltonian_update_indices = {
        r: compute_update_indices_sparse(system_matrix, h_r)
        for r, h_r in device.hamiltonians.items()
    }
    overlap_update_indices = {
        r: compute_update_indices_sparse(system_matrix, s_r)
        for r, s_r in device.overlap_matrices.items()
    }

    for r, h_r in device.hamiltonians.items():
        inplace.isub(system_matrix.data, h_r.data, hamiltonian_update_indices[r])

    for overlap in device.overlap_matrices.values():
        overlap.data *= energy

    for r, s_r in device.overlap_matrices.items():
        inplace.iadd(system_matrix.data, s_r.data, overlap_update_indices[r])

    for overlap in device.overlap_matrices.values():
        overlap.data *= 1.0 / energy


def _assemble_contact_self_energy(
    system_matrix: sparse.csr_matrix,
    device: Device,
    energy: float,
    k: np.ndarray,
) -> tuple[dict, np.ndarray]:
    sigma_obc_update_indices = {
        contact: compute_update_indices_dense(system_matrix, contact.orbital_indices)
        for contact in device.contacts
    }

    injection_per_contact = {}
    sigma_obc_per_contact = {}

    for contact in device.contacts:
        (
            injection_per_contact[contact],
            _,
            sigma_obc_per_contact[contact],
            _,
        ) = contact.compute_boundary(k * 2 * np.pi, xp.array([energy])) #cp.array for cupy

    for contact, sigma_obc in sigma_obc_per_contact.items():
        for k_t, sigma_obc_k in sigma_obc.items():
            inplace.isub_obc(
                system_matrix.data,
                sigma_obc_k[0, :, :],
                sigma_obc_update_indices[contact],
                k_t,
                contact.transverse_repetition_grid,
            )

    mode_counts = {
        contact: injection_per_contact[contact][0].shape[1] for contact in device.contacts
    }
    total_modes = int(sum(mode_counts.values()))
    rhs = xp.zeros((device.hamiltonians[(0, 0, 0)].shape[0], total_modes), dtype=xp.complex128)

    offset = 0
    for contact in device.contacts:
        nmodes = mode_counts[contact]
        rhs[contact.orbital_indices, offset : offset + nmodes] = injection_per_contact[contact][0]
        offset += nmodes

    return sigma_obc_per_contact, rhs


def export_system(
    example: Path,
    mode: MatrixMode,
    energy_index: int | None,
    energy: float | None,
    k_index: int | None,
    k_point: tuple[float, float, float] | None,
):
    config_path = _resolve_config_path(example)
    config = parse_config(config_path)
    setup_context(config)

    device = Device(config)
    qtbm = QTBM(device, config)

    energy_idx, selected_energy = _select_energy(
        _host_array(qtbm.electron_energies), energy_index=energy_index, energy=energy
    )
    k_idx, selected_k = _select_kpoint(qtbm, k_index=k_index, k_point=k_point)

    _apply_kpoint_phases(device, selected_k)

    if mode == "hamiltonian":
        system_matrix = allocate_system_matrix(device.hamiltonians, {}, [])
        h_update = {
            r: compute_update_indices_sparse(system_matrix, h_r)
            for r, h_r in device.hamiltonians.items()
        }
        for r, h_r in device.hamiltonians.items():
            inplace.isub(system_matrix.data, h_r.data, h_update[r])
        rhs = xp.zeros((system_matrix.shape[0], 0), dtype=xp.complex128)
    elif mode == "overlap":
        system_matrix = allocate_system_matrix(device.overlap_matrices, {}, [])
        h_update = {
            r: compute_update_indices_sparse(system_matrix, h_r)
            for r, h_r in device.overlap_matrices.items()
        }
        for r, h_r in device.overlap_matrices.items():
            inplace.isub(system_matrix.data, h_r.data, h_update[r])
        rhs = xp.zeros((system_matrix.shape[0], 0), dtype=xp.complex128)

    elif mode == "hamiltonian-overlap":
        system_matrix = allocate_system_matrix(device.hamiltonians, device.overlap_matrices, [])
        _assemble_base_matrix(system_matrix, device, selected_energy)
        rhs = xp.zeros((system_matrix.shape[0], 0), dtype=xp.complex128)

    elif mode == "system-with-contacts":
        system_matrix = allocate_system_matrix(
            device.hamiltonians, device.overlap_matrices, device.contacts
        )
        _assemble_base_matrix(system_matrix, device, selected_energy)
        rhs = xp.zeros((system_matrix.shape[0], 0), dtype=xp.complex128)

    elif mode == "full":
        system_matrix = allocate_system_matrix(
            device.hamiltonians, device.overlap_matrices, device.contacts
        )
        _assemble_base_matrix(system_matrix, device, selected_energy)
        _, rhs = _assemble_contact_self_energy(system_matrix, device, selected_energy, selected_k)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return system_matrix, config, rhs


def _parse_k_point(value: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values: kx,ky,kz")

    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("k-point values must be numeric") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one QTBM linear system A x = b at a specific energy and k-point. "
            "By default, writes the full assembled system."
        )
    )
    parser.add_argument(
        "example",
        type=Path,
        help=(
            "Path to an example folder or directly to config.toml. "
            "If a folder is given, the script looks for config.toml inside it."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["hamiltonian", "hamiltonian-overlap", "system-with-contacts", "full"],
        default="full",
        help=(
            "Assembly mode: hamiltonian -> only -H; hamiltonian-overlap -> E*S-H; "
            "system-with-contacts -> same as E*S-H but with contact sparsity allocated; "
            "full -> E*S-H-Sigma_contact and rhs injections."
        ),
    )
    parser.add_argument(
        "--energy-index",
        type=int,
        default=None,
        help="Energy index in the configured energy grid.",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=None,
        help="Energy value. The nearest configured energy point is used.",
    )
    parser.add_argument(
        "--k-index",
        type=int,
        default=None,
        help="k-point index in the Monkhorst-Pack grid.",
    )
    parser.add_argument(
        "--k-point",
        type=_parse_k_point,
        default=None,
        help="k-point as comma-separated values (fractional coordinates): kx,ky,kz.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("qtbm_system"),
        help="Output prefix. Files <prefix>_A.npz, <prefix>_b.npy, <prefix>_meta.json are written.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    export_system(
        example=args.example,
        mode=args.mode,
        output_prefix=args.output_prefix,
        energy_index=args.energy_index,
        energy=args.energy,
        k_index=args.k_index,
        k_point=args.k_point,
    )


if __name__ == "__main__":
    main()

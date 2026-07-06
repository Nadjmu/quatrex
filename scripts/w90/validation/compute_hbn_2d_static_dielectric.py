"""Estimate bilayer-hBN electronic screening from a physical 2D RPA mesh.

This diagnostic uses the full two-dimensional translation-block Hamiltonian,
physical lattice vectors from the extended XYZ file, and small commensurate
momentum transfers along the first reciprocal-lattice direction.

The reported head-only response neglects local-field coupling:

    epsilon_head(q, 0) = 1 - V_head(q) P_head(q, 0)

The 2D electronic polarizability is extracted directly from the insulating
small-q behavior of the density response:

    alpha_2D(q) = -e^2 P_total(q, 0) / (A_cell |q|^2).

This avoids relying on the long-wavelength singularity of the supplied
finite-range Coulomb matrix. The q -> 0 value is extrapolated linearly in q^2.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


TRANSLATION_PATTERN = re.compile(
    r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]$"
)
KB_EV_PER_K = 8.617333262145e-5
E2_EV_ANGSTROM = 14.3996454784255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--nk1", type=int, default=240)
    parser.add_argument("--nk2", type=int, default=60)
    parser.add_argument("--max-q-shift", type=int, default=8)
    parser.add_argument("--fermi-level", type=float, default=-1.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--broadening", type=float, default=1e-4)
    parser.add_argument("--spin-degeneracy", type=float, default=2.0)
    parser.add_argument("--fit-points", type=int, default=5)
    return parser.parse_args()


def load_blocks(path: Path) -> dict[tuple[int, int, int], np.ndarray]:
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    for key, value in raw.items():
        match = TRANSLATION_PATTERN.match(key)
        if match is None:
            continue
        translation = tuple(int(component) for component in match.groups())
        blocks[translation] = np.asarray(value, dtype=np.complex128)
    if not blocks:
        raise ValueError(f"No translation blocks found in {path}")
    return blocks


def read_lattice(path: Path) -> np.ndarray:
    line = path.read_text().splitlines()[1]
    match = re.search(r'Lattice="([^"]+)"', line)
    if match is None:
        raise ValueError(f"No Lattice field found in {path}")
    return np.fromstring(match.group(1), sep=" ", dtype=np.float64).reshape(3, 3)


def fermi(energies: np.ndarray, chemical_potential: float, temperature: float) -> np.ndarray:
    if temperature == 0.0:
        result = np.zeros_like(energies)
        result[energies < chemical_potential] = 1.0
        result[energies == chemical_potential] = 0.5
        return result
    argument = np.clip(
        (energies - chemical_potential) / (KB_EV_PER_K * temperature),
        -700.0,
        700.0,
    )
    return 1.0 / (np.exp(argument) + 1.0)


def bloch_matrices(
    blocks: dict[tuple[int, int, int], np.ndarray],
    fractional_points: np.ndarray,
) -> np.ndarray:
    translations = np.asarray(list(blocks), dtype=np.float64)
    matrices = np.stack(list(blocks.values()))
    phases = np.exp(2j * np.pi * fractional_points @ translations.T)
    result = np.tensordot(phases, matrices, axes=(1, 0))
    return 0.5 * (result + result.swapaxes(-1, -2).conj())


def scalar_head_polarization(
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    occupations: np.ndarray,
    shape: tuple[int, int],
    q_shift: int,
    broadening: float,
    spin_degeneracy: float,
) -> complex:
    nk1, nk2 = shape
    norb = eigenvectors.shape[-2]
    energies_grid = energies.reshape(nk1, nk2, norb)
    vectors_grid = eigenvectors.reshape(nk1, nk2, norb, norb)
    occupations_grid = occupations.reshape(nk1, nk2, norb)

    shifted_energies = np.roll(energies_grid, -q_shift, axis=0).reshape(-1, norb)
    shifted_vectors = np.roll(vectors_grid, -q_shift, axis=0).reshape(
        -1, norb, norb
    )
    shifted_occupations = np.roll(occupations_grid, -q_shift, axis=0).reshape(
        -1, norb
    )

    overlaps = np.einsum(
        "kib,kic->kbc", eigenvectors.conj(), shifted_vectors, optimize=True
    )
    form_factors = np.abs(overlaps) ** 2
    energy_difference = (
        shifted_energies[:, np.newaxis, :]
        - energies[:, :, np.newaxis]
    )
    occupation_difference = (
        shifted_occupations[:, np.newaxis, :]
        - occupations[:, :, np.newaxis]
    )
    polarization_total_density = np.sum(
        occupation_difference
        * form_factors
        / (energy_difference - 1j * broadening),
        dtype=np.complex128,
    )
    polarization_total_density *= spin_degeneracy / energies.shape[0]

    # The normalized uniform orbital vector defines the density "head".
    return polarization_total_density / norb


def main() -> None:
    args = parse_args()
    if args.nk1 < 2 or args.nk2 < 2:
        raise ValueError("Both k-grid dimensions must be at least two.")
    if not 1 <= args.max_q_shift < args.nk1 // 2:
        raise ValueError("max-q-shift must lie between 1 and nk1/2.")

    lattice = read_lattice(args.input_dir / "lattice.xyz")
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T

    k1 = np.arange(args.nk1, dtype=np.float64) / args.nk1 - 0.5
    k2 = np.arange(args.nk2, dtype=np.float64) / args.nk2 - 0.5
    k1_grid, k2_grid = np.meshgrid(k1, k2, indexing="ij")
    fractional_k = np.column_stack(
        (
            k1_grid.ravel(),
            k2_grid.ravel(),
            np.zeros(k1_grid.size),
        )
    )

    hamiltonian_blocks = load_blocks(args.input_dir / "hamiltonian.mat")
    coulomb_blocks = load_blocks(args.input_dir / "coulomb_matrix.mat")
    hamiltonian_k = bloch_matrices(hamiltonian_blocks, fractional_k)
    energies, eigenvectors = np.linalg.eigh(hamiltonian_k)
    del hamiltonian_k
    occupations = fermi(energies, args.fermi_level, args.temperature)

    norb = energies.shape[1]
    uniform_probe = np.ones(norb, dtype=np.complex128) / np.sqrt(norb)
    shifts = np.arange(1, args.max_q_shift + 1)
    fractional_q = np.column_stack(
        (
            shifts / args.nk1,
            np.zeros(shifts.size),
            np.zeros(shifts.size),
        )
    )
    q_vectors = fractional_q @ reciprocal
    q_magnitudes = np.linalg.norm(q_vectors, axis=1)
    coulomb_q = bloch_matrices(coulomb_blocks, fractional_q)

    p_head = np.empty(shifts.size, dtype=np.complex128)
    v_head = np.empty(shifts.size, dtype=np.float64)
    epsilon_head = np.empty(shifts.size, dtype=np.complex128)
    for index, shift in enumerate(shifts):
        p_head[index] = scalar_head_polarization(
            energies,
            eigenvectors,
            occupations,
            (args.nk1, args.nk2),
            int(shift),
            args.broadening,
            args.spin_degeneracy,
        )
        v_head[index] = float(
            np.real(uniform_probe.conj() @ coulomb_q[index] @ uniform_probe)
        )
        epsilon_head[index] = 1.0 - v_head[index] * p_head[index]
        print(
            f"q shift {shift}/{args.max_q_shift}: "
            f"|q|={q_magnitudes[index]:.6f} 1/Angstrom, "
            f"Re epsilon_head={epsilon_head[index].real:.8f}",
            flush=True,
        )

    fit_count = min(args.fit_points, shifts.size)
    cell_area = float(np.linalg.norm(np.cross(lattice[0], lattice[1])))
    polarization_total = norb * p_head
    alpha_2d_by_q = (
        -E2_EV_ANGSTROM
        * polarization_total.real
        / (cell_area * q_magnitudes**2)
    )
    alpha_q2_slope, alpha_2d = np.polyfit(
        q_magnitudes[:fit_count] ** 2,
        alpha_2d_by_q[:fit_count],
        1,
    )
    fitted = alpha_2d + alpha_q2_slope * q_magnitudes[:fit_count] ** 2
    residual = alpha_2d_by_q[:fit_count] - fitted
    r_squared = 1.0 - np.sum(residual**2) / np.sum(
        (alpha_2d_by_q[:fit_count] - alpha_2d_by_q[:fit_count].mean()) ** 2
    )
    supercell_epsilon = 1.0 + 4.0 * np.pi * alpha_2d / lattice[2, 2]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_prefix.with_suffix(".npz"),
        lattice=lattice,
        reciprocal_lattice=reciprocal,
        q_fractional=fractional_q,
        q_vectors_inverse_angstrom=q_vectors,
        q_magnitudes_inverse_angstrom=q_magnitudes,
        p_head=p_head,
        polarization_total=polarization_total,
        v_head=v_head,
        epsilon_head=epsilon_head,
        cell_area_angstrom2=cell_area,
        alpha_2d_by_q_angstrom=alpha_2d_by_q,
        fit_count=fit_count,
        alpha_q2_slope_angstrom3=alpha_q2_slope,
        alpha_2d_angstrom=alpha_2d,
        fit_r_squared=r_squared,
        supercell_epsilon_electronic=supercell_epsilon,
        nk1=args.nk1,
        nk2=args.nk2,
    )

    summary = {
        "nk1": args.nk1,
        "nk2": args.nk2,
        "fit_points": fit_count,
        "alpha_2d_angstrom": float(alpha_2d),
        "alpha_q2_slope_angstrom3": float(alpha_q2_slope),
        "fit_r_squared": float(r_squared),
        "cell_area_angstrom2": cell_area,
        "supercell_height_angstrom": float(lattice[2, 2]),
        "supercell_epsilon_electronic": float(supercell_epsilon),
        "smallest_q_inverse_angstrom": float(q_magnitudes[0]),
        "smallest_q_alpha_2d_angstrom": float(alpha_2d_by_q[0]),
        "scope": (
            "2D head-only electronic RPA; local-field and ionic terms excluded"
        ),
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    ax.plot(
        q_magnitudes,
        alpha_2d_by_q,
        "o-",
        color="tab:blue",
        linewidth=2.0,
        label=r"$\alpha_{2D}(q)$",
    )
    q_fit = np.linspace(0.0, q_magnitudes[fit_count - 1], 150)
    ax.plot(
        q_fit,
        alpha_2d + alpha_q2_slope * q_fit**2,
        "--",
        color="tab:orange",
        linewidth=1.8,
        label=rf"$q\to0$ fit: $\alpha_{{2D}}={alpha_2d:.3f}$ Å",
    )
    ax.scatter([0.0], [alpha_2d], marker="x", s=70, color="black")
    ax.set_title("Bilayer hBN Static Electronic 2D Polarizability")
    ax.set_xlabel(r"$|q|$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$\alpha_{2D}$ ($\mathrm{\AA}$)")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend()
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

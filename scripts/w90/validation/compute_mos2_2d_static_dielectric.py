"""Estimate monolayer-MoS2 static electronic 2D polarizability."""

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
    parser.add_argument("--nk2", type=int, default=120)
    parser.add_argument("--max-q-shift", type=int, default=8)
    parser.add_argument("--fermi-level", type=float, default=-1.154984046)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--broadening", type=float, default=1e-4)
    parser.add_argument("--spin-degeneracy", type=float, default=2.0)
    parser.add_argument("--fit-points", type=int, default=5)
    parser.add_argument(
        "--effective-thickness",
        type=float,
        default=6.15,
        help="Optional bulk-repeat thickness used to quote epsilon_eff (Angstrom).",
    )
    return parser.parse_args()


def load_blocks(path: Path) -> dict[tuple[int, int, int], np.ndarray]:
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    blocks = {}
    for key, value in raw.items():
        match = TRANSLATION_PATTERN.match(key)
        if match is not None:
            blocks[tuple(map(int, match.groups()))] = np.asarray(
                value, dtype=np.complex128
            )
    if not blocks:
        raise ValueError(f"No translation blocks found in {path}")
    return blocks


def read_lattice(path: Path) -> np.ndarray:
    line = path.read_text().splitlines()[1]
    match = re.search(r'Lattice="([^"]+)"', line)
    if match is None:
        raise ValueError(f"No Lattice field found in {path}")
    return np.fromstring(match.group(1), sep=" ").reshape(3, 3)


def fermi(energies: np.ndarray, chemical_potential: float, temperature: float):
    argument = np.clip(
        (energies - chemical_potential) / (KB_EV_PER_K * temperature),
        -700.0,
        700.0,
    )
    return 1.0 / (np.exp(argument) + 1.0)


def bloch_matrices(blocks, fractional_points):
    translations = np.asarray(list(blocks), dtype=np.float64)
    matrices = np.stack(list(blocks.values()))
    phases = np.exp(2j * np.pi * fractional_points @ translations.T)
    result = np.tensordot(phases, matrices, axes=(1, 0))
    return 0.5 * (result + result.swapaxes(-1, -2).conj())


def scalar_head_polarization(
    energies,
    eigenvectors,
    occupations,
    shape,
    q_shift,
    broadening,
    spin_degeneracy,
):
    nk1, nk2 = shape
    norb = energies.shape[1]
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
    energy_difference = (
        shifted_energies[:, np.newaxis, :] - energies[:, :, np.newaxis]
    )
    occupation_difference = (
        shifted_occupations[:, np.newaxis, :] - occupations[:, :, np.newaxis]
    )
    polarization = np.sum(
        occupation_difference
        * np.abs(overlaps) ** 2
        / (energy_difference - 1j * broadening),
        dtype=np.complex128,
    )
    return polarization * spin_degeneracy / energies.shape[0] / norb


def main() -> None:
    args = parse_args()
    lattice = read_lattice(args.input_dir / "structure.xyz")
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T

    k1 = np.arange(args.nk1) / args.nk1 - 0.5
    k2 = np.arange(args.nk2) / args.nk2 - 0.5
    k1_grid, k2_grid = np.meshgrid(k1, k2, indexing="ij")
    fractional_k = np.column_stack(
        (k1_grid.ravel(), k2_grid.ravel(), np.zeros(k1_grid.size))
    )

    blocks = load_blocks(args.input_dir / "hamiltonian.mat")
    hamiltonian_k = bloch_matrices(blocks, fractional_k)
    energies, eigenvectors = np.linalg.eigh(hamiltonian_k)
    del hamiltonian_k
    occupations = fermi(energies, args.fermi_level, args.temperature)

    # Confirm the intrinsic 14-band filling used for this 22-orbital model.
    valence_edge = float(energies[:, 13].max())
    conduction_edge = float(energies[:, 14].min())
    gap = conduction_edge - valence_edge

    shifts = np.arange(1, args.max_q_shift + 1)
    fractional_q = np.column_stack(
        (shifts / args.nk1, np.zeros(shifts.size), np.zeros(shifts.size))
    )
    q_vectors = fractional_q @ reciprocal
    q_magnitudes = np.linalg.norm(q_vectors, axis=1)
    p_head = np.empty(shifts.size, dtype=np.complex128)
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
        print(
            f"q shift {shift}/{args.max_q_shift}: "
            f"|q|={q_magnitudes[index]:.6f} 1/Angstrom",
            flush=True,
        )

    norb = energies.shape[1]
    cell_area = float(np.linalg.norm(np.cross(lattice[0], lattice[1])))
    polarization_total = norb * p_head
    alpha_by_q = (
        -E2_EV_ANGSTROM
        * polarization_total.real
        / (cell_area * q_magnitudes**2)
    )
    fit_count = min(args.fit_points, shifts.size)
    slope, alpha_2d = np.polyfit(
        q_magnitudes[:fit_count] ** 2, alpha_by_q[:fit_count], 1
    )
    fitted = alpha_2d + slope * q_magnitudes[:fit_count] ** 2
    residual = alpha_by_q[:fit_count] - fitted
    denominator = np.sum(
        (alpha_by_q[:fit_count] - alpha_by_q[:fit_count].mean()) ** 2
    )
    r_squared = 1.0 - np.sum(residual**2) / denominator
    effective_epsilon = (
        1.0 + 4.0 * np.pi * alpha_2d / args.effective_thickness
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_prefix.with_suffix(".npz"),
        lattice=lattice,
        reciprocal_lattice=reciprocal,
        q_magnitudes_inverse_angstrom=q_magnitudes,
        p_head=p_head,
        polarization_total=polarization_total,
        cell_area_angstrom2=cell_area,
        alpha_2d_by_q_angstrom=alpha_by_q,
        fit_count=fit_count,
        alpha_q2_slope_angstrom3=slope,
        alpha_2d_angstrom=alpha_2d,
        fit_r_squared=r_squared,
        valence_edge_ev=valence_edge,
        conduction_edge_ev=conduction_edge,
        band_gap_ev=gap,
        fermi_level_ev=args.fermi_level,
        effective_thickness_angstrom=args.effective_thickness,
        effective_epsilon_electronic=effective_epsilon,
        nk1=args.nk1,
        nk2=args.nk2,
    )
    summary = {
        "nk1": args.nk1,
        "nk2": args.nk2,
        "fit_points": fit_count,
        "alpha_2d_angstrom": float(alpha_2d),
        "alpha_q2_slope_angstrom3": float(slope),
        "fit_r_squared": float(r_squared),
        "cell_area_angstrom2": cell_area,
        "supercell_height_angstrom": float(lattice[2, 2]),
        "valence_edge_ev": valence_edge,
        "conduction_edge_ev": conduction_edge,
        "band_gap_ev": gap,
        "fermi_level_ev": args.fermi_level,
        "effective_thickness_angstrom": args.effective_thickness,
        "effective_epsilon_electronic": float(effective_epsilon),
        "smallest_q_inverse_angstrom": float(q_magnitudes[0]),
        "smallest_q_alpha_2d_angstrom": float(alpha_by_q[0]),
        "scope": "2D head-only electronic RPA; local-field and ionic terms excluded",
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.1, 4.25), constrained_layout=True)
    ax.plot(q_magnitudes, alpha_by_q, "o-", color="#1f5a91", label=r"$\alpha_{2D}(q)$")
    q_fit = np.linspace(0.0, q_magnitudes[fit_count - 1], 150)
    ax.plot(
        q_fit,
        alpha_2d + slope * q_fit**2,
        "--",
        color="#d95f02",
        label=rf"$q\to0$: $\alpha_{{2D}}={alpha_2d:.3f}$ Å",
    )
    ax.scatter([0.0], [alpha_2d], marker="x", color="black", s=55, zorder=3)
    ax.set_title(r"MoS$_2$ Static Electronic 2D Polarizability")
    ax.set_xlabel(r"Momentum transfer, $|q|$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"2D polarizability, $\alpha_{2D}$ ($\mathrm{\AA}$)")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend()
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

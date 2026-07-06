"""Validate dielectric response built from RPA polarization.

This script computes

    epsilon(q, omega) = I - V(q) P(q, omega)

using the same translation-resolved Hamiltonian and Coulomb matrix conventions as
the RPA screening workflow. For matrix-valued epsilon it reports an effective
trace average and a trace-averaged loss function:

    epsilon_eff = Tr epsilon / N
    loss = -Im Tr epsilon^{-1} / N

These are compact diagnostics for presentations and qualitative comparison.
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

RPAPolarization = rpa_compute.RPAPolarization
build_bloch_hamiltonian = rpa_compute.build_bloch_hamiltonian
build_uniform_brillouin_zone_mesh = rpa_compute.build_uniform_brillouin_zone_mesh
load_translation_blocks = rpa_compute.load_translation_blocks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dielectric function from RPA P(q, omega)."
    )
    parser.add_argument(
        "--hamiltonian",
        type=Path,
        default=EXAMPLE_ROOT / "inputs" / "hamiltonian.mat",
    )
    parser.add_argument(
        "--coulomb",
        type=Path,
        default=EXAMPLE_ROOT / "inputs" / "coulomb_matrix.mat",
    )
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--lattice-constant", type=float, default=1.0)
    parser.add_argument("--num-k", type=int, default=80)
    parser.add_argument("--num-q", type=int, default=41)
    parser.add_argument("--num-frequency", type=int, default=180)
    parser.add_argument("--max-frequency", type=float, default=8.0)
    parser.add_argument("--fermi-level", type=float, default=-3.6)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--broadening", type=float, default=0.05)
    parser.add_argument("--epsilon-r", type=float, default=1.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_ANALYSIS_ROOT
        / "validation_outputs"
        / "cnt_hbn_general"
        / "carbon_nanotube"
        / "dielectric_function"
        / "validation_cnt_rpa_dielectric",
    )
    return parser.parse_args()


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _compute_dielectric_response(args: argparse.Namespace):
    mesh = build_uniform_brillouin_zone_mesh(
        num_k_points=args.num_k,
        num_q_points=args.num_q,
        num_frequencies=args.num_frequency,
        max_frequency=args.max_frequency,
        lattice_constant=args.lattice_constant,
        include_zero_q=True,
    )

    hamiltonian_blocks = load_translation_blocks(args.hamiltonian)
    coulomb_blocks = load_translation_blocks(args.coulomb)

    solver = RPAPolarization(frequency_axis="real")
    polarization_result = solver.solve_from_translation_blocks(
        translation_blocks=hamiltonian_blocks,
        mesh=mesh,
        chemical_potential=args.fermi_level,
        temperature=args.temperature,
        periodic_axis=args.axis,
        lattice_constant=args.lattice_constant,
        broadening=args.broadening,
    )
    polarization = np.asarray(polarization_result.polarization, dtype=np.complex128)

    coulomb_matrices = build_bloch_hamiltonian(
        coulomb_blocks,
        mesh.q_points,
        periodic_axis=args.axis,
        lattice_constant=args.lattice_constant,
    )
    coulomb_matrices = np.asarray(coulomb_matrices, dtype=np.complex128) / args.epsilon_r

    nq, norb, _ = coulomb_matrices.shape
    nw = mesh.frequencies.size
    identity = np.eye(norb, dtype=np.complex128)
    epsilon = np.empty((nq, nw, norb, norb), dtype=np.complex128)
    epsilon_inverse_trace = np.empty((nq, nw), dtype=np.complex128)

    for q_index in range(nq):
        for frequency_index in range(nw):
            epsilon_matrix = identity - polarization[q_index, frequency_index] * coulomb_matrices[q_index]
            epsilon[q_index, frequency_index] = epsilon_matrix
            epsilon_inverse_trace[q_index, frequency_index] = np.trace(
                np.linalg.solve(epsilon_matrix, identity)
            ) / norb

    epsilon_eff = np.trace(epsilon, axis1=-2, axis2=-1) / norb
    loss = -np.imag(epsilon_inverse_trace)
    return mesh.q_points, mesh.frequencies, polarization, coulomb_matrices, epsilon_eff, loss


def _make_plots(
    *,
    output_prefix: Path,
    q_points: np.ndarray,
    frequencies: np.ndarray,
    epsilon_eff: np.ndarray,
    loss: np.ndarray,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    extent = [
        float(frequencies[0]),
        float(frequencies[-1]),
        float(q_points[0] / np.pi),
        float(q_points[-1] / np.pi),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    fields = [
        (np.real(epsilon_eff), r"Re $\epsilon_\mathrm{eff}$", "viridis"),
        (np.imag(epsilon_eff), r"Im $\epsilon_\mathrm{eff}$", "coolwarm"),
        (loss, r"Loss $-\mathrm{Im}\,\mathrm{Tr}\,\epsilon^{-1}/N$", "magma"),
    ]
    for ax, (field, title, cmap) in zip(axes, fields, strict=True):
        image = ax.imshow(
            field,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$\omega$ (eV)")
        ax.set_ylabel(r"$q / \pi$")
        fig.colorbar(image, ax=ax)
    heatmap_path = output_prefix.with_name(output_prefix.name + "_heatmaps.png")
    fig.savefig(heatmap_path, dpi=220)
    plt.close(fig)

    q_indices = sorted(
        {
            _nearest_index(q_points, 0.25 * np.pi),
            _nearest_index(q_points, 0.5 * np.pi),
            _nearest_index(q_points, 0.75 * np.pi),
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for q_index in q_indices:
        label = f"q/pi={q_points[q_index] / np.pi:.2f}"
        axes[0].plot(frequencies, np.real(epsilon_eff[q_index]), label=label)
        axes[1].plot(frequencies, np.imag(epsilon_eff[q_index]), label=label)
        axes[2].plot(frequencies, loss[q_index], label=label)
    axes[0].set_title(r"Re $\epsilon_\mathrm{eff}$")
    axes[1].set_title(r"Im $\epsilon_\mathrm{eff}$")
    axes[2].set_title("Loss")
    for ax in axes:
        ax.set_xlabel(r"$\omega$ (eV)")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False)
    line_path = output_prefix.with_name(output_prefix.name + "_q_lines.png")
    fig.savefig(line_path, dpi=220)
    plt.close(fig)

    print(f"Wrote plot: {heatmap_path.resolve()}")
    print(f"Wrote plot: {line_path.resolve()}")


def main() -> None:
    args = _parse_args()
    q_points, frequencies, polarization, coulomb_matrices, epsilon_eff, loss = (
        _compute_dielectric_response(args)
    )

    peak_loss_index = np.unravel_index(int(np.argmax(loss)), loss.shape)
    zero_q_index = _nearest_index(q_points, 0.0)
    finite_q_loss = np.delete(loss, zero_q_index, axis=0)

    output_prefix = args.output_prefix
    _make_plots(
        output_prefix=output_prefix,
        q_points=q_points,
        frequencies=frequencies,
        epsilon_eff=epsilon_eff,
        loss=loss,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_prefix.with_suffix(".npz")
    np.savez(
        data_path,
        q_points=q_points,
        frequencies=frequencies,
        polarization=polarization,
        coulomb_matrices=coulomb_matrices,
        epsilon_eff=epsilon_eff,
        loss=loss,
    )

    print(f"Hamiltonian: {args.hamiltonian.resolve()}")
    print(f"Coulomb matrix: {args.coulomb.resolve()}")
    print(f"RPA dielectric grid: nk={args.num_k}, nq={args.num_q}, nw={args.num_frequency}")
    print(
        "Peak loss at "
        f"q/pi={q_points[peak_loss_index[0]] / np.pi:.4g}, "
        f"omega={frequencies[peak_loss_index[1]]:.4g} eV, "
        f"loss={loss[peak_loss_index]:.6g}"
    )
    print(f"Max |epsilon_eff - 1| at q=0: {np.max(np.abs(epsilon_eff[zero_q_index] - 1.0)):.6e}")
    print(f"Median finite-q loss: {np.median(finite_q_loss):.6e}")
    print(f"Wrote data: {data_path.resolve()}")


if __name__ == "__main__":
    main()

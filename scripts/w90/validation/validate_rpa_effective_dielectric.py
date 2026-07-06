"""Validate an effective scalar dielectric function from RPA.

The full CNT dielectric object is matrix-valued in an orbital basis. To make a
literature-style scalar diagnostic, this script projects the Coulomb interaction
onto a clearly stated scalar channel and combines it with scalar RPA P(q, omega):

    epsilon_eff(q, omega) = 1 - v_eff(q) P(q, omega)
    loss_eff(q, omega) = -Im 1 / epsilon_eff(q, omega)

The default projection uses the largest positive eigenvalue of the Hermitian
part of V(q), i.e. the strongest positive Coulomb mode at each q.
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
        description="Validate projected scalar dielectric response from RPA."
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
    parser.add_argument("--epsilon-r", type=float, default=4.0)
    parser.add_argument(
        "--projection",
        choices=("dominant-positive", "uniform", "trace-absolute"),
        default="dominant-positive",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_ANALYSIS_ROOT
        / "validation_outputs"
        / "cnt_hbn_general"
        / "carbon_nanotube"
        / "dielectric_function"
        / "validation_cnt_rpa_effective_dielectric",
    )
    return parser.parse_args()


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _project_coulomb(coulomb_matrices: np.ndarray, projection: str) -> np.ndarray:
    v_eff = np.empty(coulomb_matrices.shape[0], dtype=np.float64)
    norb = coulomb_matrices.shape[-1]

    if projection == "uniform":
        probe = np.ones(norb, dtype=np.complex128) / np.sqrt(norb)
        for q_index, matrix in enumerate(coulomb_matrices):
            hermitian = 0.5 * (matrix + matrix.conj().T)
            v_eff[q_index] = float(np.real(probe.conj() @ hermitian @ probe))
        return v_eff

    if projection == "trace-absolute":
        trace = np.trace(coulomb_matrices, axis1=-2, axis2=-1) / norb
        return np.abs(trace).astype(np.float64, copy=False)

    if projection == "dominant-positive":
        for q_index, matrix in enumerate(coulomb_matrices):
            hermitian = 0.5 * (matrix + matrix.conj().T)
            eigenvalues = np.linalg.eigvalsh(hermitian)
            positive = eigenvalues[eigenvalues > 0.0]
            if positive.size == 0:
                raise ValueError(f"V(q_index={q_index}) has no positive eigenvalue.")
            v_eff[q_index] = float(positive[-1])
        return v_eff

    raise ValueError(f"Unsupported projection: {projection}")


def _compute(args: argparse.Namespace):
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
    result = solver.solve_from_translation_blocks(
        translation_blocks=hamiltonian_blocks,
        mesh=mesh,
        chemical_potential=args.fermi_level,
        temperature=args.temperature,
        periodic_axis=args.axis,
        lattice_constant=args.lattice_constant,
        broadening=args.broadening,
    )
    polarization = np.asarray(result.polarization, dtype=np.complex128)
    coulomb_matrices = build_bloch_hamiltonian(
        coulomb_blocks,
        mesh.q_points,
        periodic_axis=args.axis,
        lattice_constant=args.lattice_constant,
    )
    coulomb_matrices = np.asarray(coulomb_matrices, dtype=np.complex128) / args.epsilon_r
    v_eff = _project_coulomb(coulomb_matrices, args.projection)

    epsilon_eff = 1.0 - v_eff[:, np.newaxis] * polarization
    loss_eff = -np.imag(1.0 / epsilon_eff)
    return mesh.q_points, mesh.frequencies, polarization, v_eff, epsilon_eff, loss_eff


def _make_plots(
    *,
    output_prefix: Path,
    q_points: np.ndarray,
    frequencies: np.ndarray,
    v_eff: np.ndarray,
    epsilon_eff: np.ndarray,
    loss_eff: np.ndarray,
    projection: str,
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
        (loss_eff, r"$-\mathrm{Im}\,1/\epsilon_\mathrm{eff}$", "magma"),
    ]
    for ax, (field, title, cmap) in zip(axes, fields, strict=True):
        image = ax.imshow(field, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(r"$\omega$ (eV)")
        ax.set_ylabel(r"$q / \pi$")
        fig.colorbar(image, ax=ax)
    fig.suptitle(f"Effective dielectric response, projection={projection}")
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
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(q_points / np.pi, v_eff, color="#1f4e79")
    axes[0].set_title(r"$v_\mathrm{eff}(q)$")
    axes[0].set_xlabel(r"$q / \pi$")

    for q_index in q_indices:
        label = f"q/pi={q_points[q_index] / np.pi:.2f}"
        axes[1].plot(frequencies, np.real(epsilon_eff[q_index]), label=label)
        axes[2].plot(frequencies, np.imag(epsilon_eff[q_index]), label=label)
        axes[3].plot(frequencies, loss_eff[q_index], label=label)
    axes[1].set_title(r"Re $\epsilon_\mathrm{eff}$")
    axes[2].set_title(r"Im $\epsilon_\mathrm{eff}$")
    axes[3].set_title("Loss")
    for ax in axes:
        ax.grid(True, alpha=0.25, linewidth=0.5)
    for ax in axes[1:]:
        ax.set_xlabel(r"$\omega$ (eV)")
        ax.legend(frameon=False)
    line_path = output_prefix.with_name(output_prefix.name + "_q_lines.png")
    fig.savefig(line_path, dpi=220)
    plt.close(fig)

    print(f"Wrote plot: {heatmap_path.resolve()}")
    print(f"Wrote plot: {line_path.resolve()}")


def main() -> None:
    args = _parse_args()
    q_points, frequencies, polarization, v_eff, epsilon_eff, loss_eff = _compute(args)

    output_prefix = args.output_prefix
    _make_plots(
        output_prefix=output_prefix,
        q_points=q_points,
        frequencies=frequencies,
        v_eff=v_eff,
        epsilon_eff=epsilon_eff,
        loss_eff=loss_eff,
        projection=args.projection,
    )

    data_path = output_prefix.with_suffix(".npz")
    np.savez(
        data_path,
        q_points=q_points,
        frequencies=frequencies,
        polarization=polarization,
        v_eff=v_eff,
        epsilon_eff=epsilon_eff,
        loss_eff=loss_eff,
        projection=np.array(args.projection),
    )

    peak = np.unravel_index(int(np.argmax(loss_eff)), loss_eff.shape)
    zero_q = _nearest_index(q_points, 0.0)
    print(f"Projection: {args.projection}")
    print(f"v_eff min/max/median: {np.min(v_eff):.6g}, {np.max(v_eff):.6g}, {np.median(v_eff):.6g}")
    print(
        "Peak effective loss at "
        f"q/pi={q_points[peak[0]] / np.pi:.4g}, "
        f"omega={frequencies[peak[1]]:.4g} eV, "
        f"loss={loss_eff[peak]:.6g}"
    )
    print(f"Fraction loss < 0: {np.mean(loss_eff < -1e-12):.6g}")
    print(f"Max |epsilon_eff - 1| at q=0: {np.max(np.abs(epsilon_eff[zero_q] - 1.0)):.6e}")
    print(f"Wrote data: {data_path.resolve()}")


if __name__ == "__main__":
    main()

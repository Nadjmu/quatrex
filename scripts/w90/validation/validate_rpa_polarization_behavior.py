"""Validate physical trends of the RPA polarization P(q, omega).

This script checks qualitative behavior rather than matching a literature curve:

- P(q=0, omega) is near zero for the scalar density vertex.
- P(q, omega) varies smoothly with q on a uniform grid.
- P changes when chemical potential/doping changes.
- P changes when temperature changes.

The calculation uses the same RPA helpers as the dielectric-screening workflow.
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
build_uniform_brillouin_zone_mesh = rpa_compute.build_uniform_brillouin_zone_mesh
load_translation_blocks = rpa_compute.load_translation_blocks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate qualitative physical behavior of RPA P(q, omega)."
    )
    parser.add_argument(
        "--hamiltonian",
        type=Path,
        default=EXAMPLE_ROOT / "inputs" / "hamiltonian.mat",
        help="Translation-resolved Hamiltonian .mat file.",
    )
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--lattice-constant", type=float, default=1.0)
    parser.add_argument("--num-k", type=int, default=80)
    parser.add_argument("--num-q", type=int, default=41)
    parser.add_argument("--num-frequency", type=int, default=180)
    parser.add_argument("--max-frequency", type=float, default=8.0)
    parser.add_argument("--fermi-level", type=float, default=-3.6)
    parser.add_argument("--doped-fermi-level", type=float, default=-3.3)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--hot-temperature", type=float, default=1000.0)
    parser.add_argument("--broadening", type=float, default=0.05)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_ANALYSIS_ROOT
        / "validation_outputs"
        / "cnt_hbn_general"
        / "carbon_nanotube"
        / "polarization_behavior"
        / "validation_cnt_rpa_polarization",
    )
    return parser.parse_args()


def _solve_scalar_polarization(
    *,
    hamiltonian: Path,
    axis: int,
    lattice_constant: float,
    num_k: int,
    num_q: int,
    num_frequency: int,
    max_frequency: float,
    fermi_level: float,
    temperature: float,
    broadening: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = load_translation_blocks(hamiltonian)
    mesh = build_uniform_brillouin_zone_mesh(
        num_k_points=num_k,
        num_q_points=num_q,
        num_frequencies=num_frequency,
        max_frequency=max_frequency,
        lattice_constant=lattice_constant,
        include_zero_q=True,
    )
    solver = RPAPolarization(frequency_axis="real")
    result = solver.solve_from_translation_blocks(
        translation_blocks=blocks,
        mesh=mesh,
        chemical_potential=fermi_level,
        temperature=temperature,
        periodic_axis=axis,
        lattice_constant=lattice_constant,
        broadening=broadening,
    )
    return mesh.q_points, mesh.frequencies, np.asarray(result.polarization)


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _relative_norm_difference(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(a)), 1e-30)
    return float(np.linalg.norm(a - b) / denominator)


def _make_plots(
    *,
    output_prefix: Path,
    q_points: np.ndarray,
    frequencies: np.ndarray,
    p_base: np.ndarray,
    p_doped: np.ndarray,
    p_hot: np.ndarray,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    extent = [
        float(frequencies[0]),
        float(frequencies[-1]),
        float(q_points[0] / np.pi),
        float(q_points[-1] / np.pi),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    im0 = axes[0].imshow(
        np.abs(p_base),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[0].set_title(r"$|P(q,\omega)|$")
    axes[0].set_xlabel(r"$\omega$ (eV)")
    axes[0].set_ylabel(r"$q / \pi$")
    fig.colorbar(im0, ax=axes[0], label="a.u.")

    im1 = axes[1].imshow(
        np.imag(p_base),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
    )
    axes[1].set_title(r"Im $P(q,\omega)$")
    axes[1].set_xlabel(r"$\omega$ (eV)")
    axes[1].set_ylabel(r"$q / \pi$")
    fig.colorbar(im1, ax=axes[1], label="a.u.")
    heatmap_path = output_prefix.with_name(output_prefix.name + "_heatmap.png")
    fig.savefig(heatmap_path, dpi=220)
    plt.close(fig)

    q_indices = sorted(
        {
            _nearest_index(q_points, 0.0),
            _nearest_index(q_points, 0.25 * np.pi),
            _nearest_index(q_points, 0.5 * np.pi),
            _nearest_index(q_points, 0.75 * np.pi),
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    for q_index in q_indices:
        label = f"q/pi={q_points[q_index] / np.pi:.2f}"
        axes[0].plot(frequencies, np.real(p_base[q_index]), linewidth=1.4, label=label)
        axes[1].plot(frequencies, np.imag(p_base[q_index]), linewidth=1.4, label=label)
    axes[0].set_title(r"Re $P(q,\omega)$")
    axes[1].set_title(r"Im $P(q,\omega)$")
    for ax in axes:
        ax.set_xlabel(r"$\omega$ (eV)")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False)
    line_path = output_prefix.with_name(output_prefix.name + "_q_lines.png")
    fig.savefig(line_path, dpi=220)
    plt.close(fig)

    q_index = _nearest_index(q_points, 0.5 * np.pi)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    axes[0].plot(frequencies, np.abs(p_base[q_index]), label="baseline", linewidth=1.6)
    axes[0].plot(frequencies, np.abs(p_doped[q_index]), label="doped", linewidth=1.4)
    axes[0].set_title(r"Doping response at $q \approx 0.5\pi$")
    axes[1].plot(frequencies, np.abs(p_base[q_index]), label="baseline", linewidth=1.6)
    axes[1].plot(frequencies, np.abs(p_hot[q_index]), label="hotter", linewidth=1.4)
    axes[1].set_title(r"Temperature response at $q \approx 0.5\pi$")
    for ax in axes:
        ax.set_xlabel(r"$\omega$ (eV)")
        ax.set_ylabel(r"$|P|$")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False)
    sensitivity_path = output_prefix.with_name(output_prefix.name + "_sensitivity.png")
    fig.savefig(sensitivity_path, dpi=220)
    plt.close(fig)

    np.savez(
        output_prefix.with_suffix(".npz"),
        q_points=q_points,
        frequencies=frequencies,
        p_base=p_base,
        p_doped=p_doped,
        p_hot=p_hot,
    )

    print(f"Wrote plot: {heatmap_path.resolve()}")
    print(f"Wrote plot: {line_path.resolve()}")
    print(f"Wrote plot: {sensitivity_path.resolve()}")
    print(f"Wrote data: {output_prefix.with_suffix('.npz').resolve()}")


def main() -> None:
    args = _parse_args()

    q_points, frequencies, p_base = _solve_scalar_polarization(
        hamiltonian=args.hamiltonian,
        axis=args.axis,
        lattice_constant=args.lattice_constant,
        num_k=args.num_k,
        num_q=args.num_q,
        num_frequency=args.num_frequency,
        max_frequency=args.max_frequency,
        fermi_level=args.fermi_level,
        temperature=args.temperature,
        broadening=args.broadening,
    )
    _, _, p_doped = _solve_scalar_polarization(
        hamiltonian=args.hamiltonian,
        axis=args.axis,
        lattice_constant=args.lattice_constant,
        num_k=args.num_k,
        num_q=args.num_q,
        num_frequency=args.num_frequency,
        max_frequency=args.max_frequency,
        fermi_level=args.doped_fermi_level,
        temperature=args.temperature,
        broadening=args.broadening,
    )
    _, _, p_hot = _solve_scalar_polarization(
        hamiltonian=args.hamiltonian,
        axis=args.axis,
        lattice_constant=args.lattice_constant,
        num_k=args.num_k,
        num_q=args.num_q,
        num_frequency=args.num_frequency,
        max_frequency=args.max_frequency,
        fermi_level=args.fermi_level,
        temperature=args.hot_temperature,
        broadening=args.broadening,
    )

    zero_q_index = _nearest_index(q_points, 0.0)
    finite_q_norms = np.linalg.norm(np.delete(p_base, zero_q_index, axis=0), axis=1)
    zero_q_norm = float(np.linalg.norm(p_base[zero_q_index]))
    finite_q_median_norm = float(np.median(finite_q_norms))
    zero_q_relative = zero_q_norm / max(finite_q_median_norm, 1e-30)

    q_step_norms = np.linalg.norm(np.diff(p_base, axis=0), axis=1)
    p_q_norms = np.linalg.norm(p_base[:-1], axis=1)
    smoothness = float(np.median(q_step_norms / np.maximum(p_q_norms, 1e-30)))

    doping_change = _relative_norm_difference(p_base, p_doped)
    temperature_change = _relative_norm_difference(p_base, p_hot)

    peak_index = np.unravel_index(int(np.argmax(np.abs(p_base))), p_base.shape)
    print(f"Hamiltonian: {args.hamiltonian.resolve()}")
    print(f"RPA grid: nk={args.num_k}, nq={args.num_q}, nw={args.num_frequency}")
    print(f"Baseline mu={args.fermi_level:g} eV, T={args.temperature:g} K")
    print(f"Max |P| at q/pi={q_points[peak_index[0]] / np.pi:.4g}, omega={frequencies[peak_index[1]]:.4g} eV")
    print(f"||P(q=0)|| / median ||P(q!=0)|| = {zero_q_relative:.6e}")
    print(f"Median relative q-step change = {smoothness:.6e}")
    print(f"Relative change when mu -> {args.doped_fermi_level:g} eV: {doping_change:.6e}")
    print(f"Relative change when T -> {args.hot_temperature:g} K: {temperature_change:.6e}")

    _make_plots(
        output_prefix=args.output_prefix,
        q_points=q_points,
        frequencies=frequencies,
        p_base=p_base,
        p_doped=p_doped,
        p_hot=p_hot,
    )


if __name__ == "__main__":
    main()

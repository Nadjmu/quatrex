"""Validate the RPA electronic-structure input by plotting Bloch bands.

This script uses the same translation-resolved Hamiltonian convention as the
RPA implementation: H(k) = sum_R H_R exp(i k R a). It is meant as a lightweight
presentation/validation artifact for checking that the band structure entering
the RPA polarization is sensible.
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
from scipy.optimize import linear_sum_assignment


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

build_bloch_hamiltonian = rpa_compute.build_bloch_hamiltonian
diagonalize_bloch_hamiltonian = rpa_compute.diagonalize_bloch_hamiltonian
infer_periodic_axis = rpa_compute.infer_periodic_axis
load_translation_blocks = rpa_compute.load_translation_blocks


def _read_lattice_vectors(lattice_xyz: Path) -> np.ndarray:
    """Read the 3x3 lattice matrix from an extxyz-style Lattice attribute."""

    with lattice_xyz.open("r", encoding="utf-8") as handle:
        handle.readline()
        metadata = handle.readline()

    marker = 'Lattice="'
    start = metadata.find(marker)
    if start < 0:
        raise ValueError(f"Could not find Lattice attribute in {lattice_xyz}.")
    start += len(marker)
    end = metadata.find('"', start)
    if end < 0:
        raise ValueError(f"Could not parse Lattice attribute in {lattice_xyz}.")

    values = np.fromstring(metadata[start:end], sep=" ", dtype=np.float64)
    if values.size != 9:
        raise ValueError(
            f"Expected 9 lattice values in {lattice_xyz}, found {values.size}."
        )
    return values.reshape(3, 3)


def _reciprocal_lattice_vectors(lattice_vectors: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(lattice_vectors).T


def _build_hexagonal_path(
    *,
    path_name: str,
    points_per_segment: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    if points_per_segment < 2:
        raise ValueError("--path-points must be at least 2.")

    if path_name == "gamma-k-m-gamma":
        labels = ["G", "K", "M", "G"]
        point_lookup = {
            "G": np.array([0.0, 0.0], dtype=np.float64),
            "K": np.array([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
            "M": np.array([0.5, 0.0], dtype=np.float64),
        }
    elif path_name == "gamma-m-k-gamma":
        labels = ["G", "M", "K", "G"]
        point_lookup = {
            "G": np.array([0.0, 0.0], dtype=np.float64),
            "K": np.array([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
            "M": np.array([0.5, 0.0], dtype=np.float64),
        }
    elif path_name == "mos2-rect-gamma-k-m-gamma":
        labels = ["G", "K", "M", "G"]
        point_lookup = {
            "G": np.array([0.0, 0.0], dtype=np.float64),
            "K": np.array([1.0 / 3.0, 0.0], dtype=np.float64),
            "M": np.array([0.5, 0.5], dtype=np.float64),
        }
    elif path_name == "mos2-rect-k-gamma-m-k":
        labels = ["K", "G", "M", "K"]
        point_lookup = {
            "K": np.array([1.0 / 3.0, 0.0], dtype=np.float64),
            "G": np.array([0.0, 0.0], dtype=np.float64),
            "M": np.array([0.5, 0.5], dtype=np.float64),
        }
    elif path_name == "rect-gamma-x-s-y-gamma":
        labels = ["G", "X", "S", "Y", "G"]
        point_lookup = {
            "G": np.array([0.0, 0.0], dtype=np.float64),
            "X": np.array([0.5, 0.0], dtype=np.float64),
            "S": np.array([0.5, 0.5], dtype=np.float64),
            "Y": np.array([0.0, 0.5], dtype=np.float64),
        }
    else:
        raise ValueError(f"Unsupported path: {path_name}")
    high_symmetry_points = [(label, point_lookup[label]) for label in labels]

    path_parts = []
    tick_indices = [0]
    for segment_index, ((_, start), (_, end)) in enumerate(
        zip(high_symmetry_points[:-1], high_symmetry_points[1:])
    ):
        segment = np.linspace(start, end, points_per_segment, endpoint=True)
        if segment_index > 0:
            segment = segment[1:]
        path_parts.append(segment)
        tick_indices.append(tick_indices[-1] + segment.shape[0] - 1)

    return np.vstack(path_parts), tick_indices, labels


def _path_distances(
    reduced_k_points: np.ndarray,
    *,
    lattice_xyz: Path | None,
) -> np.ndarray:
    if lattice_xyz is None:
        cartesian_k_points = reduced_k_points
    else:
        reciprocal_vectors = _reciprocal_lattice_vectors(_read_lattice_vectors(lattice_xyz))
        cartesian_k_points = reduced_k_points @ reciprocal_vectors[:2, :2]

    segment_lengths = np.linalg.norm(np.diff(cartesian_k_points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _build_bloch_hamiltonian_reduced_2d(
    translation_blocks: dict[tuple[int, int, int], np.ndarray],
    reduced_k_points: np.ndarray,
) -> np.ndarray:
    first_block = next(iter(translation_blocks.values()))
    norb = first_block.shape[0]
    bloch_hamiltonian = np.zeros(
        (reduced_k_points.shape[0], norb, norb), dtype=np.complex128
    )

    for translation, block in translation_blocks.items():
        in_plane_translation = np.array(translation[:2], dtype=np.float64)
        phase_argument = 2.0 * np.pi * reduced_k_points @ in_plane_translation
        phase = np.exp(1j * phase_argument)
        bloch_hamiltonian += phase[:, np.newaxis, np.newaxis] * block[np.newaxis, :, :]
    return bloch_hamiltonian


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Bloch bands from a translation-resolved Hamiltonian."
    )
    parser.add_argument(
        "--hamiltonian",
        type=Path,
        default=EXAMPLE_ROOT / "inputs" / "hamiltonian.mat",
        help="Path to the translation-resolved Hamiltonian .mat file.",
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Periodic axis used in the Bloch phase. Defaults to inferred axis.",
    )
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=1.0,
        help="Lattice constant in the Bloch phase convention.",
    )
    parser.add_argument(
        "--num-k",
        type=int,
        default=301,
        help="Number of k-points in the validation path.",
    )
    parser.add_argument(
        "--fermi-level",
        type=float,
        default=None,
        help="Optional Fermi level to draw and use for gap reporting.",
    )
    parser.add_argument(
        "--shift-fermi",
        action="store_true",
        help="Plot energies relative to --reference-energy or EF. Requires one of them.",
    )
    parser.add_argument(
        "--reference-energy",
        type=float,
        default=None,
        help="Optional energy used only as the plotting zero when --shift-fermi is set.",
    )
    parser.add_argument(
        "--reference-label",
        default=r"E_\mathrm{ref}",
        help="Math-text label for --reference-energy in the y-axis.",
    )
    parser.add_argument(
        "--energy-window",
        type=float,
        nargs=2,
        metavar=("EMIN", "EMAX"),
        default=None,
        help="Optional y-axis energy window in eV.",
    )
    parser.add_argument(
        "--path",
        choices=(
            "1d",
            "gamma-k-m-gamma",
            "gamma-m-k-gamma",
            "mos2-rect-gamma-k-m-gamma",
            "mos2-rect-k-gamma-m-k",
            "rect-gamma-x-s-y-gamma",
        ),
        default="1d",
        help="Band path to plot in reduced reciprocal coordinates.",
    )
    parser.add_argument(
        "--path-points",
        type=int,
        default=121,
        help="Points per segment for --path gamma-k-m-gamma.",
    )
    parser.add_argument(
        "--lattice-xyz",
        type=Path,
        default=None,
        help="Optional extxyz file used to scale the 2D path distance axis.",
    )
    parser.add_argument(
        "--frontier-bands",
        type=int,
        default=None,
        help="Plot only this many valence and conduction bands nearest the reference.",
    )
    parser.add_argument(
        "--track-bands",
        action="store_true",
        help="Connect bands by adjacent-k eigenvector overlap instead of sorted energy index.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ANALYSIS_ROOT
        / "validation_outputs"
        / "cnt_hbn_general"
        / "carbon_nanotube"
        / "band_structure"
        / "rpa_bandstructure_validation.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        help="Optional NPZ path for k-points and eigenvalues.",
    )
    return parser.parse_args()


def _gap_summary(eigenvalues: np.ndarray, fermi_level: float | None) -> str:
    if fermi_level is None:
        occupied = eigenvalues[eigenvalues <= 0.0]
        unoccupied = eigenvalues[eigenvalues > 0.0]
        reference = 0.0
    else:
        occupied = eigenvalues[eigenvalues <= fermi_level]
        unoccupied = eigenvalues[eigenvalues > fermi_level]
        reference = fermi_level

    if occupied.size == 0 or unoccupied.size == 0:
        return f"No gap summary available around reference energy {reference:.6g} eV."

    valence_max = float(np.max(occupied))
    conduction_min = float(np.min(unoccupied))
    gap = conduction_min - valence_max
    return (
        f"Reference energy: {reference:.6g} eV\n"
        f"Valence maximum: {valence_max:.6g} eV\n"
        f"Conduction minimum: {conduction_min:.6g} eV\n"
        f"Gap around reference: {gap:.6g} eV"
    )


def _frontier_band_indices(
    eigenvalues: np.ndarray,
    *,
    reference_energy: float,
    bands_per_side: int,
) -> np.ndarray:
    if bands_per_side < 1:
        raise ValueError("--frontier-bands must be positive when provided.")

    valence_indices = [
        band_index
        for band_index in range(eigenvalues.shape[1])
        if np.max(eigenvalues[:, band_index]) <= reference_energy
    ]
    conduction_indices = [
        band_index
        for band_index in range(eigenvalues.shape[1])
        if np.min(eigenvalues[:, band_index]) > reference_energy
    ]
    selected = valence_indices[-bands_per_side:] + conduction_indices[:bands_per_side]
    if not selected:
        raise ValueError("No frontier bands found around the reference energy.")
    return np.array(selected, dtype=np.int64)


def _track_bands_by_overlap(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder bands so adjacent k-points maximize eigenvector overlap."""

    tracked_eigenvalues = np.empty_like(eigenvalues)
    tracked_eigenvectors = np.empty_like(eigenvectors)
    tracked_eigenvalues[0] = eigenvalues[0]
    tracked_eigenvectors[0] = eigenvectors[0]

    previous_vectors = tracked_eigenvectors[0]
    for k_index in range(1, eigenvalues.shape[0]):
        current_vectors = eigenvectors[k_index]
        overlap = np.abs(previous_vectors.conj().T @ current_vectors)
        row_indices, column_indices = linear_sum_assignment(-overlap)
        order = column_indices[np.argsort(row_indices)]
        tracked_eigenvalues[k_index] = eigenvalues[k_index, order]
        tracked_eigenvectors[k_index] = current_vectors[:, order]
        previous_vectors = tracked_eigenvectors[k_index]

    return tracked_eigenvalues, tracked_eigenvectors


def main() -> None:
    args = _parse_args()
    if args.shift_fermi and args.fermi_level is None and args.reference_energy is None:
        raise ValueError("--shift-fermi requires --fermi-level or --reference-energy.")

    blocks = load_translation_blocks(args.hamiltonian)
    axis = None
    tick_positions = None
    tick_labels = None
    reduced_k_points = None

    if args.path == "1d":
        axis = args.axis if args.axis is not None else infer_periodic_axis(blocks)
        k_points = np.linspace(-np.pi, np.pi, args.num_k, endpoint=True)
        plot_k_points = k_points / np.pi
        bloch_hamiltonian = build_bloch_hamiltonian(
            blocks,
            k_points,
            periodic_axis=axis,
            lattice_constant=args.lattice_constant,
        )
    else:
        reduced_k_points, tick_indices, tick_labels = _build_hexagonal_path(
            path_name=args.path,
            points_per_segment=args.path_points
        )
        k_points = _path_distances(reduced_k_points, lattice_xyz=args.lattice_xyz)
        plot_k_points = k_points
        tick_positions = k_points[tick_indices]
        bloch_hamiltonian = _build_bloch_hamiltonian_reduced_2d(
            blocks,
            reduced_k_points,
        )

    hermiticity_error = float(
        np.max(np.abs(bloch_hamiltonian - bloch_hamiltonian.conj().swapaxes(-1, -2)))
    )
    bands = diagonalize_bloch_hamiltonian(bloch_hamiltonian, k_points)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    eigenvalues_for_plot = bands.eigenvalues
    if args.track_bands:
        eigenvalues_for_plot, _ = _track_bands_by_overlap(
            bands.eigenvalues,
            bands.eigenvectors,
        )

    plot_eigenvalues = eigenvalues_for_plot
    energy_label = "Energy (eV)"
    reference_energy = (
        args.reference_energy if args.reference_energy is not None else args.fermi_level
    )
    fermi_line = args.fermi_level
    if args.shift_fermi:
        plot_eigenvalues = bands.eigenvalues - reference_energy
        if args.reference_energy is None:
            energy_label = r"$E - E_F$ (eV)"
        else:
            energy_label = rf"$E - {args.reference_label}$ (eV)"
        fermi_line = 0.0

    band_indices = np.arange(plot_eigenvalues.shape[1])
    if args.frontier_bands is not None:
        if reference_energy is None:
            raise ValueError("--frontier-bands requires --fermi-level or --reference-energy.")
        band_indices = _frontier_band_indices(
            bands.eigenvalues,
            reference_energy=reference_energy,
            bands_per_side=args.frontier_bands,
        )

    for band in plot_eigenvalues[:, band_indices].T:
        ax.plot(plot_k_points, band, color="#1f4e79", linewidth=0.8)

    if fermi_line is not None:
        ax.axhline(fermi_line, color="#b22222", linewidth=1.0, linestyle="--")

    if args.path == "1d":
        ax.set_xlabel(r"$k / \pi$")
    else:
        ax.set_xlabel("Wave-vector path")
        if tick_positions is not None and tick_labels is not None:
            display_labels = [
                r"$\Gamma$" if label == "G" else label for label in tick_labels
            ]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(display_labels)
            for tick_position in tick_positions:
                ax.axvline(tick_position, color="#666666", alpha=0.25, linewidth=0.7)
    ax.set_ylabel(energy_label)
    ax.set_xlim(float(plot_k_points[0]), float(plot_k_points[-1]))
    if args.energy_window is not None:
        ax.set_ylim(*args.energy_window)
    ax.set_title(args.title or f"RPA input band structure: {args.hamiltonian}")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    if args.save_data is not None:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_data,
            k_points=k_points,
            eigenvalues=bands.eigenvalues,
            periodic_axis=np.array(-1 if axis is None else axis),
            reduced_k_points=np.array([])
            if reduced_k_points is None
            else reduced_k_points,
            tick_positions=np.array([])
            if tick_positions is None
            else tick_positions,
            tick_labels=np.array([])
            if tick_labels is None
            else np.array(tick_labels),
            plotted_band_indices=band_indices,
            track_bands=np.array(args.track_bands),
            plotted_eigenvalues=plot_eigenvalues,
            hermiticity_error=np.array(hermiticity_error),
        )

    print(f"Hamiltonian: {args.hamiltonian.resolve()}")
    print(f"Translation blocks: {len(blocks)}")
    print(f"Band path: {args.path}")
    if axis is not None:
        print(f"Periodic axis: {axis}")
        print(f"k-points: {args.num_k}")
    else:
        print(f"k-points: {k_points.size}")
        if tick_labels is not None:
            print(f"Path labels: {'-'.join(tick_labels)}")
    print(f"Max Hermiticity error in H(k): {hermiticity_error:.6e}")
    print(f"Track bands by overlap: {args.track_bands}")
    print(_gap_summary(bands.eigenvalues, reference_energy))
    print(f"Wrote plot: {args.output.resolve()}")
    if args.save_data is not None:
        print(f"Wrote data: {args.save_data.resolve()}")


if __name__ == "__main__":
    main()

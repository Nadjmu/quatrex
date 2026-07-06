"""Elementwise verification for RPA implementation variants.

The serial, frequency-chunked, and q-chunked calculations should produce the
same complex matrix-valued polarization P(q, w) element by element. This script
computes those variants from the same Bloch band structure and reports strong
array-level diagnostics suitable for the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quatrex.core.config import parse_config
from quatrex.coulomb_screening.dielectric_screening.equilibrium_screening import (
    EquilibriumScreening,
)
from quatrex.coulomb_screening.dielectric_screening.rpa_compute import (
    BrillouinZoneMesh,
    ScreeningChannels,
    build_uniform_brillouin_zone_mesh,
    compute_rpa_polarization_matrix_from_bands,
)


W90_ROOT = Path("/home/sem26f25/quatrex/examples/w90")

MATERIALS = {
    "cnt": {
        "label": "CNT",
        "config": W90_ROOT
        / "carbon-nanotube"
        / "gw-unit-cell"
        / "quatrex_config_rpa_smooth.toml",
    },
    "cnt-benchmark": {
        "label": "CNT benchmark grid",
        "config": W90_ROOT
        / "data_analysis"
        / "benchmarks"
        / "equilibrium_polarization"
        / "configs"
        / "cnt_rpa.toml",
    },
    "hbn": {
        "label": "Bilayer hBN",
        "config": W90_ROOT
        / "data_analysis"
        / "comparison_configs"
        / "hbn"
        / "quatrex_config_hbn_rpa_export.toml",
    },
    "hbn-benchmark": {
        "label": "Bilayer hBN benchmark grid",
        "config": W90_ROOT
        / "data_analysis"
        / "benchmarks"
        / "equilibrium_polarization"
        / "configs"
        / "hbn_rpa.toml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("material", choices=sorted(MATERIALS))
    parser.add_argument(
        "--config",
        type=Path,
        help="Override the material config path.",
    )
    parser.add_argument(
        "--frequency-chunks",
        type=int,
        default=4,
        help="Number of frequency chunks for the balanced variant.",
    )
    parser.add_argument(
        "--q-chunks",
        type=int,
        default=4,
        help="Number of q chunks for the speed-focused variant.",
    )
    parser.add_argument("--num-k-points", type=int, help="Override k mesh size.")
    parser.add_argument("--num-q-points", type=int, help="Override q mesh size.")
    parser.add_argument(
        "--num-frequencies",
        type=int,
        help="Override frequency-grid size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=W90_ROOT
        / "data_analysis"
        / "validation_outputs"
        / "rpa_implementation_consistency",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Also save the reference and variant polarization arrays.",
    )
    return parser.parse_args()


def compute_reference(config, mesh, bands) -> np.ndarray:
    screening = config.coulomb_screening
    return compute_rpa_polarization_matrix_from_bands(
        bands,
        mesh.q_points,
        mesh.frequencies,
        chemical_potential=screening.chemical_potential,
        temperature=screening.temperature,
        state_multiplicity=screening.spin_degeneracy * screening.valley_degeneracy,
        broadening=screening.broadening,
        frequency_axis="real",
    )


def compute_frequency_chunked(config, mesh, bands, chunks: int) -> np.ndarray:
    screening = config.coulomb_screening
    pieces = []
    frequency_splits = np.array_split(np.arange(mesh.frequencies.size), chunks)
    for index, frequency_indices in enumerate(frequency_splits, start=1):
        if frequency_indices.size == 0:
            continue
        print(
            f"  balanced RPA frequency chunk {index}/{len(frequency_splits)} "
            f"({frequency_indices.size} frequencies)",
            flush=True,
        )
        pieces.append(
            compute_rpa_polarization_matrix_from_bands(
                bands,
                mesh.q_points,
                mesh.frequencies[frequency_indices],
                chemical_potential=screening.chemical_potential,
                temperature=screening.temperature,
                state_multiplicity=(
                    screening.spin_degeneracy * screening.valley_degeneracy
                ),
                broadening=screening.broadening,
                frequency_axis="real",
            )
        )
    return np.concatenate(pieces, axis=1)


def compute_q_chunked(config, mesh, bands, chunks: int) -> np.ndarray:
    screening = config.coulomb_screening
    pieces = []
    q_splits = np.array_split(np.arange(mesh.q_points.size), chunks)
    for index, q_indices in enumerate(q_splits, start=1):
        if q_indices.size == 0:
            continue
        print(
            f"  speed-focused RPA q chunk {index}/{len(q_splits)} "
            f"({q_indices.size} q-points)",
            flush=True,
        )
        pieces.append(
            compute_rpa_polarization_matrix_from_bands(
                bands,
                mesh.q_points[q_indices],
                mesh.frequencies,
                chemical_potential=screening.chemical_potential,
                temperature=screening.temperature,
                state_multiplicity=(
                    screening.spin_degeneracy * screening.valley_degeneracy
                ),
                broadening=screening.broadening,
                frequency_axis="real",
            )
        )
    return np.concatenate(pieces, axis=0)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool]:
    difference = candidate - reference
    reference_norm = np.linalg.norm(reference.ravel())
    difference_norm = np.linalg.norm(difference.ravel())
    return {
        "max_abs_delta": float(np.max(np.abs(difference))),
        "mean_abs_delta": float(np.mean(np.abs(difference))),
        "relative_frobenius_error": float(difference_norm / reference_norm),
        "bitwise_equal": bool(np.array_equal(candidate, reference)),
        "allclose_rtol_1e-12_atol_1e-12": bool(
            np.allclose(candidate, reference, rtol=1e-12, atol=1e-12)
        ),
    }


def main() -> None:
    args = parse_args()
    material = MATERIALS[args.material]
    config_path = args.config or material["config"]
    config = parse_config(config_path)
    screening = config.coulomb_screening
    num_k_points = args.num_k_points or screening.num_k_points
    num_q_points = args.num_q_points or screening.num_q_points
    num_frequencies = args.num_frequencies or screening.num_frequencies
    mesh = build_uniform_brillouin_zone_mesh(
        num_k_points=num_k_points,
        num_q_points=num_q_points,
        num_frequencies=num_frequencies,
        max_frequency=screening.max_frequency,
        lattice_constant=screening.lattice_constant,
        include_zero_q=screening.include_zero_q,
    )

    solver = EquilibriumScreening(
        channels=ScreeningChannels(
            spin_degeneracy=screening.spin_degeneracy,
            valley_degeneracy=screening.valley_degeneracy,
        ),
        matrix_polarization=True,
        frequency_axis="real",
    )
    inputs = solver.load_inputs_from_config(
        config,
        hamiltonian_matrix_name=screening.hamiltonian_matrix_name,
        coulomb_matrix_name=screening.coulomb_matrix_name,
    )
    bands = solver.polarization_solver.load_bloch_band_structure_from_blocks(
        inputs.hamiltonian_blocks,
        BrillouinZoneMesh(
            k_points=mesh.k_points,
            q_points=np.asarray([0.0]),
            frequencies=np.asarray([0.0]),
        ),
        periodic_axis=screening.periodic_axis,
        lattice_constant=screening.lattice_constant,
    )

    print(
        f"{material['label']}: computing full P(q,w) with "
        f"nk={mesh.k_points.size}, nq={mesh.q_points.size}, "
        f"nw={mesh.frequencies.size}, norb={bands.eigenvectors.shape[1]}"
    )
    print("  serial RPA reference", flush=True)
    reference = compute_reference(config, mesh, bands)
    print("  balanced RPA variant", flush=True)
    balanced = compute_frequency_chunked(
        config, mesh, bands, chunks=args.frequency_chunks
    )
    balanced_comparison = compare(reference, balanced)
    balanced_to_save = balanced if args.save_arrays else None
    if not args.save_arrays:
        del balanced

    print("  speed-focused RPA variant", flush=True)
    speed_focused = compute_q_chunked(config, mesh, bands, chunks=args.q_chunks)
    speed_focused_comparison = compare(reference, speed_focused)
    speed_focused_to_save = speed_focused if args.save_arrays else None

    result = {
        "material": material["label"],
        "config": str(config_path),
        "polarization_shape": list(reference.shape),
        "mesh": {
            "num_k_points": int(num_k_points),
            "num_q_points": int(num_q_points),
            "num_frequencies": int(num_frequencies),
        },
        "frequency_chunks": args.frequency_chunks,
        "q_chunks": args.q_chunks,
        "comparisons": {
            "balanced_frequency_chunked_vs_serial": balanced_comparison,
            "speed_focused_q_chunked_vs_serial": speed_focused_comparison,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.material}_elementwise_consistency.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    if args.save_arrays:
        np.savez_compressed(
            args.output_dir / f"{args.material}_elementwise_consistency_arrays.npz",
            serial=reference,
            balanced_frequency_chunked=balanced_to_save,
            speed_focused_q_chunked=speed_focused_to_save,
        )

    print(json.dumps(result, indent=2))
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()

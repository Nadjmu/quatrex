"""Compare periodic hBN RPA against an equilibrium Green-function bubble.

Both responses use the same Bloch Hamiltonian, occupations, momentum mesh, density
vertices, and spin degeneracy.  The Green-function response is evaluated by an
independent numerical energy convolution, making this a prefactor/normalization
test rather than another finite-device comparison.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

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


CONFIG = Path(
    "/home/sem26f25/quatrex/examples/w90/data_analysis/comparison_configs/hbn/"
    "quatrex_config_hbn_rpa_export.toml"
)
OUTPUT_DIR = Path("outputs/hbn_periodic_rpa_vs_gf_bubble")

# Use representative nonzero q values. The density response at exactly q=0
# vanishes for this orthogonal density vertex.
Q_FRACTIONS = (0.25, 0.50)
ELECTRON_BROADENING_EV = 0.025
ENERGY_MIN_EV = -30.0
ENERGY_MAX_EV = 42.0


def fermi_dirac(energies: np.ndarray, chemical_potential: float, temperature: float):
    kb_ev = 8.617333262145e-5
    argument = np.clip(
        (energies - chemical_potential) / (kb_ev * temperature), -700.0, 700.0
    )
    return 1.0 / (np.exp(argument) + 1.0)


def orbital_green_functions(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    energy_grid: np.ndarray,
    occupations: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal orbital G-retarded and G-lesser."""

    orbital_weights = np.abs(eigenvectors) ** 2
    denominator = (
        energy_grid[np.newaxis, np.newaxis, :]
        - eigenvalues[:, :, np.newaxis]
        + 1j * eta
    )
    g_retarded = np.einsum(
        "kib,kbe->kie", orbital_weights, 1.0 / denominator, optimize=True
    )
    spectral_by_band = 2.0 * eta / np.abs(denominator) ** 2
    g_lesser = 1j * np.einsum(
        "kib,kb,kbe->kie",
        orbital_weights,
        occupations,
        spectral_by_band,
        optimize=True,
    )
    return g_retarded, g_lesser


def periodic_gf_bubble_trace(
    g_retarded: np.ndarray,
    g_lesser: np.ndarray,
    q_shifts: np.ndarray,
    energy_step: float,
    frequency_indices: np.ndarray,
    state_multiplicity: float,
) -> np.ndarray:
    """Evaluate -i Tr[Gr(k+q)G<(k) + G<(k+q)Ga(k)] by convolution."""

    nk, norb, ne = g_retarded.shape
    center = ne - 1
    result = np.zeros((q_shifts.size, frequency_indices.size), dtype=np.complex128)

    for q_index, q_shift in enumerate(q_shifts):
        shifted_retarded = np.roll(g_retarded, -q_shift, axis=0)
        shifted_lesser = np.roll(g_lesser, -q_shift, axis=0)
        bubble = np.zeros(2 * ne - 1, dtype=np.complex128)

        for k_index in range(nk):
            for orbital in range(norb):
                # At positive transfer l, these convolutions evaluate:
                # sum_E G<(k,E) Gr(k+q,E+l) and
                # sum_E Ga(k,E) G<(k+q,E+l).
                bubble += fftconvolve(
                    g_lesser[k_index, orbital, ::-1],
                    shifted_retarded[k_index, orbital],
                    mode="full",
                )
                bubble += fftconvolve(
                    g_retarded[k_index, orbital, ::-1].conj(),
                    shifted_lesser[k_index, orbital],
                    mode="full",
                )

        result[q_index] = (
            -1j
            * state_multiplicity
            * energy_step
            / (2.0 * np.pi * nk)
            * bubble[center + frequency_indices]
        )

    return result


def main() -> None:
    config = parse_config(CONFIG)
    screening = config.coulomb_screening
    mesh = build_uniform_brillouin_zone_mesh(
        num_k_points=screening.num_k_points,
        num_q_points=screening.num_k_points,
        num_frequencies=screening.num_frequencies,
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

    dk = mesh.k_points[1] - mesh.k_points[0]
    desired_q = np.asarray(Q_FRACTIONS) * np.pi / screening.lattice_constant
    q_shifts = np.rint(desired_q / dk).astype(int)
    q_points = q_shifts * dk

    rpa_matrix = compute_rpa_polarization_matrix_from_bands(
        bands,
        q_points,
        mesh.frequencies,
        chemical_potential=screening.chemical_potential,
        temperature=screening.temperature,
        state_multiplicity=(
            screening.spin_degeneracy * screening.valley_degeneracy
        ),
        broadening=2.0 * ELECTRON_BROADENING_EV,
        frequency_axis="real",
    )
    rpa_trace = np.trace(rpa_matrix, axis1=2, axis2=3)

    frequency_step = mesh.frequencies[1] - mesh.frequencies[0]
    energy_grid = np.arange(
        ENERGY_MIN_EV,
        ENERGY_MAX_EV + 0.5 * frequency_step,
        frequency_step,
    )
    frequency_indices = np.rint(mesh.frequencies / frequency_step).astype(int)
    occupations = fermi_dirac(
        bands.eigenvalues, screening.chemical_potential, screening.temperature
    )
    g_retarded, g_lesser = orbital_green_functions(
        bands.eigenvalues,
        bands.eigenvectors,
        energy_grid,
        occupations,
        ELECTRON_BROADENING_EV,
    )
    gf_trace = periodic_gf_bubble_trace(
        g_retarded,
        g_lesser,
        q_shifts,
        frequency_step,
        frequency_indices,
        screening.spin_degeneracy * screening.valley_degeneracy,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_DIR / "hbn_periodic_rpa_vs_gf_bubble.npz",
        frequencies_eV=mesh.frequencies,
        q_points=q_points,
        q_over_pi=q_points * screening.lattice_constant / np.pi,
        rpa_trace=rpa_trace,
        gf_trace=gf_trace,
        electron_broadening_eV=ELECTRON_BROADENING_EV,
    )

    figure, axes = plt.subplots(len(q_points), 2, figsize=(13, 4.5 * len(q_points)))
    for row, q_fraction in enumerate(q_points * screening.lattice_constant / np.pi):
        for column, component in enumerate(("real", "imag")):
            axis = axes[row, column]
            rpa_values = getattr(rpa_trace[row], component)
            gf_values = getattr(gf_trace[row], component)
            axis.plot(mesh.frequencies, rpa_values, label="Band-sum RPA", linewidth=2)
            axis.plot(
                mesh.frequencies,
                gf_values,
                "--",
                label="Periodic GF bubble",
                linewidth=1.8,
            )
            axis.axhline(0.0, color="0.25", linewidth=0.8)
            axis.set_title(f"{component.capitalize()} response, q/pi={q_fraction:.2f}")
            axis.set_xlabel("Energy transfer (eV)")
            axis.set_ylabel("Polarization trace")
            axis.legend()
            axis.grid(alpha=0.2)
    figure.suptitle("Bilayer hBN: Periodic RPA versus Periodic Green-Function Bubble")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "hbn_periodic_rpa_vs_gf_bubble.png", dpi=220, bbox_inches="tight"
    )

    summary_figure, summary_axes = plt.subplots(
        1, len(q_points), figsize=(13, 4.8), sharey=True
    )
    if len(q_points) == 1:
        summary_axes = np.asarray([summary_axes])

    for index, q_fraction in enumerate(q_points * screening.lattice_constant / np.pi):
        scale = np.vdot(gf_trace[index], rpa_trace[index]).real / np.vdot(
            gf_trace[index], gf_trace[index]
        ).real
        relative_error = np.linalg.norm(gf_trace[index] - rpa_trace[index]) / np.linalg.norm(
            rpa_trace[index]
        )
        scaled_error = np.linalg.norm(
            scale * gf_trace[index] - rpa_trace[index]
        ) / np.linalg.norm(rpa_trace[index])
        print(
            f"q/pi={q_fraction:.2f}: direct relative error={relative_error:.6e}, "
            f"best-fit scale={scale:.8f}, scaled residual={scaled_error:.6e}"
        )

        imag_relative_error = np.linalg.norm(
            gf_trace[index].imag - rpa_trace[index].imag
        ) / np.linalg.norm(rpa_trace[index].imag)
        imag_scale = np.vdot(
            gf_trace[index].imag, rpa_trace[index].imag
        ).real / np.vdot(gf_trace[index].imag, gf_trace[index].imag).real
        imag_correlation = np.corrcoef(
            gf_trace[index].imag, rpa_trace[index].imag
        )[0, 1]
        print(
            f"q/pi={q_fraction:.2f} imaginary response: "
            f"relative error={imag_relative_error:.6e}, "
            f"best-fit scale={imag_scale:.8f}, correlation={imag_correlation:.8f}"
        )

        axis = summary_axes[index]
        axis.plot(
            mesh.frequencies,
            rpa_trace[index].imag,
            label="Band-sum RPA",
            linewidth=2.2,
        )
        axis.plot(
            mesh.frequencies,
            gf_trace[index].imag,
            "--",
            label="Periodic GF bubble",
            linewidth=1.8,
        )
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.set_title(
            f"q/pi = {q_fraction:.2f}\n"
            f"scale = {imag_scale:.4f}, error = {100 * imag_relative_error:.2f}%"
        )
        axis.set_xlabel("Energy transfer (eV)")
        axis.grid(alpha=0.2)
    summary_axes[0].set_ylabel("Im polarization trace")
    summary_axes[0].legend()
    summary_figure.suptitle(
        "Bilayer hBN Absolute-Magnitude Validation: Periodic RPA vs GF Bubble"
    )
    summary_figure.tight_layout()
    summary_figure.savefig(
        OUTPUT_DIR / "hbn_periodic_rpa_vs_gf_bubble_imag_validation.png",
        dpi=220,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()

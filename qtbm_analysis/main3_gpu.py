"""
GPU counterpart of main3.py: export of QTBM system matrices and conditioning
analysis over an energy sweep.

Differences from the CPU version
--------------------------------
M(E) and Sigma(E) are kept sparse on the device through
``cupyx.scipy.sparse`` and are transferred to host memory only when written to
disk, which is what makes the large systems tractable. Everything else,
including the definitions, the energy grid and the output layout, is identical
to main3.py; see its module docstring for the full description.

The dense generalized eigenvalue problem is solved on the host, because
``scipy.linalg.eigh`` has no CuPy equivalent that accepts a matrix pencil. That
step applies only to the small bare pencil (H, S), never to M(E).

Input
-----
QTBM example directories listed in EXAMPLES, read through
``export_qtbm_systems.export_system``.

Output
------
Written to FOLDER/matrices/<example>/, identical to main3.py:

    energies.npy, band_edge.npy, spectrum_bare.npy, condition_bare.npy,
    condition_full_svd.npy, max_singular_values.npy, min_singular_values.npy,
    M_E_<idx>.npz, Sigma_E_<idx>.npz, rhs_E_<idx>.npy

No figures are produced; see plotting/plot_qtbm_spectra.py.

Usage
-----
    python main3_gpu.py
    python plotting/plot_qtbm_spectra.py /scratch/yimili/matrices/dev_12_sorted_BENCH
"""

from pathlib import Path

import cupy as cp
import numpy as np
from scipy.linalg import eigh

from export_qtbm_systems import export_system, _host_array, _save_csr_npz

FOLDER = Path("/scratch/yimili")

EXAMPLES = [
    Path("/scratch/yimili/examples/dev_12_sorted_BENCH/qtbm"),
    Path("/scratch/yimili/examples/WS2-hBN-25_benchmark-QUATREX-DZ/qtbm"),
]

OFFSET = 1.0
OFFSET_LOW = 0.005
POINTS = 401

INSPECTION_INDICES = [0] + list(range(10, 400, 20))

# Dense analyses, O(n^3) each. Feasible only for the small examples.
RUN_SPECTRUM = False
RUN_CONDITION_BARE = False
RUN_CONDITION_FULL_SVD = False

# Sparse export of M(E), Sigma(E) and the right-hand side.
RUN_EXPORT_MATRICES = True


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_matrices(example):
    """
    Hamiltonian, overlap and configuration of one example, device-resident.

    Both matrices are negated to match the sign convention used by the QTBM
    assembly, so that M(E) = E S - H - Sigma(E) holds with these H and S.
    """
    hamiltonian, config, _ = export_system(
        example=example, mode="hamiltonian",
        energy_index=None, energy=0, k_index=None, k_point=(0, 0, 0))
    overlap, _, _ = export_system(
        example=example, mode="overlap",
        energy_index=None, energy=0, k_index=None, k_point=(0, 0, 0))
    return -hamiltonian, -overlap, config


def band_edge_of(config):
    """
    Conduction band edge in eV, falling back to the left Fermi level when the
    configuration does not set one.
    """
    edge = config.electron.conduction_band_edge
    return config.electron.left_fermi_level if edge is None else edge


def energy_grid(band_edge):
    """Energy sweep around the band edge, POINTS + 1 samples."""
    return np.linspace(band_edge - OFFSET - OFFSET_LOW, band_edge + OFFSET,
                       POINTS + 1)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def bare_spectrum(hamiltonian, overlap, band_edge):
    """
    Eigenvalues of H v = lambda S v, solved densely on the host.

    The pencil form has no CuPy equivalent, so both operands are transferred to
    the host. This is acceptable because it is applied only to the small bare
    problem, never to M(E).

    Returns (eigenvalues, eigenvalue nearest the band edge).
    """
    eigenvalues, _ = eigh(_host_array(hamiltonian.toarray()),
                          _host_array(overlap.toarray()))
    nearest = eigenvalues[np.argmin(np.abs(eigenvalues - band_edge))]
    print(f"  eigenvalue nearest the band edge: {nearest}")
    return eigenvalues, nearest


def sweep_condition_bare(hamiltonian, overlap, energies):
    """kappa_2(H - E S) along the grid, computed on the device."""
    H = hamiltonian.toarray()
    S = overlap.toarray()
    condition_numbers = []
    for energy in energies:
        print(f"  kappa_2(H - E S) at E = {energy:.4f} eV")
        condition_numbers.append(float(cp.linalg.cond(H - energy * S)))
    return np.array(condition_numbers)


def sweep_condition_full_svd(example, energies):
    """
    kappa_2(M(E)), sigma_max(M(E)) and sigma_min(M(E)) along the grid.

    The full SVD is used rather than a condition-number routine so that the two
    extreme singular values are recorded separately: a peak in kappa_2 may
    originate either from sigma_min approaching zero or from sigma_max growing,
    and the ratio does not distinguish them.
    """
    condition_numbers, sigma_max, sigma_min = [], [], []
    for energy in energies:
        print(f"  kappa_2(M(E)) at E = {energy:.4f} eV")
        A, _, _ = export_system(
            example=example, mode="full",
            energy_index=None, energy=cp.array(energy),
            k_index=None, k_point=(0, 0, 0))
        singular_values = cp.linalg.svd(A.toarray(), compute_uv=False)
        singular_values = _host_array(singular_values)
        sigma_max.append(singular_values[0])       # sorted descending
        sigma_min.append(singular_values[-1])
        condition_numbers.append(singular_values[0] / singular_values[-1])
    return (np.array(condition_numbers), np.array(sigma_max),
            np.array(sigma_min))


def export_matrices(example, hamiltonian, overlap, energies, out_dir):
    """
    Write M(E), Sigma(E) and the right-hand side at every inspection index.

    Sigma(E) is recovered as Sigma = (E S - H) - M(E) from the definition of M.
    All arithmetic stays sparse and device-resident; the transfer to host
    memory happens inside the writers.
    """
    for index in INSPECTION_INDICES:
        energy = energies[index]
        print(f"  inspection index {index}: E = {energy} eV")

        M, _, rhs = export_system(
            example=example, mode="full",
            energy_index=None, energy=cp.array(energy),
            k_index=None, k_point=(0, 0, 0))

        M = M.tocsr()
        Sigma = ((energy * overlap - hamiltonian).tocsr() - M).tocsr()

        _save_csr_npz(out_dir / f"M_E_{index}.npz", M)
        _save_csr_npz(out_dir / f"Sigma_E_{index}.npz", Sigma)
        np.save(out_dir / f"rhs_E_{index}.npy", _host_array(rhs))
        print(f"    wrote M_E_{index}.npz, Sigma_E_{index}.npz, "
              f"rhs_E_{index}.npy")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_example(example):
    name = example.parent.name
    print(f"Example: {name}")

    out_dir = FOLDER / "matrices" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    hamiltonian, overlap, config = load_matrices(example)
    band_edge = band_edge_of(config)
    energies = energy_grid(band_edge)
    print(f"  band edge = {band_edge} eV, "
          f"grid resolution = {2 * OFFSET / POINTS:.3e} eV")

    np.save(out_dir / "energies.npy", energies)
    np.save(out_dir / "band_edge.npy", np.array(band_edge))

    if RUN_SPECTRUM:
        eigenvalues, _ = bare_spectrum(hamiltonian, overlap, band_edge)
        np.save(out_dir / "spectrum_bare.npy", eigenvalues)

    if RUN_CONDITION_BARE:
        condition = sweep_condition_bare(hamiltonian, overlap, energies)
        np.save(out_dir / "condition_bare.npy", condition)

    if RUN_CONDITION_FULL_SVD:
        condition, sigma_max, sigma_min = sweep_condition_full_svd(
            example, energies)
        np.save(out_dir / "condition_full_svd.npy", condition)
        np.save(out_dir / "max_singular_values.npy", sigma_max)
        np.save(out_dir / "min_singular_values.npy", sigma_min)

    if RUN_EXPORT_MATRICES:
        export_matrices(example, hamiltonian, overlap, energies, out_dir)

    print(f"  output directory: {out_dir}")
    print(f"  plot with: python plotting/plot_qtbm_spectra.py {out_dir}\n")


def main():
    for example in EXAMPLES:
        process_example(example)


if __name__ == "__main__":
    main()

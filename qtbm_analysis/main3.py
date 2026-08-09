"""
Export of QTBM system matrices and conditioning analysis over an energy sweep.

Input
-----
One or more QTBM example directories, listed in EXAMPLES. Each is read through
``export_qtbm_systems.export_system``, which returns the Hamiltonian H, the
overlap S, the assembled system matrix M(E) and the right-hand side for a given
energy.

Definitions
-----------
The QTBM system matrix at energy E is

    M(E) = E S - H - Sigma(E),

where Sigma(E) collects the contact self-energies of the open boundary
conditions. The bare pencil (H, S) has no self-energy contribution, so the
difference between the conditioning of H - E S and that of M(E) isolates the
effect of the open boundaries.

Algorithm
---------
Per example, controlled by the RUN_* switches below:

1. Load H and S. Both are negated, matching the sign convention of the QTBM
   assembly in export_qtbm_systems.
2. Determine the conduction band edge from the configuration, falling back to
   the left Fermi level when it is not set, and build the energy grid
   [E_edge - offset, E_edge + offset].
3. Solve the generalized eigenvalue problem H v = lambda S v densely and record
   the spectrum. The eigenvalue nearest the band edge governs the conditioning
   of the pencil there.
4. Sweep the grid and record kappa_2(H - E S).
5. Sweep the grid and record kappa_2(M(E)) together with sigma_max and
   sigma_min from the full SVD. The two extreme singular values are stored
   separately because kappa_2 alone does not distinguish a peak caused by
   sigma_min approaching zero from one caused by sigma_max growing.
6. Assemble M(E) and Sigma(E) at the inspection indices and write them to disk
   as sparse triplets, along with the right-hand side.

Steps 3 to 5 are dense and scale as O(n^3); they are disabled by default and
are only feasible for the small examples. Step 6 stays sparse throughout and is
the path used for the large systems.

Output
------
Written to FOLDER/matrices/<example>/:

    energies.npy                 the energy grid, eV
    band_edge.npy                the conduction band edge, eV
    spectrum_bare.npy            eigenvalues of the pencil (H, S)
    condition_bare.npy           kappa_2(H - E S) along the grid
    condition_full_svd.npy       kappa_2(M(E)) along the grid
    max_singular_values.npy      sigma_max(M(E))
    min_singular_values.npy      sigma_min(M(E))
    M_E_<idx>.npz                M(E) at an inspection index, CSR triplet
    Sigma_E_<idx>.npz            Sigma(E) at the same index, CSR triplet
    rhs_E_<idx>.npy              the right-hand side at the same index

No figures are produced; see plotting/plot_qtbm_spectra.py, which reads these
arrays.

Usage
-----
    python main3.py
    python plotting/plot_qtbm_spectra.py /scratch/yimili/matrices/dev_12_sorted_BENCH
"""

from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from export_qtbm_systems import export_system, _host_array, _save_csr_npz

FOLDER = Path("/scratch/yimili")

EXAMPLES = [
    Path("/scratch/yimili/examples/dev_12_sorted_BENCH/qtbm"),
    Path("/scratch/yimili/examples/WS2-hBN-25_benchmark-QUATREX-DZ/qtbm"),
]

# Energy grid: [band_edge - OFFSET - OFFSET_LOW, band_edge + OFFSET], with
# POINTS + 1 samples. OFFSET_LOW shifts the lower endpoint slightly below the
# symmetric window so that the grid does not sample the band edge exactly.
OFFSET = 1.0
OFFSET_LOW = 0.005
POINTS = 401

# Energy indices at which M(E) and Sigma(E) are exported.
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
    Hamiltonian, overlap and configuration of one example.

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
    Conduction band edge in eV.

    Falls back to the left Fermi level when the configuration does not set a
    band edge; for an equilibrium calculation the two coincide.
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
    Eigenvalues of the generalized problem H v = lambda S v, solved densely.

    Returns (eigenvalues, eigenvalue nearest the band edge). The nearest
    eigenvalue is the one that drives kappa_2(H - E S) as E approaches it.
    """
    eigenvalues, _ = eigh(hamiltonian.toarray(), overlap.toarray())
    nearest = eigenvalues[np.argmin(np.abs(eigenvalues - band_edge))]
    print(f"  eigenvalue nearest the band edge: {nearest}")
    return eigenvalues, nearest


def sweep_condition_bare(hamiltonian, overlap, energies):
    """kappa_2(H - E S) along the grid, from a dense SVD at each energy."""
    H = hamiltonian.toarray()
    S = overlap.toarray()
    condition_numbers = []
    for energy in energies:
        print(f"  kappa_2(H - E S) at E = {energy:.4f} eV")
        condition_numbers.append(np.linalg.cond(np.asarray(H - energy * S)))
    return np.array(condition_numbers)


def sweep_condition_full_svd(example, energies):
    """
    kappa_2(M(E)), sigma_max(M(E)) and sigma_min(M(E)) along the grid.

    The full SVD is used rather than np.linalg.cond so that the two extreme
    singular values are available separately: a peak in kappa_2 may originate
    either from sigma_min approaching zero, which is the near-singular case of
    interest, or from sigma_max growing, and the ratio does not distinguish
    them.
    """
    condition_numbers, sigma_max, sigma_min = [], [], []
    for energy in energies:
        print(f"  kappa_2(M(E)) at E = {energy:.4f} eV")
        A, _, _ = export_system(
            example=example, mode="full",
            energy_index=None, energy=np.array(energy),
            k_index=None, k_point=(0, 0, 0))
        singular_values = np.linalg.svd(np.asarray(A.toarray()),
                                        compute_uv=False)
        sigma_max.append(singular_values[0])       # sorted descending
        sigma_min.append(singular_values[-1])
        condition_numbers.append(singular_values[0] / singular_values[-1])
    return (np.array(condition_numbers), np.array(sigma_max),
            np.array(sigma_min))


def export_matrices(example, hamiltonian, overlap, energies, out_dir):
    """
    Write M(E), Sigma(E) and the right-hand side at every inspection index.

    Sigma(E) is recovered as Sigma = (E S - H) - M(E), i.e. from the definition
    of M, rather than exported separately. All arithmetic stays sparse, so this
    path is usable at sizes where a dense representation is not.
    """
    for index in INSPECTION_INDICES:
        energy = energies[index]
        print(f"  inspection index {index}: E = {energy} eV")

        M, _, rhs = export_system(
            example=example, mode="full",
            energy_index=None, energy=np.array(energy),
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

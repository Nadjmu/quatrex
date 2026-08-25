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
Written to EXPORT_DIR/<material>/, which make_hdf5.py consolidates:

    energies.npy                 the energy grid, eV
    band_edge.npy                the conduction band edge, eV; kept under the
                                 old name for the files already exported
    conduction_band_edge.npy     the same value, under the current name
    valence_band_edge.npy        the valence band edge, eV, where it is known
    H.npz                        the bare Hamiltonian, CSR triplet
    S.npz                        the bare overlap matrix, CSR triplet
    spectrum_bare.npy            eigenvalues of the pencil (H, S)
    condition_bare.npy           kappa_2(H - E S) along the grid
    condition_full_svd.npy       kappa_2(M(E)) along the grid
    max_singular_values.npy      sigma_max(M(E))
    min_singular_values.npy      sigma_min(M(E))
    M_E_<idx>.npz                M(E) at an inspection index, CSR triplet
    Sigma_E_<idx>.npz            Sigma(E) at the same index, CSR triplet
    rhs_E_<idx>.npy              the right-hand side at the same index

No figures are produced; see plotting/matrices2/plot_qtbm_spectra.py, which reads these
arrays.

Usage
-----
    python main3.py
    python plotting/matrices2/plot_qtbm_spectra.py /scratch/yimili/matrices2/dev_12_sorted_BENCH
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from export_qtbm_systems import export_system, _host_array, _save_csr_npz

sys.path.append(str((Path(__file__).resolve().parent / "solvers").resolve()))

import cli

# Stage 1 writes one directory per example here; make_hdf5.py reads them.
EXPORT_DIR = cli.EXPORT_DIR

# Materials to process, as keys of cli.MATERIALS. The example directory, the
# band edges and the block size are taken from that registry, which is the one
# place they are edited.
EXAMPLES = ["carbon-nanotube", "si-bulk", "carbon-chain", "graphene"]

# Stride between exported energy indices. 1 exports M(E), Sigma(E) and the
# right-hand side at every point of the grid, which is what stages 2 to 5
# expect; a larger stride subsamples for a quick pass. The grid itself, and so
# the number of indices, is the `grid` of the material in cli.MATERIALS.
EXPORT_STRIDE = 1

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


def band_edges_of(config, material):
    """
    Valence and conduction band edge in eV, either may be None.

    The registry in cli.MATERIALS takes precedence, since it holds the values
    determined from the contact band structure; the QTBM configuration is the
    fallback. If neither sets a conduction edge, the left Fermi level is used,
    which coincides with it for an equilibrium calculation.

    Returns (valence, conduction).
    """
    valence = material.valence_band_edge if material else None
    conduction = material.conduction_band_edge if material else None

    if valence is None:
        valence = config.electron.valence_band_edge
    if conduction is None:
        conduction = config.electron.conduction_band_edge
    if conduction is None:
        conduction = config.electron.left_fermi_level
        print("  [warn] no conduction band edge in cli.MATERIALS or the QTBM "
              "configuration; using the left Fermi level")
    if valence is None:
        print("  [warn] no valence band edge in cli.MATERIALS or the QTBM "
              "configuration; it will be absent from the material file")
    return valence, conduction


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


def export_indices(energies):
    """Energy indices to export, every EXPORT_STRIDE-th point of the grid."""
    return list(range(0, len(energies), EXPORT_STRIDE))


def export_matrices(example, hamiltonian, overlap, energies, out_dir):
    """
    Write M(E), Sigma(E) and the right-hand side at every exported index.

    Sigma(E) is recovered as Sigma = (E S - H) - M(E), i.e. from the definition
    of M, rather than exported separately. All arithmetic stays sparse, so this
    path is usable at sizes where a dense representation is not.
    """
    for index in export_indices(energies):
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

def process_example(name):
    """Run every enabled analysis for one material of cli.MATERIALS."""
    material = cli.material(name)
    example = material.example
    if example is None:
        raise ValueError(f"cli.MATERIALS['{name}'] has no example directory")
    print(f"Example: {name}")

    out_dir = EXPORT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if material.grid is None:
        raise ValueError(
            f"cli.MATERIALS['{name}'] has no grid. Set "
            f"grid=EnergyGrid(start=..., end=..., resolution=...) in eV.")

    hamiltonian, overlap, config = load_matrices(example)
    valence, band_edge = band_edges_of(config, material)
    energies = material.grid.energies()
    print(f"  conduction band edge = {band_edge} eV, "
          f"valence band edge = {valence} eV")
    print(f"  grid: {len(energies)} points, "
          f"{energies[0]:.4f} .. {energies[-1]:.4f} eV, "
          f"resolution {material.grid.resolution:.3e} eV")
    print(f"  exporting every {EXPORT_STRIDE} index "
          f"({len(export_indices(energies))} matrices)")

    np.save(out_dir / "energies.npy", energies)
    np.save(out_dir / "band_edge.npy", np.array(band_edge))
    np.save(out_dir / "conduction_band_edge.npy", np.array(band_edge))
    if valence is not None:
        np.save(out_dir / "valence_band_edge.npy", np.array(valence))
    _save_csr_npz(out_dir / "H.npz", hamiltonian.tocsr())
    _save_csr_npz(out_dir / "S.npz", overlap.tocsr())

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
    print(f"  plot with: python plotting/matrices2/plot_qtbm_spectra.py {out_dir}\n")


def main():
    for example in EXAMPLES:
        process_example(example)


if __name__ == "__main__":
    main()

"""
Condition Number Analysis for QTBM Systems
==========================================
For each example system this script:
  1. Loads Hamiltonian H and Overlap S matrices.
  2. Solves the generalised eigenvalue problem H v = λ S v, plots the
     spectrum in the complex plane (full + zoom near the band edge), and
     reports the eigenvalue closest to the band edge.
  3. Sweeps energy across [band_edge ± offset] and plots κ(H − E·S).
  4. Builds the full system matrix M(E), plots its eigenvalues at the
     band edge, and plots κ(M(E)) over the same energy sweep.

Commented-out blocks are ablation variants — kept for reference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from export_qtbm_systems import export_system

import scipy
import cupy as cp
import cupyx
from qttools.kernels.linalg import eigvalsh as qt_eigvalsh
from cupyx.scipy.sparse.linalg import gmres
from scipy.sparse.linalg import eigsh

from export_qtbm_systems import _host_array

# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_eigenvalues_complex(w, conduction_band_edge, inspection_energy, name, suffix="", xlim=None, ylim=None):
    plt.figure()
    plt.scatter(w.real, w.imag)
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    plt.axvline(inspection_energy, color="blue", linestyle="-.", label="Inspection energy")
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.title(f"{suffix} - {name}")
    plt.xlabel("Real part (eV)")
    plt.ylabel("Imaginary part (eV)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    fname = f"plots/{name}/{suffix}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_condition_numbers(energies, condition_numbers, conduction_band_edge, name, suffix=""):
    plt.figure()
    plt.plot(energies, condition_numbers, "o-")
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    plt.title(f"{suffix} - {name}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Condition number")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    fname = f"plots/{name}/{suffix}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sparsity(hamiltonian, overlap, name):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.spy(hamiltonian, markersize=1)
    plt.title(f"H sparsity - {name}")
    plt.subplot(1, 2, 2)
    plt.spy(overlap, markersize=1)
    plt.title(f"S sparsity - {name}")
    plt.tight_layout()
    plt.savefig(f"plots/{name}/sparsity_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_matrix_diff(A_singularity, eigenvalue, H_s, S_s, name):
    diff = A_singularity.toarray() - eigenvalue * S_s.toarray() + H_s.toarray()
    plt.matshow(np.log(np.abs(diff.get())), aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.savefig(f"plots/{name}/A_singularity_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_gmres_iterations(energies, iterations, name):
    plt.figure()
    plt.plot(energies, iterations, "o-")
    plt.title(f"GMRES iterations - {name}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Iterations")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{name}/gmres_iterations_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Section functions
# ---------------------------------------------------------------------------

def load_matrices(example):
    hamiltonian, config, __ = export_system(
        example=example, mode="hamiltonian",
        energy_index=None, energy=0,
        k_index=None, k_point=(0, 0, 0),
    )
    overlap, _, __ = export_system(
        example=example, mode="overlap",
        energy_index=None, energy=0,
        k_index=None, k_point=(0, 0, 0),
    )
    hamiltonian = -hamiltonian
    overlap = -overlap
    return hamiltonian, overlap, config


def solve_and_plot_spectrum(hamiltonian, overlap, conduction_band_edge, name, offset):
    # Active: dense solver
    w, v = eigh(hamiltonian.toarray(), overlap.toarray())

    # Ablation: GPU-accelerated eigvalsh
    # w = qt_eigvalsh(hamiltonian.toarray(), oveeigenvaluesrlap.toarray()).get()

    # Ablation: sparse shift-invert
    # n = hamiltonian.shape[0]
    # k = min(20, n - 1)
    # if k <= 0:
    #     raise ValueError(f"Cannot solve eigenproblem for matrix size n={n}")
    # w, _ = eigsh(hamiltonian, k=k, M=overlap, sigma=conduction_band_edge, which="LM")
    # order = abs(w - conduction_band_edge).argsort()
    # w = w[order]

    plot_eigenvalues_complex(w, conduction_band_edge, conduction_band_edge, name, suffix="spectrum_bare")
    plot_eigenvalues_complex(w, conduction_band_edge, conduction_band_edge, name,
                             suffix="spectrum_bare_zoom_E_band",
                             xlim=(conduction_band_edge - offset, conduction_band_edge + offset),
                             ylim=(-0.1, 0.1))

    closest_index = np.argmin(abs(w - conduction_band_edge))
    closest_eigenvalue = w[closest_index]
    print(f"Eigenvalue closest to conduction band edge: {closest_eigenvalue}  (index {closest_index})")
    return w, closest_eigenvalue


def sweep_condition_numbers_bare(hamiltonian, overlap, energies, conduction_band_edge, name):
    condition_numbers = []
    for energy in energies:
        print(f"  Computing condition number for energy = {energy:.4f} eV")
        A = np.array(hamiltonian.toarray() - energy * overlap.toarray())
        condition_numbers.append(np.linalg.cond(A))
    plot_condition_numbers(energies, condition_numbers, conduction_band_edge, name, suffix="condition_bare")
    return condition_numbers


def plot_full_system_eigenvalues(example, inspection_energy, inspection_index, conduction_band_edge, name, offset):
    A_singularity, config, rhs = export_system(
        example=example, mode="full",
        energy_index=None, energy=np.array(inspection_energy),
        k_index=None, k_point=(0, 0, 0),
    )
    w = np.linalg.eig(A_singularity.toarray())[0]

    plot_eigenvalues_complex(w, conduction_band_edge, inspection_energy, name,
                             suffix=f"spectrum_full_E_{inspection_index}")
    plot_eigenvalues_complex(w, conduction_band_edge, inspection_energy, name,
                             suffix=f"spectrum_full_zoom_E_{inspection_index}",
                             xlim=(conduction_band_edge - offset, conduction_band_edge + offset),
                             ylim=(-0.1, 0.1))

    # Ablation: report M's eigenvalue closest to band edge
    # closest_index = np.argmin(abs(w - conduction_band_edge))
    # closest_eigenvalue = w[closest_index].real
    # print(f"M eigenvalue closest to band edge: {closest_eigenvalue}  (index {closest_index})")

    return A_singularity, config, rhs


def sweep_condition_numbers_full(example, energies, conduction_band_edge, name):
    condition_numbers = []
    for energy in energies:
        print(f"  Computing condition number for energy = {energy:.4f} eV")
        A, config, __ = export_system(
            example=example, mode="full",
            energy_index=None, energy=np.array(energy),
            k_index=None, k_point=(0, 0, 0),
        )
        condition_numbers.append(np.linalg.cond(np.array(A.toarray())))
    plot_condition_numbers(energies, condition_numbers, conduction_band_edge, name, suffix="condition_full")
    return condition_numbers

def sweep_condition_numbers_full_svd(example, energies, conduction_band_edge, name):
    condition_numbers = []
    max_singular_values = []
    min_singular_values = []

    for energy in energies:
        print(f"  Computing condition number for energy = {energy:.4f} eV")
        A, config, __ = export_system(
            example=example, mode="full",
            energy_index=None, energy=np.array(energy),
            k_index=None, k_point=(0, 0, 0),
        )

        singular_values = np.linalg.svd(np.array(A.toarray()), compute_uv=False)
        # singular_values is sorted descending, so max is first, min is last
        max_sv = singular_values[0]
        min_sv = singular_values[-1]

        condition_numbers.append(max_sv / min_sv)
        max_singular_values.append(max_sv)
        min_singular_values.append(min_sv)

    plot_condition_numbers(energies, condition_numbers, conduction_band_edge, name, suffix="condition_full_svd")

    return (
        np.array(condition_numbers),
        np.array(max_singular_values),
        np.array(min_singular_values),
    )


def ablation_fine_sweep(example, closest_eigenvalue, name):
    points = 1000
    offset = 0.01
    energies = np.linspace(closest_eigenvalue - offset, closest_eigenvalue + offset, points)
    condition_numbers = []
    for energy in energies:
        A, config, __ = export_system(
            example=example, mode="full",
            energy_index=None, energy=np.array(energy),
            k_index=None, k_point=(0, 0, 0),
        )
        condition_numbers.append(np.linalg.cond(np.array(A.toarray())))
    plot_condition_numbers(energies, condition_numbers, closest_eigenvalue, name, suffix="condition_full_fine_sweep")


def ablation_gmres(example, energies, name):
    iterations = []
    for energy in energies:
        A, config, rhs = export_system(
            example=example, mode="full",
            energy_index=None, energy=np.array(energy),
            k_index=None, k_point=(0, 0, 0),
        )
        counter = 0
        def counter_callback(args):
            nonlocal counter
            counter += 1
        print(rhs.shape)
        if rhs.shape[1] > 0:
            __, info = gmres(A, rhs[:, 0], callback=counter_callback)
            iterations.append(counter)
        else:
            iterations.append(-100)
        print("Counter:", counter)
    plot_gmres_iterations(energies, iterations, name)


def ablation_matrix_diff(example, eigenvalue, name):
    H_s, config = export_system(example=example, mode="hamiltonian",
        energy_index=None, energy=1, k_index=None, k_point=(0, 0, 0))
    H_s = -H_s
    S_s, config = export_system(example=example, mode="overlap",
        energy_index=None, energy=1, k_index=None, k_point=(0, 0, 0))
    S_s = -S_s
    A_singularity, config, __ = export_system(
        example=example, mode="full",
        energy_index=None, energy=np.array(eigenvalue),
        k_index=None, k_point=(0, 0, 0),
    )
    plot_matrix_diff(A_singularity, eigenvalue, H_s, S_s, name)


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

examples = [
    # Path("/capstor/store/cscs/pasc/c33/amaeder/quatrex/dev/condition_numbers/carbon-chain"),
    # Path("/capstor/store/cscs/pasc/c33/amaeder/quatrex/dev/condition_numbers/carbon-nanotube"),
    # Path("/capstor/store/cscs/pasc/c33/amaeder/quatrex/dev/condition_numbers/graphene"),
    #Path("../examples/cp2k/carbon-chain/qtbm"),
    Path("../examples/w90/carbon-nanotube/qtbm"),
]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


for example in examples:
    name = example.parent.name
    print("Example:", name)

    Path("matrices").mkdir(exist_ok=True)
    Path(f"matrices/{name}").mkdir(exist_ok=True)
    Path(f"plots/{name}").mkdir(exist_ok=True)

    # 1. Load matrices
    hamiltonian, overlap, config = load_matrices(example)
    np.save(f"matrices/{name}/H.npy", hamiltonian.toarray())
    np.save(f"matrices/{name}/S.npy", overlap.toarray())
    print("H shape:", hamiltonian.shape)
    print("S shape:", overlap.shape)

    conduction_band_edge = config.electron.conduction_band_edge
    if conduction_band_edge is None:
        conduction_band_edge = config.electron.left_fermi_level

    offset = 1
    points = 10
    offset2 = 0.005
    energies = np.linspace(conduction_band_edge - offset - offset2, conduction_band_edge + offset - offset2, points+1)
    print("Resolution:", 2 * offset / points, "eV")
    print(f"Conduction band edge: {conduction_band_edge} eV")

    # 2. Eigenvalues of H v = λ S v
    w_HS, closest_eigenvalue = solve_and_plot_spectrum(
        hamiltonian, overlap, conduction_band_edge, name, offset)
    np.save(f"matrices/{name}/spectrum_bare.npy", w_HS)

    # 3. Condition number of (H - E*S)
    #condition_numbers_bare = sweep_condition_numbers_bare(hamiltonian, overlap, energies, conduction_band_edge, name)
    #np.save(f"matrices/{name}/condition_bare.npy", condition_numbers_bare)

    # 4. Full system matrix M(E) at each inspection energy
    inspection_indices = list(range(0, 1)) #list(range(0,50))+list(range(150,201))

    for inspection_index in inspection_indices:
        inspection_energy = energies[inspection_index]
        print(f"Inspection energy [{inspection_index}]: {inspection_energy} eV")

        M_E_inspected, config, rhs = plot_full_system_eigenvalues(
            example, inspection_energy, inspection_index, conduction_band_edge, name, offset)
        M_E = M_E_inspected.toarray()
        ES_H = inspection_energy * overlap.toarray() - hamiltonian.toarray()
        Sigma = ES_H - M_E
        w_M = np.linalg.eig(M_E)[0]

        np.save(f"matrices/{name}/M_E_{inspection_index}.npy", M_E)
        np.save(f"matrices/{name}/Sigma_E_{inspection_index}.npy", Sigma)
        np.save(f"matrices/{name}/spectrum_M_E_{inspection_index}.npy", w_M)
        np.save(f"matrices/{name}/rhs_E_{inspection_index}.npy", _host_array(rhs))

    # 5. Condition number of M(E)
    #condition_numbers_full = sweep_condition_numbers_full(example, energies, conduction_band_edge, name)
    #condition_numbers_full, max_singular_values, min_singular_values = sweep_condition_numbers_full_svd(example, energies, conduction_band_edge, name)
    #np.save(f"matrices/{name}/condition_full_svd.npy", condition_numbers_full)
    #np.save(f"matrices/{name}/max_singular_values.npy", max_singular_values)
    #np.save(f"matrices/{name}/min_singular_values.npy", min_singular_values)

    # Ablation: fine sweep around closest M eigenvalue
    # ablation_fine_sweep(example, closest_eigenvalue, name)
    # Ablation: GMRES iteration count
    # ablation_gmres(example, energies, name)
    # Ablation: sparsity pattern
    # plot_sparsity(hamiltonian, overlap, name)
    # Ablation: matrix difference check
    # ablation_matrix_diff(example, conduction_band_edge, name)
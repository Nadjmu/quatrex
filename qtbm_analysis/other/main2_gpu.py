"""
Condition Number Analysis for QTBM Systems (GPU version)
=========================================================
GPU equivalent of main2.py — uses CuPy for matrix operations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from export_qtbm_systems import export_system, _host_array

import scipy
import cupy as cp
import cupyx
from qttools.kernels.linalg import eigvalsh as qt_eigvalsh
from cupyx.scipy.sparse.linalg import gmres
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_eigenvalues_complex(w, conduction_band_edge, name, suffix="", xlim=None, ylim=None):
    plt.figure()
    plt.scatter(w.real, w.imag)
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.title(f"Eigenvalues{suffix} - {name}")
    plt.xlabel("Real part (eV)")
    plt.ylabel("Imaginary part (eV)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    fname = f"plots_gpu/eigenvalues_complex{suffix.lower().replace(' ', '_')}_{name}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_condition_numbers(energies, condition_numbers, conduction_band_edge, name, suffix=""):
    plt.figure()
    plt.plot(energies, condition_numbers, "o-")
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    plt.title(f"Condition number{suffix} - {name}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Condition number")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    fname = f"plots_gpu/condition_numbers{suffix.lower().replace(' ', '_')}_{name}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sparsity(hamiltonian, overlap, name):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.spy(_host_array(hamiltonian.toarray()), markersize=1)
    plt.title(f"H sparsity - {name}")
    plt.subplot(1, 2, 2)
    plt.spy(_host_array(overlap.toarray()), markersize=1)
    plt.title(f"S sparsity - {name}")
    plt.tight_layout()
    plt.savefig(f"plots_gpu/sparsity_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_matrix_diff(A_singularity, eigenvalue, H_s, S_s, name):
    diff = A_singularity.toarray() - eigenvalue * S_s.toarray() + H_s.toarray()
    plt.matshow(np.log(np.abs(diff.get())), aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.savefig(f"plots_gpu/A_singularity_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_gmres_iterations(energies, iterations, name):
    plt.figure()
    plt.plot(energies, iterations, "o-")
    plt.title(f"GMRES iterations - {name}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Iterations")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots_gpu/gmres_iterations_{name}.png", dpi=300, bbox_inches="tight")
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
    # Active: dense solver on CPU (eigh does not support CuPy)
    w, v = eigh(hamiltonian.toarray().get(), overlap.toarray().get())

    # Ablation: GPU-accelerated eigvalsh
    # w = qt_eigvalsh(hamiltonian.toarray(), overlap.toarray()).get()

    # Ablation: sparse shift-invert
    # n = hamiltonian.shape[0]
    # k = min(20, n - 1)
    # if k <= 0:
    #     raise ValueError(f"Cannot solve eigenproblem for matrix size n={n}")
    # w, _ = eigsh(hamiltonian, k=k, M=overlap, sigma=conduction_band_edge, which="LM")
    # order = abs(w - conduction_band_edge).argsort()
    # w = w[order]

    plot_eigenvalues_complex(w, conduction_band_edge, name)
    plot_eigenvalues_complex(w, conduction_band_edge, name,
                             suffix=" near band edge",
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
        A = cp.array(hamiltonian.toarray() - energy * overlap.toarray())
        condition_numbers.append(cp.linalg.cond(A).get())
    plot_condition_numbers(energies, condition_numbers, conduction_band_edge, name, suffix=" of (H - E*S)")
    return condition_numbers


def plot_full_system_eigenvalues(example, eigenvalue, conduction_band_edge, name, offset):
    A_singularity, config, rhs = export_system(
        example=example, mode="full",
        energy_index=None, energy=cp.array(eigenvalue),
        k_index=None, k_point=(0, 0, 0),
    )
    w = cp.linalg.eig(A_singularity.toarray())[0].get()

    plot_eigenvalues_complex(w, conduction_band_edge, name, suffix=" of M")
    plot_eigenvalues_complex(w, conduction_band_edge, name,
                             suffix=" of M near band edge",
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
            energy_index=None, energy=cp.array(energy),
            k_index=None, k_point=(0, 0, 0),
        )
        A = cp.array(A.toarray())
        condition_numbers.append(cp.linalg.cond(A).get())
    plot_condition_numbers(energies, condition_numbers, conduction_band_edge, name, suffix=" of M")
    return condition_numbers


def ablation_fine_sweep(example, closest_eigenvalue, name):
    points = 1000
    offset = 0.01
    energies = np.linspace(closest_eigenvalue - offset, closest_eigenvalue + offset, points)
    condition_numbers = []
    for energy in energies:
        A, config, __ = export_system(
            example=example, mode="full",
            energy_index=None, energy=cp.array(energy),
            k_index=None, k_point=(0, 0, 0),
        )
        A = cp.array(A.toarray())
        condition_numbers.append(cp.linalg.cond(A).get())
    plot_condition_numbers(energies, condition_numbers, closest_eigenvalue, name, suffix=" of M (fine sweep)")


def ablation_gmres(example, energies, name):
    iterations = []
    for energy in energies:
        A, config, rhs = export_system(
            example=example, mode="full",
            energy_index=None, energy=cp.array(energy),
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
    H_s, config, __ = export_system(example=example, mode="hamiltonian",
        energy_index=None, energy=1, k_index=None, k_point=(0, 0, 0))
    H_s = -H_s
    S_s, config, __ = export_system(example=example, mode="overlap",
        energy_index=None, energy=1, k_index=None, k_point=(0, 0, 0))
    S_s = -S_s
    A_singularity, config, __ = export_system(
        example=example, mode="full",
        energy_index=None, energy=cp.array(eigenvalue),
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
    Path("../examples/cp2k/carbon-chain/qtbm"),
    Path("../examples/w90/carbon-nanotube/qtbm"),
]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for example in examples:
    name = example.parent.name
    print("Example:", name)

    Path("matrices").mkdir(exist_ok=True)
    Path("plots_gpu").mkdir(exist_ok=True)

    # 1. Load matrices
    hamiltonian, overlap, config = load_matrices(example)
    np.save(f"matrices/{name}_H.npy", _host_array(hamiltonian.toarray()))
    np.save(f"matrices/{name}_S.npy", _host_array(overlap.toarray()))
    print("H shape:", hamiltonian.shape)
    print("S shape:", overlap.shape)

    conduction_band_edge = config.electron.conduction_band_edge
    if conduction_band_edge is None:
        conduction_band_edge = config.electron.left_fermi_level

    # --- tunable parameter: set to None to use conduction_band_edge ---
    inspection_energy = None
    if inspection_energy is None:
        inspection_energy = conduction_band_edge
    print(f"Inspection energy: {inspection_energy} eV")

    offset = 1
    points = 200
    energies = np.linspace(conduction_band_edge - offset, conduction_band_edge + offset, points)
    print("Resolution:", 2 * offset / points, "eV")

    # 2. Eigenvalues of H v = λ S v
    w_HS, closest_eigenvalue = solve_and_plot_spectrum(
        hamiltonian, overlap, conduction_band_edge, name, offset)
    np.save(f"matrices/{name}_eigenvalues_HS.npy", w_HS)

    # 3. Condition number of (H - E*S)
    sweep_condition_numbers_bare(hamiltonian, overlap, energies, conduction_band_edge, name)

    # 4. Full system matrix M(E) at inspection_energy - save M, Sigma, rhs and eigenvalues
    A_singularity, config, rhs = plot_full_system_eigenvalues(
        example, inspection_energy, conduction_band_edge, name, offset)
    M = _host_array(A_singularity.toarray())
    ES_H = inspection_energy * _host_array(overlap.toarray()) - _host_array(hamiltonian.toarray())
    Sigma = ES_H - M
    w_M = cp.linalg.eig(cp.array(M))[0].get()

    np.save(f"matrices/{name}_M.npy", M)
    np.save(f"matrices/{name}_Sigma.npy", Sigma)
    np.save(f"matrices/{name}_eigenvalues_M.npy", w_M)
    np.save(f"matrices/{name}_rhs.npy", _host_array(rhs))

    # 5. Condition number of M(E)
    sweep_condition_numbers_full(example, energies, conduction_band_edge, name)

    # Ablation: fine sweep around closest M eigenvalue
    # ablation_fine_sweep(example, closest_eigenvalue, name)
    # Ablation: GMRES iteration count
    # ablation_gmres(example, energies, name)
    # Ablation: sparsity pattern
    # plot_sparsity(hamiltonian, overlap, name)
    # Ablation: matrix difference check
    # ablation_matrix_diff(example, conduction_band_edge, name)
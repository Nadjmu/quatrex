"""
Condition Number Analysis for QTBM Systems (GPU version)
=========================================================
GPU equivalent of the updated CPU script (main3.py). Uses CuPy /
cupyx.scipy.sparse so that, for very large systems, M(E) and Sigma(E)
are kept sparse on the GPU and only moved to host memory at save time.

Commented-out blocks are ablation variants or steps that are only
feasible for small/dense systems — kept for reference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from export_qtbm_systems import export_system, _host_array, _save_csr_npz

import scipy
import cupy as cp
import cupyx
import cupyx.scipy.sparse as cusparse
from qttools.kernels.linalg import eigvalsh as qt_eigvalsh
from cupyx.scipy.sparse.linalg import gmres
from scipy.sparse.linalg import eigsh

FOLDER = Path("/scratch/yimili")


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
    fname = FOLDER / "plots" / name / f"{suffix}.png"
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
    fname = FOLDER / "plots" / name / f"{suffix}.png"
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
    plt.savefig(FOLDER / "plots" / name / f"sparsity_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_matrix_diff(A_singularity, eigenvalue, H_s, S_s, name):
    diff = A_singularity.toarray() - eigenvalue * S_s.toarray() + H_s.toarray()
    plt.matshow(np.log(np.abs(diff.get())), aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.savefig(FOLDER / "plots" / name / f"A_singularity_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_gmres_iterations(energies, iterations, name):
    plt.figure()
    plt.plot(energies, iterations, "o-")
    plt.title(f"GMRES iterations - {name}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Iterations")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FOLDER / "plots" / name / f"gmres_iterations_{name}.png", dpi=300, bbox_inches="tight")
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
    # Active: dense solver on CPU (eigh does not support CuPy). This is
    # only for the small bare H/S problem, not the huge full system M(E).
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
        A = cp.array(hamiltonian.toarray() - energy * overlap.toarray())
        condition_numbers.append(cp.linalg.cond(A).get())
    plot_condition_numbers(energies, condition_numbers, conduction_band_edge, name, suffix="condition_bare")
    return condition_numbers


def plot_full_system_eigenvalues(example, inspection_energy, inspection_index, conduction_band_edge, name, offset):
    A_singularity, config, rhs = export_system(
        example=example, mode="full",
        energy_index=None, energy=cp.array(inspection_energy),
        k_index=None, k_point=(0, 0, 0),
    )

    # Dense full eig is infeasible at the sizes we're now targeting
    # (e.g. 40000x40000) — skip it, same as the updated CPU script.
    # w = cp.linalg.eig(A_singularity.toarray())[0].get()
    # plot_eigenvalues_complex(w, conduction_band_edge, inspection_energy, name,
    #                          suffix=f"spectrum_full_E_{inspection_index}")
    # plot_eigenvalues_complex(w, conduction_band_edge, inspection_energy, name,
    #                          suffix=f"spectrum_full_zoom_E_{inspection_index}",
    #                          xlim=(conduction_band_edge - offset, conduction_band_edge + offset),
    #                          ylim=(-0.1, 0.1))

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
    plot_condition_numbers(energies, condition_numbers, conduction_band_edge, name, suffix="condition_full")
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
    plot_condition_numbers(energies, condition_numbers, closest_eigenvalue, name, suffix="condition_full_fine_sweep")


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
    # Path("/scratch/yimili/examples/cp2k/carbon-chain/qtbm"),
    # Path("/scratch/yimili/examples/w90/carbon-nanotube/qtbm"),
    # Path("/scratch/yimili/examples/w90/si-bulk/qtbm"),
    # Path("/scratch/yimili/examples/graphene/qtbm"),
    #Path("/scratch/yimili/examples/dev_12_sorted_BENCH/qtbm"),
    #Path("/scratch/yimili/examples/WS2-hBN-25_benchmark-QUATREX-DZ/qtbm"),
    Path("/scratch/yimili/examples/dev_1_CP2K/qtbm")
]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for example in examples:
    name = example.parent.name
    print("Example:", name)

    (FOLDER / "matrices").mkdir(parents=True, exist_ok=True)
    (FOLDER / "matrices" / name).mkdir(parents=True, exist_ok=True)
    (FOLDER / "plots").mkdir(parents=True, exist_ok=True)
    (FOLDER / "plots" / name).mkdir(parents=True, exist_ok=True)

    # 1. Load matrices
    hamiltonian, overlap, config = load_matrices(example)
    # np.save(FOLDER / "matrices" / name / "H.npy", _host_array(hamiltonian.toarray()))
    # np.save(FOLDER / "matrices" / name / "S.npy", _host_array(overlap.toarray()))
    # _save_csr_npz handles the device->host transfer itself, so the
    # cupyx sparse matrix can be passed straight in:
    # _save_csr_npz(FOLDER / "matrices" / name / "H.npz", hamiltonian.tocsr())
    # _save_csr_npz(FOLDER / "matrices" / name / "S.npz", overlap.tocsr())

    conduction_band_edge = config.electron.conduction_band_edge
    if conduction_band_edge is None:
        conduction_band_edge = config.electron.left_fermi_level

    offset = 1
    points = 401
    offset2 = 0.005
    energies = np.linspace(conduction_band_edge - offset - offset2, conduction_band_edge + offset, points + 1)
    print(f"Conduction band edge: {conduction_band_edge} eV")

    # 2. Eigenvalues of H v = λ S v
    # w_HS, closest_eigenvalue = solve_and_plot_spectrum(
    #     hamiltonian, overlap, conduction_band_edge, name, offset)
    # np.save(FOLDER / "matrices" / name / "spectrum_bare.npy", w_HS)

    # 3. Condition number of (H - E*S)
    # condition_numbers_bare = sweep_condition_numbers_bare(hamiltonian, overlap, energies, conduction_band_edge, name)
    # np.save(FOLDER / "matrices" / name / "condition_bare.npy", condition_numbers_bare)

    # 4. Full system matrix M(E) at a sparse subset of inspection energies.
    # Kept sparse throughout (cupyx.scipy.sparse) — dense .toarray() at
    # e.g. 40000x40000 would be ~12.8 GB (float64) / ~25.6 GB (complex128)
    # per matrix, which is not worth doing when only M/Sigma/rhs need saving.
    inspection_indices = list(range(1, 401, 2))

    for inspection_index in inspection_indices:
        inspection_energy = energies[inspection_index]
        print(f"Inspection energy [{inspection_index}]: {inspection_energy} eV")

        M_E_inspected, config, rhs = plot_full_system_eigenvalues(
            example, inspection_energy, inspection_index, conduction_band_edge, name, offset)

        print("done with loading M_E_inspected and rhs")

        M_E = M_E_inspected.tocsr()                                     # stay sparse, on GPU
        ES_H = (inspection_energy * overlap - hamiltonian).tocsr()       # sparse arithmetic, on GPU
        Sigma = (ES_H - M_E).tocsr()

        print("starting saving sparse matrices to disk")

        # Dense full eig is infeasible at these sizes — get a few extreme
        # eigenvalues instead if ever needed:
        # w_M, _ = eigsh(M_E.get(), k=6, which="LM")

        _save_csr_npz(FOLDER / "matrices" / name / f"M_E_{inspection_index}.npz", M_E)
        print("saved M_E")
        _save_csr_npz(FOLDER / "matrices" / name / f"Sigma_E_{inspection_index}.npz", Sigma)
        print("saved Sigma_E")
        # np.save(FOLDER / "matrices" / name / f"spectrum_M_E_{inspection_index}.npy", w_M)
        np.save(FOLDER / "matrices" / name / f"rhs_E_{inspection_index}.npy", _host_array(rhs))
        print("saved rhs")

    # 5. Condition number of M(E)
    # condition_numbers_full = sweep_condition_numbers_full(example, energies, conduction_band_edge, name)

    # Ablation: fine sweep around closest M eigenvalue
    # ablation_fine_sweep(example, closest_eigenvalue, name)
    # Ablation: GMRES iteration count
    # ablation_gmres(example, energies, name)
    # Ablation: sparsity pattern
    # plot_sparsity(hamiltonian, overlap, name)
    # Ablation: matrix difference check
    # ablation_matrix_diff(example, conduction_band_edge, name)
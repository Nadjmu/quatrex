
from pathlib import Path
from export_qtbm_systems import export_system
import numpy as np

from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh  # not scipy.sparse.linalg
import matplotlib.pyplot as plt
import scipy
import cupy as cp
import cupyx
from qttools.kernels.linalg import eigvalsh as qt_eigvalsh
from cupyx.scipy.sparse.linalg import gmres

examples = [
    #Path("/capstor/store/cscs/pasc/c33/amaeder/quatrex/dev/condition_numbers/carbon-chain"),
    #Path("/capstor/store/cscs/pasc/c33/amaeder/quatrex/dev/condition_numbers/carbon-nanotube"),
    #Path("/capstor/store/cscs/pasc/c33/amaeder/quatrex/dev/condition_numbers/graphene"),
    Path("../examples/cp2k/carbon-chain/qtbm"),
    Path("../examples/w90/carbon-nanotube/qtbm"),
]


for example in examples:
    print("in path: ", example.parent.name)
    
    hamiltonian, config, __ = export_system(
        example=example,
        mode="hamiltonian",
        energy_index=None,
        energy=0,
        k_index=None,
        k_point=(0,0,0),
    )

    overlap, _, __ = export_system(
        example=example,
        mode="overlap",
        energy_index=None,
        energy=0,
        k_index=None,
        k_point=(0,0,0),
    )

    hamiltonian = -hamiltonian
    overlap = -overlap

    print(hamiltonian.shape)
    print(overlap.shape)

    conduction_band_edge = config.electron.conduction_band_edge
    if conduction_band_edge is None:
        conduction_band_edge = config.electron.left_fermi_level

    #w = qt_eigvalsh(hamiltonian.toarray(), overlap.toarray()).get()
    w, v = eigh(hamiltonian.toarray(), overlap.toarray())
    plt.figure()
    # plt.plot(w.real, w.imag, "o")
    plt.scatter(w.real, w.imag)
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    plt.title(f"Eigenvalues in the complex plane for {example.parent.name}")
    plt.xlabel("Real part (eV)")
    plt.ylabel("Imaginary part (eV)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/eigenvalues_complex_{example.parent.name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    offset = 1

    # zoom in around the conduction band edge
    plt.figure()
    plt.scatter(w.real, w.imag)
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    plt.xlim(conduction_band_edge - offset, conduction_band_edge + offset)
    plt.ylim(-0.1, 0.1)
    plt.title(f"Eigenvalues near conduction band edge for {example.parent.name}")
    plt.xlabel("Real part (eV)")
    plt.ylabel("Imaginary part (eV)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/eigenvalues_complex_zoom_{example.parent.name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # print eigenvalue closest to the conduction band edge
    closest_index = np.argmin(abs(w - conduction_band_edge))
    closest_eigenvalue = w[closest_index]
    print(f"Eigenvalue closest to conduction band edge: {closest_eigenvalue} (index {closest_index})")


    eigenvalue = conduction_band_edge
    points = 200

    print("Resolution = ", 2 * offset / points)
    
    energies = np.linspace(eigenvalue - offset, eigenvalue + offset, points)
    print("HEREEEE    -----      1111")
    
    condition_numbers = []
    for energy in energies:
        print(f"Computing condition number for energy = {energy:.4f} eV")
        A = np.array(hamiltonian.toarray() - energy * overlap.toarray())
        condition_number = np.linalg.cond(A)
        condition_numbers.append(condition_number)

    plt.figure()
    plt.plot(energies, condition_numbers, "o-")
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    plt.title(f"Condition number of (H - E S) for {example.parent.name}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Condition number")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/condition_numbers_bare_{example.parent.name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    

##################################  next system ##################################

    A_singularity, config, __ = export_system(
        example=example,
        mode="full",
        energy_index=None,
        energy=np.array(eigenvalue),
        k_index=None,
        k_point=(0,0,0),
    )

    w = np.linalg.eig(A_singularity.toarray())[0]
    

    plt.figure()
    # plt.plot(w.real, w.imag, "o")
    plt.scatter(w.real, w.imag)
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    plt.title(f"Eigenvalues in the complex plane for {example.parent.name}")
    plt.xlabel("Real part (eV)")
    plt.ylabel("Imaginary part (eV)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/eigenvalues_complex_full_{example.parent.name}.png", dpi=300, bbox_inches="tight")
    plt.close()


    # # print eigenvalue closest to the conduction band edge
    # closest_index = np.argmin(abs(w - conduction_band_edge))
    # closest_eigenvalue = w[closest_index].real
    # print(f"Eigenvalue closest to conduction band edge: {closest_eigenvalue} (index {closest_index})")


    # zoom in around the conduction band edge
    plt.figure()
    plt.scatter(w.real, w.imag)
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    plt.xlim(conduction_band_edge - offset, conduction_band_edge + offset)
    plt.ylim(-0.1, 0.1)
    plt.title(f"Eigenvalues near conduction band edge for {example.parent.name}")
    plt.xlabel("Real part (eV)")
    plt.ylabel("Imaginary part (eV)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/eigenvalues_complex_full_zoom_{example.parent.name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("HEEEEEREEEE    -----      1111")
    # eigenvalue = closest_eigenvalue
    # points = 1000
    # offset = 0.01

    # energies = np.linspace(eigenvalue - offset, eigenvalue + offset, points)

    condition_numbers = []
    for energy in energies:
        print(f"Computing condition number for energy = {energy:.4f} eV")
        A, config, __ = export_system(
            example=example,
            mode="full",
            energy_index=None,
            energy=np.array(energy),
            k_index=None,
            k_point=(0,0,0),
        )
        A = np.array(A.toarray())
        condition_number = np.linalg.cond(A)
        condition_numbers.append(condition_number)

    plt.figure()
    plt.plot(energies, condition_numbers, "o-")
    plt.axvline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    plt.title(f"Condition number of M for {example.parent.name}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Condition number")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/condition_numbers_full_{example.parent.name}.png", dpi=300, bbox_inches="tight")
    plt.close()


    # eigenvalue = closest_eigenvalue
    # points = 200
    # offset = 1

    # energies = np.linspace(eigenvalue - offset, eigenvalue + offset, points)

    # condition_numbers = []
    # for energy in energies:
    #     print(f"Computing condition number for energy = {energy:.4f} eV")
    #     A, config, __ = export_system(
    #         example=example,
    #         mode="full",
    #         energy_index=None,
    #         energy=np.array(energy),
    #         k_index=None,
    #         k_point=(0,0,0),
    #     )
    #     A = np.array(A.toarray())
    #     condition_number = np.linalg.cond(A)
    #     condition_numbers.append(condition_number.get())

    # plt.figure()
    # plt.plot(energies, condition_numbers, "o-")
    # plt.axvline(eigenvalue, color="red", linestyle="--", label="Conduction band edge")
    # plt.title(f"Condition number of M for {example.parent.name}")
    # plt.xlabel("Energy (eV)")
    # plt.ylabel("Condition number")
    # plt.yscale("log")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(f"plots/{example.parent.name}_condition_numbers_wide.png", dpi=300, bbox_inches="tight")
    # plt.close()

        
"""
    print("HEEEEEREEEE    -----      2222")
    iterations = []
    for energy in energies:
        print(f"Computing condition number for energy = {energy:.4f} eV")
        A, config, rhs = export_system(
            example=example,
            mode="full",
            energy_index=None,
            energy=np.array(energy),
            k_index=None,
            k_point=(0,0,0),
        )
        counter = 0
        def counter_callback(args):
            global counter
            counter += 1
    
        print(rhs.shape)


        if rhs.shape[1] > 0:
            __, info = gmres(A, rhs[:,0], callback=counter_callback)
            iterations.append(counter)
        else:
            iterations.append(-100)

        print("Counter:", counter)

    plt.figure()
    plt.plot(energies, iterations, "o-")
    plt.title(f"GMRES iterations for {example.parent.name}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Iterations")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/gmres_iterations_{example.parent.name}.png", dpi=300, bbox_inches="tight")
    plt.close()
"""

    # # A_singularity, config = export_system(
    # #     example=example,
    # #     mode="full",
    # #     energy_index=None,
    # #     energy=np.array(eigenvalue),
    # #     k_index=None,
    # #     k_point=(0,0,0),
    # # )
    # # H_singularity, config = export_system(
    # #     example=example,
    # #     mode="hamiltonian",
    # #     energy_index=None,
    # #     energy=1,
    # #     k_index=None,
    # #     k_point=(0,0,0),
    # # )
    # # H_singularity = -H_singularity
    # # S_singularity, config = export_system(
    # #     example=example,
    # #     mode="overlap",
    # #     energy_index=None,
    # #     energy=1,
    # #     k_index=None,
    # #     k_point=(0,0,0),
    # # )
    # # S_singularity = -S_singularity

    # # test = A_singularity.toarray() - eigenvalue * S_singularity.toarray() + H_singularity.toarray()

    # # plt.matshow(np.log(np.abs(test.get() )), aspect="auto", cmap="viridis")
    # # plt.colorbar()
    # # plt.savefig(f"plots/{example.parent.name}_A_singularity.png", dpi=300, bbox_inches="tight")
    # # plt.close()

    # energies = np.linspace(eigenvalue - offset, eigenvalue + offset, points)

    # condition_numbers = []
    # for energy in energies:
    #     print(f"Computing condition number for energy = {energy:.4f} eV")
    #     A, config = export_system(
    #         example=example,
    #         mode="full",
    #         energy_index=None,
    #         energy=np.array(energy),
    #         k_index=None,
    #         k_point=(0,0,0),
    #     )
    #     A = np.array(A.toarray())
    #     condition_number = np.linalg.cond(A)
    #     condition_numbers.append(condition_number.get())

    # plt.figure()
    # plt.plot(energies, condition_numbers, "o-")
    # plt.axvline(eigenvalue, color="red", linestyle="--", label="Conduction band edge")
    # plt.title(f"Condition number of M for {example.parent.name}")
    # plt.xlabel("Energy (eV)")
    # plt.ylabel("Condition number")
    # plt.yscale("log")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(f"plots/{example.parent.name}_condition_numbers_full.png", dpi=300, bbox_inches="tight")
    # plt.close()


    # # # solve for the 20 eigenvalues around the conduction band edge
    # # n = hamiltonian.shape[0]
    # # k = min(20, n - 1)

    # # if k <= 0:
    # #     raise ValueError(f"Cannot solve eigenproblem for matrix size n={n}")

    # # eigenvalues, _ = eigsh(
    # #     hamiltonian,
    # #     k=k,
    # #     M=overlap,
    # #     sigma=conduction_band_edge,
    # #     which="LM",
    # # )

    # # # eigsh does not guarantee order; sort by distance to the target shift.
    # # order = abs(eigenvalues - conduction_band_edge).argsort()
    # # eigenvalues = eigenvalues[order]

    # # print(f"Example: {example.parent.name}")
    # # print(f"Conduction band edge: {conduction_band_edge}")
    # # print(f"{k} eigenvalues nearest the conduction band edge:")
    # # print(eigenvalues)

    # # # plot the eigenvalues    plt.figure()
    # # plt.plot(eigenvalues, "o")
    # # plt.axhline(conduction_band_edge, color="red", linestyle="--", label="Conduction band edge")
    # # plt.title(f"Eigenvalues near conduction band edge for {example.parent.name}")
    # # plt.xlabel("Index")
    # # plt.ylabel("Energy (eV)")
    # # plt.legend()
    # # plt.grid()
    # # plt.tight_layout()
    # # plt.savefig(f"plots/{example.parent.name}_eigenvalues.png", dpi=300, bbox_inches="tight")

    # # plt.close()
    # # # spy hamiltonian and overlap
    # # plt.figure(figsize=(12, 6))
    # # plt.subplot(1, 2, 1)
    # # plt.spy(hamiltonian, markersize=1)
    # # plt.title(f"Hamiltonian sparsity pattern for {example.parent.name}")
    # # plt.subplot(1, 2, 2)
    # # plt.spy(overlap, markersize=1)
    # # plt.title(f"Overlap sparsity pattern for {example.parent.name}")
    # # plt.tight_layout()
    # # plt.savefig(f"plots/{example.parent.name}_sparsity.png", dpi=300, bbox_inches="tight")
    # # plt.close()



    


    # # w = np.linalg.eigvals(np.array(A_singularity.toarray())).get()

    # # print(w)

    # # plt.figure()
    # # # plt.plot(w.real, w.imag, "o")
    # # plt.scatter(w.real, w.imag)
    # # plt.title(f"Eigenvalues in the complex plane for {example.parent.name}")
    # # plt.xlabel("Real part (eV)")
    # # plt.ylabel("Imaginary part (eV)")
    # # plt.legend()
    # # plt.grid()
    # # plt.tight_layout()
    # # plt.savefig(f"plots/{example.parent.name}_eigenvalues_complex.png", dpi=300, bbox_inches="tight")
    # # plt.close()

    # # # zoom in around the conduction band edge
    # # plt.figure()
    # # plt.scatter(w.real, w.imag)
    # # plt.xlim(0 - 0.1, 0 + 0.1)
    # # plt.ylim(-0.1, 0.1)
    # # plt.title(f"Eigenvalues near conduction band edge for {example.parent.name}")
    # # plt.xlabel("Real part (eV)")
    # # plt.ylabel("Imaginary part (eV)")
    # # plt.legend()
    # # plt.grid()
    # # plt.tight_layout()
    # # plt.savefig(f"plots/{example.parent.name}_eigenvalues_complex_zoom.png", dpi=300, bbox_inches="tight")
    # # plt.close()

    # # # print eigenvalue closest to the conduction band edge
    # # closest_index = np.argmin(abs(w - conduction_band_edge))
    # # closest_eigenvalue = w[closest_index]
    # # print(f"Eigenvalue closest to conduction band edge: {closest_eigenvalue} (index {closest_index})")
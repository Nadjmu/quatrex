# Simulation Output

## NEGF Simulation Output

### Charge carrier densities

The charge carrier densities are computed from the lesser and greater
Green's functions and saved as `numpy` arrays (`electron_density.npy` /
`hole_density.npy`) in the
[`output_directory`](parameters/quatrex/#output_directory). These
quantities are both orbital-resolved and have shapes of `(num_energies,
*num_kpoints, num_orbitals)`. `*num_kpoints` is only present if the
simulation is performed with a transverse k-point grid (see
[`kpoint_grid`](parameters/device/#kpoint_grid)).

### Local density of states (LDOS)

Similar to the charge carrier densities, the local density of states
(LDOS) is computed from the retarded Green's function and saved as a
`numpy` array (`ldos.npy`) in the
[`output_directory`](parameters/quatrex/#output_directory). The LDOS is
also orbital-resolved and has a shape of `(num_energies, *num_kpoints,
num_orbitals)`.

### Spectral current

Two types of spectral device current can be computed with `quatrex`:

- The Meir-Wingreen current, which is computed from the lesser and
  greater Green's functions and the self-energies, and saved as a
  `numpy` array (`current_meir_wingreen.npy`). This quantity is resolved
  per transport cell and has a shape of `(num_energies, *num_kpoints,
  num_transport_cells + 1)`. It includes the contribution from the
  reservoirs into / out of the device (hence the `+ 1`). It is only
  output if [`compute_current`](parameters/solver/#compute_current) flag
  is set to `true`. In block-distributed simulations, only the contact
  currents are computed, while the remainder will be set to NaN.

- The spectral current computed from the commutator of the Hamiltonian
  and the lesser Green's function (quantum Liouville equation). This
  quantity is also resolved per transport cell and has a shape of
  `(num_energies, *num_kpoints, num_transport_cells - 1)`. This only
  includes the current flowing between the transport cells (not the
  reservoirs).

## QTBM Simulation Output

### Transmission function

The transmission function is the main output of a QTBM simulation. It is
written for every combination of leads and saved as a `numpy` array
`transmission_<xy>.npy`, where `<xy>` indicates the direction of
transport denoted by the contact name initials (e.g., `lr` for two
contacts named `"left"` and `"right"`). The transmission function has a
shape of `(*num_kpoints, num_energies)`.

### Current

The current output from a QTBM simulation is already integrated over
energy and the transverse k-points and saved as `current_<xy>.npy`,
where `<xy>` is again the direction of transport denoted by the
contacts.

### Local density of states (LDOS) per contact

The LDOS per contact contains the contribution of each contact to the
total LDOS. These are saved as `numpy` arrays `dos_<x>.npy`, where `<x>`
is the contact name. They are orbital-resolved and have a shape of
`(*num_kpoints, num_orbitals, num_energies)`.

## Self-consistent Schrödinger-Poisson Simulation Output

## Profiling and Timing Information

# The Quantum Transmitting Boundary Method

The Quantum Transmitting Boundary Method (QTBM) is one of the two
transport formalisms implemented in `quatrex` (the other being
[NEGF](negf.md)), selectable through the
[`formalism`](../parameters/quatrex.md#formalism) parameter. Instead of
computing the resolvent $\mathbf{G}^R(E)$ of the [system
matrix](quantum_transport.md), QTBM directly solves for the
*device wavefunctions*[^1]. This makes it
well suited for coherent (ballistic) transport, where no scattering
self-energies are present, at a fraction of the cost of a full NEGF
calculation.

[^1]: Brück, Sascha. Ab-initio quantum transport simulations for
    nanoelectronic devices. Diss. ETH Zurich, 2017.

## Device Wavefunction Formalism

$$
\begin{equation}
\left[E\mathbf{S} - \mathbf{H} - \mathbf{\Sigma}_{obc}(E)\right]
\boldsymbol{\Psi}(E) = \mathbf{Q}(E)
\label{eq:qtbm_linear_system}
\end{equation}
$$

Rather than inverting the system matrix, QTBM solves the linear system
in Equation $\ref{eq:qtbm_linear_system}$ for the device wavefunctions
$\boldsymbol{\Psi}(E)$, given a source term $\mathbf{Q}(E)$ that is
non-zero only on the orbitals belonging to a contact (see
[Contact](../parameters/contact.md)). Each column of $\mathbf{Q}(E)$
and $\boldsymbol{\Psi}(E)$ corresponds to a single carrier injected
from one of the contacts, so a device with several contacts and
several open channels per contact is solved as a single linear system
with multiple right-hand sides. The same boundary self-energies
$\mathbf{\Sigma}_{obc}(E)$ discussed in [OBC](obc.md) are reused here.
However, QTBM always relies on the `spectral` OBC algorithm,
since the injection vectors directly derive from the eigenpairs of 
the polynomial eigenvalue problem (see [NEVP](nevp.md)), 

## Injection Vectors

At each contact and energy, the `spectral` OBC solver solves the
polynomial eigenvalue problem (Equation $\ref{eq:poly_eig}$ in
[NEVP](nevp.md)) and classifies its eigenmodes by group velocity and
decay. Propagating modes ($|\lambda| \approx 1$) are separated by the
sign of their group velocity: those carrying flux *into the device* are
*injected* modes, while those carrying flux *back into the contact* are
*reflected*. Evanescent modes that decay into the contact
($|\lambda| > 1$) are also classified as *reflected*, while modes that
grow into the contact are discarded. Only the reflected modes are used
to build $\mathbf{\Sigma}_{obc}$, and only the (necessarily propagating)
injected modes carry current into the device. For every injected mode
$n$ with surface amplitude $\mathbf{b}_n$, a corresponding source column
is built as

$$
\begin{equation}
\mathbf{q}_n = -\mathbf{m}_{-1} \mathbf{b}_n
\label{eq:injection_vector}
\end{equation}
$$

and placed at the contact's orbital indices in $\mathbf{Q}(E)$, where
$\mathbf{m}_{-1}$ is the same coupling block used to build the boundary
self-energy in [OBC](obc.md). Physically, Equation
$\ref{eq:injection_vector}$ is the source that an incoming Bloch wave
of the semi-infinite contact induces on the finite device once the
contact degrees of freedom have been eliminated in favor of
$\mathbf{\Sigma}_{obc}$.

!!! info "Energy Batching"
    OBCs are processed in energy batches whose size
    is controlled by
    [`max_batch_size`](../parameters/qtbm.md#max_batch_size).

## Assembling and Solving the Linear System

For every $k$-point and energy, the device Hamiltonian and overlap
matrices are Bloch-summed over lattice vectors to form $\mathbf{H}$ and
$\mathbf{S}$ (see [Atomistic Material
Descriptions](quantum_transport.md#atomistic-material-descriptions)),
the boundary self-energies $\mathbf{\Sigma}_{obc}(E)$ are computed for
every contact and inserted at their respective contact blocks, and the
resulting sparse system is factorized once and solved for all injected
modes of all contacts simultaneously with a direct solver (see
[Solver](../parameters/solver.md)). Reusing a single factorization
across all right-hand-side columns is what makes QTBM cheap compared to
computing the full $\mathbf{G}^R(E)$: the number of columns to solve
for is the number injected modes, which is typically much smaller
than the number of orbitals in the device.

If a contact's periodic cell is repeated $n_y \times n_z$ times in the
two transverse directions, the boundary
self-energy and injection vectors of the small unit cell are computed
on a Monkhorst-Pack grid of transverse wavevectors and Bloch-summed
back up to the size of the full contact block before being inserted
into the device-sized system matrix.

## Low-Rank Open Boundary Formulation

Explicitly forming and inserting the dense boundary self-energy
$\mathbf{\Sigma}_{obc} = \mathbf{m}_{-1} \mathbf{g}^R \mathbf{m}_{+1}$,
whose surface Green's function $\mathbf{g}^R$ is built from the filtered
eigenpairs (see Equations $\ref{eq:retarded_boundary_self_energy}$ and
$\ref{eq:g_construction}$ in [OBC](obc.md)), into the system matrix
introduces fill-in over the contact block and breaks the Hermitian
symmetry that the bare system matrix $E\mathbf{S} - \mathbf{H}$ would
otherwise have. When
[`low_rank_obc`](../parameters/qtbm.md#low_rank_obc) is enabled, the
self-energy is never assembled. Instead, the bare (Hermitian) system is
solved with both the injection vectors and the reflected modes as
right-hand-side columns,

$$
\begin{equation}
\left[E\mathbf{S} - \mathbf{H}\right]
\begin{bmatrix} \boldsymbol{\Psi}^{(0)}_{inj} & \boldsymbol{\Psi}_{refl} \end{bmatrix}
=
\begin{bmatrix} \mathbf{Q}_{inj} & \mathbf{Q}_{refl} \end{bmatrix},
\qquad
\mathbf{Q}_{refl} = -\mathbf{m}_{-1} \mathbf{V}_{refl}
\label{eq:low_rank_system}
\end{equation}
$$

where $\mathbf{Q}_{inj}$ collects the injection vectors of Equation
$\ref{eq:injection_vector}$ and $\mathbf{Q}_{refl}$ projects the
reflected modes $\mathbf{V}_{refl}$ through the same $-\mathbf{m}_{-1}$
coupling block. The bare system matrix on the left-hand side of Equation
$\ref{eq:low_rank_system}$ carries no boundary self-energy; the open
boundaries enter only through the extra right-hand-side columns. Its
solution $\boldsymbol{\Psi}^{(0)}_{inj}$ is therefore not yet the true
injected-mode wavefunction, and $\boldsymbol{\Psi}_{refl}$ is the bare
response to the reflected-mode columns (not the modes themselves).

The true injected-mode wavefunction $\boldsymbol{\Psi}_{inj}$ is then
recovered from these bare solutions through the correction

$$
\begin{equation}
\boldsymbol{\Psi}_{inj} =
\boldsymbol{\Psi}^{(0)}_{inj} +
\boldsymbol{\Psi}_{refl}
\left[\boldsymbol{\Lambda}_{refl} - \mathbf{V}^{-1}_{refl}
\boldsymbol{\Psi}_{refl}\right]^{-1}
\mathbf{V}^{-1}_{refl} \boldsymbol{\Psi}^{(0)}_{inj}
\label{eq:low_rank_correction}
\end{equation}
$$

where $\boldsymbol{\Lambda}_{refl}$ and $\mathbf{V}^{-1}_{refl}$ are the
eigenvalues and pseudo-inverse of the reflected modes. Equation
$\ref{eq:low_rank_correction}$ is a Woodbury update that includes the
boundary self-energy in its low-rank reflected-mode representation,
$\mathbf{\Sigma}_{obc} = -\mathbf{m}_{-1} \mathbf{V}_{refl}
\boldsymbol{\Lambda}^{-1}_{refl} \mathbf{V}^{-1}_{refl}$, whose rank $k$
is the number of reflected modes. It therefore only requires solving a
dense linear system of size $k$, which is typically much smaller than
the number of contact orbitals.

Keeping the system matrix Hermitian (or, in $\Gamma$-point-only
simulations with a real Hamiltonian, real symmetric) allows the use of
symmetry-exploiting direct solvers, which further reduces the
factorization cost.

!!! warning "Ill-Conditioning Near Van Hove Singularities"
    Without the boundary self-energy, the bare system matrix loses the
    regularizing effect that $\mathbf{\Sigma}_{obc}$ has close to Van
    Hove singularities, so `low_rank_obc` can lead to a more
    ill-conditioned solve at those energies.

## Observables

Once $\boldsymbol{\Psi}(E)$ is known for every injected mode of every
contact, all transport observables are obtained directly from the
wavefunctions:

- **Transmission**: for carriers injected from contact $a$, the
  transmission into contact $b$ is computed from the outflow of
  probability current through contact $b$'s boundary, using the
  anti-Hermitian part of its boundary self-energy (or, equivalently in
  the low-rank case, its reflected modes) as a generalized broadening
  operator acting on the wavefunction restricted to contact $b$'s
  orbitals.
- **Local density of states**: obtained from the spectral weight
  $\boldsymbol{\Psi}^\dagger \mathbf{S} \boldsymbol{\Psi}$ of the
  injected wavefunctions, orbital-resolved and normalized by $2\pi$,
  including the contribution of the semi-infinite contacts through
  their Bloch-reconstructed wavefunction.
- **Contact current**: the Landauer-Büttiker current between each pair
  of contacts is obtained by integrating the transmission weighted by
  the difference of the two contacts' Fermi functions over energy and
  averaging over $k$-points.
- **Charge density**: the electron and hole densities are obtained by
  integrating the local density of states weighted by the occupation of
  each contact, separated into electron- and hole-like contributions
  around the mid-gap energy of the reference (zero-voltage) contact.
  This is what feeds back into the [electrostatics](electrostatics.md)
  solver in self-consistent calculations.

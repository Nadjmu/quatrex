# The Non-Equilibrium Green's Function Formalism

To include scattering processes in quantum transport simulations, we
need to go beyond the coherent transport formalism of the
[QTBM](qtbm.md) and use the *non-equilibrium Green's function (NEGF)
formalism*. This formalism allows us to include interactions with
phonons, photons, and other electrons on the same theoretical footing.

Because the scattering self-energies depend on the Green's functions
(and vice versa), the two must be solved together. This is done using
the self-consistent Born approximation (SCBA): we iterate between
computing the Green's functions from the current self-energies, and
computing the self-energies from the current Green's functions, until
the two are consistent with one another. Convergence of this loop
ensures that energy and momentum are conserved throughout the device.

The rest of this page breifly works through the pieces of that loop: the
[Dyson and Keldysh equations](#dyson-and-keldysh-equations) that define
the Green's functions, the general structure of the self-energies under
[SCBA](#self-consistent-born-approximation-scba), and finally the two
[interactions `quatrex` currently supports](#interacting-systems), i.e.,
the screened Coulomb interaction (electron-electron) and a simplified
optical phonon model.

## Dyson and Keldysh Equations

Under non-equilibrium conditions, the occupation of states is no longer
simply given by the Fermi-Dirac distribution, and is instead encoded in
the *lesser* and *greater* Green's functions, $\mathbf{G}^{<}(E)$ and
$\mathbf{G}^{>}(E)$, which espectively describe occupied and unoccupied
states at energy $E$.

The retarded Green's function $\mathbf{G}^R(E)$ is computed from the Dyson equation

$$
\left[E\mathbf{S} - \mathbf{H} - \mathbf{\Sigma}^R(E)\right]
\mathbf{G}^R(E) = \mathbf{I}
$$

where $\mathbf{H}$ and $\mathbf{S}$ are the device Hamiltonian and
overlap matrices, $\mathbf{I}$ is the identity, and
$\mathbf{\Sigma}^R(E)$ is the retarded self-energy describing all
[scattering processes](#interacting-systems), as well as the [open
boundaries](obc.md) (as in the QTBM formalism). The advanced Green's
function is given as $\mathbf{G}^A(E) = [\mathbf{G}^R(E)]^{\dagger}$.

The lesser and greater Green's functions are computed from the Keldysh
equation

$$
\mathbf{G}^{\lessgtr}(E) = \mathbf{G}^R(E) \mathbf{\Sigma}^{\lessgtr}(E) \mathbf{G}^A(E)
$$

where $\mathbf{\Sigma}^{\lessgtr}(E)$ are the lesser and greater
self-energies associated with the same scattering processes and
contacts.

The lesser/greater contact for electrons are computed from the retarded
contact self-energy as described in the [OBC
section](obc.md#lessergreater-open-boundary-self-energy)

## Self-Consistent Born Approximation (SCBA)

All lesser/greater self-energies and polarizations have skew-hermitian
symmetry, i.e., $\mathbf{B}^{>}(E) = -[\mathbf{B}^{<}(E)]^{\dagger}$.
The retarded self-energies and polarizations contain both a
skew-hermitian part that describes the scattering rate and a Hermitian
part that describes the energy renormalization due to the interaction.
The Hermitian part can be computed from the skew-Hermitian part using
the Kramers-Kronig relation, which is implemented in `quatrex` through a
Hilbert transform. The retarded scattering self-energy is given by

$$
\mathbf{\Sigma}^{R}(E) = \frac{1}{2} \left [\mathbf{\Sigma}^{>}(E) -
\mathbf{\Sigma}^{<}(E)\right] + \frac{1}{2\pi i} \mathcal{P}
\int_{-\infty}^{\infty} dE' \frac{\mathbf{\Sigma}^{>}(E') -
\mathbf{\Sigma}^{<}(E')}{E - E'}
$$

The same holds for polarizations. Whether the principal value integral
is actually evaluated is controlled by the
[`include_energy_renormalization`](../parameters/coulomb_screening.md#include_energy_renormalization)
parameter.

With $\mathbf{\Sigma}^{\lessgtr}(E)$ and $\mathbf{\Sigma}^{R}(E)$ in
hand, the loop closes: they enter the Dyson and Keldysh equations above
to give updated Green's functions, from which the self-energies are
recomputed. The following sections describe how
$\mathbf{\Sigma}^{\lessgtr}(E)$ itself is obtained for the interaction
`quatrex` currently supports.

## Interacting Systems

### Screened Coulomb Interaction

The Coulomb interaction is the longitudinal part of the electromagnetic
interaction. As such it has an analytic (bare) form \mathbf{V} in
vacuum. Screening by the electrons in the device modifies this bare
interaction into the *screened* Coulomb interaction $\mathbf{W}$,
obtained from its own Dyson/Keldysh pair:

$$
\begin{align}
\left[ \mathbf{I} - \mathbf{V}\mathbf{P}^R(E) \right] \mathbf{W}^R(E) =
\mathbf{V} \\
\mathbf{W}^{\lessgtr}(E) = \mathbf{W}^R(E) \mathbf{P}^{\lessgtr}(E)
\mathbf{W}^A(E)
\end{align}
$$

where $\mathbf{P}^{\lessgtr}(E)$ is the electronic polarization,
constructed as

$$
\mathbf{P}^{\lessgtr}(E) = -i \mathbf{G}^{\lessgtr}(E)
$$

and the retarded polarization is obtained from the lesser/greater parts
through the Kramers-Kronig relation, as described above.

$\mathbf{W}^{\lessgtr}(E)$ then enters the GW self-energy

$$
\mathbf{\Sigma}^{\lessgtr}_{GW}(E)
$$

In practice, computing $\mathbf{W}^R$ explicitly is unnecessary: to
obtain the GW self-energy, `quatrex` only needs
$\mathbf{W}^{\lessgtr}(E)$, and it solves for this directly by rewriting
the Keldysh equation for $\mathbf{W}$ in terms of an effective
lesser/greater polarization $\mathbf{L}^{\lessgtr}(E)$:

$$
\mathbf{W}^{\lessgtr}(E) = \left[ \mathbf{I} - \mathbf{V}\mathbf{P}^R(E)
\right]^{-1} \underbrace{\mathbf{V} \mathbf{P}^{\lessgtr}(E)
\mathbf{V}^{\dagger}}_{\mathbf{L}^{\lessgtr}(E)} \left[ \mathbf{I} -
\mathbf{V}\mathbf{P}^R(E) \right]^{-\dagger}
$$

### Phonons

Electron-phonon scattering is treated more simply: `quatrex` currently
supports only a single, dispersionless optical phonon mode (a
"monochromatic" approximation, rather than a full phonon dispersion).
The resulting phonon self-energy is

$$
\mathbf{\Sigma}^{\lessgtr}_{ph}(E) = \left[ (N_{ph}+1)
\mathbf{G}^{\lessgtr}(E \pm \hbar\omega_{ph}) + N_{ph}
\mathbf{G}^{\lessgtr}(E \mp \hbar\omega_{ph}) \right].
$$

where $\hbar\omega_{ph}$ is the phonon energy and $N_{ph} =
\left[\exp\left(\hbar\omega_{ph}/k_BT\right) - 1\right]^{-1}$ is its
Bose-Einstein occupancy at temperature $T$. The upper sign corresponds
to phonon absorption and the lower to emission, so that
$\mathbf{\Sigma}^{\lessgtr}_{ph}(E)$ combines both processes at each
energy.

# Electrostatics

!!! Note "Work in Progress"
    This section is still under construction. Please check back later for
    updates.

Solving the Poisson equation self-consistently with the Schrödinger
equation is a key part of the NEGF formalism. The electrostatic
potential is computed from the excess charge density, which is in turn
computed from the Green's functions.

$$
\nabla^{2} \phi(\mathbf{r}) =
-\frac{\rho(\mathbf{r})}{\varepsilon(\mathbf{r})}
$$

$$
n(\mathbf{r}) \equiv - 2_\mathrm{spin}
\int_{E_{CNL}(\mathbf{r})}^\infty  \frac{dE}{2\pi} G^{<}(\mathbf{r},
\mathbf{r}, E)
$$

$$
p(\mathbf{r}) \equiv 2_\mathrm{spin}
\int_{-\infty}^{E_{CNL}(\mathbf{r})}  \frac{dE}{2\pi}
G^{>}(\mathbf{r}, \mathbf{r}, E)
$$

<!-- NOTE: Included as snippet for dynamic coloring -->
--8<-- "docs/assets/images/electrostatics/excess_charge.svg"

/// figure-caption | #excess-charge
Illustration of excess electron and hole charge densities around the
band gap of a device with a potential drop along it's transport
direction $\mathbf{r}$. The charge neutrality level
$E_{CNL}(\mathbf{r})$ is shown in the middle of the band gap in green.
///

## Contact Chemical Potentials

- Write down how the Fermi level is computed via excess charge
  minimization, taking into accout the doping
- How do Fermi levels and chemical potentials relate to each other? How
  do they relate to the electrostatic potential?



## Connecting localized orbitals to real-space charge densities


- Orbital Projection

$$
G_{mn}(E) = \int d^{3}\mathbf{r} \int  d^{3}\mathbf{r'}
\, \psi^{*}_{m}(\mathbf{r})G(\mathbf{r}, \mathbf{r'};
E) \psi_{n}(\mathbf{r'})
$$

- Real-Space Projection

$$
G(\mathbf{r}, \mathbf{r'}; E) = \sum_{m,n} \psi_{m}(\mathbf{r})
G_{mn}(E) \psi^{*}_{n}(\mathbf{r'})
$$

- Mulliken Charge Projection

$$
G(\mathbf{r}, \mathbf{r'}; E) \approx \sum_{m,n} \delta(\mathbf{r} -
\mathbf{R}_{m}) G_{mn}(E) \delta(\mathbf{r'} - \mathbf{R}_{n}) = \sum_{n} G_{nn}
(E) \delta(\mathbf{r} - \mathbf{R}_{n})
$$

### Finite Element Discretization

### Multifreedom Constraints

### Dirichlet Boundary Conditions

- Gates are a bit special, involves electron affinity and metal work function

$$
\phi_\mathrm{gate} = -V_\mathrm{gate} + \Phi_\mathrm{gate} -
\chi_\mathrm{channel} - (E_\mathrm{CB} - E_\mathrm{F})
$$

## Newton-Raphson Method

$$
F[\phi(\mathbf{r})] \equiv \nabla^{2}\phi(\mathbf{r}) +
\frac{\rho[\phi(\mathbf{r})]}{\varepsilon(\mathbf{r})}
$$

$$
\begin{aligned}
    \phi_{n+1}(\mathbf{r}) &= \phi_{n}(\mathbf{r}) -
    F[\phi_{n}(\mathbf{r})] \left( \frac{\delta
    F[\phi_{n}(\mathbf{r})]}{\delta \phi_{n}(\mathbf{r})} \right) ^{-1}
    \\
    &= \phi_{n}(\mathbf{r}) - \underbrace{ F[\phi_{n}(\mathbf{r})] \left(
    \nabla^{2} + \frac{1}{\varepsilon(\mathbf{r})} \frac{\delta
    \rho[\phi_{n}(\mathbf{r})]}{\delta \phi_{n}(\mathbf{r})}\right) ^{-1}
    }_{ \Delta \phi_{n}(\mathbf{r}) }
\end{aligned}
$$

### Density Response Models

- [Notes on Fermi-Dirac Integrals](https://arxiv.org/abs/0811.0116)

<!-- NOTE: Included as snippet for dynamic coloring -->
--8<-- "docs/assets/images/electrostatics/density_model.svg"

/// figure-caption | #density-model
Illustration showing the relationship between the charge neutrality
level, electrostatic potential, and the charge density. The electronic
density of states is approximated by a single effective parabolic band,
and the charge density is computed by evaluating a Fermi-Dirac integral
of order $k$ (depends on system dimensionality).
///

$$
n(\mathbf{r}) = N_{ND}(\mathbf{r}) \mathcal{F}_{k}(\eta_{n}(\mathbf{r}))
$$

- We need to estimate a charge-neutrality level (CNL) for which we need
  a inverse Fermi-Dirac integral

$$
\rho[\phi(\mathbf{r})] = \mathcal{F}_{k}(\eta[\phi(\mathbf{r})])
\longleftrightarrow E_{CNL}(\mathbf{r}) = \mathcal{F}_{k}^{-1}(u[\rho(\mathbf{r})])
$$

- To model the charge density itself, we need to evaluate Fermi-Dirac
  integrals.

$$
\rho[\phi(\mathbf{r})] \sim \mathcal{F}_{k}(\eta[\phi(\mathbf{r})])
$$

- To model the charge density response to a change in the electrostatic
  potential, we need to evaluate the derivative of the Fermi-Dirac
  integral, which is another Fermi-Dirac integral of order $k-1$, where
  $k$ is the order of the original Fermi-Dirac integral (depends on
  system dimensionalty).

$$
\frac{\delta \rho[\phi(\mathbf{r})]}{\delta \phi(\mathbf{r})} \sim
\mathcal{F}_{k-1}(\eta[\phi(\mathbf{r})])
$$


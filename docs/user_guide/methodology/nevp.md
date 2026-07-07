# Polynomial Eigenvalue Problem Solver

$$
\begin{equation}
\sum \limits_{n=-1}^{+1} \lambda^{n} \mathbf{m}_{n} \mathbf{v} = 0
\label{eq:poly_eig}
\end{equation}
$$

A polynomial eigenvalue problem needs to be solved when using the
`spectral` methods for the open boundary conditions. The goal is to find
the eigenvalues $\lambda$ and eigenvectors $\mathbf{v}$ that satisfy
Equation $\ref{eq:poly_eig}$.

!!! info "Eigenvalue Problem Solver"
    The method for the polynomial eigenvalue problem can be set through
    the parameter [`nevp_solver`](../parameters/obc.md#nevp_solver).

## Periodicity

$$
\begin{equation}
\sum \limits_{n=-b}^{+b} \hat{\lambda}^{n} \hat{\mathbf{m}}_{n} \hat{\mathbf{v}} = 0
\label{eq:poly_eig_transport}
\end{equation}
$$

Additional periodicity properties of the contact can used to reduce the
complexity of the problem. Periodicity in transport direction leads to a
smaller polynomial eigenvalue problem with a higher degree as in
Equation $\ref{eq:poly_eig_transport}$ where $b$ corresponds to the
number of periodic repetitions in transport direction.
$\hat{\mathbf{m}}$ are subblocks of the system matrix that correspond to
the unit cell. The full eigenvectors can be reconstructed from
$\hat{\mathbf{v}}$ through Equation $\ref{eq:poly_eig_multi}$ while the
eigenvalues can be obtained from $\lambda = \hat{\lambda}^{b}$.
Currently, the periodicity has to be manually set through the parameter
[`block_sections`](../parameters/obc.md#block_sections) for NEGF
simulation while for QTBM it is automatically determined.

$$
\begin{equation}
\mathbf{v} = \begin{bmatrix} 
\hat{\mathbf{v}} \\
\lambda \hat{\mathbf{v}} \\
\lambda^{2} \hat{\mathbf{v}} \\
\vdots
\end{bmatrix}
\label{eq:poly_eig_multi}
\end{equation}
$$

Potentially, periodicity in non-transport directions can also be
exploited to further reduce the problem size. It would lead to multiple,
but smaller problems with the same degree. This is currently a feature
in development and will be further ellaborated on after full
integration.

## Linearization

Linearization of the polynomial eigenvalue problem is the simplest
solution method. There are many ways to linearize the problem, but we
currently implement the method described in [^1]. This has one advantage
that the resulting system is normal ($\mathbf{A}\vec{x} = \lambda
\vec{x}$) and not general ($\mathbf{A}\vec{x} = \lambda \mathbf{B}
\vec{x}$) which is computationally more efficient. After linearization,
the problem can be solved using the standard BLAS `geev` function.

!!! info "EIG Best Performance"
    NVIDIA has an optimized routine for the eigenvalue solving. To use
    this routine, the
    [`eig_compute_location`](../parameters/nevp.md#eig_compute_location)
    parameter should be set to `cupy`. NOTE: This configuration will be
    refactored and automatically the best option will be determined.

[^1]: Brück, Sascha. Ab-initio quantum transport simulations for
    nanoelectronic devices. Diss. ETH Zurich, 2017.

## Contour Integral

Instead of linearization, contour integral methods can be used to solve
the polynomial eigenvalue problem. The idea is to use a contour integral
to project the system onto a subspace and then solve a smaller
eigenvalue problem. These methods can be more efficient, but require
more complex implementation and parameter tuning. Many different contour
integral methods exist, but we currently support Beyn's method, but we
plan to provide more methods in the future.

![image](../../assets/images/obc/contour.svg)
/// caption
Figure 1: Annular contour used in the contour integral method.
///

As mentioned in [`obc`](obc.md) only the reflected modes contribute to
$\mathbf{g}^R$. The reflected modes separate into two groups,
propagating and decaying modes. Propagating modes have a magnitude of 1
and are located on the unit circle, while decaying modes have a
magnitude larger than 1. Thus, an annulus around the origin is
sufficient to capture all relevant eigenvalues which is shown in Figure
1. The anulus should be chosen with an inner radius slightly smaller
than 1 and a "large" outer radius. Choosing the outer radius too large
can lead to the contour method not converging, while choosing it too
small can lead to missing relevant eigenvalues. Further, the number of
quadrature points and the subspace guess size needs to be chosen. Both
parameters are not trivial to choose and a method to determine them
automatically is currently in development. 

!!! warning "Parameter Selection"
    Contour integral methods require careful selection of parameters.
    Thus, the methods are currently only recommended for advanced users.

### Beyn's Method

Beyn's method is a single iteration contour integral method. It is
derived from the first and second moment together. The method is
described in [^2] and is implemented in `quatrex` as `beyn`. The method
consists of the following steps:

- Compute the contour integral by evaluating linear systems at each
  quadrature point.
- Building the projector matrices from the contour integral using either
  a QR or SVD decomposition.
- Projecting the original system onto the subspace.
- Solving the reduced eigenvalue problem.
- Reconstructing the eigenvectors.

[^2]: Beyn, Wolf-Jürgen. "An integral method for solving nonlinear
    eigenvalue problems." Linear Algebra and its Applications 436.10
    (2012): 3839-3863.

!!! info "Parameter Selection"
    With the parameter [`use_qr`](../parameters/nevp.md#use_qr) the user
    can choose between a QR or SVD decomposition. The QR decomposition
    is faster, but leads to the contour algorithm being less robust.

## FEAST and FEAST Variants
TODO, describe them after including them in `quatrex`.

# Eigenvalue Filter

As previously mentioned, only propagating and decaying modes are needed.
Further, the modes should decay and propagate away from the device. 

A first filtering step is done for all the modes where it is checked
that the residual is smaller than a threshold (see
[`residual_tolerance`](../parameters/obc.md#residual_tolerance) and
[`residual_normalization`](../parameters/obc.md#residual_normalization)).
This ensures that the modes are indeed eigenmodes of the system. The
remaining modes are then filtered into decaying and propagating modes.

Decaying modes are filtered by three criteria:

- The imaginary part of the wavevector should be negative and smaller
  than a threshold (see [`min_decay`](../parameters/obc.md#min_decay)).
  This ensures that the mode decays away from the device.
- Slow decaying modes that are not clearly decaying are included. (see
  [`eta_decay`](../parameters/obc.md#eta_decay) and
  [`min_propagation`](../parameters/obc.md#min_propagation))
- Lastly, extra very strong decaying modes are filtered out for
  robustness. This is done by setting an upper threshold (see
  [`max_decay`](../parameters/obc.md#max_decay)) for the imaginary part
  of the wavevector.

Propagating modes are filtered by two criteria:

- The absolute value of the imaginary part of the wavevector should be
  smaller than a threshold (see
  [`min_decay`](../parameters/obc.md#min_decay)). This ensures that the
  mode is propagating.
- The group velocity ($\frac{dE}{dk}$) should be large enough (see
  [`min_propagation`](../parameters/obc.md#min_propagation)) while its
  real part should be negative. For propagating modes, the imaginary
  part of the group velocity should be zero.

!!! warning "Parameter Selection"
    Manually selecting the filtering parameters is complicated. Thus,
    only advanced users should tune them. We are working on an automatic
    selection of the parameters that will be available in a future
    release.

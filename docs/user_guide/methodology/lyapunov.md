# Lyapunov Problem

$$
\begin{equation}
\mathbf{w}^{\lessgtr} = \mathbf{q}^{\lessgtr} − \mathbf{a}\mathbf{w}^{\lessgtr}\mathbf{a}^{H}
\label{eq:lyapunov}
\end{equation}
$$

As mentioned in [`obc`](obc.md), the Lyapunov Equation
$\ref{eq:lyapunov}$ needs to be solved to compute
$\mathbf{\Sigma}^{\lessgtr}_{obc}$ (also called
$\mathbf{L}^{\lessgtr}_{obc}$) for the screened Coulomb system. 

## Derivation

!!! TODO
    include derivation of the Lyapunov equation from the derivation of
    RGF with a non-identity right hand side matrix.

## Sparsity Reduction

The Lyapunov problem can be reduced in size by exploiting the sparsity
of the matrices $\mathbf{a}$. Either zero columns or rows of the matrix
$\mathbf{a}$ can be removed. This can lead to significant speedups for
large systems with sparse matrices. The sparsity reduction is controlled
through the parameter
[`reduce_sparsity`](../parameters/lyapunov.md#reduce_sparsity). By
default, it is enabled, but it is assumed that the sparsity of the
matrix $\mathbf{a}$ can change throughout the simulation. There is
currently a "bug" in the assumption of constant sparsity and the
parameter
[`assume_constant_sparsity`](../parameters/lyapunov.md#assume_constant_sparsity)
should NOT be set to `true`.

# Solution Approaches

Similar to the solution of the fixed point problem for the retarded
boundary conditions, both iterative and direct methods can be used to
solve the Lyapunov equation. Similar considerations apply to the choice
of method. The iterative method can be more memory efficient, but can
also suffer from convergence issues. Thus, the choice of method depends
on the well-posedness of the problem and the available computational
resources. For the Lyapunov problem, convergence properties are known in
the literature. Iterative methods are stable when the magnitude of the
eigenvalues of the matrix $\mathbf{a}$ is less than one [^1].

!!! info "Algorithm Selection"
    The method for the Lyapunov problem can be set through the parameter
    [`algorithm`](../parameters/lyapunov.md#algorithm) inside
    [`lyapunov`](../parameters/lyapunov.md).

[^1]: Poloni, Federico. "Iterative and doubling algorithms for
    Riccati‐type matrix equations: A comparative introduction."
    GAMM‐Mitteilungen 43.4 (2020): e202000018.

### Iterative

#### Fixed-Point Iterations

$$
\begin{equation}
\mathbf{w}^{\lessgtr}_{n+1} = \mathbf{q}^{\lessgtr} − \mathbf{a}\mathbf{w}^{\lessgtr}_{n}\mathbf{a}^{H}
\label{eq:lyapunov_iterative}
\end{equation}
$$

The linearly convergent fixed-point iteration method is the simplest
iterative method to solve the Lyapunov problem. The convergence of the
method depends on the spectral radius of the matrix $\mathbf{a}$, which
is defined as the largest absolute value of its eigenvalues. If the
spectral radius is greater than or equal to one, the method may diverge. 

As for $\mathbf{g}^R$, simple fixed-point iterations are not exposed to
the user, but are used as a refinement step in both the direct method
and the memoizer. From experience, the iterative methods can converge
well for the Lyapunov problem, except that spurious energies can lead to
divergence. Thus, the iterative methods are not recommended for general
use.

#### Squared Smith

Similar to Sancho-Rubio, an exponentially convergent iterative method
can be derived. This doubling method is also called squared smith method
and is described in [^1]. As the fixed-point iterations, the method
convergence depends on the spectral radius of the matrix $\mathbf{a}$.

### Direct

#### Spectral Method

Solving the Lyapunov problem directly can be done by eigenvalue
decomposing the matrix $\mathbf{a}$ and then solving the Lyapunov
problem in the eigenbasis. We call this the spectral method. The method
is derived in the following:

$$
\begin{align}
\mathbf{a} &= \mathbf{V} \mathbf{\Lambda} \mathbf{V}^{-1} \\
\mathbf{w}^{\lessgtr}_{n+1} &= \mathbf{q}^{\lessgtr} − \mathbf{V} \mathbf{\Lambda} \mathbf{V}^{-1}\mathbf{w}^{\lessgtr}_{n} \mathbf{V}^{-H} \mathbf{\Lambda} \mathbf{V}^{H}\\
\hat{\mathbf{w}} &= \mathbf{V}^{-1} \mathbf{w}^{\lessgtr} \mathbf{V}^{-H} \\
\hat{\mathbf{q}} &= \mathbf{V}^{-1} \mathbf{q}^{\lessgtr} \mathbf{V}^{-H} \\
\hat{\mathbf{w}} &= \hat{\mathbf{q}} − \mathbf{\Lambda} \hat{\mathbf{w}} \mathbf{\Lambda} \\
\hat{\mathbf{w}}_{ij} &= \frac{\hat{\mathbf{q}}_{ij}}{1 - \lambda_i \lambda_j^*} \label{eq:spectral_lyapunov}\\
\mathbf{w}^{\lessgtr} &= \mathbf{V} \hat{\mathbf{w}} \mathbf{V}^{H}
\end{align}
$$

The main idea is to transform the Lyapunov problem into the eigenbasis
of the matrix $\mathbf{a}$. In this basis, the Lyapunov problem can be
solved element-wise as in Equation $\ref{eq:spectral_lyapunov}$. The
solution can then be transformed back to the original basis. The method
is efficient, but requires the eigenvalue decomposition of the matrix
$\mathbf{a}$ which can be computationally expensive for large matrices.
The matrix $\mathbf{a}$ has generally no symmetry properties, thus
LAPACK `geev` has to be used.

!!! info "EIG Best Performance"
    NVIDIA has an optimized routine for the eigenvalue solving. To use
    this routine, the
    [`eig_compute_location`](../parameters/nevp.md#eig_compute_location)
    parameter should be set to `cupy`. NOTE: This configuration will be
    refactored and automatically the best option will be determined.

As we observed some stability issues with this method, we still do a
fixed-point iteration refinement step after the spectral method. The
spectral method is currently the default method for the Lyapunov
problem, but potentially the Schur method can be more stable.

### Memoization

See [`obc`](obc.md) for a detailed description of the memoization
method. The memoization method can be used to solve the Lyapunov problem
as well and its implementation is shared with the memoization method for
the retarded boundary conditions.

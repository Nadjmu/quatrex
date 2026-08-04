"""
Smallest-magnitude eigenvalue of a large sparse matrix on CPU.

Uses inverse power iteration: sparse LU factorize once (scipy splu),
then repeatedly solve + normalize. Converges to the eigenvalue closest
to 0. Works for general (non-Hermitian) matrices; eigenvalue is complex
if the matrix is complex.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spilu, LinearOperator, gmres


def smallest_eigenvalue(A_csr, tol=1e-8, maxiter=100, gmres_tol=1e-10):
    n = A_csr.shape[0]
    A_csc = A_csr.tocsc()

    # Incomplete LU (bounded memory via drop_tol) used as a GMRES
    # preconditioner, instead of exact splu which can blow up on fill-in.
    ilu = spilu(A_csc, drop_tol=1e-5, fill_factor=10)
    M = LinearOperator(A_csc.shape, ilu.solve)

    x = np.random.rand(n).astype(A_csr.dtype)
    x /= np.linalg.norm(x)

    lam_old = 0.0 + 0.0j
    for _ in range(maxiter):
        y, info = gmres(A_csr, x, M=M, rtol=gmres_tol, atol=0)
        if info != 0:
            print(f"Warning: GMRES did not fully converge (info={info})")
        y /= np.linalg.norm(y)
        lam = complex(y.conj() @ (A_csr @ y))
        if abs(lam - lam_old) < tol * abs(lam):
            x = y
            break
        x, lam_old = y, lam

    return lam, x


def load_csr_npz(path):
    """Load a CSR matrix saved in the data/indices/indptr/shape format
    used by _save_csr_npz (not scipy's own save_npz format)."""
    with np.load(path) as f:
        return sp.csr_matrix(
            (f["data"], f["indices"], f["indptr"]),
            shape=tuple(f["shape"]),
        )


if __name__ == "__main__":
    A_csr = load_csr_npz("/scratch/yimili/matrices/dev_12_sorted_BENCH/M_E_0.npz")

    eigval, _ = smallest_eigenvalue(A_csr)
    print(f"Smallest eigenvalue: {eigval:.6e}")
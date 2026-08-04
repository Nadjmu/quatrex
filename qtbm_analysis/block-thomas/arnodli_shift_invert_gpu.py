"""
Smallest-magnitude eigenvalue of a large sparse matrix on GPU.

Uses inverse power iteration: sparse LU factorize once (cupyx splu),
then repeatedly solve + normalize. Converges to the eigenvalue closest
to 0. Works for general (non-Hermitian) matrices; eigenvalue is complex
if the matrix is complex.
"""

import cupy as cp
import cupyx.scipy.sparse as cusparse
from cupyx.scipy.sparse.linalg import splu


def smallest_eigenvalue(A_csr, tol=1e-8, maxiter=100):
    n = A_csr.shape[0]
    lu = splu(A_csr.tocsc())

    x = cp.random.rand(n).astype(A_csr.dtype)
    x /= cp.linalg.norm(x)

    lam_old = 0.0 + 0.0j
    for _ in range(maxiter):
        y = lu.solve(x)
        y /= cp.linalg.norm(y)
        lam = complex(y.conj() @ (A_csr @ y))
        if abs(lam - lam_old) < tol * abs(lam):
            x = y
            break
        x, lam_old = y, lam

    return lam, x


def load_csr_npz(path):
    """Load a CSR matrix saved in the data/indices/indptr/shape format
    used by _save_csr_npz (not scipy's own save_npz format)."""
    import numpy as np
    import scipy.sparse as sp

    with np.load(path) as f:
        return sp.csr_matrix(
            (f["data"], f["indices"], f["indptr"]),
            shape=tuple(f["shape"]),
        )


if __name__ == "__main__":
    host_csr = load_csr_npz("/scratch/yimili/matrices/dev_12_sorted_BENCH/M_E_0.npz")
    A_gpu = cusparse.csr_matrix(host_csr)

    eigval, _ = smallest_eigenvalue(A_gpu)
    print(f"Smallest eigenvalue: {eigval:.6e}")
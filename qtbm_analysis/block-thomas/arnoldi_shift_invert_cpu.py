#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spilu, LinearOperator, gmres

MATRIX_PATH = "/scratch/yimili/matrices/dev_12_sorted_BENCH/M_E_0.npz"


def load_csr_npz(path):
    with np.load(path) as f:
        return sp.csr_matrix(
            (f["data"], f["indices"], f["indptr"]),
            shape=tuple(f["shape"]),
        )


def smallest_eigenvalue(
    A,
    tol=1e-8,
    maxiter=100,
    gmres_tol=1e-10,
):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Use complex arithmetic.
    A = A.astype(np.complex128).tocsr()
    A_csc = A.tocsc()
    n = A.shape[0]

    print("[factor] Computing ILU...", flush=True)
    ilu = spilu(
        A_csc,
        drop_tol=1e-5,
        fill_factor=10,
    )

    M = LinearOperator(
        A.shape,
        matvec=ilu.solve,
        dtype=np.complex128,
    )

    rng = np.random.default_rng(1234)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x /= np.linalg.norm(x)

    for iteration in range(1, maxiter + 1):
        y, info = gmres(
            A,
            x,
            M=M,
            rtol=gmres_tol,
            atol=0.0,
            restart=50,
            maxiter=200,
        )

        if info != 0:
            raise RuntimeError(
                f"GMRES failed at iteration {iteration}: info={info}"
            )

        y_norm = np.linalg.norm(y)
        if y_norm == 0 or not np.isfinite(y_norm):
            raise RuntimeError("Invalid vector returned by GMRES")

        x = y / y_norm

        Ax = A @ x
        eigenvalue = np.vdot(x, Ax) / np.vdot(x, x)

        residual = np.linalg.norm(Ax - eigenvalue * x)
        scale = np.linalg.norm(Ax) + abs(eigenvalue) * np.linalg.norm(x)
        relative_residual = residual / max(scale, np.finfo(float).tiny)

        print(
            f"[{iteration:3d}] "
            f"lambda={eigenvalue.real:.12e} "
            f"{eigenvalue.imag:+.12e}j, "
            f"residual={relative_residual:.3e}",
            flush=True,
        )

        if relative_residual < tol:
            return eigenvalue, x

    raise RuntimeError("Inverse iteration did not converge")


def main():
    print(f"[load] {MATRIX_PATH}", flush=True)
    A = load_csr_npz(MATRIX_PATH)

    print(
        f"[load] shape={A.shape}, nnz={A.nnz}, dtype={A.dtype}",
        flush=True,
    )

    eigenvalue, _ = smallest_eigenvalue(A)

    print(
        "\nSmallest-magnitude eigenvalue: "
        f"{eigenvalue.real:.12e} "
        f"{eigenvalue.imag:+.12e}j"
    )


if __name__ == "__main__":
    main()
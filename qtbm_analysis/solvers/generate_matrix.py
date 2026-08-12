"""
Generation of the shared synthetic test system.

Purpose
-------
Writes one random complex sparse matrix, right-hand side and exact solution to
disk, so that every solver script loads the same system rather than generating
its own instance. Without this, timing and accuracy figures from different
scripts would refer to different matrices and could not be compared.

Algorithm
---------
A sparse matrix with the requested density is filled with standard complex
normal entries, then made strictly diagonally dominant by adding to each
diagonal entry the absolute row sum plus one. Diagonal dominance guarantees
that LU without pivoting is stable, so the system tests factorization
throughput rather than the pivoting strategy. The right-hand side is formed as
b = A x_true from a known x_true, which makes the forward error directly
measurable.

Output
------
For each precision, in OUTPUT_DIR:

    A_double.npz, b_double.npy, x_true_double.npy       complex128
    A_single.npz, b_single.npy, x_true_single.npy       complex64

Usage
-----
    python generate_matrix.py
"""

import os
import numpy as np
import scipy.sparse as sp

import cli

OUTPUT_DIR = str(cli.RANDOM_DIR)


def build_test_system(n=2000, density=0.002, seed=0, dtype=np.complex128):
    """
    Random sparse complex system, made diagonally dominant.

    Returns (A, b, x_true) with b = A x_true, so the forward error of any
    solver applied to this system is directly measurable.
    """
    rng = np.random.default_rng(seed)

    nnz = int(n * n * density)
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    vals = (rng.standard_normal(nnz) + 1j * rng.standard_normal(nnz)).astype(dtype)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=dtype)

    # Strict diagonal dominance makes LU stable without pivoting, so the
    # system measures factorization throughput and not the pivoting strategy.
    row_abs_sum = np.abs(A).sum(axis=1).A.flatten()
    A = A + sp.diags(row_abs_sum + 1.0)

    x_true = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
    b = A @ x_true
    return A.tocsc(), b, x_true


def save_system(A, b, x_true, tag):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sp.save_npz(os.path.join(OUTPUT_DIR, f"A_{tag}.npz"), A)
    np.save(os.path.join(OUTPUT_DIR, f"b_{tag}.npy"), b)
    np.save(os.path.join(OUTPUT_DIR, f"x_true_{tag}.npy"), x_true)
    print(f"Saved {tag}-precision system: A{A.shape}, nnz={A.nnz}, dtype={A.dtype}")


if __name__ == "__main__":
    for tag, dtype in [("double", np.complex128), ("single", np.complex64)]:
        A, b, x_true = build_test_system(dtype=dtype)
        save_system(A, b, x_true, tag)

    print(f"\nAll matrices saved to ./{OUTPUT_DIR}/")
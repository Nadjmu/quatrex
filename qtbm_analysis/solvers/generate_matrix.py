"""
Generates a complex sparse test matrix (+ RHS + true solution) and saves it
to disk, so every solver script loads the *same* matrix instead of each
generating its own random instance. This makes timing/accuracy comparisons
across solvers fair.

Produces, for each precision:
    matrix_data/A_double.npz     (complex128 sparse matrix, CSC)
    matrix_data/b_double.npy     (complex128 right-hand side)
    matrix_data/x_true_double.npy
    matrix_data/A_single.npz     (complex64 sparse matrix, CSC)
    matrix_data/b_single.npy     (complex64 right-hand side)
    matrix_data/x_true_single.npy

Usage:
    python generate_matrix.py
"""

import os
import numpy as np
import scipy.sparse as sp

OUTPUT_DIR = "/scratch/yimili/random"


def build_test_system(n=2000, density=0.002, seed=0, dtype=np.complex128):
    """Random sparse complex matrix, diagonally dominant for numerical safety."""
    rng = np.random.default_rng(seed)

    nnz = int(n * n * density)
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    vals = (rng.standard_normal(nnz) + 1j * rng.standard_normal(nnz)).astype(dtype)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=dtype)

    # Diagonal dominance keeps this a clean factorization-performance test
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
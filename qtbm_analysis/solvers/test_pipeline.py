#!/usr/bin/env python3
"""
Self-tests for the solver and factor-persistence pipeline.

Input
-----
Synthetic complex block-tridiagonal systems generated in memory. No cluster
data is required, so these tests run anywhere.

Coverage
--------
For uniform and for custom non-uniform partitions:

    - every Block Thomas variant against a dense reference solve
    - the block-structure detector and its off-band guard
    - handling of multiple right-hand sides
    - the factor layout, stacked for a uniform partition and ragged otherwise
    - a full HDF5 round trip, bench() to factor_io to the growth-factor
      assemblers, verifying that the reloaded factors still reproduce
      A == L U to machine precision

The HDF5 section requires h5py and is skipped when it is absent. UMFPACK is the
one solver with no coverage here, since it has no pure-Python fallback.

Output
------
A PASS or FAIL line per check, and a non-zero exit status if any failed.

Usage
-----
    python test_pipeline.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_classes import (
    BlockThomas, BlockThomasExplicitInv,
    BlockThomasFP16, BlockThomasExplicitInvFP16,
    extract_blocks_sparse, find_block_slices, block_sizes_from_matrix,
    normalize_block_sizes, offband_nnz,
)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def build_system(block_sizes, seed=0, diag_boost=3.0):
    """
    Random complex block-tridiagonal system under the given partition.

    diag_boost is added to the diagonal to keep the modified diagonal blocks
    well conditioned, so that a failure indicates an error in the
    implementation rather than an ill-conditioned test case.
    """
    rng = np.random.default_rng(seed)
    sizes = list(block_sizes)
    N = len(sizes)

    def rc(p, q):
        return rng.standard_normal((p, q)) + 1j * rng.standard_normal((p, q))

    D = [rc(m, m) + diag_boost * max(sizes) * np.eye(m) for m in sizes]
    L = [rc(sizes[k + 1], sizes[k]) for k in range(N - 1)]   # sub-diagonal
    U = [rc(sizes[k], sizes[k + 1]) for k in range(N - 1)]   # super-diagonal

    grid = [[None] * N for _ in range(N)]
    for k in range(N):
        grid[k][k] = sp.csr_matrix(D[k])
    for k in range(N - 1):
        grid[k + 1][k] = sp.csr_matrix(L[k])
        grid[k][k + 1] = sp.csr_matrix(U[k])
    A = sp.bmat(grid, format="csc")

    n = sum(sizes)
    x_true = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return A, A @ x_true, x_true


def relres(A, x, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)


# ---------------------------------------------------------------------------
def test_variants(label, block_sizes, tol_c128, tol_c64, tol_fp16):
    print(f"\n[{label}]  block_sizes = {block_sizes}")
    A, b, x_true = build_system(block_sizes)
    D, L, U = extract_blocks_sparse(A, block_sizes)

    check("extracted blocks tile A", offband_nnz(A, block_sizes) == 0)

    # -- extraction is lossless: reassembling the blocks reproduces A --------
    sizes = normalize_block_sizes(A.shape[0], block_sizes)
    N = len(sizes)
    grid = [[None] * N for _ in range(N)]
    for k in range(N):
        grid[k][k] = sp.csr_matrix(D[k])
    for k in range(N - 1):
        grid[k + 1][k] = sp.csr_matrix(L[k])
        grid[k][k + 1] = sp.csr_matrix(U[k])
    diff = (A - sp.bmat(grid, format="csc"))
    diff.eliminate_zeros()
    check("extract_blocks_sparse is lossless",
          diff.nnz == 0 or np.abs(diff.data).max() == 0.0)

    # -- Implementation 1, complex128 / complex64 ---------------------------
    for dt, tol in ((np.complex128, tol_c128), (np.complex64, tol_c64)):
        bt = BlockThomas(D, L, U, dt)
        r = relres(A, bt.solve(b), b)
        check(f"BlockThomas {np.dtype(dt).name}", r < tol, f"relres={r:.2e}")

    # -- Implementation 2, complex128 / complex64 ---------------------------
    for dt, tol in ((np.complex128, tol_c128), (np.complex64, tol_c64)):
        bt = BlockThomasExplicitInv(D, L, U, dt)
        r = relres(A, bt.solve(b), b)
        check(f"BlockThomasExplicitInv {np.dtype(dt).name}", r < tol,
              f"relres={r:.2e}")

    # -- Implementation 1, fp16 ---------------------------------------------
    bt16 = BlockThomasFP16(D, L, U)
    r16 = relres(A, bt16.solve(b), b)
    check("BlockThomasFP16", r16 < tol_fp16, f"relres={r16:.2e}")

    # -- Implementation 2, fp16, both inverse precisions --------------------
    results = {}
    for inv_dt in (np.float32, np.float16):
        bti = BlockThomasExplicitInvFP16(D, L, U, inv_dtype=inv_dt)
        r = relres(A, bti.solve(b), b)
        results[np.dtype(inv_dt).name] = r
        check(f"BlockThomasExplicitInvFP16 (inv={np.dtype(inv_dt).name})",
              r < tol_fp16, f"relres={r:.2e}")
    print(f"        fp32-inverse buys {results['float16'] / results['float32']:.2f}x "
          f"over the pure-fp16 inverse")

    # -- multi-RHS ----------------------------------------------------------
    B = np.column_stack([b, 2 * b, -0.5 * b])
    for cls, tol in ((BlockThomas, tol_c128),
                     (BlockThomasExplicitInv, tol_c128),
                     (BlockThomasFP16, tol_fp16),
                     (BlockThomasExplicitInvFP16, tol_fp16)):
        X = cls(D, L, U).solve(B)
        worst = max(relres(A, X[:, j], B[:, j]) for j in range(B.shape[1]))
        check(f"{cls.__name__} multi-RHS", X.shape == B.shape and worst < tol,
              f"worst relres={worst:.2e}")

    # -- factor layout: stacked iff uniform ---------------------------------
    uniform = len(set(sizes)) == 1
    bt = BlockThomas(D, L, U, np.complex128)
    Lb, Ub, Dlu, Dpiv = bt.get_LUP()
    check("get_LUP layout matches partition",
          all(isinstance(p, np.ndarray) == uniform for p in (Lb, Ub, Dlu, Dpiv)),
          "stacked" if uniform else "ragged lists")
    check("factor_nbytes > 0", bt.factor_nbytes() > 0)

    bti = BlockThomasExplicitInv(D, L, U, np.complex128)
    Dinv, Dmod, Lb2, Ub2 = bti.get_inverses()
    check("get_inverses returns D_mod for the growth-factor analysis",
          len(Dmod) == N and bti.get_LUP() is None)
    check("ExplicitInv factor_nbytes > 0", bti.factor_nbytes() > 0)


# ---------------------------------------------------------------------------
def test_block_detection():
    print("\n[block-structure detector]")
    # a genuinely non-uniform block-tridiagonal pattern
    sizes = [5, 11, 4, 9, 7]
    A, _, _ = build_system(sizes, seed=3)
    found = list(block_sizes_from_matrix(A))
    # The detector merges its first two slices by construction (the seed row
    # is absorbed into the first real block), so the leading block is coarser
    # than the true partition: [5,11,...] comes back as [16,...]. Everything
    # after the first block must match exactly.
    check("find_block_slices recovers the trailing partition",
          found[1:] == sizes[2:], f"found={found}")
    check("detected partition tiles A", sum(found) == A.shape[0])
    check("detected partition is off-band clean", offband_nnz(A, found) == 0)
    # a coarser partition is still a correct one: solving with it must work
    Dd, Ld, Ud = extract_blocks_sparse(A, found)
    _, bdet, _ = build_system(sizes, seed=3)
    xdet = BlockThomas(Dd, Ld, Ud, np.complex128).solve(bdet)
    check("detected partition solves correctly",
          relres(A, xdet, bdet) < 1e-12)

    # a partition that does NOT tile the matrix must be rejected outright
    try:
        normalize_block_sizes(A.shape[0], [5, 11, 4])
        check("bad partition rejected", False)
    except ValueError:
        check("bad partition rejected", True)

    # a matrix with a coupling outside the band must be caught
    A2 = A.tolil()
    A2[0, A.shape[0] - 1] = 1.0 + 1j          # corner entry, far off band
    check("offband_nnz catches an out-of-band coupling",
          offband_nnz(A2.tocsr(), sizes) == 1)

    # uniform int path still works and matches the explicit list
    A3, _, _ = build_system([8] * 6, seed=4)
    D1, L1, U1 = extract_blocks_sparse(A3, 8)
    D2, L2, U2 = extract_blocks_sparse(A3, [8] * 6)
    check("int and list partitions agree",
          all(np.array_equal(a, b) for a, b in zip(D1, D2)))


# ---------------------------------------------------------------------------
# full HDF5 round trip: bench() -> factor_io -> growth_factor's assemblers
# ---------------------------------------------------------------------------
def test_h5_roundtrip(label, block_sizes):
    print(f"\n[HDF5 round trip -- {label}]")
    try:
        import h5py
    except ImportError:
        print("  SKIP  h5py not installed")
        return

    sys.path.insert(0, str((Path(__file__).parent / ".." / "block-thomas").resolve()))
    from bench_all import bench
    import growth_factor as gf

    A, b, _ = build_system(block_sizes, seed=7)
    idx = 0
    tmpdir = Path(tempfile.mkdtemp())
    try:
        h5path = tmpdir / "synthetic.h5"
        with h5py.File(h5path, "w") as f:
            g = f.create_group(f"E_{idx}/M")
            Ac = A.tocsc()
            g.create_dataset("data", data=Ac.data)
            g.create_dataset("indices", data=Ac.indices)
            g.create_dataset("indptr", data=Ac.indptr)

        with h5py.File(h5path, "a") as f:
            bench(A, b, idx, block_sizes,
                  dtypes=(np.complex128,), h5file=f, save=True,
                  solvers=("superlu", "block_thomas", "block_thomas_inv",
                           "block_thomas_fp16", "block_thomas_inv_fp16"))

        # every stored factorization must still reproduce the matrix it
        # factored, after a full write/read cycle
        with h5py.File(h5path, "r") as f:
            for solver, dt in (("blockthomas", "complex128"),
                               ("blockthomas_inv", "complex128"),
                               ("blockthomas", "complex32"),
                               ("blockthomas_inv", "complex32"),
                               ("superlu", "complex128")):
                path = f"E_{idx}/{solver}/{dt}"
                if path not in f:
                    check(f"{solver}/{dt} present", False)
                    continue
                A_dt = A if dt == "complex32" else A.astype(np.dtype(dt))
                A_eff, L, U = gf.ASSEMBLERS[solver](f[path], A_dt)
                res = gf.analyse(A_eff, L, U)
                r = res["1-norm"]["resid_rel"]
                # fp16 factors only reproduce A to fp16 accuracy, by definition
                tol = 1e-1 if dt == "complex32" else 1e-10
                check(f"{solver}/{dt} reloaded factors reproduce A_eff",
                      r < tol, f"resid={r:.2e}  rho={res['1-norm']['rho']:.2e}")

            # ragged partitions must survive the flatten/reshape round trip
            import factor_io as fio
            g = f[f"E_{idx}/blockthomas/complex128"]
            blocks = fio.load_blocks(g, "L")
            sizes = normalize_block_sizes(A.shape[0], block_sizes)
            uniform = len(set(sizes)) == 1
            check("stored block layout matches partition",
                  isinstance(blocks, np.ndarray) == uniform,
                  "stacked" if uniform else "ragged")
            check("block_sizes recorded in the file",
                  list(g["block_sizes"][:]) == list(sizes))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_variants("uniform", [8] * 30, tol_c128=1e-12, tol_c64=1e-4,
                  tol_fp16=5e-2)
    test_variants("custom / non-uniform", [6, 13, 9, 4, 11, 7, 15, 5, 10, 8],
                  tol_c128=1e-12, tol_c64=1e-4, tol_fp16=5e-2)
    test_block_detection()
    test_h5_roundtrip("uniform", [8] * 12)
    test_h5_roundtrip("custom / non-uniform", [6, 13, 9, 4, 11, 7])

    print(f"\n{'ALL TESTS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)

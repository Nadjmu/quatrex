# `solvers/` — the solver library

Everything here is importable library code. Nothing in this folder is meant to
be run as a batch job; the drivers live in [`../run_bench/`](../run_bench/) and
the analysis scripts in [`../block-thomas/`](../block-thomas/).

| file | what it is |
|---|---|
| `solver_classes.py` | every solver, behind one common interface, plus the block-partition utilities |
| `bench_all.py` | the single `bench()` implementation — one place to add a solver or fix a bug |
| `factor_io.py` | HDF5 persistence of factors and metadata, and the factor-verification helpers |
| `generate_matrix.py` | fixture generator: one shared random test system so solver comparisons are fair |
| `test_pipeline.py` | self-tests, synthetic data only — no cluster files needed |

---

## The solver contract

Every class follows the same shape, so `bench()` treats them uniformly:

```python
solver = SolverClass(A_or_blocks, dtype)   # ALL factorization work — this is what you time
x      = solver.solve(b)                   # b is (n,) or (n, nrhs); returns the same shape, on host
L,U,.. = solver.get_LUP()                  # explicit factors, or None if the solver hides them
nbytes = solver.factor_nbytes()            # memory footprint of the stored factors (best effort)
```

### What each solver exposes

| solver | precision | explicit factors | reconstruction convention |
|---|---|---|---|
| `SparseLU` (SuperLU) | c128 / c64 | L, U, perm_r, perm_c | `Pr @ A @ Pc == L @ U` |
| `UMFPACK` | **c128 only** | L, U, perm_r, perm_c, R | `Pr @ diag(1/R) @ A @ Pc == L @ U` |
| `MUMPS` | c128 / c64 | none — Fortran-side, not wrapped | — |
| `GMRES` (SciPy) | c128 / c64 | ILU preconditioner only, not an LU of A | — |
| `GMRESCuPy` | c128 / c64 | none (iterative) | — |
| `CuDSS` | c128 / c64 | permutations + `lu_nnz` only | — |
| `BlockThomas` | c128 / c64 | per-block LU + pivots | `A == L @ U` exactly |
| `BlockThomasExplicitInv` | c128 / c64 | per-block explicit inverses + `D_mod` | `A == L @ U` exactly |
| `BlockThomasFP16` | fp16 | embedded-real LU at block size `2·bs` | `s·embed(A) == L @ U` |
| `BlockThomasExplicitInvFP16` | fp16 | embedded-real inverses + scales `t` | `s·embed(A) == L @ U` |

`UMFPACK` raises `TypeError` rather than silently upcasting when handed a
single-precision dtype, so a single-precision batch run stays honest.

---

## The four Block Thomas variants

All four are Higham, *Accuracy and Stability of Numerical Algorithms*, Alg. 13.3,
in the two implementations the book describes:

- **Implementation 1** — diagonal blocks factorized by GEPP; wherever the
  algorithm needs `D_mod⁻¹` it does a triangular substitution against the
  stored LU. No block is ever explicitly inverted.
- **Implementation 2** — `D_mod⁻¹` is formed **explicitly**, so every solve
  stage becomes a dense matmul. More flops (~3× an LU per block) and, per
  Higham Ch. 14, generally less accurate — but it is the shape a GPU wants.

|  | Implementation 1 | Implementation 2 |
|---|---|---|
| complex | `BlockThomas` | `BlockThomasExplicitInv` |
| half | `BlockThomasFP16` | `BlockThomasExplicitInvFP16` |

### How the fp16 variants work

Complex blocks are handled by the exact real embedding
`z = a+bi → [[a,-b],[b,a]]`, so every operation is real fp16 and no `complex32`
dtype is needed. An `m×m` complex block becomes `2m×2m` real; a rectangular
coupling block `(p,q)` becomes `(2p,2q)`, so **custom partitions work
unchanged**. The backend is numpy `float16`: numpy evaluates each op in fp32
and rounds back to fp16, i.e. the same accumulate semantics as tensor cores. A
torch/GPU backend is a mechanical swap.

Three scaling mechanisms keep the arithmetic inside fp16's narrow range
(normals only span `[6.1e-5, 65504]`). All use **exact powers of two**, so they
introduce no rounding error of their own — they only move where the
computation sits in the exponent range.

1. **Global scale `s`** — brings the largest entry of the embedded `A` to
   ~1024, leaving 64× headroom. Taken over `D`, `L` *and* `U`: the coupling
   blocks can carry the largest entries, and scaling on `D` alone lets them
   overflow.
2. **Per-block inverse scales `t[k]`** *(Implementation 2 only)* — inverses
   scale like `1/|D_mod|`, i.e. **opposite** to `A`, so the same global scale
   that keeps `A` in range drives the inverses toward underflow. Each stored
   inverse gets its own scale: `G[k] = t[k] · D_mod_hat[k]⁻¹`.
3. **Matmul overflow guard** — a dense fp16 matmul overflows purely from the
   accumulation length. With entries at ~1e3 and `bs=416` (an 832-long
   embedded accumulation) the exact product is ~1e9, far past 65504, even
   though the wanted quantity is O(1). `_matmul16` rescales by a power of two
   derived from the rigorous bound `|(A@X)ᵢⱼ| ≤ ‖A‖∞ · max|X|`. This is why
   `‖L‖∞`, `‖U‖∞`, `‖G‖∞` are precomputed at construction.

### `inv_dtype` — the mixed-precision knob

`BlockThomasExplicitInvFP16(..., inv_dtype=...)` controls the precision in
which the inverse is **formed**, independently of the fp16 storage and apply:

- `np.float32` (default) — block promoted to fp32, inverted with LAPACK,
  rounded back to fp16. Explicit inversion is the least stable step in the
  algorithm, and doing it at fp16 costs accuracy no care in the sweeps
  recovers; the fp32 inversion is `O(N)` small dense inversions and does not
  change what is stored.
- `np.float16` — genuinely all-fp16 factorization. Slower and less accurate;
  provided so the cost of the fp32 inversion can be **measured** rather than
  assumed.

On the synthetic tests the fp32 inverse is worth ~1.7–2.3× in relative
residual, and Implementation 2 with an fp32 inverse comes out *more* accurate
than Implementation 1 in pure fp16 (≈5e-4 vs ≈1.1e-3). Whether that holds on
real QTBM matrices is the open question — run `../run_bench/sweep_fp16.py`.

---

## Block partitions, uniform and custom

The partition enters the pipeline at exactly **one** place:

```python
D, L, U = extract_blocks_sparse(A, block_sizes)   # int (uniform) or a sequence (custom)
```

The solvers infer their partition from the shapes of the `D` blocks they are
handed, so none of them takes a block-size argument. For a custom partition
the off-diagonal blocks are rectangular: `L[k]` is `(n[k+1], n[k])`, `U[k]` is
`(n[k], n[k+1])`.

```python
sizes = block_sizes_from_matrix(A)     # detect from the sparsity pattern
assert offband_nnz(A, sizes) == 0      # ALWAYS check this
D, L, U = extract_blocks_sparse(A, sizes)
```

> **`offband_nnz` is not optional.** A partition that cuts through a real
> coupling does not fail loudly — `extract_blocks_sparse` silently discards
> the out-of-band entries and Block Thomas returns a plausible, *wrong* `x`.
> `bench()` checks it by default (`check_blocks=True`).

`find_block_slices` only looks **forward** (largest column index per row), so
its partition is guaranteed block-tridiagonal only for structurally symmetric
matrices — which the QTBM matrices are, but nothing enforces it. It also
merges its first two slices by construction, so the leading block comes out
coarser than the true structure; harmless, just slightly more arithmetic in
block 0.

**Factor layout follows the partition.** Uniform → `get_LUP()` returns stacked
`(N, bs, bs)` arrays, byte-identical to what was always written, so existing
notebooks and h5 files keep working. Custom → ragged lists, which `factor_io`
flattens to a 1-D buffer plus a `<name>_shapes` table. Read either back with
`factor_io.load_blocks(g, name)`, never `g[name][:]`.

---

## `bench_all.bench()`

One implementation, imported by the notebook and by every driver.

```python
metrics = bench(A, B, idx, block_sizes,
                dtypes=(np.complex128, np.complex64),
                h5file=f, save=True,
                solvers=DEFAULT_SOLVERS, exclude={"gmres": {"complex64"}})
```

- Every solver runs once per dtype. The **first** dtype defines the baseline:
  `superlu` at `dtypes[0]` is what every speedup and "vs base" error refers to.
- Result keys are `<solver>_<suffix>`: `superlu_c128`, `block_thomas_inv_c64`, …
- Skips are graceful and per-`(solver, dtype)`: no UMFPACK single-precision
  build, no GPU, a missing package — each prints a skip line and continues.
- **The fp16 variants are the exception.** They are precision-fixed, so they
  run *once* per index outside the dtype loop, under the unsuffixed keys
  `block_thomas_fp16` / `block_thomas_inv_fp16` and the storage label
  `"complex32"` (not a real numpy dtype). They are **not** in
  `DEFAULT_SOLVERS` — the fp16 kernels are pure python and orders of magnitude
  slower than LAPACK, so opt in via `bench_all.FP16_SOLVERS` when you want
  accuracy data rather than timings.

## `factor_io.py`

Solver output is appended into the **material's own** HDF5 file, as a direct
sibling of each energy index's `M`/`rhs`/`Sigma`:

```
E_<idx>/superlu/<dtype>/          L, U (sparse CSC), perm_r, perm_c, x, time_fact, time_solve
E_<idx>/umfpack/<dtype>/          + R, do_recip
E_<idx>/blockthomas/<dtype>/      L, U, Dmod_lu, Dmod_piv, block_sizes, x, times
E_<idx>/blockthomas_inv/<dtype>/  L, U, Dmod_inv, Dmod, block_sizes, x, times
E_<idx>/mumps|cudss|gmres*/<dtype>/  x, times (+ whatever metadata exists)
```

Each `save_*` deletes and recreates its own `E_<idx>/<solver>/<dtype>` group,
so re-running a combination overwrites cleanly and never touches `M`, `rhs`,
`Sigma`, `spectrum`. Pass the material file opened `"a"`/`"r+"` — there is no
separate `*_LU.h5`.

## Tests

```bash
python test_pipeline.py
```

Synthetic systems only. Covers all four Block Thomas variants against dense
references at uniform *and* non-uniform partitions, multi-RHS, the block
detector and its off-band guard, and a full `bench() → factor_io →
growth_factor` HDF5 round trip checking the reloaded factors still reproduce
`A == L @ U`. Needs `h5py` for the round-trip section (skipped without it);
UMFPACK is the one path with no local coverage.

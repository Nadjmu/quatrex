# `solvers/` — the solver library

Importable library code. Nothing here is intended to be run as a batch job; the
drivers are in [`../run_bench/`](../run_bench/), the analysis scripts in
[`../block-thomas/`](../block-thomas/), and the figures in
[`../plotting/`](../plotting/).

| File | Contents |
|---|---|
| `solver_classes.py` | every solver behind one interface, plus the block-partition utilities |
| `bench_all.py` | the single `bench()` implementation |
| `factor_io.py` | HDF5 persistence of factors and metadata, and factor verification |
| `generate_matrix.py` | the shared synthetic test system |
| `test_pipeline.py` | self-tests; synthetic data only, no cluster files required |

---

## 1. The solver interface

Every class satisfies the same contract, so `bench()` treats them uniformly:

```python
solver = SolverClass(A_or_blocks, dtype)   # ALL factorization work
x      = solver.solve(b)                   # (n,) or (n, nrhs); returns on host
L,U,.. = solver.get_LUP()                  # explicit factors, or None
nbytes = solver.factor_nbytes()            # footprint of the stored factors
```

The constructor performs the entire factorization, which is what the reported
factorization time measures. `solve` returns a host array whatever device the
solver used internally.

### What each solver exposes

| Solver | Precision | Explicit factors | Reconstruction convention |
|---|---|---|---|
| `SparseLU` (SuperLU) | c128 / c64 | L, U, perm_r, perm_c | `Pr A Pc == L U` |
| `UMFPACK` | **c128 only** | L, U, perm_r, perm_c, R | `Pr diag(1/R) A Pc == L U` |
| `MUMPS` | c128 / c64 | none; Fortran-side, unexposed | — |
| `GMRES` (SciPy) | c128 / c64 | ILU preconditioner only, not an LU of A | — |
| `GMRESCuPy` | c128 / c64 | none; iterative | — |
| `CuDSS` | c128 / c64 | permutations and `lu_nnz` only | — |
| `BlockThomas` | c128 / c64 | per-block LU and pivots | `A == L U` exactly |
| `BlockThomasExplicitInv` | c128 / c64 | per-block explicit inverses and `D_mod` | `A == L U` exactly |
| `BlockThomasFP16` | fp16 | embedded-real LU at block size `2m` | `s embed(A) == L U` |
| `BlockThomasExplicitInvFP16` | fp16 | embedded-real inverses and scales `t` | `s embed(A) == L U` |

`UMFPACK` raises `TypeError` rather than upcasting silently when given a
single-precision dtype, so a single-precision batch run records a skip rather
than a misattributed double-precision result.

`GMRESCuPy` performs no preconditioning, since cupyx provides no incomplete LU.
Its construction step is only the host-to-device transfer of `A`, so its
reported factorization time is a transfer time and is not comparable with the
direct solvers; the solve time and the iteration count are.

---

## 2. The four Block Thomas variants

All four implement Higham, *Accuracy and Stability of Numerical Algorithms*,
2nd ed., Algorithm 13.3, in the two implementations that reference describes.

| | Implementation 1 | Implementation 2 |
|---|---|---|
| complex | `BlockThomas` | `BlockThomasExplicitInv` |
| half | `BlockThomasFP16` | `BlockThomasExplicitInvFP16` |

- **Implementation 1** factorizes each modified diagonal block by Gaussian
  elimination with partial pivoting; wherever the algorithm requires
  `D_mod^-1` it performs a triangular substitution against the stored factors.
  No block is ever explicitly inverted.
- **Implementation 2** forms `D_mod^-1` explicitly, so every solve stage
  becomes a dense matrix product. More flops in the factorization, roughly 3×
  an LU per block, and by Higham Chapter 14 generally less accurate, but it is
  the shape throughput-oriented hardware executes efficiently.

The full derivation, the cost and accuracy comparison, and the complete
account of what is and is not half precision are in
[the top-level README, section 4](../README.md#4-block-thomas). What follows is
the interface summary.

### 2.1 `inv_dtype`, the mixed-precision parameter

`BlockThomasExplicitInvFP16(..., inv_dtype=...)` controls the precision in
which the inverse is **formed**, independently of the fp16 storage and
application:

- `np.float32` (default) — the block is promoted to fp32, inverted with LAPACK,
  and rounded back to fp16. Explicit inversion is the least stable step in the
  algorithm; performing it at fp16 loses accuracy that nothing downstream
  recovers, while the fp32 inversion is `O(N)` small dense inversions and
  changes neither what is stored nor the precision of the solve. **Under this
  default the factorization is not purely half precision.**
- `np.float16` — a factorization that is half precision throughout. Slower and
  markedly less accurate; provided so that the value of the higher-precision
  inversion is measured rather than assumed.

On the synthetic tests in `test_pipeline.py` the fp32 inverse is worth roughly
1.7 to 2.3× in relative residual, and implementation 2 with an fp32 inverse is
*more* accurate than implementation 1 in pure fp16, approximately 5e-4 against
1.1e-3. Whether this holds on real QTBM matrices is measured by
`../run_bench/sweep_fp16.py`.

---

## 3. Block partitions

The partition enters the pipeline at exactly one place:

```python
D, L, U = extract_blocks_sparse(A, block_sizes)   # int, or a sequence
```

The solvers infer their partition from the shapes of the diagonal blocks they
receive, so none of them takes a block-size argument. For a custom partition
the off-diagonal blocks are rectangular: `L[k]` is `(n[k+1], n[k])` and `U[k]`
is `(n[k], n[k+1])`.

```python
sizes = block_sizes_from_matrix(A)     # detect from the sparsity pattern
assert offband_nnz(A, sizes) == 0      # ALWAYS check this
D, L, U = extract_blocks_sparse(A, sizes)
```

> **`offband_nnz` is not optional.** A partition that cuts through a real
> coupling does not fail loudly: `extract_blocks_sparse` discards the
> out-of-band entries and Block Thomas returns a plausible, wrong solution.
> `bench()` checks it by default, through `check_blocks=True`.

`find_block_slices` looks forward only, using the largest column index of each
row, so its partition is guaranteed block tridiagonal only for structurally
symmetric matrices. The QTBM matrices are, but nothing enforces it. It also
merges its first two slices by construction, so the leading block is coarser
than the true structure; this is harmless, at the cost of slightly more
arithmetic in block 0.

**The factor layout follows the partition.** Uniform partitions give stacked
`(N, bs, bs)` arrays from `get_LUP()`. Custom partitions give ragged lists,
which `factor_io` flattens to a 1-D buffer plus a `<name>_shapes` table. Read
either back with `factor_io.load_blocks(g, name)`, never by indexing the
dataset directly.

---

## 4. `bench_all.bench()`

One implementation, imported by every driver.

```python
metrics = bench(A, B, idx, block_sizes,
                dtypes=(np.complex128, np.complex64),
                h5file=f, save=True,
                solvers=DEFAULT_SOLVERS, exclude={"gmres": {"complex64"}})
```

- Every solver runs once per precision. The **first** precision defines the
  baseline: `superlu` at `dtypes[0]` is what every speedup and every "vs base"
  error refers to.
- Result keys are `<solver>_<suffix>`: `superlu_c128`, `block_thomas_inv_c64`.
- Skips are per `(solver, dtype)` and are reported rather than raised: no
  UMFPACK single-precision build, no visible GPU, a missing package. This keeps
  a batch run over heterogeneous machines comparable.
- **The half-precision variants are the exception.** They are precision-fixed,
  so they run once per index outside the precision loop, under the unsuffixed
  keys `block_thomas_fp16` and `block_thomas_inv_fp16` and the storage label
  `"complex32"`, which is not a NumPy dtype. They are **not** in
  `DEFAULT_SOLVERS`: their kernels are written in NumPy and are orders of
  magnitude slower than LAPACK, so request them through `bench_all.FP16_SOLVERS`
  when accuracy rather than timing is being measured.

---

## 5. `factor_io.py`

Solver output is appended into the **material's own** HDF5 file, as a direct
sibling of each energy index's `M`, `rhs` and `Sigma`:

```
E_<idx>/superlu/<dtype>/          L, U (sparse CSC), perm_r, perm_c, x, times
E_<idx>/umfpack/<dtype>/          the same, plus R and do_recip
E_<idx>/blockthomas/<dtype>/      L, U, Dmod_lu, Dmod_piv, block_sizes, x, times
E_<idx>/blockthomas_inv/<dtype>/  L, U, Dmod_inv, Dmod, block_sizes, x, times
E_<idx>/mumps|cudss|gmres*/<dtype>/   x, times, and whatever metadata exists
```

Each `save_*` deletes and recreates its own `E_<idx>/<solver>/<dtype>` group,
so re-running a combination overwrites cleanly and never touches `M`, `rhs`,
`Sigma` or `spectrum`. Pass the material file opened `"a"` or `"r+"`; there is
no separate factor file.

`diagnose_lu_convention` determines a solver's factor reconstruction convention
empirically, by enumerating the plausible combinations of permutation direction
and row scaling and returning the one with the smallest residual. It is
intended to be run once per solver and version, so that a convention is
established rather than assumed.

---

## 6. Tests

```bash
python test_pipeline.py
```

Synthetic systems only. Covers all four Block Thomas variants against dense
references at uniform *and* non-uniform partitions, multiple right-hand sides,
the block detector and its off-band guard, and a full HDF5 round trip,
`bench()` to `factor_io` to the growth-factor assemblers, verifying that the
reloaded factors still reproduce `A == L U`. The round-trip section requires
h5py and is skipped without it. UMFPACK is the one solver with no local
coverage.

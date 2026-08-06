# `block-thomas/` — stability analysis

Post-hoc analysis scripts. They **read** factors that
[`../run_bench/`](../run_bench/) already wrote into the material HDF5 files;
none of them solves a system or mutates the h5.

| script | what it answers |
|---|---|
| `growth_factor.py` | is the factorization backward stable? how much did the factors grow? |
| `determine_custom_block_size.py` | what non-uniform block partition does this matrix have? |
| `arnoldi_shift_invert_cpu.py` / `arnoldi_shift_invert_gpu.py` | extreme eigenvalues, singular values and the condition number |

---

## `growth_factor.py`

*(was `blockthomas_growth_factor.py` / `blockthomas_stability.py` — it now
covers SuperLU and UMFPACK too, so the name no longer says "blockthomas".)*

### The question

All of these solvers are LU-based, so per Higham none is unconditionally
backward stable. The perturbation is bounded entrywise by

```
|A − L U|  ≤  γₙ · |L| |U|
```

which, in a monotone norm, gives `‖A − L U‖ ≤ γₙ · ‖ |L| |U| ‖`. A
factorization is backward stable **in practice** iff factor growth relative to
`A` stays modest.

That growth is a property of the **pivoting**, and is *independent of
cond(A)*: an ill-conditioned `A` can factor with tiny growth, and a
well-conditioned `A` can grow badly if a Schur-complement block becomes badly
scaled. Conditioning governs the accuracy of `x`; growth governs the backward
error. Confusing the two is the easiest mistake to make here.

Block Thomas pivots only *within* diagonal blocks — weaker than the global
partial pivoting with column ordering that SuperLU and UMFPACK do. Those two
are therefore the reference points that say whether exploiting the block
structure costs stability, which is why they are in this script.

### What it reports

Per `(index, solver, dtype, norm)`, for both the 1-norm and inf-norm:

| metric | definition |
|---|---|
| loose ratio | `‖L‖·‖U‖ / ‖A_eff‖` — classical, always an upper bound |
| tight ratio | `‖ |L||U| ‖ / ‖A_eff‖` — the true Wilkinson quantity, sharper |
| `rho` | `max|Uᵢⱼ| / max|A_effᵢⱼ|` — the pivot growth factor, norm-free |
| `resid_rel` | `‖A_eff − L U‖ / ‖A_eff‖` — correctness guard on the assembly |

### `A_eff` is not always `A`

Each solver's factors reconstruct a *different* matrix. Using `‖A‖` as the
denominator regardless would silently misreport UMFPACK.

| solver | what `L @ U` reproduces |
|---|---|
| `blockthomas` | `A` |
| `blockthomas_inv` | `A` |
| `superlu` | `Pr @ A @ Pc` |
| `umfpack` | `Pr @ diag(1/R) @ A @ Pc` |

Permutations alone don't change the 1- or inf-norm (a row permutation reorders
rows; a column permutation permutes entries *within* each row, leaving row
sums intact), so SuperLU could reuse `‖A‖`. UMFPACK's **row scaling** does
change it. `A_eff` is built explicitly in every case anyway — it also keeps
`resid_rel` honest, which is the guard that the UMFPACK convention still holds
for your build.

For the **fp16** groups `A_eff` is neither: those factors factored the
block-by-block real embedding of `A`, scaled by `s`, at block size `2·bs`.
Block-local embedding differs from embedding `A` globally by a permutation, so
`effective_A()` rebuilds it exactly the way the solver did, from the recorded
partition.

### Assembly

Both Block Thomas implementations reconstruct the same genuine global
block-bidiagonal factors, with `A == L_global @ U_global` exactly:

```
L_global = block-bidiagonal(I;      E_k = L_off[k−1] @ D_mod[k−1]⁻¹)
U_global = block-bidiagonal(D_mod;  U_off)
```

Implementation 1 rebuilds `D_mod` from its packed LU and pivots. Implementation
2 stores `D_mod` and its explicit inverse directly, which is *exactly* what the
assembly needs — no triangular solve required. Both handle uniform and ragged
partitions: `load_blocks()` returns a stacked array or a list, and they index
identically.

### Usage

```bash
python growth_factor.py /scratch/yimili/matrices/hdf5/graphene.h5 --idx 25
python growth_factor.py .../graphene.h5 --start 1 --end 400
python growth_factor.py .../graphene.h5 --start 1 --end 400 \
    --solvers blockthomas superlu umfpack --dtype complex128
```

Writes `<material>_growth_factor.{png,csv}` to `--plotdir` (default
`/scratch/yimili/block-thomas`). The plot is one row per norm:
ratios | `rho` | residual. `--no-plot` prints only; `--no-csv` skips the CSV.

Solvers and dtypes absent from the file are skipped with a note, so running
with the default `--solvers` on a file that only has Block Thomas is fine.

---

## `determine_custom_block_size.py`

Derives a non-uniform block partition from the sparsity pattern by growing a
reach frontier row by row: a boundary is declared once the frontier stops
advancing.

**The algorithm itself now lives in `solver_classes`**
(`find_block_slices` / `block_sizes_from_matrix`) so the detector and the
solvers that consume its output cannot drift apart. This file is the CLI and
the verification wrapper.

```bash
python determine_custom_block_size.py .../graphene.h5
python determine_custom_block_size.py .../graphene.h5 --idx 25
python determine_custom_block_size.py .../M_E_0.npz              # bare CSR triplet
python determine_custom_block_size.py .../graphene.h5 --compare-bs 416
python determine_custom_block_size.py .../graphene.h5 --emit-python
```

Reports block count, size range, `sum(bs²)` (the dense block-storage cost that
decides whether a custom partition is worth it), and **`offband_nnz`** — exit
code 1 if that is nonzero. `--compare-bs` prints the same figures for the
current uniform block size and their storage ratio. `--emit-python` prints one
line to paste into `run_benchmarks.MATERIAL_BLOCKS`.

> Two caveats, both benign for QTBM matrices but worth knowing. The detector
> only looks **forward**, so its partition is guaranteed block-tridiagonal only
> when the matrix is structurally symmetric — hence the mandatory `offband_nnz`
> check. And it merges its first two slices by construction, so the leading
> block is coarser than the true structure; a coarser partition is still
> correct, just slightly more arithmetic in block 0.

---

## Arnoldi / shift-invert

`arnoldi_shift_invert_cpu.py` (MUMPS / Block Thomas / SuperLU) and
`arnoldi_shift_invert_gpu.py` (cuDSS). Same CLI — the GPU script imports
everything but the backend list from the CPU one.

```bash
python arnoldi_shift_invert_cpu.py MATRIX [options]     # MATRIX: CSR .npz triplet
```

Two flags decide everything else:

| | `--end largest` | `--end smallest` (default) |
|---|---|---|
| `--quantity eigenvalue` | `eigs(A, which=LM)` | `eigs(A, sigma=0, OPinv=A⁻¹)` |
| `--quantity singular` | `svds(A, which=LM)` | `svds(A⁻¹, which=LM)` → `1/σ` |
| factorizations | **0**, `--backend` unused | 1 eig / 2 svd (`A` and `Aᴴ`) |

`--quantity condition` = singular at both ends + `σ_max/σ_min`, and reports
`cond·u` per precision for the mixed-precision IR question.

| flag | default | |
|---|---|---|
| `--method` | `arpack` | eig: `arpack`\|`power` · svd: `arpack`\|`propack` |
| `--backend` | `mumps`\|`cudss` | `blockthomas`, `blockthomas_inv`, `superlu`, `gmres_cupy` |
| `-k` | 1 | how many values |
| `--factor-dtype` | `c128` | `c64` factorizes in single precision; Krylov stays c128 |
| `--tol` | 1e-8 | on the *transformed* problem |
| `--ncv` | `max(2k+1,20)` | arpack basis size; raise if it stalls |
| `--maxiter` | — | arpack restarts / power iterations |
| `--no-shift-invert` | off | attack the small end of `A` directly; slow, for comparison |
| `--no-fallback` | off | don't retry propack failures with arpack |

```bash
python arnoldi_shift_invert_cpu.py MATRIX --quantity condition
python arnoldi_shift_invert_cpu.py MATRIX --quantity singular --method propack -k 5
python arnoldi_shift_invert_cpu.py MATRIX --method power --maxiter 200  # stalls
python arnoldi_shift_invert_gpu.py MATRIX --quantity singular --backend cudss
```

Three things worth knowing:

- **Residuals are against the original `A`**, not the transformed operator.
  `--tol` refers to the latter, so only the residual column says whether a
  value is trustworthy.
- **PROPACK is handicapped by scipy, not by PROPACK.** `svds` hardcodes
  `maxiter=None` in its propack branch, so it falls back to `kmax = 10*k` —
  a basis of 10 at `k=1`, against arpack's 20. Raising `-k` is the only lever.
  A failure under `--end smallest` retries with arpack on the factorizations
  already paid for.
- **`--end smallest` for singular values costs two factorizations.** `svds`
  needs `rmatvec` and neither MUMPS nor Block Thomas exposes a transpose solve.

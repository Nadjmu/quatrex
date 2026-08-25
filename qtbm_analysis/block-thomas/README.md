# `block-thomas/` — stability and spectral analysis

Post-hoc analysis. These scripts **read** factors that
[`../run_bench/`](../run_bench/) has already written into the material HDF5
files; none of them solves a system or modifies that file. They write an
analysis HDF5 file, never figures; the corresponding figures are produced by
[`../plotting/`](../plotting/).

| Script | Question addressed |
|---|---|
| `growth_factor.py` | is the factorization backward stable, and by how much did the factors grow? |
| `determine_custom_block_size.py` | what non-uniform block partition does this matrix have? |
| `arnoldi_shift_invert_cpu.py` / `_gpu.py` | extreme eigenvalues, singular values, and the condition number |

---

## 1. `growth_factor.py`

### 1.1 The question

All four solvers analysed are LU based, so by Higham none is unconditionally
backward stable. The computed factors satisfy the entrywise bound

```
    |A - L U|  <=  gamma_n |L| |U|,        gamma_n = n u / (1 - n u),
```

which in any monotone norm gives `||A - L U|| <= gamma_n || |L| |U| ||`. A
factorization is therefore backward stable in practice precisely when the
factors do not grow relative to `A`.

That growth is a property of the **pivoting**, and is **independent of
`kappa(A)`**: an ill-conditioned `A` may factor with negligible growth, and a
well-conditioned `A` may grow badly if a Schur complement becomes badly scaled.
Conditioning bounds the forward error of `x`; growth bounds the backward error.
The two must not be conflated.

Block Thomas pivots only *within* each diagonal block, which is weaker than the
global partial pivoting with column ordering performed by SuperLU and UMFPACK.
Those two are therefore the reference points that quantify what, if anything,
exploiting the block structure costs in stability.

### 1.2 What is reported

Per `(index, solver, dtype, norm)`, for both the 1-norm and the infinity norm:

| Metric | Definition |
|---|---|
| loose ratio | `||L|| ||U|| / ||A_eff||` — classical, always an upper bound |
| tight ratio | `|| |L| |U| || / ||A_eff||` — the quantity that enters the bound |
| `rho` | `max|U_ij| / max|A_eff_ij|` — the pivot growth factor, norm-free |
| `resid_rel` | `||A_eff - L U|| / ||A_eff||` — reconstruction guard |

`resid_rel` is **not** a stability metric. It verifies that the assumed factor
convention holds for the build that produced the file. If it is not near the
unit roundoff of the stored precision, the other three columns are meaningless
and must be discarded.

### 1.3 `A_eff` is solver dependent

Each solver's factors reconstruct a different matrix, and using `||A||` as the
denominator throughout would misreport UMFPACK.

| Solver | What `L U` reproduces |
|---|---|
| `block-thomas` | `A` |
| `block-thomas-inv` | `A` |
| `superlu` | `Pr A Pc` |
| `umfpack` | `Pr diag(1/R) A Pc` |

A permutation alone leaves the 1-norm and the infinity norm unchanged: a row
permutation reorders rows, and a column permutation permutes entries *within*
each row, so every row sum is preserved. SuperLU could therefore reuse `||A||`.
UMFPACK's **row scaling** does change them. `A_eff` is built explicitly in all
cases rather than special-cased, which also keeps `resid_rel` meaningful.

For the **half-precision** groups `A_eff` is neither: those factors were
computed from the real embedding of `A` applied **block by block**, scaled by a
global power of two `s`, at block size `2m`. Embedding block by block differs
from embedding `A` globally by a permutation, so `effective_A()` rebuilds the
matrix exactly as the solver did, from the recorded partition.

### 1.4 Assembly

Both Block Thomas implementations reconstruct the same genuine global
block-bidiagonal factors, with `A == L_global U_global` exactly:

```
    L_global = block-bidiagonal(I;      E_k = L_off[k-1] D_mod[k-1]^-1)
    U_global = block-bidiagonal(D_mod;  U_off)
```

Implementation 1 rebuilds `D_mod` from its packed LU and pivots.
Implementation 2 stores `D_mod` and its explicit inverse directly, which is
precisely what the assembly requires, with no triangular solve. Both handle
uniform and ragged partitions: `load_blocks()` returns a stacked array or a
list, and both are indexed identically.

### 1.5 Usage

```bash
python growth_factor.py /scratch/yimili/matrices2/hdf5/graphene.h5 --idx 25
python growth_factor.py .../graphene.h5 --start 1 --end 400
python growth_factor.py .../graphene.h5 --start 1 --end 400 \
    --solvers block-thomas superlu umfpack --dtypes complex128

python ../plotting/block-thomas/plot_growth_factor.py \
    /scratch/yimili/error-analysis-block-thomas/graphene.h5
```

Writes the `growth_factor` group of `<outdir>/<material>.h5`, with `--outdir`
defaulting to `cli.BLOCK_THOMAS_DIR`. The file is opened in append mode and
only that group is rewritten, so the `fp16_sweep` group written by
[`../run_bench/sweep_fp16.py`](../run_bench/sweep_fp16.py) into the same file
is preserved. `--no-save` prints the per-index report only.

Solver names are the canonical ones; the stored HDF5 group is resolved
internally, so the on-disk layout of the material file is unchanged.

Solvers and precisions absent from the file are reported and skipped, so
running with the default `--solvers` on a file that contains only Block Thomas
results is correct.

---

## 2. `determine_custom_block_size.py`

Derives a non-uniform block partition from the sparsity pattern by growing a
reach frontier row by row, declaring a boundary wherever the frontier stops
advancing.

**The detection algorithm resides in `solver_classes`**
(`find_block_slices` / `block_sizes_from_matrix`), so that the detector and the
solvers consuming its output cannot diverge. This file is the command-line
front end and the verification wrapper.

```bash
python determine_custom_block_size.py .../graphene.h5
python determine_custom_block_size.py .../graphene.h5 --idx 25
python determine_custom_block_size.py .../M_E_0.npz          # bare CSR triplet
python determine_custom_block_size.py .../graphene.h5 --compare-block-size 416
python determine_custom_block_size.py .../graphene.h5 --emit-python
```

Reports the block count, the size range, `sum(bs^2)` — the dense block-storage
cost that determines whether a custom partition is worthwhile — and
**`offband_nnz`**, exiting with status 1 if that is nonzero.
`--compare-block-size`
reports the same figures for a uniform partition and their storage ratio.
`--emit-python` prints a line to paste into the `blocks` field of the material's
entry in `cli.MATERIALS`.

Two properties of the detector, both benign for QTBM matrices, must be kept in
mind. It looks **forward only**, so its partition is guaranteed block
tridiagonal only for structurally symmetric matrices, which is why the
`offband_nnz` check is mandatory. And it merges its first two slices by
construction, so the leading block is coarser than the true structure; a
coarser partition remains correct, at the cost of slightly more arithmetic in
block 0.

---

## 3. Arnoldi and shift-invert

`arnoldi_shift_invert_cpu.py` (MUMPS, Block Thomas, SuperLU) and
`arnoldi_shift_invert_gpu.py` (cuDSS). The interface is identical; the GPU
script imports everything except the backend list from the CPU one, so the two
cannot diverge.

```bash
python arnoldi_shift_invert_cpu.py MATRIX [options]    # MATRIX: CSR .npz triplet
```

Two flags determine everything else, including whether a factorization is
required at all:

| | `--end largest` | `--end smallest` (default) |
|---|---|---|
| `--quantity eigenvalue` | `eigs(A, which=LM)` | `eigs(A, sigma=0, OPinv=A^-1)` |
| `--quantity singular` | `svds(A, which=LM)` | `svds(A^-1, which=LM)`, then `1/sigma` |
| factorizations | **0**; `--backend` unused | 1 for eigenvalues, 2 for singular values (`A` and `A^H`) |

`--quantity condition` computes singular values at both ends and their ratio,
and reports `kappa u` per precision, which is the quantity that decides whether
a mixed-precision refinement scheme can converge; see
[`../mixed_prec_ir/`](../mixed_prec_ir/).

| Flag | Default | Meaning |
|---|---|---|
| `--method` | `arpack` | eigenvalues: `arpack` or `power`; singular values: `arpack` or `propack` |
| `--backend` | `mumps` / `cudss` | also `block-thomas`, `block-thomas-inv`, `superlu`, `gmres-cupy` |
| `-k`, `--num-values` | 1 | how many values |
| `--factor-dtype` | `complex128` | `complex64` factorizes in single precision; the Krylov method stays at complex128 |
| `--tol` | 1e-8 | on the *transformed* problem |
| `--ncv` | `max(2k+1, 20)` | ARPACK basis size; raise if convergence stalls |
| `--max-iter` | — | ARPACK restarts or power iterations |
| `--no-shift-invert` | off | attack the small end of `A` directly; slow, for comparison |
| `--no-fallback` | off | do not retry a PROPACK failure with ARPACK |
| `--block-size` | detected | uniform Block Thomas block size |

```bash
python arnoldi_shift_invert_cpu.py MATRIX --quantity condition
python arnoldi_shift_invert_cpu.py MATRIX --quantity singular --method propack -k 5
python arnoldi_shift_invert_cpu.py MATRIX --method power --max-iter 200
python arnoldi_shift_invert_gpu.py MATRIX --quantity singular --backend cudss
```

Three properties of the implementation to be aware of:

- **Residuals are measured against the original `A`**, not against the
  transformed operator. `--tol` refers to the latter, so only the residual
  column establishes whether a value is converged.
- **PROPACK is constrained by the SciPy interface, not by PROPACK.** `svds`
  passes `maxiter=None` in its PROPACK branch, so PROPACK falls back to its own
  default `kmax = 10k`: a basis of 10 at `k = 1`, against ARPACK's default
  `ncv` of 20. Raising `-k` is the only control the public interface offers. A
  failure under `--end smallest` retries with ARPACK on the factorizations
  already computed.
- **`--end smallest` for singular values costs two factorizations.** `svds`
  requires `rmatvec`, and neither the MUMPS nor the Block Thomas Python
  interface exposes a transpose solve.

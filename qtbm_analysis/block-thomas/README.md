# `block-thomas/` — stability and spectral analysis

Post-hoc analysis. These scripts **read** what
[`../run_bench/`](../run_bench/) has already written into the material HDF5
files, and none of them modifies that file. They write an analysis HDF5 file,
never figures; the corresponding figures are produced by
[`../plotting/`](../plotting/).

`forward_error.py` is the one exception to "solves nothing": it needs a
solution more accurate than any the benchmark produced, so it factorizes once
per index and refines. It repeats none of the benchmark's own solves.

| Script | Question addressed |
|---|---|
| `growth_factor.py` | is the factorization backward stable, and by how much did the factors grow? |
| `forward_error.py` | how close is the computed solution to the true one, and do the classical bounds predict it? |
| `extract_lu.py` | what do the factors themselves look like — how much fill-in, and where? |
| `determine_custom_block_size.py` | what non-uniform block partition does this matrix have? |
| `arnoldi_shift_invert_cpu.py` / `_gpu.py` | extreme eigenvalues, singular values, and the condition number |

`error_analysis.html` is the chapter write-up that accompanies the first two:
what the growth columns measure, where the Schur complements enter the block LU
bound, and which quantity pairs with which backward error. Open it in a browser;
it is a standalone file and pulls in nothing but its fonts.

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

All three growth columns are **normwise**. They are not a normwise/componentwise
pair — that distinction belongs to the backward errors `eta` and `omega` that
`solvers/bench_all.py` measures per solve. The three differ only in how much of
the entrywise theorem each discards:

```
    |A - LU| <= gamma_n |L| |U|          entrywise, Higham ASNA Thm 9.3
 => ||A - LU|| <= gamma_n || |L||U| ||   any monotone norm      -> tight ratio
 => ||A - LU|| <= gamma_n ||L|| ||U||    submultiplicativity    -> loose ratio
```

so `tight <= loose` always, and it is `tight` that bounds the backward error of
the factorization. `loose` is reported because it is the form usually quoted,
and the gap between the two is itself worth showing. `rho` is the norm-free
scalar; `max|U|` is the standard cheap surrogate for Wilkinson's maximum over
all intermediate `A^(k)`.

### 1.2.1 The Schur-complement columns

Block Thomas is block LU, and the three columns above do not describe it fully:
they see the assembled global factors, not the recursion that produced them.
For `block-thomas` and `block-thomas-inv`, four further columns are reported,
from one SVD per diagonal block (`--no-schur` skips them):

| Metric | Definition |
|---|---|
| `schur_growth` | `max_k ||S_k||_2 / max_k ||A_kk||_2` — the block analogue of `rho` |
| `schur_norm_max` | `max_k ||S_k||_2` |
| `schur_cond_max` | `max_k kappa_2(S_k)` — conditioning of the pivot blocks |
| `inv_resid_max` | `max_k ||S_k G_k - I||_2` — implementation 2 only |

`S_1 = A_11`, `S_k = A_kk - A_{k,k-1} S_{k-1}^-1 A_{k-1,k}` is the recursion,
and the `S_k` are exactly the diagonal blocks of `U_global` (`Dmod` on disk).
Two facts make these the decisive columns of the chapter:

- Block LU's backward error carries `kappa(S_k)`, not only `||S_k||`. A leading
  principal block submatrix that is near-singular makes some `S_k`
  near-singular however well conditioned `A` is and however little anything
  grew, and there is no pivoting across blocks to prevent it. Varah's block
  diagonal dominance is the classical sufficient condition for this not to
  happen.
- Implementation 2 forms `S_k^-1` explicitly, and explicit inversion is **not**
  backward stable: the computed inverse satisfies only
  `||S G - I|| <~ c u kappa(S)`. `inv_resid_max` measures exactly that, and is
  the term implementation 1 does not pay. Read it against `schur_cond_max`: the
  two should differ by roughly the unit roundoff of the precision, and a
  synthetic check reproduces this to within a factor of two
  (`kappa_2 = 3.1e1`, `inv_resid = 5.1e-15` at complex128 and `2.0e-06` at
  complex64).

All four are 2-norm quantities and are therefore repeated across both norm
rows, exactly as `rho` is.

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
python growth_factor.py .../graphene.h5 --stride 20
python growth_factor.py .../graphene.h5 --start 900 --end 1100 \
    --solvers block-thomas superlu umfpack --dtypes complex128

python growth_factor.py .../graphene.h5 --stride 20 --no-schur

python ../plotting/block-thomas/plot_growth_factor.py \
    /scratch/yimili/error-analysis-block-thomas/graphene.h5
```

With no index selection every index the file holds is analysed. A
full-resolution sweep is a few thousand indices — the count follows from the
material's `EnergyGrid`, and is stated nowhere — and each index costs one
assembly of the global factors per (solver, precision) plus, unless
`--no-schur` is given, one SVD per diagonal block. `--stride` is the usual way
to keep a first pass short. The partition is never given on the command line:
it is read back from the `block_sizes` dataset each Block Thomas group carries,
so the analysis uses exactly the partition the solver used.

The plotting script writes two figures: `<material>_growth_factor.png` and,
when the Schur columns are present, `<material>_schur_growth.png`.

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

## 2. `forward_error.py`

### 2.1 The question

Backward stability says the computed `xhat` solves a nearby problem exactly. It
says nothing about `||xhat - x||`. The two classical bounds close that gap:

```
    ||xhat - x||_inf / ||x||_inf  <~  kappa_inf(A) * eta_inf     normwise
    ||xhat - x||_inf / ||x||_inf  <~  cond(A, x)   * omega       componentwise
```

Every quantity on the right already exists: `eta_inf` and `omega` are measured
per solve by `bench_all.backward_errors` and stored in the material file;
`kappa_inf` and the Skeel `cond(A, x)` come from
[`../condition-est/condition_est.py`](../condition-est/). What was missing is
the left-hand side, and that is what this script computes.

The stored `vs_base` column cannot serve as the forward error. Its reference is
the SuperLU complex128 solution, which carries an error of the same order as
the solutions being judged; at complex128 it compares two errors of equal size.

### 2.2 The reference solution

Iterative refinement of the SuperLU complex128 solution, with the residual
accumulated in `np.clongdouble` (80-bit on x86-64, `eps = 1.1e-19`) and the
correction solved with the same double-precision factorization. SciPy has no
sparse type in that precision, so `A xhat` is formed from the CSR arrays
directly, elementwise in extended precision and summed per row with
`np.add.reduceat`. The iterate is carried in extended precision too: rounded
back to double at every step it could never become more accurate than the
solutions it is meant to judge.

Refinement in a residual precision `u_r` converges to a relative error of order
`kappa * u_r`, so the reference beats a double-precision solution by roughly
`eps_double / eps_ext = 2e3`, not by the `u^2` a true double-double residual
would give. That limit is recorded rather than hidden: `ref_floor =
kappa_inf * eps_ext` is written per index, and a measured forward error within
an order of magnitude of it is a measurement of the reference, not of the
solver. On a synthetic system with `kappa_inf = 5e3` and an exactly represented
right-hand side, the double solve reaches `6.3e-14` and the reference `1.1e-18`.

### 2.3 Coverage

Unlike `growth_factor.py`, this script needs only `x`, never the factors, so
**MUMPS and cuDSS are included**: they store a solution like every other solver
even though python-mumps and cuDSS expose no `L` and `U`. The growth-factor
comparison covers the four solvers of `cli.FACTOR_SOLVERS`; the forward-error
and backward-error comparison covers all of them.

### 2.4 Usage

```bash
python forward_error.py /scratch/yimili/matrices2/hdf5/carbon-nanotube.h5
python forward_error.py .../carbon-chain.h5 --stride 10
python forward_error.py .../carbon-nanotube.h5 --idx 25 --no-save

python ../plotting/block-thomas/plot_forward_error.py \
    /scratch/yimili/error-analysis-block-thomas/carbon-nanotube.h5
```

Writes the `forward_error` group of `<outdir>/<material>.h5`, beside the
`growth_factor` and `fp16_sweep` groups; the file is opened in append mode and
only that group is rewritten. The same group also feeds
`plot_backward_error.py`, which reads only its eta/omega columns and is the
one figure covering all six solvers, MUMPS and cuDSS included, since backward
error needs the stored solution and nothing the two factor-less solvers
withhold. The condition file defaults to
`cli.CONDITION_DIR/<material>.h5` and is optional: without it the forward
errors are still recorded and only the bound columns are NaN.
`cond_skeel_x` requires `condition_est.py` to have been run with the Skeel
columns present — an older condition file gains them from
`condition_est.py --only-skeel`, which reuses the valid rows and recomputes
nothing else.

---

## 3. `extract_lu.py`

Assembles the global `L` and `U` of a stored factorization and writes them out,
so that the factors can be looked at rather than only measured. The assembly is
`growth_factor.ASSEMBLERS`, imported rather than repeated: the factors written
here are exactly the ones the growth columns are computed from. Nothing is
recomputed and no factorization is performed.

Per `(index, solver, dtype)` three sparse matrices are stored: `L`, `U`, and
`A_eff`, the matrix those factors reconstruct — `A` for the Block Thomas
variants, `Pr A Pc` for SuperLU, `Pr diag(1/R) A Pc` for UMFPACK, as in section
1.3. `resid_rel = ||A_eff - L U|| / ||A_eff||` is stored as an attribute and is
the same guard it is there: far above the unit roundoff of the stored
precision, the extracted factors are not the ones the solver computed.

```bash
python extract_lu.py /scratch/yimili/matrices2/hdf5/carbon-nanotube.h5 --idx 5
python extract_lu.py .../graphene.h5 --idx 25 --solvers block-thomas superlu \
    --dtypes complex128

python ../plotting/block-thomas/plot_lu_factors.py \
    /scratch/yimili/error-analysis-block-thomas/carbon-nanotube.h5 --idx 5
```

Writes `lu_factors/E_<idx>/<solver>/<dtype>/{A_eff,L,U}` as CSC triplets into
`<outdir>/<material>.h5`, beside the `growth_factor` and `forward_error`
groups. **Only the combinations extracted in a run are replaced**, not the
whole group — unlike every other script here, which rewrites its group
wholesale. An extraction is per index and costs a full copy of the factors, so
wholesale replacement would discard the earlier indices, and a sweep is not a
sensible selection in the first place: the intended use is a handful of
indices.

The figure is a 3x3 grid, one row per solver (`block-thomas`, `superlu`,
`umfpack`, whichever are present) and one column per matrix (`A_eff`, `L`,
`U`), as `log10` of the entry magnitude on one shared colour scale across the
whole grid. Fill-in and factor growth are then comparable both within a row
(a factor panel brighter than its own `A_eff`) and across solvers. The full
matrix is drawn; there is no windowing.

---

## 4. `determine_custom_block_size.py`

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
`--emit-python` prints a line in the form of the `blocks` field of
`cli.MATERIALS`.

**This script is diagnostic only.** No driver takes a partition from a
material entry or from the command line any more: every one of them detects it
from the sparsity pattern of the matrix at hand, through the same
`block_sizes_from_matrix` this script wraps. What it is for is inspecting a new
material's structure — how many blocks, how ragged, and what the dense
block-storage cost would be — before a sweep is started.

Two properties of the detector, both benign for QTBM matrices, must be kept in
mind. It looks **forward only**, so its partition is guaranteed block
tridiagonal only for structurally symmetric matrices, which is why the
`offband_nnz` check is mandatory. And it merges its first two slices by
construction, so the leading block is coarser than the true structure; a
coarser partition remains correct, at the cost of slightly more arithmetic in
block 0.

---

## 5. Arnoldi and shift-invert

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

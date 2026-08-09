# `qtbm_analysis`

Numerical study of the linear systems produced by the QTBM transport solver:
how fast the available sparse solvers factorize them, how accurately, and how
stably. Block Thomas and its reduced-precision variants are the principal
subject.

---

## 1. The problem

At each energy `E` on a sweep, QTBM assembles the system matrix

```
    M(E) = E S - H - Sigma(E),
```

where `H` and `S` are the Hamiltonian and overlap matrices of the device and
`Sigma(E)` collects the contact self-energies imposed by the open boundary
conditions. The transport calculation requires the solution of `M(E) x = b`
at every energy index, so the cost of a full simulation is dominated by a few
hundred sparse solves of the same sparsity pattern at different values of `E`.

Two properties of `M(E)` govern everything that follows.

**Block tridiagonal structure.** The device is discretized as a chain of
layers with nearest-neighbour coupling only, so under the natural ordering
`M(E)` is block tridiagonal. This admits a direct solver of cost `O(N m^3)` for
`N` blocks of size `m`, against `O(n^3)` for a dense factorization, and it is
what Block Thomas exploits.

**Conditioning.** `kappa_2(M(E))` varies by orders of magnitude across the
sweep and peaks near band edges, where `M(E)` approaches singularity. This is
what limits reduced-precision arithmetic: a solve in a precision with unit
roundoff `u` cannot resolve the solution at all once `kappa_2 u >= 1`.

---

## 2. The pipeline

Everything is organised around **one HDF5 file per material**. Solver results
are appended into that same file rather than into a separate one, so a matrix
and every factorization of it are stored together, and no analysis script ever
has to correlate two files by index.

### 2.1 Stages

```
  STAGE 1  EXPORT                    export_qtbm_systems.py
           QTBM example directory       main3.py  /  main3_gpu.py
                    │
                    │  M(E), Sigma(E), rhs per energy index, as .npz / .npy
                    ▼
  STAGE 2  CONSOLIDATE              make_hdf5.py
                    │
                    │  one <material>.h5 per material
                    ▼
        ┌────────────────────────────────────────────────────┐
        │  <material>.h5                                     │
        │    metadata/indices, metadata/energies             │
        │    global/condition_full_svd, ...                  │
        │    E_<idx>/M, E_<idx>/Sigma, E_<idx>/rhs           │
        └────────────────────────────────────────────────────┘
                    │
                    ├──────────────────────────────────────────────────┐
                    ▼                                                  │
  STAGE 3  SOLVE + PERSIST          run_bench/run_benchmarks.py        │
                    │                run_bench/gpu_run_benchmarks.py   │
                    │  appends, in place, into the same file:          │
                    │    E_<idx>/superlu/<dtype>/       L, U, perms, x │
                    │    E_<idx>/umfpack/<dtype>/       + row scaling  │
                    │    E_<idx>/blockthomas/<dtype>/   per-block LU   │
                    │    E_<idx>/blockthomas_inv/<dtype>/ inverses     │
                    │    E_<idx>/mumps|cudss|gmres*/<dtype>/  x, times │
                    ▼                                                  │
  STAGE 4  ANALYSE (reads only, writes CSV)                            │
                    │                                                  │
    block-thomas/growth_factor.py ──────► <material>_growth_factor.csv │
    run_bench/sweep_fp16.py ────────────► <material>_metrics.csv ◄─────┘
    mixed_prec_ir/c32_gmres_ir.py ─────► <material>_fp16_gmres_ir.csv
    non-normal/non-normal.py ──────────► ratio_matrix.npy, ...
                    │
                    ▼
  STAGE 5  PLOT (reads CSV / HDF5, writes figures only)
                    plotting/plot_speedup.py
                    plotting/plot_growth_factor.py
                    plotting/plot_fp16_accuracy.py
                    plotting/plot_mixed_prec_ir.py
                    plotting/plot_non_normal.py
                    plotting/plot_qtbm_spectra.py
```

### 2.2 The invariant

**Computation and visualization are strictly separated.** No script that
solves, factorizes or measures anything imports matplotlib. Every quantity a
figure displays is first written to a CSV or an HDF5 dataset, and every figure
is produced by a script in `plotting/` that reads that artefact and computes
nothing.

This has three consequences worth stating explicitly. A sweep that takes hours
on the cluster is never repeated in order to change an axis label. The numbers
behind every figure exist as a file that can be inspected, cited or replotted.
And a plotting change cannot alter a measurement.

### 2.3 Complete worked example

```bash
# Stage 1-2: export and consolidate. Cluster-side; needs the QTBM examples.
python export_qtbm_systems.py <example> --mode full --energy-index 0
python make_hdf5.py /scratch/yimili/matrices2 --material graphene

# Stage 3: solve every index with every solver, both precisions.
#          MUTATES graphene.h5 in place.
cd run_bench && python run_benchmarks.py

# Stage 4: analysis. Read-only with respect to the HDF5 file.
cd ../block-thomas
python growth_factor.py /scratch/yimili/matrices/hdf5/graphene.h5 \
    --start 1 --end 400
cd ../run_bench
python sweep_fp16.py /scratch/yimili/matrices/hdf5/graphene.h5 \
    --start 0 --end 401 --block-size 416
cd ../mixed_prec_ir
python c32_gmres_ir.py /scratch/yimili/matrices/hdf5/graphene.h5 \
    --start 0 --end 401 --block-size 416 --outdir plots

# Stage 5: figures.
cd ../plotting
python plot_speedup.py        /scratch/yimili/matrices/hdf5/graphene.h5
python plot_growth_factor.py  /scratch/yimili/block-thomas/graphene_growth_factor.csv
python plot_fp16_accuracy.py  ../run_bench/plots/graphene_metrics.csv
python plot_mixed_prec_ir.py  ../mixed_prec_ir/plots/graphene_fp16_gmres_ir.csv

# Tests: synthetic data only, no cluster files required.
cd ../solvers && python test_pipeline.py
```

### 2.4 Ordering constraints

| Stage | Requires | Reason |
|---|---|---|
| `run_benchmarks.py` | stage 2 complete | reads `E_<idx>/M` and `E_<idx>/rhs` |
| `growth_factor.py` | stage 3 complete | reads the stored factors |
| `sweep_fp16.py` | stage 3 complete | reads `blockthomas/complex128/x` as its reference |
| `plot_speedup.py` | stage 3 complete | reads the stored timings |
| every other `plot_*.py` | its stage-4 CSV | reads that CSV only |

`run_benchmarks.py` and `gpu_run_benchmarks.py` are the only scripts that
modify a material file. Everything else opens it read-only.

---

## 3. Command-line conventions

Every executable script builds its parser from `solvers/cli.py`, so an option
means the same thing and is spelled the same way wherever it appears. Adding a
script means using those helpers, not inventing a new spelling.

### 3.1 Solver names

One canonical spelling, lower-case kebab, used on every command line and in
every Python API — `bench(solvers=...)`, `mpir.SOLVER_BUILDERS`, the Arnoldi
`--backend` list, and the figure legends:

```
superlu   umfpack   mumps   gmres   gmres-cupy   cudss
block-thomas   block-thomas-inv   block-thomas-fp16   block-thomas-inv-fp16
```

**HDF5 group names are deliberately not renamed.** The groups inside a material
file (`blockthomas`, `blockthomas_inv`, `gmres_scipy`) are on-disk data;
renaming them would make every material file already written unreadable, for no
numerical benefit. The mapping lives in `cli.SOLVERS` and is applied by
`cli.h5_group()` and `cli.from_h5_group()`, so nothing outside `cli.py` needs
to know the stored spelling.

| Canonical name | Stored group | Precision level beneath it |
|---|---|---|
| `superlu` | `superlu` | `complex128`, `complex64` |
| `umfpack` | `umfpack` | `complex128` |
| `mumps` | `mumps` | `complex128`, `complex64` |
| `gmres` | `gmres_scipy` | `complex128`, `complex64` |
| `gmres-cupy` | `gmres_cupy` | `complex128`, `complex64` |
| `cudss` | `cudss` | `complex128`, `complex64` |
| `block-thomas` | `blockthomas` | `complex128`, `complex64` |
| `block-thomas-inv` | `blockthomas_inv` | `complex128`, `complex64` |
| `block-thomas-fp16` | `blockthomas` | `complex32` |
| `block-thomas-inv-fp16` | `blockthomas_inv` | `complex32` |

### 3.2 Precision names

Complex working precisions are spelled in full everywhere: `complex128`,
`complex64`, and `complex32`. The last is a storage label, not a NumPy dtype —
`np.dtype("complex32")` does not exist — and denotes the half-precision
embedded-real factorizations.

The precision in which implementation 2 forms its explicit inverses is a real
dtype and is spelled `float64`, `float32`, `float16`.

### 3.3 Option vocabulary

| Concept | Option | Notes |
|---|---|---|
| material HDF5 file | positional `h5path` | never a flag |
| matrix, right-hand side | positional `matrix`, `rhs` | `.npz` triplet, `.npy` |
| explicit energy indices | `--idx N [N ...]` | mutually exclusive with `--start` |
| index range | `--start S --end E` | inclusive at both ends |
| one solver | `--solver NAME` | canonical names |
| several solvers | `--solvers NAME [NAME ...]` | canonical names |
| working precisions | `--dtypes NAME [NAME ...]` | |
| factorization precision | `--factor-dtype NAME` | `u_f` in the refinement analysis |
| inverse-formation precision | `--inv-dtype NAME` | real dtype |
| uniform block size | `--block-size M` | |
| detected partition | `--auto-blocks` | overrides `--block-size` |
| output directory | `--outdir DIR` | never `--out`, `--plotdir`, `--out-root` |
| output label | `--material NAME` | defaults to the input stem |
| iteration limit | `--max-iter N` | never `--maxiter` |

Where a script takes no index selection it takes none; where it takes one, it
takes exactly this one. `--idx` always accepts several values, so `--idx 25`
and `--idx 0 25 50` are both valid.

### 3.4 Renames from the previous interface

| Was | Now | Where |
|---|---|---|
| `--bs` | `--block-size` | `sweep_fp16`, `single_solve`, `mpir`, `c32_gmres_ir` |
| `--block-sizes` | `--block-size` | `arnoldi_shift_invert_*` |
| `--compare-bs` | `--compare-block-size` | `determine_custom_block_size` |
| `--low-dtype` | `--factor-dtype` | `mpir` |
| `--factor-dtype c128\|c64` | `--factor-dtype complex128\|complex64` | `arnoldi_shift_invert_*` |
| `--dtype` | `--dtypes` | `growth_factor` |
| `--h5path PATH` | positional `h5path` | `sweep_fp16`, `non-normal` |
| `--material-name` | `--material` | `non-normal` |
| `--out-root`, `--out`, `--plotdir` | `--outdir` | `non-normal`, `plot_speedup`, `make_hdf5`, `growth_factor` |
| `--indices 0:401` | `--start 0 --end 401` | `non-normal` |
| `--idx N` (single) | `--idx N [N ...]` | `growth_factor`, `mpir` |
| `--maxiter` | `--max-iter` | `arnoldi_shift_invert_*` |
| `--gmres-maxiter` | `--gmres-max-iter` | `mpir`, `c32_gmres_ir` |
| `-k` | `-k`, `--num-values` | `arnoldi_shift_invert_*` |
| `block_thomas`, `blockthomas` | `block-thomas` | everywhere |
| `block_thomas_inv`, `blockthomas_inv` | `block-thomas-inv` | everywhere |
| `block_thomas_fp16` | `block-thomas-fp16` | everywhere |
| `gmres_cupy` | `gmres-cupy` | everywhere |
| `gmres_scipy` (as a CLI name) | `gmres` | figure scripts |

---

## 4. Directory layout

The three principal directories are strictly layered: library, drivers,
analysis. There is exactly one `bench()` implementation and one copy of each
solver.

| Directory | Role | README |
|---|---|---|
| [`solvers/`](solvers/) | the solver library: every solver behind one interface, the canonical names and shared CLI, the block-partition utilities, HDF5 persistence, tests | [README](solvers/README.md) |
| [`run_bench/`](run_bench/) | batch drivers: load, call `bench()`, write results back | [README](run_bench/README.md) |
| [`block-thomas/`](block-thomas/) | post-hoc stability and spectral analysis, read-only | [README](block-thomas/README.md) |
| [`mixed_prec_ir/`](mixed_prec_ir/) | mixed-precision iterative refinement, LU-IR and GMRES-IR | [README](mixed_prec_ir/README.md) |
| [`plotting/`](plotting/) | every figure in the project; computes nothing | [README](plotting/README.md) |

| Directory | Contents |
|---|---|
| `non-normal/` | departure from normality by full SVD and eigendecomposition |
| `condition_number_plots/` | conditioning studies |
| `pyginkgo/` | Ginkgo binding experiments |
| `jupyter/` | exploratory notebooks and per-material outputs |
| `other/` | superseded `main*.py` variants, retained for reference |
| `qttools/`, `quatrex/` | vendored copies of the two upstream packages, for reference |

---

## 5. Block Thomas

Both implementations of Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., Algorithm 13.3 are provided, each at complex double and
single precision and at half precision, giving four classes in
`solvers/solver_classes.py`:

| | Implementation 1 | Implementation 2 |
|---|---|---|
| complex128 / complex64 | `BlockThomas` | `BlockThomasExplicitInv` |
| half precision | `BlockThomasFP16` | `BlockThomasExplicitInvFP16` |

### 5.1 The factorization

For the block-tridiagonal matrix

```
    A = [ D_0  U_0                    ]
        [ L_0  D_1  U_1               ]
        [      L_1  D_2  ...          ]
        [                ...  U_{N-2} ]
        [           L_{N-2}  D_{N-1}  ]
```

the block LU factorization is `A = L_global U_global` with

```
    L_global = block-bidiagonal(I;      E_k = L_{k-1} D_mod_{k-1}^-1)
    U_global = block-bidiagonal(D_mod;  U_k)
```

where the modified diagonal blocks satisfy the Schur-complement recursion

```
    D_mod_0 = D_0
    D_mod_k = D_k - L_{k-1} D_mod_{k-1}^-1 U_{k-1},        k = 1 .. N-1.
```

The solve is the corresponding two-sweep substitution:

```
    forward     y_0 = b_0
                y_k = b_k - L_{k-1} D_mod_{k-1}^-1 y_{k-1}
    backward    x_{N-1} = D_mod_{N-1}^-1 y_{N-1}
                x_k     = D_mod_k^-1 (y_k - U_k x_{k+1})
```

Both implementations compute the same `D_mod` recursion and the same solve.
They differ solely in how `D_mod_k^-1` is realized.

### 5.2 Implementation 1 — LU with substitution

Each `D_mod_k` is factorized by Gaussian elimination with partial pivoting
(LAPACK `getrf`, through `scipy.linalg.lu_factor`), and every occurrence of
`D_mod_k^-1` is a triangular substitution against those stored factors
(`scipy.linalg.lu_solve`). **No block is ever explicitly inverted.**

Stored per block: the packed LU and its pivot vector, plus `L_k` and `U_k`.

### 5.3 Implementation 2 — explicit inverses

Each `D_mod_k^-1` is formed explicitly, once, and stored. Every operation that
implementation 1 performs as a triangular substitution becomes a dense matrix
product.

Higham's motivation is stated directly: with the inverse available, the forward
sweep becomes a matrix multiplication and the back substitution consists
entirely of matrix-vector products. The solve is then a sequence of GEMM calls,
which is the form throughput-oriented hardware executes most efficiently. This
is why implementation 2, and not implementation 1, is the one that matters for
half precision.

Stored per block: the explicit inverse, plus `L_k` and `U_k`. No LU factors and
no pivot vectors exist, so `get_LUP()` returns `None` and `get_inverses()` is
used instead.

### 5.4 Cost and accuracy comparison

| | Implementation 1 | Implementation 2 |
|---|---|---|
| factorization | `N` block LU factorizations | `N` explicit inversions, roughly 3× the flops of an LU each |
| solve, per stage | triangular substitution | dense matrix product |
| stored per block | packed LU + pivots | explicit inverse |
| accuracy | better; see below | worse; see below |
| hardware fit | poor for tensor cores | good |

By Higham Chapter 14, forming an inverse and multiplying is in general less
accurate than solving by substitution, because a computed inverse satisfies no
small-residual bound of the kind that holds for triangular solves. For
well-conditioned diagonal blocks the difference is modest.

`block-thomas/growth_factor.py` measures the backward stability of both against
SuperLU and UMFPACK, whose global partial pivoting with column ordering is
strictly stronger than the block-local pivoting Block Thomas performs.

### 5.5 Half precision: what is and is not fp16

This is the point on which the half-precision results must be read carefully.

#### Complex arithmetic in a real format

IEEE binary16 has no complex counterpart and NumPy provides no `complex32`.
Complex blocks are handled through the exact real embedding

```
    z = a + bi  ->  [[a, -b], [b, a]],
```

which is a ring isomorphism: complex addition and multiplication map onto their
real matrix counterparts exactly, so the embedding introduces no error. A
complex `m × m` block becomes a real `2m × 2m` block and a rectangular `(p, q)`
coupling block becomes `(2p, 2q)`, so custom non-uniform partitions carry over
unchanged. The cost is a factor of two in each dimension.

#### The three categories of non-fp16 operation

Every quantity **stored** by these classes is `float16`: the embedded
off-diagonal blocks, the packed LU factors or the explicit inverses, and the
intermediate right-hand sides during a solve. Every arithmetic **result** is
rounded to `float16` before being used again. Three categories of operation are
nevertheless not performed in `float16`, and only the last affects accuracy.

**1. Accumulation inside a product — wider, deliberately, and physically
correct.** NumPy evaluates a `float16` operation by promoting to `float32`,
computing, and rounding back. A matrix product therefore accumulates in
`float32` and rounds once per output entry, rather than rounding after every
fused multiply-add. This is not an approximation adopted for convenience: it is
exactly the accumulation model of tensor cores and of the mixed-precision GEMM
instructions such a factorization would target in practice, so the simulated
arithmetic matches the hardware being modelled. It is, however, more accurate
than a hypothetical implementation rounding to `float16` after every scalar
addition, and results should be read as characterising tensor-core arithmetic
rather than idealised half-precision arithmetic.

**2. Scalings — wider, but exact, so without accuracy consequence.** Every
scale factor is an exact power of two, chosen by `_pow2_scale`. Multiplying by
a power of two changes only the exponent field, so in the absence of overflow
or underflow it is exact in binary floating point. These multiplications are
carried out in `float32` or `float64` because the reciprocal of a scale is
frequently subnormal, or infinite, in `float16` and would be destroyed by the
format before it could be applied. Since the operations are exact, the wider
format changes no digit of any result; it only prevents the format from losing
the factor. The same holds for the row-sum norms used to choose the overflow
guard, computed in `float32`: they select an exponent and enter no result.

**3. Formation of the explicit inverse — the one genuine departure.**
`BlockThomasExplicitInvFP16` takes an `inv_dtype` parameter:

| `inv_dtype` | Behaviour | Purity |
|---|---|---|
| `np.float32` (default) | the modified diagonal block is promoted to `float32`, inverted by LAPACK, and the result rounded back to `float16` for storage and application | **not** purely half precision |
| `np.float16` | the inverse is formed by `lu_fp16` followed by substitution against the identity | half precision throughout |

The default is `float32` because explicit inversion is the least stable step in
the algorithm: performing it in `float16` loses accuracy that no subsequent
step recovers, whereas the higher-precision inversion consists of `O(N)` small
dense inversions and leaves both the stored data and the precision of the solve
unchanged. `inv_dtype=np.float16` exists so that the value of the
higher-precision inversion is *measured* rather than assumed;
`run_bench/sweep_fp16.py --inv-dtype float16` performs that measurement.

**Implementation 1 has no such parameter.** It is half precision in all three
categories, subject only to point 1.

#### Summary

| Class | Storage | Products | Inversion |
|---|---|---|---|
| `BlockThomasFP16` | fp16 | fp16 with fp32 accumulation | not applicable |
| `BlockThomasExplicitInvFP16`, default | fp16 | fp16 with fp32 accumulation | **fp32** |
| `BlockThomasExplicitInvFP16`, `inv_dtype=float16` | fp16 | fp16 with fp32 accumulation | fp16 |

#### Range control

The normal range of binary16 is `[6.10e-5, 65504]`, about five decades, against
616 for binary64. Neither the QTBM blocks nor the intermediate quantities of
the algorithm lie inside it, so three separate power-of-two scalings are
applied.

1. **Global scale `s`.** Brings the largest entry of the embedded matrix to
   about 1024, leaving a factor of 64 below overflow. Taken over the diagonal
   *and* both off-diagonals: the coupling blocks may carry the largest entries,
   and scaling on the diagonal alone would allow them to overflow.

2. **Per-block inverse scales `t[k]`** (implementation 2 only). Inverses scale
   like `1 / |D_mod|`, that is, in the direction opposite to `A`, so the same
   global scale that brings `A` into range drives its inverses towards
   underflow. Each stored inverse therefore carries its own scale:
   `G[k] = t[k] * D_mod_hat[k]^-1`.

3. **Matmul overflow guard.** A dense fp16 product overflows purely from the
   accumulation length, independently of the magnitude of the wanted result:
   with entries near `1e3` and `bs = 416`, which embeds to an accumulation of
   length 832, the exact intermediate is near `1e9` against a ceiling of 65504,
   even when the quantity the algorithm needs is `O(1)`. `_matmul16` rescales by
   a power of two derived from the rigorous bound
   `|(A X)_ij| <= ||A||_inf max|X|`. This is why `||L||_inf`, `||U||_inf` and
   `||G||_inf` are precomputed at construction.

Because all three use exact powers of two, they introduce no rounding error;
they only move where the computation sits in the exponent range.

---

## 6. Mixed-precision iterative refinement

`mixed_prec_ir/` addresses the complementary question: if a reduced-precision
factorization is not accurate enough on its own, can it still be *used*, as the
inner solver or the preconditioner of a refinement scheme?

### 6.1 The three precisions

Iterative refinement solves `A x = b` by computing a solution in a low
precision and correcting it using residuals computed in a higher one. The
modern analysis (Carson and Higham, 2017 and 2018) distinguishes:

| Symbol | Meaning | Value here |
|---|---|---|
| `u_f` | precision of the factorization | `--low-dtype`, `complex64` or half |
| `u` | working precision, in which `x` and the corrections are held | `complex128` |
| `u_r` | precision of the residual computation | `complex128` |

The classical result of Wilkinson and of Moler is that when the residual is
computed more accurately than the factorization, refinement recovers a forward
error governed by `u` rather than by `u_f`. This is the whole motivation: a
factorization is `O(n^3)` and a residual is `O(nnz)`, so accuracy characteristic
of a high-precision factorization is obtained at the cost of a low-precision
one.

### 6.2 The algorithms

Both variants share the outer loop and differ only in step 4.

```
    1. Build the solver at u_f.                    one-time factorization cost
    2. x = solver.solve(b), promoted to u.
    3. r = b - A x                                 computed at u_r
    4. Solve A dx = r                              see below
    5. x = x + dx
       repeat from 3 until ||r||/||b|| < tol, or max_iter is reached
```

**LU-IR** (`--inner direct`; Buttari et al., 2006). Step 4 is
`dx = solver.solve(r)`: a single low-precision triangular substitution reusing
the same factorization. One triangular solve per outer iteration.

**GMRES-IR** (`--inner gmres`; Carson and Higham, 2017). Step 4 solves
`A dx = r` by GMRES applied to `A` in `complex128`, left-preconditioned by
`M^-1 v = solver.solve(v)`, with `v` cast down to `u_f` and the result cast
back up. GMRES itself runs at the working precision; only the preconditioner
applications are low-precision solves, and they reuse the same factorization.

### 6.3 Why both are implemented

The practical question is the condition under which refinement converges.

| Variant | Approximate requirement | Limit at `u_f` = fp32 | at `u_f` = fp16 |
|---|---|---|---|
| LU-IR | `kappa_inf(A) u_f < 1` | `kappa ~ 1e7` | `kappa ~ 1e3` |
| GMRES-IR | substantially weaker | far higher | far higher |

QTBM matrices near a band edge exceed both LU-IR limits. GMRES-IR relaxes the
requirement because the inner Krylov method is not obliged to accept whatever
the low-precision factors produce; it iterates on the preconditioned system at
the working precision. This is what makes refinement applicable at these
condition numbers at all, and it is why `kappa_2(M)` is reported alongside
every result.

A second property matters for this project specifically: the GMRES
preconditioner requires only the *action* of the factorization as an operator,
never explicit access to `L` and `U`. The same code therefore drives SuperLU,
Block Thomas, MUMPS and cuDSS identically — including the two solvers that
expose no factors at all.

### 6.4 What is measured

`mpir.py` compares three variants of the **same** solver family, so that the
comparison isolates the effect of precision and refinement rather than of the
solver implementation:

1. the solver at `u_f` with refinement — the method under test
2. the solver at `complex128`, no refinement — the accuracy reference
3. the solver at `u_f`, no refinement — the lower bound refinement must beat

`c32_gmres_ir.py` extends this to the half-precision Block Thomas
factorization, adding two further reference points, and sweeps an energy range.
It does not reimplement the refinement drivers: it registers one additional
builder into `mpir.SOLVER_BUILDERS` at run time, so any correction to the
refinement logic applies to it automatically.

The evidence for whether half-precision preconditioning works is the
convergence history and the inner iteration counts, not any single final
number; both are recorded per index.

---

## 7. References

- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
  SIAM, 2002. Algorithm 13.3 and its two implementations; Chapter 14 on the
  accuracy of explicit inversion.
- J. H. Wilkinson, *Rounding Errors in Algebraic Processes*, 1963.
- A. Buttari et al., Mixed precision iterative refinement techniques for the
  solution of dense linear systems, *IJHPCA* 21(4), 2007.
- E. Carson and N. J. Higham, A new analysis of iterative refinement and its
  application to accurate solution of ill-conditioned sparse linear systems,
  *SIAM J. Sci. Comput.* 39(6), 2017.
- E. Carson and N. J. Higham, Accelerating the solution of linear systems by
  iterative refinement in three precisions, *SIAM J. Sci. Comput.* 40(2), 2018.

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
has to correlate two files by index. The same convention is applied one level
further out: each analysis directory holds one HDF5 file per material, into
which every analysis of that material writes its own top-level group.

**HDF5 is the only result format.** No script writes CSV, and none writes a
`.npy` result array.

### 2.1 Stages

```
  STAGE 1  EXPORT                    export_qtbm_systems.py
           QTBM example directory       main3.py  /  main3_gpu.py
                    │
                    │  M(E), Sigma(E), rhs per energy index, as .npz / .npy
                    │  into  matrices2/<material>/
                    ▼
  STAGE 2  CONSOLIDATE              make_hdf5.py
                    │
                    │  one <material>.h5 into  matrices2/hdf5/
                    ▼
        ┌────────────────────────────────────────────────────┐
        │  matrices2/hdf5/<material>.h5                      │
        │    metadata/indices, metadata/energies             │
        │    metadata attrs: band edges, grid resolution     │
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
  STAGE 4  ANALYSE (reads the material file, writes an analysis file)  │
                    │                                                  │
    block-thomas/growth_factor.py ─► error-analysis-block-thomas/      │
                                       <material>.h5 :/growth_factor   │
    block-thomas/forward_error.py ─► the same file :/forward_error     │
    run_bench/sweep_fp16.py ───────► error-analysis-block-thomas/ ◄────┘
                                       <material>.h5 :/fp16_sweep
    mixed_prec_ir/mpir.py ─────────► mixed-precision-IR/<material>/
                                       <material>.h5 :/experiments/<NNNN>/
                                                        runs, iterations
    mixed_prec_ir/mpperf.py ───────► mixed-precision-IR/<material>/
                                       <material>_perf.h5 :/experiments/<NNNN>/
                                                        runs
    non-normal/non-normal.py ──────► non-normal/
                                       <material>.h5 :/non_normality
                    │
                    ▼
  STAGE 5  PLOT (reads HDF5, writes figures only, into the same directory)
                    plotting/block-thomas/plot_growth_factor.py
                    plotting/block-thomas/plot_forward_error.py
                    plotting/block-thomas/plot_backward_error.py
                    plotting/block-thomas/plot_fp16_accuracy.py
                    plotting/mixed_prec_ir/plot_mpir.py
                    plotting/mixed_prec_ir/plot_mpperf.py
                    plotting/non-normal/plot_non_normal.py
                    plotting/matrices2/plot_qtbm_spectra.py
                    plotting/materials/bandstructure.py
```

`plotting/` mirrors this same scratch-directory layout: one subfolder per
category (`block-thomas/`, `condition-est/`, `non-normal/`, `mixed_prec_ir/`,
`matrices2/`, `materials/`), holding the scripts that plot that category's
data. `style.py` is the only file at `plotting/`'s top level.

### 2.2 The invariant

**Computation and visualization are strictly separated.** No script that
solves, factorizes or measures anything imports matplotlib. Every quantity a
figure displays is first written to an HDF5 dataset, and every figure is
produced by a script in `plotting/` that reads that dataset and computes
nothing.

This has three consequences worth stating explicitly. A sweep that takes hours
on the cluster is never repeated in order to change an axis label. The numbers
behind every figure exist as a file that can be inspected, cited or replotted.
And a plotting change cannot alter a measurement.

Each figure is written to the directory holding the data it was drawn from, so
a result and its figure are never separated.

### 2.3 Directories on the cluster

Every default path in the project is defined in `solvers/cli.py` and nowhere
else. The analysis directories are named after the thesis chapter whose figures
they hold, so the mapping from a result file to the text that uses it is direct.

| Constant | Path | Holds |
|---|---|---|
| `cli.EXAMPLES_DIR` | `/scratch/yimili/examples` | QTBM/DFT inputs, the stage 1 source |
| `cli.EXPORT_DIR` | `/scratch/yimili/matrices2` | stage 1 output, one directory per material |
| `cli.HDF5_DIR` | `/scratch/yimili/matrices2/hdf5` | stage 2 and 3, `<material>.h5` |
| `cli.BLOCK_THOMAS_DIR` | `/scratch/yimili/error-analysis-block-thomas` | growth factor, fp16 sweep, solver timings |
| `cli.CONDITION_DIR` | `/scratch/yimili/condition-est` | conditioning and singular-value figures |
| `cli.NON_NORMAL_DIR` | `/scratch/yimili/non-normal` | non-normality sweep, frames, animation |
| `cli.MIXED_PREC_DIR` | `/scratch/yimili/mixed-precision-IR` | iterative refinement sweeps |
| `cli.MATERIALS_DIR` | `/scratch/yimili/materials` | contact band structures, one directory per material |
| `cli.RANDOM_DIR` | `/scratch/yimili/random` | synthetic test matrices |

`cli.material_h5(material)` gives the stage-3 file and
`cli.analysis_h5(outdir, material)` the stage-4 file. Setting the environment
variable `QTBM_SCRATCH` relocates the whole tree, which is how the pipeline is
exercised outside the cluster; every path above derives from it.

Every script takes `--outdir` to override its own default.

`EXPORT_DIR` points at `matrices2`, the current export. An earlier export under
a different directory is read by passing it explicitly, for example
`python make_hdf5.py /scratch/yimili/matrices --all`; nothing in the pipeline
writes to it.

### 2.4 Materials and the energy grid

`cli.MATERIALS` is the one place a per-material property is edited. Each entry
carries the band edges, the Block Thomas block size, the input directories and
the contact band structure parameters, and every stage reads them from there:
`main3.py` positions the energy sweep on the conduction edge, `make_hdf5.py`
records both edges in the material file, `run_benchmarks.py` takes the block
size, and `plotting/materials/bandstructure.py` takes the inputs and the
mid-gap energy.

```python
"graphene": Material(
    inputs=EXAMPLES_DIR / "graphene/inputs",
    block_size=416,
    valence_band_edge=None,          # set once determined
    conduction_band_edge=-2.4,
    grid=EnergyGrid(start=-3.405, end=-1.400, resolution=0.005),
    mid_gap_energy=0.5,
),
```

A band edge left at `None` is not fatal: stage 1 falls back to the value in the
QTBM configuration, and stage 2 records only the edges it has. Run
`plotting/materials/bandstructure.py <material>` to determine them; it reports
both edges and the gap.

**The energy sweep is a first energy, a last energy and a step, all in eV.** It
is not derived from the band edges, and the number of energy indices is stated
nowhere; it follows from the three numbers:

| `grid` | Indices |
|---|---|
| `EnergyGrid(start=-3.405, end=-1.400, resolution=0.005)` | 402, the present setting |
| `EnergyGrid(start=-3.405, end=-1.400, resolution=0.0025)` | 803 |
| `EnergyGrid(start=-3.000, end=-2.000, resolution=0.005)` | 201 |

Changing it is the only edit required: stage 1 exports that many matrices,
stage 2 discovers how many were exported rather than assuming a count, and
every later stage reads the index list from `metadata/indices`. `main3.py`
additionally has `EXPORT_STRIDE`, which subsamples the grid for a quick pass
without changing the grid itself.

`end` is included only if it is an exact number of steps above `start`;
otherwise the grid stops at the last sample below it, and `EnergyGrid.last`
reports that sample. `EnergyGrid.around(centre, half_window, resolution)`
constructs a grid centred on a band edge, for a sweep more naturally described
that way; it stores the same three numbers.

Stage 1 saves the grid it assembled the matrices from, and stage 2 reads that
array back rather than recomputing it, so the energies recorded against an
index cannot disagree with the matrix stored under it. The registry is
consulted at stage 2 only for exports that predate that file.

Note that a sweep positioned on the conduction edge will usually not contain
the valence edge, which is then absent from the figures; widen `start` to
include it.

### 2.5 Complete worked example

```bash
# Stage 1-2: export and consolidate. Cluster-side; needs the QTBM examples.
python export_qtbm_systems.py <example> --mode full --energy-index 0
python make_hdf5.py --all

# Stage 3: solve every index with every solver, both precisions.
#          MUTATES matrices2/hdf5/graphene.h5 in place.
cd run_bench && python run_benchmarks.py

# Stage 4: analysis. Read-only with respect to the material file.
cd ../block-thomas
python growth_factor.py /scratch/yimili/matrices2/hdf5/graphene.h5 --stride 20
python forward_error.py /scratch/yimili/matrices2/hdf5/graphene.h5 --stride 20
cd ../run_bench
python sweep_fp16.py /scratch/yimili/matrices2/hdf5/graphene.h5 --stride 20
cd ../mixed_prec_ir
# One invocation per solver and precision under test; each appends a new
# numbered experiment to mixed-precision-IR/graphene/graphene.h5 and never
# overwrites, so the file records every run made.
python mpir.py /scratch/yimili/matrices2/hdf5/graphene.h5 \
    --stride 20 --solver block-thomas \
    --factor-dtype complex32 --inner gmres
# The companion performance study: one invocation covers every solver, since
# the comparison is between them. Writes graphene_perf.h5 in the same
# directory. A handful of indices, not a sweep -- each one is eight bars.
python mpperf.py /scratch/yimili/matrices2/hdf5/graphene.h5 \
    --idx 84 254 601 --solvers block-thomas mumps cudss superlu
cd ../non-normal
python non-normal.py /scratch/yimili/matrices2/hdf5/graphene.h5 --stride 20

# Stage 5: figures, each written beside the data it was drawn from.
# plotting/ mirrors the same category layout as the scratch directories.
cd ../plotting/block-thomas
python plot_growth_factor.py  /scratch/yimili/error-analysis-block-thomas/graphene.h5
python plot_forward_error.py  /scratch/yimili/error-analysis-block-thomas/graphene.h5
python plot_backward_error.py /scratch/yimili/error-analysis-block-thomas/graphene.h5
python plot_fp16_accuracy.py  /scratch/yimili/error-analysis-block-thomas/graphene.h5
cd ../mixed_prec_ir
python plot_mpir.py           /scratch/yimili/mixed-precision-IR/graphene/graphene.h5 --list
python plot_mpperf.py         /scratch/yimili/mixed-precision-IR/graphene/graphene_perf.h5
cd ../non-normal
python plot_non_normal.py     /scratch/yimili/non-normal/graphene.h5

# Tests: synthetic data only, no cluster files required.
cd ../solvers && python test_pipeline.py
```

### 2.6 Ordering constraints

| Stage | Requires | Reason |
|---|---|---|
| `run_benchmarks.py` | stage 2 complete | reads `E_<idx>/M` and `E_<idx>/rhs` |
| `growth_factor.py` | stage 3 complete | reads the stored factors |
| `forward_error.py` | stage 3 complete, and `condition_est.py` for the bound columns | reads the stored solutions and backward errors |
| `sweep_fp16.py` | stage 3 complete | reads `blockthomas/complex128/x` as its reference |
| every other `plot_*.py` | its stage-4 group | reads that group only |

`run_benchmarks.py` and `gpu_run_benchmarks.py` are the only scripts that
modify a material file. Everything else opens it read-only.

### 2.7 The analysis file

An analysis directory holds one file per material. Each stage-4 script writes
one top-level group into it, deleting and rewriting only its own group, so
re-running one analysis leaves the others intact:

```
    /scratch/yimili/error-analysis-block-thomas/graphene.h5
    ├── growth_factor/    idx, solver, dtype, norm, nA, nL, nU, ...
    ├── forward_error/    idx, solver, dtype, fwd_inf, bound_nw, bound_cw, ...
    └── fp16_sweep/       idx, relres_fp16, ..., cond_full_svd
```

A group holds one 1-D dataset per column, all of the same length, and carries
the run configuration as attributes: the source material file, the index range,
the block partition, the tolerances, and the indices that failed or were
skipped. `non_normality` is the exception in shape only: its per-rank
quantities are `(P, n)` matrices chunked one row per index, which is what makes
its sweep resumable.

`factor_io.save_table` and `factor_io.load_table` are the only readers and
writers of this format.

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
| analysis HDF5 file | positional `h5path` | the plotting scripts; same spelling |
| matrix, right-hand side | positional `matrix`, `rhs` | `.npz` triplet, `.npy` |
| explicit energy indices | `--idx N [N ...]` | mutually exclusive with `--start` |
| index range | `--start S --end E` | inclusive at both ends |
| one solver | `--solver NAME` | canonical names |
| several solvers | `--solvers NAME [NAME ...]` | canonical names |
| working precisions | `--dtypes NAME [NAME ...]` | |
| factorization precision | `--factor-dtype NAME` | `u_f` in the refinement analysis |
| inverse-formation precision | `--inv-dtype NAME` | real dtype |
| output directory | `--outdir DIR` | never `--out`, `--plotdir`, `--out-root` |
| output label | `--material NAME` | defaults to the input stem |
| iteration limit | `--max-iter N` | never `--maxiter` |

Where a script takes no index selection it takes none; where it takes one, it
takes exactly this one. `--idx` always accepts several values, so `--idx 25`
and `--idx 0 25 50` are both valid.

**There is no block-partition option.** Every driver detects the Block Thomas
partition from the sparsity pattern of the matrix it was given; the analysis
scripts read back the partition the solver recorded. A uniform block size is
never assumed and cannot be requested.

**Index ranges are a property of the file, never of a script.** With no index
selection, a driver sweeps whatever the file holds, resolved through
`cli.available_indices`, which prefers `metadata/indices` and falls back to
scanning the `E_<idx>` groups. A full-resolution material is a few thousand
indices — it follows from the material's `EnergyGrid` and is written down
nowhere — so `--stride` is how a sweep is shortened, not a hard-coded `--end`.

### 3.4 Renames from the CSV interface

| Was | Now | Where |
|---|---|---|
| positional `csv_path` | positional `h5path` | `plot_growth_factor`, `plot_fp16_accuracy` |
| positional `data_dir` | positional `h5path` | `plot_non_normal` |
| `--no-csv` | `--no-save` | `growth_factor` |
| `--save-csv` | removed | `non-normal`; the HDF5 group replaces it |
| required positional `folder` | optional, defaults to `cli.EXPORT_DIR` | `make_hdf5` |

### 3.5 Earlier renames

| Was | Now | Where |
|---|---|---|
| `--bs`, `--block-sizes`, `--block-size`, `--auto-blocks` | *removed*; the partition is always detected | `sweep_fp16`, `single_solve`, `mpir`, `arnoldi_shift_invert_*` |
| `--compare-bs` | `--compare-block-size` | `determine_custom_block_size` |
| `--low-dtype` | `--factor-dtype` | `mpir` |
| `--factor-dtype c128\|c64` | `--factor-dtype complex128\|complex64` | `arnoldi_shift_invert_*` |
| `--dtype` | `--dtypes` | `growth_factor` |
| `--h5path PATH` | positional `h5path` | `sweep_fp16`, `non-normal` |
| `--material-name` | `--material` | `non-normal` |
| `--out-root`, `--out`, `--plotdir` | `--outdir` | `non-normal`, `make_hdf5`, `growth_factor` |
| `--indices 0:401` | `--start 0 --end 401` | `non-normal` |
| `--idx N` (single) | `--idx N [N ...]` | `growth_factor`, `mpir` |
| `--maxiter` | `--max-iter` | `arnoldi_shift_invert_*` |
| `--gmres-maxiter` | `--gmres-max-iter` | `mpir` |
| `c32_gmres_ir.py` | `mpir.py --factor-dtype complex32` | `mixed_prec_ir` |
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
| [`solvers/`](solvers/) | the solver library: every solver behind one interface, the canonical names, shared CLI and scratch paths, the block-partition utilities, HDF5 persistence, tests | [README](solvers/README.md) |
| [`run_bench/`](run_bench/) | batch drivers: load, call `bench()`, write results back | [README](run_bench/README.md) |
| [`block-thomas/`](block-thomas/) | post-hoc stability and spectral analysis, read-only | [README](block-thomas/README.md) |
| [`mixed_prec_ir/`](mixed_prec_ir/) | mixed-precision iterative refinement: accuracy (`mpir.py`) and runtime (`mpperf.py`) | [README](mixed_prec_ir/README.md) |
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

The solver family and the factorization precision are chosen independently:
`--solver` selects the implementation, `--factor-dtype` the precision it runs
at. `complex128` and `complex64` are available to every solver that has them,
and `complex32` — the embedded-real `float16` factorization — to `block-thomas`
and `block-thomas-inv`, the only two implementations that have one. Which
pairings exist is read from `cli.SOLVERS` and checked at the parser; see
[`mixed_prec_ir/README.md`](mixed_prec_ir/README.md).

The evidence for whether half-precision preconditioning works is the
convergence history and the inner iteration counts, not any single final
number; both are printed per index and written to the experiment's
`iterations` table.

The outer loop stops on the four criteria of Oktay and Carson rather than on a
residual tolerance, and its convergence is reported through the factor of
Carson and Higham's Corollary 3.3, split into a conditioning term and a
correction-solver term. Every one of those quantities is reconstructed after
the run from the arrays the loop retained, never inside it, so the timing and
memory rows measure only refinement itself; see
[`mixed_prec_ir/README.md`](mixed_prec_ir/README.md).

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

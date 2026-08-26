# `mixed_prec_ir/` — mixed-precision iterative refinement

If a reduced-precision factorization is not accurate enough on its own, can it
still be *used*, as the inner solver or the preconditioner of a refinement
scheme? These scripts answer that for the QTBM systems.

All scripts here take the canonical option names; see
[the top-level README, section 3](../README.md#3-command-line-conventions).

| Script | Scope |
|---|---|
| `mpir.py` | the whole study: LU-IR and GMRES-IR over any solver in `../solvers/`, at any precision that solver supports |
| `sparse.py` | the earlier standalone study: fp32 SuperLU with fp64 refinement |
| `dense.py` | the earlier standalone study: fp32 LAPACK with fp64 refinement |

`mpir.py` writes its results into the `mpir` group of the analysis file; see
[section 6](#6-output). No figures are produced here.

---

## 1. The three precisions

Iterative refinement solves `A x = b` by computing a solution in a low
precision and correcting it using residuals computed in a higher one. The
modern analysis (Carson and Higham, 2017 and 2018) distinguishes:

| Symbol | Meaning | Value here |
|---|---|---|
| `u_f` | precision of the factorization | `--factor-dtype`; `complex64` or `complex32` |
| `u` | working precision, in which `x` and the corrections are held | `complex128` |
| `u_r` | precision of the residual computation | `complex128` |

The classical result of Wilkinson and of Moler is that when the residual is
computed more accurately than the factorization, refinement recovers a forward
error governed by `u` rather than by `u_f`. This is the entire motivation: a
factorization is `O(n^3)` and a residual is `O(nnz)`, so accuracy characteristic
of a high-precision factorization is obtained at the cost of a low-precision
one.

---

## 2. The two algorithms

Both share the outer loop and differ only in step 4.

```
    1. Build the solver at u_f.                    one-time factorization cost
    2. x = solver.solve(b), promoted to u.
    3. r = b - A x                                 computed at u_r
    4. Solve A dx = r                              see below
    5. x = x + dx
       repeat from 3 until ||r||/||b|| < tol, or max_iter is reached
```

**LU-IR** — `--inner direct`, Buttari et al. 2006. Step 4 is
`dx = solver.solve(r)`: a single low-precision triangular substitution reusing
the same factorization. One triangular solve per outer iteration.

**GMRES-IR** — `--inner gmres`, Carson and Higham 2017. Step 4 solves
`A dx = r` by GMRES applied to `A` in `complex128`, left-preconditioned by
`M^-1 v = solver.solve(v)`, with `v` cast down to `u_f` and the result cast
back up. GMRES itself runs at the working precision; only the preconditioner
applications are low-precision solves, and they reuse the same factorization.

---

## 3. Why both are implemented

The practical question is the condition under which refinement converges.

| Variant | Approximate requirement | Limit at `u_f` = c64 | at `u_f` = c32 |
|---|---|---|---|
| LU-IR | `kappa_inf(A) u_f < 1` | `kappa ~ 1e7` | `kappa ~ 1e3` |
| GMRES-IR | substantially weaker | far higher | far higher |

QTBM matrices near a band edge exceed both LU-IR limits. GMRES-IR relaxes the
requirement because the inner Krylov method is not obliged to accept whatever
the low-precision factors produce; it iterates on the preconditioned system at
the working precision. This is what makes refinement applicable at these
condition numbers at all, and it is why `kappa_2(M)` is loaded from
`global/condition_full_svd` and reported alongside every result.

A second property matters here specifically: **the GMRES preconditioner
requires only the action of the factorization as an operator**, never explicit
access to `L` and `U`. The same code therefore drives SuperLU, Block Thomas,
MUMPS and cuDSS identically, including the two solvers that expose no factors
at all.

---

## 4. `mpir.py`

Basic: one solver, one precision, default inner solve (LU-IR).

```bash
python mpir.py .../carbon-nanotube.h5 --idx 5 --solver superlu \
    --factor-dtype complex64
```

GPU, cuDSS at `complex64` — cuDSS has no `complex32` factorization; see below.

```bash
python mpir.py .../si-bulk.h5 --idx 254 --solver cudss --factor-dtype complex64
```

GMRES-IR instead of LU-IR, with the inner solve's own tolerance, restart and
iteration cap. Needed once the condition number exceeds the LU-IR bound; see
section 3.

```bash
python mpir.py .../si-bulk.h5 --idx 254 --solver mumps --factor-dtype complex64 \
    --inner gmres --gmres-tol 1e-8 --gmres-restart 30 --gmres-max-iter 50
```

Half precision (`complex32`), CPU only — Block Thomas, LU with substitution.

```bash
python mpir.py .../carbon-nanotube.h5 --idx 84 --solver block-thomas \
    --factor-dtype complex32 --inner gmres
```

Half precision, the explicit-block-inverse implementation, with its inverse
formed at `float16` instead of the `float32` default — the factorization is
then half precision throughout, at some cost in accuracy; see
[`../solvers/README.md`](../solvers/README.md).

```bash
python mpir.py .../carbon-nanotube.h5 --idx 84 --solver block-thomas-inv \
    --factor-dtype complex32 --inner gmres --inv-dtype float16
```

By energy rather than index — resolves to the nearest recorded energy;
mutually exclusive with `--idx`/`--start`/`--end`.

```bash
python mpir.py .../si-bulk.h5 --energy 6.1 6.2 6.3 \
    --solver cudss --factor-dtype complex64 --inner gmres
```

Sweeping a range. Each index prints its own full report; nothing is written to
disk (see section 6).

```bash
python mpir.py .../carbon-nanotube.h5 --stride 10 \
    --solver block-thomas --factor-dtype complex32 --inner gmres
```

Repeats with the median reported, for the noisier GPU timings.

```bash
python mpir.py .../si-bulk.h5 --idx 254 --solver cudss \
    --factor-dtype complex64 --repeats 5
```

A different reference solver — `--reference-solver` defaults to `mumps` and
supplies both `x_true` and the reference corrections `phi_solve` is measured
against.

```bash
python mpir.py .../si-bulk.h5 --idx 254 --solver cudss \
    --factor-dtype complex64 --reference-solver cudss
```

Loosening the stopping criteria, and writing the tables somewhere else.

```bash
python mpir.py .../si-bulk.h5 --idx 254 --solver cudss \
    --factor-dtype complex64 --inner gmres \
    --rho-thresh 0.9 --max-iter 30 --k-max 0 --ferr-thresh 0 --outdir ./scratch
```

Three variants of the **same** solver family are measured, so the comparison
isolates the effect of precision and refinement rather than of the solver
implementation:

1. the solver at `u_f` with refinement — the method under test
2. the solver at `complex128`, no refinement — the accuracy reference
3. the solver at `u_f`, no refinement — the lower bound refinement must beat

Reported per variant: relative residual, forward error against the reference
solution, the normwise and componentwise backward errors, wall time split into
factorization and solve, factor memory, the outer convergence history, and, for
GMRES-IR, the inner iteration counts.

### Stopping criteria

The loop does **not** stop on a fixed residual tolerance. It uses the four
criteria of [Oktay and Carson, section 2.1.1](#references), which read the
corrections the loop already produces and so cost nothing extra, plus one more
of the same kind:

| # | Condition | Option | Meaning |
|---|---|---|---|
| 1 | `‖d_{i+1}‖ / ‖x_i‖ ≤ u` | — | the correction no longer moves the iterate |
| 2 | `‖d_{i+1}‖ / ‖d_i‖ ≥ ρ_thresh` | `--rho-thresh` (0.5) | corrections stopped shrinking geometrically; a ratio above 1 is divergence |
| 3 | `iter ≥ i_max` | `--max-iter` (10) | |
| 4 | `k_GMRES ≥ k_max` | `--k-max` (`ceil(0.1n)`) | one outer step now costs about what refactorizing would; GMRES-IR only, `0` disables |
| 5 | `ferr_i / ferr_{i-1} ≥ ferr_thresh` | `--ferr-thresh` (1.0) | the forward error against the reference solution stopped improving; needs `--reference-solver`, `0` or negative disables |

**Condition 5 is not from Oktay and Carson.** It measures the thing that
actually matters rather than a proxy for it: the reference solution is the
best available estimate of the exact answer, so once the iterate stops getting
closer to it there is nothing left to gain, whatever the correction norms are
doing. It is a cheap `O(n)` vector-norm comparison — exactly like conditions 1
and 2, not an extra solve — computed from the same iterates already retained.
It complements rather than duplicates 1 and 2: a correction can keep looking
healthy by every internal measure while `ferr` has already reached the
reference's own accuracy floor (condition 5 catches this, conditions 1–2
don't), and conversely a nonnormal matrix can make corrections look erratic
well before `ferr` actually stops improving (conditions 1–2 catch this,
condition 5 doesn't). When both are true at once — typical once rounding noise
dominates — they fire together and `stop_reason` names both.

Convergence itself is declared on the normwise error estimate of Demmel et al.
that Oktay and Carson carry alongside them,

```
phi = z / (1 - rho_max),   z = ‖d_{i+1}‖/‖x_i‖,  rho_max = max_j ‖d_{j+1}‖/‖d_j‖
```

with convergence when `0 ≤ phi ≤ sqrt(n) u`. Both bounds matter: `phi` is
negative exactly when `rho_max > 1`, that is when a correction grew, and that
is divergence rather than a small error.

`u` here is the **working** precision for every variant, never `u_f`. The
question the criteria answer is whether refinement reached the accuracy the
working precision can hold; asking it at `u_f` would accept a solution only as
good as the factorization refinement exists to improve on. `u_f` enters the
metrics below as `u_s`, and the stopping test never.

Two consequences are worth expecting rather than being surprised by. Condition
2 is what normally ends a healthy run: refinement reaches the floor set by
rounding and the reference solution, the next correction fails to shrink, and
the loop stops — often one step after the accuracy stopped improving. And a run
that has genuinely converged to the reference floor may still fail the `phi`
convergence test when that floor sits above `sqrt(n) u`, since the reference
solution is itself only accurate to its own backward error, which is reported
beside it. The console reports the plain outer-iteration count and the reason
the loop stopped rather than a converged/failed verdict, since which stopping
condition fired is more informative than a binary label, and the count is what
you need to compute an average convergence rate over a sweep. `converged` is
still recorded as its own column in `runs` for filtering.

### Convergence metrics

The refinement variant is reported through the quantities of [Carson and
Higham, Corollary 3.3](#references) — the right-hand panels of their numerical
experiments:

```
phi_i = 2 u_s min(cond(A), kappa_inf(A) mu_i)  +  u_s ‖E_i‖_inf
        └────────── phi_cond ──────────────┘     └── phi_solve ──┘
```

| Metric | Meaning |
|---|---|
| `ferr_ref` | `‖x_ref − x_i‖_∞ / ‖x_ref‖_∞`, the forward error **relative to the reference solution**; not a certified forward error |
| `rho` | `ferr_ref[i+1] / ferr_ref[i]`, the contraction actually observed |
| `mu_hat` | `‖A(x_ref − x_i)‖_∞ / (‖A‖_∞ ‖x_ref − x_i‖_∞)`; small exactly when the error points along the directions `A` damps, which is what lets refinement converge past the plain bound |
| `phi_cond_hat` | `2 u_s kappa_inf(A) mu_hat` |
| `phi_solve_hat` | `‖d_i − d_i^ref‖_∞ / ‖d_i^ref‖_∞`, against a reference solve of the *same* retained residual |
| `phi_hat` | their sum, the estimate of the Corollary's `phi_i` |

`u_s` is the effective precision of the correction solve: `u_f` for LU-IR,
whose correction is whatever the low-precision factors return, and `u` for
GMRES-IR, whose inner solve runs at the working precision. That difference is
the whole reason GMRES-IR tolerates condition numbers LU-IR cannot.

Three honest caveats, all recorded in the output rather than hidden:

- **Everything is an estimate.** `x_ref` is a numerical reference, not the
  exact solution, so `ferr_ref` and everything derived from it inherit its
  error. Its own backward error is reported so the floor is visible.
- **`phi_cond` drops the `min` against `cond(A)`.** `cond(A) = ‖|A⁻¹||A|‖_∞`
  needs the inverse and is not among the condition-est pipeline's outputs.
  Dropping the `min` can only overstate `phi_cond`, so the estimate is
  conservative; the form used is recorded per row as `phi_cond_form`.
- **`phi_solve` is directional.** Under `d̂ = (I + u_s E) d` it measures
  `u_s ‖E_i d_i‖/‖d_i‖`, the error along the direction the correction actually
  took, not the worst case `u_s ‖E_i‖_∞` the Corollary bounds with. It is
  therefore a lower estimate.

**None of this is computed inside the refinement loop.** The loop retains its
iterates, corrections and residual vectors and nothing else; every metric is
reconstructed afterwards by `refinement_metrics`. The loop is timed and its
memory is measured, so a diagnostic solve performed there would corrupt exactly
the figures the report exists to give.

### The precisions, and which solver has which

`--solver` selects the implementation and `--factor-dtype` the precision it
runs at. The two are independent, which is what makes the three variants above
a comparison of precision rather than of implementation. Not every pairing
exists:

| `--factor-dtype` | Available for |
|---|---|
| `complex128` | every solver |
| `complex64` | every solver except `umfpack`, which has no single-precision build |
| `complex32` | `block-thomas` and `block-thomas-inv` only |

`mpir.solver_dtypes` reads this from `cli.SOLVERS`, which stays the one place a
solver's supported precisions are recorded, and `main` rejects an unsupported
pairing at the parser rather than letting the library raise several minutes
into a run.

`complex32` is a **storage label and not a NumPy dtype**. There is no complex
half format, so `BlockThomasFP16` and `BlockThomasExplicitInvFP16` embed each
complex block into a real one of twice the dimension and hold that in
`float16`; the general sparse direct solvers offer no half-precision
factorization at all. `--inv-dtype` applies to `block-thomas-inv` at
`complex32` only: it sets the precision the explicit block inverses are formed
in before being rounded to `float16`, and `float16` is what makes that
factorization half precision throughout. Both implementations, and what each
does and does not carry out in half precision, are documented in
[`../solvers/README.md`](../solvers/README.md).

Two consequences appear directly in the output.

**Vectors entering a `complex32` solver are cast to `complex128`,** not to
something narrower. The half-precision solvers do their own rounding and
power-of-two rescaling on every solve, so a `complex128` argument is cast
losslessly and every rounding then happens inside the solver, where the
embedding can be applied to it. Casting narrower first would insert a second,
unrelated rounding ahead of the half-precision one and charge its effect to the
method. This is `mpir.cast_dtype`, and it is why `u_f` for the `kappa_inf u_f`
bound comes from `mpir.unit_roundoff` rather than from the cast dtype.

**`complex32` does not halve the matrix again relative to `complex64`.** The
embedding writes each of the two real components of an entry twice, so an entry
costs 8 bytes either way and the redundancy gives back exactly what the
narrower format saves. This is `mpir.ITEMSIZE`, stated explicitly because there
is no dtype to infer an itemsize from and because the equality is a result
worth reporting rather than a rounding of one.

**cuDSS start-up cost.** The first cuDSS call in a process pays a fixed cost
for CUDA context creation and kernel compilation, independent of problem size
and measured at roughly 1.2 s. Left in place it would be charged in full to
whichever variant runs first, which is the refinement variant. An untimed
warm-up solve is therefore performed before any variant is measured.

**cuDSS right-hand-side binding.** cuDSS binds the right-hand-side shape at
construction, and `reset_operands` rejects any later array whose shape or
strides differ. `_CuDSSSolver` allocates one Fortran-ordered buffer and copies
each right-hand side into it, so that the rebuild-and-refactorize fallback is
never triggered. Without this the factorization would be recomputed at every
refinement step, which would defeat the method.

**Memory is reported at two levels**, both analytic and both relative to the
`complex128` factorization refinement is meant to replace.

*Factor memory* is the stored factorization, as each solver reports it through
`factor_nbytes`. It is the one figure comparable across every solver here, and
it shows the halving that motivates the method.

*Working set* is the factorization plus what the method must hold alongside it,
and it is the honest version of the claim. Refinement computes its residual at
the working precision, so it keeps `A` at `complex128` however low `u_f` is;
only a bare low-precision solve can hold `A` at `u_f`. GMRES-IR additionally
holds a Krylov basis of `restart + 1` vectors of length `n` at the working
precision — one basis, not one per right-hand side, since SciPy's `gmres` takes
a single column and `solve_gmres_ir` loops over them.

The factor halves; the working set does not. On carbon-nanotube at `n = 768`
the factor ratio is 0.50x while the working set is 0.73x for LU-IR and 0.86x
for GMRES-IR. Quote the working set where the claim is about memory saved.

Process-level figures are deliberately not reported. The Python heap sees the
Block Thomas factors, which are NumPy arrays, but not those of SuperLU, UMFPACK
or MUMPS, which live in compiled extensions, nor those of cuDSS, which live on
the device; a table mixing the two would rank solvers by where their memory
sits rather than by how much they use. Peak RSS is additionally order-dependent,
since the allocator does not return freed memory to the operating system:
whichever variant runs first is charged the whole growth and the rest measure
zero.

---

## 5. `sparse.py` and `dense.py`

The earlier standalone studies, on a single system rather than a sweep, and
against SciPy's sparse solvers and LAPACK directly rather than through the
solver library. Both compare fp32 with fp64 refinement against pure fp64 and
pure fp32.

`dense.py` has no half-precision variant because CPU LAPACK provides no fp16
factorization; that requires cuSOLVER or MAGMA on a device. The half-precision
study is therefore carried out with the Block Thomas implementations in
`../solvers/solver_classes.py`, which simulate `float16` arithmetic in NumPy
and are reached here as `--factor-dtype complex32`.

---

## 6. Output

**One invocation is one experiment, and every experiment is kept.** Each run
appends a new numbered group to `<outdir>/<material>/<material>.h5` —
`--outdir` defaults to `cli.MIXED_PREC_DIR` — rather than overwriting the last,
so the file becomes a record of what was actually run. `--no-save` suppresses
the write; `--list-experiments` prints what a file already holds.

```
<outdir>/<material>/<material>.h5
└── experiments/
    ├── 0001/          attrs: the whole run configuration
    │   ├── runs        one row per (index, variant)
    │   └── iterations  one row per (index, outer step)
    ├── 0002/
    └── ...
```

`mpir.py` writes into a *directory* per material, `<material>/<material>.h5`,
rather than the flat `<material>.h5` of `cli.analysis_h5`:
`plotting/mixed_prec_ir/plot_mpir.py` writes its figures into a subdirectory
of that same directory, one per experiment (`exp0001/`, `exp0002/`, ...), so
the whole material — the data and every figure ever drawn from it — is one
directory, `scp -r`-able as a unit. See Plotting below.

The experiment number is zero-padded because HDF5 orders keys as strings:
unpadded, `10` would sort before `2` and a listing would come out in the wrong
order. The lowest unused number is taken, so an experiment deleted by hand is
reused rather than leaving a gap.

Every other analysis in this pipeline rewrites its one group in place, which is
right for a sweep reproducible from its inputs. Refinement is not: the
interesting runs differ in precision, inner solver and stopping thresholds, and
comparing them **is** the point, so each is kept beside the configuration that
produced it.

### The experiment group's attributes

Everything needed to say what a run was, in one place: `material`, `source`,
`timestamp`, `command`, `solver`, `factor_dtype`, `inv_dtype`, `inner` and
`inner_label`, `working_dtype`, `residual_dtype`, `working_u`, `rho_thresh`,
`max_iter`, `k_max`, `ferr_thresh`, `gmres_tol`, `gmres_restart`,
`gmres_max_iter`, `reference_solver`, `repeats`, `indices`, `n_requested`, `n_skipped`,
`criteria`, `convergence_factor`, plus the material's band edges and energy
grid. They sit on the experiment group and not on the two tables, so a reader
looks in one place.

### The two tables

They are two rather than one because they have different lengths. A figure of
final accuracy against energy reads `runs` alone; a convergence trajectory or a
`phi` panel reads `iterations` alone. Both carry `idx`, so either joins back to
the other. All three variants appear in `runs`; only the refinement variant
performs outer steps, so only it appears in `iterations`.

**`runs`** — what each variant achieved and what it cost:
`idx`, `energy`, `n`, `nnz`, `n_rhs`, `n_blocks`, `solver`, `factor_dtype`,
`inner`, `variant`, `is_refined`, `u_f`, `u`, `u_s`, `kappa_2`, `kappa_inf`,
`lu_ir_bound`, `relres`, `ferr_ref`, `eta1`, `eta2`, `etainf`, `omega`,
`outer_iters`, `converged`, `rho_max`, `phi_final`, `stop_reason`,
`gmres_total`, `wall_s`, `factor_s`, `factor_symbolic_s`, `factor_numeric_s`,
`inner_s`, `factor_mb`, `working_mb`, `reference_solver`, `reference_nbe`.

**`iterations`** — the convergence trajectory and the quantities Corollary 3.3
is stated in: `outer_iteration`, `relres`, `residual_norm_inf`, `ferr_ref`,
`rho`, `etainf`, `omega`, `mu_hat`, `phi_cond_hat`, `phi_solve_hat`,
`phi_hat`, `phi_cond_form`, `z`, `v`, `rho_max`, `phi_demmel`, `ferr_ratio`,
`correction_norm_inf`, `reference_correction_norm_inf`,
`gmres_inner_iterations`, `gmres_inner_max`, `note`, alongside the same
identifying columns.

`z`, `v`, `rho_max`, `phi_demmel` and `ferr_ratio` are the stopping-criteria
quantities themselves, so a figure can show not only that a run stopped but
which condition was about to fire and how close the others were. `ferr_ratio`
is `ferr_i / ferr_{i-1}`, the quantity condition 5 tests; it is numerically the
same ratio as `rho` one step later (`ferr_ratio[i] == rho[i-1]`), but kept as
its own column since it is what the monitor actually saw at decision time,
looking backward rather than `rho`'s forward-looking convention.

Raw iterate, correction and residual **vectors are not stored**: they are
`O(n)` per step per index and a sweep would run to gigabytes, while every
quantity a figure needs is already reduced to a scalar per step here. A study
that needs the vectors themselves should re-run the single index it cares
about.

### Reading it back

`mpir.load_experiment(path, experiment=None)` returns
`(name, attrs, runs, iterations)` for one experiment, the last by default, with
the two tables as the column dicts `factor_io.load_table` returns.
`mpir.experiment_names(path)` lists what is there. The plotting script uses
exactly these.

### Plotting

[`../plotting/mixed_prec_ir/plot_mpir.py`](../plotting/mixed_prec_ir/) draws one
experiment in the layout of Carson and Higham's numerical experiments: a
two-panel figure per energy index — convergence history on the left, the `phi`
decomposition on the right — and one summary figure across the sweep. Figures
are written into `exp<NNNN>/` beside the analysis file by default, so a
material's directory ends up holding the data and every experiment's figures
together:

```
mixed-precision-IR/carbon-nanotube/
├── carbon-nanotube.h5
├── exp0001/
│   ├── carbon-nanotube_summary.png
│   └── carbon-nanotube_E84.png
└── exp0002/
    └── ...
```

```bash
python ../plotting/mixed_prec_ir/plot_mpir.py \
    /scratch/yimili/mixed-precision-IR/carbon-nanotube/carbon-nanotube.h5 --list
python ../plotting/mixed_prec_ir/plot_mpir.py \
    /scratch/yimili/mixed-precision-IR/carbon-nanotube/carbon-nanotube.h5 \
    --experiment 3 --idx 84
```

---

## References

- J. H. Wilkinson, *Rounding Errors in Algebraic Processes*, 1963.
- A. Buttari et al., Mixed precision iterative refinement techniques for the
  solution of dense linear systems, *IJHPCA* 21(4), 2007.
- E. Carson and N. J. Higham, A new analysis of iterative refinement and its
  application to accurate solution of ill-conditioned sparse linear systems,
  *SIAM J. Sci. Comput.* 39(6), 2017.
- E. Carson and N. J. Higham, Accelerating the solution of linear systems by
  iterative refinement in three precisions, *SIAM J. Sci. Comput.* 40(2), 2018.
  Corollary 3.3 is the source of the convergence factor reported here.
- E. Oktay and E. Carson, Multistage mixed precision iterative refinement,
  *Numer. Linear Algebra Appl.* 29(4), 2022; arXiv:2107.06200. Section 2.1.1 is
  the source of the stopping criteria.
- J. Demmel et al., Error bounds from extra-precise iterative refinement,
  *ACM TOMS* 32(2), 2006. The `phi` estimate the convergence test uses.

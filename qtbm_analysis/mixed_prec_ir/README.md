# `mixed_prec_ir/` — mixed-precision iterative refinement

If a reduced-precision factorization is not accurate enough on its own, can it
still be *used*, as the inner solver or the preconditioner of a refinement
scheme? These scripts answer that for the QTBM systems.

All scripts here take the canonical option names; see
[the top-level README, section 3](../README.md#3-command-line-conventions).

| Script | Scope |
|---|---|
| `mpir.py` | does refinement converge, and to what accuracy: LU-IR and GMRES-IR over any solver in `../solvers/`, at any precision that solver supports |
| `mpperf.py` | what refinement costs in runtime: the `complex64` + LU-IR solve against the `complex128` direct solve it replaces, across several solvers at a fixed `complex64` |
| `sparse.py` | the earlier standalone study: fp32 SuperLU with fp64 refinement |
| `dense.py` | the earlier standalone study: fp32 LAPACK with fp64 refinement |

The two studies are separate scripts because they are separate measurements
with incompatible requirements. `mpir.py` varies the precision and the inner
solver over one solver family and asks about accuracy; `mpperf.py` holds the
precision fixed and varies the family, and its numbers are only comparable
when every variant of one index is measured back to back in one process. They
share the refinement loops, the solver registry and the stopping rule:
`mpperf.py` imports them from `mpir.py` rather than reimplementing them, so
both studies measure the same code.

Each writes numbered experiments to its own file in the material's directory;
see [section 6](#6-output) and [section 7](#7-mpperfpy). No figures are
produced here — they are drawn by
[`../plotting/mixed_prec_ir/`](../plotting/mixed_prec_ir/).

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

A different reference solver — `--reference-solver` defaults to `extended`
(complex128 refined with a clongdouble residual, the "true" solution) and
supplies both `x_true` and the reference corrections `phi_solve` is measured
against. The other three named solvers run at plain complex128, with a
forward error of the same order as refinement's own limiting accuracy, so
they check a specific solver's own solution rather than serving as a ruler
for convergence.

```bash
python mpir.py .../si-bulk.h5 --idx 254 --solver cudss \
    --factor-dtype complex64 --reference-solver cudss
```

Raising the safety net and the convergence level, and writing the tables
somewhere else. The stopping rule itself takes no option — see
[Stopping criterion](#stopping-criterion).

```bash
python mpir.py .../si-bulk.h5 --idx 254 --solver cudss \
    --factor-dtype complex64 --inner gmres \
    --max-iter 60 --ferr-tol 1e-12 --outdir ./scratch
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

### Stopping criterion

The loop does **not** stop on a fixed residual tolerance, and no longer uses
the five conditions it once did. There is one rule:

> **stop when the forward error against the reference solution increases.**

An increase means the correction just applied made the answer worse — rounding
noise rather than refinement — so the previous iterate was the best the method
reached, and that is what gets returned. `--max-iter` (30) is a safety net
behind it, and the only thing that ends the loop when no reference solution is
available.

The rule is checked at the *top* of each pass, before the pass spends a solve
on a correction. Every criterion this module used previously looked backwards
— whether step `i` was worth taking could only be judged once step `i+1` had
been formed — so a run always paid for one correction it had already decided
it did not want. Comparing the forward error of the iterate in hand against
the previous one needs nothing from the future, so `outer_iters` counts the
corrections that produced the returned solution, and no more.

**What was deleted, and why.** The five Oktay–Carson conditions (correction no
longer moves the iterate; corrections stopped shrinking geometrically; an
iteration limit; inner GMRES too long; forward error improving by less than a
threshold) and the Demmel et al. `psi` convergence estimate. Each needed a
constant whose right value varied with `u_f` and `kappa_inf`, and every one of
them could cut off a run that was still genuinely converging — which happens
precisely at the ill-conditioned end of a sweep, biasing the iteration count
downward exactly where it is most interesting. Measured on carbon-nanotube
E_1608 at complex32, 29 corrections were still productive; the old
`--rho-thresh 0.5` would have stopped it at 13. An increase in the measured
forward error needs no constant and cannot be argued with.

### Did it converge?

Stopping and converging are separate questions. Stopping says the method got as
far as it was going to; `converged` says whether that was far enough:

```
converged  <=>  best ferr <= ferr_tol,   ferr_tol defaulting to cond(A,x) u
```

`cond(A,x)` is read from the condition-est file's `cond_skeel_x` column —
Corollary 3.3's own limiting accuracy for this system, not `sqrt(n) u`, the
level the working precision can represent in the abstract. The two differ by
orders of magnitude on an ill-conditioned index: judging convergence against
`sqrt(n) u` there mistakes the theorem's own limit for a failure, calling a run
"not converged" for stopping exactly where it was always going to stop.
`sqrt(n) u` remains the fallback where `cond_skeel_x` is unavailable — an older
condition-est file, or none at all — and `--ferr-tol` overrides either.

`u` here is the **working** precision for every variant, never `u_f`: the
question is whether refinement reached the accuracy the working precision can
hold, and asking it at `u_f` would accept a solution only as good as the
factorization refinement exists to improve on. `ferr_best` and `ferr_tol` are
both recorded per run so the verdict can always be re-derived.

### When the low-precision factorization itself fails

At `complex32` the factorization can overflow outright before refinement ever
starts — a Block Thomas block reaching `inf`/`nan` in the Schur recursion with
no power-of-two rescale left to bring it back into `float16` range, or a pivot
underflowing to exactly zero. This is caught and recorded, not left to crash
the sweep: the refined variant gets a run row with `converged = 0`,
`outer_iters = 0`, `ferr_best = nan`, and `stop_reason` naming the exception,
while `kappa_inf` and every other quantity known before any variant ran are
filled in normally — enough for a `kappa_inf`-vs-`outer_iters` figure to place
the index and colour it red. These are typically the hardest indices in a
sweep, so dropping them (silently absent, as an earlier version of this module
did) would make such a figure end early exactly where its trend matters most.
The complex128 baseline is unaffected and still runs.

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
| `phi_cond_hat` | `2 u_s min(cond(A), kappa_inf(A) mu_hat)`, with `cond(A) = ‖|A⁻¹||A|‖_∞` read from the condition-est file's `cond_skeel` column |
| `phi_cond_binding` | which half of that `min` was taken, `cond` or `kappa_mu`; `cond` means `mu_hat` bought nothing this step |
| `phi_solve_hat` | `‖d_i − d_i^ref‖_∞ / ‖d_i^ref‖_∞`, against a reference solve of the *same* retained residual |
| `phi_hat` | their sum, the estimate of the Corollary's `phi_i` |

`u_s` is the effective precision of the correction solve: `u_f` for LU-IR,
whose correction is whatever the low-precision factors return, and `u` for
GMRES-IR, whose inner solve runs at the working precision. That difference is
the whole reason GMRES-IR tolerates condition numbers LU-IR cannot.

Four honest caveats, all recorded in the output rather than hidden:

- **Everything is an estimate.** `x_ref` is a numerical reference, not the
  exact solution, so `ferr_ref` and everything derived from it inherit its
  error. Its own backward error is reported so the floor is visible.
- **`phi_cond` takes the `min` only over what the condition-est file holds.**
  Both halves bound the same quantity, `‖|A⁻¹||A||e_i|‖_∞ / ‖e_i‖_∞`: the left
  one worst-case over the direction of the error, the right one along the
  direction this step's error actually took. `cond(A)` comes from
  `cond_skeel`, which `condition_est.py` gained after `cond_inf`, so a file
  written before then carries only the `kappa_inf(A) mu_hat` half; `mu_hat` is
  in turn undefined once the error reaches zero. Dropping either half can only
  overstate `phi_cond`, so a partial estimate stays conservative, and
  `phi_cond_form` records per row which form was evaluated.
- **`cond(A, x)` is not part of `phi_cond`.** `cond_skeel_x` bounds the
  *limiting accuracy* refinement can reach, roughly `cond(A, x) u`, not the
  rate at which it gets there. It is carried in the `runs` table instead, so a
  figure can draw that floor beside `ferr_ref`.
- **`phi_solve` is directional.** Under `d̂ = (I + u_s E) d` it measures
  `u_s ‖E_i d_i‖/‖d_i‖`, the error along the direction the correction actually
  took, not the worst case `u_s ‖E_i‖_∞` the Corollary bounds with. It is
  therefore a lower estimate.
- **`phi_solve`'s reference correction is a plain complex128 solve, even with
  `--reference-solver extended`.** Reusing the extended reference's own
  multi-step clongdouble refinement here costs `O(n_rhs · outer_iters · nnz)`
  unvectorized extended-precision work — minutes per index on a QTBM block with
  nnz in the millions — for a term that never reaches the summary figure or a
  convergence verdict. `x_true` is unaffected and still built at full extended
  precision, so `ferr_ref`, `outer_iters`, `converged`, `rho`, `mu_hat` and
  `phi_cond` are all exactly as before; only `phi_solve_hat`, `phi_hat` and
  `reference_correction_norm_inf` are computed this way.

  This is sound for **LU-IR**, where `u_s = u_f` and the reference's own
  `cond(A,x) u` stays far below it. For **GMRES-IR** `u_s = u`, the same
  precision this reference carries, so at large `cond(A,x)` the two become
  comparable and `phi_solve_hat` degrades into noise — do not quote a
  high-`kappa_inf` GMRES-IR `phi_solve` from a run made this way. Restoring the
  accurate term for one experiment is a one-line change; see
  `_ExtendedReference.fast_solve`.

### The reference solution, and why it decides the statistics

Every forward error here is measured against `x_true`, so `ferr_ref` cannot go
below the reference's own error. A complex128 direct solver carries a forward
error of order `cond(A,x)·u` — the *same order* as refinement's own limiting
accuracy (3.10) — so it is not a usable ruler for a convergence study:

- **the iteration count** is corrupted, because stopping condition 5 fires when
  `ferr` stops improving, so a coarse reference ends the loop early — and it is
  coarsest exactly where `kappa_inf` is largest, flattening the trend being
  measured;
- **the contraction rate** is corrupted, because the last steps then sit on the
  reference's plateau rather than the method's.

`--reference-solver extended` refines a complex128 solve with the residual
accumulated in `np.clongdouble`, as `block-thomas/forward_error.py` does. The
gain is `eps_double / eps_ext ≈ 2×10³`, **not** `u²`: refinement in a residual
precision `u_r` converges to about `kappa_inf(A)·u_r`. Measured against an
exact rational solution the improvement is 1600–3200× over a plain SuperLU
complex128 solve. `reference_floor = kappa_inf(A)·eps_ext` is recorded per
index — a `ferr_ref` within an order of magnitude of it is measuring the
reference, not the method.

### What the sweep actually reports

The `phi_*` and `mu_hat` columns above are recorded but **not plotted**. The
one quantity a convergence-versus-conditioning study reads is

```
outer_iters   the corrections that produced the returned solution
```

together with `converged`. Those two, against `kappa_inf(A)`, are what decide
whether mixed-precision refinement is worth using on a system: the iteration
count multiplies the cost of the cheap low-precision factorization, so a method
needing thirty steps has given back what the low precision won. They are the
whole content of the summary figure — see
`plotting/mixed_prec_ir/plot_mpir.py`.

The convergence factor is a *mechanism*, not a decision variable. It stays in
the `iterations` table for the per-index figures and for the thesis text, and
nothing is derived from it automatically. An earlier version of this module
also reduced it to a per-run mean (`n_contract`, `rho_bar`, `rho_censored`);
those columns are gone, along with the two different rules tried for splitting
the geometric phase from the plateau, neither of which survived contact with
the complex32 sweeps.

`outer_iters` is only as good as the reference solution it is measured against,
which is why `--reference-solver extended` exists: a complex128 reference
carries a forward error of the same order as refinement's own limiting
accuracy, so the stopping rule would fire when the *reference* ran out rather
than when the method did — earliest exactly where `kappa_inf` is largest,
flattening the very trend being measured.

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
    │   └── iterations  one row per (index, outer step), plus one
    │                  terminal row for the returned solution
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
`inner_label`, `working_dtype`, `residual_dtype`, `working_u`,
`max_iter`, `ferr_tol`, `gmres_tol`, `gmres_restart`,
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
`cond_skeel`, `cond_skeel_x`, `lu_ir_bound`, `relres`, `ferr_ref`, `eta1`, `eta2`, `etainf`, `omega`,
`outer_iters`, `converged`, `ferr_best`, `ferr_tol`, `stop_reason`,
`gmres_total`, `gmres_avg`, `wall_s`, `factor_s`, `factor_symbolic_s`, `factor_numeric_s`,
`inner_s`, `solve_s`, `residual_s`, `other_s`, `n_solves`, `factor_mb`,
`factor_mb_reported`, `working_mb`, `reference_solver`, `reference_nbe`,
`reference_floor`.

`factor_symbolic_s` and `factor_numeric_s` split the factorization into the
analysis phase — the fill-reducing ordering and the symbolic factorization,
or for the Block Thomas families the detection and extraction of the blocks —
and the numerical phase. The analysis phase performs no floating-point
arithmetic, so it costs the same at every `u_f` and bounds the speedup a lower
precision can produce. cuDSS and MUMPS time their own phases and the Block
Thomas builders are timed in two steps; SuperLU (`scipy`'s `splu`) and UMFPACK
fuse both into one call and report `nan`.

`solve_s`, `residual_s` and `other_s` split `inner_s` the same way: the
low-precision solves, `n_solves` of them, the working-precision residuals
`b - Ax`, and what is left — the update, the stopping monitor and, for
GMRES-IR, the products with `A` and the orthogonalization. Both sub-timers run
strictly inside the `inner_s` window, so the three sum to it and `other_s` is
non-negative by construction. LU-IR performs one solve per outer step and
GMRES-IR one per inner GMRES iteration, which is most of the difference in
their cost.

`factor_mb_reported` is 0 where the backend exposed no factor size —
MUMPS when `INFOG(3)` is unreachable through the installed `python-mumps`,
cuDSS when the factorization info carries no `lu_nnz`. Zero bytes of factors
is not a possible measurement, so `factor_mb` is left at 0 rather than
guessed and `working_mb` is then a lower bound. A figure must check the flag
rather than draw the zero.

**`iterations`** — the convergence trajectory and the quantities Corollary 3.3
is stated in: `outer_iteration`, `relres`, `residual_norm_inf`, `ferr_ref`,
`rho`, `etainf`, `omega`, `mu_hat`, `phi_cond_hat`, `phi_solve_hat`,
`phi_hat`, `phi_cond_binding`, `phi_cond_form`, `ferr_ratio`,
`correction_norm_inf`, `reference_correction_norm_inf`,
`gmres_inner_iterations`, `gmres_inner_max`, `note`, alongside the same
identifying columns.

`ferr_ratio` is `ferr_i / ferr_{i-1}`, the stopping rule's own quantity: the
step where it first exceeds 1 is the step the loop stopped on. It is
numerically the same ratio as `rho` one step later
(`ferr_ratio[i] == rho[i-1]`), but kept as its own column since it is what the
monitor actually saw at decision time, looking backward rather than `rho`'s
forward-looking convention.

This table has **one row more than `outer_iters`** for a run that stopped on an
increase: the iterate whose forward error rose is recorded too, since it is the
evidence the rule acted on. It carries `ferr_ref`/`relres`/`etainf`/`omega` but
no `correction_norm_inf`, `phi_solve_hat` or `gmres_inner_iterations`, because
no correction was computed from it.

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

## 7. `mpperf.py`

The runtime study. `mpir.py` asks whether refinement converges and to what
accuracy; `mpperf.py` asks what it costs, on the assumption that it does. It
sweeps **solvers** rather than precisions.

```bash
python mpperf.py <material.h5> --idx 84 254 601
python mpperf.py <material.h5> --start 0 --end 800 --stride 200
python mpperf.py <material.h5> --idx 84 --solvers block-thomas mumps --repeats 5
python mpperf.py <material.h5> --list-experiments
```

A handful of indices, not a sweep: each index is eight bars in the figure.

### The variants

`--inner` selects the refinement, **spelled and valued exactly as in
`mpir.py`**, and the variants follow from it:

| `--inner` | variants measured | bars per solver |
|---|---|---|
| `direct` (default) | `c64_ir`, `c128` | 2 |
| `gmres` | `c64_gmres`, `c128` | 2 |
| `both` | `c64_ir`, `c64_gmres`, `c128` | 3 |

`mpir.py` runs one inner solver at a time and has no need for more; `mpperf`
compares them, so it adds `both`. `--gmres-tol`, `--gmres-restart` and
`--gmres-max-iter` are also the same names and the same defaults as
`mpir.py`'s, so a cost run and a convergence run of the same configuration
differ only in which script is called.

`c128` is measured in every case: it is the baseline each variant is read
against, and a run without it would report no speedup. The variant keys
(`c64_ir`, `c64_gmres`, `c128`) are the on-disk names and never change; the
figure reads the list from the experiment, so a two-bar and a three-bar run
draw correctly from the same script.

A timing is only meaningful for a variant that reached the target accuracy.
`c64_ir` stops on `mpir`'s rule — the forward error against the reference
increased — and `converged` is that monitor's verdict; `c128` runs no loop, is
the accuracy the study is measured against, and is recorded as converged with
its `ferr_ref` beside it so a figure can check rather than assume. A `c64_ir`
bar that did not converge is a cost that bought nothing, and the figure hatches
it.

### The three stages

Every bar is `symbolic_s + factorization_s + solve_s = total_s`.

| Column | What it contains |
|---|---|
| `symbolic_s` | the fill-reducing ordering and symbolic factorization, or for Block Thomas the detection and extraction of the blocks. No floating-point arithmetic, so it costs the same at both precisions and bounds the achievable speedup |
| `factorization_s` | the rest of the factorization: the numerical phase, plus the cast of `A` into the factorization precision and, for cuDSS, the host-to-device transfer. The stage a lower `u_f` makes cheaper |
| `solve_s` | the whole refinement iteration: every triangular solve, every `complex128` residual `b - Ax`, and for GMRES-IR the Krylov work too — the products with `A`, the orthogonalization and the least squares problem |

`solve_s` is split into `triangular_s`, `residual_s` and `krylov_s`, recorded
but never drawn: the bar totals the end-to-end cost of the variant, which is what makes
the two bars of a pair comparable, and the split explains that total rather
than restating it.

The split is worth querying, because the residual is `b - Ax` and so costs the
**same for every solver** at one index. Measured on si-bulk it is 78-95% of
`solve_s`, about 1.1 ms per right-hand-side column per outer step for all three
solvers alike -- cuDSS spends 9.5 ms of a 180 ms solve stage on the GPU and the
rest on a host `complex128` matvec. Two things follow. The quantity `u_f`
cannot reduce is the residual, not the symbolic phase as in Zounon et al.; and
the `c64_ir` bars of different solvers are mostly the same shared cost, so the
solver-dependent information is in the symbolic and factorization segments, not
in the height of the bar.

`factorization_s` is `factor_s` minus the symbolic phase, **not** the numerical
phase the backend reports: the builder also casts `A` and, for cuDSS, moves it
to the device, and that work is precision-dependent and belongs on the same
side of the split as the arithmetic. Attributing it this way is also what makes
the three stages sum to the true cost of the variant rather than to slightly
less than it. `numeric_reported_s` and `factor_s` are kept as columns so the
difference is auditable.

SuperLU (`scipy.splu`) and UMFPACK fuse the symbolic and numerical phases into
one call and expose no split. `symbolic_s` is then `nan`, `phases_split` is 0,
and the whole factorization sits in `factorization_s`; those bars have two
segments and are not comparable stage by stage with the other solvers'.

### What is not timed

Nothing that exists only to diagnose convergence is inside a timed region.
Each stage above is bracketed by its own `perf_counter` pair around the solver
call, the matrix-vector product or the builder, so the forward errors the
stopping rule watches, the backward errors `eta` and `omega`, the convergence
factor `mu_hat` and the `phi_*` terms of Corollary 3.3 all fall *between* the
brackets and are charged to nothing. `mpir.refinement_metrics`, which computes
`rho` and the `phi_*` columns, is never called by this script at all.

The one thing left out is the `O(n)` vector update `x + d`, which no timer
brackets. It is the same work in both variants and orders of magnitude below a
solve.

The refinement loop is nonetheless run **with** a reference solution, through
`--reference-solver`, so that it stops on the same rule and after the same
number of outer steps as it would in `mpir.py`. Timing a loop that ran to
`--max-iter` because it had nothing to stop on would measure the safety net.

### Precision

`u_f` is `complex64` and is not swept. The `complex32` factorizations are
hand-written NumPy kernels that embed each complex block into a real one of
twice the dimension; their run time reflects that implementation and not the
cost of half-precision arithmetic, so timing them against LAPACK, MUMPS and
cuDSS would compare implementations rather than precisions. `--factor-dtype
complex32` is rejected by the parser. Its *accuracy* behaviour is the subject
of `mpir.py`, where the comparison is meaningful.

UMFPACK is likewise absent from the default `--solvers`: it has no
single-precision build, so it cannot supply the `c64_ir` half of any pair.

### Repeats

`--repeats` runs of each variant are made, default 5, and the **median of each
stage** is reported. `total_s` is the sum of those three medians, so a bar
equals the sum of its parts — it is therefore the median of each part rather
than the median total. `total_s_min` and `total_s_max` carry the spread of the
per-repeat totals that the median hides.

### Memory

Three components summing to `working_mb`, drawn as the lower panel of the
summary figure with the same geometry as the upper one:

| Column | What it is |
|---|---|
| `matrix_mb` | `A` at `complex128`, which **both** variants hold — LU-IR forms its residual at the working precision. Identical in the two bars of a pair; it does not shrink with `u_f` |
| `factor_mb` | the stored factorization, from each solver's `factor_nbytes`. Halves exactly at `complex64` |
| `krylov_mb` | the inner GMRES basis, `(restart+1)` vectors of length `n`. Zero until a GMRES-IR variant is measured |

Most of this is analytic and needed no measurement. The matrix term is exactly
`nnz·s + nnz·4 + (n+1)·4`; Block Thomas's factorization is exactly
`s(Σ bᵢ² + 2Σ bᵢbᵢ₊₁) + 4Σ bᵢ` from the block partition, since a
block-tridiagonal LU has no fill beyond the band. Both were verified to the
last digit on si-bulk (24.91 and 43.76 MiB). What is *not* predictable is the
factorization size of a general sparse solver — that is fill-in, and depends on
the ordering and on numerical pivoting, which is why `lu_nnz` and `INFOG(3)`
exist at all.

The working set therefore falls by less than half. With `f = factor/matrix`,

    W(u) / W(u_f) = (F + M) / (F/2 + M) = 2(f+1) / (f+2)

which reproduces both measured ratios to four significant figures (Block
Thomas `f=1.757` → 1.4676 against 1.4675 measured; cuDSS `f=2.067` → 1.5082
against 1.5081). The limits are the whole argument: `f → ∞` gives 2, the
factorization dominating and memory halving with it; `f → 0` gives 1, the
working-precision matrix dominating and a halved factorization buying nothing.
si-bulk sits at `f ≈ 2`, hence ~1.5. `factor_nbytes` returns 0 where a backend exposes no size — MUMPS
where `INFOG(3)` is unreachable, cuDSS where the factorization info carries no
`lu_nnz` — and `factor_mb_reported` is 0 there. A figure must drop such a row
rather than draw it at zero.

Process-level figures — peak Python heap, peak RSS — are deliberately absent,
for the reason given in `mpir.py`: the Block Thomas factors are NumPy arrays
and visible to `tracemalloc` while SuperLU, MUMPS and cuDSS hold theirs in
compiled extensions or on the device, and peak RSS is order-dependent because
the allocator does not return freed pages. Only the per-solver figure is
comparable.

### Fairness

Every variant of one index is measured in one process, back to back, against
the same `A` and `b`, after an untimed warm-up of that solver **at both
precisions**. The reference solution is computed once per index and shared by
every solver, so its cost is charged to none of them.

The warm-up is not only for the GPU. The first call into a numerical library in
a process pays costs no later call pays: OpenBLAS builds its thread pool on the
first LAPACK call, SciPy resolves its lazy imports, cuDSS creates a CUDA context
and JITs its kernels, and the allocator faults in pages it afterwards reuses.
Left inside a measurement it all lands on whichever variant runs first — the
first solver of the first index — whose bar is then not comparable with any
other. Measured on a small block-tridiagonal system it inflated
`factorization_s` from 0.4 ms to as much as 23 ms, and it took two calls to
settle, which is why both precisions are warmed. It survives the median: an
outlier is absorbed only when it is confined to one repeat, and thread-pool and
allocator warm-up are not.

The warm-up runs per `(index, solver)`, not once per process, so a later index
with a different size or pattern is equally warm. It costs two solves against
`2 * --repeats` measured ones.

The min–max spread of the per-repeat totals is printed beside each median while
the sweep runs, and stored as `total_s_min` / `total_s_max`. A spread of more
than about 2x on a warmed solver means something is still not warm.

Solvers are compared on wall-clock cost **as installed and configured on the
machine the script runs on**. MUMPS and cuDSS are libraries with their own
threading and, for cuDSS, their own device; Block Thomas is NumPy code calling
LAPACK on dense blocks. A difference between two solvers here is a difference
between those implementations on this machine, not a property of the algorithms
in isolation. Running all four together requires a node with a GPU, and the
comparison is then between one GPU solver and three CPU solvers.

### Reproducibility

A wall-clock comparison is a statement about a machine, so `mpperf.py` records
the machine on every experiment: host, platform, `cpu_count` and
`cpu_affinity` (they differ under `taskset`), the BLAS name, version and build
configuration, every `*_NUM_THREADS` variable, what `threadpoolctl` reports the
loaded pools actually chose, the NumPy/SciPy/Python versions, and the 1/5/15
minute load averages both **before and after** the sweep. `plot_mpperf.py`
prints the important part of it along the bottom of the figure.

**Cap the BLAS thread pool.** This is not a second-order effect. OpenBLAS
defaults to one thread per core; on a 72-core node factoring dense blocks 128
to 352 wide, that was measured costing **68×** — a `complex128` Block Thomas
factorization of one si-bulk index took 254 ms with the pool capped and 17 s
without. Blocks of a few hundred rows are far too small to want 72 threads, and
the barrier cost dwarfs the arithmetic.

Which variable depends on how OpenBLAS was built. `numpy.show_config()` reports
it; with `USE_OPENMP=0` it is a pthread build and `OPENBLAS_NUM_THREADS` is
authoritative, `OMP_NUM_THREADS` only a fallback. `mpperf.py` **reads** these
and never sets them: OpenBLAS fixes its pool size at the first call, so setting
one from inside the process is unreliable, and silently overriding what was
asked for would make the recorded environment a lie.

**Do not measure on a busy node.** Other jobs make every CPU timing a
measurement of the node. The GPU solver is unaffected, which is what makes the
contamination easy to spot: cuDSS held 1.4% across five si-bulk indices while
the two CPU solvers varied by 14× and 68× on *identically sized* matrices.

Two checks run automatically:

- **Before the sweep**, the environment is printed and warned about — no thread
  cap with more than 16 cores available, or a load average above a quarter of
  them. Nothing is refused; the requirement is that the run says so.
- **After the sweep**, `check_stability` flags every row whose slowest repeat
  exceeds its fastest by more than `--stability-limit` (default 2). Contention
  only ever adds time, so a large spread means the run was interrupted, not
  that the solver is variable. The count is stored as `n_unstable` and those
  bars are outlined in red in the figure.

The median cannot catch this on its own — hiding the spread is what a median is
for. `total_s_min` and `total_s_max` are therefore stored per row and drawn as
a whisker on every bar.

**`--reduce min` on a machine you do not own.** Contention, page reclaim and
scheduling only ever *add* time, so the fastest of the repeats is the
observation least contaminated by the node and the closest to the cost of the
solver itself; `timeit.Timer.repeat` documents the same argument. It is the
right choice where one solver is being disturbed and the others are not — on
si-bulk, MUMPS's `complex128` factorization was measured at 142–157 ms
minimum at every index across three separate experiments while its medians
wandered between 154 and 607 ms, the disturbance falling on the one solver
that allocates the largest contiguous workspace.

`min` is not a way to make a bad run look good. The stability check reads
`total_s_min` and `total_s_max` whichever reducer is used, so a contaminated
run is still reported and its bars still outlined; `min` changes which number
is drawn, not whether the run is admitted. The reducer is recorded in the
experiment attributes and named in the figure title.

`median` remains the default.

### Output

```
<outdir>/<material>/<material>_perf.h5
└── experiments/
    ├── 0001/          attrs: the whole run configuration
    │   └── runs       one row per (index, solver, variant)
    └── ...
```

beside the convergence file `mpir.py` writes, with the same numbering, the same
append-only rule and the same attribute convention (`mpir.new_experiment`).

`mpperf.load_experiment(path, experiment=None)` returns `(name, attrs, runs)`;
`mpperf.perf_path(outdir, material)` gives the file path.

### The Krylov stage

GMRES-IR's Arnoldi work lands in mpir's `inner_s` but in neither `solve_s` nor
`residual_s`, and `inner_s` also contains the forward-error diagnostics the
stopping rule computes. Timing it as `inner_s - solve_s - residual_s` would
therefore charge GMRES-IR for the diagnostics this study exists to exclude.

`mpir.solve_gmres_ir` instead measures it directly, as the wall time of each
inner GMRES call **minus** the preconditioner time inside that call, which
`precond_apply` has already counted into `solve_s`. The result is neither
double counted against the triangular solves nor contaminated. Without it a
GMRES-IR bar would show only its preconditioner applications and badly
understate the method.

`gmres_total` records the inner iterations summed over every outer step and
every right-hand-side column, since that is the count multiplying the
preconditioner applications. It is `-1`, not `0`, for a variant that runs no
inner solver: a real zero would mean GMRES converged without iterating.

### Plotting

[`../plotting/mixed_prec_ir/plot_mpperf.py`](../plotting/mixed_prec_ir/) draws
one figure, `<material>_perf_summary.png`, into `perf<NNNN>/` beside the
performance file, matching the `exp<NNNN>/` that `plot_mpir.py` writes.

**`--nrhs` draws the second figure**, `<material>_perf_nrhs.png`: speedup over
the `complex128` direct solve against the number of right-hand sides, pooled
over every experiment in the file. This is the axis the cost data actually
varies along. `kappa_inf` is the right axis for the *convergence* study, where
iteration count genuinely tracks conditioning, but cost does not track it — on
si-bulk the speedups were flat across two and a half orders of magnitude of
`kappa_inf` and monotone in `n_rhs`. The reason is structural: refinement saves
half a factorization, which is fixed, and costs `(k+1)` solves and residuals,
every one of which scales with `n_rhs`. Colour is the solver and line style the
variant; each index is its own point and the line joins the median at each
`n_rhs`.

Pooled experiments are checked for comparability first — machine, BLAS, thread
caps, reducer, precision and reference solver — and any disagreement is
reported in full, since a curve assembled from differently configured runs is
not a curve. `--experiments` restricts the pool.

The summary figure itself: one group of bars per energy index, the groups
ordered by `kappa_inf(A)` and
evenly spaced regardless of it — bars of neighbouring indices would otherwise
overlap wherever two condition numbers are close, which near a band edge is
most of them. Within a group each solver contributes a pair of bars in its own
`style.SOLVER_STYLE` colour, left `c64_ir` and right `c128`, each stacked into
the three stages and shaded light to dark. Reading one pair is the whole point:
the left bar shorter than the right is the case for mixed precision at that
conditioning, and the left bar growing past the right as `kappa_inf` rises is
refinement giving back what the low precision won.

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
  *ACM TOMS* 32(2), 2006. Source of the `psi` estimate this module used
  as its convergence test until the stopping rule was reduced to a single
  forward-error comparison; kept as a pointer for the thesis text.
- M. Zounon, N. J. Higham, C. Lucas and F. Tisseur, Performance impact of
  precision reduction in sparse linear systems solvers, *PeerJ Comput. Sci.*
  8:e778, 2022. The stage breakdown `mpperf.py` draws, and why the symbolic
  phase is what keeps a sparse speedup short of 2.
- P. Amestoy, A. Buttari, N. J. Higham, J.-Y. L'Excellent, T. Mary and
  B. Vieublé, Combining sparse approximate factorizations with mixed-precision
  iterative refinement, *ACM TOMS* 49(1), 2023. The variant layout and the
  time and memory breakdowns `mpperf.py` follows.

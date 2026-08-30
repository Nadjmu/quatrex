# `mpir.py` — a practical guide

This is a self-contained reference for `mpir.py`: what it does, how to run it,
what ends up in the HDF5 file, and what the plotting script draws from it. The
module docstring (`python mpir.py --help`) carries the same material inline
with the code; this document exists to be read start to finish once, rather
than grepped.

`mpir.py` answers whether refinement converges and to what accuracy. Its
companion `mpcost.py` answers what it costs — time and memory, across solvers,
at a fixed `complex64` — reusing these same refinement loops and writing its
own file. See [README.md, section 7](README.md#7-mpcostpy).

---

## 1. What it measures

Iterative refinement solves `A x = b` by factorizing `A` at a low precision,
then repeatedly correcting the solution using residuals computed at a higher
one:

```
1. Factorize A at u_f (the low precision under test).
2. x = solve(b) at u_f, promoted to u (complex128).
3. r = b - A x, computed at complex128.
4. Solve A dx = r for the correction (see below).
5. x = x + dx. Repeat from 3 until a stopping criterion fires.
```

Step 4 is where the two variants differ:

| Variant | `--inner` | Step 4 | Effective solve precision `u_s` |
|---|---|---|---|
| **LU-IR** | `direct` | one triangular substitution through the low-precision factors | `u_f` |
| **GMRES-IR** | `gmres` | GMRES on `A` at complex128, preconditioned by the low-precision factors | `u` (complex128) |

LU-IR is cheaper per step but bound by the factorization precision: it
converges only while `kappa_inf(A) * u_f < 1`. GMRES-IR relaxes that bound
because the correction is solved at the working precision — the low-precision
factors only ever act as a preconditioner, never as the thing whose accuracy
limits the answer. This is why GMRES-IR is the variant that matters once
`kappa_inf(A) * u_f` exceeds 1, which happens routinely for QTBM matrices near
a band edge at `complex32`, and even at `complex64` for the worst-conditioned
indices.

**Every run measures three variants of the same solver family**, never just
one, so the comparison isolates precision and refinement from the solver
implementation:

1. `<solver>` at `u_f` **with** refinement — the method under test
2. `<solver>` at `complex128`, no refinement — the accuracy ceiling
3. `<solver>` at `u_f`, no refinement — the lower bound refinement must beat

### The three precisions

| Precision | Available for | Note |
|---|---|---|
| `complex128` | every solver | the accuracy ceiling |
| `complex64` | every solver except `umfpack` | UMFPACK has no single-precision build |
| `complex32` | `block-thomas`, `block-thomas-inv` only | not a NumPy dtype — see below |

`complex32` is a **storage label**, not a real dtype: there is no complex half
format, so `BlockThomasFP16` / `BlockThomasExplicitInvFP16` embed each complex
block into a real one of twice the dimension and hold that in `float16`. Two
consequences that show up directly in the output:

- Vectors entering a `complex32` solver are cast to `complex128` on the way
  in, not to something narrower — the cast is then lossless and all rounding
  happens inside the solver, where the embedding is applied. This is why
  `u_f` for the `kappa_inf * u_f` bound is read off `float16`'s roundoff, not
  off the cast dtype.
- `complex32` does **not** halve memory again relative to `complex64`. The
  real embedding stores each of the two real components of an entry twice, so
  an entry costs 8 bytes either way — the same as `complex64`.

`--solver` picks the implementation, `--factor-dtype` picks the precision
independently; not every combination exists (`solver_dtypes()` checks this
before anything is factored, and the parser rejects an unsupported pairing
immediately rather than raising minutes into a run).

---

## 2. Running it

```bash
python mpir.py <material.h5> --idx <indices> --solver <name> \
    --factor-dtype <precision> [--inner gmres] [options...]
```

### Selecting indices

`--idx 5 84 254` (explicit list), `--start 0 --end 1200` (inclusive range,
optionally `--stride N`), or `--energy 6.1 6.2` (nearest recorded energy —
mutually exclusive with `--idx`/`--start`).

### Examples

```bash
# Basic: one solver, one precision, LU-IR (the default inner solve)
python mpir.py carbon-nanotube.h5 --idx 5 --solver superlu --factor-dtype complex64

# GPU
python mpir.py si-bulk.h5 --idx 254 --solver cudss --factor-dtype complex64

# GMRES-IR, once LU-IR's kappa_inf * u_f < 1 bound is exceeded
python mpir.py si-bulk.h5 --idx 254 --solver mumps --factor-dtype complex64 \
    --inner gmres --gmres-tol 1e-8 --gmres-restart 30 --gmres-max-iter 50

# Half precision (complex32) — CPU only, Block Thomas family.
# LU-IR is too weak at these condition numbers; GMRES-IR is the point.
python mpir.py carbon-nanotube.h5 --idx 84 --solver block-thomas \
    --factor-dtype complex32 --inner gmres

# The explicit-inverse implementation, inverse formed at float16
# (factorization becomes half precision throughout, at some accuracy cost)
python mpir.py carbon-nanotube.h5 --idx 84 --solver block-thomas-inv \
    --factor-dtype complex32 --inner gmres --inv-dtype float16

# A sweep — one experiment, many indices
python mpir.py carbon-nanotube.h5 --stride 10 --solver block-thomas \
    --factor-dtype complex32 --inner gmres

# Raise the safety net and the convergence level, pick where the file goes
python mpir.py si-bulk.h5 --idx 254 --solver cudss --factor-dtype complex64 \
    --inner gmres --max-iter 60 --ferr-tol 1e-12 --outdir ./scratch

# List what's already in a file, without running anything
python mpir.py carbon-nanotube.h5 --list-experiments
```

### Every option

| Option | Default | Meaning |
|---|---|---|
| `--idx` / `--start` `--end` `--stride` / `--energy` | — | which energy indices to run (one experiment can sweep many) |
| `--solver` | — | `superlu`, `umfpack`, `mumps`, `block-thomas`, `block-thomas-inv`, `cudss` |
| `--factor-dtype` | `complex64` | `u_f`: `complex128`, `complex64`, or `complex32` (Block Thomas only) |
| `--inv-dtype` | `float32` | `block-thomas-inv` + `complex32` only: precision the explicit inverse is formed in before rounding to `float16` |
| `--inner` | `direct` | `direct` = LU-IR, `gmres` = GMRES-IR |
| `--max-iter` | `30` | safety net on the outer steps; the stopping rule itself takes no option (§3) |
| `--ferr-tol` | `cond(A,x) u` | accuracy the returned solution must reach to count as converged; falls back to `sqrt(n) u` where `cond(A,x)` is unavailable (§3) |
| `--repeats` | `1` | repeats per variant; the median is reported |
| `--reference-solver` | `extended` | supplies `x_true` and the reference corrections `phi_solve` is measured against |
| `--gmres-tol`, `--gmres-restart`, `--gmres-max-iter` | `1e-8`, `30`, `50` | inner GMRES parameters (`--inner gmres` only) |
| `--outdir` | `cli.MIXED_PREC_DIR` | the analysis file is written to `<outdir>/<material>/<material>.h5` |
| `--material` | derived from the input path | label used in the file name and figure titles |
| `--no-save` | off | print the report, append no experiment |
| `--list-experiments` | off | list the file's experiments and exit |

---

## 3. Stopping criteria

There is **no fixed residual tolerance**. On these systems the residual
reaches the working precision long before the forward error does, so a
residual tolerance would declare convergence while the solution is still
wrong — and it cannot tell slow convergence from divergence at all.

Instead there is one rule:

> **stop when the forward error against the reference solution increases.**

An increase means the correction just applied made the answer worse — rounding
noise, not refinement. The previous iterate was the best the method reached,
and that is what gets returned. `--max-iter` (30) sits behind it as a safety
net, and is the only thing that ends the loop when no reference solution is
available.

The rule is checked at the **top** of each pass, before the pass spends a solve
on a correction, so `outer_iters` counts the corrections that produced the
returned solution and no more.

**What was deleted.** The five Oktay–Carson conditions (correction too small;
corrections stopped shrinking; iteration limit; inner GMRES too long; forward
error improving by less than a threshold) and Demmel's `psi` convergence
estimate. Each needed a constant whose right value varied with `u_f` and
`kappa_inf`, and every one of them could cut off a run still genuinely
converging — which happens precisely at the ill-conditioned end of a sweep,
biasing the iteration count downward exactly where it is most interesting.
Measured on carbon-nanotube E_1608 at complex32, 29 corrections were still
productive; `--rho-thresh 0.5` would have stopped it at 13.

### Did it converge?

A separate question from stopping. Stopping says the method got as far as it
was going to; `converged` says whether that was far enough:

```
converged  <=>  best ferr <= ferr_tol,   ferr_tol defaulting to cond(A,x) * u
```

`cond(A,x)` comes from the condition-est file's `cond_skeel_x` column —
Corollary 3.3's own limiting accuracy, not `sqrt(n) u`, the level the working
precision can represent in the abstract. On an ill-conditioned index the two
differ by orders of magnitude; judging convergence against `sqrt(n) u` there
mistakes the theorem's own limit for a failure. `sqrt(n) u` is still the
fallback where `cond_skeel_x` is unavailable (an older condition-est file, or
none at all), and `--ferr-tol` overrides either.

`u` is always the **working precision** (complex128), never `u_f` — the
question is whether refinement reached the accuracy the working precision can
hold, and asking it at `u_f` would accept a solution only as good as the
factorization itself. `ferr_best` / `ferr_tol` are both recorded per run so
the verdict can be re-derived.

**When the factorization itself fails.** At `complex32` a Block Thomas block
can overflow to `inf`/`nan` before refinement ever starts, or a pivot can
underflow to exactly zero. This is caught, not left to crash the sweep: the
refined variant gets a row with `converged = 0`, `outer_iters = 0`,
`ferr_best = nan`, `stop_reason` naming the exception, and `kappa_inf` filled
in as usual — a red point on the summary figure rather than a gap. These tend
to be the hardest indices in a sweep, so a silent drop would bias exactly the
end of the `kappa_inf` axis that matters. The complex128 baseline still runs.

---

## 3b. Measuring convergence speed

One number per index, in the `runs` table:

| Column | Meaning |
|---|---|
| `outer_iters` | the corrections that produced the returned solution |
| `converged` | whether that solution reached `ferr_tol` (§3) |
| `kappa_inf` | the x axis of the summary figure |

**Plot `outer_iters` against `kappa_inf`.** That is the whole statistic. The
iteration count multiplies the cost of the cheap low-precision factorization,
so a method needing thirty steps has given back what the low precision won —
it is the only quantity here that decides whether to use mixed-precision
refinement on a system.

The convergence factor is deliberately **not** part of this. It is a
mechanism, not a decision variable; `mu_hat` and the `phi_*` columns stay in
the `iterations` table for the per-index figures and the thesis text, and
nothing is derived from them automatically.

**For a sweep, one flag matters, and it's already the default:**

```
--reference-solver extended     an accurate enough ruler; see below (default)
--max-iter 40                   a safety net that should not bind
```

**Why the reference has to be `extended`.** `ferr_ref` cannot fall below
`x_true`'s own error. A complex128 direct solve carries `cond(A,x)·u`, the same
order as refinement's limiting accuracy, so the stopping rule would fire when
the *reference* ran out rather than when the method did — earliest where
`kappa_inf` is largest, flattening the very trend being measured.
`--reference-solver extended` refines with an `np.clongdouble` residual;
measured against an exact rational solution it is 1600–3200× more accurate on
an x86 build, and far more where `np.longdouble` is IEEE binary128.
`reference_floor = kappa_inf·eps_ext` is recorded so a `ferr_ref` near it can
be recognised as measuring the reference.

Runs that stopped on `max_iter` did not finish — filter on `stop_reason` and
report their count separately rather than averaging them in.

---

## 4. Convergence metrics

Beyond the stopping decision, every experiment reconstructs the quantities of
Carson and Higham's Corollary 3.3 — the right-hand panels of their numerical
experiments — **after** the run, from the iterates/corrections/residuals the
loop retained. Nothing here is computed inside the timed loop: it would
corrupt the very timing and memory figures the report exists to give.

```
phi_i = 2 u_s min(cond(A), kappa_inf(A) mu_i)   +   u_s ||E_i||_inf
        \__________________ phi_cond __________/   \___ phi_solve ___/
```

| Quantity | Meaning |
|---|---|
| `ferr_ref` | `‖x_ref − x_i‖_∞ / ‖x_ref‖_∞` — forward error **relative to the reference solution**, not a certified forward error |
| `rho` | `ferr_ref[i+1] / ferr_ref[i]`, the contraction actually observed |
| `mu_hat` | `‖A(x_ref − x_i)‖_∞ / (‖A‖_∞ ‖x_ref − x_i‖_∞)` — small exactly when the error points along directions `A` damps, which is what lets refinement work past the plain bound |
| `phi_cond_hat` | `2 u_s min(cond(A), kappa_inf(A) mu_hat)`, with `cond(A) = ‖|A⁻¹||A|‖_∞` read from the condition-est file's `cond_skeel` column |
| `phi_cond_binding` | which half of that `min` was taken, `cond` or `kappa_mu`; `cond` means `mu_hat` bought nothing this step |
| `phi_solve_hat` | `‖d_i − d_i^ref‖_∞ / ‖d_i^ref‖_∞`, against a reference solve of the *same* retained residual |
| `phi_hat` | `phi_cond_hat + phi_solve_hat`, the estimate of Corollary 3.3's `phi_i` |

`u_s` is `u_f` for LU-IR and `u` for GMRES-IR — the whole reason GMRES-IR
tolerates condition numbers LU-IR cannot.

Four honest caveats, all visible in the output rather than hidden:

- **Everything here is an estimate.** `x_ref` is a numerical reference
  solution, not the exact one, so `ferr_ref` and everything derived from it
  inherit its error. Its own backward error is printed and saved
  (`reference_nbe`) so the floor is visible.
- **`phi_cond` takes the `min` only over what the condition-est file holds.**
  Both halves are bounds on the same quantity, so the smaller is the one the
  Corollary asserts. `cond(A)` comes from `cond_skeel`, added to
  `condition_est.py` later than `cond_inf`; a file written before then has only
  the `kappa_inf(A) mu_hat` half, and `mu_hat` itself is undefined once the
  error reaches zero. Dropping either half can only overstate `phi_cond`, so a
  partial estimate stays conservative, and `phi_cond_form` records per row
  which form was actually evaluated.
- **`cond(A, x)` is not part of `phi_cond`.** `cond_skeel_x` bounds the
  *limiting accuracy* refinement can reach, about `cond(A, x) u`, not the rate
  at which it gets there. It is carried in the `runs` table so a figure can
  draw that floor beside `ferr_ref`.
- **`phi_solve` is directional**, not the worst case. It measures
  `u_s ‖E_i d_i‖/‖d_i‖`, the error along the direction the correction actually
  took, which is a lower estimate of the worst-case `u_s ‖E_i‖_∞` the
  Corollary bounds with.
- **`phi_solve`'s reference correction is always a plain complex128 solve**,
  even under `--reference-solver extended`. Reusing the extended reference's
  own refinement loop there costs `O(n_rhs · outer_iters · nnz)` of
  unvectorized clongdouble work per index — minutes on a QTBM block with
  millions of nonzeros — for a term that never feeds the summary figure or the
  convergence decision. `x_true` still gets the full extended refinement, so
  only `phi_solve_hat`, `phi_hat` and `reference_correction_norm_inf` are
  affected; `ferr_ref`, `outer_iters`, `converged`, `rho`, `mu_hat` and
  `phi_cond_hat` are unchanged.

  Sound for **LU-IR** (`u_s = u_f`, far above the reference's own
  `cond(A,x) u`). For **GMRES-IR** `u_s = u`, the same precision the reference
  carries, so `phi_solve_hat` becomes noise once `cond(A,x)` is large — the
  regime GMRES-IR exists for. Don't quote it there from a run made this way;
  `_ExtendedReference.fast_solve` documents the one-line restore.

---

## 5. The HDF5 file

**One invocation is one experiment, and every experiment is kept.** Nothing
is ever overwritten — each run appends the next numbered group. And the file
lives in a directory of its own per material, not as a bare file, because the
plotting script's figures go into that same directory (§6) — the whole
material, data and figures together, is then one thing to `scp -r`:

```
<outdir>/<material>/<material>.h5
└── experiments/
    ├── 0001/          attrs: the whole run configuration
    │   ├── runs        one row per (index, variant)
    │   └── iterations  one row per (index, outer step), plus one
    │                  terminal row for the returned solution
    ├── 0002/
    └── 0003/
```

The number is zero-padded (HDF5 sorts keys as strings — unpadded, `10` would
sort before `2`) and is the lowest unused one, so deleting an experiment by
hand doesn't leave a permanent gap.

**Why numbered experiments rather than one rewritten table**, unlike every
other analysis in this pipeline: the interesting mpir runs *differ* in
precision, inner solver and stopping thresholds, and comparing them is the
point — so each is kept beside the configuration that produced it, rather than
the last run silently replacing the previous one.

### Experiment attributes

Everything needed to say what a run was, on the experiment group itself:
`material`, `source`, `timestamp`, `command`, `solver`, `factor_dtype`,
`inv_dtype`, `inner`, `inner_label`, `working_dtype`, `residual_dtype`,
`working_u`, `max_iter`, `ferr_tol`, `gmres_tol`,
`gmres_restart`, `gmres_max_iter`, `reference_solver`, `repeats`, `indices`, `n_requested`,
`n_skipped`, `criteria`, `convergence_factor`, plus the material's band edges
and energy grid.

### `runs` — one row per (index, variant)

```
idx, energy, n, nnz, n_rhs, n_blocks, solver, factor_dtype, inner, variant,
is_refined, u_f, u, u_s, kappa_2, kappa_inf, cond_skeel, cond_skeel_x,
lu_ir_bound,
relres, ferr_ref, eta1, eta2, etainf, omega,
outer_iters, converged, ferr_best, ferr_tol, stop_reason,
gmres_total, gmres_avg, wall_s, factor_s, factor_symbolic_s, factor_numeric_s, inner_s,
solve_s, residual_s, other_s, n_solves,
factor_mb, factor_mb_reported, working_mb, reference_solver, reference_nbe,
reference_floor
```

All three measured variants appear here (three rows per index).

The timing columns are two nested splits. `factor_symbolic_s` and
`factor_numeric_s` split the factorization into the analysis phase — the
fill-reducing ordering and the symbolic factorization, or for the Block Thomas
families the detection and extraction of the blocks — and the numerical phase.
The analysis phase performs no floating-point arithmetic, so it costs the same
at every `u_f` and bounds the speedup a lower precision can produce. cuDSS and
MUMPS time their own phases and the Block Thomas builders are timed in two
steps; SuperLU and UMFPACK fuse both into one call and report `nan`.

`solve_s`, `residual_s` and `other_s` split `inner_s` the same way: the
low-precision solves (`n_solves` of them), the working-precision residuals, and
what is left — the update, the stopping monitor and, for GMRES-IR, the products
with `A` and the orthogonalization. Both sub-timers run strictly inside the
`inner_s` window, so the three sum to it and `other_s` is non-negative by
construction.

`factor_mb_reported` is 0 where the backend exposed no factor size (MUMPS when
`INFOG(3)` is unreachable, cuDSS when `lu_nnz` is absent). Zero bytes of
factors is not a possible measurement, so `factor_mb` is left at 0 rather than
guessed and `working_mb` is then a lower bound.

These columns are what the companion cost study reads; see `mpcost.py` and
[README.md, section 7](README.md#7-mpcostpy).

### `iterations` — one row per (index, outer step), plus a terminal row

```
idx, energy, n, nnz, solver, factor_dtype, inner, variant, outer_iteration,
relres, residual_norm_inf, ferr_ref, rho, etainf, omega,
mu_hat, phi_cond_hat, phi_solve_hat, phi_hat, phi_cond_binding, phi_cond_form,
ferr_ratio,
correction_norm_inf, reference_correction_norm_inf,
gmres_inner_iterations, gmres_inner_max, note
```

**One row more than `outer_iters`** when the run stopped on an increase: the
iterate whose forward error rose is recorded too, since it is the evidence the
rule acted on. It carries `ferr_ref`/`relres`/`etainf`/`omega` but no
`correction_norm_inf`, `phi_solve_hat` or `gmres_inner_iterations`, because no
correction was computed from it; those come back `NaN`/`-1` rather than a
stale value.

Only the refinement variant performs outer steps, so only it appears here.
`ferr_ratio` is `ferr_i/ferr_{i-1}`, the stopping rule's own quantity: the step
where it first exceeds 1 is the step the loop stopped on. It equals `rho`
shifted by one step (`ferr_ratio[i] == rho[i-1]`) but is kept separately since
it's what the monitor actually saw, looking backward, at the moment it
decided.

Both tables carry `idx`, so either joins back to the other when a figure needs
both. Raw iterate/correction/residual **vectors are not stored** — `O(n)` per
step per index would run to gigabytes over a sweep, and every quantity a
figure needs is already reduced to a scalar per step. A study that needs the
vectors themselves has to re-run the single index it cares about.

### Reading it back

```python
import mpir
mpir.experiment_names(path)                    # ['0001', '0002', ...]
name, attrs, runs, iters = mpir.load_experiment(path)             # last one
name, attrs, runs, iters = mpir.load_experiment(path, 2)          # by number
```

`runs` and `iters` are the column dicts `factor_io.load_table` returns (one
NumPy array per column); `factor_io.table_rows(...)` turns either into a list
of per-row dicts.

---

## 6. Plotting

`plotting/mixed_prec_ir/plot_mpir.py` draws one experiment, in the layout of
Carson and Higham's numerical experiments. Figures are written into
`exp<NNNN>/` beside the analysis file by default:

```
<outdir>/<material>/
├── <material>.h5
└── exp0001/
    ├── <material>_summary.png
    └── <material>_E<idx>.png
```

```bash
python plot_mpir.py <material.h5> --list                    # what's in the file
python plot_mpir.py <material.h5>                            # plot the latest experiment
python plot_mpir.py <material.h5> --experiment 3              # a specific one
python plot_mpir.py <material.h5> --experiment 3 --idx 84 254 # only these indices
python plot_mpir.py <material.h5> --summary-only              # skip per-index figures
```

### Per-index figure — `exp<NNNN>/<material>_E<idx>.png`

One per energy index (capped by `--max-figures`, unless `--idx` names specific
ones). One panel: the **convergence history** — `ferr_ref` (red),
`nbe = etainf` (blue), `cbe = omega` (green) against the outer refinement step,
log scale, with a dotted line at the working precision `u`. When the run
stopped on an increase you can see it: `ferr_ref` turns upward on the last
point, and `outer_iters` in the title is the step before it.

The title carries the material, the energy, `kappa_inf`, the method, the outer
iteration count, and why the loop stopped.

The convergence-factor panel that used to sit beside it is gone. `mu_hat` and
the `phi_*` columns are still recorded in the `iterations` table; nothing
plots them.

### Summary figure — `exp<NNNN>/<material>_summary.png`

**One panel: outer iterations against `kappa_inf(A)`.** One point per index,
x on a log scale, y an integer count.

- **blue** where the run converged, **red** where it did not (§3) — including a
  point pinned at `outer_iters = 0` where the low-precision factorization
  itself failed to form (§3, "When the factorization itself fails"), which is
  otherwise indistinguishable at a glance from a run that simply made no
  progress; `stop_reason` in the `runs` table tells the two apart.
- a dashed vertical line at `kappa_inf = 1/u_f`, the classical LU-IR
  requirement `kappa_inf(A) u_f < 1`. Points to the right of it are outside
  what the theory guarantees — exactly where GMRES-IR should keep working and
  LU-IR should not.
- for **GMRES-IR only**, each point is labelled with `gmres_avg`: the mean
  inner-GMRES iteration count of one correction, averaged over every outer
  step and every right-hand side at that index. The y position already shows
  `outer_iters`, so labelling with that again would say nothing new — the
  inner count is the number a cost comparison actually needs, and it has to be
  one average rather than the full per-step table, since a point stands for
  several outer steps times several right-hand sides. `gmres_avg` is exact
  even when the run's last correction was later discarded (§3): it is computed
  from every correction actually attempted, not from `gmres_total` divided by
  `outer_iters * n_rhs`, which would undercount by leaving that correction's
  cost out of the denominator while it is still in the numerator.

Points are sorted by `kappa_inf`; indices with no `kappa_inf` in the
condition-est file are dropped, and the count of dropped indices is printed.

The y axis runs `0` to the experiment's own `--max-iter`, not to the largest
`outer_iters` actually seen in it — otherwise a run that converged in 3 steps
and one that needed 25 draw axes of very different height, and two summary
figures aren't comparable at a glance even when both used the same safety
net. `plot_mpir.py --y-max N` overrides this to compare experiments that used
different `--max-iter` values on one common scale.

This is the figure the study exists to produce. Nothing else is plotted across
the sweep: the iteration count multiplies the cost of the cheap factorization,
so it is what decides whether mixed-precision refinement is worth using.

---

## References

- E. Carson and N. J. Higham, *Accelerating the solution of linear systems by
  iterative refinement in three precisions*, SIAM J. Sci. Comput. 40(2), 2018.
  — Corollary 3.3, the convergence factor.
- E. Carson and N. J. Higham, *A new analysis of iterative refinement and its
  application to accurate solution of ill-conditioned sparse linear systems*,
  SIAM J. Sci. Comput. 39(6), 2017. — GMRES-IR.
- E. Oktay and E. Carson, *Multistage mixed precision iterative refinement*,
  Numer. Linear Algebra Appl. 29(4), 2022; arXiv:2107.06200. — section 2.1.1,
  the stopping criteria.
- J. Demmel et al., *Error bounds from extra-precise iterative refinement*,
  ACM TOMS 32(2), 2006. — the `phi` convergence-test estimate.
- A. Buttari et al., *Mixed precision iterative refinement techniques for the
  solution of dense linear systems*, IJHPCA 21(4), 2007. — LU-IR.

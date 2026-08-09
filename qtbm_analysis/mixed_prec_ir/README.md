# `mixed_prec_ir/` — mixed-precision iterative refinement

If a reduced-precision factorization is not accurate enough on its own, can it
still be *used*, as the inner solver or the preconditioner of a refinement
scheme? These scripts answer that for the QTBM systems.

All scripts here take the canonical option names; see
[the top-level README, section 3](../README.md#3-command-line-conventions).

| Script | Scope |
|---|---|
| `mpir.py` | the library and the single-index driver: LU-IR and GMRES-IR over any solver in `../solvers/` |
| `c32_gmres_ir.py` | half-precision Block Thomas as a GMRES-IR preconditioner, swept over an energy range |
| `sparse.py` | the earlier standalone study: fp32 SuperLU with fp64 refinement |
| `dense.py` | the earlier standalone study: fp32 LAPACK with fp64 refinement |

No figures are produced here. `c32_gmres_ir.py` writes a CSV, which
[`../plotting/plot_mixed_prec_ir.py`](../plotting/) renders.

---

## 1. The three precisions

Iterative refinement solves `A x = b` by computing a solution in a low
precision and correcting it using residuals computed in a higher one. The
modern analysis (Carson and Higham, 2017 and 2018) distinguishes:

| Symbol | Meaning | Value here |
|---|---|---|
| `u_f` | precision of the factorization | `--factor-dtype`; `complex64` or half |
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

| Variant | Approximate requirement | Limit at `u_f` = fp32 | at `u_f` = fp16 |
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

```bash
python mpir.py .../carbon-nanotube.h5 --idx 5 --solver superlu \
    --factor-dtype complex64
python mpir.py .../carbon-nanotube.h5 --idx 5 --solver block-thomas \
    --block-size 32 --factor-dtype complex64
python mpir.py .../si-bulk.h5 --idx 254 --solver mumps --factor-dtype complex64 \
    --inner gmres --gmres-tol 1e-8 --gmres-restart 30 --gmres-max-iter 50
python mpir.py .../si-bulk.h5 --idx 254 --solver cudss --factor-dtype complex64
```

Three variants of the **same** solver family are measured, so the comparison
isolates the effect of precision and refinement rather than of the solver
implementation:

1. the solver at `u_f` with refinement — the method under test
2. the solver at `complex128`, no refinement — the accuracy reference
3. the solver at `u_f`, no refinement — the lower bound refinement must beat

Reported per variant: relative residual, forward error against the reference
solution, wall time, peak Python heap and peak RSS above baseline, factor
memory, the outer convergence history, and, for GMRES-IR, the inner iteration
counts.

### Practical notes

**UMFPACK has no single-precision build**, so `solver_classes.UMFPACK` raises
`TypeError` for `--factor-dtype complex64`. This is a property of the library, not
of this script; select a different solver for a low-precision comparison.

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

**Memory is measured two ways** because neither suffices alone. `tracemalloc`
records the Python heap exactly but does not observe allocations inside
compiled extensions; the RSS poller observes those but samples at a few
milliseconds and may miss short peaks.

---

## 5. `c32_gmres_ir.py`

Extends the study to the half-precision Block Thomas factorization and sweeps
an energy range.

```bash
python c32_gmres_ir.py .../carbon-nanotube.h5 --idx 84 --block-size 32
python c32_gmres_ir.py .../carbon-nanotube.h5 --start 0 --end 401 \
    --block-size 32 --outdir plots
python ../plotting/plot_mixed_prec_ir.py plots/carbon-nanotube_fp16_gmres_ir.csv
```

The refinement drivers are not reimplemented. This script imports `mpir` and
registers one additional builder, `block-thomas-fp16`, into
`mpir.SOLVER_BUILDERS` at run time — an
in-memory dict insertion, not a modification of any file — so that `mpir`'s own
`solve_gmres_ir`, `solve_mixed_ir`, `solve_direct` and `benchmark_solver` drive
the half-precision solver unchanged. Any correction to the refinement logic
therefore applies here automatically.

Five variants are compared per energy index, on the same matrix and right-hand
side:

| Variant | Role |
|---|---|
| fp16 direct | the lower bound |
| fp16 + LU-IR | establishes whether plain refinement suffices |
| fp16 + GMRES-IR | **the variant under test** |
| c64 + GMRES-IR | a reference point at a precision where refinement is known to work |
| c128 direct | the accuracy ceiling |

`x_true` comes from SuperLU at `complex128`, the same convention `mpir` uses,
so forward errors are comparable across both scripts.

### The cast precision for the half-precision variants

`mpir` casts each vector with `v.astype(low_dtype)` before handing it to the
preconditioner. There is no `complex32` in NumPy, and `BlockThomasFP16`
performs its own rounding to `float16` and its own power-of-two rescaling
internally on every solve. `complex128` is therefore passed as the cast dtype:
the cast is then lossless and all precision loss occurs inside the
half-precision solver, where it belongs. Passing `complex64` would silently
insert an additional rounding step ahead of the half-precision one and would
misattribute its effect.

### Output

`<outdir>/<material>_fp16_gmres_ir.csv`, in long format, one row per
`(index, variant)`, with the run configuration in the header lines. A verbose
per-index log goes to stdout. The convergence history and the inner iteration
counts, rather than any single final number, are the evidence for whether
half-precision preconditioning works.

---

## 6. `sparse.py` and `dense.py`

The earlier standalone studies, on a single system rather than a sweep, and
against SciPy's sparse solvers and LAPACK directly rather than through the
solver library. Both compare fp32 with fp64 refinement against pure fp64 and
pure fp32.

`dense.py` has no half-precision variant because CPU LAPACK provides no fp16
factorization; that requires cuSOLVER or MAGMA on a device. The half-precision
study is therefore carried out with the Block Thomas implementations in
`../solvers/solver_classes.py`, which simulate fp16 arithmetic in NumPy.

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

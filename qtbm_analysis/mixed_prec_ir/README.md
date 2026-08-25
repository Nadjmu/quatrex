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

Nothing is written to disk and no figures are produced.

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
python mpir.py .../carbon-nanotube.h5 --start 0 --end 400 \
    --solver block-thomas --factor-dtype complex32 --inner gmres
```

Repeats with the median reported, for the noisier GPU timings.

```bash
python mpir.py .../si-bulk.h5 --idx 254 --solver cudss \
    --factor-dtype complex64 --repeats 5
```

A different reference solver for the forward error — `--reference-solver`
defaults to `superlu`.

```bash
python mpir.py .../si-bulk.h5 --idx 254 --solver cudss \
    --factor-dtype complex64 --reference-solver mumps
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

`mpir.py` writes nothing to disk. Its report is the console table, the
convergence history and the inner iteration counts, and for a sweep it is run
once per index. There is no plotting script or saved table for this study yet;
add one against whichever variants and metrics the thesis figure needs.

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

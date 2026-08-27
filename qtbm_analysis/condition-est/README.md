# `condition-est/` — condition number estimation

How ill-conditioned is `M(E)`, and in which sense? Every later chapter needs an
answer: the mixed-precision refinement bound is stated in `kappa_inf`, the
forward-error bound of the Block Thomas chapter needs both a normwise and a
componentwise condition number, and the test indices of both are selected by
`kappa_2` bucket. These scripts produce all five over a full energy sweep.

All scripts here take the canonical option names; see
[the top-level README, section 3](../README.md#3-command-line-conventions).

| Script | Scope |
|---|---|
| `condition_est.py` | the sweep: five condition numbers of `M(E)` per energy index, written to the analysis file |
| `list_by_kappa.py` | reads that file back and lists indices bucketed by `kappa_2`, for choosing test cases |

Figures are drawn by
[`../plotting/condition-est/plot_condition.py`](../plotting/condition-est/plot_condition.py);
nothing here imports matplotlib
([section 2.2 of the top-level README](../README.md#22-the-invariant)).

---

## 1. What is being estimated

Five numbers per matrix, three normwise and two componentwise.

| Column | Definition | How it is obtained |
|---|---|---|
| `cond_1` | `\|\|M\|\|_1 \|\|M^-1\|\|_1` | exact norm × estimated inverse norm |
| `cond_inf` | `\|\|M\|\|_inf \|\|M^-1\|\|_inf` | exact norm × estimated inverse norm |
| `cond_2` | `sigma_max / sigma_min` | both singular values computed, not estimated |
| `cond_skeel` | `\|\| \|M^-1\| \|M\| \|\|_inf` | estimated |
| `cond_skeel_x` | `\|\| \|M^-1\| \|M\| \|x\| \|\|_inf / \|\|x\|\|_inf` | estimated, worst right-hand side column |

`||M||_1` and `||M||_inf` are the maximum absolute column and row sum
of a sparse matrix and cost one pass over the entries. The whole difficulty is
`M^-1`, which is dense even when `M` is not and is never formed.

The five are reported separately rather than reduced to one number. The first
three are the same quantity in three norms and are bounded against each other
by factors of `n`; the last two are a different quantity and can be orders of
magnitude smaller than `cond_inf`.

---

## 2. The estimator

### 2.1 The problem

Estimating `kappa(M)` means estimating `||M^-1||` without forming `M^-1`.
The naive route — `n` solves against the columns of the identity — costs `n`
triangular substitutions per matrix, which over a sweep of thousands of indices
is out of reach. The exact `kappa_2` route — a full SVD — is dense `O(n^3)` and
destroys the sparsity entirely.

The observation that makes the problem tractable is that a *norm* of `M^-1`
does not require `M^-1` itself, only the ability to apply it to a vector. One
LU factorization supplies that, and every estimate below is a small number of
triangular solves against it.

### 2.2 Hager's algorithm and the Higham–Tisseur block generalization

`||A||_1 = max_j sum_i |a_ij|` is the maximum of the convex function
`f(x) = ||Ax||_1` over the unit 1-norm ball, and a convex function on a
convex set attains its maximum at a vertex — here at a column of the identity.
Hager's algorithm (1984) is a gradient ascent on `f`: from the current `x` it
computes `Ax`, takes `xi = sign(Ax)`, computes `A^T xi` — an element of the
subdifferential — and moves to the unit vector `e_j` where `|A^T xi|` is
largest. It stops when no vertex improves on the current value. Every step uses
only one product with `A` and one with `A^T`; `A` is never inspected.

Higham (1988) turned this into the LAPACK condition estimator (`xLACON`, driving
`xGECON`), adding the convergence and cycling safeguards and an extra probe
vector that defeats the matrices on which pure ascent stalls. Higham and Tisseur
(2000) generalized it to a *block* method that iterates on `t` columns at once
rather than one: the ascent then explores `t` vertices per step, which both
tightens the estimate and makes the failure cases far rarer, at the price of
`t` times as many products. This is LAPACK's `xLACN2`, MATLAB's `normest1`, and
SciPy's `scipy.sparse.linalg.onenormest` — the function used here. `--onenorm-t`
sets `t`, defaulting to `2`, the value Higham and Tisseur recommend.

Two properties matter for how the results are read:

- **It is a lower bound.** The estimate never exceeds the true norm. In practice
  it is within a small factor — almost always exact or nearly so at `t = 2` —
  but `cond_1`, `cond_inf`, `cond_skeel` and `cond_skeel_x` are therefore
  estimates *from below* of exactly defined quantities. `cond_2` is not.
- **It is matrix-free.** It needs `A @ v` and `A^H @ v` and nothing else, so
  `A` can be any `LinearOperator`. This is what allows the same routine to
  estimate four different quantities from one factorization.

### 2.3 The four operators

`inverse_operators()` builds `M^-1` and `M^-H` as `LinearOperator`s over one
`splu` factorization, using its `trans="N"` and `trans="H"` solves.

```
    ||M^-1||_1    = onenormest(M^-1)
    ||M^-1||_inf  = ||M^-H||_1  = onenormest(M^-H)
```

The infinity norm is reached through the adjoint rather than being estimated
directly, so that both inverse norms are one-norm estimates and `M^-T` is never
needed. The identity is `||A||_inf = ||A^H||_1`, exact for complex `A`.

> **Adjoint, not transpose.** SciPy takes `LinearOperator.rmatvec` to be the
> *adjoint*, and `onenormest` forms `A.H` from it. A transpose solve in that
> slot silently estimates the norm of a different operator whenever `M` is
> complex, which it always is here. Both operators are therefore built from the
> `trans="N"` and `trans="H"` solves of the same factorization, which are exact
> adjoints of one another. The same care is taken in
> `abs_inverse_apply_norm()`.

---

## 3. `kappa_2`, and why it is not estimated the same way

`cond_2` needs the two ends of the singular spectrum. Neither is a 1-norm, so
the Hager–Higham machinery does not apply; both come from ARPACK through
`scipy.sparse.linalg.svds`.

`sigma_max` is the dominant singular value and is computed on `M` itself, with
no factorization: it is the well-conditioned end for a Krylov method and
converges quickly.

`sigma_min` is the difficult end. Asking `svds` for `which="SM"` works through
`M^H M`, whose condition number is `kappa_2(M)^2`, and converges badly for
exactly the matrices this chapter is about. Instead it is reached by
**shift-invert**: `M = U S V^H` implies `M^-1 = V S^-1 U^H`, so the singular
values of `M^-1` are the reciprocals of those of `M` and

```
    sigma_min(M) = 1 / sigma_max(M^-1)
```

The smallest singular value of `M` is thus the *largest* of `M^-1` — again the
well-conditioned end — and `M^-1` is supplied as the same `LinearOperator` the
norm estimates already use. This is the same argument as in
[`../block-thomas/arnoldi_shift_invert_cpu.py`](../block-thomas/arnoldi_shift_invert_cpu.py),
which applies it to a single matrix with a choice of backends; here the backend
is fixed to SuperLU because the factorization is required anyway.

`--svd-tol` (default `1e-10`) and `--svd-ncv` control convergence; raise `ncv`
if a clustered spectrum stalls.

Note the asymmetry this creates. `cond_2` is a converged computation of both
singular values, while `cond_1` and `cond_inf` are lower bounds. The gap between
the three curves in the figure therefore mixes the choice of norm with the slack
of the estimator, and is not read as either alone.

---

## 4. The Skeel condition numbers

### 4.1 Why a componentwise condition number

The normwise perturbation bound

```
    ||x - xhat||_inf / ||x||_inf  <~  kappa_inf(M) * eta_inf
```

pairs `kappa_inf` with the normwise backward error `eta_inf`. It measures the
perturbation as a single number against `||M||`, which means a matrix with
badly scaled rows is charged for its largest row everywhere. Skeel (1979)
replaces the norm by an entrywise comparison, giving

```
    cond(M, x) = || |M^-1| |M| |x| ||_inf / ||x||_inf
    cond(M)    = || |M^-1| |M| ||_inf
```

and the companion bound

```
    ||x - xhat||_inf / ||x||_inf  <~  cond(M, x) * omega
```

where `omega` is the componentwise backward error of Oettli and Prager, the
smallest `eps` for which `xhat` solves a system whose entries are each perturbed
by at most a relative `eps`. Because `|M^-1| |M|` is invariant under row
scaling of `M`, `cond(M) <= kappa_inf(M)` always, and often by orders of
magnitude; `cond(M, x)` is smaller still and is the sharpest of the three.

This is precisely the pairing
[`../block-thomas/forward_error.py`](../block-thomas/forward_error.py) needs: it
forms `bound_nw = cond_inf * eta_inf` and `bound_cw = cond_skeel_x * omega` and
plots the observed forward error against both.

### 4.2 Estimating them without forming `|M^-1|`

`|M^-1|` is as dense as `M^-1` and just as unavailable, so the estimator is
applied a second time, through an identity. For a nonnegative vector `g`,

```
    || |M^-1| g ||_inf  =  || M^-1 diag(g) ||_inf  =  || diag(g) M^-H ||_1
```

The middle step holds because `|M^-1| g` is the vector of row sums of
`|M^-1 diag(g)|`, whose largest entry is that matrix's infinity norm; the last
is again `||A||_inf = ||A^H||_1`. So `abs_inverse_apply_norm()` builds
`X = diag(g) M^-H` as a `LinearOperator` — one triangular solve and one scaling
per product — and calls `onenormest` on it.

Both Skeel numbers are then the same computation with a different `g`, using
that `|M^-1| |M|` is a nonnegative matrix, whose infinity norm is therefore
the largest entry of its action on the vector of ones:

| Quantity | `g` |
|---|---|
| `cond_skeel` | `\|M\| e` |
| `cond_skeel_x` | `\|M\| \|x\|` |

`cond_skeel_x` needs a solution, so `E_<idx>/rhs` is read from the material file
and solved with the factorization already in hand; the worst column is kept, and
the column is `NaN` where the right-hand side has none. This is the estimate
LAPACK forms inside `xyyRFS` for its componentwise error bound, and the one
Demmel et al. (2006) refine with extra precision.

---

## 5. Why these estimators and not others

The chapter's survey, and where each option lands.

| Approach | Cost per matrix | Guarantee | Used here |
|---|---|---|---|
| Explicit `M^-1` | `n` solves | exact | no — infeasible over a sweep |
| Full SVD | dense `O(n^3)` | exact `kappa_2` | only as the reference `global/condition_full_svd`, written once at export time |
| LINPACK estimator (Cline et al., 1979) | a few solves | lower bound, no safeguards | no — superseded |
| Hager / `xLACON` (1984, 1988) | `~5` solves | lower bound | superseded by the block form |
| **Higham–Tisseur `xLACN2` / `onenormest`** | `~4t` solves | lower bound, usually tight | **yes** — all four estimated columns |
| **Shift-invert `svds`** | one factorization + Krylov | converged, both ends | **yes** — `cond_2` |
| Randomized / statistical (Kenney–Laub SCE) | a few products | probabilistic confidence | no — a probabilistic interval is harder to defend as an input to a deterministic error bound |

Four reasons decide it:

1. **`kappa_inf` is not optional.** The LU-IR convergence condition is stated as
   `kappa_inf(M) u_f < 1` ([`../mixed_prec_ir/README.md`](../mixed_prec_ir/README.md),
   section 3). The chapter that follows needs this specific number in this
   specific norm, and `onenormest` on `M^-H` is the standard way to it.
2. **`kappa_2` is the norm everything else is stated in.** The singular
   spectrum, the band-edge behaviour and the non-normality study all live in the
   2-norm, and `cond_2` is comparable against the full-SVD reference stored at
   export time — which is how the estimator itself is validated.
3. **The componentwise pair is required by the forward-error chapter.** A
   normwise bound alone cannot explain a forward error that sits orders of
   magnitude below `kappa_inf * eta_inf`; `cond_skeel_x * omega` can.
4. **Marginal cost.** One `splu` per index dominates. Given it, `cond_1`,
   `cond_inf`, `cond_skeel` and `cond_skeel_x` are four `onenormest` calls —
   roughly `4t` triangular solves each — and `cond_2` two Krylov runs. Computing
   all five instead of one costs a small multiple of the factorization that had
   to be paid anyway, which is why nothing is left out.

`cond_1` is the clearest case of point 4: nothing downstream reads it, but the
adjoint operator it needs already exists for `cond_inf`, so it comes essentially
free and serves as a consistency check — `cond_1` and `cond_inf` must agree to
within a factor of `n`, and a violation means the estimator, not the matrix.

---

## 6. Cost and structure of one row

```
    splu(M)                                  once, dominates
    ||M||_1, ||M||_inf                       one pass over the entries
    onenormest(M^-1)          ~4t solves
    onenormest(M^-H)          ~4t solves
    svds(M,   k=1, LM)                       Krylov, no factorization
    svds(M^-1, k=1, LM)                      Krylov on the factorization
    onenormest(diag(|M|e) M^-H)   ~4t solves
    per rhs column: one solve, then ~4t more
```

The matrix is cast to `complex128` and `csc` once; `splu` wants CSC, and the
material files already store CSC triplets. `M` and the
factorization are dropped and `gc.collect()` called after every row, since a
sweep holds thousands of them in succession.

---

## 7. Running it

Every material in the standard scratch layout, every index:

```bash
python condition_est.py
```

One material, thinned — one row costs a factorization and two Krylov runs, so
`--stride` is the cheap way to a first pass over a dense sweep:

```bash
python condition_est.py --material carbon-nanotube --stride 10
```

An explicit file, resuming an interrupted run. Each row is flushed as soon as it
is produced and marked in `valid`, so `--resume` recomputes nothing already
done:

```bash
python condition_est.py /scratch/yimili/matrices2/hdf5/graphene.h5 \
    --stride 10 --resume
```

Filling only the Skeel columns of a sweep made before they existed. No singular
values and no `onenormest` of `M^-1` itself, so it is a fraction of a full row:

```bash
python condition_est.py --material si-bulk --only-skeel
```

Tightening the estimator, or helping a clustered spectrum converge:

```bash
python condition_est.py --material graphene --onenorm-t 4 --svd-ncv 40
```

A material file that fails is warned about and skipped rather than discarding
the sweeps of the others, each of which costs hours.

---

## 8. Output

`<outdir>/<material>.h5`, group `condition`, opened in append mode so that other
analyses of the same material are preserved.

| Dataset | Shape | Meaning |
|---|---|---|
| `indices` | `(P,)` | energy index of each row |
| `valid` | `(P,)` | row fully computed |
| `nnz` | `(P,)` | nnz of `M`, `-1` if unknown |
| `norm1`, `norminf` | `(P,)` | exact norms of `M` |
| `norm1_inv`, `norminf_inv` | `(P,)` | estimated norms of `M^-1` |
| `sigma_max`, `sigma_min` | `(P,)` | extreme singular values |
| `cond_1`, `cond_inf`, `cond_2` | `(P,)` | the normwise condition numbers |
| `cond_skeel`, `cond_skeel_x` | `(P,)` | the componentwise ones |
| `seconds` | `(P,)` | wall time of the row |

Group attributes carry `material`, `source`, `n_indices`, `stride`,
`onenorm_t`, `svd_tol` and the material metadata — including `grid_energy_min`,
`resolution` and the band edges, which is what lets every consumer convert an
index to an energy.

The row dimension `P` is fixed by the index selection when the group is created,
so reopening it for a different selection is an error rather than a silent
partial overwrite; use `--overwrite` or a different `--outdir`. A group written
before a column existed is extended with a `NaN` column rather than rejected.

### Consumers

| Reader | Columns |
|---|---|
| [`../plotting/condition-est/plot_condition.py`](../plotting/condition-est/plot_condition.py) | `cond_1`, `cond_2`, `cond_inf` |
| [`../block-thomas/forward_error.py`](../block-thomas/forward_error.py) | `cond_inf`, `cond_skeel`, `cond_skeel_x` |
| [`../mixed_prec_ir/mpir.py`](../mixed_prec_ir/mpir.py) | `cond_2`, `cond_inf` |
| `list_by_kappa.py` | `cond_2`, `cond_inf` |

---

## 9. `list_by_kappa.py`

Selecting the test indices for the later chapters. Buckets the valid rows by
`kappa_2` in decade-wide bins above `--first-edge`, and lists the index, energy
and both condition numbers of each row.

```bash
python list_by_kappa.py /scratch/yimili/condition-est/carbon-nanotube.h5
python list_by_kappa.py .../si-bulk.h5 --first-edge 1000 --max-per-bin 20
```

`kappa_2` is the bucketing variable because it is the one computed rather than
estimated; `kappa_inf` is listed beside it as the check against the LU-IR bound,
which is stated in the infinity norm. Rows whose energy falls inside the band gap
are dropped — there is no transport there and the conditioning is not
representative — which needs `valence_band_edge` and `conduction_band_edge` in
the group attributes; without them every index is listed and a warning is
printed. The listing goes to `--out` as well as to the terminal, since an
uncapped sweep over thousands of indices is not usable as scrollback.

---

## References

- W. W. Hager, Condition estimates, *SIAM J. Sci. Statist. Comput.* 5(2), 1984.
  The ascent algorithm every estimator here descends from.
- N. J. Higham, FORTRAN codes for estimating the one-norm of a real or complex
  matrix, with applications to condition estimation, *ACM TOMS* 14(4), 1988.
  The safeguarded version that became LAPACK's `xLACON`.
- N. J. Higham and F. Tisseur, A block algorithm for matrix 1-norm estimation,
  with an application to 1-norm pseudospectra, *SIAM J. Matrix Anal. Appl.*
  21(4), 2000. The block generalization; `t = 2` is their recommendation and the
  default of `--onenorm-t`.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM,
  2002. Chapter 7 for the normwise and componentwise perturbation theory,
  Chapter 15 for condition estimation.
- R. D. Skeel, Scaling for numerical stability in Gaussian elimination,
  *J. ACM* 26(3), 1979. The componentwise condition number.
- W. Oettli and W. Prager, Compatibility of approximate solution of linear
  equations with given error bounds, *Numer. Math.* 6, 1964. The componentwise
  backward error `omega`.
- M. Arioli, J. W. Demmel and I. S. Duff, Solving sparse linear systems with
  sparse backward error, *SIAM J. Matrix Anal. Appl.* 10(2), 1989. The pairing
  of `omega` with the Skeel condition number, and its use in refinement.
- A. K. Cline, C. B. Moler, G. W. Stewart and J. H. Wilkinson, An estimate for
  the condition number of a matrix, *SIAM J. Numer. Anal.* 16(2), 1979. The
  LINPACK estimator.
- C. S. Kenney and A. J. Laub, Small-sample statistical condition estimates for
  general matrix functions, *SIAM J. Sci. Comput.* 15(1), 1994. The
  probabilistic alternative not taken here.
- J. Demmel, Y. Hida, W. Kahan, X. S. Li, S. Mukherjee and E. J. Riedy, Error
  bounds from extra-precise iterative refinement, *ACM TOMS* 32(2), 2006. The
  componentwise bound as LAPACK computes it.

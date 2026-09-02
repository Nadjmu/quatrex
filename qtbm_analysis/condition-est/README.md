# `condition-est/` — condition number estimation

How ill-conditioned is `M(E)`, and in which sense? The refinement bound in
[`../mixed_prec_ir/`](../mixed_prec_ir/) is stated in `kappa_inf`. The
forward-error bound in [`../block-thomas/`](../block-thomas/) needs a normwise
and a componentwise condition number together. The test indices for both are
chosen by `kappa_2` bucket. These scripts produce all of it over a full energy
sweep, using estimators rather than the exact quantities, because the exact
quantities are not affordable at this scale. This file explains the
estimators, how they work, and — the point of most of what follows — what they
cost against the alternative of computing the exact answer.

| Script | Scope |
|---|---|
| `condition_est.py` | the sweep: five condition numbers of `M(E)` per energy index, estimated |
| `exact_condition.py` | the same five quantities, computed exactly, at a hand-picked or `--all` selection of indices, as the reference the estimator is checked against |
| `list_by_kappa.py` | reads `condition_est.py`'s output back and lists indices bucketed by `kappa_2`, for choosing test cases |

Figures are drawn by
[`../plotting/condition-est/plot_condition.py`](../plotting/condition-est/plot_condition.py);
nothing here imports matplotlib
([top-level README, section 2.2](../README.md#22-the-invariant)).

---

## 1. What is being estimated

Five numbers per matrix: three normwise, two componentwise.

| Column | Definition | How it is obtained |
|---|---|---|
| `cond_1` | ‖M‖₁ ‖M⁻¹‖₁ | exact norm × estimated inverse norm |
| `cond_inf` | ‖M‖∞ ‖M⁻¹‖∞ | exact norm × estimated inverse norm |
| `cond_2` | σ_max / σ_min | both singular values computed, not estimated |
| `cond_skeel` | ‖ \|M⁻¹\| \|M\| ‖∞ | estimated |
| `cond_skeel_x` | ‖ \|M⁻¹\| \|M\| \|x\| ‖∞ / ‖x‖∞ | estimated, worst right-hand side column |

`‖M‖₁` and `‖M‖∞` are the largest absolute column and row sum of a sparse
matrix and cost one pass over the nonzero entries. The difficulty everywhere
else is `M⁻¹`, which is dense even when `M` is sparse and is never formed.

`cond_1` is not plotted (see section 4), but is kept: `kappa_1(M) =
kappa_inf(M^T)` exactly, and `M(E) = ES − H − Σ(E)` is nearly complex
symmetric, so `cond_1 ≈ cond_inf` here far more closely than the general
factor-of-`n` bound between the two norms allows. Agreement between them is
therefore a check that the `trans="N"` and `trans="H"` solves used to build
them are both correct, not a second measurement of conditioning.

The remaining four are kept apart rather than reduced to one number, because
they are not the same quantity. `cond_inf` and `cond_skeel`/`cond_skeel_x` obey
the ladder

```
cond_skeel_x  ≤  cond_skeel  ≤  kappa_inf
```

with equality only in special cases; the gaps are explained in section 3.

---

## 2. The estimator

### 2.1 Why an estimator at all

Two exact routes exist, and both are too expensive for a sweep of thousands of
energy indices.

**Explicit inverse.** Solving `M X = I` costs `n` triangular substitutions
against one factorization — the same asymptotic cost as forming `M⁻¹` by
`getrf` + `getri`, `O(n³)` (Golub & Van Loan, *Matrix Computations*, 4th ed.,
§3.2, gives ≈`2n³` flops for the two steps together on a dense matrix; a sparse
`M` gains nothing here, since the columns of `I` produce a dense right-hand
side one at a time and `M⁻¹` itself is dense).

**Full SVD.** `σ_max` and `σ_min` from a full singular value decomposition,
`O(n³)` again (Golub & Van Loan, §8.6: computing the singular values alone,
without `U` or `V`, is dominated by the bidiagonal reduction step at roughly
`(8/3)n³` flops — a few times the cost of one `LU`, but the same cubic order).

Neither is affordable per index. What makes an estimate possible instead is
that a *norm* of `M⁻¹` does not require `M⁻¹` itself — only the ability to
multiply by it. One sparse LU factorization supplies exactly that, and every
estimated column below is a handful of triangular solves against it.

### 2.2 Hager's ascent, and what LAPACK made of it

`‖A‖₁ = max_j Σ_i |a_ij|` is the maximum of the convex function `f(x) =
‖Ax‖₁` over the unit 1-norm ball. A convex function on a convex set reaches
its maximum at a vertex of that set — here, a column of the identity — so
computing the norm reduces to a search over vertices, and a search needs only
products.

| Year | Contribution |
|---|---|
| 1979 | Cline, Moler, Stewart & Wilkinson: the LINPACK estimator, a few solves, a lower bound, no safeguards against the matrices that defeat it. |
| 1984 | Hager: the ascent itself. From `x`, compute `Ax`, set `ξ = sign(Ax)`, compute `Aᵀξ`, move to `e_j` at the largest entry of `Aᵀξ`; stop when no vertex improves. One product with `A` and one with `Aᵀ` per step. |
| 1988 | Higham: convergence and anti-cycling safeguards, plus an extra probe vector for the matrices where plain ascent stalls. This is LAPACK's `xLACON` / `xGECON`. |
| 2000 | Higham & Tisseur: the block form, ascending on `t` columns at once instead of one. `t` corners are tested per step, tightening the estimate and making failure far rarer, at `t` times the cost. This is LAPACK's `xLACN2`, MATLAB's `normest1`, and SciPy's `onenormest` — the routine `condition_est.py` calls. |

`--onenorm-t` sets `t`; the default is 2, the authors' own recommendation.

Two properties of this estimator govern how the output is read.

- **It is a lower bound.** The result never exceeds the true norm. In
  practice it is close — at `t = 2` usually exact or nearly so — but
  `cond_1`, `cond_inf`, `cond_skeel` and `cond_skeel_x` are estimates *from
  below* of exactly-defined quantities. `cond_2` is not.
- **It needs only `A @ v` and `A^H @ v`.** `A` can therefore be any
  `LinearOperator`, never an explicit matrix. This is the property that lets
  one factorization serve four different quantities.

### 2.3 The operators built from one factorization

`inverse_operators()` wraps one `splu` factorization as `M⁻¹` and `M⁻ᴴ`,
using its `trans="N"` and `trans="H"` solves.

```
‖A‖∞ = ‖A^H‖₁   ⟹   ‖M⁻¹‖∞ = onenormest(M⁻ᴴ)
```

Both inverse norms are therefore 1-norm estimates of an operator the same
factorization supplies. `M⁻ᵀ` is never formed.

> **Adjoint, not transpose.** SciPy's `LinearOperator.rmatvec` is the
> *adjoint*, and `onenormest` builds `A.H` from it. A transpose solve in that
> slot silently estimates the norm of a different operator whenever `M` is
> complex — which it always is here — with no error raised. Both operators
> are built from the `trans="N"` and `trans="H"` solves of the same
> factorization, which are exact adjoints of one another; the same care is
> taken in `abs_inverse_apply_norm()`.

---

## 3. The Skeel numbers, and the ladder against `kappa_inf`

### 3.1 Why a componentwise measure

The normwise bound

```
‖x − x̂‖∞ / ‖x‖∞  ≲  kappa_inf(M) · eta_inf
```

pairs `cond_inf` with the normwise backward error `eta_inf`. It compares the
perturbation of `M` to a single number, `‖M‖`, so a matrix with badly scaled
rows is charged everywhere for its largest row.

Skeel (1979) compares entrywise instead, giving `cond(M) = ‖ |M⁻¹| |M| ‖∞`
and `cond(M, x) = ‖ |M⁻¹| |M| |x| ‖∞ / ‖x‖∞`, paired with the componentwise
backward error `omega` of Oettli & Prager (1964):

```
‖x − x̂‖∞ / ‖x‖∞  ≲  cond(M, x) · omega
```

### 3.2 The ladder is exact, not just an inequality

```
cond(M, x)  ≤  cond(M)  ≤  kappa_inf(M)
```

The first step: `cond(M)` is the supremum of `cond(M, x)` over every `x`,
attained at `x = e`. The second step: `‖ |M⁻¹| |M| ‖∞ ≤ ‖M⁻¹‖∞ ‖M‖∞` by
submultiplicativity, but the identity behind the gap is sharper —

```
cond(M) = min over positive diagonal D of kappa_inf(D M)
```

(Bauer 1963; van der Sluis 1969; Higham, *ASNA*, 2nd ed., §7.3). `cond(M)` is
the infinity-norm condition number *after optimal row scaling*, so
`kappa_inf / cond(M)` is precisely the factor by which row scaling inflates
`kappa_inf` — a number `plot_condition.py` prints as `[scaling]`, the median
of that ratio over the sweep, since it is exactly the headroom row
equilibration (LAPACK `xGEEQU`) could recover.

### 3.3 Estimating the Skeel numbers without `|M⁻¹|`

`|M⁻¹|` is as dense and as unaffordable as `M⁻¹`. The same block estimator is
applied a second time, through a third identity: for nonnegative `g`,

```
‖ |M⁻¹| g ‖∞  =  ‖ M⁻¹ diag(g) ‖∞  =  ‖ diag(g) M⁻ᴴ ‖₁
```

`|M⁻¹|g` is the vector of row sums of `|M⁻¹ diag(g)|`; its largest entry is
that matrix's infinity norm, and the last equality is section 2.3's identity
again. `abs_inverse_apply_norm()` builds `X = diag(g) M⁻ᴴ` as a
`LinearOperator` — one triangular solve and one scaling per product — and
calls `onenormest` on it.

Both Skeel numbers are the same call with a different `g`, using that `|M⁻¹|
|M|` is nonnegative, so its infinity norm is the largest entry of its action
on a vector:

| Quantity | `g` |
|---|---|
| `cond_skeel` | `\|M\| e` |
| `cond_skeel_x` | `\|M\| \|x\|`, worst column of `E_<idx>/rhs` |

`cond_skeel_x` is `NaN` at an index whose right-hand side has no columns —
inside the band gap, where there is no transport and therefore no `rhs`
stored. `plot_condition.py` masks rather than deletes those points, so the
curve breaks there instead of a straight segment bridging across the gap
(see the module docstring of `plot_condition.py` for the mechanism).

---

## 4. Cost — the estimator against the alternatives

This is the section to cite when arguing that estimation, rather than exact
computation, is the only workable choice over a sweep.

### 4.1 Asymptotic cost, per energy index

| Operation | Cost | Where |
|---|---|---|
| Sparse LU factorization of `M` | amortized elsewhere — the same factorization every solver in this project already needs | `splu(M)` |
| One `onenormest` call | ≈`4t` triangular solves against the existing factorization | `cond_1`, `cond_inf`, `cond_skeel`, `cond_skeel_x` |
| `cond_2` (shift-invert `svds`, two calls) | a handful of Krylov iterations, each one more solve against the existing factorization | section 3 of the module docstring in `plot_condition.py`; the argument itself is in `condition_est.py`'s `singular_extremes()` |
| **Total per index (estimator)** | **one sparse factorization, plus `O(t)` triangular solves** | `condition_est.py` |
| Dense inverse (`getri` after `getrf`) | `O(n³)`, ≈`2n³` flops | `exact_condition.py`, `--no-svd` off |
| Full SVD, singular values only (`gesdd`) | `O(n³)`, dominated by bidiagonal reduction at ≈`(8/3)n³` flops | `exact_condition.py`, `--no-inverse` off |
| **Total per index (exact)** | **two dense `O(n³)` factorizations** | `exact_condition.py --all` |

The estimator's cost is dominated entirely by the one sparse factorization
every solve in this project already pays for. `M` is a block-tridiagonal
QTBM system matrix; its `LU` factors fill in only within the block
bandwidth, so a sparse or block factorization costs `O(n b²)` for block size
`b`, not `O(n³)` — see [`../block-thomas/README.md`](../block-thomas/README.md)
for the factorization this project actually uses. `cli.MATERIALS` fixes a
block size `b` for each material (`carbon-nanotube`: 32, `si-bulk`: 256,
`carbon-chain`: 104, `graphene`: 416), an order of magnitude or two below
`n`; two of the four matrix sizes were checked directly on the cluster in the
course of this work (`carbon-nanotube`: `n = 768`, `si-bulk`: `n = 3840`),
and at both, `n b²` is smaller than `n³` by several orders of magnitude. The
other two were not independently checked here; run the `n` query of
section 4.3's companion, `h5py.File(...)['E_0/M'].attrs['shape'][0]`, on the
material file before quoting a number for them. The `4t` solves the estimator adds on top of
that factorization are triangular substitutions against the same sparse
factors — themselves far below `O(n²)`, let alone `O(n³)`.

The exact computation has no such structure to exploit. `getri` and `gesdd`
both operate on the dense `n × n` array regardless of how sparse `M` was;
sparsity buys nothing once the matrix is densified, which is exactly why
`exact_condition.py` requires an explicit index selection (`--idx`,
`--start`/`--end`, or the deliberate opt-in `--all`) rather than defaulting
to every index the way `condition_est.py` does.

### 4.2 Memory

A dense `complex128` array costs `16n²` bytes, and `exact_condition.py` holds
several live at once (`M`, `M⁻¹`, the array `svd` reduces internally).
`--max-n` (default 8000, ≈1 GiB per array) refuses an index above that size
rather than letting a job exhaust memory partway through a sweep. For
`si-bulk` at `n = 3840`, one array is ≈225 MiB; for a hypothetical `n =
8000`, ≈1 GiB. The sparse `M` itself, by contrast, costs `nnz` complex
entries plus two index arrays — for these materials, a small fraction of one
dense array's footprint.

### 4.3 Measured cost, not just asymptotic

Both scripts record wall time per row, in the `seconds` column of their own
output group (`condition/seconds` for the estimator, `condition_exact/seconds`
for the exact reference), so the actual gap on this hardware is already in
the data rather than needing to be argued from complexity alone. To compare
them for one material:

```python
import h5py, numpy as np
with h5py.File("/scratch/yimili/condition-est/<material>.h5") as f:
    est = f["condition/seconds"][f["condition/valid"][:]]
    print("estimator: median", np.median(est), "s/row, n =", len(est))
    if "condition_exact" in f:
        exact = f["condition_exact/seconds"][f["condition_exact/valid"][:]]
        print("exact:     median", np.median(exact), "s/row, n =", len(exact))
```

This is the number to quote in the thesis: it is specific to the matrix size
and the machine the sweep ran on, which an asymptotic complexity count is
not. Section 4.1's `O(n b²)` versus `O(n³)` argument is why the gap has the
shape it does; this is what the gap actually is.

### 4.4 Why not skip the exact computation entirely

If the estimator is a lower bound, and the exact computation is expensive,
the remaining question is whether the estimate can simply be trusted. It can
be checked cheaply, which is the entire purpose of `exact_condition.py`: a
*handful* of indices, spanning the observed range of `kappa_2` (chosen with
`list_by_kappa.py`), computed exactly and compared against the estimate at
the same indices. `plot_condition.py`'s ratio panel and the `[slack]` lines
it prints are that comparison. The accuracy of the estimator on these
matrices is therefore not asserted here — it is a result of the sweep and is
read off the figures, not off this file. What this section establishes is
only that checking a handful of indices exactly, rather than every index,
is what makes the check affordable at all.

---

## 5. Why these estimators, and not other options

| Method | Cost per index | Guarantee | Used here |
|---|---|---|---|
| Explicit `M⁻¹` | `n` solves, `O(n³)` overall | exact | no — infeasible over a sweep |
| Full SVD | `O(n³)` | exact `kappa_2` | only via `exact_condition.py`, on a handful of indices |
| LINPACK estimator (1979) | a few solves | lower bound, unsafeguarded | no — superseded |
| Hager / `xLACON` (1984, 1988) | ≈5 solves | lower bound | no — superseded by the block form |
| **Higham–Tisseur `xLACN2` / `onenormest`** | ≈`4t` solves | lower bound, usually tight | **yes** — `cond_1`, `cond_inf`, `cond_skeel`, `cond_skeel_x` |
| **Shift-invert `svds`** | one factorization + Krylov | converged, both ends | **yes** — `cond_2` |
| Statistical (Kenney–Laub, 1994) | a few products | probabilistic confidence interval | no — an interval is a poor input to a bound the thesis states as a hard ceiling |

`kappa_inf` is required by name in the LU-IR convergence condition
`kappa_inf(M) u_f < 1` (see [`../mixed_prec_ir/README.md`](../mixed_prec_ir/README.md)).
`kappa_2` is the norm the rest of the thesis is stated in, and is the only
column checkable against the full SVD stored once at export time
(`global/condition_full_svd`). The Skeel pair is required by
[`../block-thomas/forward_error.py`](../block-thomas/forward_error.py)'s
componentwise bound. `cond_1` costs nothing beyond what `cond_inf` already
needs and serves only as a consistency check (section 1).

---

## 6. Kahan's theorem, for the spectral figure

`1 / kappa_2(M)` is the relative 2-norm distance from `M` to the nearest
singular matrix (Kahan 1966; also Demmel & Kahan). A `kappa_2` spike at a
particular energy is therefore not merely "hard to solve" — it says `M(E)` at
that energy is, in the 2-norm, within a specific relative distance of being
exactly singular. This is the reading the spectral figure supports that the
normwise/componentwise figure does not.

---

## 7. Output

### `condition_est.py` → `<outdir>/<material>.h5`, group `condition`

| Dataset | Shape | Meaning |
|---|---|---|
| `indices` | `(P,)` | energy index of each row |
| `valid` | `(P,)` | row fully computed |
| `nnz` | `(P,)` | nnz of `M`, `-1` if unknown |
| `norm1`, `norminf` | `(P,)` | exact norms of `M` |
| `norm1_inv`, `norminf_inv` | `(P,)` | estimated norms of `M⁻¹` |
| `sigma_max`, `sigma_min` | `(P,)` | extreme singular values |
| `cond_1`, `cond_inf`, `cond_2` | `(P,)` | normwise condition numbers |
| `cond_skeel`, `cond_skeel_x` | `(P,)` | componentwise condition numbers |
| `seconds` | `(P,)` | wall time of the row |

Group attributes: `material`, `source` (the material file's path — read by
`list_by_kappa.py` to count right-hand sides), `n_indices`, `stride`,
`onenorm_t`, `svd_tol`, plus the material metadata (`grid_energy_min`,
`resolution`, band edges).

The row count `P` is fixed by the index selection at creation; reopening the
group with a different selection is an error (`--overwrite` or a different
`--outdir` to proceed). A group written before a column existed gains that
column filled with `NaN` rather than being rejected.

### `exact_condition.py` → the same file, group `condition_exact`

Same `SCALAR_DATASETS` naming with `_exact` appended (`cond_inf_exact`,
`cond_skeel_exact`, `cond_skeel_x_exact`, `cond_2_exact`, ...), plus `n` and
`nnz`. Attributes additionally record `max_n`, `did_inverse`, `did_svd`.

### Consumers

| Reader | Columns |
|---|---|
| `plotting/condition-est/plot_condition.py` | `cond_inf`, `cond_skeel`, `cond_skeel_x`, `cond_2`, and the `_exact` columns where present |
| `block-thomas/forward_error.py` | `cond_inf`, `cond_skeel`, `cond_skeel_x` |
| `mixed_prec_ir/mpir.py` | `cond_2`, `cond_inf` |
| `list_by_kappa.py` | `cond_2`, `cond_inf`, and `E_<idx>/rhs` from the material file directly |

---

## 8. Running it

Every material in the standard scratch layout, every index, estimated:

```bash
python condition_est.py --materials carbon-nanotube carbon-chain si-bulk graphene
```

One material, resuming an interrupted run — rows already marked valid are
skipped:

```bash
python condition_est.py --materials si-bulk --resume
```

Filling only the Skeel columns of a sweep made before they existed — one
factorization and a few solves per row, no singular values:

```bash
python condition_est.py --materials si-bulk --only-skeel
```

Exact reference on a handful of indices spanning the observed `kappa_2` range
(pick them with `list_by_kappa.py` first):

```bash
python exact_condition.py --material carbon-chain --idx 84 512 1200 1800 2400
```

Exact reference on every index — only for a small matrix; check `n` against
`--max-n` first, and expect this to run for a long time on a large one:

```bash
python exact_condition.py --material carbon-nanotube --all
```

Buckets by `kappa_2`, right-hand-side counts included, one material file at a
time (no `--materials` option on this script):

```bash
python list_by_kappa.py /scratch/yimili/condition-est/graphene.h5
```

Plotting, after the above:

```bash
python ../plotting/condition-est/plot_condition.py
```

---

## References

- H. A. van der Sluis, Condition numbers and equilibration of matrices,
  *Numer. Math.* 14, 1969; F. L. Bauer, Optimally scaled matrices,
  *Numer. Math.* 5, 1963. `cond(M)` as the row-equilibrated `kappa_inf(M)`.
- W. Oettli and W. Prager, Compatibility of approximate solution of linear
  equations with given error bounds, *Numer. Math.* 6, 1964. The
  componentwise backward error `omega`.
- A. K. Cline, C. B. Moler, G. W. Stewart and J. H. Wilkinson, An estimate for
  the condition number of a matrix, *SIAM J. Numer. Anal.* 16(2), 1979. The
  LINPACK estimator.
- R. D. Skeel, Scaling for numerical stability in Gaussian elimination,
  *J. ACM* 26(3), 1979. The componentwise condition number.
- W. Kahan, Numerical linear algebra, *Canadian Math. Bulletin* 9, 1966.
  `1/kappa_2` as the relative distance to the nearest singular matrix.
- W. W. Hager, Condition estimates, *SIAM J. Sci. Statist. Comput.* 5(2),
  1984. The ascent algorithm.
- N. J. Higham, FORTRAN codes for estimating the one-norm of a real or
  complex matrix, with applications to condition estimation, *ACM TOMS*
  14(4), 1988. LAPACK's `xLACON`.
- M. Arioli, J. W. Demmel and I. S. Duff, Solving sparse linear systems with
  sparse backward error, *SIAM J. Matrix Anal. Appl.* 10(2), 1989. Pairs
  `omega` with the Skeel condition number.
- C. S. Kenney and A. J. Laub, Small-sample statistical condition estimates
  for general matrix functions, *SIAM J. Sci. Comput.* 15(1), 1994. The
  probabilistic alternative not taken here.
- N. J. Higham and F. Tisseur, A block algorithm for matrix 1-norm
  estimation, with an application to 1-norm pseudospectra,
  *SIAM J. Matrix Anal. Appl.* 21(4), 2000. `t = 2` is their recommendation.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
  SIAM, 2002. Ch. 7 for normwise/componentwise perturbation theory, ch. 15
  for condition estimation.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
  Hopkins University Press, 2013. §3.2 for `LU`/inverse flop counts, §8.6 for
  SVD flop counts — the `O(n³)` figures in section 4.
- T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006. Sparse
  factorization cost as a function of fill-in and elimination order — the
  basis for the `O(n b²)` figure in section 4.1.
- J. Demmel, Y. Hida, W. Kahan, X. S. Li, S. Mukherjee and E. J. Riedy, Error
  bounds from extra-precise iterative refinement, *ACM TOMS* 32(2), 2006.
  The componentwise bound as LAPACK computes it.

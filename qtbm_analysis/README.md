# `qtbm_analysis/`

Thesis-side analysis of the linear systems QTBM produces: how fast the
available sparse solvers factor them, how accurately, and how stably — with
Block Thomas and its reduced-precision variants as the main subject.

## The pipeline

Everything hangs off **one HDF5 file per material**. Solver results are
appended into that same file rather than a separate one, so a matrix and every
factorization of it live side by side.

```
  export_qtbm_systems.py / make_hdf5.py
            │  writes E_<idx>/{M, rhs, Sigma, spectrum}, metadata/
            ▼
  ┌──────────────────────────────┐
  │  <material>.h5               │◄─── run_bench/run_benchmarks.py
  │    E_0/M, rhs, Sigma …       │       appends E_<idx>/<solver>/<dtype>/
  │    E_0/superlu/complex128/   │       via solvers/bench_all.bench()
  │    E_0/blockthomas/…         │
  │    E_0/blockthomas_inv/…     │
  └──────────────────────────────┘
            │  reads
            ├──────────► block-thomas/growth_factor.py    backward stability
            └──────────► run_bench/sweep_fp16.py          fp16 accuracy
```

Three folders carry this, and they are strictly layered — library, drivers,
analysis:

| folder | role | README |
|---|---|---|
| [`solvers/`](solvers/) | the solver library, one common interface, plus factor I/O | [README](solvers/README.md) |
| [`run_bench/`](run_bench/) | batch drivers — load, call `bench()`, write back | [README](run_bench/README.md) |
| [`block-thomas/`](block-thomas/) | post-hoc stability analysis, reads only | [README](block-thomas/README.md) |

There is exactly one `bench()` implementation (`solvers/bench_all.py`) and one
copy of each solver (`solvers/solver_classes.py`); the drivers are thin.

### Typical order of operations

```bash
# 1. benchmark + persist factors (mutates <material>.h5 in place)
cd run_bench && python run_benchmarks.py

# 2. backward-stability / growth factor across solvers
cd ../block-thomas && python growth_factor.py .../graphene.h5 --start 1 --end 400

# 3. half-precision accuracy sweep (needs step 1's c128/c64 results as reference)
cd ../run_bench && python sweep_fp16.py --h5path .../graphene.h5 --start 0 --end 401 --bs 416

# tests — synthetic data only, no cluster files needed
cd ../solvers && python test_pipeline.py
```

## Other folders

| folder | contents |
|---|---|
| `jupyter/` | exploratory notebooks (matrix analysis, Hermiticity, LU checks, Weyl) + per-material plots |
| `mixed_prec_ir/` | mixed-precision iterative refinement — GMRES-IR, dense and sparse |
| `condition_number_plots/`, `non-normal/` | conditioning and non-normality studies |
| `pyginkgo/` | Ginkgo binding experiments |
| `solvers/plots/`, `other/` | outputs, and superseded `main*.py` variants |
| `qttools/`, `quatrex/` | vendored copies of the two packages, for reference |

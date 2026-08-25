# `run_bench/` — batch drivers

Thin scripts that load material data, call into [`../solvers/`](../solvers/),
and write results back. **No benchmarking logic lives here**: there is exactly
one `bench()` implementation, in `solvers/bench_all.py`, so a correction or a
new solver added there is picked up by every driver.

**No figures are produced here either.** Every driver writes its results into
an HDF5 file, and the corresponding figure is produced by a script in
[`../plotting/`](../plotting/). See
[the pipeline description](../README.md#2-the-pipeline).

Each script resolves `../solvers` relative to its own file rather than to the
working directory, so it runs correctly from anywhere, and builds its parser
from `solvers/cli.py`, so the option names match every other script; see
[the top-level README, section 3](../README.md#3-command-line-conventions).

| Script | Function |
|---|---|
| `run_benchmarks.py` | full sweep: all materials, all solvers, both precisions, appended into each material's HDF5 file |
| `gpu_run_benchmarks.py` | the GPU solvers only, appended into the same files |
| `single_solve.py` | one matrix, one right-hand side, chosen solvers: timing and peak RSS. The diagnostic entry point |
| `gpu_single_solve.py` | the same for cuDSS |
| `sweep_fp16.py` | accuracy sweep of the two half-precision Block Thomas variants, writing an analysis file |

---

## `run_benchmarks.py`

```bash
python run_benchmarks.py
python ../plotting/block-thomas/plot_speedup.py /scratch/yimili/matrices2/hdf5/graphene.h5
```

Sweeps every material of `cli.MATERIALS` that declares a block size, appending
solver results into each
material's own HDF5 file as `E_<idx>/<solver>/<dtype>/`, siblings of
`E_<idx>/M`. **This mutates the source file in place**; copy it first if the
original must be preserved. Indices whose right-hand side has zero columns are
skipped, and appear as gaps in the figures produced downstream.

### Choosing the block partition

`BLOCK_MODE` selects where Block Thomas obtains its partition:

| Mode | Source | Notes |
|---|---|---|
| `"uniform"` | `block_size` of `cli.MATERIALS` | the default |
| `"custom"` | `blocks` of `cli.MATERIALS` | an error if the material has no entry |
| `"auto"` | detected per material | from the sparsity pattern |

The sparsity pattern is identical at every energy index, so `"auto"` detects
the partition once per material from the first matrix and reuses it. Both
non-uniform modes verify `offband_nnz == 0` before anything runs.

To populate the `blocks` field of a material, generate a line and paste it into
`cli.MATERIALS`:

```bash
python ../block-thomas/determine_custom_block_size.py <material>.h5 --emit-python
```

### `EXCLUDE`

Drops specific `(solver, dtype)` combinations deliberately. GMRES at single
precision does not reach a useful tolerance on these systems, so it is excluded
rather than recorded as a failure:

```python
EXCLUDE = {"gmres": {"complex64"}, "gmres-cupy": {"complex64"}}
```

---

## `single_solve.py`

The diagnostic entry point. One matrix, one right-hand side, explicit progress
at every stage, and peak RSS alongside the solver-reported factor memory.

```bash
python single_solve.py --solvers superlu mumps
python single_solve.py --solvers block-thomas block-thomas-inv --block-size 104
python single_solve.py --solvers block-thomas --auto-blocks
python single_solve.py --solvers block-thomas-inv-fp16 --block-size 32 \
    --inv-dtype float16
python single_solve.py /path/M.npz /path/rhs.npy --solvers superlu umfpack
```

The two memory figures answer different questions: `factor_nbytes` is what the
solver believes it stores, while the peak RSS delta is what the process
actually consumed, including workspace and fill-in the solver does not account
for.

Block solvers require either `--block-size` or `--auto-blocks`; the partition is
checked with `offband_nnz` and the run **aborts** rather than returning a
silently wrong solution. `--inv-dtype` sets the precision in which the
half-precision explicit-inverse variant forms its inverses; see the
[solvers README](../solvers/README.md#31-inv_dtype-the-mixed-precision-parameter).

---

## `sweep_fp16.py`

Accuracy sweep of both half-precision variants across a material's energy
range, against the stored `complex128` Block Thomas solution as reference.

```bash
python sweep_fp16.py .../carbon-nanotube.h5 --start 0 --end 401 --block-size 32
python sweep_fp16.py .../graphene.h5 --start 0 --end 401 --auto-blocks
python sweep_fp16.py .../graphene.h5 --idx 0 25 50 --inv-dtype float16
python ../plotting/block-thomas/plot_fp16_accuracy.py \
    /scratch/yimili/error-analysis-block-thomas/graphene.h5
```

Reads `E_<idx>/blockthomas/complex128/x` and the `complex64` equivalent, so
**`run_benchmarks.py` must have run first**.

Writes the `fp16_sweep` group of `<outdir>/<material>.h5`, with `--outdir`
defaulting to `cli.BLOCK_THOMAS_DIR`, holding the columns `idx, relres_fp16,
relres_fp16_inv, fwd_err_fp16_vs_c128, fwd_err_fp16_inv_vs_c128,
fwd_err_c64_vs_c128, cond_full_svd`. The run configuration and the failed
indices with their reasons are group attributes. The file is opened in append
mode and only that group is rewritten, so the `growth_factor` group written by
[`../block-thomas/growth_factor.py`](../block-thomas/growth_factor.py) into the
same file is preserved.

Unlike the other drivers this one does **not** write into the material file. It
is read-only with respect to it by design, so an accuracy sweep may be repeated
freely.

The residual and the forward error must be read together. The residual measures
backward error and is expected near the half-precision unit roundoff
`u = 2^-11`; the forward error additionally carries `kappa_2(M)`, so a large
forward error with a residual near `u` indicates ill-conditioning rather than
an unstable factorization.

`--inv-dtype float16` is the informative comparison: it makes implementation 2
a factorization that is half precision throughout, so the gap against the
`float32` default measures exactly what the higher-precision inversion buys.

---

## GPU drivers

`gpu_run_benchmarks.py` runs only the GPU solvers and appends them into the
same material files. It does not rerun any CPU solver, so it may be executed on
a GPU node without repeating the CPU sweep; `plotting/block-thomas/plot_speedup.py` reads
the SuperLU baseline already present in the file.

```bash
python gpu_run_benchmarks.py
python ../plotting/block-thomas/plot_speedup.py <material>.h5 --solvers cudss --suffix _gpu
```

Both GPU drivers complete without error on a CPU-only machine: availability is
checked per solver, and each prints a skip line.

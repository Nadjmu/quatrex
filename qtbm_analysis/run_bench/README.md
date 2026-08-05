# `run_bench/` — batch drivers

Thin scripts that load material data, call into
[`../solvers/`](../solvers/), and write results back. **No benchmarking logic
lives here** — there is exactly one `bench()` implementation, in
`solvers/bench_all.py`, so a fix or a new solver added there is picked up by
every driver automatically.

Each script resolves `../solvers` relative to **its own file**, not the cwd, so
they work from anywhere.

| script | what it does |
|---|---|
| `run_benchmarks.py` | full sweep over all materials, all solvers, both dtypes → h5 + speedup plot |
| `gpu_run_benchmarks.py` | the GPU-only solvers, reusing the stored SuperLU baseline for plotting |
| `single_solve.py` | one matrix, one RHS, chosen solvers — timing + peak RSS. The debugging entry point |
| `gpu_single_solve.py` | same, GPU |
| `sweep_fp16.py` | accuracy sweep of the two half-precision Block Thomas variants |

---

## `run_benchmarks.py`

```bash
python run_benchmarks.py
```

Sweeps every material in `MATERIAL_BS`, appending solver results **into each
material's own HDF5 file** — `E_<idx>/superlu/<dtype>/` and friends, siblings
of `E_<idx>/M`. **This mutates the source file in place**; back it up first if
you want the original untouched. A two-panel speedup-vs-SuperLU-c128 plot goes
to `<hdf5_dir>/../plots/<material>_speedup.png`.

### Choosing the block partition

`BLOCK_MODE` at the top of the file selects where Block Thomas gets its
partition:

| mode | source | notes |
|---|---|---|
| `"uniform"` | `MATERIAL_BS` | the historical behaviour, and the default |
| `"custom"` | `MATERIAL_BLOCKS` | errors if the material has no entry |
| `"auto"` | detected per material | from the sparsity pattern, once, not per index |

The sparsity pattern is identical at every energy index, so `"auto"` detects
the partition **once per material** off the first matrix and reuses it. Both
non-uniform modes verify `offband_nnz == 0` before running anything.

To populate `MATERIAL_BLOCKS`, generate a line and paste it in:

```bash
python ../block-thomas/determine_custom_block_size.py <material>.h5 --emit-python
```

### `EXCLUDE`

Drops specific `(solver, dtype)` combinations on purpose — currently GMRES at
single precision, both CPU and CuPy:

```python
EXCLUDE = {"gmres": {"complex64"}, "gmres_cupy": {"complex64"}}
```

---

## `single_solve.py`

The script to reach for when something is wrong. One matrix, one RHS, explicit
progress at every stage, and peak RSS alongside the solver-reported factor
memory.

```bash
python single_solve.py --solvers superlu mumps
python single_solve.py --solvers block_thomas block_thomas_inv --bs 104
python single_solve.py --solvers block_thomas --auto-blocks
python single_solve.py --solvers block_thomas_inv_fp16 --bs 32 --inv-dtype float16
python single_solve.py /path/M.npz /path/rhs.npy --solvers superlu umfpack
```

Block solvers need either `--bs` or `--auto-blocks`; the partition is checked
with `offband_nnz` and the run **aborts** rather than returning a silently
wrong `x`. `--inv-dtype` sets the precision in which the fp16 explicit-inverse
variant forms its inverses (see the [solvers README](../solvers/README.md)).

---

## `sweep_fp16.py`

Accuracy sweep of both half-precision variants across a material's energy
range, against the stored `complex128` Block Thomas solution as reference.

```bash
python sweep_fp16.py --h5path .../carbon-nanotube.h5 --start 0 --end 401 --bs 32
python sweep_fp16.py --h5path .../graphene.h5 --start 0 --end 401 --auto-blocks
python sweep_fp16.py --h5path .../graphene.h5 --start 0 --end 50 --inv-dtype float16
```

Reads `E_<idx>/blockthomas/complex128/x` and `.../complex64/x`, so
**`run_benchmarks.py` must have run first**. Writes to `plots/`:

- `<material>_metrics.csv` — `idx, relres_fp16, relres_fp16_inv,
  fwd_err_fp16_vs_c128, fwd_err_fp16_inv_vs_c128, fwd_err_c64_vs_c128,
  cond_full_svd`
- `<material>_metrics.txt` — the same plus run metadata and failed indices
- `<material>_relres_fwderr.png`, `<material>_forward_accuracy.png`

Unlike the other drivers this one does **not** write into the h5 file; it only
reads. It is read-only by design so an accuracy sweep can be re-run freely.

`--inv-dtype float16` is the interesting comparison: it makes Implementation 2
a genuinely all-half-precision factorization, so the gap against the default
`float32` measures exactly what the higher-precision inversion buys.

---

## GPU drivers

`gpu_run_benchmarks.py` runs only the GPU solvers and pulls the previously
stored `superlu_c128` result back out of the h5 file so the speedup baseline
still works without rerunning anything on CPU. It reuses `run_benchmarks.py`'s
plotting code verbatim, writing `<material>_speedup_gpu.png`.

Both GPU drivers no-op cleanly on a CPU-only machine — `gpu_available()` is
checked per solver, and each prints a skip line rather than failing.

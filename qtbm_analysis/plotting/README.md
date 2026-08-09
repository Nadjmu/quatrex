# `plotting/` — figures

Every figure in the project is produced here, and **nothing here computes a
result**. Each script reads an artefact that a compute script has already
written, a CSV or an HDF5 dataset, and renders it. No script in `solvers/`,
`run_bench/`, `block-thomas/`, `mixed_prec_ir/` or `non-normal/` imports
matplotlib.

This separation means a sweep that takes hours on the cluster is never repeated
in order to change an axis label, the numbers behind every figure exist as a
file that can be inspected or replotted, and a change to a figure cannot alter
a measurement.

| Script | Input | Figures produced |
|---|---|---|
| `plot_speedup.py` | a material HDF5 file | `<material>_speedup.png` |
| `plot_growth_factor.py` | `<material>_growth_factor.csv` | `<material>_growth_factor.png` |
| `plot_fp16_accuracy.py` | `<material>_metrics.csv` | `_relres_fwderr.png`, `_forward_accuracy.png`, `_error_vs_condition.png` |
| `plot_mixed_prec_ir.py` | `<material>_fp16_gmres_ir.csv` | `_ir_accuracy.png`, `_ir_iterations.png`, `_ir_error_vs_condition.png` |
| `plot_non_normal.py` | the `non-normal/` output directory | `frames/E_*.png`, `non_normal_shift.gif` |
| `plot_qtbm_spectra.py` | a `main3.py` output directory | `_spectrum.png`, `_condition.png`, `_singular_values.png` |
| `style.py` | — | library; not executable |

---

## Conventions

`style.py` defines the shared conventions, so that the same solver is drawn
identically in the timing, stability and accuracy figures and the three may be
compared directly:

- **Colour and marker identify the solver**, keyed by the HDF5 group name that
  `solvers/factor_io.py` writes: `blockthomas`, `gmres_scipy`, and so on. These
  are not the internal result keys used by `solvers/bench_all.py`
  (`block_thomas`, `gmres`); a script reading a CSV must map its own labels onto
  these keys first.
- **Line style identifies the precision.** `complex32` is not a NumPy dtype; it
  is the storage label used for the half-precision embedded-real
  factorizations.
- `FP16_UNIT_ROUNDOFF` is `u = 2^-11` for IEEE binary16, the reference level
  against which half-precision residuals are read.

All scripts use the `Agg` backend and write PNG; none opens a window.

---

## Which producer feeds which figure

```
run_bench/run_benchmarks.py       ──► <material>.h5              ──► plot_speedup.py
run_bench/gpu_run_benchmarks.py   ──► the same file              ──► plot_speedup.py --suffix _gpu
block-thomas/growth_factor.py     ──► *_growth_factor.csv        ──► plot_growth_factor.py
run_bench/sweep_fp16.py           ──► *_metrics.csv              ──► plot_fp16_accuracy.py
mixed_prec_ir/c32_gmres_ir.py     ──► *_fp16_gmres_ir.csv        ──► plot_mixed_prec_ir.py
non-normal/non-normal.py          ──► ratio_matrix.npy, ...      ──► plot_non_normal.py
main3.py / main3_gpu.py           ──► energies.npy, ...          ──► plot_qtbm_spectra.py
```

`plot_speedup.py` is the one script that reads the HDF5 file rather than a CSV,
because the timings it needs are already stored there and no intermediate
artefact is required.

---

## Usage

```bash
python plot_speedup.py        /scratch/yimili/matrices/hdf5/graphene.h5
python plot_speedup.py        .../graphene.h5 --solvers cudss --suffix _gpu
python plot_growth_factor.py  /scratch/yimili/block-thomas/graphene_growth_factor.csv
python plot_fp16_accuracy.py  ../run_bench/plots/graphene_metrics.csv
python plot_mixed_prec_ir.py  ../mixed_prec_ir/plots/graphene_fp16_gmres_ir.csv
python plot_non_normal.py     /scratch/yimili/non-normal/carbon-chain --ping-pong
python plot_qtbm_spectra.py   /scratch/yimili/matrices/dev_12_sorted_BENCH
```

Every script defaults its output directory to that of its input, and accepts
`--outdir` to override it. Each has a `--help` describing the quantities it
plots and how they are to be read.

---

## What the figures show

**`plot_speedup.py`.** Factorization time and solve time relative to the
SuperLU complex128 baseline, on a logarithmic axis with the unit line marked.
Ratios are taken per energy index, so a missing index in one series cannot bias
another. Shaded spans mark indices whose right-hand side has zero columns and
which the benchmark driver therefore skipped.

**`plot_growth_factor.py`.** Per norm: the growth ratios, the pivot growth
factor `rho`, and the reconstruction residual. The tight ratio is the quantity
that enters the backward-error bound; the loose ratio over-estimates it and is
drawn faded. The residual panel is a correctness guard, not a stability metric.

**`plot_fp16_accuracy.py`.** Residual against forward error, with the fp16 unit
roundoff drawn as a reference. A residual near `u` with a forward error far
above it indicates ill-conditioning rather than an unstable factorization. The
second figure isolates the cost of dropping from single to half precision on
the same algorithm; the third plots the forward error against `kappa_2(M)` with
the reference line `kappa_2 u`.

**`plot_mixed_prec_ir.py`.** Accuracy per variant, iteration counts, and error
against conditioning. The inner GMRES iteration count is the quantity that
determines whether refinement is cheaper than factorizing at higher precision.

**`plot_non_normal.py`.** Per rank, `sigma_i / |lambda_i|`, and its cumulative
logarithmic form. For a normal matrix the first is identically 1 and the second
identically 0, so deviation measures departure from normality. Axis limits are
computed once over the whole sweep so that frames are comparable, and rows are
read from the memory-mapped arrays one at a time.

**`plot_qtbm_spectra.py`.** The pencil spectrum, the conditioning of the bare
pencil against that of the full system matrix, and the two extreme singular
values. The last figure matters because a peak in `kappa_2` may arise either
from `sigma_min` approaching zero, the near-singular case of interest, or from
`sigma_max` growing, and the ratio alone does not distinguish them.

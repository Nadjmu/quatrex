# `plotting/` — figures

Every figure in the project is produced here, and **nothing here computes a
result**. Each script reads an HDF5 dataset that a compute script has already
written and renders it. No script in `solvers/`, `run_bench/`, `block-thomas/`,
`mixed_prec_ir/` or `non-normal/` imports matplotlib.

This separation means a sweep that takes hours on the cluster is never repeated
in order to change an axis label, the numbers behind every figure exist as a
file that can be inspected or replotted, and a change to a figure cannot alter
a measurement.

| Script | Input | Figures produced |
|---|---|---|
| `plot_speedup.py` | material file, timing datasets | `<material>_speedup.png` |
| `plot_growth_factor.py` | analysis file, `growth_factor` | `<material>_growth_factor.png` |
| `plot_fp16_accuracy.py` | analysis file, `fp16_sweep` | `_relres_fwderr.png`, `_forward_accuracy.png`, `_error_vs_condition.png` |
| `plot_non_normal.py` | analysis file, `non_normality` | `<material>_frames/E_*.png`, `<material>_non_normal.gif` |
| `plot_qtbm_spectra.py` | a `main3.py` output directory | `_spectrum.png`, `_condition.png`, `_singular_values.png` |
| `bandstructure.py` | `examples/<material>/inputs` | `bandstructure.png`, `bandstructure_zoom.png`, `hamiltonian_matrix.png` |
| `style.py` | — | library; not executable |

Every script that reads an analysis file takes it as the positional `h5path`
and defaults its output directory to that file's own directory, so a figure is
written beside the data it was drawn from. The three that read something else
default as follows.

| Script | Default output directory |
|---|---|
| `plot_speedup.py` | `cli.BLOCK_THOMAS_DIR` |
| `plot_qtbm_spectra.py` | `cli.CONDITION_DIR` |
| `bandstructure.py` | `cli.MATERIALS_DIR/<material>` |

---

## Conventions

`style.py` defines the shared conventions, so that the same solver is drawn
identically in the timing, stability and accuracy figures and the three may be
compared directly:

- **Colour and marker identify the solver**, keyed by the canonical solver name
  defined in `solvers/cli.py`: `block-thomas`, `gmres`, and so on — the same
  spelling every command line uses. A script that reads an HDF5 file converts
  the stored group name with `cli.from_h5_group` first, and `plot_speedup.py`
  goes the other way with `cli.h5_group` when it opens the file.
- **Line style identifies the precision.** `complex32` is not a NumPy dtype; it
  is the storage label used for the half-precision embedded-real
  factorizations.
- `FP16_UNIT_ROUNDOFF` is `u = 2^-11` for IEEE binary16, the reference level
  against which half-precision residuals are read.

- **The sweep axis is energy in eV**, never the energy index. The index is a
  position in a grid whose range and resolution are a per-material choice, so a
  figure labelled by index cannot be compared across materials or across two
  runs at different resolutions. The conversion is carried in the metadata of
  every material and analysis file, as `grid_energy_min` and `resolution`, and
  applied by `style.energies_of`. A file written before those attributes
  existed carries neither; the axis then falls back to the index, and
  `style.axis_label` labels it accordingly.
- **The band edges are marked** on every energy axis by `style.mark_band_edges`,
  the valence edge in blue and the conduction edge in red, matching
  `bandstructure.py`. Only edges recorded in the file and lying inside the swept
  range are drawn: a sweep positioned on the conduction edge commonly excludes
  the valence edge, and a line outside the data would widen the axis for
  nothing. The function returns the edges it drew, which a figure building its
  own legend must use so that the legend names only lines the figure contains.

All scripts use the `Agg` backend and write PNG; none opens a window.

---

## Which producer feeds which figure

```
run_bench/run_benchmarks.py      ─► matrices2/hdf5/<material>.h5            ─► plot_speedup.py
run_bench/gpu_run_benchmarks.py  ─► the same file                           ─► plot_speedup.py --suffix _gpu
block-thomas/growth_factor.py    ─► error-analysis-block-thomas/            ─► plot_growth_factor.py
                                     <material>.h5 :/growth_factor
run_bench/sweep_fp16.py          ─► the same file :/fp16_sweep              ─► plot_fp16_accuracy.py
non-normal/non-normal.py         ─► non-normal/<material>.h5                ─► plot_non_normal.py
                                     :/non_normality
main3.py / main3_gpu.py          ─► matrices2/<material>/energies.npy, ...  ─► plot_qtbm_spectra.py
```

`plot_speedup.py` reads the material file itself, because the timings it needs
are already stored there and no intermediate artefact is required. Its figure
is written to the Block Thomas analysis directory, beside the stability and
accuracy figures drawn from the same solver runs.

---

## Usage

```bash
python plot_speedup.py        /scratch/yimili/matrices2/hdf5/graphene.h5
python plot_speedup.py        .../graphene.h5 --solvers cudss --suffix _gpu
python plot_growth_factor.py  /scratch/yimili/error-analysis-block-thomas/graphene.h5
python plot_fp16_accuracy.py  /scratch/yimili/error-analysis-block-thomas/graphene.h5
python plot_non_normal.py     /scratch/yimili/non-normal/carbon-chain.h5 --ping-pong
python plot_qtbm_spectra.py   /scratch/yimili/matrices2/dev_12_sorted_BENCH
python bandstructure.py       graphene si-bulk
```

Every script accepts `--outdir` to override its default, `--material` to set
the label used in filenames and titles, and where applicable `--solvers` and
`--dtypes` in the canonical spellings. Each has a `--help` describing the
quantities it plots and how they are to be read.

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

**`plot_non_normal.py`.** Its panels are per rank rather than per energy, so it
has no energy axis to mark; the energy of the frame appears in the frame title
instead. Per rank, `sigma_i / |lambda_i|`, and its cumulative
logarithmic form. For a normal matrix the first is identically 1 and the second
identically 0, so deviation measures departure from normality. Axis limits are
computed once over the whole sweep so that frames are comparable, and the
`(P, n)` datasets are read one row at a time, so the sweep is never resident.

**`plot_qtbm_spectra.py`.** The pencil spectrum, the conditioning of the bare
pencil against that of the full system matrix, and the two extreme singular
values. The last figure matters because a peak in `kappa_2` may arise either
from `sigma_min` approaching zero, the near-singular case of interest, or from
`sigma_max` growing, and the ratio alone does not distinguish them.

**`bandstructure.py`.** The contact band structure of each configured device,
full and zoomed on the gap, with the valence and conduction band edges marked,
plus the magnitude of the leading Hamiltonian blocks. It is the one script here
that computes what it plots, since the eigensolve is its subject rather than a
measurement to be reused; it reads the device Hamiltonian directly and writes
one directory per material under `cli.MATERIALS_DIR`. The materials it can
process, their block sizes and their mid-gap energies come from `cli.MATERIALS`;
run it to determine the band edges, then record them there.

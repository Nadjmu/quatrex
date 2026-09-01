# `plotting/` — figures

Every figure in the project is produced here, and **nothing here computes a
result**. Each script reads an HDF5 dataset that a compute script has already
written and renders it. No script in `solvers/`, `run_bench/`, `block-thomas/`,
`mixed_prec_ir/` or `non-normal/` imports matplotlib.

This separation means a sweep that takes hours on the cluster is never repeated
in order to change an axis label, the numbers behind every figure exist as a
file that can be inspected or replotted, and a change to a figure cannot alter
a measurement.

Every script also writes a plain-text companion beside its figures — one
`<material>_<analysis>_data.txt` per run — holding the plot command, the
provenance the source file records, the options the figures were drawn under,
and every numeric series behind them as an aligned table and again as TSV. A
figure is then reproducible and readable without opening the HDF5 file: the
numbers can be replotted, diffed between runs, or handed to a reader — or a
language model — to interpret directly. `plot_mpir.py` and `plot_mpperf.py`
predate this and keep their own richer `_report.txt`; every other script goes
through `style.write_data_report`.

`plotting/` mirrors the same per-category layout as the scratch directories:
one subfolder per analysis, holding the scripts that plot it.

```
plotting/
├── style.py                  shared conventions; not executable; see below
├── block-thomas/
│   ├── plot_growth_factor.py
│   ├── plot_forward_error.py
│   ├── plot_backward_error.py
│   ├── plot_lu_factors.py
│   └── plot_fp16_accuracy.py
├── condition-est/
│   └── plot_condition.py
├── non-normal/
│   └── plot_non_normal.py
├── mixed_prec_ir/
│   ├── plot_mpir.py
│   └── plot_mpperf.py
├── matrices2/
│   ├── plot_qtbm_spectra.py
│   └── plot_rhs.py
└── materials/
    └── bandstructure.py
```

Every script below also writes a `_data.txt` companion (see above); the column
lists only the figures.

| Script | Input | Figures produced |
|---|---|---|
| `block-thomas/plot_growth_factor.py` | analysis file, `growth_factor` | `<material>_growth_factor.png`, `<material>_schur_growth.png` |
| `block-thomas/plot_forward_error.py` | analysis file, `forward_error` | `<material>_forward_error.png` |
| `block-thomas/plot_backward_error.py` | analysis file, `forward_error` | `<material>_backward_error.png` |
| `block-thomas/plot_lu_factors.py` | analysis file, `lu_factors` | `<material>_E<idx>_<solver>_<dtype>_lu.png` |
| `block-thomas/plot_fp16_accuracy.py` | analysis file, `fp16_sweep` | `_relres_fwderr.png`, `_forward_accuracy.png`, `_error_vs_condition.png` |
| `condition-est/plot_condition.py` | analysis file, `condition` | `<material>_condition.png`, `condition_all.png` |
| `mixed_prec_ir/plot_mpir.py` | convergence file, one `experiments/<NNNN>` | `exp<NNNN>/<material>_E<idx>.png`, `exp<NNNN>/<material>_summary.png`, `<material>_report.txt` |
| `mixed_prec_ir/plot_mpperf.py` | performance file, one `experiments/<NNNN>` | `perf<NNNN>/<material>_perf_summary.png`, `<material>_perf_report.txt` |
| `non-normal/plot_non_normal.py` | analysis file, `non_normality` | `<material>_frames/E_*.png`, `<material>_non_normal.gif` |
| `matrices2/plot_qtbm_spectra.py` | a `main3.py` output directory | `_spectrum.png`, `_condition.png`, `_singular_values.png` |
| `matrices2/plot_rhs.py` | material file, `E_<idx>/rhs` | `<material>_rhs.png` |
| `materials/bandstructure.py` | `examples/<material>/inputs` | `bandstructure.png`, `bandstructure_zoom.png`, `hamiltonian_matrix.png` |
| `style.py` | — | library; not executable |

Every script that reads an analysis file takes it as the positional `h5path`
and defaults its output directory to that file's own directory, so a figure is
written beside the data it was drawn from. The scripts that read something else
default as follows.

| Script | Default output directory |
|---|---|
| `matrices2/plot_qtbm_spectra.py` | `cli.CONDITION_DIR` |
| `matrices2/plot_rhs.py` | `cli.CONDITION_DIR` |
| `materials/bandstructure.py` | `cli.MATERIALS_DIR/<material>` |
| `mixed_prec_ir/plot_mpir.py` | `exp<NNNN>/` beside the analysis file — one subdirectory per experiment inside the material's own directory, so it stays `scp -r`-able as a unit |
| `mixed_prec_ir/plot_mpperf.py` | `perf<NNNN>/` beside the performance file, the same convention |

---

## Conventions

`style.py` defines the shared conventions, so that the same solver is drawn
identically in the timing, stability and accuracy figures and the three may be
compared directly. It lives at `plotting/`'s top level, one directory above
every script that imports it; each script resolves it and `solvers/cli.py`
relative to its own file, so the two-level nesting costs nothing at the call
site.

- **Colour and marker identify the solver**, keyed by the canonical solver name
  defined in `solvers/cli.py`: `block-thomas`, `gmres`, and so on — the same
  spelling every command line uses. A script that reads an HDF5 file converts
  the stored group name with `cli.from_h5_group` first; `cli.h5_group` goes the
  other way, for a script that opens the material file directly.
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
  `materials/bandstructure.py`. Only edges recorded in the file and lying
  inside the swept range are drawn: a sweep positioned on the conduction edge
  commonly excludes the valence edge, and a line outside the data would widen
  the axis for nothing. The function returns the edges it drew, which a figure
  building its own legend must use so that the legend names only lines the
  figure contains.

All scripts use the `Agg` backend and write PNG; none opens a window.

---

## Which producer feeds which figure

```
run_bench/run_benchmarks.py      ─► matrices2/hdf5/<material>.h5            ─► block-thomas/growth_factor.py
run_bench/gpu_run_benchmarks.py  ─► the same file                           ─► block-thomas/forward_error.py
block-thomas/growth_factor.py    ─► error-analysis-block-thomas/            ─► block-thomas/plot_growth_factor.py
                                     <material>.h5 :/growth_factor
block-thomas/forward_error.py    ─► the same file :/forward_error           ─► block-thomas/plot_forward_error.py
                                                                              ─► block-thomas/plot_backward_error.py
run_bench/sweep_fp16.py          ─► the same file :/fp16_sweep              ─► block-thomas/plot_fp16_accuracy.py
condition-est/condition_est.py   ─► condition-est/<material>.h5 :/condition ─► condition-est/plot_condition.py
mixed_prec_ir/mpir.py            ─► mixed-precision-IR/<material>/           ─► mixed_prec_ir/plot_mpir.py
                                     <material>.h5 :/experiments/<NNNN>/
                                       {runs,iterations}
mixed_prec_ir/mpperf.py          ─► mixed-precision-IR/<material>/           ─► mixed_prec_ir/plot_mpperf.py
                                     <material>_perf.h5 :/experiments/<NNNN>/
                                       runs
non-normal/non-normal.py         ─► non-normal/<material>.h5                ─► non-normal/plot_non_normal.py
                                     :/non_normality
main3.py / main3_gpu.py          ─► matrices2/<material>/energies.npy, ...  ─► matrices2/plot_qtbm_spectra.py
make_hdf5.py                     ─► matrices2/hdf5/<material>.h5 :/E_*/rhs ─► matrices2/plot_rhs.py
```

The two `mixed_prec_ir/` scripts are the ones here that read a *numbered
experiment* rather than a single fixed group: `mpir.py` and `mpperf.py` append one per
invocation and never overwrite, so `--experiment` selects which run to draw
and defaults to the last. `--list` prints what a file holds. See
[`../mixed_prec_ir/README.md`](../mixed_prec_ir/README.md#6-output).

---

## Usage

```bash
cd block-thomas
python plot_growth_factor.py  /scratch/yimili/error-analysis-block-thomas/graphene.h5
python plot_forward_error.py  /scratch/yimili/error-analysis-block-thomas/graphene.h5
python plot_backward_error.py /scratch/yimili/error-analysis-block-thomas/graphene.h5
python plot_fp16_accuracy.py  /scratch/yimili/error-analysis-block-thomas/graphene.h5

cd ../condition-est
python plot_condition.py      /scratch/yimili/condition-est/graphene.h5

cd ../mixed_prec_ir
python plot_mpir.py           /scratch/yimili/mixed-precision-IR/graphene/graphene.h5 --list
python plot_mpir.py           .../graphene/graphene.h5 --experiment 3 --idx 84 254
python plot_mpperf.py         .../graphene/graphene_perf.h5 --list
python plot_mpperf.py         .../graphene/graphene_perf.h5 --experiment 2

cd ../non-normal
python plot_non_normal.py     /scratch/yimili/non-normal/carbon-chain.h5 --ping-pong

cd ../matrices2
python plot_qtbm_spectra.py   /scratch/yimili/matrices2/dev_12_sorted_BENCH
python plot_rhs.py            /scratch/yimili/matrices2/hdf5/graphene.h5

cd ../materials
python bandstructure.py       graphene si-bulk
```

Every script accepts `--outdir` to override its default, `--material` to set
the label used in filenames and titles, and where applicable `--solvers` and
`--dtypes` in the canonical spellings. Each has a `--help` describing the
quantities it plots and how they are to be read.

---

## What the figures show

**`block-thomas/plot_growth_factor.py`.** Per norm: the growth ratios, the
pivot growth factor `rho`, and the reconstruction residual. The tight ratio is
the quantity that enters the backward-error bound; the loose ratio
over-estimates it and is drawn faded. The residual panel is a correctness
guard, not a stability metric.

A second figure, `<material>_schur_growth.png`, covers the Schur-complement
recursion of the two Block Thomas variants: block growth, the conditioning of
the pivot blocks, and the residual of implementation 2's explicit inverses. It
is drawn only when the analysis file carries those columns.

**`block-thomas/plot_backward_error.py`.** omega (componentwise, primary row)
and eta_inf (normwise, secondary row) against energy, one column per working
precision present — splitting by precision is what makes six solvers
distinguishable at all, since folding both precisions onto one axis spreads
omega over 10+ decades. The one figure in the chapter where MUMPS and cuDSS
appear alongside the four factor-exposing solvers, since backward error needs
only the stored solution. A unit-roundoff reference line is drawn per column.

**`block-thomas/plot_forward_error.py`.** Two rows, one column per working
precision present: the measured forward error against energy with the
reference floor `kappa_inf * eps_ext` drawn beneath it, and `fwd_inf / bound`
against energy with unity marked — the panel that answers the chapter's
question directly, since "the bound holds" reads as "the line sits below 1"
rather than requiring a diagonal-distance judgement. An earlier version drew a
fwd-against-bound scatter with the line `y = x` instead; mathematically the
same information, but harder to read for a comparison whose entire point is
"below one," so the ratio panel replaced it.

**`block-thomas/plot_fp16_accuracy.py`.** Residual against forward error, with
the fp16 unit roundoff drawn as a reference. A residual near `u` with a forward
error far above it indicates ill-conditioning rather than an unstable
factorization. The second figure isolates the cost of dropping from single to
half precision on the same algorithm; the third plots the forward error
against `kappa_2(M)` with the reference line `kappa_2 u`.

**`condition-est/plot_condition.py`.** `kappa_1`, `kappa_2` and `kappa_inf` of
`M(E)` together on one logarithmic axis per material. `kappa_1` and `kappa_inf`
rest on a norm estimate that is a lower bound and may sit slightly below the
exact value; `kappa_2` comes from both extreme singular values directly. Where
the three curves diverge beyond the factor-of-`n` equivalence of norms, it is
the shape of `M^-1` rather than the conditioning that differs between them.

**`mixed_prec_ir/plot_mpir.py`.** One experiment of a mixed-precision
refinement study, in the layout of Carson and Higham's numerical experiments.
Per energy index, two panels: on the left the convergence history — forward
error against the reference solution, and the normwise and componentwise
backward errors — with a dotted line at the working precision `u`; on the right
the convergence factor of Corollary 3.3, split into its conditioning term
`phi_cond` and its correction-solver term `phi_solve`, with the observed
contraction `rho` beside them and a dotted line at 1. Refinement converges
while `phi` stays comfortably below that line, and the split says which of the
two is binding. `phi` is drawn wide and pale because it is the sum of the other
two: whichever dominates coincides with it exactly and would otherwise be
hidden underneath. One summary figure then covers the whole sweep — what
refinement recovered against what the unrefined low-precision solve reached,
the outer iteration counts, and the inner GMRES counts.

**`mixed_prec_ir/plot_mpperf.py`.** One experiment of the companion
performance study, as a single figure: runtime against conditioning.

One group of bars per energy index, the groups ordered by `kappa_inf(A)` and
evenly spaced regardless of it — bars of neighbouring indices would otherwise
overlap wherever two condition numbers are close, which near a band edge is
most of them. Within a group each solver contributes a pair of bars in its own
`style.SOLVER_STYLE` colour: the left is the `complex64` factorization with
LU-IR, the right the `complex128` direct solve it is meant to replace. Reading
one pair is the whole point — the left bar shorter than the right is the case
for mixed precision at that conditioning, and the left bar growing past the
right as `kappa_inf` rises is refinement giving back what the low precision
won.

Each bar is stacked into `symbolic_s`, `factorization_s` and `solve_s`,
separated by black lines and shaded light to dark within the solver's colour.
The symbolic stage performs no floating-point arithmetic, so it is the same
height in both bars of a pair and is the part a lower precision cannot shrink;
a pair whose bars differ little is usually a pair whose symbolic stage
dominates — the explanation Zounon et al. give for sparse speedups falling
short of 2. SuperLU fuses the symbolic phase into the numerical one and reports
no split, so its bars have two segments and are marked in the legend. A left
bar is hatched where refinement did not reach the target accuracy: its height
is then a cost that bought nothing and no speedup should be read from that
pair.

The figure aggregates nothing. Every bar is one measured median, so a handful
of indices is the intended input — each one is eight bars.

Memory is recorded by `mpperf.py` and not yet drawn.

**`non-normal/plot_non_normal.py`.** Its panels are per rank rather than per
energy, so it has no energy axis to mark; the energy of the frame appears in
the frame title instead. Per rank, `sigma_i / |lambda_i|`, and its cumulative
logarithmic form. For a normal matrix the first is identically 1 and the second
identically 0, so deviation measures departure from normality. Axis limits are
computed once over the whole sweep so that frames are comparable, and the
`(P, n)` datasets are read one row at a time, so the sweep is never resident.

**`matrices2/plot_qtbm_spectra.py`.** The pencil spectrum, the conditioning of
the bare pencil against that of the full system matrix, and the two extreme
singular values. The last figure matters because a peak in `kappa_2` may arise
either from `sigma_min` approaching zero, the near-singular case of interest,
or from `sigma_max` growing, and the ratio alone does not distinguish them.

**`matrices2/plot_rhs.py`.** The number of injected contact modes along the
energy sweep. `main3.py` writes `rhs` at every exported index including those
with zero modes, so the band gap shows up as a run of zeros rather than a gap
in the sweep.

**`materials/bandstructure.py`.** The contact band structure of each configured
device, full and zoomed on the gap, with the valence and conduction band edges
marked, plus the magnitude of the leading Hamiltonian blocks. It is the one
script here that computes what it plots, since the eigensolve is its subject
rather than a measurement to be reused; it reads the device Hamiltonian
directly and writes one directory per material under `cli.MATERIALS_DIR`. The
materials it can process, their block sizes and their mid-gap energies come
from `cli.MATERIALS`; run it to determine the band edges, then record them
there.

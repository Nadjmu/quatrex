#!/usr/bin/env python3
"""
Runtime and LU-IR speedup against the number of BLAS threads.

Input
-----
One or more performance files written by mixed_prec_ir/run_thread_sweep.py,
each holding one experiment per thread count for one material. The thread
count is not passed on the command line: it is read back from each
experiment's own env_OPENBLAS_NUM_THREADS attribute, which mpperf.py records
because it never sets it.

The figure
----------
One column per material, two rows sharing the thread axis.

The top row is the wall-clock time of the complex128 direct solve. It is the
denominator of every speedup below it, and it is what a thread count is
normally chosen to minimise.

The bottom row is the LU-IR speedup over that solve, with a line at 1. Above
the line the complex64 factorization with refinement is faster than solving
in complex128 directly; below it, slower.

The two rows together are the point of the figure. Reading the top row alone
says more threads are better. Reading the bottom row alone says fewer threads
are better for mixed precision. Neither is the whole statement, and the
crossing in the bottom row is at a different thread count for each solver.

The reason the speedup falls as threads are added is that the factorization
parallelises and the complex128 residual does not. The residual is a sparse
matrix-vector product per right-hand side per refinement step, it is memory
bound, and it is recomputed in full precision no matter what u_f is. Adding
threads shrinks the part of the total that LU-IR makes cheaper and leaves the
part it pays for unchanged.

The x axis is log2, since the counts double. Each point is the median over
the indices measured at that thread count; individual indices are drawn as
faint markers behind the line where more than one was measured.

Usage
-----
    python plot_mpthreads.py <material>_perf.h5 [more.h5 ...]
    python plot_mpthreads.py .../threads/*/*_perf.h5 --outdir .../threads
    python plot_mpthreads.py .../si-bulk_perf.h5 --solvers block-thomas mumps
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.append(str((_HERE / "..").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "solvers").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "mixed_prec_ir").resolve()))

import cli                                                        # noqa: E402
from factor_io import table_rows                                  # noqa: E402
from mpperf import experiment_names, load_experiment              # noqa: E402
import style                                                      # noqa: E402

REFINED = "c64_ir"
REFERENCE = "c128"
BREAK_EVEN = 1.0


def thread_count(attrs):
    """
    The BLAS thread cap an experiment ran under.

    OPENBLAS_NUM_THREADS is authoritative on this node: NumPy is linked
    against a pthread OpenBLAS build, where OMP_NUM_THREADS is only a
    fallback. An experiment that recorded neither is not part of a sweep and
    is reported rather than guessed at.
    """
    for key in ("env_OPENBLAS_NUM_THREADS", "env_OMP_NUM_THREADS"):
        value = attrs.get(key)
        if isinstance(value, bytes):
            value = value.decode()
        if value not in (None, "", b""):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def collect(h5path, solvers=None):
    """
    (material, {solver: {threads: (ref_ms, [speedups])}}, skipped).

    A row counts only where it converged and where the complex128 solve of the
    same index and solver was measured: the speedup is a ratio against that
    solve, and without it there is no ratio to take.
    """
    material, data, skipped = None, {}, []
    for name in experiment_names(h5path):
        _, attrs, columns = load_experiment(h5path, name)
        runs = table_rows(columns)
        material = material or attrs.get("material") or Path(h5path).stem
        n = thread_count(attrs)
        if n is None:
            skipped.append((name, "no thread count recorded"))
            continue

        ref, refined = {}, {}
        for row in runs:
            key = (int(row["idx"]), row["solver"])
            if row["variant"] == REFERENCE:
                ref[key] = float(row["total_s"])
            elif row["variant"] == REFINED and row["converged"]:
                refined[key] = float(row["total_s"])

        for key, t in refined.items():
            idx, solver = key
            if solvers and solver not in solvers:
                continue
            t_ref = ref.get(key)
            if not t_ref or t <= 0:
                continue
            slot = data.setdefault(solver, {}).setdefault(n, ([], []))
            slot[0].append(t_ref * 1e3)
            slot[1].append(t_ref / t)

    return material, data, skipped


def _order(data):
    """Solvers in the canonical order, thread counts ascending."""
    known = [s for s in style.SOLVER_STYLE if s in data]
    return known + [s for s in sorted(data) if s not in known]


def plot(files, out_path, solvers=None):
    collected, skipped_all = [], []
    for path in files:
        material, data, skipped = collect(path, solvers)
        skipped_all += [(material, n, why) for n, why in skipped]
        if data:
            collected.append((material, data))
        else:
            print(f"  [skip] {path}: no usable rows")

    if not collected:
        print("  [skip] nothing to draw")
        return

    n_col = len(collected)
    fig, axes = plt.subplots(2, n_col, sharex=True,
                             figsize=(1.4 + 3.5 * n_col, 7.2),
                             squeeze=False)

    every_solver = []
    for col, (material, data) in enumerate(collected):
        ax_t, ax_s = axes[0][col], axes[1][col]
        for solver in _order(data):
            label, colour, marker = style.SOLVER_STYLE.get(
                solver, (solver, "#666666", "o"))
            if solver not in every_solver:
                every_solver.append(solver)
            counts = sorted(data[solver])
            ref_ms = [float(np.median(data[solver][n][0])) for n in counts]
            speed = [float(np.median(data[solver][n][1])) for n in counts]

            ax_t.plot(counts, ref_ms, marker=marker, ms=4.5, lw=1.6,
                      color=colour, label=label)
            ax_s.plot(counts, speed, marker=marker, ms=4.5, lw=1.6,
                      color=colour, label=label)
            # The individual indices, so a median over two is not mistaken
            # for a measurement of one.
            for n in counts:
                per_index = data[solver][n][1]
                if len(per_index) > 1:
                    ax_s.plot([n] * len(per_index), per_index, marker=marker,
                              ms=3.0, lw=0, color=colour, alpha=0.35)

        ax_t.set_title(material, fontsize=10)
        ax_t.set_yscale("log")
        ax_t.grid(True, which="both", alpha=0.25, lw=0.5)
        ax_s.grid(True, alpha=0.25, lw=0.5)
        ax_s.axhline(BREAK_EVEN, color="black", lw=1.0)
        ax_s.set_xscale("log", base=2)
        all_counts = sorted({n for d in data.values() for n in d})
        ax_s.set_xticks(all_counts)
        ax_s.set_xticklabels([str(n) for n in all_counts])
        ax_s.set_xlabel("BLAS threads")
        if col == 0:
            ax_t.set_ylabel("complex128 direct solve [ms]")
            ax_s.set_ylabel("LU-IR speedup over complex128")

    # One shared speedup axis, so the columns can be compared to each other
    # and not only within themselves. Taken from the data rather than from the
    # drawn artists: the break-even line is an artist too, and would pull the
    # limits towards 1 on a panel that never goes near it.
    every = [v for _material, data in collected for byn in data.values()
             for _ref, speeds in byn.values() for v in speeds]
    lo, hi = min(every + [BREAK_EVEN]), max(every + [BREAK_EVEN])
    pad = 0.05 * max(hi - lo, 0.1)
    for ax in axes[1]:
        ax.set_ylim(lo - pad, hi + pad)
    for ax in axes[1][1:]:
        ax.tick_params(labelleft=False)

    axes[1][-1].annotate("break-even", xy=(1.0, BREAK_EVEN),
                         xycoords=("axes fraction", "data"),
                         xytext=(-4, 3), textcoords="offset points",
                         ha="right", va="bottom", fontsize=8)

    handles = [plt.Line2D([], [], color=style.SOLVER_STYLE.get(
                   s, (s, "#666666", "o"))[1],
                   marker=style.SOLVER_STYLE.get(s, (s, "#666666", "o"))[2],
                   ms=4.5, lw=1.6,
                   label=style.SOLVER_STYLE.get(s, (s, "#666666", "o"))[0])
               for s in every_solver]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "cost of mixed-precision refinement against the BLAS thread count\n"
        "top: the complex128 solve the speedup is taken against;  "
        "bottom: LU-IR over it, above 1 = mixed precision is faster",
        fontsize=9)
    fig.subplots_adjust(top=0.87, bottom=0.14, left=0.075, right=0.985,
                        hspace=0.10, wspace=0.08)
    style.plot_provenance()
    style.save_figure(fig, out_path)
    print(f"wrote {out_path}")

    for material, name, why in skipped_all:
        print(f"  [note] {material} experiment {name} not drawn: {why}")

    _report(collected, out_path.with_name(out_path.stem + "_data.txt"), files)


def _report(collected, path, files_drawn):
    """The numbers behind the figure, as every plotting script here writes."""
    series = {}
    for material, data in collected:
        for solver in _order(data):
            counts = sorted(data[solver])
            series[f"{material}  {solver}"] = {
                "threads": np.array(counts),
                "c128_ms": np.array([float(np.median(data[solver][n][0]))
                                     for n in counts]),
                "speedup": np.array([float(np.median(data[solver][n][1]))
                                     for n in counts]),
                "n_indices": np.array([len(data[solver][n][1])
                                       for n in counts]),
            }
    style.write_data_report(
        path, title="LU-IR speedup against BLAS thread count",
        source=[str(p) for p in files_drawn],
        series=series,
        config={"reference variant": REFERENCE, "refined variant": REFINED,
                "point": "median over the indices at that thread count"})
    print(f"wrote {path}")


def main():
    ap = cli.new_parser(__doc__)
    ap.add_argument("h5path", nargs="+",
                    help="performance files written by run_thread_sweep.py, "
                         "one per material")
    ap.add_argument("--solvers", nargs="+", default=None, metavar="NAME",
                    help="draw only these solvers")
    cli.add_output(ap, outdir_help="output directory (default: beside the "
                                   "first input file)")
    args = ap.parse_args()

    files = [Path(p) for p in args.h5path]
    missing = [p for p in files if not p.exists()]
    if missing:
        sys.exit(f"no such file: {', '.join(str(p) for p in missing)}")

    outdir = Path(args.outdir) if args.outdir else files[0].parent
    outdir.mkdir(parents=True, exist_ok=True)
    plot(files, outdir / "perf_threads.png", solvers=args.solvers)


if __name__ == "__main__":
    main()

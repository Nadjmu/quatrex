#!/usr/bin/env python3
"""
The same measurement as run_perf_overnight.py, repeated at a range of thread
counts, to show that the thread count decides the answer and not only the
runtime.

The batch reports one configuration: 8 threads. At 8 threads LU-IR is slower
than the complex128 direct solve on si-bulk. At 1 and 2 threads it is faster.
Both are correct measurements of different machines. A reader given only the
8-thread figure has no way to know the other exists, so the sweep is measured
and drawn beside it.

The reason the count matters is that threading speeds up the factorization,
which is the phase u_f halves, but not the complex128 residual, which is
memory bound and is recomputed at every refinement step. The more threads the
factorization gets, the less of the total it accounts for, and the less there
is for a cheaper factorization to save.

What is measured
----------------
Two indices per material, taken from the lowest n_rhs group in
run_perf_overnight.SELECTION -- the lowest and the highest kappa_inf in that
group. Two is enough because the thread response is a property of the matrix
size and the solver, not of kappa_inf or n_rhs, both of which the main batch
already varies. Holding n_rhs fixed leaves the thread count as the only
variable in the figure.

Results go to <ANALYSIS_DIR>/threads/<material>/<material>_perf.h5, a separate
file from the main batch, so that the pooled n_rhs figure cannot accidentally
draw a curve across two different thread counts.

Usage
-----
    python3 run_thread_sweep.py
    python3 run_thread_sweep.py --threads 1 2 4 8
    python3 run_thread_sweep.py --materials si-bulk graphene
    python3 run_thread_sweep.py --replot

Each thread count is a separate process. The cap has to be set before numpy is
imported, so it cannot be changed inside one.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MPPERF = HERE / "mpperf.py"
PLOT = HERE.parent / "plotting" / "mixed_prec_ir" / "plot_mpthreads.py"

sys.path.insert(0, str(HERE))
# run_perf_overnight reads --threads from sys.argv at import time, to set the
# cap before numpy loads. This script's --threads means something else -- a
# list of counts to sweep -- so the import is done with an empty argv. Nothing
# here depends on the cap that import sets: every run below is launched with
# an explicit environment.
_argv, sys.argv = sys.argv, sys.argv[:1]
from run_perf_overnight import (                                 # noqa: E402
    ANALYSIS_DIR, HDF5_DIR, MATERIAL_ORDER, SELECTION, SOLVERS, banner)
sys.argv = _argv

SWEEP_DIR = ANALYSIS_DIR / "threads"
LOG_DIR = Path("/scratch/yimili/mpperf_thread_sweep_logs")

THREAD_COUNTS = [1, 2, 4, 8, 16, 32]
REPEATS = "9"
REDUCE = "min"
INNER = "direct"
N_INDICES = 2

_EXPERIMENT_RE = re.compile(r"wrote .+:/experiments/(\d+)")


def sweep_indices(material):
    """
    Two indices for one material: the ends of the kappa_inf range of its
    lowest n_rhs group.

    The group's indices are already ordered by kappa_inf in SELECTION, so the
    ends are the first and the last. Taking them from the lowest n_rhs keeps
    the residual cost -- the part that does not thread -- as small as it gets,
    which is the case least favourable to the point being made.
    """
    rhs_values = sorted(r for (m, r) in SELECTION if m == material)
    if not rhs_values:
        return None, []
    rhs = rhs_values[0]
    indices = SELECTION[(material, rhs)]
    return rhs, [indices[0], indices[-1]][:N_INDICES]


def run_one(material, indices, threads, log_dir):
    """One mpperf.py invocation at one thread count. Returns the experiment
    name, or None if it failed."""
    log = log_dir / f"{material}__t{threads}.log"
    argv = [sys.executable, str(MPPERF), str(HDF5_DIR / f"{material}.h5"),
            "--material", material,
            "--outdir", str(SWEEP_DIR),
            "--idx", *[str(i) for i in indices],
            "--solvers", *SOLVERS,
            "--inner", INNER,
            "--repeats", REPEATS,
            "--reduce", REDUCE]

    # A fresh environment per run: the cap is read when numpy is imported, so
    # it can only be changed by starting another process.
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)

    with open(log, "w") as f:
        f.write(f"OMP_NUM_THREADS={threads} "
                f"OPENBLAS_NUM_THREADS={threads} "
                + " ".join(argv) + "\n\n")
        f.flush()
        result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT,
                                env=env)

    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode}) -- see {log}", flush=True)
        return None
    match = _EXPERIMENT_RE.search(log.read_text())
    if match is None:
        print(f"    ran but wrote no experiment -- see {log}", flush=True)
        return None
    return f"{int(match.group(1)):04d}"


def draw(materials):
    """Draw the sweep figure over whatever the sweep file holds."""
    paths = [SWEEP_DIR / m / f"{m}_perf.h5" for m in materials]
    have = [p for p in paths if p.exists()]
    if not have:
        print("  [skip] no sweep files to draw", flush=True)
        return
    argv = [sys.executable, str(PLOT), *[str(p) for p in have],
            "--outdir", str(SWEEP_DIR)]
    r = subprocess.run(argv, capture_output=True, text=True)
    print(r.stdout + r.stderr, flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", nargs="+", type=int, default=THREAD_COUNTS,
                    metavar="N", help=f"thread counts to measure "
                                      f"(default {THREAD_COUNTS})")
    ap.add_argument("--materials", nargs="+", default=MATERIAL_ORDER,
                    metavar="NAME", help="materials to sweep")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from what is already measured, solve nothing")
    args = ap.parse_args()

    if args.replot:
        draw(args.materials)
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    todo = [(m, args.threads) for m in args.materials]
    total = sum(len(t) for _, t in todo)
    done = ok = 0

    for material, threads in todo:
        rhs, indices = sweep_indices(material)
        if not indices:
            print(f"  [skip] {material}: no group in SELECTION", flush=True)
            continue
        for n in threads:
            done += 1
            banner(f"[{done}/{total}] {material}  n_rhs={rhs}  "
                   f"indices {indices}  threads={n}")
            name = run_one(material, indices, n, LOG_DIR)
            if name:
                ok += 1
                print(f"    -> experiment {name}", flush=True)

    banner(f"sweep done: {ok}/{total} runs succeeded")
    draw(args.materials)


if __name__ == "__main__":
    main()

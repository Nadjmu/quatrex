#!/usr/bin/env python3
"""
One overnight batch: every (precision, solver, inner) combination the thesis
needs, for all four materials, run sequentially and plotted immediately after
with the y-max its tier uses.

    half   (complex32, Block Thomas only): LU-IR, GMRES-IR at 3 gmres_tol
    single (complex64, four solvers):      LU-IR, GMRES-IR at 3 gmres_tol

20 mpir.py invocations per material, 80 total. See mixed_prec_ir/README.md
and MPIR_GUIDE.md for what each run measures.

Sequential and single-process on purpose: attelas1 is shared, and pinning
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS here -- rather than trusting the
caller's shell -- is what keeps 80 runs from becoming 80 chances to forget,
after mpir-thread-contention-debugging showed what an unpinned run on this
node's factorizations actually costs (up to 68x, mpperf.py).

The analysis files already hold experiments from before this batch, so
experiment numbers do NOT start at 1 here. Nothing in this script assumes
they do: each run's actual experiment number is read back from mpir.py's own
"wrote ... :/experiments/NNNN" line, and EXPERIMENT_INDEX.html -- rewritten
after every single run, not just at the end -- is what tells you which
number is which afterwards.

Usage
-----
    nohup python3 run_overnight.py >&! /scratch/yimili/mpir_overnight_logs/driver.log &

(tcsh: nohup ... & backgrounds and detaches; >&! forces the redirect past
noclobber. Check progress with `tail -f` on that log or on EXPERIMENT_INDEX.html.)
"""
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

# Read, never set from inside a subprocess's own code: fixed here, in the
# parent, so every child inherits it. See mpperf.py's THREAD_VARS for why
# this is the one thing that must never be left to chance on this node.
THREADS = "8"
os.environ["OMP_NUM_THREADS"] = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS

HERE = Path(__file__).resolve().parent
MPIR = HERE / "mpir.py"
PLOT = HERE.parent / "plotting" / "mixed_prec_ir" / "plot_mpir.py"
HDF5_DIR = Path("/scratch/yimili/matrices2/hdf5")
ANALYSIS_DIR = Path("/scratch/yimili/mixed-precision-IR")
LOG_DIR = Path("/scratch/yimili/mpir_overnight_logs")
INDEX_HTML = ANALYSIS_DIR / "EXPERIMENT_INDEX.html"
# created in main(), not at import time -- importing this module for
# testing must not require /scratch to exist locally.

MATERIALS = {
    "carbon-nanotube": [2600, 2534, 2491, 2051, 2047, 384, 2035, 201, 454,
                        479, 108, 784, 659, 733, 536, 609, 543, 593, 571,
                        1617, 1614, 1612, 1610, 1609, 1608, 1607, 1606,
                        1601, 1605, 1602, 1604, 1603],
    "carbon-chain": [5855, 5856, 5672, 190, 5500, 469, 539, 595, 5413, 677,
                     706, 732, 746, 5392, 776, 780, 788, 812, 823, 833, 987,
                     848, 854, 934, 999, 867, 876, 912, 909, 880, 904, 882,
                     901, 886, 898, 887, 888, 895, 890, 894, 891, 893, 892],
    "si-bulk": [2718, 2692, 2740, 234, 2564, 2510, 392, 406, 2234, 632, 692,
                2348, 182, 888, 2274, 1876, 810, 640, 38, 1978, 668, 58,
                1932, 1864, 2106, 32, 30],
    "graphene": [15, 209, 365, 1991, 1833, 1756, 1647, 1646, 1631, 1632,
                 1633, 1642, 1641, 1636, 1639, 1637, 1638],
}

SOLVER_LABELS = {"block-thomas": "Block Thomas (LU)", "mumps": "MUMPS",
                 "cudss": "cuDSS", "superlu": "SuperLU"}
SOLVERS_SINGLE = ["block-thomas", "mumps", "cudss", "superlu"]
GMRES_TOL_HALF = [1e-1, 1e-2, 1e-3]
GMRES_TOL_SINGLE = [1e-4, 1e-5, 1e-6]


def run_specs():
    """
    One (label, description, argv_extra, y_max) per run, 20 per material.

    description is the human-readable line EXPERIMENT_INDEX.html shows next
    to the number mpir.py actually assigned; label is the same thing made
    filesystem-safe, for the per-run log file name.
    """
    specs = []

    specs.append(("half_lu-ir",
                  "(half, double, double) Block Thomas (LU) LU-IR",
                  ["--solver", "block-thomas", "--factor-dtype", "complex32"],
                  30))
    for tol in GMRES_TOL_HALF:
        specs.append((f"half_gmres-ir_tol{tol:g}",
                      f"(half, double, double) Block Thomas (LU) "
                      f"GMRES-IR tol={tol:g}",
                      ["--solver", "block-thomas", "--factor-dtype", "complex32",
                       "--inner", "gmres", "--gmres-tol", str(tol)],
                      20))

    for solver in SOLVERS_SINGLE:
        specs.append((f"single_lu-ir_{solver}",
                      f"(single, double, double) {SOLVER_LABELS[solver]} LU-IR",
                      ["--solver", solver, "--factor-dtype", "complex64"],
                      10))
    for solver in SOLVERS_SINGLE:
        for tol in GMRES_TOL_SINGLE:
            specs.append((f"single_gmres-ir_{solver}_tol{tol:g}",
                          f"(single, double, double) {SOLVER_LABELS[solver]} "
                          f"GMRES-IR tol={tol:g}",
                          ["--solver", solver, "--factor-dtype", "complex64",
                           "--inner", "gmres", "--gmres-tol", str(tol)],
                          10))
    return specs


_EXPERIMENT_RE = re.compile(r"wrote .+:/experiments/(\d+)")


def banner(msg):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n{'=' * 78}\n{ts}  {msg}\n{'=' * 78}", flush=True)


def write_index_html(records, failures):
    """
    (Re)write EXPERIMENT_INDEX.html from scratch. Called after every run
    rather than once at the end, so a run interrupted at 3am still leaves a
    correct index for everything that finished before it.
    """
    by_material = {}
    for r in records:
        by_material.setdefault(r["material"], []).append(r)

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>mpir overnight batch -- experiment index</title>",
             "<style>",
             "body{font-family:system-ui,sans-serif;margin:2em;color:#222}",
             "h1{font-size:1.3em}h2{margin-top:2em;border-bottom:2px solid #ccc}",
             "table{border-collapse:collapse;width:100%;margin-top:0.5em}",
             "th,td{border:1px solid #ccc;padding:4px 10px;text-align:left;"
             "font-size:0.92em}",
             "th{background:#eee}tr:nth-child(even){background:#f8f8f8}",
             "code{background:#eee;padding:1px 4px;border-radius:3px}",
             ".fail{color:#a00}",
             "</style></head><body>",
             f"<h1>mpir.py overnight batch -- experiment index</h1>",
             f"<p>Generated {datetime.datetime.now().isoformat(timespec='seconds')}"
             f" by run_overnight.py. Experiment numbers were already in use in"
             f" these files before this batch, so they do not start at 1 --"
             f" this page is the only record of which number is which run.</p>"]

    for material, rows in by_material.items():
        h5path = ANALYSIS_DIR / material / f"{material}.h5"
        parts.append(f"<h2>{material}</h2>")
        parts.append(f"<p><code>{h5path}</code></p>")
        parts.append("<table><tr><th>experiment</th><th>configuration</th>"
                     "<th>y-max</th><th>started</th><th>log</th></tr>")
        for r in sorted(rows, key=lambda r: int(r["experiment"])):
            parts.append(f"<tr><td>{r['experiment']}</td>"
                        f"<td>{r['description']}</td>"
                        f"<td>{r['y_max']}</td>"
                        f"<td>{r['timestamp']}</td>"
                        f"<td><code>{r['log']}</code></td></tr>")
        parts.append("</table>")

    if failures:
        parts.append("<h2 class='fail'>failed (no experiment written)</h2>")
        parts.append("<table><tr><th>material</th><th>configuration</th>"
                     "<th>started</th><th>log</th></tr>")
        for f in failures:
            parts.append(f"<tr class='fail'><td>{f['material']}</td>"
                        f"<td>{f['description']}</td>"
                        f"<td>{f['timestamp']}</td>"
                        f"<td><code>{f['log']}</code></td></tr>")
        parts.append("</table>")

    parts.append("</body></html>")
    INDEX_HTML.write_text("\n".join(parts))


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    specs = run_specs()
    total = len(specs) * len(MATERIALS)
    records, failures = [], []
    done = 0

    for material, indices in MATERIALS.items():
        h5path = HDF5_DIR / f"{material}.h5"
        analysis_h5 = ANALYSIS_DIR / material / f"{material}.h5"

        for label, description, extra, y_max in specs:
            done += 1
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            banner(f"[{done}/{total}] {material}: {description}")

            log = LOG_DIR / f"{material}__{label}.log"
            argv = ([sys.executable, str(MPIR), str(h5path),
                    "--idx", *[str(i) for i in indices],
                    "--outdir", str(ANALYSIS_DIR)] + extra)

            with open(log, "w") as f:
                f.write(" ".join(argv) + "\n\n")
                f.flush()
                result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)

            if result.returncode != 0:
                print(f"  FAILED (exit {result.returncode}) -- see {log}", flush=True)
                failures.append(dict(material=material, description=description,
                                     timestamp=ts, log=log))
                write_index_html(records, failures)
                continue

            match = _EXPERIMENT_RE.search(log.read_text())
            if match is None:
                print(f"  ran, but no experiment number found in {log} -- "
                      f"--no-save was likely in effect, or the run wrote nothing",
                      flush=True)
                failures.append(dict(material=material, description=description,
                                     timestamp=ts, log=log))
                write_index_html(records, failures)
                continue
            exp_name = match.group(1)
            print(f"  -> experiment {exp_name}", flush=True)

            plog = LOG_DIR / f"{material}__{label}__plot.log"
            pargv = [sys.executable, str(PLOT), str(analysis_h5),
                     "--experiment", exp_name, "--y-max", str(y_max)]
            with open(plog, "w") as f:
                f.write(" ".join(pargv) + "\n\n")
                f.flush()
                pr = subprocess.run(pargv, stdout=f, stderr=subprocess.STDOUT)
            if pr.returncode != 0:
                print(f"  (plot failed, exit {pr.returncode} -- see {plog}; "
                      f"the experiment itself is still recorded)", flush=True)

            records.append(dict(material=material, experiment=exp_name,
                                description=description, y_max=y_max,
                                timestamp=ts, log=log))
            write_index_html(records, failures)

    banner(f"batch done: {len(records)}/{total} succeeded, "
          f"{len(failures)} failed -- see {INDEX_HTML}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
One overnight batch of the performance study: eight energy indices for every
(material, n_rhs) group that has any, run sequentially and plotted immediately
after.

    n_rhs = 1   carbon-nanotube, carbon-chain, si-bulk
    n_rhs = 2   carbon-nanotube, carbon-chain, graphene
    n_rhs = 3   carbon-nanotube, si-bulk
    n_rhs = 4   carbon-nanotube, carbon-chain, si-bulk, graphene

12 mpperf.py invocations, 96 indices, 4 solvers and 2 variants each. Groups
with no indices at that n_rhs are absent from SELECTION and are not run. This
is the cost counterpart of run_overnight.py, which does the same for accuracy.

One experiment per group rather than one per material: the summary figure
draws a group of bars per index, and a material's four n_rhs groups together
would be 32 of them side by side, which is past the width at which a pair of
bars can still be compared. The four stay comparable through the pooled
speedup-against-n_rhs figure drawn per material at the end.

Why the indices are what they are
---------------------------------
Eight per group, spaced across the log10(kappa_inf) range that group covers
and including both ends. Cost is flat in kappa_inf and monotone in n_rhs, so
the eight are there to show that flatness holds over the whole range, while
n_rhs is what the groups vary.

Unstable runs
-------------
A row whose slowest repeat exceeds its fastest by more than the stability
limit is not a measurement of the solver -- it is a measurement of whatever
else the node was doing (see mpperf.py's check_stability). Such a run is
repeated, up to MAX_ATTEMPTS times, and the attempt with the fewest unstable
rows is the one plotted and indexed. Every attempt is kept in the analysis
file, since numbered experiments are never overwritten, and the index page
marks the superseded ones.

That is the honest way to remove the red outlines from a figure: measure
again on a quiet node until the repeats agree, rather than raising the limit
until nothing trips it.

Usage
-----
    nohup python3 run_perf_overnight.py >&! /scratch/yimili/mpperf_overnight_logs/driver.log &

(tcsh: nohup ... & detaches; >&! forces the redirect past noclobber.)

    python3 run_perf_overnight.py --preflight

checks every index in SELECTION against the material files -- that it exists,
that its n_rhs is what the group claims, and that its kappa_inf is valid --
and runs nothing. A stale selection costs a whole night, so this runs
automatically before the batch as well.

    python3 run_perf_overnight.py --replot

redraws every experiment this batch wrote and runs no solves.

    python3 run_perf_overnight.py --threads 2

runs the batch at a different thread cap. See the comment on THREADS: the
count changes which precision wins, not only how long the batch takes.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

# Read, never set from inside a subprocess's own code: fixed here, in the
# parent, so every child inherits it. mpperf.py deliberately never sets these
# -- it reports what it was given -- and an unpinned OpenBLAS on this node
# costs up to 68x on a Block Thomas factorization.
#
# The count is not a tuning detail. Threading speeds up the factorization,
# which is what u_f halves, but not the complex128 residual, which is memory
# bound, so it moves the answer rather than just the runtime. Measured on
# si-bulk idx 1864, complex128 Block Thomas and the LU-IR speedup beside it:
#
#     threads     1      2      4      8     16     32
#     c128    224.3  150.4  103.0   81.4   77.4   83.7  ms
#     speedup  1.06   1.07   0.97   0.79   0.83   0.87
#
# Refinement wins at one and two threads and loses from four up. On
# carbon-nanotube the whole curve is flat (9.2-9.6 ms from 1 to 16): its
# blocks are 33x33, below the size at which OpenBLAS splits the work at all.
# 8 is the default because it is near the floor for the large materials,
# harmless for the small ones, and one fixed value keeps the four materials
# comparable. Whatever is chosen, mpperf.py records it in every experiment.
#
# Parsed before argparse because the environment must be set before numpy or
# any solver library is imported; too late once a pool exists.
def _threads_from_argv(default="8"):
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return default


THREADS = _threads_from_argv()
if not THREADS.isdigit() or int(THREADS) < 1:
    sys.exit(f"--threads must be a positive integer, got {THREADS!r}")
os.environ["OMP_NUM_THREADS"] = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS

import h5py  # noqa: E402  (after the thread cap, before anything numeric)

HERE = Path(__file__).resolve().parent
MPPERF = HERE / "mpperf.py"
PLOT = HERE.parent / "plotting" / "mixed_prec_ir" / "plot_mpperf.py"
HDF5_DIR = Path("/scratch/yimili/matrices2/hdf5")
ANALYSIS_DIR = Path("/scratch/yimili/mixed-precision-IR")
LOG_DIR = Path("/scratch/yimili/mpperf_overnight_logs")
INDEX_HTML = ANALYSIS_DIR / "PERF_EXPERIMENT_INDEX.html"

# Eight indices per group, spread over that group's kappa_inf range. Keyed by
# (material, n_rhs); a group with no indices at that n_rhs is simply absent.
SELECTION = {
    ("carbon-nanotube", 1): [926, 994, 995, 996, 1600, 1601, 1602, 1603],
    ("carbon-chain", 1): [805, 929, 903, 895, 894, 891, 893, 892],
    ("si-bulk", 1): [910, 944, 962, 960, 952, 956, 954, 958],

    ("carbon-nanotube", 2): [1726, 1677, 1646, 1625, 1614, 1608, 1605, 1604],
    ("carbon-chain", 2): [8, 470, 602, 706, 762, 787, 796, 800],
    ("graphene", 2): [461, 1630, 1634, 1640, 1636, 1639, 1637, 1638],

    ("carbon-nanotube", 3): [1820, 1774, 777, 764, 788, 734, 727, 724],
    ("si-bulk", 3): [1800, 826, 796, 874, 804, 806, 884, 808],

    ("carbon-nanotube", 4): [358, 1851, 450, 484, 709, 527, 591, 575],
    ("carbon-chain", 4): [5855, 5327, 5504, 4917, 4900, 5397, 5394, 4885],
    ("si-bulk", 4): [746, 1794, 1852, 666, 718, 1866, 870, 1864],
    ("graphene", 4): [15, 187, 315, 358, 1995, 1845, 1767, 1715],
}

# Materials in the order their groups are run, so one material's file is
# written contiguously and its pooled n_rhs figure can be drawn as soon as its
# last group finishes.
MATERIAL_ORDER = ["carbon-nanotube", "carbon-chain", "si-bulk", "graphene"]

SOLVERS = ["block-thomas", "mumps", "cudss", "superlu"]
REPEATS = "9"
REDUCE = "min"
# LU-IR only. The GMRES-IR inner solve has never been measured on this node,
# so it is not what an unattended overnight batch should be finding out.
INNER = "direct"
MAX_ATTEMPTS = 3

_EXPERIMENT_RE = re.compile(r"wrote .+:/experiments/(\d+)")


def groups():
    """(material, n_rhs, indices) in run order: material outer, n_rhs inner."""
    out = []
    for material in MATERIAL_ORDER:
        for (mat, rhs), indices in sorted(SELECTION.items(),
                                          key=lambda kv: kv[0][1]):
            if mat == material:
                out.append((material, rhs, indices))
    return out


def perf_h5(material):
    """Where mpperf.py writes this material's performance file."""
    return ANALYSIS_DIR / material / f"{material}_perf.h5"


def banner(msg):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n{'=' * 78}\n{ts}  {msg}\n{'=' * 78}", flush=True)


def preflight():
    """
    Check every index in SELECTION against the material file it names.

    Three things go wrong with a hand-picked index list, and all three are
    silent at run time: an index that is not in the file at all, one whose
    n_rhs is not the group's (the listing it was copied from was for another
    material, or the file was regenerated), and one whose kappa_inf never
    converged, which the summary figure drops. Each costs a group's worth of
    the night, so they are found here instead.

    Returns the number of problems.
    """
    banner("preflight: checking SELECTION against the material files")
    problems = 0
    for material in MATERIAL_ORDER:
        path = HDF5_DIR / f"{material}.h5"
        if not path.exists():
            print(f"  MISSING material file {path}", flush=True)
            problems += sum(len(v) for (m, _), v in SELECTION.items()
                            if m == material)
            continue

        cond = {}
        cond_path = Path("/scratch/yimili/condition-est") / f"{material}.h5"
        if cond_path.exists():
            try:
                with h5py.File(cond_path, "r") as cf:
                    g = cf["condition"]
                    cond = dict(zip(g["indices"][()].tolist(),
                                    zip(g["cond_inf"][()].tolist(),
                                        g["valid"][()].tolist())))
            except Exception as exc:                       # noqa: BLE001
                print(f"  [note] {material}: condition file unreadable "
                      f"({exc}); kappa is not checked", flush=True)

        with h5py.File(path, "r") as f:
            for (mat, rhs), indices in sorted(SELECTION.items(),
                                              key=lambda kv: kv[0][1]):
                if mat != material:
                    continue
                bad, kappas = [], []
                for i in indices:
                    key = f"E_{i}"
                    if key not in f:
                        bad.append(f"idx {i}: not in the material file")
                        continue
                    rhs_ds = f[key]["rhs"]
                    n = rhs_ds.shape[1] if rhs_ds.ndim > 1 else 1
                    if n != rhs:
                        bad.append(f"idx {i}: n_rhs is {n}, group claims {rhs}")
                    k, valid = cond.get(i, (float("nan"), None))
                    if valid is False:
                        bad.append(f"idx {i}: kappa_inf did not converge")
                    elif k == k:
                        kappas.append(k)
                span = (f"{min(kappas):.1e}..{max(kappas):.1e}"
                        if kappas else "unknown")
                mark = "ok " if not bad else "BAD"
                print(f"  {mark} {material:<17} n_rhs={rhs}  "
                      f"{len(indices)} indices  kappa_inf {span}", flush=True)
                for line in bad:
                    print(f"        {line}", flush=True)
                problems += len(bad)

    banner(f"preflight: {problems} problem(s)")
    return problems


def experiment_attrs(material, name):
    """
    n_unstable and the row count of one written experiment.

    Read back from the file rather than scraped from stdout: mpperf.py stores
    the stability verdict as an attribute, so this cannot drift from what the
    run actually decided.
    """
    with h5py.File(perf_h5(material), "r") as f:
        g = f["experiments"][name]
        return (int(g.attrs.get("n_unstable", -1)),
                int(g["runs"]["idx"].shape[0]))


def run_group(material, rhs, indices, attempt, log_dir):
    """
    One mpperf.py invocation. Returns (experiment_name, n_unstable, n_rows,
    log_path), with experiment_name None if the run failed or wrote nothing.
    """
    log = log_dir / f"{material}__rhs{rhs}__try{attempt}.log"
    argv = [sys.executable, str(MPPERF), str(HDF5_DIR / f"{material}.h5"),
            "--material", material,
            "--outdir", str(ANALYSIS_DIR),
            "--idx", *[str(i) for i in indices],
            "--solvers", *SOLVERS,
            "--inner", INNER,
            "--repeats", REPEATS,
            "--reduce", REDUCE]

    with open(log, "w") as f:
        f.write(" ".join(argv) + "\n\n")
        f.flush()
        result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode}) -- see {log}", flush=True)
        return None, -1, 0, log

    match = _EXPERIMENT_RE.search(log.read_text())
    if match is None:
        print(f"    ran but wrote no experiment -- see {log}", flush=True)
        return None, -1, 0, log

    name = f"{int(match.group(1)):04d}"
    n_unstable, n_rows = experiment_attrs(material, name)
    return name, n_unstable, n_rows, log


def draw(material, name, log_dir, label):
    """Draw one experiment's summary figure and report."""
    plog = log_dir / f"{label}__plot.log"
    argv = [sys.executable, str(PLOT), str(perf_h5(material)),
            "--experiment", name]
    with open(plog, "w") as f:
        f.write(" ".join(argv) + "\n\n")
        f.flush()
        r = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"    (plot failed, exit {r.returncode} -- see {plog}; the "
              f"experiment itself is recorded)", flush=True)
    return r.returncode == 0


def draw_nrhs(material, names, log_dir):
    """
    The pooled speedup-against-n_rhs figure for one material.

    Restricted to this batch's experiments: the file also holds older runs
    made with other reducers and repeat counts, and pooling those into one
    line would compare measurements that are not comparable.
    """
    if len(names) < 2:
        print(f"  [skip] {material}: only {len(names)} group(s), the n_rhs "
              f"figure needs at least two", flush=True)
        return
    plog = log_dir / f"{material}__nrhs__plot.log"
    argv = [sys.executable, str(PLOT), str(perf_h5(material)),
            "--nrhs", "--experiments", *names]
    with open(plog, "w") as f:
        f.write(" ".join(argv) + "\n\n")
        f.flush()
        r = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)
    print(f"  n_rhs figure over experiments {', '.join(names)}: "
          f"{'ok' if r.returncode == 0 else f'FAILED, see {plog}'}", flush=True)


def write_index_html(records, failures):
    """
    (Re)write PERF_EXPERIMENT_INDEX.html from scratch after every group, so a
    batch interrupted at 3am still leaves a correct index for what finished.
    """
    by_material = {}
    for r in records:
        by_material.setdefault(r["material"], []).append(r)

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>mpperf overnight batch -- experiment index</title>",
             "<style>",
             "body{font-family:system-ui,sans-serif;margin:2em;color:#222}",
             "h1{font-size:1.3em}h2{margin-top:2em;border-bottom:2px solid #ccc}",
             "table{border-collapse:collapse;width:100%;margin-top:0.5em}",
             "th,td{border:1px solid #ccc;padding:4px 10px;text-align:left;"
             "font-size:0.92em}",
             "th{background:#eee}tr:nth-child(even){background:#f8f8f8}",
             "code{background:#eee;padding:1px 4px;border-radius:3px}",
             ".fail{color:#a00}.old{color:#888}.warn{color:#b45309}",
             "</style></head><body>",
             "<h1>mpperf.py overnight batch -- experiment index</h1>",
             f"<p>Generated "
             f"{datetime.datetime.now().isoformat(timespec='seconds')} by "
             f"run_perf_overnight.py. {REPEATS} repeats, reduced by "
             f"{REDUCE}, {THREADS} threads, solvers "
             f"{', '.join(SOLVERS)}. Experiment numbers continue from what "
             f"was already in these files, so they do not start at 1.</p>",
             "<p>A row marked <span class='old'>superseded</span> is an "
             "attempt whose repeats disagreed; it was re-run and is kept only "
             "because numbered experiments are never overwritten.</p>"]

    for material, rows in by_material.items():
        parts.append(f"<h2>{material}</h2>")
        parts.append(f"<p><code>{perf_h5(material)}</code></p>")
        parts.append("<table><tr><th>experiment</th><th>n_rhs</th>"
                     "<th>indices</th><th>rows</th><th>unstable</th>"
                     "<th>attempt</th><th>started</th><th>log</th></tr>")
        for r in sorted(rows, key=lambda r: (r["rhs"], int(r["experiment"]))):
            cls = " class='old'" if r["superseded"] else (
                " class='warn'" if r["n_unstable"] else "")
            state = ("superseded" if r["superseded"]
                     else f"{r['n_unstable']} of {r['n_rows']}")
            parts.append(f"<tr{cls}><td>{r['experiment']}</td>"
                         f"<td>{r['rhs']}</td>"
                         f"<td>{' '.join(str(i) for i in r['indices'])}</td>"
                         f"<td>{r['n_rows']}</td>"
                         f"<td>{state}</td>"
                         f"<td>{r['attempt']} of {MAX_ATTEMPTS}</td>"
                         f"<td>{r['timestamp']}</td>"
                         f"<td><code>{r['log'].name}</code></td></tr>")
        parts.append("</table>")

    if failures:
        parts.append("<h2 class='fail'>failed (no experiment written)</h2>")
        parts.append("<table><tr><th>material</th><th>n_rhs</th>"
                     "<th>started</th><th>log</th></tr>")
        for f in failures:
            parts.append(f"<tr class='fail'><td>{f['material']}</td>"
                         f"<td>{f['rhs']}</td><td>{f['timestamp']}</td>"
                         f"<td><code>{f['log'].name}</code></td></tr>")
        parts.append("</table>")

    parts.append("</body></html>")
    INDEX_HTML.write_text("\n".join(parts))


def replot():
    """
    Redraw every experiment this batch wrote, and nothing else.

    The figures are a function of the stored experiment, so a change to
    plot_mpperf.py reaches them by running it again; the solves cost a night
    and do not need repeating. Which experiments those are is read from the
    index page, so an experiment made outside this batch is left alone.
    """
    if not INDEX_HTML.exists():
        print(f"no {INDEX_HTML} -- nothing to replot; run the batch first")
        return
    text = INDEX_HTML.read_text()
    drawn = 0
    for material in MATERIAL_ORDER:
        # The index lists each material's experiments in its own section.
        section = text.split(f"<h2>{material}</h2>")
        if len(section) < 2:
            continue
        body = section[1].split("<h2>")[0]
        names = [m for m in re.findall(r"<tr[^>]*><td>(\d{4})</td>", body)]
        kept = [n for n, row in zip(names, body.split("<tr")[1:])
                if "superseded" not in row]
        banner(f"replot: {material} ({len(kept)} experiments)")
        for name in kept:
            argv = [sys.executable, str(PLOT), str(perf_h5(material)),
                    "--experiment", name]
            r = subprocess.run(argv, capture_output=True, text=True)
            print(f"  {name}: {'ok' if r.returncode == 0 else 'FAILED'}",
                  flush=True)
            if r.returncode != 0:
                print(r.stdout + r.stderr, flush=True)
            drawn += r.returncode == 0
        if kept:
            draw_nrhs(material, kept, LOG_DIR)
    banner(f"replot done: {drawn} experiments redrawn")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true",
                    help="check SELECTION against the material files and exit")
    ap.add_argument("--replot", action="store_true",
                    help="redraw this batch's experiments and run no solves")
    ap.add_argument("--force", action="store_true",
                    help="run the batch even if the preflight found problems")
    ap.add_argument("--threads", default=THREADS, metavar="N",
                    help=f"OMP_NUM_THREADS and OPENBLAS_NUM_THREADS for every "
                         f"run (default {THREADS}). Read before argparse, "
                         f"since the cap must be set before numpy is "
                         f"imported; listed here so --help shows it")
    args = ap.parse_args()

    if args.replot:
        replot()
        return

    if args.preflight:
        sys.exit(1 if preflight() else 0)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if preflight() and not args.force:
        print("\nrefusing to start: fix SELECTION, or pass --force to run "
              "anyway.\nA stale index list costs the whole night, which is "
              "why this is checked first.", flush=True)
        sys.exit(1)

    todo = groups()
    records, failures = [], []
    kept_by_material = {}
    done = 0

    for material, rhs, indices in todo:
        done += 1
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        banner(f"[{done}/{len(todo)}] {material}  n_rhs={rhs}  "
               f"{len(indices)} indices")

        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"  attempt {attempt} of {MAX_ATTEMPTS}", flush=True)
            name, n_unstable, n_rows, log = run_group(
                material, rhs, indices, attempt, LOG_DIR)
            if name is None:
                failures.append(dict(material=material, rhs=rhs,
                                     timestamp=ts, log=log))
                write_index_html(records, failures)
                break
            attempts.append(dict(material=material, rhs=rhs, indices=indices,
                                 experiment=name, n_unstable=n_unstable,
                                 n_rows=n_rows, attempt=attempt,
                                 timestamp=ts, log=log, superseded=False))
            print(f"    -> experiment {name}: {n_unstable} of {n_rows} rows "
                  f"unstable", flush=True)
            if n_unstable == 0:
                break
            if attempt < MAX_ATTEMPTS:
                print(f"    repeats disagreed; measuring again", flush=True)

        if not attempts:
            continue

        # Keep the attempt that measured the machine least.
        best = min(attempts, key=lambda a: a["n_unstable"])
        for a in attempts:
            a["superseded"] = a is not best
        records.extend(attempts)

        if best["n_unstable"]:
            print(f"  still {best['n_unstable']} unstable row(s) after "
                  f"{len(attempts)} attempt(s); drawing experiment "
                  f"{best['experiment']} with them outlined", flush=True)
        draw(material, best["experiment"], LOG_DIR, f"{material}__rhs{rhs}")
        kept_by_material.setdefault(material, []).append(best["experiment"])
        write_index_html(records, failures)

    for material, names in kept_by_material.items():
        banner(f"pooled n_rhs figure: {material}")
        draw_nrhs(material, names, LOG_DIR)

    kept = [r for r in records if not r["superseded"]]
    unstable = sum(r["n_unstable"] for r in kept)
    banner(f"batch done: {len(kept)}/{len(todo)} groups measured, "
           f"{len(failures)} failed, {unstable} unstable row(s) remaining "
           f"-- see {INDEX_HTML}")


if __name__ == "__main__":
    main()
